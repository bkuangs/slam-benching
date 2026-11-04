"""SP_SLAM3 learned relocalization settings (patch 0007): MixVPR keyframe descriptors + top-K retrieval."""

import json
from pathlib import Path

import cv2
import pytest
import yaml

from slambench.algorithms import spslam3 as sp
from slambench.algorithms.base import RunContext
from slambench.algorithms.spslam3 import SpSlam3Adapter
from slambench.types import SensorMode

from synthetic import FakeDataset

REPO = Path(__file__).resolve().parents[1]
LEARNED_KEYS = set(sp.LEARNED_RELOC_DEFAULTS)


def _mapping_run(root, settings=None):
    run = root / "runs" / "map_cfg" / "20240101T000000"
    (run / "algorithm").mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": "success"}))
    s = SpSlam3Adapter({}).resolve_settings({"save_map": True, **(settings or {})})
    (run / "config.resolved.yaml").write_text(yaml.safe_dump({"algorithm": {"name": "spslam3", "settings": s}}))
    (run / "algorithm" / "atlas.osa").write_bytes(b"atlas")
    return run


def _ctx(tmp_path, settings=None):
    ds = FakeDataset(tmp_path, "s1")
    (tmp_path / "prep").mkdir(exist_ok=True)
    prep = ds.prepare_offline(SensorMode.STEREO, tmp_path / "prep")
    (tmp_path / "x.sif").write_bytes(b"")
    a = SpSlam3Adapter({"container": str(tmp_path / "x.sif")})
    return a, ds, RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.STEREO, prep,
                             a.resolve_settings(settings or {}))


def _settings_text(a, ctx):
    a.prepare(ctx)
    return (ctx.algo_dir / sp.SETTINGS_FILE).read_text()


def test_unused_learned_reloc_settings_leave_settings_and_file_unchanged(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    assert not set(ctx.settings) & LEARNED_KEYS
    text = _settings_text(a, ctx)
    assert "GlobalDescriptor" not in text and "LearnedReloc" not in text
    # explicitly disabled == absent in the settings file
    ctx2 = RunContext(tmp_path, tmp_path / "algo2", ds, SensorMode.STEREO, ctx.prepared,
                      a.resolve_settings({"global_descriptors": False}))
    assert _settings_text(a, ctx2) == text


def test_global_descriptors_for_the_mapping_session(tmp_path):
    a, ds, ctx = _ctx(tmp_path, {"save_map": True, "global_descriptors": True})
    text = _settings_text(a, ctx)
    assert 'GlobalDescriptor.model_path: "/opt/SP_SLAM3/mixvpr.pt"\n' in text
    assert "LearnedReloc" not in text
    assert a.validate(SensorMode.STEREO, ds, ctx.settings) == []


def test_learned_reloc_settings_file(tmp_path):
    run = _mapping_run(tmp_path, {"global_descriptors": True})
    a, ds, ctx = _ctx(tmp_path, {"load_map": str(run), "learned_reloc": True, "learned_reloc_topk": 7,
                                 "learned_reloc_verifier": "bow"})
    assert ctx.settings["learned_reloc_min_inliers"] == 50
    assert ctx.settings["learned_reloc_consistent_frames"] == 3
    a.prepare(ctx)
    fs = cv2.FileStorage(str(ctx.algo_dir / sp.SETTINGS_FILE), cv2.FILE_STORAGE_READ)
    assert fs.getNode("GlobalDescriptor.model_path").string() == sp.MIXVPR_MODEL
    for key, val in (("LearnedReloc.enable", 1), ("LearnedReloc.topK", 7), ("LearnedReloc.minInliers", 50),
                     ("LearnedReloc.consistentFrames", 3)):
        assert fs.getNode(key).isInt() and int(fs.getNode(key).real()) == val, key
    assert fs.getNode("LearnedReloc.verifier").string() == "bow"
    assert int(fs.getNode("LearnedReloc.recoverDescriptors").real()) == 1
    assert fs.getNode("LearnedReloc.logFile").string() == str((ctx.algo_dir / "learned_reloc.csv").resolve())
    assert "--relocalize-only" in a.command(ctx)
    assert a.validate(SensorMode.STEREO, ds, ctx.settings) == []


def test_learned_reloc_rejected_settings(tmp_path):
    run = _mapping_run(tmp_path, {"global_descriptors": True})
    a = SpSlam3Adapter({})
    base = {"load_map": str(run), "learned_reloc": True}
    for bad in ({"learned_reloc": True},  # no load_map
                {**base, "reuse_mode": "merge"},
                {**base, "learned_reloc": 1}, {"global_descriptors": "yes"},
                {**base, "learned_reloc_topk": 0}, {**base, "learned_reloc_topk": 2.5},
                {**base, "learned_reloc_min_inliers": 5}, {**base, "learned_reloc_consistent_frames": 0},
                {**base, "learned_reloc_verifier": "orb"},
                {**base, "learned_reloc_recover_descriptors": 1},
                {**base, "lightglue": False}):
        with pytest.raises(ValueError):
            a.resolve_settings(bad)
    # the BoW verifier does not need LightGlue
    a.resolve_settings({**base, "learned_reloc_verifier": "bow", "lightglue": False})


def test_learned_reloc_needs_a_map_with_descriptors(tmp_path):
    run = _mapping_run(tmp_path)
    a, ds, ctx = _ctx(tmp_path, {"load_map": str(run), "learned_reloc": True})
    problems = a.validate(SensorMode.STEREO, ds, ctx.settings)
    assert len(problems) == 1 and "global_descriptors: true" in problems[0]


def test_legacy_r3_image_keeps_map_reuse_but_refuses_learned_reloc(tmp_path, monkeypatch):
    r3 = "3fc07c4b60ad6e1c5564bfa178fd55fe19e831c6"
    assert r3 in sp.MAP_REUSE_LEGACY_DRIVER_SOURCE_SHA1 and r3 in sp.LEGACY_DRIVER_SOURCE_SHA1
    assert sp.driver_source_sha1() != r3

    class R:
        returncode = 0
        stdout = f"spslam3_commit={sp.SPSLAM3_COMMIT}\ndriver_source_sha1={r3}\n"

    def run(cmd, **kw):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return R()
    monkeypatch.setattr(sp.subprocess, "run", run)
    mrun = _mapping_run(tmp_path, {"global_descriptors": True})
    a, ds, ctx = _ctx(tmp_path, {"load_map": str(mrun)})
    a.options["apptainer"] = "apptainer"
    info = a.version_info()
    assert "r3" in info["legacy_driver_source"]
    assert a.validate(SensorMode.STEREO, ds, ctx.settings) == []
    s = a.resolve_settings({"load_map": str(mrun), "learned_reloc": True})
    problems = a.validate(SensorMode.STEREO, ds, s)
    assert len(problems) == 1 and "patch 0007" in problems[0]
    # the pre-0006 image still refuses map reuse
    R.stdout = f"spslam3_commit={sp.SPSLAM3_COMMIT}\ndriver_source_sha1=463cc0adb1f40a28ef3c26bb22d320dea6089a30\n"
    a.version_info()
    assert any("map reuse needs" in p for p in a.validate(SensorMode.STEREO, ds, ctx.settings))


def test_legacy_r4_image_only_without_descriptor_recovery(tmp_path, monkeypatch):
    r4 = "af171357f101d83cf7599de2665aa54b6ed77050"
    assert r4 in sp.LEARNED_RELOC_NO_RECOVERY_DRIVER_SOURCE_SHA1 and r4 in sp.MAP_REUSE_LEGACY_DRIVER_SOURCE_SHA1
    assert sp.driver_source_sha1() != r4

    class R:
        returncode = 0
        stdout = f"spslam3_commit={sp.SPSLAM3_COMMIT}\ndriver_source_sha1={r4}\n"

    def run(cmd, **kw):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return R()
    monkeypatch.setattr(sp.subprocess, "run", run)
    mrun = _mapping_run(tmp_path, {"global_descriptors": True})
    a, ds, ctx = _ctx(tmp_path, {"load_map": str(mrun)})
    a.options["apptainer"] = "apptainer"
    assert "r4" in a.version_info()["legacy_driver_source"]
    base = {"load_map": str(mrun), "learned_reloc": True}
    problems = a.validate(SensorMode.STEREO, ds, a.resolve_settings(base))
    assert len(problems) == 1 and "learned_reloc_recover_descriptors" in problems[0]
    for ok in ({**base, "learned_reloc_recover_descriptors": False}, {**base, "learned_reloc_verifier": "bow"},
               {"global_descriptors": True}):
        assert a.validate(SensorMode.STEREO, ds, a.resolve_settings(ok)) == [], ok


def test_summarize_learned_reloc_log(tmp_path):
    header = ("timestamp,frame_id,mode,n_db,desc_ms,retrieval_ms,verify_ms,total_ms,cand_kf_ids,cand_sims,"
              "cand_lg_matches,cand_mp_matches,cand_pnp_inliers,cand_final_inliers,cand_verify_ms,best_rank,"
              "best_kf_id,best_inliers,accepted,consistent_count,confirmed,dist_to_prev,angle_to_prev,"
              "tx,ty,tz,qx,qy,qz,qw")
    rows = ["1.000000000,1,lost,100,8.0,0.2,30.0,38.2,5;9,0.61;0.55,120;40,90;20,70;0,80;0,10.0;5.0,0,5,80,1,1,0,"
            "-1,-1,0,0,0,0,0,0,1",
            "1.100000000,2,lost,100,9.0,0.2,32.0,41.2,5;9,0.62;0.50,10;4,3;1,0;0,0;0,4.0;2.0,-1,-1,-1,0,0,0,"
            "-1,-1,,,,,,,",
            "1.200000000,3,lost,100,7.0,0.3,31.0,38.3,6,0.70,150,95,80,90,12.0,0,6,90,1,1,1,0.05,1.0,"
            "0,0,0,0,0,0,1"]
    (tmp_path / "learned_reloc.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    s = sp.summarize_learned_reloc(tmp_path / "learned_reloc.csv")
    assert s["n_attempts"] == 3 and s["n_attempts_lost"] == 3
    assert s["n_accepted_single_frame"] == 2 and s["n_confirmed"] == 1
    assert s["first_confirmed_timestamp_s"] == pytest.approx(1.2)
    assert s["n_with_verified_candidate"] == 2 and s["n_database_keyframes"] == 100
    assert s["desc_ms"]["median"] == pytest.approx(8.0)
    assert s["best_inliers"]["max"] == 90
    assert s["top1_similarity"]["median"] == pytest.approx(0.62)


def test_container_definition_bakes_mixvpr():
    definition = (REPO / "third_party" / "spslam3" / "spslam3.def").read_text()
    assert "/opt/SP_SLAM3/mixvpr.pt" in definition and "mixvpr_model_sha256" in definition
    assert (REPO / "third_party" / "spslam3" / "patches" / "0007-learned-relocalization.patch").exists()


def test_diagnose_classifies_retrieval_and_verification():
    import numpy as np
    from slambench.evaluation.learned_reloc import diagnose

    def T(x):
        m = np.eye(4)
        m[0, 3] = x
        return m
    # session A: keyframes 0..4 at x = 0, 10, 20, 30, 40 m; session B query frames at x = 0, 20, 31
    gt_a = (np.arange(5, dtype=np.int64) * 10**9, np.stack([T(10.0 * i) for i in range(5)]))
    gt_b = (np.arange(3, dtype=np.int64) * 10**9 + 100 * 10**9, np.stack([T(0.0), T(20.0), T(31.0)]))
    kf_ts = {i: i * 10**9 for i in range(5)}

    def row(ts, cands, accepted, pose_x=None, confirmed=False):
        return {"timestamp_ns": ts, "frame_id": ts // 10**9, "mode": "lost", "n_db": 5, "desc_ms": 3.0,
                "retrieval_ms": 0.2, "verify_ms": 50.0, "total_ms": 53.2, "cand_kf_ids": cands,
                "cand_sims": [0.5] * len(cands), "cand_lg_matches": [100] * len(cands),
                "cand_mp_matches": [80] * len(cands), "cand_pnp_inliers": [0] * len(cands),
                "cand_final_inliers": [0] * len(cands), "best_rank": 0 if accepted else -1,
                "best_kf_id": cands[0] if accepted else -1, "best_inliers": 60 if accepted else -1,
                "accepted": accepted, "consistent_count": 1, "confirmed": confirmed,
                "pose": None if pose_x is None else np.array([pose_x, 0, 0, 0, 0, 0, 1.0])}
    rows = [row(100 * 10**9, [0, 3], True, 0.1, True),        # correct accept
            row(101 * 10**9, [4, 3], False),                   # retrieval miss (query at 20 m)
            row(102 * 10**9, [3, 1], False),                   # positive (30 m, 1 m away) but not verified
            ]
    rows.append(row(102 * 10**9, [3], True, 25.0))           # wrong-place accept
    rows[0]["cand_final_inliers"] = [60, 0]
    res = diagnose(rows, kf_ts, gt_a, gt_b, np.eye(3), np.zeros(3), np.eye(4), positive_m=2.0)
    outs = [d["outcome"] for d in res["attempts"]]
    assert outs == ["accepted", "retrieval_miss", "verification_fail", "accepted"]
    assert res["attempts"][0]["err_m"] == pytest.approx(0.1)
    assert res["attempts"][2]["cand_positive"] == [True, False]
    s = res["summary"]
    assert s["recall_at_1"] == pytest.approx(0.75) and s["recall_at_k"] == pytest.approx(0.75)
    assert s["n_accepted_single_frame"] == 2 and s["n_false_single_frame_accepts"] == 1
    assert s["n_confirmed"] == 1 and s["n_false_confirmed"] == 0
    assert s["accept_rate_given_positive_retrieved"] == pytest.approx(2 / 3)
    pc = s["positive_candidates"]
    assert (pc["n"], pc["n_pass_gate"], pc["min_inliers"]) == (3, 1, 50)
    assert pc["lg_median"] == 100 and pc["final_median"] == 0
