"""Map reuse (atlas save / load) for the ORB-SLAM3-family adapters: settings, generated files,
command, map-frame bookkeeping, collection and fixed-frame scoring (no image)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from slambench.algorithms import orbslam3 as orb
from slambench.algorithms import orbslam3_map_reuse as mr
from slambench.algorithms import spslam3 as sp
from slambench.algorithms.base import RunContext
from slambench.algorithms.orbslam3 import OrbSlam3Adapter
from slambench.algorithms.spslam3 import SpSlam3Adapter
from slambench.types import RunStatus, SensorMode

from synthetic import FakeDataset, sample

ADAPTERS = [OrbSlam3Adapter, SpSlam3Adapter]
SETTINGS_FILE = {OrbSlam3Adapter: "orbslam3_settings.yaml", SpSlam3Adapter: sp.SETTINGS_FILE}


def _ctx(tmp_path, settings=None, adapter=OrbSlam3Adapter, name="s1"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds = FakeDataset(tmp_path, name)
    (tmp_path / "prep").mkdir(exist_ok=True)
    prep = ds.prepare_offline(SensorMode.STEREO, tmp_path / "prep")
    a = adapter({"container": str(tmp_path / "x.sif")})
    return a, ds, RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.STEREO, prep,
                             a.resolve_settings(settings or {}))


def _kf_csv(path, poses):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map_id", "keyframe_id", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
                    "in_final_frame"])
        for i, p in enumerate(poses):
            w.writerow([0, i, i, *p, 1])


KF_POSES = [[0, 0, 0, 0, 0, 0, 1], [1.0, 0, 0, 0, 0, 0, 1], [2.0, 0.5, 0, 0, 0, 0, 1]]


def _mapping_run(root, adapter=OrbSlam3Adapter, settings=None, run_id="20240101T000000", status="success"):
    """runs/<config>/<run id>/ of a finished mapping session (save_map: true)."""
    run = root / "runs" / "map_cfg" / run_id
    (run / "algorithm").mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": status}))
    s = adapter({}).resolve_settings({"save_map": True, **(settings or {})})
    (run / "config.resolved.yaml").write_text(yaml.safe_dump({"algorithm": {"name": adapter.name, "settings": s}}))
    (run / "algorithm" / "atlas.osa").write_bytes(b"atlas")
    _kf_csv(run / "algorithm" / "keyframes.csv", KF_POSES)
    return run


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_unused_map_reuse_settings_leave_files_and_command_unchanged(tmp_path, adapter):
    a, ds, ctx = _ctx(tmp_path / "a", adapter=adapter)
    assert not set(ctx.settings) & set(mr.MAP_REUSE_DEFAULTS)
    a.prepare(ctx)
    text = (ctx.algo_dir / SETTINGS_FILE[adapter]).read_text()
    assert "System." not in text and not (ctx.algo_dir / "map_source.json").exists()
    cmd = a.command(ctx)
    assert "--relocalize-only" not in cmd and not any(c.endswith(":ro") for c in cmd)
    # explicit defaults resolve to the same files and command
    b, _, ctx2 = _ctx(tmp_path / "b", {"save_map": False, "load_map": None}, adapter=adapter)
    b.prepare(ctx2)
    assert (ctx2.algo_dir / SETTINGS_FILE[adapter]).read_text() == text.replace(str(tmp_path / "a"),
                                                                                  str(tmp_path / "b"))
    assert b.command(ctx2) == [c.replace(str(tmp_path / "a"), str(tmp_path / "b")) for c in cmd]
    assert a._map_reuse_validate(SensorMode.STEREO, ctx.settings) == []
    assert b._map_reuse_validate(SensorMode.STEREO, ctx2.settings) == []


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_rejected_map_reuse_settings(tmp_path, adapter):
    a = adapter({"container": str(tmp_path / "x.sif")})
    for bad in ({"save_map": "yes"}, {"reuse_mode": "fuse"},
                {"save_map": True, "load_map": str(tmp_path), "reuse_mode": "relocalize"}):
        with pytest.raises(ValueError):
            a.resolve_settings(bad)
    s = a.resolve_settings({"save_map": True, "load_map": str(tmp_path), "reuse_mode": "merge"})
    assert s["save_map"] and s["reuse_mode"] == "merge"


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_save_map_settings_line(tmp_path, adapter):
    a, ds, ctx = _ctx(tmp_path, {"save_map": True}, adapter=adapter)
    a.prepare(ctx)
    text = (ctx.algo_dir / SETTINGS_FILE[adapter]).read_text()
    assert text.splitlines()[-1] == f'System.SaveAtlasToFile: "{ctx.algo_dir.resolve() / "atlas"}"'
    assert "System.LoadAtlasFromFile" not in text
    assert "--relocalize-only" not in a.command(ctx)


def test_latest_successful_mapping_run_is_resolved(tmp_path):
    old = _mapping_run(tmp_path, run_id="20240101T000000")
    _mapping_run(tmp_path, run_id="20240301T000000", status="crash")
    assert mr.resolve_map_reuse_settings(dict(mr.MAP_REUSE_DEFAULTS, load_map=str(old.parent)))["load_map"] == str(old)
    new = _mapping_run(tmp_path, run_id="20240201T000000")
    s = mr.resolve_map_reuse_settings(dict(mr.MAP_REUSE_DEFAULTS, load_map=str(old.parent)))
    assert s["load_map"] == str(new)
    assert mr.atlas_to_load(s) == new / "algorithm" / "atlas.osa"
    assert mr.mapping_run_dir(mr.atlas_to_load(s)) == new
    for p in (new / "algorithm", new / "algorithm" / "atlas.osa"):
        assert mr.atlas_to_load({"load_map": str(p)}) == new / "algorithm" / "atlas.osa"


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("mode", ["relocalize", "merge"])
def test_load_map_files_command_and_checks(tmp_path, adapter, mode):
    run = _mapping_run(tmp_path, adapter)
    a, ds, ctx = _ctx(tmp_path / "loc", {"load_map": str(run.parent), "reuse_mode": mode}, adapter=adapter)
    (tmp_path / "loc" / "x.sif").write_bytes(b"")
    assert a.validate(SensorMode.STEREO, ds, ctx.settings) == []
    a.prepare(ctx)
    text = (ctx.algo_dir / SETTINGS_FILE[adapter]).read_text()
    assert text.splitlines()[-1] == f'System.LoadAtlasFromFile: "{run / "algorithm" / "atlas"}"'
    src = json.loads((ctx.algo_dir / "map_source.json").read_text())
    assert src["map_run"] == str(run) and src["atlas_sha256_before"] == mr.file_sha256(run / "algorithm" / "atlas.osa")
    cmd = a.command(ctx)
    assert ("--relocalize-only" in cmd) == (mode == "relocalize")
    assert f"{run / 'algorithm'}:{run / 'algorithm'}:ro" in cmd
    assert cmd.index(f"{run / 'algorithm'}:{run / 'algorithm'}:ro") < cmd.index(str(a.container))

    other = OrbSlam3Adapter if adapter is SpSlam3Adapter else SpSlam3Adapter
    (run / "config.resolved.yaml").write_text(yaml.safe_dump({"algorithm": {"name": other.name, "settings": {}}}))
    assert "saved by" in a.validate(SensorMode.STEREO, ds, ctx.settings)[0]
    key = "orb_n_features" if adapter is OrbSlam3Adapter else "sp_n_features"
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(
        {"algorithm": {"name": adapter.name, "settings": adapter({}).resolve_settings({"save_map": True, key: 5})}}))
    assert key in a.validate(SensorMode.STEREO, ds, ctx.settings)[0]
    (run / "algorithm" / "atlas.osa").unlink()
    assert "not found" in a.validate(SensorMode.STEREO, ds, ctx.settings)[0]


def test_merge_needs_loop_closing_and_visual_only(tmp_path):
    run = _mapping_run(tmp_path, settings={"loop_closure": False})
    a, ds, ctx = _ctx(tmp_path / "loc", {"load_map": str(run), "reuse_mode": "merge", "loop_closure": False})
    (tmp_path / "loc" / "x.sif").write_bytes(b"")
    assert "needs loop_closure" in a.validate(SensorMode.STEREO, ds, ctx.settings)[0]
    assert "visual-only" in " ".join(a._map_reuse_validate(SensorMode.STEREO_INERTIAL, ctx.settings))


@pytest.mark.parametrize("module,adapter,legacy", [(orb, OrbSlam3Adapter, "f11551745c3712629ce54436cbcc9eec685d5992"),
                                                   (sp, SpSlam3Adapter, "463cc0adb1f40a28ef3c26bb22d320dea6089a30")])
def test_legacy_image_accepted_without_map_reuse_only(tmp_path, monkeypatch, module, adapter, legacy):
    assert legacy in module.LEGACY_DRIVER_SOURCE_SHA1
    commit = orb.ORBSLAM3_COMMIT if adapter is OrbSlam3Adapter else sp.SPSLAM3_COMMIT
    key = "orbslam3_commit" if adapter is OrbSlam3Adapter else "spslam3_commit"

    class R:
        returncode = 1
        stdout = f"{key}={commit}\ndriver_source_sha1={legacy}\n"
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: R())
    a, ds, ctx = _ctx(tmp_path, adapter=adapter)
    info = a.version_info()
    assert info["legacy_driver_source"].endswith("before map reuse (patch 0006)")
    assert a._map_reuse_validate(SensorMode.STEREO, ctx.settings) == []
    s = a.resolve_settings({"save_map": True})
    assert "needs an image built from this repository's driver" in a._map_reuse_validate(SensorMode.STEREO, s)[0]


def test_map_frame_check(tmp_path):
    _kf_csv(tmp_path / "a.csv", KF_POSES)
    _kf_csv(tmp_path / "b.csv", KF_POSES)
    assert mr.map_frame_check(tmp_path / "a.csv", tmp_path / "b.csv")["same_frame"] is True
    _kf_csv(tmp_path / "b.csv", [KF_POSES[0], KF_POSES[1], [2.0, 0.6, 0, 0, 0, 0, 1]])
    c = mr.map_frame_check(tmp_path / "a.csv", tmp_path / "b.csv")
    assert c["same_frame"] is False and abs(c["max_translation_diff_m"] - 0.1) < 1e-9
    assert mr.map_frame_check(tmp_path / "a.csv", tmp_path / "missing.csv")["same_frame"] is False


def _row(i, state, pose, **kw):
    return {"frame_idx": i, "timestamp_ns": 1000 + i, "state": state, "pose": None if pose is None else np.array(pose),
            "track_time_s": 0.01, **kw}


def test_map_reuse_rows_transform_and_merge_exclusion():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T[:3, 3] = [1, 2, 3]
    p = [1.0, 0, 0, 0, 0, 0, 1]
    log = [_row(0, 4, None, in_loaded_map=True, relocalized=False),
           _row(1, 2, p, in_loaded_map=True, relocalized=True),
           _row(2, 2, p, in_loaded_map=True, relocalized=False, vo_only=True),
           _row(3, 3, p, in_loaded_map=True, relocalized=False)]
    rows = mr.map_reuse_rows(log, T, "relocalize")
    assert [r["localized"] for r in rows] == [False, True, True, False]
    assert [r["loc_event"] for r in rows] == [False, True, False, False]
    assert np.allclose(rows[1]["pose"][:3], [1, 3, 3])
    assert np.allclose(Rotation.from_quat(rows[1]["pose"][3:]).as_matrix(), T[:3, :3])
    assert rows[2]["vo_only"] and rows[0]["pose"] is None

    log = [_row(0, 2, p, in_loaded_map=False), _row(1, 2, p, in_loaded_map=False),
           _row(2, 2, p, in_loaded_map=True), _row(3, 2, p, in_loaded_map=True)]
    rows = mr.map_reuse_rows(log, np.eye(4), "merge")
    assert [r["localized"] for r in rows] == [False, False, True, True]
    assert [r["loc_event"] for r in rows] == [False, False, True, False]
    assert rows[0]["pose"] is None


def _write_session_b(ctx, ds, run, T_final_map, first_loc=5, mode="relocalize"):
    """Driver outputs of a session that tracked GT poses expressed in the saved map's frame."""
    d = ctx.algo_dir
    d.mkdir(parents=True, exist_ok=True)
    gt = sample(ds.gt, ds.frames)
    T_map_final = np.linalg.inv(T_final_map)
    cols = ["frame_idx", "timestamp_ns", "state", "map_id", "map_changed", "track_time_s", "coord_session",
            "map_kfs", "map_points", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "in_loaded_map", "relocalized",
            "vo_only"]
    with open(d / "frame_log.csv", "w", newline="") as f, open(d / "trajectory_euroc.txt", "w") as g:
        w = csv.writer(f)
        w.writerow(cols)
        for i, t in enumerate(ds.frames):
            ok = i >= first_loc
            Tw = np.eye(4)
            Tw[:3, :3] = Rotation.from_quat(gt.quaternions_xyzw[i]).as_matrix()
            Tw[:3, 3] = gt.positions[i]
            Tm = T_map_final @ Tw
            pose = list(Tm[:3, 3]) + list(Rotation.from_matrix(Tm[:3, :3]).as_quat())
            in_map = 1 if mode == "relocalize" or ok else 0
            state = 2 if (ok or mode == "merge") else 4
            w.writerow([i, t, state, 0, 0, 0.02, 0, 3, 10, *(pose if state == 2 else [""] * 7), in_map,
                        int(mode == "relocalize" and i == first_loc), 0])
            if state == 2:
                g.write(f"{float(t):.6f} " + " ".join(f"{v:.9f}" for v in pose) + "\n")
    (d / "driver_status.json").write_text(json.dumps({"status": "completed", "message": ""}))
    (d / "loaded_map.json").write_text(json.dumps({"map_id": 0, "n_keyframes": 3, "n_map_points": 10,
                                                   "init_time_s": 1.5, "T_final_map": T_final_map.tolist()}))
    _kf_csv(d / "loaded_map" / "keyframes.csv", KF_POSES)


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("mode", ["relocalize", "merge"])
def test_collect_session_b_and_fixed_frame_metrics(tmp_path, adapter, mode):
    from slambench.evaluation.map_reuse import map_reuse_metrics
    run = _mapping_run(tmp_path, adapter)
    a, ds, ctx = _ctx(tmp_path / "loc", {"load_map": str(run), "reuse_mode": mode}, adapter=adapter)
    a.prepare(ctx)
    T_final_map = np.eye(4)
    T_final_map[:3, 3] = [0.5, -1, 0]
    _write_session_b(ctx, ds, run, T_final_map, mode=mode)
    out = a.collect(ctx, 0, False)
    assert out.status == RunStatus.SUCCESS, out.message
    m = out.extra["map_reuse"]
    assert m["atlas_unchanged"] and m["map_frame_check"]["same_frame"] and m["reuse_mode"] == mode
    assert m["first_localization_frame_idx"] == 5 and m["n_localized_poses"] == len(ds.frames) - 5
    assert m["n_relocalizations"] == (1 if mode == "relocalize" else 0)
    assert m["merged_into_loaded_map"] == (True if mode == "merge" else None)
    assert m["merge_frame_idx"] == (5 if mode == "merge" else None)

    # GT expressed in the map frame by the mapping run's (here: identity) map->GT transform
    log, _ = a.read_frame_log(ctx.algo_dir / "frame_log.csv")
    rows = mr.map_reuse_rows(log, T_final_map, mode)
    fixed = {"R": np.eye(3), "t": np.zeros(3)}
    gt = sample(ds.gt, ds.gt.timestamps_ns)
    res = map_reuse_metrics(rows, np.array(ds.frames), gt, fixed, {}, {"wall_time_s": 9.0}, m, 0.02, tmp_path)
    assert res["initialization"]["frame_idx"] == 5
    assert res["error_fixed_frame"]["translation_m"]["max"] < 1e-6
    assert res["error_fixed_frame"]["translation_m"]["p95"] < 1e-6
    n = len(ds.frames)
    assert abs(res["coverage"]["localized_fraction"] - (n - 5) / n) < 0.01

    # a modified atlas, a different map frame, or never localizing invalidate the run
    (run / "algorithm" / "atlas.osa").write_bytes(b"changed")
    assert a.collect(ctx, 0, False).status == RunStatus.CRASH
    (run / "algorithm" / "atlas.osa").write_bytes(b"atlas")
    _kf_csv(ctx.algo_dir / "loaded_map" / "keyframes.csv", [KF_POSES[0], KF_POSES[1], [3, 0, 0, 0, 0, 0, 1]])
    assert a.collect(ctx, 0, False).status == RunStatus.CRASH
    _kf_csv(ctx.algo_dir / "loaded_map" / "keyframes.csv", KF_POSES)
    _write_session_b(ctx, ds, run, T_final_map, first_loc=len(ds.frames), mode=mode)
    assert a.collect(ctx, 0, False).status == RunStatus.INIT_FAILURE


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_collect_saving_session_reports_map_size(tmp_path, adapter):
    a, ds, ctx = _ctx(tmp_path, {"save_map": True}, adapter=adapter)
    a.prepare(ctx)
    d = ctx.algo_dir
    gt = sample(ds.gt, ds.frames)
    with open(d / "frame_log.csv", "w", newline="") as f, open(d / "trajectory_euroc.txt", "w") as g:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_ns", "state", "map_id", "track_time_s", "coord_session",
                    "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for i, t in enumerate(ds.frames):
            p = list(gt.positions[i]) + list(gt.quaternions_xyzw[i])
            w.writerow([i, t, 2, 0, 0.02, 0, *p])
            g.write(f"{float(t):.6f} " + " ".join(f"{v:.9f}" for v in p) + "\n")
    (d / "driver_status.json").write_text(json.dumps({"status": "completed", "message": ""}))
    _kf_csv(d / "keyframes.csv", KF_POSES)
    assert a.collect(ctx, 0, False).status == RunStatus.MISSING_OUTPUT
    (d / "atlas.osa").write_bytes(b"x" * 2000)
    out = a.collect(ctx, 0, False)
    assert out.status == RunStatus.SUCCESS, out.message
    sm = out.extra["saved_map"]
    assert sm["exists"] and sm["n_keyframes_final_map"] == 3 and abs(sm["atlas_size_mb"] - 0.002) < 1e-9
    assert out.artifacts["saved_atlas"] == d / "atlas.osa"


def test_map_reuse_configs_are_consistent():
    repo = Path(__file__).resolve().parents[1]
    cfgs = [(p, yaml.safe_load(p.read_text())) for p in sorted(repo.glob("configs/*mapreuse*.yaml"))]
    cfgs = [(p, c) for p, c in cfgs if c["algorithm"]["name"] in ("orbslam3", "spslam3")]
    assert cfgs
    for p, c in cfgs:
        s = c["algorithm"]["settings"]
        assert s["reuse_mode"] in mr.REUSE_MODES and "load_map" in s, p.name
        name = Path(s["load_map"]).name
        mapcfg = yaml.safe_load((repo / "configs" / f"{name}.yaml").read_text())
        ms = mapcfg["algorithm"]["settings"]
        assert ms["save_map"] is True and mapcfg["algorithm"]["name"] == c["algorithm"]["name"], p.name
        diff = {k for k in set(ms) | set(s) if k not in mr._FREE_SETTINGS and ms.get(k) != s.get(k)}
        assert not diff, (p.name, diff)
