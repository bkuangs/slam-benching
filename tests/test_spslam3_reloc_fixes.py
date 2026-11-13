"""SP_SLAM3 patch 0008 settings: unit-norm descriptors, keyframe density, map anchoring."""

import cv2
import pytest

from slambench.algorithms import spslam3 as sp
from slambench.types import SensorMode

from test_spslam3_learned_reloc import _ctx, _mapping_run, _settings_text


def _fs(a, ctx):
    a.prepare(ctx)
    return cv2.FileStorage(str(ctx.algo_dir / sp.SETTINGS_FILE), cv2.FILE_STORAGE_READ)


def test_unused_fix_settings_leave_settings_and_file_unchanged(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    assert not set(ctx.settings) & set(sp.SP_FIX_DEFAULTS)
    text = _settings_text(a, ctx)
    for key in ("SuperPoint.unitDescriptors", "bowDescriptorScale", "SPmatcher.", "Tracking.kf", "MapAnchor"):
        assert key not in text
    a2, _, ctx2 = _ctx(tmp_path, {"sp_unit_descriptors": False, "map_anchor": False})
    ctx2.algo_dir = tmp_path / "algo2"
    assert _settings_text(a2, ctx2) == text


def test_unit_descriptors_use_calibrated_thresholds_and_bow_scale(tmp_path):
    a, ds, ctx = _ctx(tmp_path, {"sp_unit_descriptors": True, "sp_n_features": 1600})
    fs = _fs(a, ctx)
    assert int(fs.getNode("SuperPoint.unitDescriptors").real()) == 1
    assert fs.getNode("SPmatcher.thLow").real() == pytest.approx(sp.SP_UNIT_TH_LOW)
    assert fs.getNode("SPmatcher.thHigh").real() == pytest.approx(sp.SP_UNIT_TH_HIGH)
    assert fs.getNode("SuperPoint.bowDescriptorScale").real() == pytest.approx((256 / 1600) ** 0.5)
    assert sp.SP_UNIT_TH_LOW < sp.SP_UNIT_TH_HIGH
    # explicit values win; keyframe-density keys are written only when set
    a, ds, ctx = _ctx(tmp_path, {"sp_unit_descriptors": True, "sp_th_low": 0.6, "sp_th_high": 1.0,
                                 "sp_bow_descriptor_scale": 1.0, "kf_max_points": 300, "kf_ref_ratio": 0.9})
    fs = _fs(a, ctx)
    assert fs.getNode("SPmatcher.thLow").real() == pytest.approx(0.6)
    assert fs.getNode("SuperPoint.bowDescriptorScale").real() == pytest.approx(1.0)
    assert int(fs.getNode("Tracking.kfMaxPoints").real()) == 300
    assert fs.getNode("Tracking.kfRefRatio").real() == pytest.approx(0.9)
    assert fs.getNode("Tracking.kfCloseTracked").empty()


@pytest.mark.parametrize("bad", [{"sp_unit_descriptors": 1}, {"sp_th_low": 0.9, "sp_th_high": 0.5},
                                 {"sp_th_low": -1.0}, {"kf_max_points": 1.5}, {"map_anchor": True},
                                 {"map_anchor_min_inliers": 5}])
def test_rejected_fix_settings(tmp_path, bad):
    with pytest.raises(ValueError):
        _ctx(tmp_path, bad)


def test_map_anchor_settings_and_map_consistency(tmp_path):
    run = _mapping_run(tmp_path, {"global_descriptors": True, "sp_unit_descriptors": True})
    a, ds, ctx = _ctx(tmp_path, {"load_map": str(run), "learned_reloc": True, "sp_unit_descriptors": True,
                                 "map_anchor": True, "map_anchor_k": 2})
    fs = _fs(a, ctx)
    assert int(fs.getNode("MapAnchor.enable").real()) == 1 and int(fs.getNode("MapAnchor.k").real()) == 2
    assert int(fs.getNode("MapAnchor.minInliers").real()) == 30
    assert fs.getNode("MapAnchor.logFile").string() == str((ctx.algo_dir / sp.MAP_ANCHOR_LOG).resolve())
    assert a.validate(SensorMode.STEREO, ds, ctx.settings) == []
    # a session B whose descriptors differ from the map's is refused (map-reuse settings check)
    s = a.resolve_settings({"load_map": str(run), "learned_reloc": True})
    problems = a.validate(SensorMode.STEREO, ds, s)
    assert len(problems) == 1 and "sp_unit_descriptors: map=True here=None" in problems[0]
    # matcher thresholds and keyframe policy are not stored in the map, so session B may change them
    s = a.resolve_settings({"load_map": str(run), "learned_reloc": True, "sp_unit_descriptors": True,
                            "sp_th_high": 0.9, "kf_ref_ratio": 0.9})
    assert a.validate(SensorMode.STEREO, ds, s) == []


def test_legacy_r5_image_runs_learned_reloc_but_not_patch_0008(tmp_path, monkeypatch):
    r5 = "9be049ce57cce22a2e59d986c38c09fc1f0c4fca"
    assert r5 in sp.LEGACY_DRIVER_SOURCE_SHA1 and sp.driver_source_sha1() != r5

    class R:
        returncode = 0
        stdout = f"spslam3_commit={sp.SPSLAM3_COMMIT}\ndriver_source_sha1={r5}\n"

    def run(cmd, **kw):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return R()
    monkeypatch.setattr(sp.subprocess, "run", run)
    mrun = _mapping_run(tmp_path, {"global_descriptors": True})
    a, ds, ctx = _ctx(tmp_path, {"load_map": str(mrun), "learned_reloc": True})
    a.options["apptainer"] = "apptainer"
    assert "r5" in a.version_info()["legacy_driver_source"]
    assert a.validate(SensorMode.STEREO, ds, ctx.settings) == []
    s = a.resolve_settings({"load_map": str(mrun), "learned_reloc": True, "map_anchor": True})
    assert "map_anchor" in sp.mr._FREE_SETTINGS
    problems = a.validate(SensorMode.STEREO, ds, s)
    assert len(problems) == 1 and "patch 0008" in problems[0]


def test_summarize_map_anchor_log(tmp_path):
    p = tmp_path / sp.MAP_ANCHOR_LOG
    p.write_text("timestamp,frame_id,kf_ids,lg_matches,mp_matches,pose_inliers,local_inliers,"
                 "correction_m,correction_deg,accepted,ms\n"
                 "1.0,10,3;4,200,80,60,120,0.4,1.0,1,30.0\n"
                 "2.0,11,3,50,5,0,0,-1,-1,0,12.0\n")
    out = sp.summarize_map_anchor(p)
    assert out["n_attempts"] == 2 and out["n_accepted"] == 1
    assert out["accepted_correction_m"]["median"] == pytest.approx(0.4)


def test_reloc_monitor_settings(tmp_path):
    run = _mapping_run(tmp_path, {"global_descriptors": True})
    base = {"load_map": str(run), "learned_reloc": True}
    a, ds, ctx = _ctx(tmp_path, base)
    text = _settings_text(a, ctx)
    assert "RelocMonitor" not in text and "reloc_monitor" not in ctx.settings
    a2, _, ctx2 = _ctx(tmp_path, {**base, "reloc_monitor": False})
    assert _settings_text(a2, ctx2) == text
    a, ds, ctx = _ctx(tmp_path, {**base, "reloc_monitor": True, "reloc_monitor_period": 10,
                                 "reloc_monitor_max_dev_m": 1.5, "reloc_monitor_max_vo_s": 2})
    fs = _fs(a, ctx)
    assert int(fs.getNode("RelocMonitor.enable").real()) == 1
    assert int(fs.getNode("RelocMonitor.period").real()) == 10
    assert fs.getNode("RelocMonitor.maxDevM").real() == pytest.approx(1.5)
    assert fs.getNode("RelocMonitor.maxDevDeg").real() == pytest.approx(3.0)
    assert fs.getNode("RelocMonitor.maxVOs").real() == pytest.approx(2.0)
    assert fs.getNode("RelocMonitor.maxUnverifiedS").real() == 0
    assert fs.getNode("RelocMonitor.logFile").string() == str((ctx.algo_dir / sp.RELOC_MONITOR_LOG).resolve())
    # session-B only: the map run did not use it
    assert a.validate(SensorMode.STEREO, ds, ctx.settings) == []


@pytest.mark.parametrize("bad", [{"reloc_monitor": True}, {"reloc_monitor": 1},
                                 {"reloc_monitor_period": 0}, {"reloc_monitor_max_dev_m": 0},
                                 {"reloc_monitor_max_vo_s": -1.0}])
def test_rejected_reloc_monitor_settings(tmp_path, bad):
    with pytest.raises(ValueError):
        _ctx(tmp_path, bad)


def test_legacy_r6_image_runs_patch_0008_but_not_the_monitor(tmp_path, monkeypatch):
    r6 = "c9516337cbad1d13b52bb9afbf7e24c6d637be36"
    assert r6 in sp.LEGACY_DRIVER_SOURCE_SHA1 and sp.driver_source_sha1() != r6

    class R:
        returncode = 0
        stdout = f"spslam3_commit={sp.SPSLAM3_COMMIT}\ndriver_source_sha1={r6}\n"

    def run(cmd, **kw):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return R()
    monkeypatch.setattr(sp.subprocess, "run", run)
    mrun = _mapping_run(tmp_path, {"global_descriptors": True, "sp_unit_descriptors": True})
    base = {"load_map": str(mrun), "learned_reloc": True, "sp_unit_descriptors": True, "map_anchor": True}
    a, ds, ctx = _ctx(tmp_path, base)
    a.options["apptainer"] = "apptainer"
    assert "r6" in a.version_info()["legacy_driver_source"]
    assert a.validate(SensorMode.STEREO, ds, ctx.settings) == []
    problems = a.validate(SensorMode.STEREO, ds, a.resolve_settings({**base, "reloc_monitor": True}))
    assert len(problems) == 1 and "patch 0009" in problems[0]


def test_summarize_reloc_monitor_log(tmp_path):
    p = tmp_path / sp.RELOC_MONITOR_LOG
    p.write_text("timestamp,frame_id,trigger,tracked_inliers,verified,verified_inliers,dev_m,dev_deg,decision,"
                 "local_inliers,since_verified_s,ms\n"
                 "1.0,10,period,300,vpr,120,0.5,0.4,agree,0,0.000,250.0\n"
                 "2.0,25,period,280,prior,90,6.0,4.0,reset,150,0.000,270.0\n"
                 "3.0,40,period,200,0,30,-1,-1,unverified,0,1.000,240.0\n"
                 "4.0,41,unverified_timeout,200,0,0,-1,-1,lost,0,2.000,0.1\n")
    out = sp.summarize_reloc_monitor(p)
    assert out["n_checks"] == 3 and out["n_verified"] == 2 and out["n_verified_by_prior"] == 1
    assert out["n_reset"] == 1 and out["n_agree"] == 1 and out["n_unverified"] == 1 and out["n_lost"] == 1
    assert out["verified_dev_m"]["max"] == pytest.approx(6.0)
