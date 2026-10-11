"""Replay export: sessions stay separate, errors match the evaluator, raw fallback is labelled."""

import json
from pathlib import Path

import numpy as np
import pytest

rr = pytest.importorskip("rerun")
rd = pytest.importorskip("rerun.dataframe")

from slambench.viz import rerun_export  # noqa: E402

from test_online import MAP_POINTS_GT, register, run  # noqa: E402,F401  (fixture + scenario runner)


def _entities(path):
    return {c.entity_path for c in rd.load_recording(str(path)).schema().component_columns()}


def _scalars(path, entity):
    tbl = rd.load_recording(str(path)).view(index="frame", contents=entity).select().read_all()
    col = [c for c in tbl.column_names if "Scalars:scalars" in c][0]
    return np.concatenate([np.asarray(x) for x in tbl[col].to_pylist() if x is not None])


def test_merge_run_replay_has_separate_sessions_and_matching_errors(tmp_path):
    import cv2
    d, st, m = run(tmp_path, "merge")
    prep = json.loads((d / "prepared_input.json").read_text())["directory"]
    frames = [l.split(",") for l in open(f"{prep}/frames.csv").read().splitlines()[1:]]
    for t, left, right in frames[:10]:           # only the first 10 frames have images
        for p in (left, right):
            out = Path(prep) / p
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.full((24, 32, 3), 128, np.uint8))
    rep = rerun_export.export_run(d, tmp_path / "r.rrd", stride=5, log=lambda *_: None)
    assert rep["sessions"] == {"final": [0], "online": [0, 1]} and rep["aligned"] == {"final": True, "online": True}
    assert rep["n_images"] == 4 and rep["n_images_missing"] == 2 * (len(range(0, 300, 5)) - 2)
    ents = _entities(tmp_path / "r.rrd")
    assert {"/world/online/session_0/path", "/world/online/session_1/path", "/world/final/path",
            "/world/ground_truth/path", "/camera/left", "/camera/right", "/events/session",
            "/plots/tracking/coordinate_session"} <= ents
    assert not any("session_-1" in e for e in ents)
    for s in m["accuracy"]["online"]["sessions"]:
        v = _scalars(tmp_path / "r.rrd", f"/plots/error/online/session_{s['session']['session_id']}")
        assert len(v) == s["ate"]["translation_m"]["n"]
        assert np.sqrt(np.mean(v ** 2)) == pytest.approx(s["ate"]["translation_m"]["rmse"], abs=1e-9)


def test_raw_unaligned_fallback_is_labelled(tmp_path, monkeypatch):
    d, st, m = run(tmp_path, "map_reset")
    import slambench.runner as runner
    monkeypatch.setattr(runner, "make_dataset", lambda cfg: (_ for _ in ()).throw(FileNotFoundError("gone")))
    rep = rerun_export.export_run(d, tmp_path / "raw.rrd", images=False, log=lambda *_: None)
    assert rep["aligned"] == {"final": False, "online": False} and "raw unaligned" in rep["note"]
    assert rep["sessions"]["online"] == [0, 1]   # map ids as sessions (legacy log without coord_session)
    ents = _entities(tmp_path / "raw.rrd")
    assert not any(e.startswith("/plots/error") for e in ents)


def _column(path, entity, component, static=False):
    tbl = rd.load_recording(str(path)).view(index=None if static else "frame",
                                             contents=entity).select().read_all()
    col = [c for c in tbl.column_names if component in c][0]
    return [x for x in tbl[col].to_pylist() if x is not None]


def test_sparse_map_lands_on_ground_truth_and_features_overlay(tmp_path):
    import cv2
    from slambench.artifacts import read_tracked_features
    d, st, m = run(tmp_path, "drift")
    out = json.loads((d / "algorithm_output.json").read_text())
    assert set(out["artifacts"]) == {"tracked_features", "sparse_map_points", "sparse_map_keyframes"}
    assert out["extra"]["sparse_map"] == {"n_points": 41, "n_points_in_final_frame": 40, "n_maps": 2}
    feats, truncated = read_tracked_features(Path(out["artifacts"]["tracked_features"]))
    assert truncated and len(feats) == 20
    prep = json.loads((d / "prepared_input.json").read_text())["directory"]
    left = Path(prep) / open(f"{prep}/frames.csv").read().splitlines()[1].split(",")[1]
    left.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(left), np.zeros((40, 1280, 3), np.uint8))       # wider than max_width: scaled by 0.5
    rep = rerun_export.export_run(d, tmp_path / "m.rrd", stride=1, log=lambda *_: None)
    assert rep["map_points_shown"] == 40 and rep["features_frames"] == 20
    pts = np.asarray(_column(tmp_path / "m.rrd", "/world/final/map_points", "Points3D:positions",
                                static=True)[0])
    assert np.allclose(pts, MAP_POINTS_GT, atol=1e-4)               # final alignment puts the map on GT
    xy = np.asarray(_column(tmp_path / "m.rrd", "/camera/left/features", "Points2D:positions")[0])
    assert np.allclose(xy[:, 0], (10.0 + np.arange(10)) * 0.5) and np.allclose(xy[:, 1], 10.0)
    inl = np.concatenate([np.asarray(x) for x in _column(tmp_path / "m.rrd", "/plots/features/inliers",
                                                          "Scalars:scalars")])
    assert len(inl) == 20 and np.all(inl == 5)
