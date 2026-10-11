"""Replay export: sessions stay separate, errors match the evaluator, raw fallback is labelled."""

import json
from pathlib import Path

import numpy as np
import pytest

rr = pytest.importorskip("rerun")
rd = pytest.importorskip("rerun.dataframe")

from slambench.viz import rerun_export  # noqa: E402

from test_online import register, run  # noqa: E402,F401  (fixture + scenario runner)


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
