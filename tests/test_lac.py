"""LAC adapter tests: settings, driver settings, command, output collection (no image), and the
driver's LAC-rover -> OpenCV camera pose convention."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from slambench.algorithms import ALGORITHMS
from slambench.algorithms.base import RunContext
from slambench.algorithms.lac import POSE_COLUMNS, LacAdapter, driver_source_sha1
from slambench.trajectory import make_T, read_tum
from slambench.types import RunStatus, SensorMode

from synthetic import FakeDataset, sample

DRIVER = Path(__file__).resolve().parents[1] / "third_party" / "lac" / "slambench_lac_driver.py"
LOG_COLS = ["frame_idx", "timestamp_ns", "state", "keyframe", "n_tracked", "n_new_points", "n_loop_closures",
            "track_time_s", "backend_time_s", "frame_time_s", *POSE_COLUMNS]


def _driver():
    spec = importlib.util.spec_from_file_location("slambench_lac_driver", DRIVER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _vec(T):
    return list(T[:3, 3]) + list(Rotation.from_matrix(T[:3, :3]).as_quat())


def _ctx(tmp_path, settings=None):
    ds = FakeDataset(tmp_path, "s1")
    (tmp_path / "prep").mkdir(exist_ok=True)
    prep = ds.prepare_offline(SensorMode.STEREO, tmp_path / "prep")
    a = LacAdapter({"container": str(tmp_path / "x.sif")})
    return a, ds, RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.STEREO, prep,
                             a.resolve_settings(settings or {}))


def _write_outputs(ctx, ds, failed=(), status="completed"):
    d = ctx.algo_dir
    d.mkdir(parents=True, exist_ok=True)
    gt = sample(ds.gt, ds.frames).poses()
    with open(d / "frame_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLS)
        for i, t in enumerate(ds.frames):
            st = 0 if i == 0 else (2 if i in failed else 1)
            w.writerow([i, t, st, int(i % 10 == 0), 900, 50, 1 if i > 20 else 0, 0.03, 0.002, 0.032, *_vec(gt[i])])
    with open(d / "trajectory_final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", *POSE_COLUMNS])
        for i, t in enumerate(ds.frames):
            w.writerow([t, *_vec(gt[i])])
    (d / "map_points.ply").write_bytes(b"ply\n")
    (d / "loop_closures.csv").write_text("from_pose_idx,to_pose_idx\n0,21\n")
    (d / "driver_status.json").write_text(json.dumps({"status": status, "message": "",
                                                      "gpu": {"name": "Fake GPU", "max_memory_allocated_bytes": 1}}))


def test_registered_and_rejects_bad_modes_settings(tmp_path):
    assert ALGORITHMS["lac"] is LacAdapter
    a, ds, ctx = _ctx(tmp_path)
    s = ctx.settings
    assert "does not support sensor mode 'mono'" in a.validate(SensorMode.MONO, ds, s)[0]
    assert any("container not found" in p for p in a.validate(SensorMode.STEREO, ds, s))
    for bad in ({"nope": 1}, {"keyframe_interval_s": 0}, {"lc_min_matches": 0}, {"lc_min_score": 2},
                {"max_frames": 1.5}, {"playback_rate": -1}, {"cpu_threads": 0},
                {"max_depth_m": 0}, {"min_disparity_px": -1}, {"max_row_diff_px": 0},
                {"vo_reference": "map"}, {"vo_ref_max_flow_px": 0}, {"vo_ref_min_match_frac": 1.5},
                {"frame_stride": 0}, {"frame_stride": 1.5}, {"features": "sift"}, {"orb_n_features": 0}):
        with pytest.raises(ValueError):
            a.resolve_settings(bad)
    assert a.resolve_settings({"max_frames": 5})["max_frames"] == 5
    assert a.resolve_settings({"cpu_threads": None})["cpu_threads"] is None
    assert a.estimate_frame(SensorMode.STEREO) == "cam0"
    assert driver_source_sha1() and len(driver_source_sha1()) == 40


def test_driver_settings_and_command(tmp_path):
    a, ds, ctx = _ctx(tmp_path, {"playback_rate": 0, "loop_closure": False})
    a.prepare(ctx)
    p = json.loads((ctx.algo_dir / "lac_settings.json").read_text())
    c = ds.stereo_calibration()
    assert p["camera"] == {"width": c.left.width, "height": c.left.height, "fx": c.left.fx, "fy": c.left.fy,
                           "cx": c.left.cx, "cy": c.left.cy, "baseline_m": c.baseline_m}
    assert p["playback_rate"] == 0.0 and p["keyframe_interval_s"] == 1.0
    assert p["stereo"] == {"max_depth_m": None, "min_disparity_px": None, "max_row_diff_px": None}
    assert p["vo"] == {"reference": "previous", "max_flow_px": 10.0, "min_match_frac": 0.5}
    assert p["features"] == {"type": "superpoint"}
    assert p["loop_closure"] == {"enabled": False, "min_score": 0.5, "min_matches": 100,
                                 "distance_threshold_m": 1.0, "angle_threshold_deg": 5.0}
    cmd = a.command(ctx)
    assert cmd[:6] == [a.apptainer, "exec", "--nv", "--cleanenv",
                       "--env", "OMP_NUM_THREADS=1,MKL_NUM_THREADS=1"]
    assert cmd[-5:-3] == ["python", "/opt/slambench/slambench_lac_driver.py"]
    assert cmd[-3].endswith("lac_settings.json") and cmd[-1] == str(ctx.algo_dir.resolve())


def test_collect_success(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    _write_outputs(ctx, ds, failed={5, 6})
    out = a.collect(ctx, 0, False)
    assert out.status is RunStatus.SUCCESS, out.message
    kinds = {t.kind: t for t in out.all_trajectories()}
    assert set(kinds) == {"online", "final"} and all(t.frame == "cam0" for t in kinds.values())
    tr = read_tum(kinds["final"].path)
    assert len(tr.timestamps_ns) == len(ds.frames)
    assert np.allclose(tr.positions, sample(ds.gt, tr.timestamps_ns).positions, atol=1e-6)
    e = out.extra
    assert e["n_frames_tracking_ok"] == len(ds.frames) - 2
    assert e["state_counts"] == {"init": 1, "vo": len(ds.frames) - 3, "vo_failed_identity": 2}
    assert e["n_loop_closures"] == 1 and e["gpu"]["name"] == "Fake GPU"
    assert abs(e["frontend_time_s"]["median"] - 0.03) < 1e-9
    assert set(out.artifacts) == {"sparse_map_ply", "loop_closures", "frame_log"}
    with open(out.frame_diagnostics) as f:
        rows = list(csv.DictReader(f))
    assert rows[5]["tracking_state"] == "vo_failed_identity" and rows[5]["tracking_ok"] == "0"
    assert rows[7]["tracking_ok"] == "1" and rows[7]["in_final_trajectory"] == "1"


def test_frame_stride_input_view(tmp_path):
    a, ds, ctx = _ctx(tmp_path, {"frame_stride": 3})
    a.prepare(ctx)
    inp = ctx.algo_dir / "input"
    rows = (inp / "frames.csv").read_text().splitlines()
    src = (ctx.prepared.directory / "frames.csv").read_text().splitlines()
    assert rows[0] == src[0] and rows[1:] == src[1::3]
    for d in (p for p in ctx.prepared.directory.iterdir() if p.is_dir()):
        assert (inp / d.name).is_symlink() and (inp / d.name).resolve() == d.resolve()
    cmd = a.command(ctx)
    assert cmd[-2] == str(inp.resolve())
    # unstrided runs still read the prepared input directly
    (tmp_path / "s1run").mkdir()
    a1, _, ctx1 = _ctx(tmp_path / "s1run")
    a1.prepare(ctx1)
    assert a1.command(ctx1)[-2] == str(ctx1.prepared.directory.resolve()) and not (ctx1.algo_dir / "input").exists()


def test_collect_failures(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    _write_outputs(ctx, ds, failed=set(range(1, len(ds.frames))))
    assert a.collect(ctx, 0, False).status is RunStatus.INIT_FAILURE
    _write_outputs(ctx, ds, status="error")
    assert a.collect(ctx, 1, False).status is RunStatus.CRASH
    assert a.collect(ctx, 0, False).status is RunStatus.CRASH
    assert a.collect(ctx, None, True).status is RunStatus.TIMEOUT
    _write_outputs(ctx, ds)
    (ctx.algo_dir / "trajectory_final.csv").unlink()
    assert a.collect(ctx, 0, False).status is RunStatus.MISSING_OUTPUT
    (ctx.algo_dir / "frame_log.csv").unlink()
    assert a.collect(ctx, 0, False).status is RunStatus.MISSING_OUTPUT


def test_driver_pose_convention():
    """The driver's cam0 pose must invert LAC's solve_vision_pnp rover-pose construction (camera at
    the rover origin): rover R = R_cv @ OPENCV_TO_CAMERA_PASSIVE, PASSIVE = ACTIVE.T."""
    d = _driver()
    A = d.OPENCV_TO_CAMERA_ACTIVE
    # LAC frame: x forward, y left, z up; OpenCV: x right, y down, z forward
    assert np.allclose(A @ [0, 0, 1], [1, 0, 0]) and np.allclose(A @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(A @ [0, 1, 0], [0, 0, -1])
    rng = np.random.default_rng(0)
    for _ in range(5):
        T_w_cv = make_T(Rotation.random(random_state=rng).as_matrix(), rng.normal(size=3))
        T_w_rover = T_w_cv.copy()
        T_w_rover[:3, :3] = T_w_cv[:3, :3] @ A.T
        assert np.allclose(d.rover_to_opencv_pose(T_w_rover), T_w_cv)
        # a point seen by the camera maps to the same world point either way
        p_cv = rng.normal(size=3)
        p_rover = A @ p_cv
        assert np.allclose(T_w_rover[:3, :3] @ p_rover + T_w_rover[:3, 3],
                           T_w_cv[:3, :3] @ p_cv + T_w_cv[:3, 3])
    t, q = d.mat_to_tq(T_w_cv)
    assert np.allclose(make_T(Rotation.from_quat(q).as_matrix(), t), T_w_cv)


def test_driver_keyframe_due_and_ply(tmp_path):
    d = _driver()
    assert d.keyframe_due(5, None, 1.0)
    assert not d.keyframe_due(1_500_000_000, 1_000_000_000, 1.0)
    assert d.keyframe_due(2_000_000_000, 1_000_000_000, 1.0)
    d.write_ply(tmp_path / "m.ply", np.arange(6).reshape(2, 3), [3, 1])
    raw = (tmp_path / "m.ply").read_bytes()
    head, body = raw.split(b"end_header\n")
    assert b"element vertex 2" in head and len(body) == 2 * 16
    rec = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<i4")])
    assert rec["z"].tolist() == [2.0, 5.0] and rec["label"].tolist() == [3, 1]


def test_driver_stereo_filter():
    d = _driver()
    kl = np.array([[100.0, 50.0], [100.0, 50.0], [100.0, 50.0], [100.0, 50.0], [100.0, 50.0]])
    kr = np.array([[90.0, 50.0], [99.9, 50.0], [101.0, 50.0], [95.0, 53.0], [100.0, 50.0]])
    # LAC behaviour: only depth > max_depth (fb=30: 3 m, 300 m, -30 m, 6 m, ~3e9 m) is dropped
    keep, depth = d.stereo_filter(kl, kr, 30.0, 56.5)
    assert keep.tolist() == [True, False, True, True, False]
    assert np.allclose(depth[[0, 3]], [3.0, 6.0])
    keep, _ = d.stereo_filter(kl, kr, 30.0, 56.5, min_disparity_px=0.0, max_row_diff_px=1.5)
    assert keep.tolist() == [True, False, False, False, False]
    keep, _ = d.stereo_filter(kl, kr, 30.0, 5.0)
    assert keep.tolist() == [True, False, True, False, False]


def test_driver_reference_switch():
    d = _driver()
    assert not d.reference_switch_due(3.0, 80, 100, 10.0, 0.5)
    assert d.reference_switch_due(10.0, 80, 100, 10.0, 0.5)
    assert d.reference_switch_due(3.0, 49, 100, 10.0, 0.5)
    assert d.reference_switch_due(float("nan"), 0, 100, 10.0, 0.5)

    fp = d.FramePoints(np.zeros((4, 3)), np.array([3, 3, 3, 3]), np.array([0, 2]))
    assert fp.lengths.tolist() == [1, 0, 1, 0]


def test_driver_orb_features_and_matching():
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    d = _driver()
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (3, 3), 0)
    f1 = d.orb_features(img, 300, 7, 8, 1.2)
    n = f1["keypoints"].shape[1]
    assert 50 < n <= 300 and f1["descriptors"].shape == (1, n, 32) and f1["descriptors"].dtype == torch.uint8
    assert f1["keypoint_scores"].shape == (1, n) and f1["image_size"].tolist() == [[320, 240]]
    shifted = np.roll(img, 5, axis=1)                      # pure 5 px horizontal shift
    f2 = d.orb_features(shifted, 300, 7, 8, 1.2)
    m, sc = d.orb_match(f1, f2, 0.8, 100)
    assert m.dtype == torch.int64 and m.shape[1] == 2 and len(m) > 30 and len(sc) == len(m)
    flow = f2["keypoints"][0][m[:, 1]] - f1["keypoints"][0][m[:, 0]]
    assert float(torch.median(flow[:, 0])) == pytest.approx(5.0, abs=0.5)
    assert float(torch.median(flow[:, 1].abs())) < 0.5 and bool(((sc > 0) & (sc <= 1)).all())
    empty = d.orb_features(np.zeros((60, 80), np.uint8), 100, 7, 8, 1.2)
    assert empty["keypoints"].shape == (1, 0, 2) and len(d.orb_match(empty, f1, 0.8, 100)[0]) == 0
