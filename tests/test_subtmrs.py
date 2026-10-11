"""SubT-MRS (SmokeRoom) adapter tests on a tiny synthetic sequence using the published calibration."""

import json
import sys

import cv2
import numpy as np
import pytest

from slambench import runner
from slambench.algorithms import ALGORITHMS
from slambench.algorithms.base import AlgorithmAdapter, AlgorithmOutput, RunContext
from slambench.algorithms.orbslam3 import OrbSlam3Adapter
from slambench.config import DEFAULTS, deep_merge, validate_config
from slambench.datasets import DATASETS
from slambench.datasets.subtmrs import SubtMrsAdapter, mei_project, pinhole_to_mei_maps
from slambench.trajectory import Trajectory, write_tum
from slambench.types import RunStatus, SensorMode

from synthetic import random_world, write_diag

SEQ = "SubT_MRS_Hawkins_Smoke_Handheld"

# Values from SubT_MRS_Hawkins_Smoke_Handheld_{Intrinsics,Extrinsics}.yaml (superodometry.com).
INTRINSICS = """%YAML 1.2
---
rgb_camera:
  model_type: MEI
  camera_name: camera
  image_width: 640
  image_height: 480
  mirror_parameters:
    xi: 1.583106303248484
  distortion_parameters:
    k1: -0.059098218173967695
    k2: 0.1793477408661115
    p1: 0.0016819528105368057
    p2: -0.0005887999624264534
  projection_parameters:
    gamma1: 758.3153257832925
    gamma2: 676.6492212772476
    u0: 318.27111164892506
    v0: 239.79816832491474
"""
EXTRINSICS = """%YAML 1.2
---
rgb_camera_to_imu:
  rows: 4
  cols: 4
  data: [-0.04279531, -0.00237969,  0.99908103,  0.19499356,
         -0.99880330, -0.02359508, -0.04283961,  0.04340662,
          0.02367534, -0.99971877, -0.00136708, -0.01782382,
          0.00000000,  0.00000000,  0.00000000,  1.00000000]
"""
MEI = dict(xi=1.583106303248484, k1=-0.059098218173967695, k2=0.1793477408661115,
           p1=0.0016819528105368057, p2=-0.0005887999624264534, gamma1=758.3153257832925,
           gamma2=676.6492212772476, u0=318.27111164892506, v0=239.79816832491474)
POINT = np.array([0.4, -0.25, 1.0])     # 3D point (camera frame) drawn into every synthetic frame
N_FRAMES, T0 = 5, 1676767774054630249


@pytest.fixture
def seq_root(tmp_path):
    seq = tmp_path / SEQ
    cam = seq / "SmokeRoom_Folder" / "cam_0"
    cam.mkdir(parents=True)
    (seq / "SmokeRoom_Folder" / "imu").mkdir()
    gt = seq / "SubT_MRS_Smoke_Room" / "integrated_odom"
    gt.mkdir(parents=True)
    (seq / f"{SEQ}_Intrinsics.yaml").write_text(INTRINSICS)
    (seq / f"{SEQ}_Extrinsics.yaml").write_text(EXTRINSICS)
    u, v = mei_project(POINT, **MEI)
    ts = [T0 + i * 40_000_000 for i in range(N_FRAMES)]  # on the 5 ms GT grid
    for i in range(N_FRAMES):
        img = np.zeros((480, 640, 3), np.uint8)
        cv2.circle(img, (int(round(u)), int(round(v))), 3, (255, 255, 255), -1)
        cv2.imwrite(str(cam / f"{i}.png"), img)
    (cam / "timestamps.txt").write_text("\n".join(map(str, ts)) + "\n")
    t_imu = T0 - 10_000_000 + np.arange(60) * 5_000_000
    with open(seq / "SmokeRoom_Folder" / "imu" / "imu_data.csv", "w") as f:
        f.write("timestamp, q_x, q_y, q_z, q_w, ang_vel_x, ang_vel_y, ang_vel_z, lin_acc_x, lin_acc_y, lin_acc_z\n")
        for k, t in enumerate(t_imu):
            f.write(f"{t},0,0,0,1,{0.01 * k},0.02,0.03,-0.17,-0.04,9.81\n")
    with open(gt / "odometry_data.csv", "w") as f:
        f.write("timestamp, x, y, z, q_x, q_y, q_z, q_w\n")
        for k, t in enumerate(t_imu):
            yaw = 0.01 * k
            f.write(f"{t}, {0.01 * k}, {0.002 * k * k}, {0.001 * k}, 0, 0, {np.sin(yaw / 2)}, {np.cos(yaw / 2)}\n")
    return tmp_path


def test_registered():
    assert DATASETS["subtmrs"] is SubtMrsAdapter


def test_mei_projection_matches_closed_form():
    assert np.allclose(mei_project(np.array([0, 0, 2.0]), **MEI), [MEI["u0"], MEI["v0"]])
    p = dict(MEI, k1=0, k2=0, p1=0, p2=0)
    P = np.array([0.3, -0.2, 0.8])
    z = P[2] + p["xi"] * np.linalg.norm(P)
    assert np.allclose(mei_project(P, **p), [p["gamma1"] * P[0] / z + p["u0"], p["gamma2"] * P[1] / z + p["v0"]])


def test_pinhole_maps_sample_projection_of_pinhole_rays():
    f, w, h = 250.0, 64, 48
    mx, my = pinhole_to_mei_maps(MEI, w, h, f)
    u, v = 10, 37
    ray = np.array([(u - (w - 1) / 2) / f, (v - (h - 1) / 2) / f, 1.0])
    assert np.allclose([mx[v, u], my[v, u]], mei_project(ray, **MEI), atol=1e-3)


def test_extrinsic_direction(seq_root):
    ds = SubtMrsAdapter(seq_root, SEQ)
    T = ds.T_gt_body_sensor("cam0")       # pose of camera in IMU (GT body) frame
    optical_axis_in_imu = T[:3, :3] @ [0, 0, 1]
    assert optical_axis_in_imu[0] > 0.99  # camera looks along IMU +x (forward)
    assert T[0, 3] > 0.19                 # camera mounted ahead of the IMU
    assert np.allclose(ds.imu_calibration().T_cam0_imu @ T, np.eye(4), atol=1e-6)
    assert np.allclose(ds.T_gt_body_sensor("imu"), np.eye(4))


def test_validation_per_mode(seq_root):
    ds = SubtMrsAdapter(seq_root, SEQ)
    assert ds.validate(SensorMode.MONO) == []
    assert any("cam1" in p for p in ds.validate(SensorMode.STEREO))
    assert any("lidar" in p for p in ds.validate(SensorMode.LIDAR))
    assert any("noise" in p for p in ds.validate(SensorMode.MONO_INERTIAL))
    noise = dict(gyro_noise_density=1e-4, accel_noise_density=1e-3, gyro_random_walk=1e-6,
                 accel_random_walk=1e-5, source="test")
    assert any("not verified" in p for p in
               SubtMrsAdapter(seq_root, SEQ, {"imu_noise": noise}).validate(SensorMode.MONO_INERTIAL))
    ok = SubtMrsAdapter(seq_root, SEQ, {"imu_noise": dict(noise, verified=True)})
    assert ok.validate(SensorMode.MONO_INERTIAL) == []
    calib = ds.stereo_calibration()
    assert calib.right is None and json.dumps(calib.to_dict(), default=str)


def test_unknown_sequence_needs_paths(tmp_path):
    with pytest.raises(Exception):
        SubtMrsAdapter(tmp_path, "nope").sequence_info()


def test_prepare_resamples_to_pinhole(seq_root, tmp_path):
    ds = SubtMrsAdapter(seq_root, SEQ)
    prep = ds.prepare_cached(SensorMode.MONO_INERTIAL, tmp_path / "cache")
    m = prep.manifest
    assert m["complete"] and prep.n_frames == N_FRAMES and prep.n_imu == 60
    assert m["conversions"] == ["mei_to_pinhole_remap"] and m["preprocessing"] == []
    assert m["pinhole_valid_pixel_fraction"] == pytest.approx(1.0)
    rows = (prep.directory / "frames.csv").read_text().splitlines()
    assert rows[1] == f"{T0},cam0/0.png,"
    imu = np.loadtxt(prep.imu_csv, delimiter=",", skiprows=1)
    assert np.allclose(imu[0, 1:], [0.0, 0.02, 0.03, -0.17, -0.04, 9.81])

    # The dot drawn at the MEI projection of POINT must appear at its pinhole projection.
    c = ds.stereo_calibration().left
    img = cv2.imread(str(prep.directory / "cam0" / "0.png"), cv2.IMREAD_GRAYSCALE)
    vs, us = np.nonzero(img > 128)
    expected = [c.fx * POINT[0] / POINT[2] + c.cx, c.fy * POINT[1] / POINT[2] + c.cy]
    assert np.allclose([us.mean(), vs.mean()], expected, atol=1.0)

    # Cache hit; different pinhole settings produce a different preparation.
    assert ds.prepare_cached(SensorMode.MONO_INERTIAL, tmp_path / "cache").directory == prep.directory
    other = SubtMrsAdapter(seq_root, SEQ, {"pinhole_focal": 200})
    assert other.preparation_key(SensorMode.MONO_INERTIAL) != ds.preparation_key(SensorMode.MONO_INERTIAL)


def test_ground_truth(seq_root):
    gt = SubtMrsAdapter(seq_root, SEQ).ground_truth()
    assert gt.body_frame == "imu" and gt.has_orientation and len(gt.timestamps_ns) == 60
    assert np.allclose(gt.positions[3], [0.03, 0.018, 0.003])


def test_orbslam3_mono_settings(seq_root, tmp_path):
    ds = SubtMrsAdapter(seq_root, SEQ)
    prep = ds.prepare_cached(SensorMode.MONO, tmp_path / "cache")
    algo = OrbSlam3Adapter({"container": str(tmp_path / "missing.sif")})
    ctx = RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.MONO, prep, algo.resolve_settings({}))
    text = algo._settings_yaml(ctx)
    assert 'Camera.type: "PinHole"' in text and "Camera1.k1: 0.0" in text
    assert f"Camera1.fx: {ds.pinhole_focal()!r}" in text and "Camera.fps: 25" in text
    problems = algo.validate(SensorMode.STEREO, ds, ctx.settings)
    assert any("single camera" in p or "not found" in p for p in problems)


class _MonoFromGt(AlgorithmAdapter):
    """Fake mono estimator: cam0 poses from the reference trajectory, in a scaled random world."""
    name = "mono_from_gt"
    supported_modes = frozenset({SensorMode.MONO})

    def estimate_frame(self, mode):
        return "cam0"

    def version_info(self):
        return {"fake": "1"}

    def prepare(self, ctx):
        ctx.algo_dir.mkdir(parents=True, exist_ok=True)

    def command(self, ctx):
        return [sys.executable, "-c", "pass"]

    def collect(self, ctx, returncode, timed_out):
        frames = np.loadtxt(ctx.prepared.frames_csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
        gt = ctx.dataset.ground_truth()
        idx = np.searchsorted(gt.timestamps_ns, frames)
        assert np.array_equal(gt.timestamps_ns[idx], frames)
        imu = Trajectory(frames, gt.positions[idx], gt.quaternions_xyzw[idx], "imu")
        cam = imu.transformed(T_left=random_world(7), T_right=ctx.dataset.T_gt_body_sensor("cam0"))
        cam = Trajectory(cam.timestamps_ns, 3.7 * cam.positions, cam.quaternions_xyzw, "cam0")
        write_tum(cam, ctx.run_dir / "trajectory.tum")
        write_diag(ctx.run_dir / "frame_diagnostics.csv", frames, np.ones(len(frames), bool))
        return AlgorithmOutput(RunStatus.SUCCESS, "", ctx.run_dir / "trajectory.tum", "cam0",
                               ctx.run_dir / "frame_diagnostics.csv",
                               {"tracking_state": True, "per_frame_time": True, "map_id": True})


def test_runner_end_to_end_mono(seq_root, tmp_path, monkeypatch):
    monkeypatch.setitem(ALGORITHMS, _MonoFromGt.name, _MonoFromGt)
    cfg = {"name": "smoke", "output_root": str(tmp_path / "runs"), "work_dir": str(tmp_path / "work"),
           "dataset": {"name": "subtmrs", "root": str(seq_root), "sequence": SEQ},
           "algorithm": {"name": "mono_from_gt", "mode": "mono"},
           "execution": {"timeout_s": 60, "memory_poll_s": 0.05},
           "evaluation": {"plots": False, "alignment": "sim3", "association_max_dt_s": 0.005,
                          "session_min_matched": 3,
                          "rpe": [{"delta": 1, "unit": "frames"}]}}
    d = runner.run_experiment(validate_config(deep_merge(DEFAULTS, cfg)), log=lambda *_: None)
    st = json.loads((d / "status.json").read_text())
    assert st["status"] == "success" and st["eval_status"] == "ok", st
    info = json.loads((d / "dataset_info.json").read_text())
    assert info["stereo_calibration"]["right"] is None
    m = json.loads((d / "eval" / "metrics.json").read_text())
    ate = m["accuracy"]["final"]["ate"]
    assert m["accuracy"]["final"]["association"]["n_matched"] == N_FRAMES
    assert m["accuracy"]["final"]["ground_truth"]["body_frame"] == "imu"
    assert ate["translation_m"]["rmse"] < 1e-6 and ate["rotation_deg"]["rmse"] < 1e-4


def test_stereo_rejected_by_runner(seq_root, tmp_path):
    cfg = {"name": "smoke_stereo", "output_root": str(tmp_path / "runs"), "work_dir": str(tmp_path / "work"),
           "dataset": {"name": "subtmrs", "root": str(seq_root), "sequence": SEQ},
           "algorithm": {"name": "orbslam3", "mode": "stereo",
                         "options": {"container": str(tmp_path / "missing.sif")}}}
    d = runner.run_experiment(validate_config(deep_merge(DEFAULTS, cfg)), log=lambda *_: None)
    st = json.loads((d / "status.json").read_text())
    assert st["status"] == "config_rejected" and "cam1" in st["message"]


def test_crop_bottom_keeps_pinhole_geometry(seq_root, tmp_path):
    with pytest.raises(ValueError):
        SubtMrsAdapter(seq_root, SEQ, {"crop_bottom_px": 480})
    full = SubtMrsAdapter(seq_root, SEQ).stereo_calibration().left
    ds = SubtMrsAdapter(seq_root, SEQ, {"crop_bottom_px": 140})
    c = ds.stereo_calibration().left
    assert (c.width, c.height) == (640, 340)
    assert (c.fx, c.fy, c.cx, c.cy) == (full.fx, full.fy, full.cx, full.cy)
    prep = ds.prepare_cached(SensorMode.MONO, tmp_path / "cache")
    assert prep.manifest["preprocessing"][0].startswith("crop_bottom_140px")
    img = cv2.imread(str(prep.directory / "cam0" / "0.png"), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (340, 640)
    vs, us = np.nonzero(img > 128)
    expected = [c.fx * POINT[0] / POINT[2] + c.cx, c.fy * POINT[1] / POINT[2] + c.cy]
    assert np.allclose([us.mean(), vs.mean()], expected, atol=1.0)


def test_changed_source_data_invalidates_prepared_cache(seq_root, tmp_path):
    import shutil
    cache = tmp_path / "cache"
    first = SubtMrsAdapter(seq_root, SEQ).prepare_cached(SensorMode.MONO_INERTIAL, cache)
    assert first.manifest["preparation_inputs"]["dataset_root"] == str(seq_root.resolve())
    assert SubtMrsAdapter(seq_root, SEQ).prepare_cached(SensorMode.MONO_INERTIAL, cache).directory == first.directory

    imu = seq_root / SEQ / "SmokeRoom_Folder" / "imu" / "imu_data.csv"
    imu.write_text(imu.read_text().replace(",9.81\n", ",9.80\n"))
    second = SubtMrsAdapter(seq_root, SEQ).prepare_cached(SensorMode.MONO_INERTIAL, cache)
    assert second.directory != first.directory
    assert np.loadtxt(second.imu_csv, delimiter=",", skiprows=1)[0, 6] == pytest.approx(9.80)

    img = seq_root / SEQ / "SmokeRoom_Folder" / "cam_0" / "2.png"
    cv2.imwrite(str(img), np.full((480, 640, 3), 7, np.uint8))
    third = SubtMrsAdapter(seq_root, SEQ).prepare_cached(SensorMode.MONO_INERTIAL, cache)
    assert third.directory not in (first.directory, second.directory)

    calib = seq_root / SEQ / f"{SEQ}_Intrinsics.yaml"
    calib.write_text(calib.read_text().replace("xi: 1.583106303248484", "xi: 1.5"))
    fourth = SubtMrsAdapter(seq_root, SEQ).prepare_cached(SensorMode.MONO_INERTIAL, cache)
    assert fourth.directory not in (first.directory, second.directory, third.directory)

    # An identical copy of the sequence under another root is prepared separately.
    other = tmp_path / "copy"
    shutil.copytree(seq_root / SEQ, other / SEQ)
    assert SubtMrsAdapter(other, SEQ).preparation_key(SensorMode.MONO_INERTIAL) != \
        SubtMrsAdapter(seq_root, SEQ).preparation_key(SensorMode.MONO_INERTIAL)


def test_nonzero_imu_time_offset_is_rejected(seq_root):
    noise = dict(gyro_noise_density=1e-4, accel_noise_density=1e-3, gyro_random_walk=1e-6,
                 accel_random_walk=1e-5, source="test", verified=True)
    ok = SubtMrsAdapter(seq_root, SEQ, {"imu_noise": noise})
    assert ok.validate(SensorMode.MONO_INERTIAL) == []
    shifted = SubtMrsAdapter(seq_root, SEQ, {"imu_noise": noise, "time_offset_s": 0.004})
    probs = shifted.validate(SensorMode.MONO_INERTIAL)
    assert len(probs) == 1 and "time offset 0.004" in probs[0] and "not apply" in probs[0]
    assert shifted.validate(SensorMode.MONO) == []
