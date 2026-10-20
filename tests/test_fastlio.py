"""FAST-LIO adapter tests: settings, native parameters, command and output collection (no image),
plus the LunaPolaris LiDAR scan format helpers."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from slambench.algorithms import ALGORITHMS
from slambench.algorithms.base import RunContext
from slambench.algorithms.fastlio import POSE_COLUMNS, FastLioAdapter, driver_source_sha1
from slambench.datasets.lunapolaris import (LIDAR_POINT_DTYPE, livox_points, read_lidar_scan,
                                            write_lidar_scan)
from slambench.trajectory import make_T, read_tum
from slambench.types import RunStatus, SensorMode

from synthetic import FakeDataset, sample

LOG_COLS = ["frame_idx", "timestamp_ns", "status", "pose_valid", "lidar_beg_time", "lidar_end_time", "n_imu",
            *POSE_COLUMNS, *("imu_" + c for c in POSE_COLUMNS), *("ext_" + c for c in POSE_COLUMNS),
            "vx", "vy", "vz", "bgx", "bgy", "bgz", "bax", "bay", "baz",
            "n_points_raw", "n_points_undistort", "n_points_down", "n_effective",
            "track_time_s", "preprocess_time_s", "read_time_s", "map_time_s", "feed_wall_s", "done_wall_s"]

T_IMU_LIDAR = make_T(Rotation.from_euler("y", -27, degrees=True).as_matrix(), [0.05, -0.03, -0.14])


class FakeLidarDataset(FakeDataset):
    supports_lidar = True

    def lidar_calibration(self):
        return {"imu_source": "fake", "T_imu_lidar": T_IMU_LIDAR.tolist(), "source": "fake", "time_offset_s": None}

    def T_gt_body_sensor(self, sensor_frame):
        return np.eye(4) if sensor_frame in ("cam0", "lidar") else None


def _vec(T):
    return list(T[:3, 3]) + list(Rotation.from_matrix(T[:3, :3]).as_quat())


def _ctx(tmp_path, settings=None):
    ds = FakeLidarDataset(tmp_path, "s1")
    (tmp_path / "prep").mkdir(exist_ok=True)
    prep = ds.prepare_offline(SensorMode.LIDAR_INERTIAL, tmp_path / "prep")
    a = FastLioAdapter({"container": str(tmp_path / "x.sif")})
    return a, ds, RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.LIDAR_INERTIAL, prep,
                             a.resolve_settings(settings or {}))


def _write_outputs(ctx, ds, n_init=3, status="completed"):
    d = ctx.algo_dir
    d.mkdir(parents=True, exist_ok=True)
    gt = sample(ds.gt, ds.frames).poses()
    with open(d / "frame_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLS)
        for i, t in enumerate(ds.frames):
            ok = i >= n_init
            st = "ok" if ok else ("first_scan" if i == 0 else "initializing")
            pose = _vec(gt[i]) * 3 + [0.0] * 9 if ok else [""] * 30
            w.writerow([i, t, st, int(ok), (t - 100_000_000) / 1e9, t / 1e9, 8, *pose,
                        45000, 15000, 3000, 2500 if ok else 0, 0.02, 0.003, 0.001, 0.0005,
                        (t - ds.frames[0]) / 1e9, (t - ds.frames[0]) / 1e9 + 0.03])
    (d / "fastlio_map.ply").write_bytes(b"ply\n")
    (d / "driver_status.json").write_text(json.dumps({"status": status, "message": "input exhausted"}))
    (d / "driver_timing.json").write_text(json.dumps({"wall_s": 20.0, "n_ok": len(ds.frames) - n_init}))


def test_registered_and_rejects_bad_modes_settings(tmp_path):
    assert ALGORITHMS["fastlio"] is FastLioAdapter
    a, ds, ctx = _ctx(tmp_path)
    s = ctx.settings
    assert "does not support sensor mode 'stereo'" in a.validate(SensorMode.STEREO, ds, s)[0]
    assert any("container not found" in p for p in a.validate(SensorMode.LIDAR_INERTIAL, ds, s))
    assert any("no LiDAR calibration" in p for p in
               a.validate(SensorMode.LIDAR_INERTIAL, FakeDataset(tmp_path, "s1"), s))
    for bad in ({"nope": 1}, {"scan_line": 0}, {"filter_size_surf": 0}, {"map_voxel": -1},
                {"point_filter_num": 1.5}, {"max_iteration": True}):
        with pytest.raises(ValueError):
            a.resolve_settings(bad)
    assert a.estimate_frame(SensorMode.LIDAR_INERTIAL) == "lidar"
    assert driver_source_sha1() and len(driver_source_sha1()) == 40


def test_parameters_and_command(tmp_path):
    a, ds, ctx = _ctx(tmp_path, {"playback_rate": 0, "blind": 0.7})
    a.prepare(ctx)
    p = yaml.safe_load((ctx.algo_dir / "params.yaml").read_text())
    assert p["preprocess"] == {"lidar_type": 1, "scan_line": 6, "blind": 0.7, "timestamp_unit": 3, "scan_rate": 10}
    R = np.array(p["mapping"]["extrinsic_R"]).reshape(3, 3)          # row major
    assert np.allclose(R, T_IMU_LIDAR[:3, :3]) and np.allclose(p["mapping"]["extrinsic_T"], T_IMU_LIDAR[:3, 3])
    assert p["mapping"]["extrinsic_est_en"] is False
    assert p["common"]["time_offset_lidar_to_imu"] == 0.0
    assert p["slambench"]["input_dir"] == str(ctx.prepared.directory.resolve())
    assert p["slambench"]["playback_rate"] == 0.0 and p["slambench"]["map_voxel"] == 0.02
    assert not any(p["publish"].values()) and p["pcd_save"]["pcd_save_en"] is False
    cmd = a.command(ctx)
    assert cmd[:3] == [a.apptainer, "exec", "--cleanenv"] and cmd[-3:-1] == ["/bin/bash", "-c"]
    script = cmd[-1]
    assert "rosmaster --core" in script and "rosparam load" in script
    assert "fastlio_mapping __name:=laserMapping; rc=$?" in script and script.endswith("exit $rc")


def test_collect_success(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    _write_outputs(ctx, ds)
    out = a.collect(ctx, 0, False)
    assert out.status is RunStatus.SUCCESS, out.message
    kinds = {t.kind: t for t in out.all_trajectories()}
    assert set(kinds) == {"online", "final"} and all(t.frame == "lidar" for t in kinds.values())
    tr = read_tum(kinds["final"].path)
    assert len(tr.timestamps_ns) == len(ds.frames) - 3
    gt = sample(ds.gt, tr.timestamps_ns)
    assert np.allclose(tr.positions, gt.positions, atol=1e-6)
    e = out.extra
    assert e["n_frames_tracking_ok"] == len(ds.frames) - 3
    assert e["status_counts"] == {"first_scan": 1, "initializing": 2, "ok": len(ds.frames) - 3}
    assert e["max_abs_lidar_end_minus_frame_ts_s"] < 1e-6
    assert abs(e["latency_s"]["median"] - 0.03) < 1e-9
    assert set(out.artifacts) == {"lidar_map", "frame_log"}
    with open(out.frame_diagnostics) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["tracking_state"] == "first_scan" and rows[0]["tracking_ok"] == "0"
    assert rows[5]["tracking_ok"] == "1" and rows[5]["in_final_trajectory"] == "1"


def test_collect_failures(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    _write_outputs(ctx, ds, n_init=len(ds.frames))
    out = a.collect(ctx, 0, False)
    assert out.status is RunStatus.INIT_FAILURE and not out.all_trajectories()
    _write_outputs(ctx, ds, status="error")
    assert a.collect(ctx, 3, False).status is RunStatus.CRASH
    assert a.collect(ctx, 0, False).status is RunStatus.CRASH
    assert a.collect(ctx, None, True).status is RunStatus.TIMEOUT
    (ctx.algo_dir / "frame_log.csv").unlink()
    (ctx.algo_dir / "driver_status.json").write_text(json.dumps({"status": "completed"}))
    assert a.collect(ctx, 0, False).status is RunStatus.MISSING_OUTPUT


def test_lidar_scan_round_trip(tmp_path):
    pts = np.zeros(5, LIDAR_POINT_DTYPE)
    pts["x"] = np.arange(5)
    pts["offset_ns"] = [0, 10, 5, 20, 99_000_000]
    pts["line"] = [0, 1, 2, 3, 5]
    write_lidar_scan(tmp_path / "s.bin", 1_700_000_000_123_456_789, pts)
    assert (tmp_path / "s.bin").stat().st_size == 24 + 5 * 20
    start, back = read_lidar_scan(tmp_path / "s.bin")
    assert start == 1_700_000_000_123_456_789 and np.array_equal(back, pts)


def test_livox_points_decodes_pointcloud2():
    names = ["x", "y", "z", "intensity", "tag", "line", "timestamp"]
    dt = np.dtype({"names": names, "formats": ["<f4", "<f4", "<f4", "<f4", "u1", "u1", "<f8"],
                   "offsets": [0, 4, 8, 12, 16, 17, 18], "itemsize": 26})
    a = np.zeros(3, dt)
    a["x"], a["intensity"], a["tag"], a["line"] = [1, 2, 3], [0, 17, 255], [0, 16, 0], [0, 3, 5]
    a["timestamp"] = [2.0**60, 2.0**60 + 2**12, 2.0**60 + 2**26]   # f64: 256 ns steps at ~1e18
    msg = SimpleNamespace(fields=[SimpleNamespace(name=n, offset=dt.fields[n][1]) for n in names],
                          point_step=26, width=3, height=1, data=a.tobytes())
    t, pts = livox_points(msg)
    assert t.tolist() == [2**60, 2**60 + 2**12, 2**60 + 2**26]
    assert pts["reflectivity"].tolist() == [0, 17, 255] and pts["line"].tolist() == [0, 3, 5]
    a["intensity"][1] = 3.5
    msg.data = a.tobytes()
    with pytest.raises(ValueError):
        livox_points(msg)
