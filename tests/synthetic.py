"""Synthetic trajectories, diagnostics and fake adapters for tests."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from slambench.algorithms.base import FRAME_DIAGNOSTICS_COLUMNS, AlgorithmAdapter, AlgorithmOutput
from slambench.datasets.base import DatasetAdapter, write_frames_csv
from slambench.trajectory import Trajectory, make_T, write_tum
from slambench.types import (CameraIntrinsics, GroundTruth, PreparedInput, RunStatus, SensorMode,
                             SequenceInfo, StereoCalibration, StreamInfo)

T0 = 1_700_000_000_000_000_000


def circle_gt(duration_s=60.0, rate_hz=30.0, radius=5.0):
    n = int(duration_s * rate_hz) + 1
    t = T0 + np.round(np.arange(n) / rate_hz * 1e9).astype(np.int64)
    a = np.linspace(0, 2 * np.pi, n)
    pos = np.c_[radius * np.cos(a), radius * np.sin(a), 0.3 * np.sin(3 * a)]
    yaw = a + np.pi / 2
    q = Rotation.from_euler("zyx", np.c_[yaw, 0.05 * np.sin(5 * a), np.zeros(n)]).as_quat()
    return Trajectory(t, pos, q, "body")


def frame_times(duration_s=60.0, rate_hz=15.0, offset_ns=0):
    n = int(duration_s * rate_hz)
    return T0 + offset_ns + np.round(np.arange(n) / rate_hz * 1e9).astype(np.int64)


def sample(gt: Trajectory, ts: np.ndarray) -> Trajectory:
    """Exact GT poses at the GT samples nearest to ts (ts chosen to coincide with GT)."""
    idx = np.searchsorted(gt.timestamps_ns, ts)
    idx = np.clip(idx, 0, len(gt) - 1)
    return Trajectory(ts, gt.positions[idx], gt.quaternions_xyzw[idx], gt.frame)


def random_world(seed=0):
    r = Rotation.from_rotvec(np.random.default_rng(seed).normal(size=3)).as_matrix()
    return make_T(r, [10.0, -3.0, 2.0])


def write_diag(path: Path, ts, ok, map_id=None, track_time=0.01):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_DIAGNOSTICS_COLUMNS)
        for i, t in enumerate(ts):
            w.writerow([int(t), "ok" if ok[i] else "lost", int(ok[i]),
                        0 if map_id is None else int(map_id[i]), track_time, int(ok[i])])


# ---------------------------------------------------------------------------
class FakeDataset(DatasetAdapter):
    """In-memory dataset: stereo pinhole + circle GT; no IMU calibration."""
    name = "fake"

    def __init__(self, root, sequence, options=None):
        super().__init__(root, sequence, options)
        self.gt = circle_gt(20.0)
        self.frames = frame_times(20.0)

    def sequence_info(self):
        s = {c: StreamInfo(c, "camera", c, len(self.frames), 15.0, int(self.frames[0]), int(self.frames[-1]))
             for c in ("cam0", "cam1")}
        s["imu"] = StreamInfo("imu", "imu", "imu", 100, 200.0, int(self.frames[0]), int(self.frames[-1]))
        return SequenceInfo(self.name, self.sequence, str(self.root), 20.0, s, True)

    def stereo_calibration(self):
        c = CameraIntrinsics("pinhole", 640, 360, 260.0, 260.0, 320.0, 180.0, [0, 0, 0, 0], True)
        return StereoCalibration(c, c, make_T(np.eye(3), [0.12, 0, 0]), 0.12, "fake")

    def imu_calibration(self):
        return None

    def ground_truth(self):
        g = self.gt
        return GroundTruth(g.timestamps_ns, g.positions, g.quaternions_xyzw, "world", "body", "fake")

    def T_gt_body_sensor(self, sensor_frame):
        return np.eye(4) if sensor_frame == "cam0" else None

    def prepare_offline(self, mode, out_dir):
        n = write_frames_csv(Path(out_dir) / "frames.csv", [(t, f"cam0/{t}.png", f"cam1/{t}.png")
                                                            for t in self.frames])
        return PreparedInput(Path(out_dir), mode, n, 0, {"n_frames": n, "preprocessing": []})


FAKE_ESTIMATOR = r'''
import sys, csv, numpy as np
frames_csv, out_dir, behaviour = sys.argv[1], sys.argv[2], sys.argv[3]
ts = np.loadtxt(frames_csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
if behaviour == "crash":
    sys.exit(-11 % 256)
with open(out_dir + "/poses.txt", "w") as f:
    for t in ts:
        f.write(f"{t}\n")
print("fake estimator done")
'''


class FakeAlgorithm(AlgorithmAdapter):
    """Runs a tiny python 'estimator'; collect() synthesises poses = GT + 0.05 m alternating z error."""
    name = "fake_algo"
    supported_modes = frozenset({SensorMode.STEREO})

    def estimate_frame(self, mode):
        return "cam0"

    def version_info(self):
        return {"fake": "1.0"}

    def prepare(self, ctx):
        ctx.algo_dir.mkdir(parents=True, exist_ok=True)
        (ctx.algo_dir / "est.py").write_text(FAKE_ESTIMATOR)

    def command(self, ctx):
        return [sys.executable, str(ctx.algo_dir / "est.py"), str(ctx.prepared.frames_csv),
                str(ctx.algo_dir), self.options.get("behaviour", "ok")]

    def collect(self, ctx, returncode, timed_out):
        if timed_out:
            return AlgorithmOutput(RunStatus.TIMEOUT, "timed out", estimate_frame="cam0")
        if returncode != 0:
            return AlgorithmOutput(RunStatus.CRASH, f"exit {returncode}", estimate_frame="cam0")
        ts = np.loadtxt(ctx.algo_dir / "poses.txt", dtype=np.int64)
        gt = ctx.dataset.gt
        est = sample(gt, ts)
        err = np.where(np.arange(len(ts)) % 2 == 0, 0.05, -0.05)
        est = Trajectory(est.timestamps_ns, est.positions + np.c_[0 * err, 0 * err, err],
                         est.quaternions_xyzw, "cam0")
        est = est.transformed(T_left=random_world(3))
        write_tum(est, ctx.run_dir / "trajectory.tum")
        write_diag(ctx.run_dir / "frame_diagnostics.csv", ts, np.ones(len(ts), bool))
        return AlgorithmOutput(RunStatus.SUCCESS, "", ctx.run_dir / "trajectory.tum", "cam0",
                               ctx.run_dir / "frame_diagnostics.csv",
                               {"tracking_state": True, "per_frame_time": True, "map_id": True})
