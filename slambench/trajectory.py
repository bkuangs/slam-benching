"""Normalized trajectory format and SE(3) helpers.

Normalized trajectory file (``trajectory.tum``):
    TUM RGB-D format, one pose per line, space separated:
        timestamp tx ty tz qx qy qz qw
    * timestamp: seconds with 9 decimals, written from the integer-nanosecond
      original sensor timestamps (header stamps) - not re-based.
    * pose: T_world_sensor, the pose of the *estimate frame* (recorded in the
      ``trajectory.json`` sidecar, e.g. "cam0" = left camera optical frame) in
      the estimator's own world frame. Units: metres.
    * Only poses the estimator actually produced are written: no
      interpolation and no filling of tracking gaps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def T_from_quat(t, q_xyzw) -> np.ndarray:
    return make_T(Rotation.from_quat(q_xyzw).as_matrix(), t)


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


@dataclass
class Trajectory:
    timestamps_ns: np.ndarray               # int64
    positions: np.ndarray                   # (N,3)
    quaternions_xyzw: Optional[np.ndarray]  # (N,4) or None (position only)
    frame: str = ""                         # body frame the poses refer to
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.timestamps_ns = np.asarray(self.timestamps_ns, dtype=np.int64)
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        if self.quaternions_xyzw is not None:
            self.quaternions_xyzw = np.asarray(self.quaternions_xyzw, dtype=float).reshape(-1, 4)
        if len(self.timestamps_ns) != len(self.positions):
            raise ValueError("timestamps and positions differ in length")

    def __len__(self) -> int:
        return len(self.timestamps_ns)

    @property
    def has_orientation(self) -> bool:
        return self.quaternions_xyzw is not None

    @property
    def timestamps_s(self) -> np.ndarray:
        return self.timestamps_ns.astype(np.float64) * 1e-9

    def poses(self) -> np.ndarray:
        if not self.has_orientation:
            raise ValueError("trajectory has no orientation")
        T = np.tile(np.eye(4), (len(self), 1, 1))
        if len(self):
            T[:, :3, :3] = Rotation.from_quat(self.quaternions_xyzw).as_matrix()
            T[:, :3, 3] = self.positions
        return T

    def subset(self, idx) -> "Trajectory":
        return Trajectory(self.timestamps_ns[idx], self.positions[idx],
                          None if self.quaternions_xyzw is None else self.quaternions_xyzw[idx],
                          self.frame, dict(self.meta))

    def sorted_unique(self) -> "Trajectory":
        order = np.argsort(self.timestamps_ns, kind="stable")
        ts = self.timestamps_ns[order]
        keep = np.ones(len(ts), dtype=bool)
        keep[1:] = np.diff(ts) > 0
        return self.subset(order[keep])

    def transformed(self, T_left: Optional[np.ndarray] = None,
                    T_right: Optional[np.ndarray] = None) -> "Trajectory":
        """Return poses T_left @ T_i @ T_right (T_right requires orientation)."""
        if T_right is not None and not self.has_orientation:
            raise ValueError("cannot change the body frame of a position-only trajectory")
        if self.has_orientation:
            P = self.poses()
            if T_left is not None:
                P = T_left[None] @ P
            if T_right is not None:
                P = P @ T_right[None]
            q = Rotation.from_matrix(P[:, :3, :3]).as_quat() if len(P) else np.zeros((0, 4))
            return Trajectory(self.timestamps_ns, P[:, :3, 3], q, self.frame, dict(self.meta))
        pos = self.positions
        if T_left is not None:
            pos = pos @ T_left[:3, :3].T + T_left[:3, 3]
        return Trajectory(self.timestamps_ns, pos, None, self.frame, dict(self.meta))


def format_ns(t_ns: int) -> str:
    t_ns = int(t_ns)
    return f"{t_ns // 1_000_000_000}.{t_ns % 1_000_000_000:09d}"


def write_tum(traj: Trajectory, path: Path, sidecar: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = traj.quaternions_xyzw
    with open(path, "w") as f:
        for i in range(len(traj)):
            p = traj.positions[i]
            qq = q[i] if q is not None else (np.nan,) * 4
            f.write(f"{format_ns(traj.timestamps_ns[i])} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                    f"{qq[0]:.9f} {qq[1]:.9f} {qq[2]:.9f} {qq[3]:.9f}\n")
    meta = {"format": "tum", "timestamp_unit": "s", "position_unit": "m",
            "quaternion_order": "xyzw", "pose_convention": "T_world_sensor",
            "frame": traj.frame, "n_poses": len(traj)}
    meta.update(traj.meta)
    if sidecar:
        meta.update(sidecar)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def parse_ts_ns(token: str) -> int:
    """Parse a decimal seconds string into integer nanoseconds without float rounding."""
    token = token.strip()
    sign = -1 if token.startswith("-") else 1
    token = token.lstrip("+-")
    if "." in token:
        whole, frac = token.split(".", 1)
        frac = (frac + "000000000")[:9]
        return sign * (int(whole or 0) * 1_000_000_000 + int(frac))
    return sign * int(token) * 1_000_000_000


def read_tum(path: Path) -> Trajectory:
    ts, pos, quat = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            c = line.split()
            ts.append(parse_ts_ns(c[0]))
            pos.append([float(x) for x in c[1:4]])
            quat.append([float(x) for x in c[4:8]])
    quat = np.array(quat, dtype=float).reshape(-1, 4)
    q = None if (len(quat) and np.isnan(quat).all()) else quat
    meta, frame = {}, ""
    side = Path(path).with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text())
        frame = meta.get("frame", "")
    return Trajectory(np.array(ts, dtype=np.int64), np.array(pos).reshape(-1, 3), q, frame, meta)
