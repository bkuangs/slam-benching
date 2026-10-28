"""Benchmark sessions: one image sequence with per-frame GT poses and rectified stereo calibration.

A session directory (written by scripts/vpr_prepare_sessions.py) holds
    session.json  {name, kind: sim|real, image_root, fx, fy, cx, cy, baseline_m, width, height, ...}
    poses.npz     T (N,4,4) GT cam0 (left, OpenCV optical) -> world, valid (N,) bool, t_ns (N,) int64
rows aligned with image_root/frames.csv (timestamp_ns,left_path,right_path).
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Session:
    name: str
    kind: str
    image_root: str
    left: List[str]
    right: List[str]
    T: np.ndarray
    valid: np.ndarray
    t_ns: np.ndarray
    K: np.ndarray
    baseline_m: float
    depth_dir: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.left)

    def left_path(self, i: int) -> str:
        return os.path.join(self.image_root, self.left[i])

    def right_path(self, i: int) -> str:
        return os.path.join(self.image_root, self.right[i])

    def depth_path(self, i: int) -> Optional[str]:
        if self.depth_dir is None:
            return None
        return os.path.join(self.depth_dir, os.path.basename(self.left[i]))


def read_frames_csv(image_root: str):
    t, left, right = [], [], []
    with open(os.path.join(image_root, "frames.csv")) as f:
        next(f)
        for line in f:
            a, b, c = line.strip().split(",")
            t.append(int(a))
            left.append(b)
            right.append(c)
    return np.array(t, np.int64), left, right


def load_session(session_dir: str) -> Session:
    meta = json.load(open(os.path.join(session_dir, "session.json")))
    z = np.load(os.path.join(session_dir, "poses.npz"))
    t, left, right = read_frames_csv(meta["image_root"])
    if len(left) != len(z["T"]):
        raise ValueError(f"{session_dir}: {len(left)} frames but {len(z['T'])} poses")
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1.0]])
    return Session(meta["name"], meta["kind"], meta["image_root"], left, right, z["T"], z["valid"].astype(bool),
                   t, K, float(meta["baseline_m"]), meta.get("depth_dir"), meta)


def write_session(session_dir: str, meta: dict, T: np.ndarray, valid: np.ndarray, t_ns: np.ndarray) -> None:
    os.makedirs(session_dir, exist_ok=True)
    np.savez(os.path.join(session_dir, "poses.npz"), T=T, valid=valid, t_ns=t_ns)
    with open(os.path.join(session_dir, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)
