"""Readers for optional estimator artifacts: per-frame tracked features and the sparse map.

Formats (written by the ORB-SLAM3 driver; other adapters may produce the same files):

features.bin
    8-byte magic ``SBFEAT01``, then one record per frame (little endian):
    int64 timestamp_ns, int32 frame_idx, int32 n, then n x {float32 x, float32 y, uint8 status}.
    x, y are pixel coordinates in the estimator's input image; status 0 = keypoint without map
    point, 1 = matched to a map point (inlier), 2 = matched but rejected as outlier.
    A record cut short (e.g. by a crash) ends the file.

map_points.csv
    ``map_id,point_id,x,y,z,n_observations,in_final_frame``; rows with in_final_frame=1 are in the
    world frame of the run's final trajectory, other rows in their own map's frame.

keyframes.csv
    ``map_id,keyframe_id,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,in_final_frame`` (camera pose in world).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

FEATURES_MAGIC = b"SBFEAT01"
FEATURE_STATUS = {0: "unmatched", 1: "inlier", 2: "outlier"}
_REC = np.dtype([("x", "<f4"), ("y", "<f4"), ("status", "u1")])


def read_tracked_features(path: Path) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], bool]:
    """{timestamp_ns: (xy float32 (n,2), status uint8 (n,))}, truncated flag."""
    data = Path(path).read_bytes()
    if data[:8] != FEATURES_MAGIC:
        raise ValueError(f"{path}: not a slambench features file")
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    pos, end = 8, len(data)
    while pos < end:
        if pos + 16 > end:
            return out, True
        t = int(np.frombuffer(data, "<i8", 1, pos)[0])
        n = int(np.frombuffer(data, "<i4", 1, pos + 12)[0])
        pos += 16
        if n < 0 or pos + n * _REC.itemsize > end:
            return out, True
        rec = np.frombuffer(data, _REC, n, pos)
        pos += n * _REC.itemsize
        out[t] = (np.stack([rec["x"], rec["y"]], axis=1), rec["status"].copy())
    return out, False


def read_sparse_map(points_csv: Path, keyframes_csv: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Map points (and keyframe positions) as arrays; empty arrays if a file is missing/empty."""
    res: Dict[str, np.ndarray] = {}
    p = np.loadtxt(points_csv, delimiter=",", skiprows=1, ndmin=2) if Path(points_csv).exists() else np.zeros((0, 7))
    if p.size == 0:
        p = np.zeros((0, 7))
    res.update({"map_id": p[:, 0].astype(int), "xyz": p[:, 2:5], "n_obs": p[:, 5].astype(int),
                "in_final": p[:, 6].astype(bool)})
    if keyframes_csv is not None:
        k = (np.loadtxt(keyframes_csv, delimiter=",", skiprows=1, ndmin=2)
             if Path(keyframes_csv).exists() else np.zeros((0, 11)))
        if k.size == 0:
            k = np.zeros((0, 11))
        res.update({"kf_map_id": k[:, 0].astype(int), "kf_t_ns": k[:, 2].astype(np.int64),
                    "kf_xyz": k[:, 3:6], "kf_in_final": k[:, 10].astype(bool)})
    return res
