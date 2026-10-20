"""Geometric map quality against a reference point cloud (e.g. LiDAR scans placed with ground-truth poses).

Building blocks used by ``scripts/eval_map_quality.py``:

* :class:`PoseInterpolator` – ground-truth poses at arbitrary times (linear position, slerp rotation),
  with a body->sensor extrinsic applied and a check for gaps in the ground truth.
* :class:`VoxelAccumulator` – streaming voxel-grid merge of many scans (centroid and hit count per voxel).
* :func:`umeyama_se3` – rigid alignment of an estimated map frame to the ground-truth frame from
  trajectory positions (the same kind of alignment ATE uses), so map errors include pose errors.
* :func:`visible_mask` – which reference points a camera could have seen (frustum, depth limit and a
  coarse z-buffer against occlusion).
* :func:`score` – accuracy (estimated -> reference), completeness (visible reference -> estimated)
  and F-score at distance thresholds, the usual reconstruction-benchmark metrics.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


class PoseInterpolator:
    """T_world_sensor(t) from body poses (t_ns, positions, quaternions xyzw) and a fixed T_body_sensor."""

    def __init__(self, t_ns: np.ndarray, positions: np.ndarray, quats_xyzw: np.ndarray,
                 T_body_sensor: Optional[np.ndarray] = None, max_gap_s: float = 0.1):
        t_ns = np.asarray(t_ns, dtype=np.int64)
        if np.any(np.diff(t_ns) <= 0):
            raise ValueError("pose timestamps must be strictly increasing")
        self.t0 = int(t_ns[0])
        self.s = (t_ns - self.t0) * 1e-9
        self.p = np.asarray(positions, dtype=float)
        self.slerp = Slerp(self.s, Rotation.from_quat(quats_xyzw))
        self.T_bs = np.eye(4) if T_body_sensor is None else np.asarray(T_body_sensor, dtype=float)
        self.max_gap_s = max_gap_s

    def _s(self, t_ns) -> np.ndarray:
        return (np.asarray(t_ns, dtype=np.int64) - self.t0) * 1e-9

    def valid(self, t_ns) -> np.ndarray:
        """True where t lies inside the ground truth and between samples at most max_gap_s apart."""
        s = self._s(t_ns)
        i = np.searchsorted(self.s, s)
        ok = (s >= self.s[0]) & (s <= self.s[-1])
        i = np.clip(i, 1, len(self.s) - 1)
        return ok & ((self.s[i] - self.s[i - 1]) <= self.max_gap_s)

    def poses(self, t_ns) -> np.ndarray:
        """(N, 4, 4) T_world_sensor at the given times (must be valid())."""
        s = np.clip(self._s(t_ns), self.s[0], self.s[-1])
        T = np.tile(np.eye(4), (len(s), 1, 1))
        T[:, :3, :3] = self.slerp(s).as_matrix()
        T[:, :3, 3] = np.stack([np.interp(s, self.s, self.p[:, k]) for k in range(3)], axis=1)
        return T @ self.T_bs


class VoxelAccumulator:
    """Merges point batches into a voxel grid, keeping each voxel's centroid and number of hits."""

    _OFF = 1 << 20  # 21 bits per axis: +-2^20 voxels

    def __init__(self, voxel: float, flush_points: int = 5_000_000):
        self.voxel = float(voxel)
        self.flush_points = flush_points
        self.keys = np.empty(0, np.int64)
        self.sums = np.empty((0, 3))
        self.counts = np.empty(0, np.int64)
        self._buf: list = []
        self._nbuf = 0

    def _key(self, p: np.ndarray) -> np.ndarray:
        q = np.floor(p / self.voxel).astype(np.int64) + self._OFF
        if q.min(initial=0) < 0 or q.max(initial=0) >= (1 << 21):
            raise ValueError("points outside the voxel key range")
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    def add(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points):
            self._buf.append(points)
            self._nbuf += len(points)
            if self._nbuf >= self.flush_points:
                self._flush()

    def _merge(self, keys, sums, counts) -> None:
        keys = np.concatenate([self.keys, keys])
        sums = np.concatenate([self.sums, sums])
        counts = np.concatenate([self.counts, counts])
        u, inv = np.unique(keys, return_inverse=True)
        self.sums = np.stack([np.bincount(inv, weights=sums[:, j], minlength=len(u)) for j in range(3)], axis=1)
        self.counts = np.bincount(inv, weights=counts, minlength=len(u)).astype(np.int64)
        self.keys = u

    def _flush(self) -> None:
        if not self._buf:
            return
        p = np.concatenate(self._buf)
        self._buf, self._nbuf = [], 0
        self._merge(self._key(p), p, np.ones(len(p), np.int64))

    def merge(self, other: "VoxelAccumulator") -> None:
        if other.voxel != self.voxel:
            raise ValueError("voxel sizes differ")
        self._flush()
        other._flush()
        self._merge(other.keys, other.sums, other.counts)

    def result(self, min_count: int = 1):
        """(centroids (N, 3), counts (N,)) of voxels hit at least min_count times."""
        self._flush()
        keep = self.counts >= min_count
        return self.sums[keep] / self.counts[keep, None], self.counts[keep]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    acc = VoxelAccumulator(voxel)
    acc.add(points)
    return acc.result()[0]


def umeyama_se3(src: np.ndarray, dst: np.ndarray):
    """R, t minimizing sum |dst - (R src + t)|^2 (no scale)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    return R, md - R @ ms


def visible_mask(ref: np.ndarray, T_world_cam: Iterable[np.ndarray], fx: float, fy: float, cx: float,
                 cy: float, width: int, height: int, max_depth: float, min_depth: float = 0.2,
                 zbuf_decimation: int = 4, occlusion_tol: float = 0.15) -> np.ndarray:
    """Reference points inside the camera frustum (depth in [min_depth, max_depth]) of at least one pose
    and not behind a nearer reference surface (z-buffer on a grid of zbuf_decimation-pixel cells)."""
    ref = np.asarray(ref, float)
    tree = cKDTree(ref)
    vis = np.zeros(len(ref), bool)
    W, H = -(-width // zbuf_decimation), -(-height // zbuf_decimation)
    for T in T_world_cam:
        idx = np.asarray(tree.query_ball_point(T[:3, 3], max_depth), dtype=np.int64)
        if not len(idx):
            continue
        pc = (ref[idx] - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * pc[:, 0] / z + cx
            v = fy * pc[:, 1] / z + cy
        ok = (z >= min_depth) & (z <= max_depth) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not ok.any():
            continue
        idx, z = idx[ok], z[ok]
        pix = (v[ok] // zbuf_decimation).astype(np.int64) * W + (u[ok] // zbuf_decimation).astype(np.int64)
        zbuf = np.full(W * H, np.inf)
        np.minimum.at(zbuf, pix, z)
        vis[idx[z <= zbuf[pix] + occlusion_tol]] = True
    return vis


def region_mask(points: np.ndarray, ref: np.ndarray, cell: float = 0.25, dilate: int = 1) -> np.ndarray:
    """Points inside the space the reference covers: cells of size `cell` containing reference points,
    dilated by `dilate` cells. Estimated points outside it cannot be checked against the reference."""
    def keys(q):
        q = q + (1 << 20)
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    qr = np.unique(np.floor(np.asarray(ref) / cell).astype(np.int64), axis=0)
    offs = np.array([(i, j, k) for i in range(-dilate, dilate + 1) for j in range(-dilate, dilate + 1)
                     for k in range(-dilate, dilate + 1)], dtype=np.int64)
    occupied = np.unique(keys((qr[:, None, :] + offs[None]).reshape(-1, 3)))
    return np.isin(keys(np.floor(np.asarray(points) / cell).astype(np.int64)), occupied)


def score(est: np.ndarray, ref: np.ndarray, ref_visible: np.ndarray,
          taus: Sequence[float] = (0.05, 0.10), ref_tree: Optional[cKDTree] = None) -> Dict[str, float]:
    """Accuracy / completeness / F-score.

    precision@tau: fraction of estimated points within tau of the reference (all reference points);
    recall@tau: fraction of *visible* reference points within tau of the estimate; F = harmonic mean.
    """
    est = np.asarray(est, float)
    ref_tree = ref_tree or cKDTree(ref)
    d_acc = ref_tree.query(est)[0] if len(est) else np.empty(0)
    vis_pts = np.asarray(ref)[ref_visible]
    d_comp = cKDTree(est).query(vis_pts)[0] if len(est) else np.full(len(vis_pts), np.inf)
    out = {"n_points": int(len(est)), "n_ref_visible": int(len(vis_pts)),
           "accuracy_median_m": float(np.median(d_acc)) if len(d_acc) else float("nan"),
           "accuracy_p90_m": float(np.percentile(d_acc, 90)) if len(d_acc) else float("nan"),
           "completeness_median_m": float(np.median(d_comp)) if len(d_comp) else float("nan")}
    for tau in taus:
        tag = f"{round(tau * 100)}cm"
        P = float(np.mean(d_acc <= tau)) if len(d_acc) else 0.0
        R = float(np.mean(d_comp <= tau)) if len(d_comp) else 0.0
        out[f"precision_{tag}"] = P
        out[f"recall_{tag}"] = R
        out[f"fscore_{tag}"] = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return out
