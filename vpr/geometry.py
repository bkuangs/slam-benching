"""Geometric (lighting-independent) verification of relocalization candidates.

A candidate pose from PnP places the query's own 3D points (depth or LiDAR, query camera frame) in the map's
world frame. Robust point-to-plane ICP against the map's 3D points around the candidate then tests three
things: do the shapes agree (fitness, residual), does ICP leave the PnP pose nearly where it was (correction),
and does the geometry constrain all six degrees of freedom (conditioning). Flat ground "agrees" with any pose
shifted along the ground, so the conditioning test is what separates agreement from absence of evidence.

Pure numpy / scipy; unit-tested on synthetic data in tests/test_geometry.py.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree


def transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def voxel_downsample(P: np.ndarray, voxel: float) -> np.ndarray:
    """One point (the centroid) per occupied voxel."""
    if len(P) == 0 or voxel <= 0:
        return P
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, P)
    return out / cnt[:, None]


def depth_to_points(depth_m: np.ndarray, K: np.ndarray, stride: int = 4, z_min: float = 0.1,
                    z_max: float = 60.0) -> np.ndarray:
    """Camera-frame (OpenCV optical) points from a z-depth image in metres, on a pixel grid of `stride`."""
    d = depth_m[::stride, ::stride]
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    ok = (d > z_min) & (d < z_max) & np.isfinite(d)
    z = d[ok].astype(np.float64)
    x = (u[ok] - K[0, 2]) / K[0, 0] * z
    y = (v[ok] - K[1, 2]) / K[1, 1] * z
    return np.c_[x, y, z]


def estimate_normals(P: np.ndarray, k: int = 12):
    """PCA normals (unit, arbitrary sign) and planarity (l1 - l0) / l2 in [0, 1] from k nearest neighbours."""
    k = min(k, len(P))
    _, idx = cKDTree(P).query(P, k)
    nb = P[idx] - P[idx].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb) / k)
    return v[:, :, 0], (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)


class MapCloud:
    """World-frame points with normals and a KD-tree (points of low planarity are dropped)."""

    def __init__(self, P: np.ndarray, voxel: float = 0.0, k: int = 20, min_planarity: float = 0.3):
        P = voxel_downsample(np.asarray(P, float), voxel)
        if len(P) >= 3:
            n, planar = estimate_normals(P, k)
            keep = planar > min_planarity
            P, n = P[keep], n[keep]
        else:
            n = np.zeros_like(P)
        self.P, self.n = P, n
        self.tree = cKDTree(P) if len(P) else None

    def __len__(self):
        return len(self.P)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """4x4 transform for a small twist xi = (omega, v) (rotation by Rodrigues, translation v)."""
    w, v = xi[:3], xi[3:]
    th = np.linalg.norm(w)
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        R = np.eye(3) + W
    else:
        R = np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th**2 * W @ W
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, v
    return T


def pose_delta(Ta: np.ndarray, Tb: np.ndarray):
    """(translation m, rotation deg) between two poses."""
    dR = Ta[:3, :3].T @ Tb[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), float(ang)


def conditioning(P: np.ndarray, n: np.ndarray, w: Optional[np.ndarray] = None):
    """Conditioning of the point-to-plane information matrix (as in scripts/lidar_degeneracy_probe.py).

    Points are centred on their (weighted) centroid and scaled to unit RMS radius, so the result is unitless:
    H = sum w J^T J / sum w with J = [p x n, n]. Returns (cond_t, cond_r, cond6, weakest 6-vector):
    cond_t = lambda_min of the translation block (sum n n^T / N, at most 1/3; 0 on a plane),
    cond_r = lambda_min of the rotation block, cond6 = lambda_min of the full 6x6 matrix."""
    if len(P) < 6:
        return 0.0, 0.0, 0.0, np.zeros(6)
    w = np.ones(len(P)) if w is None else w
    c = (w[:, None] * P).sum(0) / w.sum()
    Pc = P - c
    s = np.sqrt((w * (Pc**2).sum(1)).sum() / w.sum())
    Pc = Pc / max(s, 1e-9)
    J = np.hstack([np.cross(Pc, n), n])
    H = (J * w[:, None]).T @ J / w.sum()
    ev, V = np.linalg.eigh(H)
    return (float(np.linalg.eigvalsh(H[3:, 3:])[0]), float(np.linalg.eigvalsh(H[:3, :3])[0]), float(ev[0]),
            V[:, 0])


@dataclass
class IcpResult:
    T: np.ndarray            # refined query-camera -> world
    fitness: float           # share of query points that are inliers after ICP (see icp_point_to_plane)
    rmse: float              # point-to-plane RMS residual of the inliers (m)
    n_inliers: int
    n_points: int
    corr_t: float            # |ICP correction| from the initial pose (m)
    corr_r: float            # (deg)
    cond_t: float
    cond_r: float
    cond6: float
    iterations: int
    converged: bool


def icp_point_to_plane(Q: np.ndarray, M: MapCloud, T0: np.ndarray, max_corr: Sequence[float],
                       inlier_dist: float, nn_radius: Optional[float] = None, iters: int = 30,
                       huber: Optional[float] = None, min_corr: int = 30, eps: float = 1e-5) -> IcpResult:
    """Robust point-to-plane ICP of query points Q (sensor frame) onto map M, starting at pose T0.

    `max_corr` is a coarse-to-fine schedule of correspondence distances (one stage per value, up to `iters`
    iterations each). Residuals get Huber weights with threshold `huber` (default inlier_dist / 2). Rotation
    updates are about the centroid of the transformed query cloud, which keeps the 6x6 system well scaled.
    After ICP a query point is an inlier if its nearest map point is within `nn_radius` (default
    3 * inlier_dist, i.e. the map covers it) and its point-to-plane residual is below `inlier_dist`.
    Conditioning is computed on the inliers."""
    huber = inlier_dist / 2 if huber is None else huber
    nn_radius = 3 * inlier_dist if nn_radius is None else nn_radius
    T = T0.copy()
    it_total, converged = 0, False
    nq = len(Q)
    fail = IcpResult(T0.copy(), 0.0, np.inf, 0, nq, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
    if nq < min_corr or M.tree is None or len(M) < min_corr:
        return fail
    for dmax in max_corr:
        converged = False
        for _ in range(iters):
            it_total += 1
            P = transform(T, Q)
            d, j = M.tree.query(P, distance_upper_bound=dmax)
            ok = np.isfinite(d)
            if ok.sum() < min_corr:
                return fail
            p, m, n = P[ok], M.P[j[ok]], M.n[j[ok]]
            r = ((p - m) * n).sum(1)
            a = np.abs(r)
            w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
            c = p.mean(0)
            J = np.hstack([np.cross(p - c, n), n])
            H = (J * w[:, None]).T @ J
            g = (J * w[:, None]).T @ r
            try:
                xi = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
            except np.linalg.LinAlgError:
                return fail
            D = se3_exp(xi)
            # rotate about c: x -> R (x - c) + c + v
            D[:3, 3] = D[:3, 3] + c - D[:3, :3] @ c
            T = D @ T
            if np.linalg.norm(xi[:3]) < eps and np.linalg.norm(xi[3:]) < eps:
                converged = True
                break
    P = transform(T, Q)
    d, j = M.tree.query(P, distance_upper_bound=nn_radius)
    ok = np.isfinite(d)
    jj = np.where(ok, j, 0)
    res_all = ((P - M.P[jj]) * M.n[jj]).sum(1)
    ok &= np.abs(res_all) < inlier_dist
    if ok.sum() < 6:
        fail.iterations = it_total
        return fail
    p, n, res = P[ok], M.n[j[ok]], res_all[ok]
    ct, cr, c6, _ = conditioning(p, n)
    dt, dr = pose_delta(T0, T)
    return IcpResult(T, float(ok.mean()), float(np.sqrt(np.mean(res**2))), int(ok.sum()), nq, dt, dr, ct, cr, c6,
                     it_total, converged)


def basin_test(Q: np.ndarray, M: MapCloud, T_ref: np.ndarray, shift_m: float, yaw_deg: float,
               up: Sequence[float] = (0.0, 0.0, 1.0), **icp_kwargs):
    """Noise-robust degeneracy test: restart ICP from T_ref perturbed along the directions that flat ground
    leaves free (horizontal translation and yaw about the world `up` axis, applied about the query cloud's
    centroid) and return the largest (translation m, rotation deg) by which a restart ends away from T_ref.

    Where the geometry pins the pose, every restart returns to T_ref (spread ~0); on a plane the restarts stay
    where they were put (spread ~ shift_m / yaw_deg). Unlike the Hessian eigenvalues, this is not inflated by
    noisy normals."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    c = transform(T_ref, Q).mean(0)
    worst_t, worst_r = 0.0, 0.0
    for s_a, s_b, s_y in ((1, 0, 1), (-1, 0, -1), (0, 1, -1), (0, -1, 1)):
        D = se3_exp(np.r_[np.radians(yaw_deg * s_y) * up, 0, 0, 0])
        D[:3, 3] = c - D[:3, :3] @ c + shift_m * (s_a * a + s_b * b)
        r = icp_point_to_plane(Q, M, D @ T_ref, **icp_kwargs)
        if r.fitness == 0:
            return np.inf, np.inf
        dt, dr = pose_delta(r.T, T_ref)
        worst_t, worst_r = max(worst_t, dt), max(worst_r, dr)
    return worst_t, worst_r
