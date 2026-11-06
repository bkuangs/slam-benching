"""Synthetic tests for vpr/geometry.py (ICP and registration conditioning)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vpr import geometry as G  # noqa: E402


def _bumpy(n=6000, seed=0):
    """Non-degenerate terrain: ground with Gaussian bumps of different sizes (constrains x, y and yaw)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-3, 3, (n, 2))
    z = (0.4 * np.exp(-((xy[:, 0] - 1) ** 2 + (xy[:, 1] - 0.5) ** 2) / 0.3)
         + 0.3 * np.exp(-((xy[:, 0] + 1.2) ** 2 + (xy[:, 1] + 1) ** 2) / 0.2)
         + 0.25 * np.exp(-((xy[:, 0] + 0.5) ** 2 + (xy[:, 1] - 1.8) ** 2) / 0.15))
    return np.c_[xy, z]


def _plane(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-3, 3, (n, 2)), np.zeros(n)]


def _pose(rx_deg, ry_deg, rz_deg, t):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    T[:3, 3] = t
    return T


def test_voxel_downsample_one_point_per_voxel():
    P = np.array([[0.01, 0.01, 0.01], [0.02, 0.03, 0.04], [0.51, 0.0, 0.0]])
    out = G.voxel_downsample(P, 0.1)
    assert len(out) == 2
    assert np.allclose(sorted(out[:, 0]), [0.015, 0.51])


def test_depth_to_points_round_trip():
    K = np.array([[200.0, 0, 32], [0, 200.0, 24], [0, 0, 1]])
    depth = np.full((48, 64), 2.0)
    depth[0, 0] = 0.0
    P = G.depth_to_points(depth, K, stride=1)
    assert len(P) == 48 * 64 - 1
    uv = P[:, :2] / P[:, 2:] * 200.0 + [32, 24]
    assert np.allclose(P[:, 2], 2.0) and uv.min() >= 0 and uv[:, 0].max() <= 63


def test_conditioning_plane_vs_corner():
    plane = _plane()
    n_plane = np.tile([0, 0, 1.0], (len(plane), 1))
    ct, cr, c6, weak = G.conditioning(plane, n_plane)
    assert ct < 1e-9 and c6 < 1e-9
    # three orthogonal planes constrain everything
    rng = np.random.default_rng(1)
    pts, nrm = [], []
    for ax in range(3):
        p = rng.uniform(0, 2, (2000, 3))
        p[:, ax] = 0
        pts.append(p)
        nrm.append(np.tile(np.eye(3)[ax], (2000, 1)))
    ct2, cr2, c62, _ = G.conditioning(np.vstack(pts), np.vstack(nrm))
    assert ct2 > 0.3 and c62 > 0.01


def test_icp_recovers_small_offset_on_bumpy_terrain():
    W = _bumpy()
    M = G.MapCloud(W, voxel=0.05)
    T_true = _pose(0, 0, 0, [0, 0, 0])
    Q = _bumpy(3000, seed=5)  # different samples of the same surface, expressed in a "sensor" frame = world
    T0 = _pose(1.5, -1.0, 3.0, [0.15, -0.1, 0.05])
    r = G.icp_point_to_plane(Q, M, T0, max_corr=[0.5, 0.2, 0.1], inlier_dist=0.05)
    dt, dr = G.pose_delta(r.T, T_true)
    assert dt < 0.02 and dr < 0.5
    assert r.fitness > 0.9 and r.rmse < 0.01
    assert abs(r.corr_t - np.linalg.norm([0.15, -0.1, 0.05])) < 0.03
    assert r.cond_t > 1e-3  # bumps constrain horizontal translation


def test_icp_on_plane_agrees_with_shifted_pose_but_is_ill_conditioned():
    M = G.MapCloud(_plane(), voxel=0.05)
    Q = _plane(2000, seed=3)[:, :] * [0.5, 0.5, 1]
    T0 = _pose(0, 0, 10.0, [0.8, -0.6, 0.0])  # wrong in x, y and yaw: a plane cannot tell
    r = G.icp_point_to_plane(Q, M, T0, max_corr=[0.3, 0.1], inlier_dist=0.05)
    assert r.fitness > 0.95 and r.rmse < 1e-3   # shape "agrees"
    assert r.corr_t < 0.01                       # ICP does not move along the plane
    assert r.cond_t < 1e-6                       # but the conditioning shows there is no evidence


def test_icp_rejects_wrong_shape():
    M = G.MapCloud(_bumpy(), voxel=0.05)
    Q = _plane(2000, seed=3) * [0.5, 0.5, 1]
    Q[:, 2] += 0.6  # flat patch 0.6 m above the ground: no surface near it
    r = G.icp_point_to_plane(Q, M, np.eye(4), max_corr=[0.2], inlier_dist=0.05)
    assert r.fitness < 0.2


def test_basin_test_separates_plane_from_bumps():
    kw = dict(max_corr=[0.5, 0.2, 0.1], inlier_dist=0.05)
    Mb = G.MapCloud(_bumpy(20000), voxel=0.05)
    Qb = _bumpy(3000, seed=5) * [0.6, 0.6, 1]
    rb = G.icp_point_to_plane(Qb, Mb, np.eye(4), **kw)
    tb, _ = G.basin_test(Qb, Mb, rb.T, 0.2, 4.0, **kw)
    Mp = G.MapCloud(_plane(), voxel=0.05)
    Qp = _plane(2000, seed=3) * [0.5, 0.5, 1]
    rp = G.icp_point_to_plane(Qp, Mp, np.eye(4), **kw)
    tp, _ = G.basin_test(Qp, Mp, rp.T, 0.2, 4.0, **kw)
    assert tb < 0.03
    assert tp > 0.15
