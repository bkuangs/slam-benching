import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from slambench.evaluation.map_quality import (align_by_orientation, PoseInterpolator, VoxelAccumulator, region_mask, score,
                                              umeyama_se3, visible_mask, voxel_downsample)


def test_pose_interpolator_midpoint_extrinsic_and_gaps():
    t = np.array([0, 1_000_000_000, 2_000_000_000, 5_000_000_000])
    p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    q = Rotation.from_euler("z", [0, 90, 90, 90], degrees=True).as_quat()
    T_bs = np.eye(4)
    T_bs[:3, 3] = [1, 0, 0]
    pi = PoseInterpolator(t, p, q, T_bs, max_gap_s=1.5)
    T = pi.poses([500_000_000])[0]
    assert np.allclose(Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True), [0, 0, 45])
    # sensor 1 m along the body x axis, body rotated 45 deg about z
    assert np.allclose(T[:3, 3], [0.5 + np.sqrt(0.5), np.sqrt(0.5), 0])
    assert pi.valid([500_000_000, 1_500_000_000]).all()
    assert not pi.valid([3_000_000_000]).any()          # 3 s gap in the ground truth
    assert not pi.valid([-1, 6_000_000_000]).any()      # outside


def test_voxel_accumulator_centroids_counts_and_merge():
    a, b = VoxelAccumulator(0.1, flush_points=2), VoxelAccumulator(0.1)
    a.add([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03]])
    a.add([[0.55, 0.0, 0.0]])
    b.add([[0.05, 0.05, 0.05], [-0.05, 0, 0]])
    a.merge(b)
    c, n = a.result()
    order = np.lexsort(c.T[::-1])
    c, n = c[order], n[order]
    assert n.tolist() == [1, 3, 1]
    assert np.allclose(c[1], [0.03, 0.03, 0.03])
    assert len(a.result(min_count=2)[0]) == 1
    assert len(voxel_downsample(np.random.default_rng(0).random((1000, 3)), 0.5)) == 8


def test_umeyama_recovers_rigid_transform():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    R = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    t = np.array([1.0, -2.0, 0.5])
    R_est, t_est = umeyama_se3(src, src @ R.T + t)
    assert np.allclose(R_est, R) and np.allclose(t_est, t)


def _wall_and_box():
    # wall at z = 3 in front of a camera at the origin looking along +z, a box 1 m behind the wall
    g = np.arange(-1, 1, 0.02)
    xx, yy = np.meshgrid(g, g)
    wall = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 3.0)]
    hidden = wall + [0, 0, 1.0]
    behind_cam = wall * [1, 1, -1]
    return wall, hidden, behind_cam


def test_visible_mask_frustum_depth_and_occlusion():
    wall, hidden, behind = _wall_and_box()
    ref = np.vstack([wall, hidden, behind])
    vis = visible_mask(ref, [np.eye(4)], fx=100, fy=100, cx=100, cy=100, width=200, height=200, max_depth=5)
    n = len(wall)
    assert vis[:n].mean() > 0.99
    assert vis[n:2 * n].mean() < 0.01         # occluded by the wall
    assert not vis[2 * n:].any()              # behind the camera
    vis_near = visible_mask(ref, [np.eye(4)], 100, 100, 100, 100, 200, 200, max_depth=2)
    assert not vis_near.any()


def test_region_mask_and_score():
    wall, hidden, _ = _wall_and_box()
    assert region_mask(wall + [0, 0, 0.1], wall).all()
    assert not region_mask(wall + [0, 0, 5.0], wall).any()
    vis = np.ones(len(wall), bool)
    s = score(wall + [0, 0, 0.03], wall, vis)
    assert s["precision_5cm"] == 1.0 and s["recall_5cm"] == 1.0 and s["fscore_5cm"] == 1.0
    half = wall[wall[:, 0] < 0]
    s = score(np.vstack([half, half + [0, 0, 0.5]]), wall, vis)
    assert s["precision_5cm"] == pytest.approx(0.5)
    assert s["recall_5cm"] == pytest.approx(0.5, abs=0.05)
    assert s["fscore_10cm"] == pytest.approx(0.5, abs=0.02)


def test_align_by_orientation_pins_tilt_on_planar_path():
    rng = np.random.default_rng(0)
    n = 50
    p_gt = np.c_[rng.uniform(0, 5, n), rng.uniform(0, 5, n), np.zeros(n)]  # planar path
    R_gt = Rotation.random(n, random_state=1).as_matrix()
    R_true = Rotation.from_euler("xyz", [2.0, -1.5, 30.0], degrees=True).as_matrix()
    t_true = np.array([0.3, -1.0, 0.2])
    # estimate frame: x_gt = R_true x_est + t_true
    p_est = (p_gt - t_true) @ R_true
    R_est = np.einsum("ji,njk->nik", R_true, R_gt)
    R, t = align_by_orientation(R_est, p_est, R_gt, p_gt)
    np.testing.assert_allclose(R, R_true, atol=1e-9)
    np.testing.assert_allclose(t, t_true, atol=1e-9)
