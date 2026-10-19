#!/usr/bin/env python3
"""Top-down figure comparing the maps of an ORB-SLAM3 and an RTAB-Map stereo run of one sequence.

Panels: ORB-SLAM3 sparse map points, the TSDF surface from scripts/dense_from_keyframes.py (<orb run>/dense/),
RTAB-Map's exported dense cloud and its OctoMap (occupied and free cells). RTAB-Map's map is moved into
ORB-SLAM3's world frame with the rigid (SE3, Umeyama) transform that best aligns RTAB-Map's final node
positions to ORB-SLAM3's final keyframe positions (matched by timestamp, both are left-camera poses),
so all panels share axes. Plotted points: x (right) vs z (forward) of ORB-SLAM3's first camera frame.

Usage: python scripts/plot_dense_maps.py <orbslam3 run> <rtabmap run> <out.png> [--title TEXT]
Needs open3d and matplotlib.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd


def umeyama_se3(src: np.ndarray, dst: np.ndarray):
    """R, t minimizing |dst - (R src + t)|."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    return R, md - R @ ms


def read_cloud(path: Path):
    pc = o3d.io.read_point_cloud(str(path))
    cols = np.clip(np.asarray(pc.colors), 0, 1) if pc.has_colors() else None
    return np.asarray(pc.points), cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orb_run", type=Path)
    ap.add_argument("rtab_run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    orb, rtab = args.orb_run / "algorithm", args.rtab_run / "algorithm"

    kf = pd.read_csv(orb / "keyframes.csv")
    kf = kf[kf.in_final_frame == 1].sort_values("timestamp_ns")
    nodes = pd.read_csv(rtab / "nodes.csv")
    nodes = nodes[nodes.in_final == 1].sort_values("timestamp_ns")
    t_n = nodes.timestamp_ns.to_numpy(np.int64)
    t_k = kf.timestamp_ns.to_numpy(np.int64)
    j = np.clip(np.searchsorted(t_k, t_n), 1, len(t_k) - 1)
    j -= (t_n - t_k[j - 1]) < (t_k[j] - t_n)
    ok = np.abs(t_k[j] - t_n) < 0.5e9  # keyframes and RTAB-Map nodes are not the same frames
    src = nodes[["tx", "ty", "tz"]].to_numpy()[ok]
    dst = kf[["tx", "ty", "tz"]].to_numpy()[j[ok]]
    R, t = umeyama_se3(src, dst)
    res = np.linalg.norm(dst - (src @ R.T + t), axis=1)
    to_orb = lambda p: p @ R.T + t

    sparse = pd.read_csv(orb / "map_points.csv")
    sparse = sparse[sparse.in_final_frame == 1][["x", "y", "z"]].to_numpy()
    tsdf, tsdf_c = read_cloud(args.orb_run / "dense" / "cloud.ply")
    cloud, cloud_c = read_cloud(rtab / "rtabmap_cloud.ply")
    octo, octo_c = read_cloud(rtab / "rtabmap_octomap.ply")
    # rtabmap-export writes all OctoMap leaves: occupied cells in image color, free cells white
    # (OctoMap::createCloud); separate them by that color (a rare saturated occupied cell counts as free).
    is_free = np.all(octo_c > 0.999, axis=1)
    free, octo = to_orb(octo[is_free]), octo[~is_free]
    cloud, octo = to_orb(cloud), to_orb(octo)
    rtab_traj = to_orb(nodes[["tx", "ty", "tz"]].to_numpy())

    allxz = np.vstack([tsdf, cloud])[:, [0, 2]]
    lo, hi = np.percentile(allxz, 0.5, axis=0) - 0.5, np.percentile(allxz, 99.5, axis=0) + 0.5
    fig, ax = plt.subplots(2, 2, figsize=(13, 12), sharex=True, sharey=True)
    ax = ax.ravel()
    hlim = lambda y: dict(vmin=np.percentile(-y, 2), vmax=np.percentile(-y, 98))  # outliers off the scale
    ax[0].scatter(sparse[:, 0], sparse[:, 2], s=0.4, c=-sparse[:, 1], cmap="viridis", rasterized=True,
                  **hlim(sparse[:, 1]))
    ax[0].set_title(f"ORB-SLAM3 sparse map: {len(sparse):,} points (color = height)")
    ax[1].scatter(tsdf[:, 0], tsdf[:, 2], s=0.3, c=tsdf_c, rasterized=True)
    ax[1].set_title(f"ORB-SLAM3 keyframes + SGBM + 5 cm TSDF: {len(tsdf):,} points")
    ax[2].scatter(cloud[:, 0], cloud[:, 2], s=0.3, c=cloud_c, rasterized=True)
    ax[2].set_title(f"RTAB-Map dense cloud (2 cm voxels): {len(cloud):,} points")
    order = np.argsort(-octo[:, 1])  # draw highest cells last (y points down)
    ax[3].scatter(free[:, 0], free[:, 2], s=0.5, c="#cfe3f5", marker="s", rasterized=True, label="free")
    ax[3].scatter(octo[order, 0], octo[order, 2], s=0.5, c=-octo[order, 1], cmap="viridis", marker="s",
                  rasterized=True, **hlim(octo[:, 1]))
    ax[3].set_title(f"RTAB-Map OctoMap 5 cm: {len(octo):,} occupied (color = height), {len(free):,} free")
    for i, a in enumerate(ax):
        if i < 2:
            a.plot(kf.tx, kf.tz, "r-", lw=0.6, label="ORB-SLAM3 keyframes")
        else:
            a.plot(rtab_traj[:, 0], rtab_traj[:, 2], "r-", lw=0.6, label="RTAB-Map nodes")
        a.set_aspect("equal")
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(lo[1], hi[1])
        a.legend(loc="lower right", fontsize=8)
    for a in ax[2:]:
        a.set_xlabel("x (m)")
    for a in ax[::2]:
        a.set_ylabel("z (m)")
    fig.suptitle(f"{args.title}\ntop-down in ORB-SLAM3's world frame; RTAB-Map aligned by its trajectory "
                 f"(SE3, {ok.sum()} node pairs, residual RMS {np.sqrt(np.mean(res ** 2)) * 100:.1f} cm)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print(f"{args.out}: aligned {ok.sum()} node pairs, residual RMS {np.sqrt(np.mean(res ** 2)):.3f} m")


if __name__ == "__main__":
    main()
