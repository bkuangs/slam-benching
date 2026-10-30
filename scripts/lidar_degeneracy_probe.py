#!/usr/bin/env python3
"""LiDAR registration conditioning on LunaPolaris scans: full scan vs ground-only crops.

    python scripts/lidar_degeneracy_probe.py --seq 01 11 [--scans 16] [--plot-dir DIR]

For evenly spaced scans, points are mapped into the rover ground-plane frame (leo01/base_footprint,
/tf_static only) and cropped (max height above it, max horizontal range). Per crop: points kept, share of
vertical surfaces (|n_z| < 0.5), and the conditioning of the point-to-plane ICP information matrix
H = sum J^T J, J = [p x n, n] (per point, on a 20k-point subsample with planar normals): translation and
rotation blocks' lambda_min / lambda_max and the weakest direction (|x, y, z| components; x forward,
z up). Values near 0 mean registration does not constrain that direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slambench.datasets.lunapolaris import LunaPolarisAdapter, crop_lidar_points, livox_points  # noqa: E402

CROPS = [("full", None, None), ("h0.5", 0.5, None), ("h0.5_r3", 0.5, 3.0), ("h0.3_r2.5", 0.3, 2.5)]


def normals(P: np.ndarray, k: int = 12):
    _, idx = cKDTree(P).query(P, k)
    nb = P[idx] - P[idx].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb) / k)
    return v[:, :, 0], (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)


def conditioning(P: np.ndarray, n: np.ndarray):
    J = np.hstack([np.cross(P, n), n])
    H = J.T @ J / len(P)
    wt, vt = np.linalg.eigh(H[3:, 3:])
    wr, vr = np.linalg.eigh(H[:3, :3])
    return wt[0] / wt[2], np.abs(vt[:, 0]), wr[0] / wr[2], np.abs(vr[:, 0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs="+", required=True)
    ap.add_argument("--root", type=Path, default=Path.home() / "scratch/LunaPolaris")
    ap.add_argument("--scans", type=int, default=16)
    ap.add_argument("--plot-dir", type=Path)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for seq in a.seq:
        ds = LunaPolarisAdapter(a.root, f"sequence_{seq}")
        T = ds.static_tf().lookup(ds.options["lidar_crop_frame"], ds.options["lidar_frame"])
        res = {c[0]: [] for c in CROPS}
        with ds._reader() as r:
            conns = [c for c in r.connections if c.topic == ds.options["lidar_topic"]]
            t0, t1 = r.start_time, r.end_time
            for k in range(a.scans):
                ts = t0 + int((k + 0.5) / a.scans * (t1 - t0))
                for c, _, raw in r.messages(connections=conns, start=ts, stop=ts + int(3e8)):
                    _, pts = livox_points(r.deserialize(raw, c.msgtype))
                    break
                pts = pts[np.c_[pts["x"], pts["y"], pts["z"]].any(axis=1)]
                P = np.c_[pts["x"], pts["y"], pts["z"]].astype(float) @ T[:3, :3].T + T[:3, 3]
                for name, h, rr in CROPS:
                    Q = P[crop_lidar_points(pts, T, h, rr)]
                    if len(Q) < 200:
                        continue
                    Q = Q[rng.choice(len(Q), min(len(Q), 20000), replace=False)]
                    n, planar = normals(Q)
                    Q, n = Q[planar > 0.3], n[planar > 0.3]
                    res[name].append((len(Q), *conditioning(Q, n), np.mean(np.abs(n[:, 2]) < 0.5)))
                if a.plot_dir is not None and k == a.scans // 4:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    for axis, (name, h, rr) in zip(ax, (CROPS[0], CROPS[3])):
                        Q = P[crop_lidar_points(pts, T, h, rr)]
                        axis.scatter(Q[:, 0], Q[:, 1], s=0.2, c=Q[:, 2], cmap="viridis")
                        axis.set_aspect("equal")
                        axis.set_title(f"seq{seq} {name} ({len(Q)} returns), top view, colour = height")
                    a.plot_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(a.plot_dir / f"lidar_crop_seq{seq}.png", dpi=80)
                    plt.close(fig)
        print(f"\nsequence_{seq} ({a.scans} scans; medians)")
        print("| crop | planar points | vertical-surface share | translation cond. (min) | weakest translation |x,y,z| "
              "| rotation cond. | weakest rotation axis |x,y,z| |")
        print("|---|---|---|---|---|---|---|")
        for name, rows in res.items():
            if not rows:
                continue
            col = list(zip(*rows))
            print(f"| {name} | {np.median(col[0]):.0f} | {np.median(col[5]):.2f} | {np.median(col[1]):.3f} "
                  f"({min(col[1]):.4f}) | {np.round(np.median(col[2], 0), 2)} | {np.median(col[3]):.3f} | "
                  f"{np.round(np.median(col[4], 0), 2)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
