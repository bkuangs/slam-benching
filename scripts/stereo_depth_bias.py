"""Bias of LAC's stereo keypoint depths against a depth reference.

Input: matches from scripts/lac_stereo_depth_probe.py (left/right SuperPoint keypoints matched by LightGlue).
Reference depth at each left keypoint:
  --gt-depth DIR        simulator z-depth images (DIR/<timestamp_ns>.png, uint16 mm), bilinear;
  --lidar-run RUN       LunaPolaris: the cached LiDAR reference map of eval_map_quality.py (--reference),
                        seen from the ground-truth cam0 pose of RUN's dataset (near-surface depth of the
                        reference points projecting within --radius px of the keypoint).

Reports, per depth bin, the median ratio LAC depth / reference depth and the median disparity error
(LAC disparity - fx*b/z_ref, px). A constant disparity offset d0 > 0 shrinks depths by z/(1 + z*d0/(fx*b)),
i.e. more at long range; a calibration (baseline/focal) error is a depth-independent ratio.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_matches(path: Path, max_row_diff: float, frame_step: int = 1):
    z = np.load(path)
    cam = json.loads(str(z["camera"]))
    m = {k: z[k] for k in ("t_ns", "uL", "vL", "uR", "vR")}
    keep = (np.abs(m["vL"] - m["vR"]) <= max_row_diff) & (m["uL"] - m["uR"] > 0)
    keep &= np.isin(m["t_ns"], np.unique(m["t_ns"])[::frame_step])
    return cam, {k: v[keep] for k, v in m.items()}, int(len(keep))


def bilinear(img, u, v):
    h, w = img.shape
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (u0 >= 0) & (v0 >= 0) & (u0 < w - 1) & (v0 < h - 1)
    out = np.full(len(u), np.nan)
    a, b, uu, vv = u[ok] - u0[ok], v[ok] - v0[ok], u0[ok], v0[ok]
    q = [img[vv, uu], img[vv, uu + 1], img[vv + 1, uu], img[vv + 1, uu + 1]]
    # reject depth discontinuities (corners more than 5% apart)
    qa = np.stack(q).astype(float)
    smooth = (qa.max(0) - qa.min(0)) <= 0.05 * qa.min(0)
    val = (1 - a) * (1 - b) * qa[0] + a * (1 - b) * qa[1] + (1 - a) * b * qa[2] + a * b * qa[3]
    val[(~smooth) | (qa.min(0) <= 0)] = np.nan
    out[ok] = val
    return out


def ref_from_gt_depth(m, depth_dir: Path):
    z = np.full(len(m["uL"]), np.nan)
    for t in np.unique(m["t_ns"]):
        sel = m["t_ns"] == t
        img = cv2.imread(str(depth_dir / f"{t}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        z[sel] = bilinear(img.astype(np.float64) / 1000.0, m["uL"][sel], m["vL"][sel])
    return z


def ref_from_lidar(m, cam, run: Path, reference: Path, radius: float, max_depth: float):
    import sys
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_map_quality import adapter_for
    from slambench.evaluation.map_quality import PoseInterpolator

    pts = np.load(reference)["points"]
    ad = adapter_for(run)
    gt = ad.ground_truth()
    pose = PoseInterpolator(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, ad.T_gt_body_sensor("cam0"))
    z = np.full(len(m["uL"]), np.nan)
    ts = np.unique(m["t_ns"])
    ok_t = pose.valid(ts)
    for t, T in zip(ts[ok_t], pose.poses(ts[ok_t])):
        sel = np.flatnonzero(m["t_ns"] == t)
        Tcw = np.linalg.inv(T)
        pc = pts @ Tcw[:3, :3].T + Tcw[:3, 3]
        pc = pc[(pc[:, 2] > 0.2) & (pc[:, 2] < max_depth)]
        u = cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"]
        v = cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"]
        inside = (u > -radius) & (v > -radius) & (u < cam["width"] + radius) & (v < cam["height"] + radius)
        u, v, d = u[inside], v[inside], pc[inside, 2]
        if not len(d):
            continue
        tree = cKDTree(np.c_[u, v])
        for i, nb in zip(sel, tree.query_ball_point(np.c_[m["uL"][sel], m["vL"][sel]], radius)):
            if len(nb) >= 3:
                dd = np.sort(d[nb])
                front = dd[dd <= dd[0] + 0.10]   # nearest surface (ignore points behind it)
                if len(front) >= 3:
                    z[i] = np.median(front)
    return z


def report(name, cam, m, z_ref, n_raw, bins):
    fb = cam["fx"] * cam["baseline_m"]
    disp = m["uL"] - m["uR"]
    z_lac = fb / disp
    ok = np.isfinite(z_ref) & (z_ref > 0)
    z_lac, z_ref, disp = z_lac[ok], z_ref[ok], disp[ok]
    derr = disp - fb / z_ref
    ratio = z_lac / z_ref
    out = {"name": name, "matches_raw": n_raw, "matches_with_reference": int(ok.sum()),
           "median_ratio": float(np.median(ratio)), "median_disp_err_px": float(np.median(derr)),
           "keypoint_subpixel_fraction": float(np.mean(np.abs(m["uL"] - np.round(m["uL"])) > 1e-3)),
           "bins": []}
    # robust fit of disparity error: derr = d0 + k * disp  (k != 0: focal/baseline scale error)
    inl = np.abs(derr - np.median(derr)) < 3 * 1.4826 * np.median(np.abs(derr - np.median(derr)))
    A = np.c_[np.ones(inl.sum()), fb / z_ref[inl]]
    (d0, k), *_ = np.linalg.lstsq(A, derr[inl], rcond=None)
    out["fit"] = {"disp_offset_px": float(d0), "disp_scale_minus_1": float(k), "inliers": float(inl.mean())}
    print(f"\n== {name}: {ok.sum():,}/{n_raw:,} matches with reference depth")
    print(f"   median depth ratio LAC/ref {out['median_ratio']:.3f}, median disparity error "
          f"{out['median_disp_err_px']:+.3f} px; fit: offset {d0:+.3f} px, scale {k:+.4f}; "
          f"sub-pixel keypoints {out['keypoint_subpixel_fraction'] * 100:.0f}%")
    print(f"   {'ref depth (m)':>14} {'n':>7} {'ratio p25/med/p75':>22} {'disp err px (med)':>18}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (z_ref >= lo) & (z_ref < hi)
        if s.sum() < 20:
            continue
        q = np.percentile(ratio[s], [25, 50, 75])
        out["bins"].append({"lo": lo, "hi": hi, "n": int(s.sum()), "ratio_p25_50_75": q.tolist(),
                            "disp_err_median_px": float(np.median(derr[s]))})
        print(f"   {f'{lo:.1f}-{hi:.1f}':>14} {s.sum():>7,} {q[0]:>7.3f}/{q[1]:.3f}/{q[2]:.3f} "
              f"{np.median(derr[s]):>+18.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matches", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gt-depth", type=Path)
    g.add_argument("--lidar-run", type=Path)
    ap.add_argument("--reference", type=Path, help="eval_map_quality reference_*.npz (with --lidar-run)")
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--max-row-diff", type=float, default=2.0, help="drop stereo matches off the epipolar line")
    ap.add_argument("--frame-step", type=int, default=1, help="use every Nth probed frame")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    cam, m, n_raw = load_matches(args.matches, args.max_row_diff, args.frame_step)
    if args.gt_depth:
        z = ref_from_gt_depth(m, args.gt_depth)
    else:
        if not args.reference:
            ap.error("--lidar-run needs --reference")
        z = ref_from_lidar(m, cam, args.lidar_run, args.reference, args.radius, args.max_depth)
    res = report(args.matches.stem, cam, m, z, n_raw, [0.5, 1, 1.5, 2, 3, 4, 6, 10])
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
