#!/usr/bin/env python3
"""Calibrate SP_SLAM3's L2 descriptor thresholds for unit-norm SuperPoint descriptors.

Upstream SP_SLAM3 (SPDetector::computeDescriptors) normalizes each descriptor *channel* over the
image's keypoints instead of each descriptor, so its descriptors have norm ~sqrt(256/N) (~0.46 at
N = 1200 keypoints) and an image-dependent per-channel scale. SPmatcher's thresholds (TH_LOW 0.25,
TH_HIGH 0.55, stereo (TH_LOW+TH_HIGH)/2) were tuned on those descriptors. With the fix
(``sp_unit_descriptors``) the same thresholds would reject almost every match, so they are mapped to
the unit-norm descriptor space by matching the recall of true correspondences: for each buggy
threshold T, the unit threshold is the quantile of the unit-norm distances of true matches at the
fraction of true matches the buggy descriptors accept at T.

True correspondences come from SuperPoint + LightGlue (cvg/LightGlue, MagicLeap SuperPoint weights,
the same weights as SP_SLAM3) on prepared LunaPolaris frames: stereo pairs (left/right, rectified,
|dy| <= 1.5 px, disparity >= 0) and temporal pairs (left frame t vs t+k, fundamental-matrix RANSAC
inliers). The buggy descriptors are emulated from the unit ones (b = u / ||u[:, c]|| over the
image's keypoints; SP_SLAM3 samples an already unit-normalized dense map, so this is exact up to
the bilinear interpolation shrinkage). Run inside the LAC image on a GPU node:

  apptainer exec --nv --env TORCH_HOME=/storage/ice1/3/6/bkuang7/slam-bench-work/torch_home \
      --bind /storage/ice1/3/6/bkuang7 lac_26c3a2b.sif python scripts/sp_descriptor_threshold_calibration.py \
      --prepared /storage/ice1/3/6/bkuang7/slam-bench-work/prepared/lunapolaris/sequence_01/stereo_deb052b8f12e \
      --out sp_threshold_calibration.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

BUGGY_THRESHOLDS = {"TH_LOW": 0.25, "stereo": 0.40, "TH_HIGH": 0.55}


def per_channel_normalized(u: np.ndarray) -> np.ndarray:
    """SP_SLAM3's (buggy) normalization of an [N, 256] descriptor matrix: every channel over the keypoints."""
    return u / np.maximum(np.linalg.norm(u, axis=0, keepdims=True), 1e-12)


def load_gray(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
    return im


def features(extractor, im: np.ndarray, device):
    t = torch.from_numpy(im).float().div(255.0)[None, None].to(device)
    with torch.no_grad():
        f = extractor({"image": t})
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prepared", type=Path, nargs="+", required=True, help="prepared stereo dirs (frames.csv)")
    ap.add_argument("--n-pairs", type=int, default=60, help="frames sampled per prepared dir")
    ap.add_argument("--offsets", type=int, nargs="+", default=[2, 6], help="temporal pair offsets (frames)")
    ap.add_argument("--max-keypoints", type=int, default=1200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints, detection_threshold=1e-5).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    dist = {"stereo": {"unit": [], "buggy": [], "imp_unit": [], "imp_buggy": []},
            "temporal": {"unit": [], "buggy": [], "imp_unit": [], "imp_buggy": []}}
    norms = []
    for prep in args.prepared:
        with open(prep / "frames.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        idx = np.linspace(0, len(rows) - 1 - max(args.offsets), args.n_pairs).astype(int)
        for i in idx:
            imL = load_gray(prep / rows[i]["left_path"])
            pairs = [("stereo", imL, load_gray(prep / rows[i]["right_path"]))]
            pairs += [("temporal", imL, load_gray(prep / rows[i + k]["left_path"])) for k in args.offsets]
            fL = features(extractor, imL, device)
            for kind, im0, im1 in pairs:
                f0 = fL
                f1 = features(extractor, im1, device)
                with torch.no_grad():
                    m = matcher({"image0": f0, "image1": f1})["matches"][0].cpu().numpy()
                k0 = f0["keypoints"][0].cpu().numpy()
                k1 = f1["keypoints"][0].cpu().numpy()
                u0 = f0["descriptors"][0].cpu().numpy().astype(np.float64)
                u1 = f1["descriptors"][0].cpu().numpy().astype(np.float64)
                if len(m) < 20:
                    continue
                p0, p1 = k0[m[:, 0]], k1[m[:, 1]]
                if kind == "stereo":
                    keep = (np.abs(p0[:, 1] - p1[:, 1]) <= 1.5) & (p0[:, 0] - p1[:, 0] >= 0)
                else:
                    _, inl = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 1.0, 0.999)
                    if inl is None:
                        continue
                    keep = inl.ravel().astype(bool)
                m = m[keep]
                if len(m) < 10:
                    continue
                b0, b1 = per_channel_normalized(u0), per_channel_normalized(u1)
                norms += list(np.linalg.norm(b0, axis=1))
                for space, d0, d1 in (("unit", u0, u1), ("buggy", b0, b1)):
                    D = np.sqrt(np.maximum(0.0, ((d0[m[:, 0], None, :] - d1[None, :, :]) ** 2).sum(-1)))
                    true = D[np.arange(len(m)), m[:, 1]]
                    D[np.arange(len(m)), m[:, 1]] = np.inf
                    # nearest impostor (excluding keypoints within 3 px of the true partner)
                    near = np.linalg.norm(k1[None, :, :] - k1[m[:, 1]][:, None, :], axis=-1) < 3.0
                    D[near] = np.inf
                    dist[kind][space] += list(true)
                    dist[kind]["imp_" + space] += list(D.min(axis=1))

    out = {"buggy_descriptor_norm": {"median": float(np.median(norms)), "p5": float(np.percentile(norms, 5)),
                                     "p95": float(np.percentile(norms, 95))}}
    for kind, d in dist.items():
        d = {k: np.asarray(v) for k, v in d.items()}
        res = {"n_true_matches": int(len(d["unit"]))}
        for name, tb in BUGGY_THRESHOLDS.items():
            recall = float(np.mean(d["buggy"] <= tb))
            tu = float(np.quantile(d["unit"], recall)) if recall > 0 else 0.0
            res[name] = {"buggy_threshold": tb, "true_match_recall": recall,
                         "buggy_impostor_rate": float(np.mean(d["imp_buggy"] <= tb)),
                         "unit_threshold": tu, "unit_impostor_rate": float(np.mean(d["imp_unit"] <= tu))}
        for space in ("unit", "buggy"):
            res[space + "_true_distance"] = {q: float(np.quantile(d[space], float(q) / 100))
                                             for q in ("10", "50", "90", "99")}
            res[space + "_impostor_distance"] = {q: float(np.quantile(d["imp_" + space], float(q) / 100))
                                                 for q in ("1", "10", "50")}
        out[kind] = res
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
