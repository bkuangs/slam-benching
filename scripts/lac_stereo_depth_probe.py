"""Dump LAC's stereo keypoint matches (SuperPoint + LightGlue, exactly as LAC's tracker computes them)
for every Nth frame, for comparison against a depth reference (scripts/stereo_depth_bias.py).

Runs inside the LAC container (needs CUDA):
  apptainer exec --nv -B <dirs> <lac.sif> python scripts/lac_stereo_depth_probe.py \
      <lac_settings.json> <input dir with frames.csv> <out.npz> [--step N]

lac_settings.json is the driver settings of a LAC run (only "camera" is used). The output holds, per
match, the frame timestamp, left/right keypoint pixels; depth = fx * b / (uL - uR).
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

DRIVER = "/opt/slambench/slambench_lac_driver.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("settings", type=Path)
    ap.add_argument("input", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("drv", DRIVER)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    cam = json.loads(args.settings.read_text())["camera"]
    drv.patch_lac_params(cam)

    import cv2
    import torch
    from lac.slam.feature_tracker import FeatureTracker
    from lac.slam.semantic_feature_tracker import MAX_STEREO_MATCHES

    torch.set_grad_enabled(False)
    tracker = FeatureTracker({"FrontLeft": {"width": cam["width"], "height": cam["height"]}})
    with open(args.input / "frames.csv", newline="") as f:
        frames = list(csv.DictReader(f))[:: args.step]
    cols = {k: [] for k in ("t_ns", "uL", "vL", "uR", "vR")}
    for r in frames:
        L = cv2.imread(str(args.input / r["left_path"]), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(args.input / r["right_path"]), cv2.IMREAD_GRAYSCALE)
        fl, fr = tracker.extract_feats(L), tracker.extract_feats(R)
        m = tracker.match_feats(fl, fr, MAX_STEREO_MATCHES)
        kl = fl["keypoints"][0][m[:, 0]].cpu().numpy()
        kr = fr["keypoints"][0][m[:, 1]].cpu().numpy()
        cols["t_ns"].append(np.full(len(kl), int(r["timestamp_ns"]), np.int64))
        cols["uL"].append(kl[:, 0]); cols["vL"].append(kl[:, 1])
        cols["uR"].append(kr[:, 0]); cols["vR"].append(kr[:, 1])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, camera=json.dumps(cam), **out)
    print(f"{len(frames)} frames, {len(out['uL'])} matches -> {args.out}")


if __name__ == "__main__":
    main()
