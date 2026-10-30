#!/usr/bin/env python3
"""Flare headroom probe: does a light source in view degrade correspondences and local drift?

    python scripts/flare_headroom_probe.py OUT_DIR --seq 11 --prepared <prepared stereo dir> \
        --runs '~/scratch/slam-bench-work/runs/lunapolaris_seq11_orbslam3_stereo_lean/*' [--step 3]

Per sampled left frame (every --step frames):
* `fs` flare-in-view score: for each saturated (>= 250) blob of >= 300 px, the mean grey of a 30 px
  ring around it divided by (1 + mean |Laplacian| in the ring / 4), max over blobs. A light source in
  view gives a bright, smooth veiling halo; sunlit rocks / regolith are bright but textured. Classes:
  none < 0.17 <= moderate < 0.40 <= strong (checked by eye on seq11; seq01/02 score 0 % flare).
* saturated fraction, 5th-percentile grey (veiling glare lifts the black level), ORB keypoints,
  row-constrained ORB stereo matches, ORB temporal RANSAC inliers (t -> t+3).
Per run: 1-s local drift |d_est - d_gt| of the online trajectory from the regular evaluator
(`compare_variants.local_drift`), joined to the nearest sampled frame.
Writes frames.csv, drift.csv and summary.md to OUT_DIR. Run from the repository root.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_variants as cv  # noqa: E402

BINS = [-1.0, 0.17, 0.40, 9.0]
LABELS = ["none", "moderate", "strong"]
RING = np.ones((61, 61), np.uint8)


def flare_score(g: np.ndarray) -> float:
    n, lab, st, _ = cv2.connectedComponentsWithStats((g >= 250).astype(np.uint8), 8)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F))
    best = 0.0
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 300:
            continue
        m = (lab == k).astype(np.uint8)
        ring = (cv2.dilate(m, RING) > 0) & (m == 0)
        best = max(best, (g[ring].mean() / 255.0) / (1.0 + lap[ring].mean() / 4.0))
    return float(best)


def frame_stats(args):
    d, rows, i = args
    ts, lp, rp = rows[i]
    L = cv2.imread(str(d / lp), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(str(d / rp), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kl, dl = orb.detectAndCompute(L, None)
    kr, dr = orb.detectAndCompute(R, None)
    rec = {"t_ns": int(ts), "fs": flare_score(L), "sat": float((L >= 250).mean()),
           "p05": float(np.percentile(L, 5)), "n_kp": len(kl), "stereo": 0, "temporal_inl": np.nan}
    if dl is not None and dr is not None:
        for m in bf.match(dl, dr):
            a, b = kl[m.queryIdx].pt, kr[m.trainIdx].pt
            rec["stereo"] += int(m.distance <= 64 and abs(a[1] - b[1]) <= 2.0 and 0.5 < a[0] - b[0] < 96)
    if i + 3 < len(rows) and dl is not None:
        k2, d2 = orb.detectAndCompute(cv2.imread(str(d / rows[i + 3][1]), cv2.IMREAD_GRAYSCALE), None)
        ms = [m for m in bf.match(dl, d2) if m.distance <= 50] if d2 is not None else []
        if len(ms) >= 8:
            p1 = np.float32([kl[m.queryIdx].pt for m in ms])
            p2 = np.float32([k2[m.trainIdx].pt for m in ms])
            _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.999)
            rec["temporal_inl"] = int(mask.sum()) if mask is not None else 0
    return rec


def input_ts_fallback(rd: Path) -> np.ndarray:
    """Input timestamps; falls back to frame_diagnostics.csv when the prepared cache was purged."""
    try:
        return ORIG_INPUT_TS(rd)
    except FileNotFoundError:
        return pd.read_csv(Path(rd) / "frame_diagnostics.csv").timestamp_ns.values


ORIG_INPUT_TS = cv._input_timestamps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--prepared", required=True, type=Path, help="prepared stereo dir with frames.csv")
    ap.add_argument("--runs", nargs="*", default=[], help="run-directory globs to score")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cv._input_timestamps = input_ts_fallback

    d = a.prepared.expanduser()
    rows = pd.read_csv(d / "frames.csv").values.tolist()
    with Pool(a.workers) as p:
        recs = p.map(frame_stats, [(d, rows, i) for i in range(0, len(rows), a.step)], chunksize=32)
    fr = pd.DataFrame(recs).sort_values("t_ns")
    fr["flare"] = pd.cut(fr.fs, BINS, labels=LABELS)
    fr.to_csv(a.out / "frames.csv", index=False)

    lines = [f"# Flare headroom probe, sequence {a.seq}", "",
             "Frame share: " + ", ".join(f"{k} {v:.1%}" for k, v in
                                          fr.flare.value_counts(normalize=True).reindex(LABELS).items()), "",
             "| flare | frames | p05 grey | ORB stereo matches median / p5 | temporal inliers median / p5 |",
             "|---|---|---|---|---|"]
    for f, g in fr.groupby("flare", observed=True):
        lines.append(f"| {f} | {len(g)} | {g.p05.median():.0f} | {g.stereo.median():.0f} / "
                     f"{np.percentile(g.stereo, 5):.0f} | {g.temporal_inl.median():.0f} / "
                     f"{np.nanpercentile(g.temporal_inl, 5):.0f} |")

    ds_cache, drift = {}, []
    for rd in sorted({r for pat in a.runs for r in glob.glob(os.path.expanduser(pat))}):
        run = cv.load_run(Path(rd), ds_cache)
        t, e = cv.local_drift(run, "online", 1.0)
        est = Path(rd).parent.name
        drift.append(pd.DataFrame({"t_ns": run["t0"] + (t * 1e9).astype(np.int64), "drift1s": e,
                                   "run": Path(rd).name, "estimator": est}))
    if drift:
        dd = pd.merge_asof(pd.concat(drift).sort_values("t_ns"), fr[["t_ns", "flare"]], on="t_ns",
                           direction="nearest", tolerance=int(0.15e9)).dropna(subset=["flare"])
        dd.to_csv(a.out / "drift.csv", index=False)
        lines += ["", "1-s local drift (mm): RMS / p99 per flare class; runs pooled", "",
                  "| estimator | runs | " + " | ".join(LABELS) + " | strong share of top-1 % windows |",
                  "|---|---|---|---|---|---|"]
        for est, g in dd.groupby("estimator"):
            cells = []
            for f in LABELS:
                x = g[g.flare == f].drift1s.values * 1000
                cells.append(f"{np.sqrt(np.mean(x ** 2)):.1f} / {np.percentile(x, 99):.1f}" if len(x) else "–")
            top = g[g.drift1s >= g.drift1s.quantile(0.99)]
            lines.append(f"| {est} | {g.run.nunique()} | " + " | ".join(cells) +
                         f" | {(top.flare == 'strong').mean():.0%} (base {(g.flare == 'strong').mean():.0%}) |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
