#!/usr/bin/env python3
"""FAST-LIO2 divergence report for the ground-only / point-count-control runs.

    python scripts/fastlio_divergence_report.py RUN_DIR [RUN_DIR ...] [--window-ate 120]

Per run, from eval/metrics.json: final ATE / RPE, coverage and loss events (what the estimator reports).
From algorithm/frame_log.csv (no ground truth): median effective matches (`n_effective`) over the first
60 s, the first time the estimated speed exceeds --max-speed (physically impossible for the platform),
and the first time n_effective stays 0 for >= 10 consecutive scans.
--window-ate S also scores the first S seconds on their own (own SE(3) alignment) through the regular
evaluator, i.e. accuracy before divergence (reads ground truth from the dataset; slow).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def first(t: np.ndarray, mask: np.ndarray) -> float:
    i = np.nonzero(mask)[0]
    return float(t[i[0]]) if len(i) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--max-speed", type=float, default=1.0, help="m/s; LunaPolaris rover moves at ~0.05 m/s")
    ap.add_argument("--window-ate", type=float, default=None)
    a = ap.parse_args()
    ds_cache = {}
    if a.window_ate:
        import compare_variants as cv
        from slambench.evaluation.metrics import umeyama
        orig = cv._input_timestamps

        def input_ts(run_dir):
            try:
                return orig(run_dir)
            except FileNotFoundError:     # prepared cache purged
                return pd.read_csv(Path(run_dir) / "frame_diagnostics.csv").timestamp_ns.values
        cv._input_timestamps = input_ts
    print("| run | ATE m | ATE max m | RPE 1 m (m / deg) | coverage | loss events | n_eff median, first 60 s "
          f"| speed > {a.max_speed:g} m/s at | n_eff = 0 (>= 10 scans) at |" + (" window ATE m |" if a.window_ate else ""))
    print("|---|---|---|---|---|---|---|---|---|" + ("---|" if a.window_ate else ""))
    for rd in a.runs:
        m = json.loads((rd / "eval/metrics.json").read_text())
        acc, cov = m["accuracy"]["final"], m["coverage"]
        f = pd.read_csv(rd / "algorithm/frame_log.csv")
        f = f[f.pose_valid == 1].reset_index(drop=True)
        t = (f.timestamp_ns - f.timestamp_ns.iloc[0]).values * 1e-9
        speed = np.linalg.norm(f[["vx", "vy", "vz"]].values, axis=1)
        zero = np.convolve((f.n_effective.values == 0).astype(int), np.ones(10, int), "valid") >= 10
        cells = [f"{rd.parent.name}/{rd.name[-6:]}", f"{acc['ate']['translation_m']['rmse']:.3g}",
                 f"{acc['ate']['translation_m']['max']:.3g}",
                 f"{acc['rpe'][0]['translation_m']['rmse']:.3g} / {acc['rpe'][0]['rotation_deg']['rmse']:.3g}",
                 f"{cov['frame_coverage']:.3f}", str(cov["n_tracking_loss_events"]),
                 f"{np.median(f.n_effective.values[t < 60]):.0f}", f"{first(t, speed > a.max_speed):.0f} s",
                 f"{first(t[:len(zero)], zero):.0f} s"]
        if a.window_ate:
            sv = cv.load_run(rd, ds_cache)["final_series"]
            w = (sv["t_ns"] - sv["t_ns"][0]) * 1e-9 < a.window_ate
            R, tr, _ = umeyama(sv["est_pos"][w], sv["gt_pos"][w], False)
            e = np.linalg.norm(sv["est_pos"][w] @ R.T + tr - sv["gt_pos"][w], axis=1)
            cells.append(f"{np.sqrt(np.mean(e ** 2)):.3f}")
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
