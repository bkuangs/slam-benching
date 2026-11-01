#!/usr/bin/env python3
"""Ground-truth diagnostics of SP_SLAM3 learned-relocalization runs (algorithm/learned_reloc.csv).

For each session-B run: retrieval recall of the logged top-K keyframes, single-frame accepts and
their errors in the fixed map frame, false accepts, and whether failed attempts were retrieval
misses or verification failures (see slambench/evaluation/learned_reloc.py). Writes
<run>/eval/learned_reloc_diagnostics.json and prints one table row per run.

    python scripts/learned_reloc_diagnostics.py ~/scratch/slam-bench-work/runs/<config>[/<run id>] ...

A config output directory resolves to its latest successful run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slambench.algorithms import orbslam3_map_reuse as mr  # noqa: E402
from slambench.evaluation.learned_reloc import diagnose, read_log  # noqa: E402
from slambench.evaluation.map_reuse import fixed_alignment_from  # noqa: E402
from slambench.runner import make_dataset  # noqa: E402
from slambench.trajectory import Trajectory  # noqa: E402

POSITIVE_M = {"omnilrs": 2.0, "lunapolaris": 0.5}


def latest_run(p: Path) -> Path:
    if (p / "algorithm").is_dir():
        return p
    for run in sorted((c for c in p.iterdir() if c.is_dir()), reverse=True):
        try:
            if json.loads((run / "status.json").read_text()).get("status") in ("success", "init_failure"):
                return run
        except (OSError, ValueError):
            continue
    raise FileNotFoundError(f"no finished run in {p}")


def gt_cam0(cfg):
    ds = make_dataset(cfg)
    gt = ds.ground_truth()
    T_bs = ds.T_gt_body_sensor("cam0")
    t = Trajectory(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, gt.body_frame).transformed(T_right=T_bs)
    return np.asarray(t.timestamps_ns, np.int64), t.poses()


def run_one(run: Path) -> dict:
    cfg = yaml.safe_load((run / "config.resolved.yaml").read_text())
    algo = run / "algorithm"
    src = json.loads((algo / "map_source.json").read_text())
    map_run = Path(src["map_run"])
    map_cfg = yaml.safe_load((map_run / "config.resolved.yaml").read_text())
    fixed = fixed_alignment_from(map_run)
    loaded = mr.read_loaded_map(algo)
    kf_ts = {}
    import csv
    with open(algo / "loaded_map" / "keyframes.csv", newline="") as f:
        for r in csv.DictReader(f):
            kf_ts[int(r["keyframe_id"])] = int(r["timestamp_ns"])
    rows = read_log(algo / "learned_reloc.csv")
    pos_m = POSITIVE_M.get(cfg["dataset"]["name"], 1.0)
    res = diagnose(rows, kf_ts, gt_cam0(map_cfg), gt_cam0(cfg), fixed["R"], fixed["t"],
                   np.array(loaded["T_final_map"]), pos_m,
                   max_dt_s=cfg.get("evaluation", {}).get("association_max_dt_s", 0.02))
    res["summary"].update({"run": str(run), "positive_m": pos_m, "positive_deg": 30.0,
                           "map_run": str(map_run)})
    out = run / "eval" / "learned_reloc_diagnostics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return res["summary"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    keys = ("n_attempts", "recall_at_1", "recall_at_k", "n_accepted", "n_retrieval_miss", "n_verification_fail",
            "accept_rate_given_positive_retrieved", "positive_candidate_mp_matches_median",
            "n_false_single_frame_accepts", "n_confirmed", "n_false_confirmed")
    print("| run | " + " | ".join(keys) + " | desc ms | verify ms |")
    for p in args.runs:
        try:
            s = run_one(latest_run(p.expanduser()))
        except Exception as exc:  # noqa: BLE001
            print(f"| {p.name} | error: {type(exc).__name__}: {exc} |")
            continue
        vals = []
        for k in keys:
            v = s.get(k)
            vals.append("—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
        print(f"| {p.name} | " + " | ".join(vals) +
              f" | {s.get('desc_ms', {}).get('median', float('nan')):.1f} | "
              f"{s.get('verify_ms', {}).get('median', float('nan')):.1f} |")


if __name__ == "__main__":
    main()
