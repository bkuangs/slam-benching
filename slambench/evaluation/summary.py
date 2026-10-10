"""Aggregate run directories into one CSV row per run (failed runs included)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

COLUMNS = [
    "run_id", "status", "message", "eval_status", "dataset", "sequence", "algorithm", "mode",
    "loop_closure", "playback_rate", "alignment", "association_max_dt_s",
    "ate_trans_rmse_m", "ate_trans_median_m", "ate_trans_max_m", "ate_rot_rmse_deg",
    "diag_const_rot_offset_deg",
    "rpe_1_spec", "rpe_1_trans_rmse_m", "rpe_1_rot_rmse_deg", "rpe_1_pairs",
    "rpe_2_spec", "rpe_2_trans_rmse_m", "rpe_2_rot_rmse_deg", "rpe_2_pairs",
    "n_matched", "evaluated_duration_s", "evaluated_duration_fraction", "matched_frame_fraction",
    "coverage_source", "frame_coverage", "time_coverage", "time_to_first_ok_s",
    "n_loss_events", "n_unrecovered", "recovery_mean_s", "recovery_max_s", "n_maps",
    "final_trajectory_frame_coverage",
    "wall_time_s", "peak_rss_mb", "cpu_user_s", "per_frame_time_mean_s", "per_frame_time_p95_s",
    "frac_frames_over_period", "run_dir",
]


def _get(d: Any, *keys, default=""):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, list) and isinstance(k, int) and k < len(d):
            d = d[k]
        else:
            return default
    if isinstance(d, dict) and d.get("available") is False:
        return default
    return d


def run_row(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    st = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
    m = {}
    if (run_dir / "eval" / "metrics.json").exists():
        m = json.loads((run_dir / "eval" / "metrics.json").read_text())
    cfg = st.get("config_summary", {})
    acc, cov, tim = m.get("accuracy", {}), m.get("coverage", {}), m.get("timing", {})
    row = {
        "run_id": st.get("run_id", run_dir.name), "status": st.get("status", "unknown"),
        "message": st.get("message", ""), "eval_status": st.get("eval_status", ""),
        "dataset": cfg.get("dataset", ""), "sequence": cfg.get("sequence", ""),
        "algorithm": cfg.get("algorithm", ""), "mode": cfg.get("mode", ""),
        "loop_closure": cfg.get("loop_closure", ""), "playback_rate": cfg.get("playback_rate", ""),
        "alignment": _get(acc, "alignment", "type"),
        "association_max_dt_s": _get(acc, "association", "max_dt_s"),
        "ate_trans_rmse_m": _get(acc, "ate", "translation_m", "rmse"),
        "ate_trans_median_m": _get(acc, "ate", "translation_m", "median"),
        "ate_trans_max_m": _get(acc, "ate", "translation_m", "max"),
        "ate_rot_rmse_deg": _get(acc, "ate", "rotation_deg", "rmse"),
        "diag_const_rot_offset_deg": _get(acc, "diagnostics", "constant_body_rotation_offset", "angle_deg"),
        "n_matched": _get(acc, "association", "n_matched"),
        "evaluated_duration_s": _get(acc, "evaluated_coverage", "evaluated_duration_s"),
        "evaluated_duration_fraction": _get(acc, "evaluated_coverage", "evaluated_duration_fraction"),
        "matched_frame_fraction": _get(acc, "evaluated_coverage", "matched_frame_fraction"),
        "coverage_source": _get(cov, "source"),
        "frame_coverage": _get(cov, "frame_coverage"), "time_coverage": _get(cov, "time_coverage"),
        "time_to_first_ok_s": _get(cov, "time_to_first_ok_s"),
        "n_loss_events": _get(cov, "n_tracking_loss_events"), "n_unrecovered": _get(cov, "n_unrecovered"),
        "recovery_mean_s": _get(cov, "recovery_duration_s", "mean"),
        "recovery_max_s": _get(cov, "recovery_duration_s", "max"),
        "n_maps": _get(cov, "n_maps"),
        "final_trajectory_frame_coverage": _get(cov, "final_trajectory_frame_coverage"),
        "wall_time_s": st.get("wall_time_s", ""), "peak_rss_mb": st.get("peak_rss_mb", ""),
        "cpu_user_s": st.get("cpu_user_s", ""),
        "per_frame_time_mean_s": _get(tim, "per_frame_time_s", "mean"),
        "per_frame_time_p95_s": _get(tim, "per_frame_time_s", "p95"),
        "frac_frames_over_period": _get(tim, "realtime", "fraction_frames_over_period"),
        "run_dir": str(run_dir),
    }
    for k in (1, 2):
        r = _get(acc, "rpe", k - 1, default={})
        row[f"rpe_{k}_spec"] = f"{r['delta']:g} {r['delta_unit']}" if r else ""
        row[f"rpe_{k}_trans_rmse_m"] = _get(r, "translation_m", "rmse")
        row[f"rpe_{k}_rot_rmse_deg"] = _get(r, "rotation_deg", "rmse")
        row[f"rpe_{k}_pairs"] = _get(r, "n_pairs")
    return row


def find_runs(root: Path) -> List[Path]:
    return sorted(p.parent for p in Path(root).rglob("status.json"))


def write_summary(run_dirs: Iterable[Path], out_csv: Path) -> Path:
    rows = [run_row(d) for d in run_dirs]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    return out_csv
