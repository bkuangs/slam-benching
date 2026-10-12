"""Aggregate run directories into one CSV row per run (failed runs included)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

HEAD_COLUMNS = [
    "run_id", "experiment", "status", "stage", "error_stage", "message", "estimator_outcome", "eval_status",
    "dataset", "sequence", "algorithm", "mode", "loop_closure", "clahe", "clahe_clip_limit", "playback_rate", "association_max_dt_s",
]
KINDS = ("online", "final")
KIND_COLUMNS = [
    "alignment", "sim3_scale", "n_sessions", "n_sessions_pooled", "n_matched",
    "ate_trans_rmse_m", "ate_trans_median_m", "ate_trans_max_m", "ate_rot_rmse_deg",
    "diag_const_rot_offset_deg", "evaluated_duration_s", "evaluated_duration_fraction",
    "matched_frame_fraction", "unavailable_reason",
]
TAIL_COLUMNS = [
    "coverage_source", "frame_coverage", "time_coverage", "time_to_first_ok_s",
    "n_loss_events", "n_unrecovered", "recovery_mean_s", "recovery_max_s", "n_maps",
    "final_trajectory_frame_coverage",
    "wall_time_s", "peak_rss_mb", "cpu_user_s", "per_frame_time_mean_s", "per_frame_time_p95_s",
    "frac_frames_over_period", "run_dir", "config_source",
]
#: Static columns; RPE columns ("<kind>_rpe_<delta><unit>_...") are added per configured interval.
COLUMNS = HEAD_COLUMNS + [f"{k}_{c}" for k in KINDS for c in KIND_COLUMNS] + TAIL_COLUMNS


def rpe_spec_name(r: Dict[str, Any]) -> str:
    return f"{r['delta']:g}{r['delta_unit']}"


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


def _kind_columns(kind: str, acc: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if not isinstance(acc, dict) or not acc:
        return row
    if acc.get("available") is False:
        row[f"{kind}_unavailable_reason"] = acc.get("reason", "")
        return row
    p = f"{kind}_"
    row.update({
        p + "alignment": _get(acc, "alignment", "type"),
        p + "sim3_scale": _get(acc, "alignment", "scale") if _get(acc, "alignment", "type") == "sim3" else "",
        p + "n_sessions": acc.get("n_sessions", 1 if "ate" in acc else ""),
        p + "n_sessions_pooled": acc.get("n_sessions_pooled", ""),
        p + "n_matched": _get(acc, "association", "n_matched"),
        p + "ate_trans_rmse_m": _get(acc, "ate", "translation_m", "rmse"),
        p + "ate_trans_median_m": _get(acc, "ate", "translation_m", "median"),
        p + "ate_trans_max_m": _get(acc, "ate", "translation_m", "max"),
        p + "ate_rot_rmse_deg": _get(acc, "ate", "rotation_deg", "rmse"),
        p + "diag_const_rot_offset_deg": _get(acc, "diagnostics", "constant_body_rotation_offset", "angle_deg"),
        p + "evaluated_duration_s": _get(acc, "evaluated_coverage", "evaluated_duration_s"),
        p + "evaluated_duration_fraction": _get(acc, "evaluated_coverage", "evaluated_duration_fraction"),
        p + "matched_frame_fraction": _get(acc, "evaluated_coverage", "matched_frame_fraction"),
    })
    if isinstance(_get(acc, "ate", default={}), dict) and acc["ate"].get("available") is False:
        row[p + "unavailable_reason"] = acc["ate"].get("reason", "")
    for r in acc.get("rpe") or []:
        if not isinstance(r, dict) or "delta" not in r:
            continue
        q = f"{p}rpe_{rpe_spec_name(r)}_"
        row[q + "trans_rmse_m"] = _get(r, "translation_m", "rmse")
        row[q + "rot_rmse_deg"] = _get(r, "rotation_deg", "rmse")
        row[q + "pairs"] = r.get("n_pairs", "")
    return row


def _blank_none(v: Any) -> Any:
    return "" if v is None else v


def _resolved_setting(run_dir: Path, key: str) -> Any:
    p = run_dir / "config.resolved.yaml"
    if not p.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(p.read_text()) or {}
    except Exception:  # noqa: BLE001 - unreadable config: leave the column empty
        return None
    return ((cfg.get("algorithm") or {}).get("settings") or {}).get(key)


def run_row(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    st = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
    m = {}
    if (run_dir / "eval" / "metrics.json").exists():
        m = json.loads((run_dir / "eval" / "metrics.json").read_text())
    cfg = st.get("config_summary", {})
    if "clahe" not in cfg:  # runs recorded before the clahe column existed
        cfg = {**cfg, "clahe": _resolved_setting(run_dir, "clahe")}
    if "clahe_clip_limit" not in cfg:  # before the clip limit was configurable it was fixed at 3.0
        clip = _resolved_setting(run_dir, "clahe_clip_limit")
        cfg = {**cfg, "clahe_clip_limit": clip if clip is not None else (3.0 if cfg.get("clahe") else None)}
    acc, cov, tim = m.get("accuracy", {}), m.get("coverage", {}), m.get("timing", {})
    if "ate" in acc or "available" in acc:  # metrics.json written before online/final evaluation
        acc = {"final": acc}
    first = next((a for a in acc.values() if isinstance(a, dict) and "association" in a), {})
    row = {
        "run_id": st.get("run_id", run_dir.name),
        "experiment": cfg.get("experiment", run_dir.parent.name),  # run dirs are <output_root>/<experiment>/<run_id>
        "status": st.get("status", "unknown"),
        "stage": st.get("stage", ""), "error_stage": _get(st, "error", "stage"),
        "message": st.get("message", ""), "estimator_outcome": st.get("estimator_outcome", ""),
        "eval_status": st.get("eval_status", ""),
        "dataset": cfg.get("dataset", ""), "sequence": cfg.get("sequence", ""),
        "algorithm": cfg.get("algorithm", ""), "mode": cfg.get("mode", ""),
        "loop_closure": cfg.get("loop_closure", ""), "clahe": _blank_none(cfg.get("clahe")),
        "clahe_clip_limit": _blank_none(cfg.get("clahe_clip_limit")),
        "playback_rate": cfg.get("playback_rate", ""),
        "association_max_dt_s": _get(first, "association", "max_dt_s"),
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
        "run_dir": str(run_dir), "config_source": cfg.get("source", ""),
    }
    for kind in KINDS:
        row.update(_kind_columns(kind, acc.get(kind)))
    return row


def find_runs(root: Path) -> List[Path]:
    return sorted(p.parent for p in Path(root).rglob("status.json"))


def write_summary(entries: Iterable[Union[Path, Dict[str, Any]]], out_csv: Path) -> Path:
    """One row per entry: a run directory, or a dict for an experiment that has no run directory."""
    rows = [dict(e) if isinstance(e, dict) else run_row(e) for e in entries]
    rpe_cols = sorted({k for r in rows for k in r if "_rpe_" in k},
                      key=lambda k: (KINDS.index(k.split("_")[0]) if k.split("_")[0] in KINDS else 9, k))
    fixed_kind = [f"{k}_{c}" for k in KINDS for c in KIND_COLUMNS]
    columns = HEAD_COLUMNS + fixed_kind + rpe_cols + TAIL_COLUMNS
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    return out_csv
