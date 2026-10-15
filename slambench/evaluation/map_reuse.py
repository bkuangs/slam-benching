"""Map-reuse (localization against a saved map) metrics.

A localization run loads the map saved by an earlier *mapping* run and reports poses in that map's
frame. Its poses are evaluated in that fixed frame: the map->ground-truth transform is the rigid
SE(3) alignment fitted once on the mapping run's final trajectory (its eval/metrics.json), never
re-fitted on the localization run. A localization into the wrong place in the map therefore shows up
as a large error instead of being absorbed by an alignment.

Reported (all on data time unless stated):
* initialization: time / frames / distance travelled until the first accepted localization and
  the processing time spent until then (the map load is reported separately)
* coverage: fraction of input frames with a localized pose, and with a localized pose whose error
  is below each threshold ("correctly localized coverage")
* localization events (accepted loop-closure / proximity links to map nodes) and false
  relocalizations: events after which the pose error exceeds the false-relocalization threshold
* pose error distribution of localized frames in the fixed map frame
* computation: per-frame and per-map-update processing time, real-time factor, map load time,
  wall time, peak RSS
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..trajectory import Trajectory
from .metrics import _rot_angle_deg, _stats, associate, unavailable

DEFAULT_MAP_REUSE: Dict[str, Any] = {
    "error_thresholds_m": [0.1, 0.25, 0.5],
    "false_relocalization_m": 0.5,     # event is false if the post-event error exceeds this ...
    "false_relocalization_deg": 15.0,  # ... or the rotation error exceeds this
    "staleness_s": [1.0, 5.0],         # coverage with an accepted localization at most this old
}


def fixed_alignment_from(map_run: Path) -> Dict[str, Any]:
    """The mapping run's single-session SE(3) map->GT alignment of its final trajectory."""
    p = Path(map_run) / "eval" / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"mapping run not evaluated: {p} missing")
    m = json.loads(p.read_text())
    acc = m.get("accuracy", {}).get("final", {})
    sessions = acc.get("sessions") or []
    if acc.get("available") is False or len(sessions) != 1:
        raise ValueError(f"mapping run {map_run} final trajectory must be one evaluated coordinate session "
                         f"(found {len(sessions)})")
    al = sessions[0]["alignment"]
    if al.get("type") != "se3":
        raise ValueError(f"mapping run alignment must be se3, not {al.get('type')}")
    return {"R": np.array(al["R"]), "t": np.array(al["t"]), "map_run": str(map_run),
            "mapping_run_id": m.get("run_id"), "mapping_sequence": m.get("sequence"),
            "mapping_final_ate_rmse_m": acc.get("ate", {}).get("translation_m", {}).get("rmse"),
            "T_gt_body_trajectory_frame": acc.get("ground_truth", {}).get("T_gt_body_trajectory_frame")}


def map_reuse_metrics(log_rows: List[Dict[str, Any]], input_ts: np.ndarray, gt_sensor: Trajectory,
                      fixed: Dict[str, Any], cfg: Dict[str, Any], run_status: Dict[str, Any],
                      map_info: Dict[str, Any], max_dt_s: float, out_dir: Path = None) -> Dict[str, Any]:
    c = dict(DEFAULT_MAP_REUSE)
    c.update(cfg or {})
    input_ts = np.asarray(input_ts, dtype=np.int64)
    t0 = int(input_ts[0])
    dts = np.diff(input_ts) * 1e-9
    own = np.r_[dts, np.median(dts) if len(dts) else 0.0]  # frame i owns [t_i, t_i+1)
    seq_dur = float(own.sum())
    own_by_ts = dict(zip(input_ts.tolist(), own.tolist()))
    res: Dict[str, Any] = {
        "label": ("map reuse: poses evaluated in the saved map's fixed frame with the mapping run's "
                  "map->GT transform (no alignment fitted on this run)"),
        "settings": c, "fixed_alignment": {k: v for k, v in fixed.items() if k not in ("R", "t")},
        "map": map_info, "n_input_frames": int(len(input_ts)), "sequence_duration_s": seq_dur}
    res["fixed_alignment"]["R"] = fixed["R"].tolist()
    res["fixed_alignment"]["t"] = fixed["t"].tolist()

    R, t = fixed["R"], fixed["t"]
    rows = [r for r in log_rows if r["pose"] is not None]
    ts = np.array([r["timestamp_ns"] for r in rows], dtype=np.int64)
    ei, gi, _ = associate(ts, gt_sensor.timestamps_ns, max_dt_s)
    err_t = np.full(len(rows), np.nan)
    err_r = np.full(len(rows), np.nan)
    if len(ei):
        P = np.array([rows[k]["pose"] for k in ei])
        E = Trajectory(ts[ei], P[:, :3], P[:, 3:], "cam0")
        est_pos = E.positions @ R.T + t
        err_t[ei] = np.linalg.norm(est_pos - gt_sensor.positions[gi], axis=1)
        if gt_sensor.has_orientation:
            err_r[ei] = _rot_angle_deg(R[None] @ E.poses()[:, :3, :3], gt_sensor.poses()[gi][:, :3, :3])
    by_ts = {r["timestamp_ns"]: (r, err_t[k], err_r[k]) for k, r in enumerate(rows)}

    # GT distance travelled up to each input frame (nearest GT sample)
    _, gj, _ = associate(input_ts, gt_sensor.timestamps_ns, 1.0)
    gpos = gt_sensor.positions
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(gpos, axis=0), axis=1))]
    gt_dist = cum[gj] - cum[gj[0]] if len(gj) == len(input_ts) else None

    # ---- initialization
    first = next((r for r in log_rows if r["localized"] and r["pose"] is not None), None)
    if first is None:
        res["initialization"] = {"localized": False, "reason": "never localized against the map"}
    else:
        k = int(np.searchsorted(input_ts, first["timestamp_ns"]))
        proc = sum(r["track_time_s"] for r in log_rows if r["timestamp_ns"] <= first["timestamp_ns"])
        e = by_ts[first["timestamp_ns"]]
        res["initialization"] = {
            "localized": True, "frame_idx": first["frame_idx"],
            "data_time_s": float((first["timestamp_ns"] - t0) * 1e-9),
            "processing_time_s": float(proc),
            "gt_distance_travelled_m": float(gt_dist[k]) if gt_dist is not None else None,
            "error_at_first_localization_m": None if np.isnan(e[1]) else float(e[1]),
            "rotation_error_at_first_localization_deg": None if np.isnan(e[2]) else float(e[2])}

    # ---- events
    events, last_loc_ts = [], None
    prev_err = None
    for r in log_rows:
        cur = by_ts.get(r["timestamp_ns"])
        if r["loc_event"]:
            et = None if cur is None or np.isnan(cur[1]) else float(cur[1])
            er = None if cur is None or np.isnan(cur[2]) else float(cur[2])
            false = (et is not None and et > c["false_relocalization_m"]) or \
                    (er is not None and er > c["false_relocalization_deg"])
            events.append({"frame_idx": r["frame_idx"], "rel_s": float((r["timestamp_ns"] - t0) * 1e-9),
                           "correction_m": r["loc_correction_m"], "error_before_m": prev_err,
                           "error_after_m": et, "rotation_error_after_deg": er,
                           "since_previous_s": None if last_loc_ts is None
                           else float((r["timestamp_ns"] - last_loc_ts) * 1e-9),
                           "false_relocalization": bool(false), "evaluable": et is not None})
            last_loc_ts = r["timestamp_ns"]
        if cur is not None and r["localized"] and not np.isnan(cur[1]):
            prev_err = float(cur[1])
    n_eval = sum(e["evaluable"] for e in events)
    n_false = sum(e["false_relocalization"] for e in events)
    res["localization_events"] = {
        "n_events": len(events), "n_evaluable": n_eval, "n_false_relocalizations": n_false,
        "false_relocalization_fraction": n_false / n_eval if n_eval else None,
        "n_rejected_hypotheses": int(sum(r["loc_rejected"] for r in log_rows)),
        "correction_m": _stats(np.array([e["correction_m"] for e in events])) if events else None,
        "interval_s": _stats(np.array([e["since_previous_s"] for e in events[1:]])) if len(events) > 1 else None,
        "events": events}

    # ---- coverage (time-weighted over input frames)
    log_by_ts = {r["timestamp_ns"]: r for r in log_rows}
    loc_w, gt_w = 0.0, 0.0
    stale_w = {s: 0.0 for s in c["staleness_s"]}
    thr_w = {th: 0.0 for th in c["error_thresholds_m"]}
    last_ev = None
    for tns in input_ts.tolist():
        r = log_by_ts.get(tns)
        w = own_by_ts[tns]
        if r is None:
            continue
        if r["loc_event"]:
            last_ev = tns
        cur = by_ts.get(tns)
        if not (r["localized"] and cur is not None):
            continue
        loc_w += w
        if not np.isnan(cur[1]):
            gt_w += w
        for s in stale_w:
            if last_ev is not None and (tns - last_ev) * 1e-9 <= s:
                stale_w[s] += w
        for th in thr_w:
            if not np.isnan(cur[1]) and cur[1] <= th:
                thr_w[th] += w
    res["coverage"] = {
        "localized_fraction": loc_w / seq_dur if seq_dur else None,
        "localized_fraction_after_first_localization": (
            loc_w / float(sum(own_by_ts[x] for x in input_ts.tolist() if x >= first["timestamp_ns"]))
            if first is not None else None),
        "fresh_localization_fraction": {f"{s:g}s": v / seq_dur for s, v in stale_w.items()},
        "correctly_localized_fraction": {f"{th:g}m": v / seq_dur for th, v in thr_w.items()},
        "localized_with_gt_fraction": gt_w / seq_dur if seq_dur else None,
        "correctly_localized_fraction_of_localized_with_gt": (
            {f"{th:g}m": v / gt_w for th, v in thr_w.items()} if gt_w else None),
        "note": "fractions of the input duration; 'correctly' = localized and translation error <= threshold; "
                "localized frames without a GT sample within association_max_dt_s cannot be checked"}

    # ---- error distribution of localized frames
    lt = np.array([by_ts[r["timestamp_ns"]][1] for r in rows if r["localized"]])
    lr = np.array([by_ts[r["timestamp_ns"]][2] for r in rows if r["localized"]])
    lt, lr = lt[~np.isnan(lt)], lr[~np.isnan(lr)]
    res["error_fixed_frame"] = ({"n_samples": int(len(lt)), "translation_m": _stats(lt),
                                 "rotation_deg": _stats(lr) if len(lr) else unavailable("no orientation")}
                                if len(lt) else unavailable("no localized pose matched to ground truth"))
    pt = np.array([by_ts[r["timestamp_ns"]][1] for r in rows if not r["localized"]])
    pt = pt[~np.isnan(pt)]
    res["error_prior_only"] = ({"n_samples": int(len(pt)), "translation_m": _stats(pt),
                                "note": "before the first localization: RTAB-Map's prior (last pose saved "
                                        "in the map + odometry); not counted as localized"}
                               if len(pt) else None)

    # ---- computation
    tt = np.array([r["track_time_s"] for r in log_rows])
    st = np.array([r["slam_time_s"] for r in log_rows if r["slam_update"]])
    total = float(tt.sum())
    res["computation"] = {
        "map_load_s": map_info.get("map_load", {}).get("init_time_s"),
        "per_frame_time_s": _stats(tt) if len(tt) else None,
        "map_update_time_s": _stats(st) if len(st) else None,
        "n_map_updates": int(len(st)),
        "total_processing_s": total,
        "real_time_factor": seq_dur / total if total > 0 else None,
        "wall_time_s": run_status.get("wall_time_s"), "peak_rss_mb": run_status.get("peak_rss_mb"),
        "cpu_user_s": run_status.get("cpu_user_s")}

    if out_dir is not None:
        with open(out_dir / "map_reuse_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ns", "rel_s", "localized", "loc_event", "trans_err_m", "rot_err_deg"])
            for k, r in enumerate(rows):
                w.writerow([r["timestamp_ns"], f"{(r['timestamp_ns'] - t0) * 1e-9:.3f}", int(r["localized"]),
                            int(r["loc_event"]), "" if np.isnan(err_t[k]) else f"{err_t[k]:.4f}",
                            "" if np.isnan(err_r[k]) else f"{err_r[k]:.3f}"])
        try:
            res["plot"] = _plot(out_dir / "map_reuse.png", rows, err_t, ts, t0, events, c)
        except Exception as exc:  # noqa: BLE001
            res["plot"] = f"failed: {exc}"
    return res


def _plot(path: Path, rows, err_t, ts, t0, events, c) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rel = (ts - t0) * 1e-9
    loc = np.array([r["localized"] for r in rows], dtype=bool)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rel[~loc], err_t[~loc], ".", ms=2, color="0.6", label="prior only (not localized)")
    ax.plot(rel[loc], err_t[loc], ".", ms=2, color="C0", label="localized")
    for e in events:
        ax.axvline(e["rel_s"], color="C3" if e["false_relocalization"] else "C2", lw=0.6, alpha=0.6)
    ax.axhline(c["false_relocalization_m"], color="C3", ls="--", lw=0.8, label="false-relocalization threshold")
    ax.set_yscale("log")
    ax.set_xlabel("time since first frame (s)")
    ax.set_ylabel("translation error in map frame (m)")
    ax.set_title("map reuse: error in the saved map's fixed frame (green/red lines = accepted localizations)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
