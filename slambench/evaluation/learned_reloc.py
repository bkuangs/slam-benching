"""Per-attempt diagnostics of SP_SLAM3's learned relocalization (patch 0007, ``learned_reloc.csv``).

Every relocalization attempt of a session-B run is classified with ground truth:

* **retrieval**: a top-K keyframe is *positive* when its GT pose (session A's GT at the keyframe's
  timestamp) is within ``positive_m`` metres and ``positive_deg`` degrees of the query's GT pose
  (the place-recognition benchmark's positives: 2 m / 30 deg sim, 0.5 m / 30 deg real);
* **verification**: whether the best verified candidate passed the inlier gate (single-frame
  accept), and the error of its pose in the fixed map frame (mapping run's map->GT transform);
  an accept is *false* beyond ``false_m`` / ``false_deg`` (the map-reuse false-relocalization
  threshold);
* **outcome** of an attempt: ``accepted`` / ``retrieval_miss`` (no positive in the top-K) /
  ``verification_fail`` (a positive was retrieved but nothing passed the gate).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .metrics import associate


def read_log(path: Path) -> List[Dict[str, Any]]:
    """learned_reloc.csv -> rows with parsed lists / numbers."""
    def ints(s):
        return [int(x) for x in s.split(";")] if s else []

    def floats(s):
        return [float(x) for x in s.split(";")] if s else []
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {"timestamp_ns": int(round(float(r["timestamp"]) * 1e9)), "frame_id": int(r["frame_id"]),
                   "mode": r["mode"], "n_db": int(r["n_db"]),
                   "desc_ms": float(r["desc_ms"]), "retrieval_ms": float(r["retrieval_ms"]),
                   "verify_ms": float(r["verify_ms"]), "total_ms": float(r["total_ms"]),
                   "cand_kf_ids": ints(r["cand_kf_ids"]), "cand_sims": floats(r["cand_sims"]),
                   "cand_lg_matches": ints(r["cand_lg_matches"]), "cand_mp_matches": ints(r["cand_mp_matches"]),
                   "cand_pnp_inliers": ints(r["cand_pnp_inliers"]),
                   "cand_final_inliers": ints(r["cand_final_inliers"]),
                   "best_rank": int(r["best_rank"]), "best_kf_id": int(r["best_kf_id"]),
                   "best_inliers": int(r["best_inliers"]), "accepted": r["accepted"] == "1",
                   "consistent_count": int(r["consistent_count"]), "confirmed": r["confirmed"] == "1",
                   "pose": None}
            if r.get("tx") not in (None, ""):
                row["pose"] = np.array([float(r[k]) for k in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")])
            out.append(row)
    return out


def _T(pose7) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose7[3:]).as_matrix()
    T[:3, 3] = pose7[:3]
    return T


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _gt_lookup(ts_ns: Sequence[int], gt_ts: np.ndarray, gt_T: np.ndarray, max_dt_s: float) -> List[Optional[np.ndarray]]:
    ei, gi, _ = associate(np.asarray(ts_ns, np.int64), gt_ts, max_dt_s)
    out: List[Optional[np.ndarray]] = [None] * len(ts_ns)
    for e, g in zip(ei, gi):
        out[e] = gt_T[g]
    return out


def diagnose(rows: List[Dict[str, Any]], kf_ts: Dict[int, int], gt_a: tuple, gt_b: tuple,
             R_map: np.ndarray, t_map: np.ndarray, T_final_map: np.ndarray, positive_m: float,
             positive_deg: float = 30.0, false_m: float = 0.5, false_deg: float = 15.0,
             max_dt_s: float = 0.02, min_inliers: int = 50) -> Dict[str, Any]:
    """Classify every attempt. gt_a / gt_b: (timestamps_ns [N], camera poses T_gt_cam [N,4,4]) of the
    mapping / query session; R_map, t_map: the mapping run's map->GT alignment; T_final_map: the
    driver's active-map -> saved-final-frame transform (loaded_map.json)."""
    q_gt = _gt_lookup([r["timestamp_ns"] for r in rows], gt_b[0], gt_b[1], max_dt_s)
    kf_ids = sorted(kf_ts)
    kf_gt = dict(zip(kf_ids, _gt_lookup([kf_ts[k] for k in kf_ids], gt_a[0], gt_a[1], max_dt_s)))
    T_gt_map = np.eye(4)
    T_gt_map[:3, :3], T_gt_map[:3, 3] = R_map, t_map
    per = []
    for r, Tq in zip(rows, q_gt):
        d: Dict[str, Any] = {"frame_id": r["frame_id"], "timestamp_ns": r["timestamp_ns"], "mode": r["mode"],
                             "accepted": r["accepted"], "confirmed": r["confirmed"], "has_gt": Tq is not None}
        pos = []
        for k in r["cand_kf_ids"]:
            Tk = kf_gt.get(k)
            if Tq is None or Tk is None:
                pos.append(None)
                continue
            pos.append(bool(np.linalg.norm(Tk[:3, 3] - Tq[:3, 3]) <= positive_m
                            and _angle_deg(Tk[:3, :3], Tq[:3, :3]) <= positive_deg))
        d["cand_positive"] = pos
        d["top1_positive"] = pos[0] if pos else None
        d["any_positive"] = None if Tq is None else any(p for p in pos if p)
        d["best_positive"] = pos[r["best_rank"]] if r["best_rank"] >= 0 and r["best_rank"] < len(pos) else None
        d["err_m"] = d["err_deg"] = None
        if r["pose"] is not None and Tq is not None:
            T = T_gt_map @ T_final_map @ _T(r["pose"])
            d["err_m"] = float(np.linalg.norm(T[:3, 3] - Tq[:3, 3]))
            d["err_deg"] = _angle_deg(T[:3, :3], Tq[:3, :3])
        d["false_accept"] = bool(r["accepted"] and d["err_m"] is not None
                                 and (d["err_m"] > false_m or d["err_deg"] > false_deg))
        if r["accepted"]:
            d["outcome"] = "accepted"
        elif d["any_positive"] is None:
            d["outcome"] = "no_gt"
        elif not d["any_positive"]:
            d["outcome"] = "retrieval_miss"
        else:
            d["outcome"] = "verification_fail"
        per.append(d)
    return {"attempts": per, "summary": summarize(rows, per, min_inliers)}


def summarize(rows: List[Dict[str, Any]], per: List[Dict[str, Any]], min_inliers: int = 50) -> Dict[str, Any]:
    n = len(rows)
    s: Dict[str, Any] = {"n_attempts": n}
    if not n:
        return s
    with_gt = [d for d in per if d["has_gt"]]
    s["n_with_gt"] = len(with_gt)
    for k in ("accepted", "retrieval_miss", "verification_fail", "no_gt"):
        s[f"n_{k}"] = sum(d["outcome"] == k for d in per)
    if with_gt:
        s["recall_at_1"] = float(np.mean([bool(d["top1_positive"]) for d in with_gt]))
        s["recall_at_k"] = float(np.mean([bool(d["any_positive"]) for d in with_gt]))
    acc = [d for d in per if d["accepted"]]
    s["n_accepted_single_frame"] = len(acc)
    s["n_false_single_frame_accepts"] = sum(d["false_accept"] for d in acc)
    s["n_confirmed"] = sum(d["confirmed"] for d in per)
    s["n_false_confirmed"] = sum(d["false_accept"] for d in per if d["confirmed"])
    errs = [d["err_m"] for d in acc if d["err_m"] is not None]
    if errs:
        s["accepted_err_m_median"] = float(np.median(errs))
    # verification on attempts whose top-K contained a positive
    pos_rows = [(r, d) for r, d in zip(rows, per) if d["any_positive"]]
    if pos_rows:
        s["accept_rate_given_positive_retrieved"] = float(np.mean([r["accepted"] for r, _ in pos_rows]))
        best_mp = [max([m for m, p in zip(r["cand_mp_matches"], d["cand_positive"]) if p] or [0])
                   for r, d in pos_rows]
        s["positive_candidate_mp_matches_median"] = float(np.median(best_mp))
    # every GT-positive candidate: matches at each verification stage and the inlier gate
    stages = {"lg": "cand_lg_matches", "mp": "cand_mp_matches", "pnp": "cand_pnp_inliers",
              "final": "cand_final_inliers"}
    cand = {k: [] for k in stages}
    for r, d in zip(rows, per):
        for i, p in enumerate(d["cand_positive"]):
            if p:
                for k, col in stages.items():
                    cand[k].append(r[col][i] if i < len(r[col]) else 0)
    if cand["final"]:
        fin = np.array(cand["final"])
        s["positive_candidates"] = {
            "n": int(fin.size), "min_inliers": int(min_inliers),
            "n_pass_gate": int((fin >= min_inliers).sum()),
            "pass_rate": float((fin >= min_inliers).mean()),
            **{f"{k}_median": float(np.median(v)) for k, v in cand.items()},
            **{f"{k}_p90": float(np.percentile(v, 90)) for k, v in cand.items()}}
    for k in ("desc_ms", "retrieval_ms", "verify_ms", "total_ms"):
        v = np.array([r[k] for r in rows])
        s[k] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
    return s
