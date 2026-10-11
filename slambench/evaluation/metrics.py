"""Trajectory accuracy, tracking coverage and timing metrics.

Conventions (also in docs/evaluation.md):

* Association: each *valid* estimate sample is matched to the nearest
  ground-truth sample in time; the match is kept only if
  |t_est - t_gt| <= association_max_dt_s. No interpolation of either
  trajectory. Each estimate sample is used at most once.
* Valid estimate samples: if estimator tracking states are available, only
  poses from frames whose state is "tracking OK" are evaluated; poses the
  estimator wrote while recently-lost/extrapolating are counted and excluded.
* Frames: ground truth (pose of the GT body) is converted to the estimate's
  sensor frame with the dataset's T_gt_body_sensor before association.
* Alignment: "se3" (rigid, default for metric stereo/inertial runs), "sim3"
  (explicit opt-in, labelled; intended for scale-ambiguous monocular), or
  "none". Umeyama alignment over all matched samples (evo).
* ATE: translation error after alignment, metres. Rotation ATE (deg) only
  when both trajectories carry orientation.
* RPE: computed only between matched samples inside the same contiguous
  segment. A segment breaks where consecutive matched samples are more than
  segment_max_gap_s apart, where the estimator was not tracking OK in between,
  or where the estimator's map id changes. Pairs never span a tracking gap.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..trajectory import Trajectory

DEFAULT_EVAL: Dict[str, Any] = {
    "enabled": True,
    "alignment": "se3",                 # se3 | sim3 | none
    "association_max_dt_s": 0.02,
    "segment_max_gap_s": 0.5,
    "rpe": [{"delta": 1.0, "unit": "m"}, {"delta": 1.0, "unit": "s"}],
    "valid_states": "tracking_ok",      # tracking_ok | all_output
    "session_min_matched": 20,          # map sessions with fewer matched samples are not pooled
    "plots": True,
}

ACCURACY_LABEL = ("aligned benchmark accuracy: the alignment is fitted over each whole coordinate "
                  "session using all of its ground truth, including samples after each pose was "
                  "emitted; it measures trajectory shape/consistency, not a causal real-time error")


def unavailable(reason: str) -> Dict[str, Any]:
    return {"available": False, "reason": reason}


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "rmse": float(np.sqrt(np.mean(x ** 2))), "mean": float(np.mean(x)),
            "median": float(np.median(x)), "std": float(np.std(x)), "min": float(np.min(x)),
            "max": float(np.max(x))}


# ----------------------------------------------------------------------------
# frame diagnostics
# ----------------------------------------------------------------------------
@dataclass
class FrameDiagnostics:
    timestamps_ns: np.ndarray
    state: np.ndarray            # str
    tracking_ok: np.ndarray      # bool
    map_id: np.ndarray           # int, -1 if unknown
    track_time_s: np.ndarray     # float, nan if unknown
    in_final: np.ndarray         # bool

    @classmethod
    def read(cls, path: Path) -> "FrameDiagnostics":
        with open(path) as f:
            rows = list(csv.DictReader(f))
        g = lambda k, conv, default: np.array([conv(r[k]) if r.get(k, "") != "" else default for r in rows])
        return cls(g("timestamp_ns", int, 0).astype(np.int64), g("tracking_state", str, ""),
                   g("tracking_ok", int, 0).astype(bool), g("map_id", int, -1).astype(int),
                   g("track_time_s", float, np.nan).astype(float), g("in_final_trajectory", int, 0).astype(bool))


def coverage_from_states(diag: FrameDiagnostics, input_ts: np.ndarray) -> Dict[str, Any]:
    """Tracking coverage, loss events and recoveries from estimator tracking states.

    Time attribution: frame i "owns" the interval [t_i, t_{i+1}) of the input
    frame sequence; the last frame owns the median frame period.
    """
    input_ts = np.asarray(input_ts, dtype=np.int64)
    n_in = len(input_ts)
    ok_by_ts = dict(zip(diag.timestamps_ns.tolist(), diag.tracking_ok.tolist()))
    map_by_ts = dict(zip(diag.timestamps_ns.tolist(), diag.map_id.tolist()))
    ok = np.array([bool(ok_by_ts.get(t, False)) for t in input_ts.tolist()])
    processed = np.array([t in ok_by_ts for t in input_ts.tolist()])
    dt = np.diff(input_ts) * 1e-9
    med = float(np.median(dt)) if len(dt) else 0.0
    own = np.r_[dt, med]
    seq_dur = float(own.sum())

    first_ok = int(np.argmax(ok)) if ok.any() else None
    events: List[Dict[str, Any]] = []
    if first_ok is not None:
        i = first_ok
        while i < n_in:
            if not ok[i]:
                start = i
                j = i
                while j < n_in and not ok[j]:
                    j += 1
                ev = {"lost_at_s": float(input_ts[start] * 1e-9),
                      "lost_at_rel_s": float((input_ts[start] - input_ts[0]) * 1e-9),
                      "n_frames": int(j - start), "recovered": j < n_in}
                if j < n_in:
                    ev["recovery_duration_s"] = float((input_ts[j] - input_ts[start]) * 1e-9)
                    m0, m1 = map_by_ts.get(int(input_ts[start - 1]), -1), map_by_ts.get(int(input_ts[j]), -1)
                    ev["recovered_in_new_map"] = bool(m0 != m1) if (m0 >= 0 and m1 >= 0) else None
                else:
                    ev["recovery_duration_s"] = None
                    ev["unrecovered_duration_s"] = float(own[start:].sum())
                states = diag.state[np.isin(diag.timestamps_ns, input_ts[start:j])]
                ev["states"] = sorted(set(states.tolist()))
                events.append(ev)
                i = j
            else:
                i += 1
    rec = [e["recovery_duration_s"] for e in events if e["recovered"]]
    maps = sorted(set(int(m) for m in diag.map_id.tolist() if m >= 0))
    return {
        "source": "estimator_tracking_state",
        "n_frames_input": int(n_in),
        "n_frames_processed": int(processed.sum()),
        "n_frames_tracking_ok": int(ok.sum()),
        "frame_coverage": float(ok.sum() / n_in) if n_in else None,
        "sequence_duration_s": seq_dur,
        "tracked_duration_s": float(own[ok].sum()),
        "time_coverage": float(own[ok].sum() / seq_dur) if seq_dur > 0 else None,
        "time_to_first_ok_s": float((input_ts[first_ok] - input_ts[0]) * 1e-9) if first_ok is not None else None,
        "n_tracking_loss_events": len(events),
        "n_recovered": len(rec),
        "n_unrecovered": len(events) - len(rec),
        "n_recovered_in_new_map": sum(1 for e in events if e.get("recovered_in_new_map")),
        "recovery_duration_s": _stats(np.array(rec)) if rec else {"n": 0},
        "total_lost_duration_s": float(own[first_ok:][~ok[first_ok:]].sum()) if first_ok is not None else seq_dur,
        "map_ids": maps,
        "n_maps": len(maps),
        "n_frames_in_final_trajectory": int(diag.in_final.sum()),
        "final_trajectory_frame_coverage": float(diag.in_final.sum() / n_in) if n_in else None,
        "loss_events": events,
    }


def coverage_proxy_from_output(traj: Trajectory, input_ts: np.ndarray, keyframe_only: bool) -> Dict[str, Any]:
    """Output-availability proxy, used only when no tracking states exist."""
    n_in = len(input_ts)
    return {
        "source": "PROXY_output_availability",
        "warning": "estimator tracking states unavailable; this is the fraction of input frames with "
                   "an output pose, not verified tracking" + (" (keyframe-only output: strongly "
                                                              "underestimates tracked frames)" if keyframe_only else ""),
        "n_frames_input": int(n_in),
        "n_output_poses": int(len(traj)),
        "output_frame_fraction_proxy": float(len(traj) / n_in) if n_in else None,
        "n_tracking_loss_events": unavailable("no tracking states"),
        "recovery_duration_s": unavailable("no tracking states"),
    }


# ----------------------------------------------------------------------------
# association / alignment / errors
# ----------------------------------------------------------------------------
def associate(est_ts: np.ndarray, gt_ts: np.ndarray, max_dt_s: float):
    """Nearest-neighbour association within tolerance. Returns (est_idx, gt_idx, dt_s)."""
    est_ts, gt_ts = np.asarray(est_ts, np.int64), np.asarray(gt_ts, np.int64)
    if len(est_ts) == 0 or len(gt_ts) == 0:
        return np.array([], int), np.array([], int), np.array([])
    j = np.clip(np.searchsorted(gt_ts, est_ts), 1, len(gt_ts) - 1) if len(gt_ts) > 1 else np.zeros(len(est_ts), int)
    if len(gt_ts) > 1:
        cand = np.stack([j - 1, j])
        d = np.abs(gt_ts[cand] - est_ts[None])
        j = cand[np.argmin(d, axis=0), np.arange(len(est_ts))]
    dt = (est_ts - gt_ts[j]) * 1e-9
    keep = np.abs(dt) <= max_dt_s
    return np.nonzero(keep)[0], j[keep], dt[keep]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Least-squares R, t, s with dst ~= s * R @ src + t (Umeyama 1991)."""
    from evo.core.geometry import umeyama_alignment
    R, t, s = umeyama_alignment(src.T, dst.T, with_scale)
    return R, t, float(s)


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    Rrel = np.einsum("nji,njk->nik", Ra, Rb)
    return np.degrees(np.linalg.norm(Rotation.from_matrix(Rrel).as_rotvec(), axis=1))


def segments_from(ts_ns: np.ndarray, max_gap_s: float, diag: Optional[FrameDiagnostics]) -> np.ndarray:
    """Segment id per matched sample; breaks at time gaps, non-OK frames in between, map changes."""
    seg = np.zeros(len(ts_ns), dtype=int)
    if len(ts_ns) == 0:
        return seg
    bad_ts = np.array([], np.int64)
    map_by_ts = {}
    if diag is not None:
        bad_ts = np.sort(diag.timestamps_ns[~diag.tracking_ok])
        map_by_ts = dict(zip(diag.timestamps_ns.tolist(), diag.map_id.tolist()))
    for k in range(1, len(ts_ns)):
        a, b = ts_ns[k - 1], ts_ns[k]
        brk = (b - a) * 1e-9 > max_gap_s
        if not brk and len(bad_ts):
            lo = np.searchsorted(bad_ts, a, side="right")
            hi = np.searchsorted(bad_ts, b, side="left")
            brk = hi > lo
        if not brk and map_by_ts:
            brk = map_by_ts.get(int(a), -1) != map_by_ts.get(int(b), -1)
        seg[k] = seg[k - 1] + int(brk)
    return seg


def rpe(est_P: np.ndarray, gt_P: np.ndarray, est_pos: np.ndarray, gt_pos: np.ndarray, ts_ns: np.ndarray,
        seg: np.ndarray, delta: float, unit: str, has_rot: bool) -> Dict[str, Any]:
    """Relative pose error for pairs (i, j) inside one segment separated by `delta` `unit`."""
    if unit not in ("m", "s", "frames"):
        raise ValueError(f"RPE unit must be m, s or frames, not {unit!r}")
    pairs = []
    for s in np.unique(seg):
        idx = np.nonzero(seg == s)[0]
        if len(idx) < 2:
            continue
        if unit == "m":
            step = np.linalg.norm(np.diff(gt_pos[idx], axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(step)]
            for a in range(len(idx)):
                b = np.searchsorted(cum, cum[a] + delta, side="left")
                if b < len(idx):
                    pairs.append((idx[a], idx[b]))
        elif unit == "s":
            t = ts_ns[idx]
            for a in range(len(idx)):
                b = np.searchsorted(t, t[a] + int(round(delta * 1e9)), side="left")
                if b < len(idx):
                    pairs.append((idx[a], idx[b]))
        else:
            d = int(delta)
            pairs += [(idx[a], idx[a + d]) for a in range(len(idx) - d)]
    out: Dict[str, Any] = {"delta": delta, "delta_unit": unit, "n_pairs": len(pairs),
                           "pairs_within_valid_segments_only": True}
    if not pairs:
        out["translation_m"] = unavailable("no sample pairs separated by delta inside a valid segment")
        return out
    i, j = np.array(pairs).T
    if has_rot:
        Eg = np.linalg.inv(gt_P[i]) @ gt_P[j]
        Ee = np.linalg.inv(est_P[i]) @ est_P[j]
        E = np.linalg.inv(Eg) @ Ee
        te = np.linalg.norm(E[:, :3, 3], axis=1)
        re = np.degrees(np.linalg.norm(Rotation.from_matrix(E[:, :3, :3]).as_rotvec(), axis=1))
        out["translation_m"] = _stats(te)
        out["rotation_deg"] = _stats(re)
        out["_trans_err"], out["_rot_err"] = te, re
        out["frame"] = "relative motion expressed in the frame of sample i"
    else:
        e = (est_pos[j] - est_pos[i]) - (gt_pos[j] - gt_pos[i])
        out["_trans_err"] = np.linalg.norm(e, axis=1)
        out["translation_m"] = _stats(out["_trans_err"])
        out["rotation_deg"] = unavailable("ground truth or estimate has no orientation")
        out["frame"] = "displacement difference in the aligned world frame (position-only variant)"
    return out


def evaluate_trajectory(est: Trajectory, gt: Trajectory, cfg: Dict[str, Any],
                        diag: Optional[FrameDiagnostics] = None,
                        input_ts: Optional[np.ndarray] = None,
                        scale_ambiguous: bool = False) -> Dict[str, Any]:
    """Accuracy metrics for an estimate already expressed in the same sensor frame as `gt`."""
    c = dict(DEFAULT_EVAL)
    c.update(cfg or {})
    res: Dict[str, Any] = {"settings": {k: c[k] for k in ("alignment", "association_max_dt_s",
                                                          "segment_max_gap_s", "rpe", "valid_states")}}
    est = est.sorted_unique()
    gt = gt.sorted_unique()

    # 1. valid estimate samples
    n_out = len(est)
    if diag is not None and c["valid_states"] == "tracking_ok":
        ok_ts = set(diag.timestamps_ns[diag.tracking_ok].tolist())
        valid = np.array([t in ok_ts for t in est.timestamps_ns.tolist()], dtype=bool)
        res["valid_sample_rule"] = "estimator tracking state OK"
    else:
        valid = np.ones(n_out, dtype=bool)
        res["valid_sample_rule"] = "all output poses (no tracking states used)"
    res["n_output_poses"] = n_out
    res["n_output_poses_excluded_not_tracking_ok"] = int((~valid).sum())
    est_v = est.subset(np.nonzero(valid)[0])

    # 2. association
    ei, gi, dt = associate(est_v.timestamps_ns, gt.timestamps_ns, c["association_max_dt_s"])
    res["association"] = {"method": "nearest_neighbour_no_interpolation",
                          "max_dt_s": c["association_max_dt_s"], "n_valid_estimates": len(est_v),
                          "n_matched": int(len(ei)), "n_unmatched": int(len(est_v) - len(ei)),
                          "abs_dt_s": _stats(np.abs(dt))}
    if len(ei) < 3:
        res["ate"] = unavailable(f"only {len(ei)} matched samples")
        res["rpe"] = [unavailable("insufficient matched samples")]
        return res
    E = est_v.subset(ei)
    G = gt.subset(gi)
    has_rot = E.has_orientation and G.has_orientation

    # 3. alignment
    align = c["alignment"]
    if align not in ("se3", "sim3", "none"):
        raise ValueError(f"alignment must be se3, sim3 or none, not {align!r}")
    if align == "sim3":
        R, t, s = umeyama(E.positions, G.positions, True)
        res["alignment"] = {"type": "sim3", "label": "Sim(3) scale-corrected (scale-ambiguous estimate)",
                            "scale": s}
    elif align == "se3":
        R, t, s = umeyama(E.positions, G.positions, False)
        res["alignment"] = {"type": "se3", "label": "rigid SE(3)", "scale": 1.0}
    else:
        R, t, s = np.eye(3), np.zeros(3), 1.0
        res["alignment"] = {"type": "none", "label": "no alignment", "scale": 1.0}
    if scale_ambiguous and align != "sim3":
        res["alignment"]["warning"] = "scale-ambiguous estimate evaluated without scale correction"
    res["alignment"]["R"] = R.tolist()
    res["alignment"]["t"] = np.asarray(t).tolist()
    est_pos = s * E.positions @ R.T + t
    est_P = None
    if has_rot:
        est_P = E.poses()
        est_P[:, :3, :3] = R[None] @ est_P[:, :3, :3]
        est_P[:, :3, 3] = est_pos
    gt_P = G.poses() if has_rot else None

    # 4. ATE
    trans_err = np.linalg.norm(est_pos - G.positions, axis=1)
    ate = {"translation_m": _stats(trans_err), "unit": "m"}
    rot_err = _rot_angle_deg(gt_P[:, :3, :3], est_P[:, :3, :3]) if has_rot else None
    ate["rotation_deg"] = (_stats(rot_err) if has_rot
                           else unavailable("ground truth or estimate has no orientation"))
    res["ate"] = ate
    if has_rot:
        # Diagnostic only (never applied): a constant rotation between the GT-derived sensor frame
        # and the estimated sensor frame points at a GT-body -> sensor extrinsic error.
        Rb = Rotation.from_matrix(np.einsum("nji,njk->nik", gt_P[:, :3, :3], est_P[:, :3, :3]))
        mb = Rb.mean()
        resid = (mb.inv() * Rb).magnitude()
        res["diagnostics"] = {"constant_body_rotation_offset": {
            "angle_deg": float(np.degrees(mb.magnitude())),
            "rotvec_deg_in_estimate_frame": np.degrees(mb.as_rotvec()).tolist(),
            "rotation_rmse_deg_if_removed": float(np.degrees(np.sqrt(np.mean(resid ** 2)))),
            "note": "not applied to any metric; a large constant offset with small residual indicates "
                    "an error in the GT-body-to-sensor extrinsic rather than estimator error, and it "
                    "also inflates translational RPE by ~delta*sin(angle)"}}

    # 5. evaluated coverage
    seg = segments_from(E.timestamps_ns, c["segment_max_gap_s"], diag)
    ev_dur = 0.0
    for sid in np.unique(seg):
        tt = E.timestamps_ns[seg == sid]
        ev_dur += (tt[-1] - tt[0]) * 1e-9
    evc = {"n_matched_samples": int(len(ei)), "n_segments": int(seg.max() + 1),
           "evaluated_duration_s": float(ev_dur),
           "evaluated_span_s": float((E.timestamps_ns[-1] - E.timestamps_ns[0]) * 1e-9)}
    if input_ts is not None and len(input_ts) > 1:
        in_dur = (input_ts[-1] - input_ts[0]) * 1e-9
        evc["matched_frame_fraction"] = float(len(ei) / len(input_ts))
        evc["evaluated_duration_fraction"] = float(ev_dur / in_dur) if in_dur > 0 else None
        gi_all = associate(np.asarray(input_ts), gt.timestamps_ns, c["association_max_dt_s"])[0]
        evc["input_frames_with_gt_fraction"] = float(len(gi_all) / len(input_ts))
    res["evaluated_coverage"] = evc

    # 6. RPE
    res["rpe"] = [rpe(est_P, gt_P, est_pos, G.positions, E.timestamps_ns, seg, float(r["delta"]),
                      r["unit"], has_rot) for r in c["rpe"]]
    res["_series"] = {"t_ns": E.timestamps_ns, "est_pos": est_pos, "gt_pos": G.positions,
                      "trans_err": trans_err, "rot_err": rot_err, "segment": seg}
    return res


# ----------------------------------------------------------------------------
# coordinate sessions (independent map frames)
# ----------------------------------------------------------------------------
def split_sessions(traj: Trajectory, coordinate_sessions: str, diag: Optional[FrameDiagnostics]):
    """[(map_id or None, sample indices)] in order of first appearance, plus a note.

    ``map_id``: every pose belongs to the session of the estimator map active when it was emitted;
    each map has its own world frame, so sessions are aligned and scored independently and RPE
    never pairs samples from different maps.
    """
    if coordinate_sessions == "single":
        return [(None, np.arange(len(traj)))], None
    if diag is None or np.all(diag.map_id < 0):
        return ([(None, np.arange(len(traj)))],
                "map ids unavailable: poses assumed to share one coordinate frame")
    map_by_ts = dict(zip(diag.timestamps_ns.tolist(), diag.map_id.tolist()))
    ids = np.array([map_by_ts.get(int(t), -1) for t in traj.timestamps_ns.tolist()])
    out = []
    for m in dict.fromkeys(ids.tolist()):
        out.append((int(m), np.nonzero(ids == m)[0]))
    note = None
    if any(m == -1 for m, _ in out):
        note = "some poses have no map id in the frame diagnostics; they form session -1"
    return out, note


def _concat(parts, key):
    arrs = [p[key] for p in parts if p.get(key) is not None]
    return np.concatenate(arrs) if arrs else None


def evaluate_sessions(traj: Trajectory, gt: Trajectory, cfg: Dict[str, Any], coordinate_sessions: str,
                      diag: Optional[FrameDiagnostics] = None, input_ts: Optional[np.ndarray] = None,
                      scale_ambiguous: bool = False) -> Dict[str, Any]:
    """evaluate_trajectory() per coordinate session, plus pooled statistics over eligible sessions.

    Pooled ATE/RPE combine the per-sample/per-pair errors of sessions that were each aligned on
    their own; nothing is ever aligned across sessions. Sessions with fewer than
    session_min_matched matched samples are reported but not pooled (alignment would overfit).
    """
    c = dict(DEFAULT_EVAL)
    c.update(cfg or {})
    traj = traj.sorted_unique()
    split, note = split_sessions(traj, coordinate_sessions, diag)
    in_ts = np.asarray(input_ts) if input_ts is not None else None
    in_dur = float((in_ts[-1] - in_ts[0]) * 1e-9) if in_ts is not None and len(in_ts) > 1 else None
    sessions, pooled_parts = [], []
    for map_id, idx in split:
        sub = traj.subset(idx)
        r = evaluate_trajectory(sub, gt, c, diag, input_ts, scale_ambiguous)
        span = float((sub.timestamps_ns[-1] - sub.timestamps_ns[0]) * 1e-9) if len(sub) else 0.0
        info = {"map_id": map_id, "n_poses": int(len(sub)),
                "first_rel_s": float((sub.timestamps_ns[0] - in_ts[0]) * 1e-9) if in_ts is not None else None,
                "span_s": span, "span_fraction_of_input": span / in_dur if in_dur else None}
        n_m = r["association"]["n_matched"]
        ok = isinstance(r.get("ate"), dict) and r["ate"].get("available", True) and n_m >= c["session_min_matched"]
        r["session"] = info
        r["included_in_pooled"] = bool(ok)
        if not ok:
            r["excluded_reason"] = (f"{n_m} matched samples < session_min_matched "
                                    f"({c['session_min_matched']})")
        else:
            pooled_parts.append(r)
        sessions.append(r)

    res: Dict[str, Any] = {
        "label": ACCURACY_LABEL, "coordinate_sessions": coordinate_sessions,
        "settings": {k: c[k] for k in ("alignment", "association_max_dt_s", "segment_max_gap_s", "rpe",
                                       "valid_states", "session_min_matched")},
        "n_sessions": len(sessions), "n_sessions_pooled": len(pooled_parts),
        "n_poses_in_unpooled_sessions": int(sum(r["session"]["n_poses"] for r in sessions
                                                if not r["included_in_pooled"])),
        "sessions": sessions}
    if note:
        res["session_note"] = note
    res["association"] = {
        "method": "nearest_neighbour_no_interpolation", "max_dt_s": c["association_max_dt_s"],
        "n_valid_estimates": int(sum(r["association"]["n_valid_estimates"] for r in sessions)),
        "n_matched": int(sum(r["association"]["n_matched"] for r in pooled_parts))}
    if not pooled_parts:
        res["ate"] = unavailable("no coordinate session with enough matched samples")
        res["rpe"] = [unavailable("no coordinate session with enough matched samples")]
        return res

    series = [r["_series"] for r in pooled_parts]
    seg_off, segs = 0, []
    for sv in series:
        segs.append(sv["segment"] + seg_off)
        seg_off += int(sv["segment"].max()) + 1
    sess_idx = np.concatenate([np.full(len(sv["t_ns"]), k) for k, sv in enumerate(series)])
    res["_series"] = {"t_ns": _concat(series, "t_ns"), "est_pos": _concat(series, "est_pos"),
                      "gt_pos": _concat(series, "gt_pos"), "trans_err": _concat(series, "trans_err"),
                      "rot_err": _concat(series, "rot_err"), "segment": np.concatenate(segs),
                      "session": sess_idx}
    res["alignment"] = {"type": c["alignment"], "label": pooled_parts[0]["alignment"]["label"]
                        + (" (fitted independently per map session)" if len(split) > 1 else ""),
                        "per_session": [{"map_id": r["session"]["map_id"], "scale": r["alignment"]["scale"]}
                                        for r in pooled_parts]}
    if len(pooled_parts) == 1:
        res["alignment"]["scale"] = pooled_parts[0]["alignment"]["scale"]
    rot = res["_series"]["rot_err"]
    res["ate"] = {"translation_m": _stats(res["_series"]["trans_err"]), "unit": "m",
                  "rotation_deg": _stats(rot) if rot is not None and len(rot) == len(res["_series"]["t_ns"])
                  else unavailable("ground truth or estimate has no orientation")}
    largest = max(pooled_parts, key=lambda r: r["association"]["n_matched"])
    if "diagnostics" in largest:
        res["diagnostics"] = dict(largest["diagnostics"])
        res["diagnostics"]["from_session_map_id"] = largest["session"]["map_id"]
    ev = [r["evaluated_coverage"] for r in pooled_parts]
    evc = {"n_matched_samples": int(sum(e["n_matched_samples"] for e in ev)),
           "n_segments": int(sum(e["n_segments"] for e in ev)),
           "evaluated_duration_s": float(sum(e["evaluated_duration_s"] for e in ev))}
    if in_ts is not None and len(in_ts) > 1:
        evc["matched_frame_fraction"] = evc["n_matched_samples"] / len(in_ts)
        evc["evaluated_duration_fraction"] = evc["evaluated_duration_s"] / in_dur if in_dur else None
        evc["input_frames_with_gt_fraction"] = ev[0].get("input_frames_with_gt_fraction")
    res["evaluated_coverage"] = evc
    rpes = []
    for k, spec in enumerate(c["rpe"]):
        parts = [r["rpe"][k] for r in pooled_parts if k < len(r["rpe"])]
        out = {"delta": float(spec["delta"]), "delta_unit": spec["unit"],
               "n_pairs": int(sum(p.get("n_pairs", 0) for p in parts)),
               "pairs_within_valid_segments_only": True, "pairs_within_one_map_session_only": True}
        te = _concat(parts, "_trans_err")
        re = _concat(parts, "_rot_err")
        out["translation_m"] = _stats(te) if te is not None and len(te) else unavailable(
            "no sample pairs separated by delta inside a valid segment")
        out["rotation_deg"] = (_stats(re) if re is not None and len(re)
                               else unavailable("no rotation errors (no pairs or no orientation)"))
        rpes.append(out)
    res["rpe"] = rpes
    return res


# ----------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------
def timing_metrics(diag: Optional[FrameDiagnostics], input_ts: np.ndarray, run_status: Dict[str, Any],
                   playback_rate: Optional[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "wall_time_s": run_status.get("wall_time_s"),
        "peak_rss_mb": run_status.get("peak_rss_mb"),
        "peak_rss_method": run_status.get("peak_rss_method"),
        "cpu_user_s": run_status.get("cpu_user_s"),
        "cpu_system_s": run_status.get("cpu_system_s"),
        "playback_rate": playback_rate,
        "playback": ("throttled to recorded rate x %.3g" % playback_rate) if playback_rate
        else "unthrottled (as fast as possible)",
    }
    dts = np.diff(np.asarray(input_ts)) * 1e-9
    period = float(np.median(dts)) if len(dts) else None
    out["input_frame_period_s"] = period
    if diag is None or np.all(np.isnan(diag.track_time_s)):
        out["per_frame_time_s"] = unavailable("estimator does not expose per-frame processing time")
        return out
    tt = diag.track_time_s[~np.isnan(diag.track_time_s)]
    out["per_frame_time_s"] = {**_stats(tt), "p95": float(np.percentile(tt, 95)),
                               "what": "wall time of the estimator's per-frame tracking call (tracking "
                                       "thread only; mapping/loop-closing run concurrently)"}
    out["tracking_throughput_fps"] = float(len(tt) / tt.sum()) if tt.sum() > 0 else None
    if period:
        over = tt > period
        out["realtime"] = {
            "frame_period_s": period,
            "fraction_frames_over_period": float(over.mean()),
            "tracking_p95_within_period": bool(np.percentile(tt, 95) <= period),
            "note": "real-time feasibility judged from per-frame tracking time vs. recorded frame period "
                    "on this machine; offline wall time is not a real-time measure"
                    + (" (input was throttled, so wall time ~ sequence duration)" if playback_rate else ""),
        }
    return out


def strip_private(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [strip_private(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


def write_json(path: Path, d: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(strip_private(d), indent=2, default=str))
