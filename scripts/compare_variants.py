#!/usr/bin/env python3
"""Compare repeated runs of two (or more) configuration variants of the same sequence.

    python scripts/compare_variants.py -o OUT_DIR \
        --variant plain='~/scratch/slam-bench-work/runs/REPEATS/subtmrs_smokeroom_orbslam3_mono/*' \
        --variant clahe='~/scratch/slam-bench-work/runs/REPEATS/subtmrs_smokeroom_orbslam3_mono_clahe/*' \
        [--window 260 345] [--thresholds 0.25 0.5 1.0]

Accuracy comes from the regular evaluator (same association, valid-sample rule, per-coordinate-session
alignment); per-sample errors are recomputed in memory because metrics.json does not store them.
All errors are *aligned benchmark accuracy*: each coordinate session is aligned on its own over its
whole duration. A new monocular map is a new session with its own frame and scale, so its error says
nothing about whether its poses can be related to the earlier map at the time they are emitted.

Writes OUT_DIR/report.md, runs.csv, window_bins.csv, and window_online_error.png.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from slambench.algorithms.base import trajectories_from_output_json
from slambench.evaluation.evaluate import _evaluate_kind, _input_timestamps
from slambench.evaluation.metrics import DEFAULT_EVAL, UNKNOWN_SESSION, FrameDiagnostics, coverage_from_states
from slambench.runner import make_dataset
from slambench.types import SensorMode

KINDS = ("online", "final")


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_run(rd: Path, ds_cache: Dict[str, Any]) -> Dict[str, Any]:
    cfg = yaml.safe_load((rd / "config.resolved.yaml").read_text())
    status = json.loads((rd / "status.json").read_text())
    key = json.dumps(cfg["dataset"], sort_keys=True)
    if key not in ds_cache:
        ds = make_dataset(cfg)
        ds_cache[key] = (ds, ds.ground_truth())
    ds, gt = ds_cache[key]
    ecfg = dict(DEFAULT_EVAL)
    ecfg.update(cfg.get("evaluation") or {})
    ao = json.loads((rd / "algorithm_output.json").read_text())
    trajs = {t.kind: t for t in trajectories_from_output_json(ao)}
    its = _input_timestamps(rd)
    diag = FrameDiagnostics.read(Path(ao["frame_diagnostics"]))
    mode = SensorMode(cfg["algorithm"]["mode"])
    env = json.loads((rd / "environment.json").read_text()) if (rd / "environment.json").exists() else {}
    run: Dict[str, Any] = {"dir": rd, "run_id": rd.name, "cfg": cfg, "status": status, "ecfg": ecfg,
                           "input_ts": its, "t0": int(its[0]), "diag": diag,
                           "coverage": coverage_from_states(diag, its),
                           "host": f"{env.get('hostname', '?')} ({env.get('cpu_model', '?')})",
                           "git": ((env.get("git") or {}).get("commit") or "")[:7]
                           + ("+dirty" if (env.get("git") or {}).get("dirty") else "")}
    for k in KINDS:
        acc, _ = _evaluate_kind(trajs[k], ds, gt, ecfg, diag, its, mode) if k in trajs else ({}, None)
        run[k] = acc
        sv = acc.get("_series") if isinstance(acc, dict) else None
        if sv is not None:
            pooled = [s for s in acc["sessions"] if s["included_in_pooled"]]
            sid = np.array([pooled[i]["session"]["session_id"] for i in sv["session"]]) if "session" in sv \
                else np.zeros(len(sv["t_ns"]), int)
            run[k + "_err"] = {int(t): float(e) for t, e in zip(sv["t_ns"].tolist(), sv["trans_err"].tolist())}
            run[k + "_series"] = {**sv, "session_id": sid, "rel": (sv["t_ns"] - run["t0"]) * 1e-9}
        else:
            run[k + "_err"], run[k + "_series"] = {}, None
    return run


def settings_fingerprint(run: Dict[str, Any]) -> str:
    c = run["cfg"]
    keep = {"dataset": c["dataset"], "algorithm": {k: c["algorithm"].get(k) for k in ("name", "mode", "settings")},
            "evaluation": c.get("evaluation")}
    keep["algorithm"]["settings"] = {k: v for k, v in (keep["algorithm"]["settings"] or {}).items() if k != "clahe"}
    return json.dumps(keep, sort_keys=True, default=str)


# ----------------------------------------------------------------------------
# per-run metrics
# ----------------------------------------------------------------------------
def main_outage(run: Dict[str, Any], window) -> Dict[str, Any]:
    """The first tracking loss (after initialisation) starting inside `window`, and how it ended."""
    d, t0 = run["diag"], run["t0"]
    rel = (d.timestamps_ns - t0) * 1e-9
    ok, cs, mp = d.tracking_ok, d.coordinate_session, d.map_id
    first_ok = int(np.argmax(ok)) if ok.any() else None
    out: Dict[str, Any] = {"loss_rel_s": None}
    if first_ok is None:
        return out
    idx = [i for i in range(first_ok + 1, len(ok)) if ok[i - 1] and not ok[i] and window[0] <= rel[i] < window[1]]
    if not idx:
        return out
    i = idx[0]
    j = i
    while j < len(ok) and not ok[j]:
        j += 1
    prev_cs, prev_map = int(cs[i - 1]), int(mp[i - 1])
    has_cs = not np.all(cs == UNKNOWN_SESSION)
    out.update({"loss_rel_s": float(rel[i]), "last_ok_session": prev_cs, "last_ok_map": prev_map,
                "states_during": sorted(set(d.state[i:j].tolist()))})
    nm = [k for k in range(i, j) if mp[k] != prev_map]
    out["new_map_created_rel_s"] = float(rel[nm[0]]) if nm else None
    if j < len(ok):
        k = j
        while k < len(ok) and cs[k] == -1:  # first frame of a new frame is marked ambiguous
            k += 1
        new_cs, new_map = int(cs[min(k, len(ok) - 1)]), int(mp[j])
        out.update({"recovered_rel_s": float(rel[j]), "outage_s": float(rel[j] - rel[i]),
                    "recovered_map": new_map, "recovered_session": new_cs,
                    "recovery_kind": ("new map (re-initialised)" if new_map != prev_map else
                                      "same map id (no coordinate-session log)" if not has_cs else
                                      "relocalised in same map and frame" if new_cs == prev_cs else
                                      "same map id, new coordinate frame")})
        # later frame change while tracking, without a map-id change: map merge / re-expression
        # (the frame where it happens is marked -1, so compare consecutive non-ambiguous OK frames)
        later, last = [], None
        for m in range(k + 1, len(ok)) if has_cs else []:
            if not ok[m]:
                last = None
                continue
            if cs[m] == -1:
                continue
            if last is not None and cs[m] != cs[last] and mp[m] == mp[last]:
                later.append(m)
            last = m
        out["later_frame_change_rel_s"] = float(rel[later[0]]) if later else None
    else:
        out.update({"recovered_rel_s": None, "outage_s": float(rel[-1] - rel[i]), "recovery_kind": "not recovered"})
    in_final = d.in_final
    post = (rel >= (out.get("recovered_rel_s") or 1e18)) & ok
    out["post_recovery_frames_in_final"] = float(in_final[post].mean()) if post.any() else None
    return out


def frac_good(run: Dict[str, Any], kind: str, thr: float) -> float:
    """Fraction of input frames with a valid evaluated pose whose error is below thr."""
    e = np.array(list(run[kind + "_err"].values()))
    return float((e < thr).sum() / len(run["input_ts"])) if len(e) else 0.0


def err_stats(errs: List[float]) -> Dict[str, Any]:
    e = np.asarray(errs, float)
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))), "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)), "max": float(e.max())}


def local_drift(run: Dict[str, Any], kind: str, dt_s: float = 1.0):
    """(rel time of pair start, |Δest - Δgt| over dt_s) for pairs inside one session and segment."""
    sv = run[kind + "_series"]
    if sv is None:
        return np.zeros(0), np.zeros(0)
    t, P, G = sv["t_ns"], sv["est_pos"], sv["gt_pos"]
    key = sv["session_id"] * 100000 + sv["segment"]
    out_t, out_e = [], []
    j = 0
    for i in range(len(t)):
        j = max(j, i + 1)
        while j < len(t) and (t[j] - t[i]) * 1e-9 < dt_s:
            j += 1
        if j >= len(t):
            break
        if key[j] != key[i] or (t[j] - t[i]) * 1e-9 > dt_s * 1.5:
            continue
        out_t.append(sv["rel"][i])
        out_e.append(np.linalg.norm((P[j] - P[i]) - (G[j] - G[i])))
    return np.array(out_t), np.array(out_e)


# ----------------------------------------------------------------------------
# report helpers
# ----------------------------------------------------------------------------
def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def dist(vals, nd=2) -> str:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return "–"
    if len(v) == 1:
        return fmt(float(v[0]), nd)
    return f"{np.median(v):.{nd}f} [{v.min():.{nd}f}–{v.max():.{nd}f}]"


def table(header: List[str], rows: List[List[Any]]) -> str:
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |\n"
    return s


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True, help="NAME=GLOB of run directories")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--window", nargs=2, type=float, default=[260.0, 345.0],
                    help="seconds since the first frame: degraded interval to inspect")
    ap.add_argument("--bin", type=float, default=5.0)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    W = tuple(a.window)

    ds_cache: Dict[str, Any] = {}
    variants: Dict[str, List[Dict[str, Any]]] = {}
    for spec in a.variant:
        name, pat = spec.split("=", 1)
        dirs = sorted(Path(p) for p in glob.glob(os.path.expanduser(pat)) if (Path(p) / "status.json").exists())
        runs = []
        for d in dirs:
            st = json.loads((d / "status.json").read_text())
            if st.get("status") not in ("success", "crash", "timeout") or not (d / "algorithm_output.json").exists():
                print(f"skip {d} (status {st.get('status')})")
                continue
            prep_dir = Path(json.loads((d / "prepared_input.json").read_text()).get("directory", ""))
            if not (prep_dir / "frames.csv").exists():
                print(f"skip {d} (prepared input {prep_dir} no longer exists)")
                continue
            print(f"load {name}: {d.name}")
            runs.append(load_run(d, ds_cache))
        variants[name] = runs
    names = list(variants)
    all_runs = [r for n in names for r in variants[n]]
    if not all_runs:
        raise SystemExit("no successful runs")

    rep: List[str] = ["# Variant comparison: " + " vs ".join(f"`{n}` (n={len(variants[n])})" for n in names), ""]
    # identical inputs / settings check
    fps = {n: sorted({settings_fingerprint(r) for r in variants[n]}) for n in names}
    same_inputs = len({json.dumps(r["cfg"]["dataset"], sort_keys=True) for r in all_runs}) == 1
    prep = {json.loads((r["dir"] / "prepared_input.json").read_text()).get("directory") for r in all_runs}
    gits = sorted({r["git"] for r in all_runs})
    rep += ["## Controls",
            f"* Dataset config identical across all runs: {'yes' if same_inputs else '**NO**'}; prepared input "
            f"directories: {', '.join(f'`{p}`' for p in sorted(prep))}",
            f"* Settings identical except `clahe` across variants: "
            f"{'yes' if len({f for v in fps.values() for f in v}) == 1 else '**NO** – check configs'}",
            f"* Git commits: {', '.join(gits) or 'unknown'}; hosts: "
            f"{', '.join(sorted({str(r['host']) for r in all_runs}))}",
            "* Accuracy = aligned benchmark accuracy (Sim(3) fitted per coordinate session over the whole session, "
            "using its ground truth). `online` = pose returned at each frame; `final` = trajectory saved at "
            "shutdown (largest map only). New monocular maps have their own frame and scale.", ""]

    # ---------------- per-run table
    rows_csv = []
    for n in names:
        for r in variants[n]:
            cov, mo = r["coverage"], main_outage(r, W)
            row = {"variant": n, "run_id": r["run_id"], "status": r["status"].get("status"), "host": r["host"],
                   "frame_coverage": cov["frame_coverage"], "time_to_first_ok_s": cov["time_to_first_ok_s"],
                   "lost_after_init_s": cov["total_lost_duration_s"], "n_loss_events": cov["n_tracking_loss_events"],
                   "n_maps": cov["n_maps"], "final_traj_frame_coverage": cov["final_trajectory_frame_coverage"],
                   **{f"outage_{k}": v for k, v in mo.items() if k != "states_during"}}
            for k in KINDS:
                acc = r[k]
                ate = acc.get("ate", {}) if isinstance(acc, dict) else {}
                row[f"{k}_ate_rmse_m"] = (ate.get("translation_m") or {}).get("rmse") if ate.get("available", True) else None
                row[f"{k}_n_sessions"] = acc.get("n_sessions")
                row[f"{k}_eval_duration_fraction"] = (acc.get("evaluated_coverage") or {}).get("evaluated_duration_fraction")
                for th in a.thresholds:
                    row[f"{k}_frac_frames_err_lt_{th:g}m"] = frac_good(r, k, th)
            r["row"] = row
            rows_csv.append(row)
    cols = list(dict.fromkeys(k for row in rows_csv for k in row))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_csv)

    th_mid = a.thresholds[len(a.thresholds) // 2]
    rep += ["## Per run",
            "`loss`/`re-OK` = first tracking loss inside the window and the first tracked frame after it; "
            "`merge` = a later change of coordinate frame without a map-id change while tracking (map merge); "
            "`final cov.` = fraction of input frames in the final (largest-map) trajectory.", "",
            table(["variant", "run", "status", "frame cov.", "loss events", "lost after init (s)", "loss at (s)",
                   "re-OK at (s)", "outage (s)", "recovery", "merge at (s)", "final cov.", "online ATE (m)",
                   "final ATE (m)", f"online frames <{th_mid:g} m"],
                  [[n, r["run_id"], r["row"]["status"], r["row"]["frame_coverage"], r["row"]["n_loss_events"],
                    r["row"]["lost_after_init_s"],
                    r["row"].get("outage_loss_rel_s"), r["row"].get("outage_recovered_rel_s"),
                    r["row"].get("outage_outage_s"), r["row"].get("outage_recovery_kind") or "–",
                    r["row"].get("outage_later_frame_change_rel_s"), r["row"]["final_traj_frame_coverage"],
                    r["row"]["online_ate_rmse_m"], r["row"]["final_ate_rmse_m"],
                    r["row"][f"online_frac_frames_err_lt_{th_mid:g}m"]] for n in names for r in variants[n]])]

    # ---------------- distributions
    def col(n, k):
        return [r["row"].get(k) for r in variants[n]]

    def count(n, pred):
        return f"{sum(1 for r in variants[n] if pred(r['row']))}/{len(variants[n])}"
    rep += ["## Distributions (median [min–max] over runs)",
            table(["metric"] + names,
                  [["runs that crashed / timed out", *[count(n, lambda x: x["status"] != "success") for n in names]],
                   ["runs whose post-loss map merged back", *[count(n, lambda x: x.get("outage_later_frame_change_rel_s") is not None) for n in names]],
                   ["runs re-losing tracking after recovery (>1 loss event)", *[count(n, lambda x: (x["n_loss_events"] or 0) > 1) for n in names]]])]
    metrics = [("frame coverage", "frame_coverage", 3), ("tracking lost after init (s)", "lost_after_init_s", 1),
               (f"loss inside {W[0]:g}–{W[1]:g} s starts at (s)", "outage_loss_rel_s", 1),
               ("tracking OK again at (s)", "outage_recovered_rel_s", 1), ("outage duration (s)", "outage_outage_s", 1),
               ("online ATE RMSE, full evaluated coverage (m)", "online_ate_rmse_m", 3),
               ("online evaluated duration fraction", "online_eval_duration_fraction", 3),
               ("final ATE RMSE, full evaluated coverage (m)", "final_ate_rmse_m", 3),
               ("final evaluated duration fraction", "final_eval_duration_fraction", 3),
               ("final trajectory frame coverage", "final_traj_frame_coverage", 3)]
    metrics += [(f"{k} frames with error < {th:g} m (fraction of all input frames)",
                 f"{k}_frac_frames_err_lt_{th:g}m", 3) for k in KINDS for th in a.thresholds]
    rep += ["",
            table(["metric"] + names, [[lbl] + [dist(col(n, k), nd) for n in names] for lbl, k, nd in metrics])]

    # ---------------- shared valid timestamps
    rep += ["## Error on shared valid timestamps",
            "Timestamps where **every** run (all variants) has a valid, GT-matched pose. Each run keeps its own "
            "per-session alignment (fitted on its full coverage), so this removes the difference in *which* "
            "frames are scored but not the influence of extra frames on the alignment.", ""]
    for k in KINDS:
        sets = [set(r[k + "_err"]) for r in all_runs]
        common = set.intersection(*sets) if sets else set()
        rows = []
        for n in names:
            per = [err_stats([r[k + "_err"][t] for t in common]) for r in variants[n]]
            full = [err_stats(list(r[k + "_err"].values())) for r in variants[n]]
            rows.append([n, dist([p.get("rmse") for p in per], 3), dist([p.get("median") for p in per], 3),
                         dist([p.get("p90") for p in per], 3), dist([f.get("rmse") for f in full], 3),
                         dist([f.get("n", 0) / len(r["input_ts"]) for f, r in zip(full, variants[n])], 3)])
        rep += [f"**{k}**: {len(common)} shared timestamps "
                f"({len(common) / len(all_runs[0]['input_ts']):.1%} of input frames)", "",
                table(["variant", "RMSE shared (m)", "median shared (m)", "p90 shared (m)",
                       "RMSE full coverage (m)", "evaluated frame fraction"], rows)]

    if len(names) == 2:
        A, B = names
        rep += [f"### Pairwise shared timestamps (every {A}×{B} run pair)",
                f"For each pair, the frames both runs have a valid pose for. Δ = {B} − {A} on those frames "
                f"(negative favours {B}). For `final`, the shared set is empty when the two final trajectories "
                "kept different maps (the largest map is not the same part of the sequence).", ""]
        rows = []
        for k in KINDS:
            pp = []
            for ra in variants[A]:
                for rb in variants[B]:
                    c = set(ra[k + "_err"]) & set(rb[k + "_err"])
                    if len(c) < 20:
                        pp.append({"n": len(c)})
                        continue
                    sa = err_stats([ra[k + "_err"][t] for t in c])
                    sb = err_stats([rb[k + "_err"][t] for t in c])
                    pp.append({"n": len(c), "a": sa["rmse"], "b": sb["rmse"], "d": sb["rmse"] - sa["rmse"],
                               "dmed": sb["median"] - sa["median"]})
            v = [p for p in pp if "d" in p]
            nin = len(all_runs[0]["input_ts"])
            rows.append([k, f"{len(v)}/{len(pp)}", dist([p["n"] / nin for p in pp], 3),
                         dist([p["a"] for p in v], 3), dist([p["b"] for p in v], 3), dist([p["d"] for p in v], 3),
                         dist([p["dmed"] for p in v], 3), f"{sum(p['d'] < 0 for p in v)}/{len(v)}"])
        rep += [table(["trajectory", "pairs with ≥20 shared", "shared fraction of input", f"{A} RMSE (m)",
                       f"{B} RMSE (m)", "Δ RMSE (m)", "Δ median (m)", f"pairs where {B} lower"], rows)]

    # ---------------- exclusive coverage (pairwise)
    if len(names) == 2:
        A, B = names
        rep += [f"## Frames tracked by one variant but not the other (online, all {A}×{B} run pairs)",
                "For each pair, frames with a valid online pose in one run and none in the other. "
                "Error = that run's online error (per-session alignment).", ""]
        rows = []
        for x, y in ((B, A), (A, B)):
            per_pair = []
            for rx in variants[x]:
                for ry in variants[y]:
                    ex = sorted(set(rx["online_err"]) - set(ry["online_err"]))
                    e = [rx["online_err"][t] for t in ex]
                    rel = np.array([(t - rx["t0"]) * 1e-9 for t in ex])
                    inw = (rel >= W[0]) & (rel < W[1]) if len(rel) else np.zeros(0, bool)
                    per_pair.append({"n": len(e), "n_win": int(inw.sum()), **{
                        f"lt{th:g}": (np.sum(np.array(e) < th) if e else 0) for th in a.thresholds},
                        "med": np.median(e) if e else None, "p90": np.percentile(e, 90) if e else None,
                        "med_win": np.median(np.array(e)[inw]) if inw.any() else None})
            fr = 1.0 / np.mean([len(r["input_ts"]) for r in all_runs])
            rows.append([f"{x} only (vs {y})", dist([p["n"] * fr for p in per_pair], 3),
                         dist([p["n_win"] * fr for p in per_pair], 3), dist([p["med"] for p in per_pair], 3),
                         dist([p["med_win"] for p in per_pair], 3), dist([p["p90"] for p in per_pair], 3)]
                        + [dist([p[f"lt{th:g}"] / p["n"] if p["n"] else None for p in per_pair], 2)
                           for th in a.thresholds])
        rep += [table(["frames", "fraction of input", f"… inside {W[0]:g}–{W[1]:g} s", "median err (m)",
                       "median err in window (m)", "p90 err (m)"] + [f"share < {th:g} m" for th in a.thresholds],
                      rows)]

    # ---------------- window inspection
    edges = np.arange(W[0], W[1] + 1e-9, a.bin)
    bins_rows = []
    for n in names:
        for r in variants[n]:
            d, rel_d = r["diag"], (r["diag"].timestamps_ns - r["t0"]) * 1e-9
            dt_t, dt_e = local_drift(r, "online")
            for lo, hi in zip(edges[:-1], edges[1:]):
                m_d = (rel_d >= lo) & (rel_d < hi)
                sv = r["online_series"]
                m_e = (sv["rel"] >= lo) & (sv["rel"] < hi) if sv is not None else np.zeros(0, bool)
                m_l = (dt_t >= lo) & (dt_t < hi)
                bins_rows.append({"variant": n, "run_id": r["run_id"], "bin_start_s": lo,
                                  "tracking_ok_fraction": float(d.tracking_ok[m_d].mean()) if m_d.any() else None,
                                  "sessions": " ".join(str(s) for s in sorted(set(d.coordinate_session[m_d & d.tracking_ok].tolist()))),
                                  "maps": " ".join(str(s) for s in sorted(set(d.map_id[m_d].tolist()))),
                                  "online_err_median_m": float(np.median(sv["trans_err"][m_e])) if m_e.any() else None,
                                  "online_err_max_m": float(sv["trans_err"][m_e].max()) if m_e.any() else None,
                                  "drift_1s_median_m": float(np.median(dt_e[m_l])) if m_l.any() else None})
    with open(out / "window_bins.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bins_rows[0]))
        w.writeheader()
        w.writerows(bins_rows)
    hdr = ["bin (s)"]
    rows = []
    for lo in edges[:-1]:
        row = [f"{lo:g}–{lo + a.bin:g}"]
        for n in names:
            b = [x for x in bins_rows if x["variant"] == n and x["bin_start_s"] == lo]
            row += [dist([x["tracking_ok_fraction"] for x in b], 2), dist([x["online_err_median_m"] for x in b], 2),
                    dist([x["drift_1s_median_m"] for x in b], 3)]
        rows.append(row)
    for n in names:
        hdr += [f"{n} OK frac", f"{n} online err (m)", f"{n} 1-s drift (m)"]
    rep += [f"## Online error and map continuity, {W[0]:g}–{W[1]:g} s",
            f"Per {a.bin:g}-s bin, median [min–max] over runs of: fraction of frames tracked OK, median online "
            "error (per-session alignment), median 1-s relative drift |Δest−Δgt| (alignment-independent except "
            "for scale). Bins with no tracked frames show –.", "", table(hdr, rows)]
    cont = []
    for n in names:
        for r in variants[n]:
            mo = main_outage(r, W)
            cont.append([n, r["run_id"], mo.get("loss_rel_s"), mo.get("new_map_created_rel_s"),
                         mo.get("recovered_rel_s"), mo.get("recovery_kind") or "–",
                         f"{mo.get('last_ok_map')}→{mo.get('recovered_map')}" if mo.get("loss_rel_s") else "–",
                         mo.get("later_frame_change_rel_s"), mo.get("post_recovery_frames_in_final"),
                         ", ".join(mo.get("states_during", []))])
    rep += ["Map continuity per run:", "",
            table(["variant", "run", "loss at (s)", "new map at (s)", "OK again at (s)", "recovery", "map",
                   "later frame change, same map id (s)", "post-recovery frames in final traj.", "states while lost"],
                  cont)]

    # ---------------- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        lo, hi = W[0] - 10, W[1] + 10
        for vi, n in enumerate(names):
            for ri, r in enumerate(variants[n]):
                sv = r["online_series"]
                if sv is not None:
                    m = (sv["rel"] >= lo) & (sv["rel"] <= hi)
                    for s in np.unique(sv["session_id"][m]):
                        mm = m & (sv["session_id"] == s)
                        ax[0].plot(sv["rel"][mm], sv["trans_err"][mm], ".", ms=1.5, color=colors[vi],
                                   label=n if (ri == 0 and s == np.unique(sv["session_id"][m])[0]) else None)
                rel_d = (r["diag"].timestamps_ns - r["t0"]) * 1e-9
                m = (rel_d >= lo) & (rel_d <= hi)
                y = vi * (len(variants[n]) + 1) + ri
                ax[1].fill_between(rel_d[m], y, y + 0.8, where=r["diag"].tracking_ok[m], color=colors[vi], step="post")
        ax[0].set_ylabel("online error (m)\nper-session Sim(3)")
        ax[0].set_yscale("log")
        ax[0].legend(markerscale=8)
        ax[0].grid(alpha=.3)
        ax[1].set_ylabel("tracking OK\n(one row per run)")
        ax[1].set_yticks([])
        ax[1].set_xlabel("time since first frame (s)")
        fig.tight_layout()
        fig.savefig(out / "window_online_error.png", dpi=130)
        rep += ["", "![online error and tracking around the degraded interval](window_online_error.png)"]
    except Exception as exc:  # noqa: BLE001
        rep += ["", f"(plot failed: {exc})"]

    (out / "report.md").write_text("\n".join(rep) + "\n")
    print(f"wrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
