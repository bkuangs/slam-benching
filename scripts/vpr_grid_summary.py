#!/usr/bin/env python3
"""Classical-vs-learned relocalization grid tables from a scripts/vpr_benchmark.py output directory.

    python scripts/vpr_grid_summary.py <results dir> [--gates 20 50] [--out tables.md]

Rows = verifier (local features x matcher), columns = retrieval; cells = success % (gross false positives %) of
all queries, pooled over the cross-lighting pairs and, separately, the same-lighting pairs. A query is localized
when the best of the top-K candidates (most PnP inliers) has >= gate inliers (20 = benchmark gate, 50 = ORB-SLAM3 /
SP_SLAM3 Relocalization acceptance). Pure python, runs in the slambench venv.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

RETRIEVAL = ["bow", "sp_bow", "sp_bow_s1", "mixvpr", "anyloc", "cosplace", "oracle"]
RETRIEVAL_NAMES = {"bow": "ORB DBoW2", "sp_bow": "SP DBoW3", "sp_bow_s1": "SP DBoW3 s1", "mixvpr": "MixVPR",
                   "anyloc": "AnyLoc", "cosplace": "CosPlace", "oracle": "oracle"}
VERIFIERS = ["orb", "orb_slam", "sp_slam", "sp_slam_s1", "sp_slam_s1_lvl2", "sp_nn", "sp_lg", "aliked_lg", "disk_lg"]
VERIFIER_NAMES = {"orb": "ORB + BF NNDR", "orb_slam": "ORB + ORB-SLAM3 matching",
                  "sp_slam": "SP + SP_SLAM3 matching (BoW scale 0.46)", "sp_slam_s1": "SP + SP_SLAM3 matching (scale 1)",
                  "sp_slam_s1_lvl2": "SP + SP_SLAM3 matching (scale 1, level-2 nodes)", "sp_nn": "SP + mutual NN",
                  "sp_lg": "SP + LightGlue", "aliked_lg": "ALIKED + LightGlue", "disk_lg": "DISK + LightGlue"}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.0f}"


def short(s):
    return s.replace("real_", "").replace("sim_", "")


def outcome(r, gate, cfg):
    """success | fp | gross | none for one e2e row at an inlier gate."""
    if int(r["n_inliers"]) < gate or r["err_t_m"] == "":
        return "none"
    et, er = float(r["err_t_m"]), float(r["err_r_deg"])
    if et < cfg["succ_m"] and er < cfg["succ_deg"]:
        return "success"
    return "gross" if et > cfg["pos_m"] or er > 10.0 else "fp"


def rates(rows, gate, cfg):
    c = Counter(outcome(r, gate, cfg) for r in rows)
    n = max(len(rows), 1)
    return c["success"] / n, (c["gross"]) / n, (c["fp"] + c["gross"]) / n


def grid(out, rows, pairs, gate, K, cfg, title):
    sel = [r for r in rows if (r["map"], r["query"]) in pairs and r["K"] == str(K)]
    ret = [m for m in RETRIEVAL if any(r["method"] == m for r in sel)]
    ver = [v for v in VERIFIERS if any(r["verifier"] == v for r in sel)]
    g = defaultdict(list)
    for r in sel:
        g[r["verifier"], r["method"]].append(r)
    nq = len(g[ver[0], ret[0]]) if ver and ret else 0
    out.append(f"\n**{title}** (gate {gate} inliers, top-{K}, {nq} queries): success % (gross FP %)\n")
    out.append("| verifier | " + " | ".join(RETRIEVAL_NAMES[m] for m in ret) + " |")
    out.append("|---|" + "---|" * len(ret))
    for v in ver:
        cells = []
        for m in ret:
            s, gr, _ = rates(g[v, m], gate, cfg)
            cells.append(f"{pct(s)} ({pct(gr)})")
        out.append(f"| {VERIFIER_NAMES.get(v, v)} | " + " | ".join(cells) + " |")


def stages(out, rows, pairs):
    top1 = [r for r in rows if (r["map"], r["query"]) in pairs and r["method"] == "oracle" and r["K"] == "1"]
    sel = [r for r in top1 if r.get("stage")]
    if not sel:
        return
    nm = defaultdict(list)
    for r in top1:
        if r.get("n_matches", "") != "":
            nm[r["verifier"]].append(float(r["n_matches"]))
    by = defaultdict(Counter)
    for r in sel:
        by[r["verifier"]][r["stage"]] += 1
    names = ["few_bow", "pnp_fail", "bow", "projection", "projection2"]
    out.append("\n**SLAM-style verifiers, oracle top-1, where relocalization ends** (% of queries; few_bow = < 15 "
               "SearchByBoW matches, pnp_fail = < 10 PnP inliers, bow / projection(2) = final pose after the BoW "
               "matches / first (second) projection pass; matches = median / 90th percentile of SearchByBoW matches, or of "
               "LightGlue matches before PnP)\n")
    out.append("| verifier | " + " | ".join(names) + " | matches |")
    out.append("|---|" + "---|" * (len(names) + 1))
    for v in [v for v in VERIFIERS if v in by or v in nm]:
        n = sum(by[v].values())
        cells = [pct(by[v][s] / n) if n else "–" for s in names]
        cells.append(f"{np.median(nm[v]):.0f} / {np.percentile(nm[v], 90):.0f}" if nm[v] else "–")
        out.append(f"| {VERIFIER_NAMES.get(v, v)} | " + " | ".join(cells) + " |")


def recall(out, d, pairs):
    rows = read_csv(os.path.join(d, "recall.csv"))
    rec = {(r["method"], r["map"], r["query"]): r for r in rows}
    ret = [m for m in RETRIEVAL if any(r["method"] == m for r in rows)]
    out.append("\n**Retrieval Recall@1 / @5** (%)\n")
    out.append("| map → query | queries (with positive) | " + " | ".join(RETRIEVAL_NAMES[m] for m in ret) + " |")
    out.append("|---|---|" + "---|" * len(ret))
    for mp, q in pairs:
        r0 = rec[ret[0], mp, q]
        out.append(f"| {short(mp)} → {short(q)} | {r0['n_queries']} ({r0['n_with_positive']}) | " + " | ".join(
            f"{pct(float(rec[m, mp, q]['R@1']))} / {pct(float(rec[m, mp, q]['R@5']))}" for m in ret) + " |")


def timing(out, d):
    p = os.path.join(d, "timing.json")
    if not os.path.exists(p):
        return
    t = json.load(open(p))["timing"]
    keys = sorted(k for k in t if k.startswith(("verify/", "extract/")))
    out.append("\n**Timing** (median ms per call; verify/slam_* = SearchByBoW + PnP + projection passes, "
               "verify/pnp_* = PnP only, verify/match_* = GPU matching)\n")
    out.append("| step | median ms | calls |")
    out.append("|---|---|---|")
    for k in keys:
        out.append(f"| {k} | {t[k]['median_ms']:.1f} | {t[k]['n']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--gates", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--K", type=int, nargs="+", default=[5])
    ap.add_argument("--out")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.results, "meta.json")))
    cfg = meta["config"]
    pairs = [tuple(p) for p in meta["pairs"]]
    cross = {p for p in pairs if p[0] != p[1]}
    same = {p for p in pairs if p[0] == p[1]}
    out = [f"# Relocalization grid: {meta['suite']} ({os.path.basename(os.path.normpath(a.results))})",
           f"\nsuccess: < {cfg['succ_m']} m and < {cfg['succ_deg']}°; gross FP: localized > {cfg['pos_m']} m or "
           f"> 10° off. Every cell uses the same database frames, stereo depth and PnP RANSAC."]
    recall(out, a.results, pairs)
    rows = read_csv(os.path.join(a.results, "e2e.csv"))
    for gate in a.gates:
        for K in a.K:
            grid(out, rows, cross, gate, K, cfg, f"Cross lighting ({', '.join(f'{short(m)}→{short(q)}' for m, q in sorted(cross))})")
            if same:
                grid(out, rows, same, gate, K, cfg, "Same lighting (controls)")
    for p in sorted(cross):
        grid(out, rows, {p}, a.gates[0], a.K[0], cfg, f"{short(p[0])} → {short(p[1])}")
    stages(out, rows, cross)
    timing(out, a.results)
    text = "\n".join(out) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
