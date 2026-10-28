#!/usr/bin/env python3
"""Markdown tables from scripts/vpr_benchmark.py outputs (recall.csv, e2e.csv, timing.json, meta.json).

    python scripts/vpr_summary.py --sim <results/sim> --real <results/real> [--out tables.md]
Pure python (csv/json), runs in the slambench venv.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

METHOD_NAMES = {"bow": "DBoW2 (ORB, ORBvoc)", "anyloc": "AnyLoc (DINOv2-G VLAD)", "mixvpr": "MixVPR (R50, 4096)",
                "cosplace": "CosPlace (R50, 2048)", "oracle": "oracle (GT nearest)"}
ORDER = ["bow", "anyloc", "mixvpr", "cosplace", "oracle"]


def short(s):
    return s.replace("sim_", "").replace("real_", "")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.0f}"


def methods_in(rows):
    have = {r["method"] for r in rows}
    return [m for m in ORDER if m in have]


def sim_recall_tables(d, out):
    rows = read_csv(os.path.join(d, "recall.csv"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    conds = meta["config"]["sessions"]
    rec = {(r["method"], r["map"], r["query"]): r for r in rows}
    for k in ("R@1", "R@5"):
        out.append(f"\n**Sim {k}** (%, rows = map lighting, columns = query lighting)\n")
        cols = [short(c) for c in conds]
        out.append("| method | map \\ query | " + " | ".join(cols) + " | cross-lighting mean |")
        out.append("|---|---|" + "---|" * (len(cols) + 1))
        for m in methods_in(rows):
            for mp in conds:
                vals = [float(rec[m, mp, q][k]) for q in conds]
                cross = [float(rec[m, mp, q][k]) for q in conds if q != mp]
                out.append(f"| {METHOD_NAMES[m] if mp == conds[0] else ''} | {short(mp)} | "
                           + " | ".join(pct(v) for v in vals) + f" | {pct(sum(cross) / len(cross))} |")
    bow = [r for r in rows if r["method"] == "bow" and r.get("bow_shared_words_best_positive_median")]
    if bow:
        out.append("\nDBoW2 shared words (median over queries): best positive / retrieved top-1\n")
        out.append("| map \\ query | " + " | ".join(short(c) for c in conds) + " |")
        out.append("|---|" + "---|" * len(conds))
        for mp in conds:
            out.append(f"| {short(mp)} | " + " | ".join(
                f"{float(rec['bow', mp, q]['bow_shared_words_best_positive_median']):.0f} / "
                f"{float(rec['bow', mp, q]['bow_shared_words_top1_median']):.0f}" for q in conds) + " |")


def real_recall_table(d, out):
    rows = read_csv(os.path.join(d, "recall.csv"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    pairs = [tuple(p) for p in meta["pairs"]]
    rec = {(r["method"], r["map"], r["query"]): r for r in rows}
    out.append("\n**Real (LunaPolaris) Recall@1 / @5 / @10** (%)\n")
    out.append("| map → query | queries (with positive) | " + " | ".join(METHOD_NAMES[m] for m in methods_in(rows)) + " |")
    out.append("|---|---|" + "---|" * len(methods_in(rows)))
    for mp, q in pairs:
        r0 = rec["bow", mp, q] if ("bow", mp, q) in rec else rec[methods_in(rows)[0], mp, q]
        cells = [f"{pct(float(rec[m, mp, q]['R@1']))} / {pct(float(rec[m, mp, q]['R@5']))} / "
                 f"{pct(float(rec[m, mp, q]['R@10']))}" for m in methods_in(rows)]
        out.append(f"| {short(mp)} → {short(q)} | {r0['n_queries']} ({r0['n_with_positive']}) | " + " | ".join(cells) + " |")


GROSS = {"m": 2.0, "deg": 10.0}  # set per suite from the positive radius in e2e_tables()


def is_gross(r):
    return r["status"] == "false_positive" and (float(r["err_t_m"] or "inf") > GROSS["m"] or
                                                float(r["err_r_deg"] or "inf") > GROSS["deg"])


def e2e_stats(rows):
    """(n, success rate, false-positive rate, gross false-positive rate) over all queries."""
    n = len(rows)
    if not n:
        return 0, float("nan"), float("nan"), float("nan")
    s = sum(r["status"] == "success" for r in rows)
    fp = sum(r["status"] == "false_positive" for r in rows)
    return n, s / n, fp / n, sum(is_gross(r) for r in rows) / n


def cell(rows):
    _, s, fp, g = e2e_stats(rows)
    return f"{pct(s)} / {pct(fp)} / {pct(g)}"


def e2e_tables(d, out, suite):
    path = os.path.join(d, "e2e.csv")
    if not os.path.exists(path):
        out.append(f"\n(no e2e.csv in {d})\n")
        return
    rows = read_csv(path)
    meta = json.load(open(os.path.join(d, "meta.json")))
    pairs = [tuple(p) for p in meta["pairs"]]
    g = defaultdict(list)
    for r in rows:
        g[r["method"], r["verifier"], r["depth"], r["K"], r["map"], r["query"]].append(r)
    combos = sorted({(r["method"], r["verifier"], r["depth"], r["K"]) for r in rows},
                    key=lambda c: (ORDER.index(c[0]), c[1] != "sp_lg", c[2] != "gt", c[3]))
    cfg = meta["config"]
    GROSS["m"] = cfg["pos_m"]
    out.append(f"\n**{suite} end-to-end relocalization**: success % / FP % / gross FP % of all queries "
               f"(success: error < {cfg['succ_m']} m and < {cfg['succ_deg']}°; localized = best of top-K by PnP "
               f"inliers, ≥ 20 inliers; FP = localized but outside the success threshold; gross FP = error "
               f"> {cfg['pos_m']} m or > {GROSS['deg']:.0f}°, i.e. wrong place)\n")
    if suite == "sim":
        groups = {"same lighting": [p for p in pairs if p[0] == p[1]],
                  "cross lighting (12 pairs)": [p for p in pairs if p[0] != p[1]]}
        key = [("sim_nominal", "sim_low_sun"), ("sim_low_sun", "sim_glare"), ("sim_nominal", "sim_glare"),
               ("sim_glare", "sim_nominal")]
        cols = list(groups) + [f"{short(a)}→{short(b)}" for a, b in key]
    else:
        groups = {}
        key = pairs
        cols = [f"{short(a)}→{short(b)}" for a, b in key]
    hdr = "| retrieval | verifier | 3D | K | " + " | ".join(cols) + " |"
    out.append(hdr)
    out.append("|---|---|---|---|" + "---|" * len(cols))
    for c in combos:
        cells = []
        for name, ps in groups.items():
            cells.append(cell([r for p in ps for r in g[(*c, *p)]]))
        for p in key:
            cells.append(cell(g[(*c, *p)]))
        out.append(f"| {METHOD_NAMES[c[0]]} | {c[1]} | {c[2]} | {c[3]} | " + " | ".join(cells) + " |")
    n = {p: e2e_stats(g[(*combos[0], *p)])[0] for p in pairs}
    out.append(f"\nqueries per pair: {', '.join(f'{short(a)}→{short(b)} {k}' for (a, b), k in n.items())}")
    if suite == "sim":
        conds = cfg["sessions"]
        for depth in sorted({c[2] for c in combos}):
            out.append(f"\n**Sim 4×4, SuperPoint+LightGlue, top-5, {depth} depth**: success % / FP % / gross FP %\n")
            out.append("| retrieval | map \\ query | " + " | ".join(short(x) for x in conds) + " |")
            out.append("|---|---|" + "---|" * len(conds))
            for m in [c[0] for c in combos if c[1:] == ("sp_lg", depth, "5")]:
                for mp in conds:
                    cells = [cell(g[m, "sp_lg", depth, "5", mp, q]) for q in conds]
                    out.append(f"| {METHOD_NAMES[m] if mp == conds[0] else ''} | {short(mp)} | " + " | ".join(cells) + " |")
    # verification failure analysis: retrieval correct (positive chosen) vs not
    out.append(f"\n{suite}: localized queries split by whether the chosen database frame is a GT positive "
               "(K=5, SP+LG; cross-session pairs pooled)\n")
    out.append("| retrieval | 3D | localized % | chosen frame positive % (of localized) | success % (of localized, positive) "
               "| gross FP % (of localized, not positive) | median err of localized m / ° |")
    out.append("|---|---|---|---|---|---|---|")
    cross = [p for p in pairs if p[0] != p[1]]
    for c in combos:
        if c[1] != "sp_lg" or c[3] != "5":
            continue
        allr = [r for p in cross for r in g[(*c, *p)]]
        rr = [r for r in allr if r["status"] != "none"]
        posr = [r for r in rr if r["db_is_positive"] == "1"]
        neg = [r for r in rr if r["db_is_positive"] != "1"]
        med = lambda k: statistics.median(float(r[k]) for r in rr) if rr else float("nan")  # noqa: E731
        out.append(f"| {METHOD_NAMES[c[0]]} | {c[2]} | {pct(len(rr) / max(len(allr), 1))} | "
                   f"{pct(len(posr) / max(len(rr), 1))} | {pct(e2e_stats(posr)[1])} | {pct(e2e_stats(neg)[3])} | "
                   f"{med('err_t_m'):.3f} / {med('err_r_deg'):.2f} |")
    # stricter inlier gate: re-threshold the best candidate's inlier count
    thrs = (20, 50, 100, 200)
    out.append(f"\n{suite}: inlier gate on the best of top-5 (SP+LG, cross-session pairs pooled): "
               "success % / gross FP % of all queries\n")
    out.append("| retrieval | 3D | " + " | ".join(f"≥ {t} inliers" for t in thrs) + " |")
    out.append("|---|---|" + "---|" * len(thrs))
    for c in combos:
        if c[1] != "sp_lg" or c[3] != "5":
            continue
        allr = [r for p in cross for r in g[(*c, *p)]]
        cells = []
        for t in thrs:
            loc = [r for r in allr if r["status"] != "none" and int(r["n_inliers"]) >= t]
            s = sum(r["status"] == "success" for r in loc) / max(len(allr), 1)
            gr = sum(is_gross(r) for r in loc) / max(len(allr), 1)
            cells.append(f"{pct(s)} / {gr * 100:.1f}")
        out.append(f"| {METHOD_NAMES[c[0]]} | {c[2]} | " + " | ".join(cells) + " |")


def timing_table(dirs, out):
    out.append("\n**Timing and memory** (per query frame, batch 1; GPU named per suite)\n")
    out.append("| suite | GPU | method | extract median ms | retrieval ms/query (db size) | bytes per db frame | GPU peak MB | params M |")
    out.append("|---|---|---|---|---|---|---|---|")
    for suite, d in dirs.items():
        p = os.path.join(d, "timing.json")
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        meta = json.load(open(os.path.join(d, "meta.json")))
        ndb = {s: len(v) for s, v in meta["db"].items()}
        for m in ORDER[:-1]:
            ex = t["timing"].get(f"extract/{m}")
            if not ex:
                continue
            ret = [(k.split("/")[2], v["mean_ms"]) for k, v in t["timing"].items() if k.startswith(f"retrieve/{m}/")]
            r = " , ".join(f"{v:.2f} ({ndb[s]})" for s, v in ret[:1])
            mem = t["memory"].get(m, {})
            out.append(f"| {suite} | {t['gpu']} | {METHOD_NAMES[m]} | {ex['median_ms']:.1f} | {r} | "
                       f"{t['descriptor_bytes_per_db_frame'].get(m, 0):,.0f} | {mem.get('gpu_peak_mb', 0):.0f} | "
                       f"{mem.get('params_M', 0):.0f} |")
    out.append("\n**Verification cost** (median ms per call)\n")
    out.append("| suite | " + " | ".join(["SP extract", "LightGlue match", "ORB extract", "ORB match", "PnP (SP)", "SGBM stereo"]) + " |")
    out.append("|---|---|---|---|---|---|---|")
    for suite, d in dirs.items():
        p = os.path.join(d, "timing.json")
        if not os.path.exists(p):
            continue
        t = json.load(open(p))["timing"]
        get = lambda k: f"{t[k]['median_ms']:.1f}" if k in t else "–"  # noqa: E731
        out.append(f"| {suite} | " + " | ".join([get("verify/extract_sp_lg"), get("verify/match_sp_lg"),
                                                 get("verify/extract_orb"), get("verify/match_orb"),
                                                 get("verify/pnp_sp_lg_stereo"), get("verify/stereo_sgbm")]) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim")
    ap.add_argument("--real")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = []
    if a.sim:
        sim_recall_tables(a.sim, out)
    if a.real:
        real_recall_table(a.real, out)
    if a.sim:
        e2e_tables(a.sim, out, "sim")
    if a.real:
        e2e_tables(a.real, out, "real")
    timing_table({k: v for k, v in (("sim", a.sim), ("real", a.real)) if v}, out)
    text = "\n".join(out) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
