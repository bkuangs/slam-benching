"""Summarise cross_lighting_reloc.py output (pairs.csv) into markdown tables.

    python sim/omnilrs/cross_lighting_summary.py <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

d = pd.read_csv(os.path.join(sys.argv[1], "pairs.csv"))
d["ok"] = (d.err_t_m < 0.25) & (d.err_r_deg < 2.0)
d["ok_fine"] = (d.err_t_m < 0.05) & (d.err_r_deg < 0.5)
d["cross"] = d["map"] != d["query"]
order = ["orb", "sp_nn", "sp_lg", "aliked_lg", "disk_lg"]
methods = [m for m in order if m in set(d.method)]
cond = ["nominal", "low_sun", "dim", "glare"]


def pct(x):
    return f"{100 * x:.0f}%"


print("## Success rate (err < 0.25 m and < 2 deg), cross-lighting pairs only, by frame offset\n")
offs = sorted(d.offset.unique())
print("| method | " + " | ".join(f"{o} frames ({o * 2 / 15:.1f} m)" for o in offs) + " | all | median inliers |")
print("|---" * (len(offs) + 3) + "|")
for m in methods:
    x = d[(d.method == m) & d.cross]
    cells = [pct(x[x.offset == o].ok.mean()) for o in offs]
    print(f"| {m} | " + " | ".join(cells) + f" | {pct(x.ok.mean())} | {x.n_inliers.median():.0f} |")

print("\n## Success rate by query lighting (map = nominal, all offsets)\n")
print("| method | " + " | ".join(cond) + " |")
print("|---" * (len(cond) + 1) + "|")
for m in methods:
    x = d[(d.method == m) & (d["map"] == "nominal")]
    print(f"| {m} | " + " | ".join(pct(x[x["query"] == q].ok.mean()) for q in cond) + " |")

print("\n## Map x query lighting matrix (success rate, all offsets)\n")
for m in methods:
    x = d[d.method == m]
    print(f"**{m}**\n\n| map \\ query | " + " | ".join(cond) + " |")
    print("|---" * (len(cond) + 1) + "|")
    for c in cond:
        print(f"| {c} | " + " | ".join(pct(x[(x['map'] == c) & (x['query'] == q)].ok.mean()) for q in cond) + " |")
    print()

print("## Accuracy of successful localizations, cross-lighting (median) and timing\n")
print("| method | median t err (cm) | median r err (deg) | fine success (<5 cm, <0.5 deg) | extract ms | match ms |")
print("|---|---|---|---|---|---|")
for m in methods:
    x = d[(d.method == m) & d.cross]
    s = x[x.ok]
    t = json.load(open(os.path.join(sys.argv[1], f"timing_{m}.json")))
    print(f"| {m} | {100 * s.err_t_m.median():.1f} | {s.err_r_deg.median():.2f} | {pct(x.ok_fine.mean())} | "
          f"{t['extract_ms']:.1f} | {t['match_ms']:.1f} |")
