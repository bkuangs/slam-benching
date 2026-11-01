# usage: python scripts/vpr_texture_alias.py [tol_m=0.5] [vpr_work_dir]
"""Quantify how much the 5 m texture repeat in the Lunaryard sim drives VPR / relocalization errors.

An error is "aliased" when the wrong database frame is offset from the query by ~(5a, 5b) m
(each xy component within TOL of a multiple of 5 m, offset not ~0) with a similar heading.
"""
import csv, itertools, collections, sys
import numpy as np

V = sys.argv[2] if len(sys.argv) > 2 else "/storage/ice1/3/6/bkuang7/slam-bench-work/vpr"
TOL, PERIOD, HDG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5, 5.0, 30.0
S = ["sim_nominal", "sim_low_sun", "sim_dim", "sim_glare"]
P = {s: np.load(f"{V}/sessions/{s}/poses.npz")["T"] for s in S}


def heading(T):
    z = T[:3, 2]  # optical axis in world
    return np.degrees(np.arctan2(z[1], z[0]))


def offset(m, q, i, j):
    a, b = P[m][i], P[q][j]
    d = a[:2, 3] - b[:2, 3]
    dh = abs((heading(a) - heading(b) + 180) % 360 - 180)
    return d, dh


def aliased(d, dh):
    r = np.abs(d - PERIOD * np.round(d / PERIOD))
    return bool(np.all(r < TOL) and np.linalg.norm(d) > 2.0 and dh < HDG)


# chance rate: random wrong db frames that happen to satisfy the criterion
rng = np.random.default_rng(0)
n = len(P["sim_nominal"])
hits = tot = 0
for _ in range(200000):
    i, j = rng.integers(n, size=2)
    d, dh = offset("sim_nominal", "sim_nominal", i, j)
    if np.linalg.norm(d) > 2.0:
        tot += 1
        hits += aliased(d, dh)
print(f"tolerance {TOL} m; chance that a random wrong frame looks aliased: {100*hits/tot:.1f}%\n")

# 1) retrieval top-1 errors
print("Retrieval, cross-lighting pairs (12): wrong top-1 retrievals and the aliased share")
print("method   | R@1  | wrong top-1 | aliased (of all queries) | R@1 if aliased ones were correct")
for meth in ["bow", "anyloc", "mixvpr", "cosplace"]:
    nq = ok = wrong = al = 0
    for m, q in itertools.permutations(S, 2):
        z = np.load(f"{V}/results/sim/rank_{m}__{q}.npz")
        for k in range(len(z["q"])):
            if not z["pos"][k].any():
                continue
            nq += 1
            t = z[meth][k, 0]
            if z["pos"][k, t]:
                ok += 1
                continue
            wrong += 1
            al += aliased(*offset(m, q, z["db"][t], z["q"][k]))
    print(f"{meth:8s} | {100*ok/nq:4.1f} | {100*wrong/nq:5.1f}%      | {100*al/nq:5.1f}% ({100*al/max(wrong,1):.0f}% of wrong)"
          f"        | <= {100*(ok+al)/nq:4.1f}")

# 2) end-to-end false positives (sp_lg, gt depth, K=5, cross-lighting)
print("\nEnd to end (SP+LG, GT depth, top-5, >=20 inliers), cross-lighting pairs")
print("method   | success | FP    | gross FP (>2 m) | aliased gross FP | aliased w/ >=50 inl | >=100 inl")
rows = [r for r in csv.DictReader(open(f"{V}/results/sim/e2e.csv"))
        if r["verifier"] == "sp_lg" and r["depth"] == "gt" and r["K"] == "5" and r["map"] != r["query"]]
for meth in ["oracle", "bow", "anyloc", "mixvpr", "cosplace"]:
    R = [r for r in rows if r["method"] == meth]
    c = collections.Counter()
    for r in R:
        c[r["status"]] += 1
        if r["status"] != "false_positive":
            continue
        if float(r["err_t_m"]) > 2.0:
            c["gross"] += 1
            d, dh = offset(r["map"], r["query"], int(r["db_frame"]), int(r["q_frame"]))
            if aliased(d, dh):
                c["al"] += 1
                c["al50"] += int(r["n_inliers"]) >= 50
                c["al100"] += int(r["n_inliers"]) >= 100
    N = len(R)
    f = lambda k: f"{100*c[k]/N:4.1f}%"
    print(f"{meth:8s} | {f('success')}   | {f('false_positive')} | {f('gross')}           | {f('al')} ({100*c['al']/max(c['gross'],1):.0f}%)    "
          f"| {f('al50')}               | {f('al100')}")
