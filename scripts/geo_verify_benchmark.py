#!/usr/bin/env python3
"""Geometric verification of cross-lighting relocalization candidates (offline).

For each query, the MixVPR top-5 candidates (rank files of scripts/vpr_benchmark.py) are matched with
SuperPoint+LightGlue and localized with PnP RANSAC (as in the VPR benchmark). Each candidate's PnP pose then
places the query's own 3D points (sim: GT depth or SGBM stereo; real: SGBM stereo or Livox LiDAR, full scan or
ground-only crop) in the map, where robust point-to-plane ICP against the map's 3D points around the candidate
(the candidate frame plus neighbouring map frames, placed with the map session's GT poses) measures shape
agreement, the ICP correction and the registration conditioning (vpr/geometry.py). Accept rules (inlier gate,
low inlier floor, low floor + geometry, ...) are then compared offline. See docs/results/geometric_verification.md.

Stages (each writes small files into --out):
  match     GPU, LAC image: per-candidate PnP (n_inliers, pose)        -> cands_<map>__<query>.npz
  lidar     slambench venv (rosbags): Livox scans nearest the needed frames, camera frame, voxelized
                                                                        -> lidar_<session>.npz
  geometry  CPU (LAC image or venv): ICP + conditioning per candidate   -> geo_<map>__<query>.npz
  summary   thresholds tuned on the tuning pairs, tables for all pairs  -> summary.md, summary.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim" / "omnilrs"))
from vpr import geometry as G  # noqa: E402
from vpr.sessions import load_session  # noqa: E402
from vpr.stereo import StereoDepth  # noqa: E402

VPR = "/storage/ice1/3/6/bkuang7/slam-bench-work/vpr"
TOPK = 5
MIN_PNP = 6            # localize() needs >= 6 RANSAC inliers; "PnP succeeded"
GATE = 50              # current SP_SLAM3 learned-relocalization gate
FLOORS = [8, 12, 20, 30]

# LunaPolaris /tf_static (identical for sequence_01/02/11; printed by `lidar` into lidar_<session>.npz too):
# camera -> rover ground plane (leo01/base_footprint), used for the ground-only crop of stereo points.
T_FOOT_CAM0 = np.array([[0.0, -0.4674, 0.8841, 0.1091], [-1.0, 0.0, 0.0, 0.06], [0.0, -0.8841, -0.4674, 0.4749],
                        [0, 0, 0, 1.0]])
MAX_LIDAR_DT = 0.1  # s; Livox runs at 10 Hz
CROP_H, CROP_R = 0.5, 3.0   # ground-only crop as in docs/research/lidar_ground_only.md

SUITES = {
    "sim": {
        "pairs": [("sim_nominal", "sim_dim"), ("sim_nominal", "sim_low_sun"), ("sim_nominal", "sim_glare")],
        # sim_dim has no correct low-inlier candidates (all queries match with >= 50), so it cannot tune a threshold
        "tune": [("sim_nominal", "sim_low_sun")],
        "results": f"{VPR}/results/sim", "pnp_depth": "gt", "num_disp": 32,
        "succ_m": 0.25, "succ_deg": 2.0, "gross_m": 2.0, "gross_deg": 10.0,
        "sources": ["depth", "stereo"],
        # geometry parameters per source: depth stride, z range, voxel, map neighbour radius, ICP schedule
        "geo": {"stride": 4, "z": (0.3, 30.0), "voxel": 0.10, "nb_m": 3.0, "nb_deg": 45.0, "nb_max": 8,
                "max_corr": [1.0, 0.5, 0.25], "inlier": 0.10, "max_pts": 3000, "basin": (0.5, 5.0)},
    },
    "real": {
        "pairs": [("real_seq01", "real_seq02"), ("real_seq01", "real_seq11"), ("real_seq11", "real_seq01"),
                  ("real_seq01", "real_seq01"), ("real_seq11", "real_seq11")],
        # 01->02 has no correct low-inlier candidates; tune on one cross-day direction, hold out the other
        "tune": [("real_seq01", "real_seq11")],
        "results": f"{VPR}/results/real", "pnp_depth": "stereo", "num_disp": 96,
        "succ_m": 0.10, "succ_deg": 2.0, "gross_m": 0.5, "gross_deg": 10.0, "loose_m": 0.25,
        "sources": ["stereo", "stereo_ground", "lidar", "lidar_ground"],
        "geo": {"stride": 2, "z": (0.3, 6.0), "voxel": 0.03, "nb_m": 1.0, "nb_deg": 45.0, "nb_max": 8,
                "max_corr": [0.3, 0.15, 0.08], "inlier": 0.04, "max_pts": 3000, "basin": (0.15, 5.0)},
        "geo_lidar": {"voxel": 0.05, "max_corr": [0.3, 0.15, 0.08], "inlier": 0.05, "max_pts": 3000},
    },
}


def tag(m, q):
    return f"{m}__{q}"


def imread(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return im


# ------------------------------------------------------------------------------------------------- match
def cmd_match(a):
    import torch
    from cross_lighting_reloc import Extractors, localize, pose_error

    cfg = SUITES[a.suite]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ex = Extractors(dev)
    sessions = {}
    feats, depth, tm = {}, {}, {"extract": [], "match": [], "pnp": [], "sgbm": []}

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def feat(s, i):
        if (s, i) not in feats:
            im = imread(sessions[s].left_path(i))
            sync()
            t = time.perf_counter()
            f = ex.extract("sp_lg", im)
            sync()
            tm["extract"].append(time.perf_counter() - t)
            feats[s, i] = f
        return feats[s, i]

    def dmap(s, i):
        if (s, i) not in depth:
            S = sessions[s]
            if cfg["pnp_depth"] == "gt":
                depth[s, i] = cv2.imread(S.depth_path(i), cv2.IMREAD_UNCHANGED)
            else:
                t = time.perf_counter()
                depth[s, i] = StereoDepth(S.K[0, 0], S.baseline_m, cfg["num_disp"]).depth_mm(
                    imread(S.left_path(i)), imread(S.right_path(i)))
                tm["sgbm"].append(time.perf_counter() - t)
        return depth[s, i]

    for m, q in cfg["pairs"]:
        for s in (m, q):
            sessions.setdefault(s, load_session(os.path.join(a.sessions, s)))
        out = os.path.join(a.out, f"cands_{tag(m, q)}.npz")
        if os.path.exists(out) and not a.force:
            print("exists", out)
            continue
        z = np.load(os.path.join(cfg["results"], f"rank_{tag(m, q)}.npz"))
        db, qi, pos, rk = z["db"], z["q"], z["pos"], z["mixvpr"][:, :TOPK]
        if a.limit:
            sel = np.linspace(0, len(qi) - 1, a.limit).astype(int)
            qi, pos, rk = qi[sel], pos[sel], rk[sel]
        nq = len(qi)
        n_inl = np.zeros((nq, TOPK), int)
        n_match = np.zeros((nq, TOPK), int)
        T = np.full((nq, TOPK, 4, 4), np.nan)
        err = np.full((nq, TOPK, 2), np.inf)
        t0 = time.time()
        for r, j in enumerate(qi):
            fq = feat(q, int(j))
            for k, c in enumerate(rk[r]):
                i = int(db[c])
                fr = feat(m, i)
                sync()
                t = time.perf_counter()
                mt = ex.match("sp_lg", fr, fq)
                sync()
                tm["match"].append(time.perf_counter() - t)
                t = time.perf_counter()
                Te, n, _ = localize(fr["kp"], fq["kp"], mt, dmap(m, i), sessions[m].T[i], sessions[m].K)
                tm["pnp"].append(time.perf_counter() - t)
                n_match[r, k], n_inl[r, k] = len(mt), n
                if Te is not None:
                    T[r, k] = Te
                    err[r, k] = pose_error(Te, sessions[q].T[int(j)])
            if len(feats) > 4000:
                feats = {key: v for key, v in feats.items() if key[0] == m}
        np.savez(out, db=db, q=qi, cand=rk, pos=pos[np.arange(nq)[:, None], rk], n_inl=n_inl, n_match=n_match,
                 T=T, err=err)
        print(f"{m} -> {q}: {nq} queries {time.time() - t0:.0f}s", flush=True)
    json.dump({k: {"mean_ms": 1e3 * float(np.mean(v)), "median_ms": 1e3 * float(np.median(v)), "n": len(v)}
               for k, v in tm.items() if v} | {"device": torch.cuda.get_device_name() if dev == "cuda" else "cpu"},
              open(os.path.join(a.out, f"timing_match_{a.suite}.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------- lidar
def cmd_lidar(a):
    """Livox scans nearest the query / map frames of the real pairs, in the left camera frame (voxelized)."""
    from slambench.datasets.lunapolaris import LunaPolarisAdapter, crop_lidar_points, livox_points

    cfg = SUITES["real"]
    need = {}
    for m, q in cfg["pairs"]:
        z = np.load(os.path.join(cfg["results"], f"rank_{tag(m, q)}.npz"))
        need.setdefault(m, set()).update(int(i) for i in z["db"])
        need.setdefault(q, set()).update(int(i) for i in z["q"])
    voxel = cfg["geo_lidar"]["voxel"]
    for s, frames in need.items():
        out = os.path.join(a.out, f"lidar_{s}.npz")
        if os.path.exists(out) and not a.force:
            print("exists", out)
            continue
        S = load_session(os.path.join(a.sessions, s))
        frames = np.array(sorted(frames))
        tf = S.t_ns[frames]
        ds = LunaPolarisAdapter(Path(a.lunapolaris), S.meta["sequence"])
        tree = ds.static_tf()
        T_body_cam0 = ds.T_gt_body_sensor("cam0")
        T_cam0_lidar = np.linalg.inv(T_body_cam0) @ ds.T_gt_body_sensor("lidar")
        T_foot_lidar = tree.lookup(ds.options["lidar_crop_frame"], ds.options["lidar_frame"])
        best_dt = np.full(len(frames), np.inf)
        best = [None] * len(frames)
        t0 = time.time()
        n_msgs = 0
        with ds._reader() as r:
            conns = [c for c in r.connections if c.topic == ds.options["lidar_topic"]]
            for c, tb, raw in r.messages(connections=conns):
                # bag receive time is close to the scan time; skip scans that cannot be nearest
                if np.min(np.abs(tf - tb)) > 5.0e9:
                    continue
                ts, pts = livox_points(r.deserialize(raw, c.msgtype))
                n_msgs += 1
                nz = np.c_[pts["x"], pts["y"], pts["z"]].any(axis=1)
                if nz.sum() < 100:
                    continue
                mid = int((ts[nz].min() + ts[nz].max()) // 2)
                dt = np.abs(tf - mid)
                upd = dt < best_dt
                if not upd.any():
                    continue
                pts = pts[nz]
                crop = crop_lidar_points(pts, T_foot_lidar, CROP_H, CROP_R)
                P = np.c_[pts["x"], pts["y"], pts["z"]].astype(float)
                Pc = G.transform(T_cam0_lidar, P)
                full = G.voxel_downsample(Pc, voxel)
                ground = G.voxel_downsample(Pc[crop], voxel)
                for k in np.flatnonzero(upd):
                    best_dt[k], best[k] = dt[k], (full, ground)
        empty = np.zeros((0, 3))
        full_l = [b[0] if b is not None else empty for b in best]
        ground_l = [b[1] if b is not None else empty for b in best]
        np.savez_compressed(out, frames=frames, dt_ns=best_dt.astype(np.int64),
                            full=np.concatenate(full_l).astype(np.float16),
                            full_off=np.cumsum([0] + [len(x) for x in full_l]),
                            ground=np.concatenate(ground_l).astype(np.float16),
                            ground_off=np.cumsum([0] + [len(x) for x in ground_l]),
                            T_cam0_lidar=T_cam0_lidar, T_foot_lidar=T_foot_lidar,
                            T_foot_cam0=tree.lookup(ds.options["lidar_crop_frame"], ds.options["cam0_frame"]))
        print(f"{s}: {len(frames)} frames from {n_msgs} scans in {time.time() - t0:.0f}s; |dt| median "
              f"{np.median(best_dt) / 1e6:.1f} ms max {best_dt.max() / 1e6:.1f} ms; points/frame full "
              f"{np.median([len(x) for x in full_l]):.0f} ground {np.median([len(x) for x in ground_l]):.0f}; "
              f"{os.path.getsize(out) / 1e6:.0f} MB", flush=True)


# ---------------------------------------------------------------------------------------------- geometry
class Clouds:
    """Per-frame sensor-frame point clouds for one session and source."""

    def __init__(self, S, suite, source, out_dir):
        self.S, self.suite, self.source = S, suite, source
        cfg = SUITES[suite]
        self.g = dict(cfg["geo"]) | (cfg.get("geo_lidar", {}) if source.startswith("lidar") else {})
        self.stereo = StereoDepth(S.K[0, 0], S.baseline_m, cfg["num_disp"])
        self.lidar = None
        if source.startswith("lidar"):
            z = np.load(os.path.join(out_dir, f"lidar_{S.name}.npz"))
            key = "ground" if source.endswith("ground") else "full"
            self.lidar = ({int(f): k for k, f in enumerate(z["frames"])}, z[key], z[key + "_off"], z["dt_ns"])
        self.time = []
        self.memo = None  # set to a dict to cache per-frame clouds (map side)

    def __call__(self, i):
        if self.memo is not None and i in self.memo:
            return self.memo[i]
        t = time.perf_counter()
        g = self.g
        if self.lidar is not None:
            idx, P, off, dt_ns = self.lidar
            k = idx[i]
            # a scan further than MAX_LIDAR_DT from the image counts as missing (geometry then fails)
            pts = P[off[k]:off[k + 1]].astype(np.float64) if dt_ns[k] <= MAX_LIDAR_DT * 1e9 else np.zeros((0, 3))
        else:
            if self.source == "depth":
                d = cv2.imread(self.S.depth_path(i), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
            else:
                d = self.stereo.depth_mm(imread(self.S.left_path(i)), imread(self.S.right_path(i))) / 1000.0
            pts = G.depth_to_points(d, self.S.K, g["stride"], *g["z"])
            if self.source.endswith("ground"):
                f = G.transform(T_FOOT_CAM0, pts)
                pts = pts[(f[:, 2] <= CROP_H) & (np.hypot(f[:, 0], f[:, 1]) <= CROP_R)]
            pts = G.voxel_downsample(pts, g["voxel"])
        self.time.append(time.perf_counter() - t)
        if self.memo is not None:
            self.memo[i] = pts
        return pts


def map_neighbours(S, db, i, g):
    """Map (database) frames within nb_m and nb_deg of map frame i (i first), at most nb_max."""
    Ti = S.T[i]
    d = np.linalg.norm(S.T[db, :3, 3] - Ti[:3, 3], axis=1)
    cosang = np.clip(np.einsum("ij,j->i", S.T[db, :3, 2], Ti[:3, 2]), -1, 1)
    ok = (d <= g["nb_m"]) & (np.degrees(np.arccos(cosang)) <= g["nb_deg"])
    cand = [int(x) for x in db[ok][np.argsort(d[ok])] if int(x) != i]
    return [i] + cand[:g["nb_max"] - 1]


_W = {}


def _geo_query(r):
    """ICP results of every PnP-successful candidate of query row r, for every source."""
    W = _W
    z, Sm = W["cands"], W["Sm"]
    j = int(z["q"][r])
    out = []
    for src in W["sources"]:
        qc, mc, g = W["qclouds"][src], W["mclouds"][src], W["g"][src]
        ks = [k for k in range(TOPK) if z["n_inl"][r, k] >= MIN_PNP and np.isfinite(z["T"][r, k, 0, 0])]
        if not ks:
            continue
        t = time.perf_counter()
        Q = qc(j)
        t_q = time.perf_counter() - t
        rng = np.random.default_rng(j)
        if len(Q) > g["max_pts"]:
            Q = Q[rng.choice(len(Q), g["max_pts"], replace=False)]
        for k in ks:
            i = int(z["db"][z["cand"][r, k]])
            M = mc(i)
            t = time.perf_counter()
            res = G.icp_point_to_plane(Q, M, z["T"][r, k], g["max_corr"], g["inlier"])
            t_icp = time.perf_counter() - t
            t = time.perf_counter()
            bt, br = (G.basin_test(Q, M, res.T, *g["basin"], max_corr=g["max_corr"], inlier_dist=g["inlier"])
                      if res.fitness > 0 else (np.inf, np.inf))
            t_basin = time.perf_counter() - t
            et, er = G.pose_delta(res.T, W["Sq"].T[j])
            out.append((r, k, src, res.fitness, res.rmse, res.n_inliers, res.n_points, res.corr_t, res.corr_r,
                        res.cond_t, res.cond_r, res.cond6, int(res.converged), et, er, len(M), t_q, t_icp, bt, br,
                        t_basin))
    return out


class MapClouds:
    def __init__(self, S, db, clouds, g):
        self.S, self.db, self.clouds, self.g, self.cache, self.time = S, db, clouds, g, {}, []

    def __call__(self, i):
        if i not in self.cache:
            t = time.perf_counter()
            parts = []
            for n in map_neighbours(self.S, self.db, i, self.g):
                parts.append(G.transform(self.S.T[n], self.clouds(n)))
            self.cache[i] = G.MapCloud(np.vstack(parts), voxel=self.g["voxel"])
            self.time.append(time.perf_counter() - t)
        return self.cache[i]


def cmd_geometry(a):
    from multiprocessing import get_context

    cfg = SUITES[a.suite]
    sources = a.sources or cfg["sources"]
    for m, q in cfg["pairs"]:
        out = os.path.join(a.out, f"geo_{tag(m, q)}.npz")
        if os.path.exists(out) and not a.force:
            print("exists", out)
            continue
        t0 = time.time()
        z = dict(np.load(os.path.join(a.out, f"cands_{tag(m, q)}.npz")))
        Sm, Sq = load_session(os.path.join(a.sessions, m)), load_session(os.path.join(a.sessions, q))
        _W.clear()
        _W.update(cands=z, Sm=Sm, Sq=Sq, sources=sources, qclouds={}, mclouds={}, g={})
        # map clouds for every candidate frame, built once before forking
        need = sorted({int(z["db"][c]) for r in range(len(z["q"])) for c in z["cand"][r]})
        map_t = {}
        for src in sources:
            mcl = Clouds(Sm, a.suite, src, a.out)
            _W["g"][src] = mcl.g
            _W["qclouds"][src] = Clouds(Sq, a.suite, src, a.out)
            mcl.memo = {}
            mc = MapClouds(Sm, z["db"], mcl, mcl.g)
            for i in need:
                mc(i)
            _W["mclouds"][src] = mc
            map_t[src] = (float(np.mean(mc.time)), float(np.mean(mcl.time)), float(np.mean([len(c) for c in mc.cache.values()])))
            print(f"  {m} -> {q} {src}: {len(need)} map clouds in {sum(mc.time):.0f}s "
                  f"({np.mean([len(c) for c in mc.cache.values()]):.0f} pts)", flush=True)
        with get_context("fork").Pool(a.workers) as pool:
            rows = [x for part in pool.imap(_geo_query, range(len(z["q"])), chunksize=4) for x in part]
        cols = ["r", "k", "src", "fitness", "rmse", "n_icp_inl", "n_pts", "corr_t", "corr_r", "cond_t", "cond_r",
                "cond6", "converged", "icp_err_t", "icp_err_r", "n_map", "t_query_cloud", "t_icp", "basin_t", "basin_r",
                "t_basin"]
        arr = {c: np.array([x[n] for x in rows]) for n, c in enumerate(cols)}
        np.savez(out, **arr, map_cloud_s={k: v[0] for k, v in map_t.items()},
                 frame_cloud_s={k: v[1] for k, v in map_t.items()})
        print(f"{m} -> {q}: {len(rows)} ICP runs in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------- summary
def load_pair(out_dir, suite, m, q):
    """Per-candidate table: inliers, PnP errors, positive flag and geometry per source."""
    cfg = SUITES[suite]
    z = dict(np.load(os.path.join(out_dir, f"cands_{tag(m, q)}.npz")))
    gz = dict(np.load(os.path.join(out_dir, f"geo_{tag(m, q)}.npz"), allow_pickle=True))
    nq = len(z["q"])
    geo = {}
    for src in cfg["sources"]:
        sel = gz["src"] == src
        g = {}
        for c in ("fitness", "rmse", "corr_t", "corr_r", "cond_t", "cond_r", "cond6", "icp_err_t", "icp_err_r",
                  "t_query_cloud", "t_icp", "n_icp_inl", "basin_t", "basin_r", "t_basin"):
            v = np.full((nq, TOPK), np.nan)
            v[gz["r"][sel].astype(int), gz["k"][sel].astype(int)] = gz[c][sel].astype(float)
            g[c] = v
        geo[src] = g
    return z, geo


def classify(et, er, cfg, loose=False):
    if loose:
        succ = et < cfg["loose_m"]
    else:
        succ = (et < cfg["succ_m"]) & (er < cfg["succ_deg"])
    gross = (et > cfg["gross_m"]) | (er > cfg["gross_deg"])
    return succ, gross


def geo_pass(g, th):
    """(pass, rejected only by conditioning), boolean (nq, K). Shape + correction: fitness, rmse, ICP correction.
    Conditioning: Hessian translation eigenvalue cond_t >= th["cond"] and basin spread <= th["basin"] (m).
    Missing geometry (NaN) fails."""
    with np.errstate(invalid="ignore"):
        ok = (g["fitness"] >= th["fit"]) & (g["rmse"] <= th["rmse"]) & (g["corr_t"] <= th["corr_t"]) & \
             (g["corr_r"] <= th["corr_r"])
        cond = (g["cond_t"] >= th["cond"]) & (g["basin_t"] <= th["basin"])
    return ok & cond, ok & ~cond


def apply_rule(rule, z, g, th, floor):
    """(accepted mask (nq, K), score for picking among accepted)."""
    n = z["n_inl"]
    pnp = (n >= MIN_PNP) & np.isfinite(z["T"][..., 0, 0])
    gp = geo_pass(g, th)[0] if g is not None else None
    if rule == "A":
        acc = n >= GATE
    elif rule == "B":
        acc = n >= floor
    elif rule == "C":
        acc = (n >= floor) & gp
    elif rule == "D":
        acc = (n >= GATE) | ((n >= floor) & gp)
    elif rule == "E":
        acc = pnp & gp
    else:
        raise ValueError(rule)
    acc &= pnp
    score = np.where(acc, g["fitness"] if rule == "E" else n, -np.inf)
    return acc, score


def evaluate(z, g, cfg, rule, th, floor, use_icp_pose=False, loose=False):
    acc, score = apply_rule(rule, z, g, th, floor)
    et, er = z["err"][..., 0], z["err"][..., 1]
    pnp = (z["n_inl"] >= MIN_PNP) & np.isfinite(z["T"][..., 0, 0])
    succ_pnp, gross_pnp = classify(et, er, cfg, loose)
    if use_icp_pose and g is not None:
        refined = acc & geo_pass(g, th)[0]   # the ICP pose replaces PnP only where geometry passed
        et = np.where(refined, g["icp_err_t"], et)
        er = np.where(refined, g["icp_err_r"], er)
    succ_c, gross_c = classify(et, er, cfg, loose)
    nq = len(z["q"])
    best = np.argmax(score, 1)
    loc = acc.any(1)
    rows = np.arange(nq)
    s_q = loc & succ_c[rows, best]
    gross_q = loc & gross_c[rows, best]
    fp_q = loc & ~succ_c[rows, best]
    correct = pnp & succ_pnp     # candidate-level counts always use the PnP pose
    wrong_place = pnp & gross_pnp
    errs = et[rows, best][loc]
    return {"n_q": nq, "success": s_q.mean(), "fp": fp_q.mean(), "gross_fp": gross_q.mean(),
            "localized": loc.mean(),
            "acc_correct": (acc & correct).sum() / max(correct.sum(), 1), "n_correct": int(correct.sum()),
            "acc_gross_cands": int((acc & wrong_place).sum()), "n_gross_cands": int(wrong_place.sum()),
            "med_err_t": float(np.median(errs)) if len(errs) else float("nan"),
            "med_err_r": float(np.median(er[rows, best][loc])) if loc.any() else float("nan")}


def tune(datasets, cfg, floor, grid, max_gross_cands=0):
    """Thresholds maximizing accepted correct low-inlier candidates (floor <= n < GATE) on the tuning pairs,
    with at most max_gross_cands wrong-place candidates accepted."""
    best, best_key = None, None
    for th in grid:
        n_ok, n_bad = 0, 0
        for z, g in datasets:
            n = z["n_inl"]
            pnp = (n >= MIN_PNP) & np.isfinite(z["T"][..., 0, 0])
            succ, gross = classify(z["err"][..., 0], z["err"][..., 1], cfg)
            low = pnp & (n >= floor) & (n < GATE) & geo_pass(g, th)[0]
            n_ok += int((low & succ).sum())
            n_bad += int((low & gross).sum())
        # ties: prefer stricter conditioning, then stricter shape thresholds
        key = (n_bad <= max_gross_cands, n_ok, -n_bad, th["cond"], -th["basin"], th["fit"], -th["rmse"])
        if best_key is None or key > best_key:
            best, best_key = dict(th), key
    return best, {"accepted_correct_low": best_key[1], "accepted_gross_low": -best_key[2]}


def make_grid(cfg):
    inl = cfg["geo"]["inlier"]
    shift = cfg["geo"]["basin"][0]
    sim = cfg is SUITES["sim"]
    grid = []
    for fit in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for rm in (0.5 * inl, 0.6 * inl, 0.7 * inl, 1.0 * inl):
            for ct in (0.5, 0.25, 0.1):
                for cond in (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
                    for basin in (np.inf, 0.5 * shift, 0.3 * shift, 0.15 * shift):
                        grid.append({"fit": fit, "rmse": rm, "corr_t": ct * (2 if sim else 1), "corr_r": 5.0,
                                     "cond": cond, "basin": basin})
    return grid


def pct(x):
    return f"{100 * x:.1f}"


def cmd_summary(a):
    lines, js = ["# Geometric verification: summary tables (generated by scripts/geo_verify_benchmark.py summary)", ""], {}
    for suite in a.suites:
        cfg = SUITES[suite]
        data = {}
        for m, q in cfg["pairs"]:
            if os.path.exists(os.path.join(a.out, f"geo_{tag(m, q)}.npz")):
                data[m, q] = load_pair(a.out, suite, m, q)
        if not data:
            continue
        pairs = list(data)
        tune_pairs = [p for p in (a.tune.get(suite) if a.tune.get(suite) else cfg["tune"]) if p in data]
        js[suite] = {"thresholds": {}, "results": {}}
        lines += [f"## {suite}", "", f"Thresholds tuned on {', '.join(f'{m}→{q}' for m, q in tune_pairs)} "
                  "(maximize accepted correct low-inlier candidates with 0 wrong-place candidates accepted); "
                  "all other pairs are held out.", ""]
        grid = make_grid(cfg)
        ths = {}
        for src in cfg["sources"]:
            for floor in FLOORS:
                th, info = tune([(data[p][0], data[p][1][src]) for p in tune_pairs], cfg, floor, grid)
                ths[src, floor] = th
                js[suite]["thresholds"][f"{src}/{floor}"] = th | info
        lines += ["### Tuned thresholds", "", "| source | floor | fitness ≥ | rmse ≤ m | correction ≤ m / ° | cond_t ≥ | basin ≤ m | "
                  "accepted correct low-inlier cands (tune) | accepted wrong-place (tune) |", "|---|---|---|---|---|---|---|---|---|"]
        for (src, floor), th in ths.items():
            info = js[suite]["thresholds"][f"{src}/{floor}"]
            lines.append(f"| {src} | {floor} | {th['fit']} | {th['rmse']:.3f} | {th['corr_t']} / {th['corr_r']} | "
                         f"{th['cond']} | {th['basin']:.3f} | {info['accepted_correct_low']} | {info['accepted_gross_low']} |")
        lines.append("")
        # main table: rules x pairs
        for loose in ([False, True] if suite == "real" else [False]):
            for src in cfg["sources"]:
                hdr = (f"### {suite}, geometry from {src}" + (" (success = translation < 0.25 m only)" if loose else "")
                       + ": success % / FP % / gross FP % of queries; accepted share of correct candidates %")
                lines += [hdr, "", "| rule | floor | " + " | ".join(f"{m}→{q}" for m, q in pairs) + " |",
                          "|---|---|" + "---|" * len(pairs)]
                for rule in ("A", "B", "C", "D", "E"):
                    for floor in (FLOORS if rule in "BCD" else [FLOORS[0] if rule == "E" else 0]):
                        th = ths[src, floor if rule != "A" else FLOORS[0]]
                        cells = []
                        for p in pairs:
                            z, geo = data[p]
                            r = evaluate(z, geo[src], cfg, rule, th, floor, loose=loose)
                            js[suite]["results"][f"{src}/{rule}/{floor}/{'loose' if loose else 'strict'}/{p[0]}->{p[1]}"] = r
                            cells.append(f"{pct(r['success'])} / {pct(r['fp'])} / {pct(r['gross_fp'])}; "
                                         f"{pct(r['acc_correct'])}")
                        lines.append(f"| {rule} | {floor if rule in 'BCDE' else '–'} | " + " | ".join(cells) + " |")
                # rule D with the ICP-refined pose replacing PnP for every accepted candidate
                floor = FLOORS[0]
                cells = []
                for p in pairs:
                    z, geo = data[p]
                    r = evaluate(z, geo[src], cfg, "D", ths[src, floor], floor, use_icp_pose=True, loose=loose)
                    js[suite]["results"][f"{src}/D_icp/{floor}/{'loose' if loose else 'strict'}/{p[0]}->{p[1]}"] = r
                    cells.append(f"{pct(r['success'])} / {pct(r['fp'])} / {pct(r['gross_fp'])}; {pct(r['acc_correct'])}")
                lines.append(f"| D, ICP pose | {floor} | " + " | ".join(cells) + " |")
                lines.append("")
        # conditioning rejections, ICP-refined pose, candidate stats
        lines += [f"### {suite}: candidate-level breakdown (floor {FLOORS[0]} ≤ inliers < {GATE}, tuned thresholds)", "",
                  "| source | pair | low-inlier cands | correct | wrong place | pass shape+correction | of which rejected by "
                  "conditioning (correct / wrong place) | accepted (correct / wrong place) |", "|---|---|---|---|---|---|---|---|"]
        for src in cfg["sources"]:
            th = ths[src, FLOORS[0]]
            for p in pairs:
                z, geo = data[p]
                g = geo[src]
                n = z["n_inl"]
                low = (n >= FLOORS[0]) & (n < GATE) & np.isfinite(z["T"][..., 0, 0])
                succ, gross = classify(z["err"][..., 0], z["err"][..., 1], cfg)
                ok, cond_rej = geo_pass(g, th)
                lines.append(f"| {src} | {p[0]}→{p[1]} | {int(low.sum())} | {int((low & succ).sum())} | "
                             f"{int((low & gross).sum())} | {int((low & (ok | cond_rej)).sum())} | "
                             f"{int((low & cond_rej & succ).sum())} / {int((low & cond_rej & gross).sum())} | "
                             f"{int((low & ok & succ).sum())} / {int((low & ok & gross).sum())} |")
        lines.append("")
        lines += [f"### {suite}: PnP vs ICP-refined pose of accepted candidates (rule D, floor {FLOORS[0]})", "",
                  "| source | pair | median err PnP m / ° | median err ICP m / ° | success PnP % | success ICP % |",
                  "|---|---|---|---|---|---|"]
        for src in cfg["sources"]:
            th = ths[src, FLOORS[0]]
            for p in pairs:
                z, geo = data[p]
                acc, _ = apply_rule("D", z, geo[src], th, FLOORS[0])
                gi = acc & np.isfinite(geo[src]["icp_err_t"])
                if not gi.any():
                    continue
                e0t, e0r = z["err"][..., 0][gi], z["err"][..., 1][gi]
                e1t, e1r = geo[src]["icp_err_t"][gi], geo[src]["icp_err_r"][gi]
                s0 = classify(e0t, e0r, cfg)[0].mean()
                s1 = classify(e1t, e1r, cfg)[0].mean()
                lines.append(f"| {src} | {p[0]}→{p[1]} | {np.median(e0t):.3f} / {np.median(e0r):.2f} | "
                             f"{np.median(e1t):.3f} / {np.median(e1r):.2f} | {pct(s0)} | {pct(s1)} |")
        lines.append("")
        # timing
        lines += [f"### {suite}: geometry timing (ms, single thread, per query or per candidate)", "",
                  "| source | query cloud / query | ICP / candidate | basin test / candidate | map cloud / candidate frame (offline in SLAM) |",
                  "|---|---|---|---|---|"]
        for src in cfg["sources"]:
            tq, ti, tb, tm_ = [], [], [], []
            for p in pairs:
                gz = np.load(os.path.join(a.out, f"geo_{tag(*p)}.npz"), allow_pickle=True)
                sel = gz["src"] == src
                tq += list(gz["t_query_cloud"][sel])
                ti += list(gz["t_icp"][sel])
                tb += list(gz["t_basin"][sel])
                tm_.append(gz["map_cloud_s"].item()[src])
            lines.append(f"| {src} | {1e3 * np.median(tq):.1f} | {1e3 * np.median(ti):.1f} | {1e3 * np.median(tb):.1f} | {1e3 * np.mean(tm_):.0f} |")
        tf = os.path.join(a.out, f"timing_match_{suite}.json")
        if os.path.exists(tf):
            t = json.load(open(tf))
            lines += ["", "Matching stage (" + t.get("device", "") + "): " + ", ".join(
                f"{k} {v['median_ms']:.1f} ms" for k, v in t.items() if isinstance(v, dict)), ""]
        lines.append("")
    open(os.path.join(a.out, f"{a.name}.md"), "w").write("\n".join(lines))
    json.dump(js, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1, default=float)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("match", "lidar", "geometry", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=f"{VPR}/geo_verify")
        p.add_argument("--sessions", default=f"{VPR}/sessions")
        p.add_argument("--force", action="store_true")
        if name in ("match", "geometry"):
            p.add_argument("--suite", choices=list(SUITES), required=True)
        if name == "match":
            p.add_argument("--limit", type=int, default=0, help="debug: queries per pair")
        if name == "lidar":
            p.add_argument("--lunapolaris", default=os.path.expanduser("~/scratch/LunaPolaris"))
        if name == "geometry":
            p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
            p.add_argument("--sources", nargs="+")
        if name == "summary":
            p.add_argument("--suites", nargs="+", default=list(SUITES))
            p.add_argument("--tune", nargs="+", default=[], metavar="SUITE:MAP:QUERY",
                           help="override the tuning pairs, e.g. real:real_seq11:real_seq01")
            p.add_argument("--name", default="summary", help="output file stem")
    a = ap.parse_args()
    if a.cmd == "summary":
        tune = {}
        for t in a.tune:
            su, m, q = t.split(":")
            tune.setdefault(su, []).append((m, q))
        a.tune = tune
    os.makedirs(a.out, exist_ok=True)
    {"match": cmd_match, "lidar": cmd_lidar, "geometry": cmd_geometry, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
