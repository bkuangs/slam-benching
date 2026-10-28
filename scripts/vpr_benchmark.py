#!/usr/bin/env python3
"""Visual place recognition benchmark: AnyLoc / MixVPR / CosPlace vs DBoW2 (ORB) retrieval, plus
end-to-end relocalization (top-K retrieval -> SuperPoint+LightGlue or ORB matching -> PnP RANSAC).

Sessions come from scripts/vpr_prepare_sessions.py. Run inside the LAC image on a GPU
(scripts/vpr_benchmark.sbatch); summarise with scripts/vpr_summary.py. See docs/results/place_recognition.md.
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim" / "omnilrs"))
from cross_lighting_reloc import Extractors, localize, pose_error  # noqa: E402
from vpr import metrics as M  # noqa: E402
from vpr.dbow import BowDatabase, Vocabulary  # noqa: E402
from vpr.models import DinoV2Facet, GlobalExtractor, OrbSlamLikeORB, kmeans_cosine, vlad  # noqa: E402
from vpr.sessions import load_session  # noqa: E402
from vpr.stereo import StereoDepth  # noqa: E402

SUITES = {
    "sim": {
        "sessions": ["sim_nominal", "sim_low_sun", "sim_dim", "sim_glare"],
        "pairs": None,  # full 4x4
        "pos_m": 2.0, "pos_deg": 30.0, "succ_m": 0.25, "succ_deg": 2.0,
        "db": {"every": 5, "offset": 0}, "q": {"every": 5, "offset": 2},
        "num_disp": 32,
    },
    "real": {
        "sessions": ["real_seq01", "real_seq02", "real_seq11"],
        "pairs": [("real_seq01", "real_seq11"), ("real_seq11", "real_seq01"), ("real_seq01", "real_seq02"),
                  ("real_seq02", "real_seq01"), ("real_seq11", "real_seq11"), ("real_seq01", "real_seq01")],
        "pos_m": 0.5, "pos_deg": 30.0, "succ_m": 0.10, "succ_deg": 2.0,
        "db": {"min_dist_m": 0.2, "min_angle_deg": 15.0, "min_gap": 7},  # keyframe-like, >= 0.47 s apart
        "q": {"every": 15, "offset": 7},                                 # 1 Hz, never a database frame
        "num_disp": 96,
    },
}
METHODS = ["bow", "anyloc", "mixvpr", "cosplace"]
MIN_INLIERS = 20
TOPN = 25


def imread(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return im


def select_frames(S, cfg):
    db_cfg, q_cfg = cfg["db"], cfg["q"]
    if "every" in db_cfg:
        db = [i for i in range(db_cfg["offset"], len(S), db_cfg["every"]) if S.valid[i]]
    else:
        db = M.subsample_by_motion(S.T, S.valid, db_cfg["min_dist_m"], db_cfg["min_angle_deg"], db_cfg["min_gap"])
    q = [i for i in range(q_cfg["offset"], len(S), q_cfg["every"]) if S.valid[i]]
    q = [i for i in M.queries_avoiding(q, db, 1, len(S)) if S.valid[i]]
    return db, q


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, key, dt):
        self.t.setdefault(key, []).append(dt)

    def summary(self):
        return {k: {"mean_ms": 1e3 * float(np.mean(v)), "median_ms": 1e3 * float(np.median(v)), "n": len(v)}
                for k, v in self.t.items()}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_peak_reset():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb():
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def compute_descriptors(a, sessions, frames, frames_db, maps, timer, mem):
    """desc[method][(map_session or None, session, frame)] -> vector / BoW tuple."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    desc = {}
    if "bow" in a.methods:
        t0 = time.time()
        voc = Vocabulary.load(os.path.join(a.weights, "ORBvoc.txt"))
        print(f"ORBvoc loaded in {time.time() - t0:.0f}s ({voc.n_words} words)", flush=True)
        orb, d = OrbSlamLikeORB(), {}
        for s, idx in frames.items():
            for i in idx:
                im = imread(sessions[s].left_path(i))
                t = time.perf_counter()
                d[None, s, i] = voc.transform(orb(im))
                timer.add("extract/bow", time.perf_counter() - t)
        desc["bow"] = d
        mem["bow"] = {"gpu_peak_mb": 0.0, "note": "CPU (OpenCV ORB + numpy DBoW2)"}
    for name in [m for m in a.methods if m in ("mixvpr", "cosplace")]:
        gpu_peak_reset()
        ex = GlobalExtractor(name, dev, a.weights)
        d = {}
        for s, idx in frames.items():
            for i in idx:
                im = imread(sessions[s].left_path(i))
                sync()
                t = time.perf_counter()
                d[None, s, i] = ex(im)
                sync()
                timer.add(f"extract/{name}", time.perf_counter() - t)
        desc[name] = d
        mem[name] = {"gpu_peak_mb": gpu_peak_mb(), "params_M": sum(p.numel() for p in ex.model.parameters()) / 1e6}
        del ex
    if "anyloc" in a.methods:
        gpu_peak_reset()
        dino = DinoV2Facet(dev, a.anyloc_arch, a.anyloc_layer, "value")
        n_params = sum(p.numel() for p in dino.model.parameters()) / 1e6
        # vocabularies from each map session's database frames only (never query frames)
        vocab, rng = {}, np.random.default_rng(0)
        for m in maps:
            idx = frames_db[m]
            per = int(np.ceil(a.vocab_patches / len(idx)))
            samples = []
            for i in idx:
                f = dino(imread(sessions[m].left_path(i)))
                samples.append(f[torch.from_numpy(rng.permutation(len(f))[:per]).to(f.device)].half())
            t = time.time()
            vocab[m] = kmeans_cosine(torch.cat(samples).float(), a.vlad_k)
            print(f"AnyLoc vocab {m}: {sum(len(x) for x in samples)} patches from {len(idx)} db frames, "
                  f"k-means {time.time() - t:.0f}s", flush=True)
            del samples
        d = {}
        for s, idx in frames.items():
            for i in idx:
                im = imread(sessions[s].left_path(i))
                sync()
                t = time.perf_counter()
                f = dino(im)
                for j, m in enumerate(maps):
                    v = vlad(f, vocab[m])
                    if j == 0:
                        sync()
                        timer.add("extract/anyloc", time.perf_counter() - t)  # one vocabulary per query
                    d[m, s, i] = v.half().cpu().numpy()
        desc["anyloc"] = d
        mem["anyloc"] = {"gpu_peak_mb": gpu_peak_mb(), "params_M": n_params, "arch": a.anyloc_arch,
                         "layer": a.anyloc_layer, "facet": "value", "vlad_k": a.vlad_k,
                         "n_patches": int(f.shape[0]), "patch_dim": int(f.shape[1])}
        del dino
        gpu_peak_reset()
    return desc


def retrieval(desc, method, m, q, db_idx, q_idx, timer):
    """(similarity (n_q, n_db), retrieval seconds per query, bytes per database frame)."""
    if method == "bow":
        dbase = BowDatabase([desc["bow"][None, m, i] for i in db_idx])
        sim = np.zeros((len(q_idx), len(db_idx)))
        shared = np.zeros((len(q_idx), len(db_idx)), np.int64)
        t = time.perf_counter()
        for r, i in enumerate(q_idx):
            sim[r], shared[r] = dbase.query(desc["bow"][None, q, i])
        dt = (time.perf_counter() - t) / max(len(q_idx), 1)
        timer.add(f"retrieve/bow/{m}", dt)
        return sim, dt, dbase.nbytes_per_entry(), shared
    key = m if method == "anyloc" else None
    D = np.stack([desc[method][key, m, i] for i in db_idx]).astype(np.float32)
    Q = np.stack([desc[method][key, q, i] for i in q_idx]).astype(np.float32)
    t = time.perf_counter()
    sim = Q @ D.T
    M.rank_by_similarity(sim, TOPN)
    dt = (time.perf_counter() - t) / max(len(q_idx), 1)
    timer.add(f"retrieve/{method}/{m}", dt)
    return sim, dt, D.shape[1] * 4, None


class Verifier:
    """Caches local features and database depth; verifies (map frame, query frame) with PnP RANSAC."""

    def __init__(self, sessions, cfg, timer, verifiers, depth_sources):
        self.S, self.timer, self.verifiers, self.depth_sources = sessions, timer, verifiers, depth_sources
        self.ex = Extractors("cuda" if torch.cuda.is_available() else "cpu")
        self.feat, self.depth = {}, {}
        self.pool = ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1)))
        self.stereo = {s: StereoDepth(S.K[0, 0], S.baseline_m, cfg["num_disp"]) for s, S in sessions.items()}

    def features(self, method, s, i):
        k = (method, s, i)
        if k not in self.feat:
            im = imread(self.S[s].left_path(i))
            sync()
            t = time.perf_counter()
            f = self.ex.extract(method, im)
            sync()
            self.timer.add(f"verify/extract_{method}", time.perf_counter() - t)
            if "t" in f:
                f["t"] = {kk: v.cpu() for kk, v in f["t"].items()}
            self.feat[k] = f
        return self.feat[k]

    def depth_map(self, src, s, i):
        k = (src, s, i)
        if k not in self.depth:
            if src == "gt":
                self.depth[k] = cv2.imread(self.S[s].depth_path(i), cv2.IMREAD_UNCHANGED)
            else:
                L, R = imread(self.S[s].left_path(i)), imread(self.S[s].right_path(i))
                t = time.perf_counter()
                self.depth[k] = self.stereo[s].depth_mm(L, R)
                self.timer.add("verify/stereo_sgbm", time.perf_counter() - t)
        return self.depth[k]

    def _pnp(self, v, src, m, i, q, j, kp_r, kp_q, mt):
        t = time.perf_counter()
        Te, n, _ = localize(kp_r, kp_q, mt, self.depth_map(src, m, i), self.S[m].T[i], self.S[m].K)
        dt = time.perf_counter() - t
        et, er = pose_error(Te, self.S[q].T[j]) if Te is not None else (np.inf, np.inf)
        return (v, src), (n, et, er), dt

    def verify_many(self, m, frames_r, q, j):
        """{i: {(verifier, depth_src): (n_inliers, err_t, err_r)}} for map frames (m, i) vs query frame (q, j).

        Matching runs sequentially on the GPU; the PnP RANSAC calls (OpenCV releases the GIL) run in a thread
        pool. Per-call timings are recorded individually, so they are per-candidate single-thread costs."""
        jobs = []
        for i in frames_r:
            for src in self.depth_sources:
                self.depth_map(src, m, i)
            for v in self.verifiers:
                fr, fq = self.features(v, m, i), self.features(v, q, j)
                if "t" in fr:
                    dev = self.ex.dev
                    fr = {**fr, "t": {k: x.to(dev) for k, x in fr["t"].items()}}
                    fq = {**fq, "t": {k: x.to(dev) for k, x in fq["t"].items()}}
                sync()
                t = time.perf_counter()
                mt = self.ex.match(v, fr, fq)
                sync()
                self.timer.add(f"verify/match_{v}", time.perf_counter() - t)
                jobs += [(i, v, src, fr["kp"], fq["kp"], mt) for src in self.depth_sources]
        out = {i: {} for i in frames_r}
        futs = [(i, self.pool.submit(self._pnp, v, src, m, i, q, j, kr, kq, mt)) for i, v, src, kr, kq, mt in jobs]
        for i, fu in futs:
            key, val, dt = fu.result()
            out[i][key] = val
            self.timer.add(f"verify/pnp_{key[0]}_{key[1]}", dt)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), required=True)
    ap.add_argument("--sessions", default="/storage/ice1/3/6/bkuang7/slam-bench-work/vpr/sessions")
    ap.add_argument("--weights", default="/storage/ice1/3/6/bkuang7/slam-bench-work/vpr_weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--anyloc-arch", default="dinov2_vitg14")
    ap.add_argument("--anyloc-layer", type=int, default=31)
    ap.add_argument("--vlad-k", type=int, default=32)
    ap.add_argument("--vocab-patches", type=int, default=200000, help="patches sampled for each VLAD vocabulary")
    ap.add_argument("--verifiers", nargs="+", default=["sp_lg", "orb"])
    ap.add_argument("--e2e-stride", type=int, default=1, help="use every n-th query for end-to-end relocalization")
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--limit-queries", type=int, default=0, help="debug: cap queries/db per session")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = SUITES[a.suite]
    sessions = {s: load_session(os.path.join(a.sessions, s)) for s in cfg["sessions"]}
    pairs = cfg["pairs"] or list(itertools.product(cfg["sessions"], cfg["sessions"]))
    maps = sorted({m for m, _ in pairs}, key=cfg["sessions"].index)

    frames_db, frames_q = {}, {}
    for s, S in sessions.items():
        frames_db[s], frames_q[s] = select_frames(S, cfg)
        if a.limit_queries:
            frames_db[s] = frames_db[s][::max(1, len(frames_db[s]) // a.limit_queries)]
            frames_q[s] = frames_q[s][::max(1, len(frames_q[s]) // a.limit_queries)]
        print(f"{s}: {len(frames_db[s])} db frames, {len(frames_q[s])} queries", flush=True)
    frames = {s: sorted(set(frames_db[s]) | set(frames_q[s])) for s in sessions}
    json.dump({"suite": a.suite, "config": cfg, "args": vars(a), "pairs": pairs,
               "db": frames_db, "queries": frames_q,
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)

    timer, mem = Timer(), {}
    desc = compute_descriptors(a, sessions, frames, frames_db, maps, timer, mem)

    # ---- retrieval
    rec_rows, rankings, bytes_per = [], {}, {}
    for m, q in pairs:
        db_idx, q_idx = frames_db[m], frames_q[q]
        Tdb, Tq = sessions[m].T[db_idx], sessions[q].T[q_idx]
        pos = M.positives(Tq, Tdb, cfg["pos_m"], cfg["pos_deg"])
        rk = {"oracle": M.oracle_ranking(Tq, Tdb, cfg["pos_deg"], TOPN)}
        extra = {}
        for method in a.methods:
            sim, dt, nb, shared = retrieval(desc, method, m, q, db_idx, q_idx, timer)
            rk[method] = M.rank_by_similarity(sim, TOPN)
            bytes_per[method] = nb
            if shared is not None:  # DBoW2 diagnostics: shared words with best positive vs top-1
                has = pos.any(1)
                best_pos = np.where(pos, shared, -1).max(1)
                extra["bow_shared_words_best_positive_median"] = float(np.median(best_pos[has])) if has.any() else None
                extra["bow_shared_words_top1_median"] = float(np.median(shared[np.arange(len(q_idx)), rk[method][:, 0]]))
        for method, r in rk.items():
            row = {"map": m, "query": q, "method": method, **M.recall_at_k(r, pos, (1, 5, 10))}
            if method == "bow":
                row.update(extra)
            rec_rows.append(row)
            print(f"  {m} -> {q} {method:8s} R@1 {row['R@1']:.3f} R@5 {row['R@5']:.3f} R@10 {row['R@10']:.3f}",
                  flush=True)
        rankings[m, q] = rk
        np.savez(os.path.join(a.out, f"rank_{m}__{q}.npz"), db=db_idx, q=q_idx, pos=pos,
                 **{k: v for k, v in rk.items()})
    keys = sorted({k for r in rec_rows for k in r}, key=lambda k: list(rec_rows[0]).index(k)
                  if k in rec_rows[0] else 99)
    with open(os.path.join(a.out, "recall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rec_rows)

    descriptor_bytes = {k: float(v) for k, v in bytes_per.items()}
    del desc

    # ---- end-to-end relocalization
    if not a.no_e2e:
        depth_sources = ["gt", "stereo"] if a.suite == "sim" else ["stereo"]
        ver = Verifier(sessions, cfg, timer, a.verifiers, depth_sources)
        e2e_path = os.path.join(a.out, "e2e.csv")
        fields = ["map", "query", "q_frame", "method", "verifier", "depth", "K", "status", "db_frame",
                  "db_is_positive", "has_positive", "n_inliers", "err_t_m", "err_r_deg"]
        with open(e2e_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for m, q in pairs:
                t0 = time.time()
                db_idx, q_idx = frames_db[m], frames_q[q]
                rk = rankings[m, q]
                z = np.load(os.path.join(a.out, f"rank_{m}__{q}.npz"))
                pos = z["pos"]
                for r in range(0, len(q_idx), a.e2e_stride):
                    j = q_idx[r]
                    cands = sorted({int(c) for meth in rk for c in rk[meth][r, :5]})
                    vr = ver.verify_many(m, [db_idx[c] for c in cands], q, j)
                    res = {c: vr[db_idx[c]] for c in cands}
                    for meth, (v, src), K in itertools.product(rk, itertools.product(a.verifiers, depth_sources),
                                                               (1, 5)):
                        top = [int(c) for c in rk[meth][r, :K]]
                        best = max(top, key=lambda c: res[c][v, src][0])
                        n, et, er = res[best][v, src]
                        if n < MIN_INLIERS:
                            status = "none"
                        elif et < cfg["succ_m"] and er < cfg["succ_deg"]:
                            status = "success"
                        else:
                            status = "false_positive"
                        w.writerow({"map": m, "query": q, "q_frame": j, "method": meth, "verifier": v, "depth": src,
                                    "K": K, "status": status, "db_frame": db_idx[best],
                                    "db_is_positive": int(pos[r, best]), "has_positive": int(pos[r].any()),
                                    "n_inliers": n, "err_t_m": round(et, 4) if np.isfinite(et) else "",
                                    "err_r_deg": round(er, 3) if np.isfinite(er) else ""})
                f.flush()
                # drop query-session features not needed later to bound memory
                print(f"e2e {m} -> {q}: {len(q_idx[::a.e2e_stride])} queries {time.time() - t0:.0f}s", flush=True)
                if len(ver.feat) > 6000:
                    ver.feat.clear()

    json.dump({"timing": timer.summary(), "memory": mem, "descriptor_bytes_per_db_frame": descriptor_bytes,
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"},
              open(os.path.join(a.out, "timing.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
