"""Cross-lighting relocalization benchmark on the aerial Lunaryard renders.

A "map" frame rendered under lighting A (3D points from its GT depth and GT pose) is matched against a "query"
frame rendered under lighting B, k frames further along the same trajectory (k * ~0.13 m of motion). The query
pose is estimated with PnP + RANSAC and compared with GT. This isolates the feature extractor/matcher from the
rest of a SLAM system: same images, same geometry, same solver for every method.

Run inside the LAC image (torch + lightglue + OpenCV), on a CUDA GPU:
    apptainer exec --nv lac_26c3a2b.sif python sim/omnilrs/cross_lighting_reloc.py --out <dir>
"""
import argparse
import itertools
import json
import os
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

CONDITIONS = ["nominal", "low_sun", "dim", "glare"]
METHODS = ["orb", "sp_nn", "sp_lg", "aliked_lg", "disk_lg"]
MAX_KP = 1200  # same budget as the ORB-SLAM3 / SP_SLAM3 configs


def load_seq(root):
    rows = open(os.path.join(root, "frames.csv")).read().split()[1:]
    names = [r.split(",")[1] for r in rows]
    gt = np.loadtxt(os.path.join(root, "gt_cam0.tum"))
    T = np.tile(np.eye(4), (len(gt), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(gt[:, 4:8]).as_matrix()
    T[:, :3, 3] = gt[:, 1:4]
    calib = json.load(open(os.path.join(root, "calib.json")))
    K = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1]])
    return names, T, K


class Extractors:
    def __init__(self, device):
        from lightglue import ALIKED, DISK, LightGlue, SuperPoint

        self.dev = device
        self.orb = cv2.ORB_create(nfeatures=MAX_KP, scaleFactor=1.2, nlevels=8, fastThreshold=7)
        self.nets = {
            "sp": SuperPoint(max_num_keypoints=MAX_KP, detection_threshold=0.0).eval().to(device),
            "aliked": ALIKED(max_num_keypoints=MAX_KP, detection_threshold=0.0).eval().to(device),
            "disk": DISK(max_num_keypoints=MAX_KP).eval().to(device),
        }
        self.lg = {k: LightGlue(features={"sp": "superpoint"}.get(k, k)).eval().to(device) for k in self.nets}

    @torch.no_grad()
    def extract(self, method, bgr):
        if method == "orb":
            kps, desc = self.orb.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
            pts = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
            return {"kp": pts, "desc": desc if desc is not None else np.zeros((0, 32), np.uint8)}
        net = method.split("_")[0]
        img = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float().div(255).to(self.dev)
        f = self.nets[net].extract(img)
        return {"t": f, "kp": f["keypoints"][0].cpu().numpy()}

    @torch.no_grad()
    def match(self, method, a, b):
        if method == "orb":
            if len(a["desc"]) < 2 or len(b["desc"]) < 2:
                return np.zeros((0, 2), int)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            fwd = bf.knnMatch(a["desc"], b["desc"], k=2)
            bwd = {m.queryIdx: m.trainIdx for m in bf.match(b["desc"], a["desc"])}
            out = [(m.queryIdx, m.trainIdx) for m, n in (p for p in fwd if len(p) == 2)
                   if m.distance < 0.8 * n.distance and bwd.get(m.trainIdx) == m.queryIdx and m.distance <= 100]
            return np.array(out, int).reshape(-1, 2)
        if method.endswith("_nn"):
            da, db = a["t"]["descriptors"][0], b["t"]["descriptors"][0]
            if len(da) < 2 or len(db) < 2:
                return np.zeros((0, 2), int)
            d = torch.cdist(da, db)
            v, j = d.topk(2, dim=1, largest=False)
            back = d.argmin(0)
            i = torch.arange(len(da), device=d.device)
            ok = (v[:, 0] < 0.9 * v[:, 1]) & (back[j[:, 0]] == i)
            return torch.stack([i[ok], j[ok, 0]], 1).cpu().numpy()
        net = method.split("_")[0]
        r = self.lg[net]({"image0": a["t"], "image1": b["t"]})
        return r["matches"][0].cpu().numpy()


def pose_error(T_est, T_gt):
    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])), float(ang)


def localize(kp_ref, kp_q, matches, depth_ref, T_ref, K):
    """PnP from map 3D points (ref depth) to query pixels.

    Returns (T_world_query or None, n_inliers, indices into `matches` of the PnP inliers)."""
    none = np.zeros(0, int)
    if len(matches) < 6:
        return None, 0, none
    u = kp_ref[matches[:, 0]]
    ui = np.clip(np.round(u).astype(int), 0, [depth_ref.shape[1] - 1, depth_ref.shape[0] - 1])
    z = depth_ref[ui[:, 1], ui[:, 0]].astype(np.float64) / 1000.0
    ok = (z > 0.1) & (z < 60.0)
    if ok.sum() < 6:
        return None, 0, none
    rays = np.linalg.solve(K, np.c_[u[ok], np.ones(ok.sum())].T).T
    Xc = rays * z[ok, None]
    Xw = (T_ref[:3, :3] @ Xc.T).T + T_ref[:3, 3]
    q = kp_q[matches[ok, 1]].astype(np.float64)
    good, rvec, tvec, inl = cv2.solvePnPRansac(Xw, q, K, None, iterationsCount=2000, reprojectionError=3.0,
                                               confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not good or inl is None or len(inl) < 6:
        return None, 0 if inl is None else len(inl), none
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(Xw[inl], q[inl], K, None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = (-R.T @ tvec).ravel()
    return T, len(inl), np.flatnonzero(ok)[inl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default="/storage/ice1/3/6/bkuang7/omnilrs/renders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=10, help="map frame spacing")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 8, 15, 30], help="query-map frame offsets")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ex = Extractors(dev)

    roots = {c: os.path.join(a.renders, f"lunaryard_aerial_{c}") for c in CONDITIONS}
    names, T, K = load_seq(roots["nominal"])
    n = len(names)
    refs = [i for i in range(0, n - max(a.offsets), a.every)]
    needed = sorted({i + k for i in refs for k in a.offsets} | set(refs))
    rows = []
    for method in a.methods:
        t0 = time.time()
        feats, t_ext = {}, []
        for c in CONDITIONS:
            for i in needed:
                img = cv2.imread(os.path.join(roots[c], names[i]))
                s = time.time()
                feats[c, i] = ex.extract(method, img)
                t_ext.append(time.time() - s)
        depth = {(c, i): cv2.imread(os.path.join(roots[c], "depth0", os.path.basename(names[i])),
                                    cv2.IMREAD_UNCHANGED) for c in CONDITIONS for i in refs}
        t_match = []
        for (cr, cq), k, i in itertools.product(itertools.product(CONDITIONS, CONDITIONS), a.offsets, refs):
            if cr == cq and k == 0:
                continue  # identical image
            fr, fq = feats[cr, i], feats[cq, i + k]
            s = time.time()
            m = ex.match(method, fr, fq)
            t_match.append(time.time() - s)
            Te, ninl, _ = localize(fr["kp"], fq["kp"], m, depth[cr, i], T[i], K)
            et, er = pose_error(Te, T[i + k]) if Te is not None else (np.inf, np.inf)
            rows.append({"method": method, "map": cr, "query": cq, "offset": k, "frame": i, "n_kp_map": len(fr["kp"]),
                         "n_kp_query": len(fq["kp"]), "n_matches": len(m), "n_inliers": ninl, "err_t_m": et,
                         "err_r_deg": er})
        print(f"{method}: {time.time() - t0:.0f}s extract {1e3 * np.mean(t_ext):.1f} ms "
              f"match {1e3 * np.mean(t_match):.1f} ms", flush=True)
        json.dump({"extract_ms": 1e3 * float(np.mean(t_ext)), "match_ms": 1e3 * float(np.mean(t_match))},
                  open(os.path.join(a.out, f"timing_{method}.json"), "w"))
    import csv

    with open(os.path.join(a.out, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", len(rows), "pairs")


if __name__ == "__main__":
    main()
