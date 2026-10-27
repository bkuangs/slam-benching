"""Replay video for an aerial Lunaryard sweep: camera views for every lighting condition plus a top-down map
with ground truth and SE(3)-aligned estimates, drawn progressively.

    python sim/omnilrs/replay_video.py --lighting glare --out replay_glare.mp4

Needs opencv and imageio-ffmpeg (pip install imageio-ffmpeg) in the slambench venv.
"""
import argparse
import glob
import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

CONDITIONS = ["nominal", "low_sun", "dim", "glare"]
METHODS = [("orbslam3_stereo", "ORB-SLAM3", (60, 180, 255)), ("spslam3_stereo", "SP_SLAM3", (80, 220, 80)),
           ("lac_stereo_kfvo", "LAC kfvo", (230, 120, 230))]
GT_COLOR = (255, 255, 255)


def read_tum(path):
    a = np.loadtxt(path)
    return np.round(a[:, 0] * 1e9).astype(np.int64), a[:, 1:4]


def align_se3(est, gt):
    """Rigid (no scale) Umeyama alignment of est positions onto gt."""
    me, mg = est.mean(0), gt.mean(0)
    U, _, Vt = np.linalg.svd((gt - mg).T @ (est - me))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return (R @ (est - me).T).T + mg


def latest_run(runs, name):
    d = sorted(glob.glob(os.path.join(runs, name, "*", "trajectory.tum")))
    return d[-1] if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lighting", default="glare", choices=CONDITIONS, help="condition whose estimates are mapped")
    ap.add_argument("--out", required=True)
    ap.add_argument("--renders", default="/storage/ice1/3/6/bkuang7/omnilrs/renders")
    ap.add_argument("--runs", default="/storage/ice1/3/6/bkuang7/slam-bench-work/runs")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--step", type=int, default=1, help="use every Nth frame (speeds up playback)")
    a = ap.parse_args()

    root = os.path.join(a.renders, f"lunaryard_aerial_{a.lighting}")
    t_gt, p_gt = read_tum(os.path.join(root, "gt_cam0.tum"))
    tracks = []
    for key, label, color in METHODS:
        f = latest_run(a.runs, f"omnilrs_lunaryard_aerial_{a.lighting}_{key}")
        if f is None:
            print("missing run", key)
            continue
        t, p = read_tum(f)
        # TUM seconds at ~1.8e9 lose sub-microsecond precision, so match to the nearest stamp within 1 ms.
        j = np.clip(np.searchsorted(t, t_gt), 1, len(t) - 1)
        j = np.where(np.abs(t[j - 1] - t_gt) < np.abs(t[j] - t_gt), j - 1, j)
        m = np.abs(t[j] - t_gt) < 1_000_000
        est = np.full_like(p_gt, np.nan)
        est[m] = p[j[m]]
        est[m] = align_se3(est[m], p_gt[m])
        err = np.linalg.norm(est - p_gt, axis=1)
        tracks.append((label, color, est, err))
        print(f"{label}: SE3 ATE {np.sqrt(np.nanmean(err ** 2)):.3f} m ({os.path.dirname(f)})")

    # Layout: 4 camera tiles (480x270) on top, map (960x540) + error plot (960x540) below.
    TW, TH, W, H = 480, 270, 1920, 810
    dem = np.load(os.path.join(root, "dem.npy"))
    res = 0.05
    pad = 4.0
    x0, x1 = p_gt[:, 0].min() - pad, p_gt[:, 0].max() + pad
    y0, y1 = p_gt[:, 1].min() - pad, p_gt[:, 1].max() + pad
    scale = min(940 / (x1 - x0), 520 / (y1 - y0))
    mw, mh = int((x1 - x0) * scale), int((y1 - y0) * scale)
    ox, oy = (960 - mw) // 2, 270 + (540 - mh) // 2

    sub = dem[int(y0 / res):int(y1 / res), int(x0 / res):int(x1 / res)]
    gy, gx = np.gradient(sub, res)
    shade = np.clip(0.6 + 0.8 * (-gx * 0.7 - gy * 0.7) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
    shade = cv2.resize((shade * 150).astype(np.uint8), (mw, mh))
    shade = cv2.flip(shade, 0)  # image row 0 = max y
    base = np.zeros((H, W, 3), np.uint8)
    base[oy:oy + mh, ox:ox + mw] = cv2.cvtColor(shade, cv2.COLOR_GRAY2BGR)

    def px(p):
        return np.stack([ox + (p[:, 0] - x0) * scale, oy + mh - (p[:, 1] - y0) * scale], 1)

    # Error plot frame.
    ex0, ey0, ew, eh = 1030, 310, 860, 440
    emax = max(np.nanmax(e) for *_, e in tracks) * 1.1 if tracks else 1.0
    tsec = (t_gt - t_gt[0]) * 1e-9
    cv2.rectangle(base, (ex0, ey0), (ex0 + ew, ey0 + eh), (90, 90, 90), 1)
    for k in range(5):
        v = emax * k / 4
        yy = int(ey0 + eh - v / emax * eh)
        cv2.line(base, (ex0, yy), (ex0 + ew, yy), (50, 50, 50), 1)
        cv2.putText(base, f"{v:.2f}", (ex0 - 55, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(base, "position error vs GT (m), SE(3)-aligned", (ex0, ey0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (230, 230, 230), 1)
    cv2.putText(base, f"top-down, lighting = {a.lighting} (white = GT)", (20, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (230, 230, 230), 1)
    for k, (label, color, _, e) in enumerate(tracks):
        cv2.putText(base, f"{label}  ATE {np.sqrt(np.nanmean(e ** 2)):.3f} m", (ex0 + 10, ey0 + 25 + 22 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    gt_px = px(p_gt)
    est_px = [px(e) for _, _, e, _ in tracks]
    err_px = [np.stack([ex0 + tsec / tsec[-1] * ew, ey0 + eh - np.nan_to_num(e) / emax * eh], 1) for *_, e in tracks]

    names = [os.path.basename(l.split(",")[1]) for l in open(os.path.join(root, "frames.csv")).read().split()[1:]]
    ffmpeg = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                               "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(a.fps / a.step), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out], stdin=subprocess.PIPE)
    for i in range(0, len(t_gt), a.step):
        fr = base.copy()
        for k, c in enumerate(CONDITIONS):
            img = cv2.imread(os.path.join(a.renders, f"lunaryard_aerial_{c}", "cam0", names[i]))
            tile = cv2.resize(img, (TW, TH)) if img is not None else np.zeros((TH, TW, 3), np.uint8)
            if c == a.lighting:
                cv2.rectangle(tile, (1, 1), (TW - 2, TH - 2), (0, 220, 255), 3)
            cv2.putText(tile, c, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            fr[0:TH, k * TW:(k + 1) * TW] = tile
        n = i + 1
        cv2.polylines(fr, [gt_px[:n].astype(np.int32)], False, GT_COLOR, 2, cv2.LINE_AA)
        for (label, color, _, _), ep, rp in zip(tracks, est_px, err_px):
            ok = np.isfinite(ep[:n]).all(1)
            cv2.polylines(fr, [ep[:n][ok].astype(np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.polylines(fr, [rp[:n].astype(np.int32)], False, color, 2, cv2.LINE_AA)
            if ok[-1]:
                cv2.circle(fr, tuple(ep[n - 1].astype(int)), 5, color, -1)
        cv2.circle(fr, tuple(gt_px[n - 1].astype(int)), 6, GT_COLOR, -1)
        cv2.putText(fr, f"t = {tsec[i]:6.1f} s", (ex0 + ew - 150, ey0 + eh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 230, 230), 1)
        ffmpeg.stdin.write(fr.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
