"""Match-visualisation video for the cross-lighting relocalization benchmark.

For each map frame i (lighting A) and query frame i + offset (lighting B), every method's matches are drawn
side by side: PnP inliers in green, rejected matches in red, with the per-frame pose error.

    apptainer exec --nv lac_26c3a2b.sif python sim/omnilrs/cross_lighting_video.py \
        --map nominal --query glare --offset 8 --out reloc_nominal_to_glare.mp4

Needs an ffmpeg binary: --ffmpeg (default: imageio-ffmpeg's from the slambench venv).
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_lighting_reloc import CONDITIONS, METHODS, Extractors, load_seq, localize, pose_error  # noqa: E402

LABELS = {"orb": "ORB", "sp_nn": "SuperPoint + NN", "sp_lg": "SuperPoint + LightGlue",
          "aliked_lg": "ALIKED + LightGlue", "disk_lg": "DISK + LightGlue"}
VENV_FFMPEG = ("/storage/ice1/3/6/bkuang7/slam-bench-work/venv/lib64/python3.9/site-packages/imageio_ffmpeg/"
               "binaries/ffmpeg-linux-x86_64-v7.0.2")
TW, TH, PANEL = 480, 270, 320
OK_T, OK_R = 0.25, 2.0


def draw_row(img_a, img_b, kp_a, kp_b, matches, inl, label, stats, ok):
    sx, sy = TW / img_a.shape[1], TH / img_a.shape[0]
    row = np.zeros((TH, 2 * TW + PANEL, 3), np.uint8)
    row[:, :TW] = cv2.resize(img_a, (TW, TH))
    row[:, TW:2 * TW] = cv2.resize(img_b, (TW, TH))
    is_in = np.zeros(len(matches), bool)
    is_in[inl] = True
    for draw_inliers in (False, True):  # outliers underneath
        for (ia, ib), good in zip(matches, is_in):
            if good != draw_inliers:
                continue
            pa = (int(kp_a[ia, 0] * sx), int(kp_a[ia, 1] * sy))
            pb = (int(kp_b[ib, 0] * sx) + TW, int(kp_b[ib, 1] * sy))
            color = (60, 220, 60) if good else (40, 40, 220)
            cv2.line(row, pa, pb, color, 1, cv2.LINE_AA)
            cv2.circle(row, pa, 2, color, -1)
            cv2.circle(row, pb, 2, color, -1)
    x = 2 * TW + 12
    cv2.putText(row, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(row, "LOCALIZED" if ok else "FAILED", (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (60, 220, 60) if ok else (40, 40, 220), 2, cv2.LINE_AA)
    for k, s in enumerate(stats):
        cv2.putText(row, s, (x, 105 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default="/storage/ice1/3/6/bkuang7/omnilrs/renders")
    ap.add_argument("--map", default="nominal", choices=CONDITIONS)
    ap.add_argument("--query", default="glare", choices=CONDITIONS)
    ap.add_argument("--offset", type=int, default=8)
    ap.add_argument("--step", type=int, default=2, help="map frame spacing")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--ffmpeg", default=VENV_FFMPEG)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ex = Extractors("cuda" if torch.cuda.is_available() else "cpu")
    rm = os.path.join(a.renders, f"lunaryard_aerial_{a.map}")
    rq = os.path.join(a.renders, f"lunaryard_aerial_{a.query}")
    names, T, K = load_seq(rm)
    W, H = 2 * TW + PANEL, 40 + TH * len(a.methods)
    ff = subprocess.Popen([a.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "23", a.out], stdin=subprocess.PIPE)
    tally = {m: [0, 0] for m in a.methods}
    frames = range(0, len(names) - a.offset, a.step)
    for n, i in enumerate(frames):
        img_a = cv2.imread(os.path.join(rm, names[i]))
        img_b = cv2.imread(os.path.join(rq, names[i + a.offset]))
        depth = cv2.imread(os.path.join(rm, "depth0", os.path.basename(names[i])), cv2.IMREAD_UNCHANGED)
        fr = np.zeros((H, W, 3), np.uint8)
        cv2.putText(fr, f"map: {a.map} (left)   query: {a.query} (right), {a.offset * 2 / 15:.1f} m further along"
                    f"   frame {i}", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
        for r, m in enumerate(a.methods):
            fa, fb = ex.extract(m, img_a), ex.extract(m, img_b)
            matches = ex.match(m, fa, fb)
            Te, ninl, inl = localize(fa["kp"], fb["kp"], matches, depth, T[i], K)
            et, er = pose_error(Te, T[i + a.offset]) if Te is not None else (np.inf, np.inf)
            ok = et < OK_T and er < OK_R
            tally[m][0] += ok
            tally[m][1] += 1
            stats = [f"matches {len(matches)}  inliers {ninl}",
                     f"error {100 * et:.1f} cm  {er:.2f} deg" if np.isfinite(et) else "error -",
                     f"running success {100 * tally[m][0] / tally[m][1]:.0f}%"]
            fr[40 + r * TH:40 + (r + 1) * TH] = draw_row(img_a, img_b, fa["kp"], fb["kp"], matches, inl,
                                                         LABELS.get(m, m), stats, ok)
        ff.stdin.write(fr.tobytes())
        if n % 100 == 0:
            print(f"{n}/{len(frames)}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out, {m: f"{100 * s / max(t, 1):.0f}%" for m, (s, t) in tally.items()})


if __name__ == "__main__":
    main()
