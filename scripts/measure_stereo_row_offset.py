#!/usr/bin/env python3
"""Measure the vertical (row) misalignment of a prepared rectified stereo input.

KLT-tracks corners from the left to the right image of every N-th frame (forward-backward checked)
and fits dy = y_left - y_right = a + b (x - cx) + c (y - cy). For correctly rectified images a, b, c
are ~0. A constant a with negligible b, c is a pure row shift that can be undone with a dataset
option such as LunaPolaris ``right_row_shift_px: a``.

Usage: measure_stereo_row_offset.py <prepared_stereo_dir> [--every 400]
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_dir", type=Path)
    ap.add_argument("--every", type=int, default=400)
    a = ap.parse_args()
    rows = list(csv.reader(open(a.prepared_dir / "frames.csv")))[1:]
    X, Y, DY = [], [], []
    shape = None
    for r in rows[:: a.every]:
        L = cv2.imread(str(a.prepared_dir / r[1]), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(a.prepared_dir / r[2]), cv2.IMREAD_GRAYSCALE)
        shape = L.shape
        p = cv2.goodFeaturesToTrack(L, 800, 0.005, 8)
        if p is None:
            continue
        q, st, _ = cv2.calcOpticalFlowPyrLK(L, R, p, None, winSize=(21, 21), maxLevel=3)
        b, st2, _ = cv2.calcOpticalFlowPyrLK(R, L, q, None, winSize=(21, 21), maxLevel=3)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(b - p, axis=2)[:, 0] < 0.3)
        p, q = p[ok, 0], q[ok, 0]
        dx = p[:, 0] - q[:, 0]
        k = (dx > 1) & (dx < 120)
        X += list(p[k, 0]); Y += list(p[k, 1]); DY += list(p[k, 1] - q[k, 1])
    X, Y, DY = map(np.array, (X, Y, DY))
    if len(DY) < 50:
        raise SystemExit(f"only {len(DY)} stereo correspondences")
    cy, cx = shape[0] / 2, shape[1] / 2
    A = np.c_[np.ones_like(X), X - cx, Y - cy]
    inl = np.abs(DY - np.median(DY)) < 5
    for _ in range(10):  # iteratively reweighted: drop mismatches (> 3 MAD, at least 0.5 px)
        c, *_ = np.linalg.lstsq(A[inl], DY[inl], rcond=None)
        r = DY - A @ c
        thr = max(0.5, 3 * 1.4826 * np.median(np.abs(r[inl])))
        new = np.abs(r) < thr
        if (new == inl).all():
            break
        inl = new
    print(f"{len(DY)} correspondences from {len(rows[:: a.every])} frames, {inl.mean():.2f} inliers")
    print(f"dy = y_left - y_right = {c[0]:.3f} + {c[1]:.5f} (x - {cx:g}) + {c[2]:.5f} (y - {cy:g})")
    print(f"median dy {np.median(DY):.3f} px, inlier residual RMS {np.sqrt(np.mean(r[inl] ** 2)):.3f} px")
    print(f"max |b, c| contribution over the image: {abs(c[1]) * cx + abs(c[2]) * cy:.3f} px")


if __name__ == "__main__":
    main()
