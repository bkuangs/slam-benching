#!/usr/bin/env python3
"""Regenerate the figures shown in README.md (writes docs/figures/).

Needs the prepared LunaPolaris stereo inputs, the VPR session poses and the Lunaryard aerial renders
on scratch. Numbers in the relocalization chart come from docs/research_summary.md (claims 5 and 10).
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

SCRATCH = Path("/storage/ice1/3/6/bkuang7")
SESSIONS = SCRATCH / "slam-bench-work/vpr/sessions"
RENDERS = SCRATCH / "omnilrs/renders"
OUT = Path(__file__).resolve().parents[1] / "docs/figures"


def label(img, text):
    d = ImageDraw.Draw(img)
    w = 8 * len(text) + 12
    d.rectangle([0, 0, w, 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 0))
    return img


def session_frame(name, i=None, near=None):
    s = json.loads((SESSIONS / name / "session.json").read_text())
    p = np.load(SESSIONS / name / "poses.npz")
    T, valid, t = p["T"], p["valid"], p["t_ns"]
    idx = np.flatnonzero(valid)
    if near is None:
        k = idx[len(idx) // 2] if i is None else idx[i]
    else:
        pos, fwd = near[:3, 3], near[:3, 2]
        cost = np.linalg.norm(T[idx, :3, 3] - pos, axis=1) + 2.0 * (1 - T[idx, :3, 2] @ fwd)
        k = idx[np.argmin(cost)]
    img = Image.open(Path(s["image_root"]) / "cam0" / f"{t[k]}.png").convert("RGB")
    return img, T[k]


def cross_day():
    a, Ta = session_frame("real_seq11", i=4000)
    b, _ = session_frame("real_seq01", near=Ta)
    out = Image.new("RGB", (a.width * 2 + 8, a.height), "white")
    out.paste(label(b, "seq01 (map session)"), (0, 0))
    out.paste(label(a, "seq11 (other day, same place)"), (a.width + 8, 0))
    out.save(OUT / "lunapolaris_cross_day.jpg", quality=85)


def lighting_strip(frame=900, w=480):
    tiles = []
    for name in ["nominal", "low_sun", "dim", "glare"]:
        files = sorted(glob.glob(str(RENDERS / f"lunaryard_aerial_{name}" / "cam0" / "*.png")))
        im = Image.open(files[frame]).convert("RGB")
        im = im.resize((w, int(im.height * w / im.width)))
        tiles.append(label(im, name))
    out = Image.new("RGB", (w * 4 + 6 * 3, tiles[0].height), "white")
    for j, im in enumerate(tiles):
        out.paste(im, (j * (w + 6), 0))
    out.save(OUT / "lunaryard_lighting.jpg", quality=85)


def reloc_chart():
    cases = ["sim\nnominal→low_sun", "sim\nnominal→glare", "real\nseq01→seq11", "real\nseq11→seq01"]
    stock = [0, 0, 0, 0]
    learned = [99.9, 98, 90, 93]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)
    ax.bar(x - 0.2, stock, 0.4, label="stock DBoW relocalization (ORB-SLAM3 / SP_SLAM3)", color="#bbbbbb")
    b = ax.bar(x + 0.2, learned, 0.4, label="learned: MixVPR retrieval + LightGlue verification", color="#2a7ab9")
    ax.bar_label(b, fmt="%g%%", fontsize=8)
    for xi in x:
        ax.text(xi - 0.2, 2, "0%", ha="center", fontsize=8)
    ax.set_xticks(x, cases, fontsize=8)
    ax.set_ylabel("frames localized in the\nsaved map (%)", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_title("Map reuse after a lighting change (0 false relocalizations)", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.28))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "map_reuse_relocalization.png", bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cross_day()
    lighting_strip()
    reloc_chart()
