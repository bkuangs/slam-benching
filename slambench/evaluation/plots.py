"""Diagnostic plots (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .metrics import FrameDiagnostics


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _lost_spans(diag: FrameDiagnostics, t0_ns: int):
    spans, start = [], None
    for t, ok in zip(diag.timestamps_ns, diag.tracking_ok):
        if not ok and start is None:
            start = t
        elif ok and start is not None:
            spans.append(((start - t0_ns) * 1e-9, (t - t0_ns) * 1e-9))
            start = None
    if start is not None:
        spans.append(((start - t0_ns) * 1e-9, (diag.timestamps_ns[-1] - t0_ns) * 1e-9))
    return spans


COLORS = {"final": "tab:red", "online": "tab:blue"}
SESSION_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]


def _trajectory_plot(plt, path: Path, kind: str, acc: Dict[str, Any], gt_all: Optional[np.ndarray],
                     title: str) -> None:
    s = acc["_series"]
    est, gt, seg = s["est_pos"], s["gt_pos"], s["segment"]
    sess = s.get("session", np.zeros(len(seg), int))
    multi = len(np.unique(sess)) > 1
    per_session = acc.get("alignment", {}).get("per_session", [])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (a, b, names) in zip(axes, [(0, 1, "xy"), (0, 2, "xz")]):
        if gt_all is not None:
            ax.plot(gt_all[:, a], gt_all[:, b], color="0.8", lw=1, label="ground truth (all)")
        ax.plot(gt[:, a], gt[:, b], ".", color="k", ms=1.5, label="ground truth (matched)")
        labelled = set()
        for sid in np.unique(seg):
            m = seg == sid
            k = int(sess[m][0])
            color = SESSION_COLORS[k % len(SESSION_COLORS)] if multi else COLORS[kind]
            lab = None
            if k not in labelled:
                mid = per_session[k]["map_id"] if k < len(per_session) else k
                lab = f"{kind} estimate, map {mid}" if multi else f"{kind} estimate ({acc['alignment']['label']})"
                labelled.add(k)
            ax.plot(est[m, a], est[m, b], "-", color=color, lw=1, label=lab)
        ax.set_xlabel(f"{names[0]} [m]")
        ax.set_ylabel(f"{names[1]} [m]")
        ax.axis("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    ate = acc["ate"]["translation_m"]
    fig.suptitle(f"{title}\n{kind} trajectory: ATE RMSE {ate['rmse']:.3f} m over {ate['n']} matched samples "
                 f"({acc.get('n_sessions_pooled', 1)} map session(s), each aligned separately; "
                 "gaps not interpolated)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_all(out_dir: Path, accs: Dict[str, Dict[str, Any]], gt_positions_sensor: Dict[str, np.ndarray],
             diag: Optional[FrameDiagnostics], input_ts: np.ndarray, title: str) -> Dict[str, str]:
    """trajectory[_online].png per evaluated trajectory kind, plus error_and_tracking.png."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    t0 = int(input_ts[0]) if len(input_ts) else 0
    series = {k: a for k, a in (accs or {}).items() if isinstance(a, dict) and a.get("_series") is not None}

    for kind, acc in series.items():
        p = out_dir / ("trajectory.png" if kind == "final" else f"trajectory_{kind}.png")
        _trajectory_plot(plt, p, kind, acc, gt_positions_sensor.get(kind), title)
        files[f"trajectory_{kind}"] = p.name

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    if series:
        for kind, acc in series.items():
            s = acc["_series"]
            axes[0].plot((s["t_ns"] - t0) * 1e-9, s["trans_err"], ".", ms=2, color=COLORS.get(kind, "k"),
                         label=f"{kind} (RMSE {acc['ate']['translation_m']['rmse']:.3f} m)")
        axes[0].set_ylabel("ATE translation [m]")
        axes[0].legend(loc="upper left", fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "accuracy unavailable", ha="center", transform=axes[0].transAxes)
    if diag is not None:
        for a, b in _lost_spans(diag, t0):
            for ax in axes:
                ax.axvspan(a, b, color="tab:orange", alpha=0.25, lw=0)
        if not np.all(np.isnan(diag.track_time_s)):
            axes[1].plot((diag.timestamps_ns - t0) * 1e-9, diag.track_time_s * 1e3, lw=0.6)
            if len(input_ts) > 1:
                axes[1].axhline(np.median(np.diff(input_ts)) * 1e-6, color="k", ls="--", lw=0.8,
                                label="frame period")
                axes[1].legend(loc="upper right", fontsize=8)
            axes[1].set_ylabel("tracking time [ms]")
    axes[1].set_xlabel("time since first input frame [s]  (orange: estimator not tracking OK)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    p = out_dir / "error_and_tracking.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files["error_and_tracking"] = p.name
    return files
