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


def plot_all(out_dir: Path, acc: Dict[str, Any], gt_positions_sensor: Optional[np.ndarray],
             diag: Optional[FrameDiagnostics], input_ts: np.ndarray, title: str) -> Dict[str, str]:
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    t0 = int(input_ts[0]) if len(input_ts) else 0
    s = acc.get("_series") if acc else None
    label = acc.get("alignment", {}).get("label", "") if acc else ""

    if s is not None:
        est, gt, seg = s["est_pos"], s["gt_pos"], s["segment"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (a, b, names) in zip(axes, [(0, 1, "xy"), (0, 2, "xz")]):
            if gt_positions_sensor is not None:
                ax.plot(gt_positions_sensor[:, a], gt_positions_sensor[:, b], color="0.8", lw=1,
                        label="ground truth (all)")
            ax.plot(gt[:, a], gt[:, b], ".", color="k", ms=1.5, label="ground truth (matched)")
            for k, sid in enumerate(np.unique(seg)):
                m = seg == sid
                ax.plot(est[m, a], est[m, b], "-", color="tab:red", lw=1,
                        label=f"estimate ({label})" if k == 0 else None)
            ax.set_xlabel(f"{names[0]} [m]")
            ax.set_ylabel(f"{names[1]} [m]")
            ax.axis("equal")
            ax.grid(alpha=0.3)
        axes[0].legend(loc="best", fontsize=8)
        ate = acc["ate"]["translation_m"]
        fig.suptitle(f"{title}\nATE RMSE {ate['rmse']:.3f} m over {ate['n']} matched samples "
                     f"(segments drawn separately; gaps not interpolated)")
        fig.tight_layout()
        p = out_dir / "trajectory.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files["trajectory"] = p.name

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    if s is not None:
        axes[0].plot((s["t_ns"] - t0) * 1e-9, s["trans_err"], ".", ms=2, color="tab:red")
        axes[0].set_ylabel("ATE translation [m]")
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
