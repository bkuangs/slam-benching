"""Evaluate one run directory against its dataset's ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from ..trajectory import Trajectory, read_tum
from ..types import SensorMode
from .metrics import (DEFAULT_EVAL, FrameDiagnostics, coverage_from_states, coverage_proxy_from_output,
                      evaluate_trajectory, timing_metrics, unavailable, write_json)
from .plots import plot_all


def _input_timestamps(run_dir: Path) -> np.ndarray:
    prep = json.loads((run_dir / "prepared_input.json").read_text())
    frames_csv = Path(prep["directory"]) / "frames.csv"
    return np.loadtxt(frames_csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)


def evaluate_run(run_dir: Path, dataset=None, eval_cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Compute metrics for `run_dir`, write eval/metrics.json and plots, return the metrics."""
    from ..runner import make_dataset  # local import to avoid a cycle

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    ecfg = dict(DEFAULT_EVAL)
    ecfg.update(cfg.get("evaluation") or {})
    ecfg.update(eval_cfg or {})
    mode = SensorMode(cfg["algorithm"]["mode"])
    dataset = dataset or make_dataset(cfg)
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)

    algo_out = json.loads((run_dir / "algorithm_output.json").read_text())
    input_ts = _input_timestamps(run_dir)
    diag = None
    if algo_out.get("frame_diagnostics") and Path(algo_out["frame_diagnostics"]).exists():
        diag = FrameDiagnostics.read(Path(algo_out["frame_diagnostics"]))

    m: Dict[str, Any] = {
        "run_id": status.get("run_id"), "run_status": status.get("status"),
        "dataset": cfg["dataset"]["name"], "sequence": cfg["dataset"]["sequence"],
        "algorithm": cfg["algorithm"]["name"], "mode": mode.value,
        "loop_closure": (cfg["algorithm"].get("settings") or {}).get("loop_closure"),
        "estimate_frame": algo_out.get("estimate_frame"),
        "settings": {k: ecfg[k] for k in ("alignment", "association_max_dt_s", "segment_max_gap_s",
                                           "rpe", "valid_states")},
    }
    if mode.is_scale_ambiguous and ecfg["alignment"] == "se3":
        m["settings"]["note"] = "monocular run evaluated with SE(3); set alignment: sim3 for scale correction"

    # Coverage
    traj = None
    if algo_out.get("trajectory") and Path(algo_out["trajectory"]).exists():
        traj = read_tum(Path(algo_out["trajectory"]))
    if diag is not None:
        m["coverage"] = coverage_from_states(diag, input_ts)
    elif traj is not None:
        m["coverage"] = coverage_proxy_from_output(traj, input_ts, bool(algo_out.get("keyframe_only")))
    else:
        m["coverage"] = unavailable("no tracking states and no trajectory output")

    # Accuracy
    gt = dataset.ground_truth()
    acc = None
    gt_sensor_pos = None
    if traj is None:
        m["accuracy"] = unavailable(f"no trajectory output (run status: {status.get('status')})")
    elif gt is None:
        m["accuracy"] = unavailable("dataset provides no ground truth for this sequence")
    else:
        T_bs = dataset.T_gt_body_sensor(algo_out["estimate_frame"])
        gt_traj = Trajectory(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, gt.body_frame)
        if T_bs is None:
            m["accuracy"] = unavailable(f"transform from GT body {gt.body_frame!r} to estimate frame "
                                        f"{algo_out['estimate_frame']!r} unknown")
        elif not gt.has_orientation and not np.allclose(T_bs[:3, 3], 0):
            m["accuracy"] = unavailable("position-only ground truth with a non-zero lever arm to the "
                                        "estimate frame cannot be expressed in the estimate frame")
        else:
            gt_sensor = gt_traj.transformed(T_right=T_bs) if gt.has_orientation else gt_traj
            gt_sensor_pos = gt_sensor.positions
            acc = evaluate_trajectory(traj, gt_sensor, ecfg, diag, input_ts, mode.is_scale_ambiguous)
            acc["ground_truth"] = {"source": gt.source, "reference_frame": gt.reference_frame,
                                   "body_frame": gt.body_frame, "has_orientation": gt.has_orientation,
                                   "T_gt_body_estimate_frame": T_bs.tolist(), "notes": gt.notes}
            m["accuracy"] = acc

    playback = (cfg["algorithm"].get("settings") or {}).get("playback_rate")
    m["timing"] = timing_metrics(diag, input_ts, status, playback)

    if ecfg.get("plots"):
        title = f"{m['algorithm']} {m['mode']} | {m['dataset']}/{m['sequence']} | {m['run_id']}"
        m["plots"] = plot_all(out_dir, acc, gt_sensor_pos, diag, input_ts, title)
    write_json(out_dir / "metrics.json", m)
    return m
