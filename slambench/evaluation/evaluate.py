"""Evaluate one run directory against its dataset's ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from ..algorithms.base import TRAJECTORY_KINDS, TrajectoryOutput, trajectories_from_output_json
from ..trajectory import Trajectory, read_tum
from ..types import SensorMode
from .metrics import (DEFAULT_EVAL, FrameDiagnostics, coverage_from_states, coverage_proxy_from_output,
                      evaluate_sessions, timing_metrics, unavailable, write_json)
from .plots import plot_all


def _input_timestamps(run_dir: Path) -> np.ndarray:
    prep = json.loads((run_dir / "prepared_input.json").read_text())
    frames_csv = Path(prep["directory"]) / "frames.csv"
    return np.loadtxt(frames_csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)


def _evaluate_kind(t: TrajectoryOutput, dataset, gt, ecfg, diag, input_ts, mode):
    """Accuracy of one declared trajectory; returns (metrics, gt positions in its sensor frame)."""
    meta = t.to_dict()
    if not t.path.exists():
        return {"available": False, "reason": f"trajectory file missing: {t.path}", "trajectory": meta}, None
    traj = read_tum(t.path)
    if len(traj) == 0:
        return {**unavailable("trajectory file is empty"), "trajectory": meta}, None
    if gt is None:
        return {**unavailable("dataset provides no ground truth for this sequence"), "trajectory": meta}, None
    T_bs = dataset.T_gt_body_sensor(t.frame)
    gt_traj = Trajectory(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, gt.body_frame)
    if T_bs is None:
        return {**unavailable(f"transform from GT body {gt.body_frame!r} to trajectory frame {t.frame!r} "
                              "unknown"), "trajectory": meta}, None
    if not gt.has_orientation and not np.allclose(T_bs[:3, 3], 0):
        return {**unavailable("position-only ground truth with a non-zero lever arm to the trajectory "
                              "frame cannot be expressed in that frame"), "trajectory": meta}, None
    gt_sensor = gt_traj.transformed(T_right=T_bs) if gt.has_orientation else gt_traj
    acc = evaluate_sessions(traj, gt_sensor, ecfg, t.coordinate_sessions, diag, input_ts,
                            mode.is_scale_ambiguous)
    acc["trajectory"] = meta
    acc["ground_truth"] = {"source": gt.source, "reference_frame": gt.reference_frame,
                           "body_frame": gt.body_frame, "has_orientation": gt.has_orientation,
                           "T_gt_body_trajectory_frame": T_bs.tolist(), "notes": gt.notes}
    return acc, gt_sensor.positions


def _map_reuse(run_dir, cfg, dataset, gt, ecfg, status, mr_info, input_ts, out_dir):
    """Fixed-frame map-reuse metrics of a localization run (see map_reuse.py)."""
    from .map_reuse import fixed_alignment_from, map_reuse_metrics
    if gt is None:
        return unavailable("dataset provides no ground truth for this sequence")
    try:
        fixed = fixed_alignment_from(Path(mr_info["map_run"]))
    except (OSError, ValueError, KeyError) as exc:
        return unavailable(f"no fixed map->GT transform: {exc}")
    T_bs = dataset.T_gt_body_sensor("cam0")
    if T_bs is None or not gt.has_orientation:
        return unavailable("ground truth cannot be expressed in the cam0 frame")
    if fixed.get("T_gt_body_trajectory_frame") is not None and \
            not np.allclose(fixed["T_gt_body_trajectory_frame"], T_bs, atol=1e-9):
        return unavailable("GT body->cam0 transform differs between the mapping run and this run")
    gt_sensor = Trajectory(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, gt.body_frame).transformed(T_right=T_bs)
    algo = cfg["algorithm"]["name"]
    prior_note = None
    if algo in ("orbslam3", "spslam3"):
        from ..algorithms import orbslam3_map_reuse as omr
        from ..algorithms.orbslam3 import OrbSlam3Adapter
        log_rows, _ = OrbSlam3Adapter.read_frame_log(run_dir / "algorithm" / "frame_log.csv")
        load = omr.read_loaded_map(run_dir / "algorithm")
        if not load.get("T_final_map"):
            return unavailable("no loaded_map.json (atlas not loaded?)")
        rows = omr.map_reuse_rows(log_rows, np.array(load["T_final_map"]), mr_info.get("reuse_mode", "relocalize"))
    else:
        from ..algorithms.rtabmap import RtabmapAdapter
        rows, _ = RtabmapAdapter.read_frame_log(run_dir / "algorithm" / "frame_log.csv")
        prior_note = ("before the first localization: RTAB-Map's prior (last pose saved in the map + "
                      "odometry); not counted as localized")
    if not rows:
        return unavailable("no frame log")
    return map_reuse_metrics(rows, input_ts, gt_sensor, fixed, ecfg.get("map_reuse"), status, mr_info,
                             ecfg["association_max_dt_s"], out_dir, prior_note=prior_note)


def evaluate_run(run_dir: Path, dataset=None, eval_cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Compute metrics for `run_dir`, write eval/metrics.json and plots, return the metrics.

    Every trajectory the run declared (online and/or final) is evaluated separately in its own
    sensor frame; failed runs are evaluated too (e.g. the online prefix before a crash).
    """
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
    trajs = {t.kind: t for t in trajectories_from_output_json(algo_out)}
    input_ts = _input_timestamps(run_dir)
    diag = None
    if algo_out.get("frame_diagnostics") and Path(algo_out["frame_diagnostics"]).exists():
        diag = FrameDiagnostics.read(Path(algo_out["frame_diagnostics"]))

    m: Dict[str, Any] = {
        "run_id": status.get("run_id"), "run_status": status.get("status"),
        "dataset": cfg["dataset"]["name"], "sequence": cfg["dataset"]["sequence"],
        "algorithm": cfg["algorithm"]["name"], "mode": mode.value,
        "loop_closure": (cfg["algorithm"].get("settings") or {}).get("loop_closure"),
        "trajectories": {k: t.to_dict() for k, t in trajs.items()},
        "settings": {k: ecfg[k] for k in ("alignment", "association_max_dt_s", "segment_max_gap_s",
                                           "rpe", "valid_states", "session_min_matched")},
    }
    if mode.is_scale_ambiguous and ecfg["alignment"] == "se3":
        m["settings"]["note"] = "monocular run evaluated with SE(3); set alignment: sim3 for scale correction"

    # Coverage (from tracking states; otherwise a labelled proxy from whatever output exists)
    if diag is not None:
        m["coverage"] = coverage_from_states(diag, input_ts)
    else:
        any_t = trajs.get("online") or trajs.get("final")
        if any_t is not None and any_t.path.exists():
            m["coverage"] = coverage_proxy_from_output(read_tum(any_t.path), input_ts, any_t.keyframe_only)
        else:
            m["coverage"] = unavailable("no tracking states and no trajectory output")

    # Accuracy, per trajectory kind
    gt = dataset.ground_truth()
    m["accuracy"], gt_pos = {}, {}
    for kind in TRAJECTORY_KINDS:
        t = trajs.get(kind)
        if t is None:
            m["accuracy"][kind] = unavailable(f"run produced no {kind} trajectory "
                                              f"(run status: {status.get('status')})")
            continue
        m["accuracy"][kind], gt_pos[kind] = _evaluate_kind(t, dataset, gt, ecfg, diag, input_ts, mode)

    mr_info = (algo_out.get("extra") or {}).get("map_reuse")
    if mr_info is not None:
        m["map_reuse"] = _map_reuse(run_dir, cfg, dataset, gt, ecfg, status, mr_info, input_ts, out_dir)

    playback = (cfg["algorithm"].get("settings") or {}).get("playback_rate")
    m["timing"] = timing_metrics(diag, input_ts, status, playback)

    if ecfg.get("plots"):
        title = f"{m['algorithm']} {m['mode']} | {m['dataset']}/{m['sequence']} | {m['run_id']}"
        m["plots"] = plot_all(out_dir, m["accuracy"], gt_pos, diag, input_ts, title)
    write_json(out_dir / "metrics.json", m)
    return m
