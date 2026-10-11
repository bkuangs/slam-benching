"""Algorithm adapter interface.

An algorithm adapter wraps one external estimator. It declares which sensor
modes it supports, turns dataset calibration + experiment settings into the
estimator's native configuration, builds the exact command, and converts raw
outputs into the normalized trajectory / frame-diagnostics formats.

See docs/adapter_interface.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..datasets.base import DatasetAdapter
from ..types import PreparedInput, RunStatus, SensorMode

#: Columns of the normalized per-frame diagnostics file (frame_diagnostics.csv).
FRAME_DIAGNOSTICS_COLUMNS = [
    "timestamp_ns",        # original frame timestamp
    "tracking_state",      # estimator-native state name, lower case
    "tracking_ok",         # 1 if the estimator reports a valid tracked pose for this frame
    "map_id",              # estimator map/session id ('' if unavailable)
    "track_time_s",        # per-frame processing time of the tracking call ('' if unavailable)
    "in_final_trajectory", # 1 if a pose for this frame is in the normalized trajectory
]


@dataclass
class RunContext:
    run_dir: Path
    algo_dir: Path
    dataset: DatasetAdapter
    mode: SensorMode
    prepared: PreparedInput
    settings: Dict[str, Any]          # resolved algorithm settings


#: Trajectory kinds the evaluator knows about.
TRAJECTORY_KINDS = ("online", "final")


@dataclass
class TrajectoryOutput:
    """One normalized trajectory produced by a run.

    kind
        ``online``: the pose the estimator emitted right after processing each frame (what a
        vehicle would have used at that moment; it may already include earlier loop closures).
        ``final``: poses exported after shutdown, including all later optimization.
    frame
        Logical sensor frame of the poses ("cam0", "imu"); each trajectory may differ.
    coordinate_sessions
        ``single``: all poses share one world frame.
        ``map_id``: each pose is expressed in the frame of the estimator map that was active when
        it was emitted (frame_diagnostics map_id); maps are evaluated as independent sessions.
    """
    kind: str
    path: Path
    frame: str
    description: str = ""
    coordinate_sessions: str = "single"
    keyframe_only: bool = False

    def __post_init__(self):
        if self.kind not in TRAJECTORY_KINDS:
            raise ValueError(f"trajectory kind must be one of {TRAJECTORY_KINDS}, not {self.kind!r}")
        if self.coordinate_sessions not in ("single", "map_id"):
            raise ValueError("coordinate_sessions must be 'single' or 'map_id'")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "path": str(self.path), "frame": self.frame,
                "description": self.description, "coordinate_sessions": self.coordinate_sessions,
                "keyframe_only": self.keyframe_only}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryOutput":
        return cls(d["kind"], Path(d["path"]), d["frame"], d.get("description", ""),
                   d.get("coordinate_sessions", "single"), bool(d.get("keyframe_only", False)))


@dataclass
class AlgorithmOutput:
    status: RunStatus
    message: str = ""
    trajectory: Optional[Path] = None          # legacy: single (final) normalized TUM file
    estimate_frame: str = ""                   # legacy: sensor frame of `trajectory`
    frame_diagnostics: Optional[Path] = None   # normalized CSV, FRAME_DIAGNOSTICS_COLUMNS
    diagnostics_available: Dict[str, bool] = field(default_factory=dict)
    keyframe_only: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    trajectories: List[TrajectoryOutput] = field(default_factory=list)

    def all_trajectories(self) -> List[TrajectoryOutput]:
        """Declared trajectories; a legacy `trajectory` counts as the final trajectory."""
        out = list(self.trajectories)
        if self.trajectory is not None and not any(t.kind == "final" for t in out):
            out.append(TrajectoryOutput("final", Path(self.trajectory), self.estimate_frame,
                                        "estimator output trajectory", "single", self.keyframe_only))
        return out

    def to_dict(self) -> Dict[str, Any]:
        trajs = self.all_trajectories()
        final = next((t for t in trajs if t.kind == "final"), None)
        return {"status": self.status.value, "message": self.message,
                "trajectories": [t.to_dict() for t in trajs],
                # legacy keys (final trajectory) kept for older tools
                "trajectory": str(final.path) if final else None,
                "estimate_frame": final.frame if final else self.estimate_frame,
                "frame_diagnostics": str(self.frame_diagnostics) if self.frame_diagnostics else None,
                "diagnostics_available": self.diagnostics_available,
                "keyframe_only": self.keyframe_only, "extra": self.extra}


def trajectories_from_output_json(d: Dict[str, Any]) -> List[TrajectoryOutput]:
    """Trajectories listed in algorithm_output.json (also reads files written before `trajectories`)."""
    if d.get("trajectories") is not None:
        return [TrajectoryOutput.from_dict(t) for t in d["trajectories"]]
    if d.get("trajectory"):
        return [TrajectoryOutput("final", Path(d["trajectory"]), d.get("estimate_frame", ""),
                                 "estimator output trajectory", "single", bool(d.get("keyframe_only")))]
    return []


class AlgorithmAdapter(ABC):
    #: registry name used in experiment configs (``algorithm.name``)
    name: str = ""
    #: sensor modes this adapter can run
    supported_modes: frozenset = frozenset()

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options = dict(options or {})

    def resolve_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Fill defaults so the saved resolved config fully determines the run."""
        return dict(settings)

    def validate(self, mode: SensorMode, dataset: DatasetAdapter, settings: Dict[str, Any]) -> List[str]:
        """Problems preventing this algorithm from running `mode` on `dataset`."""
        if mode not in self.supported_modes:
            return [f"algorithm {self.name!r} does not support sensor mode {mode.value!r} "
                    f"(supported: {sorted(m.value for m in self.supported_modes)})"]
        return []

    @abstractmethod
    def estimate_frame(self, mode: SensorMode) -> str:
        """Logical sensor frame ("cam0", "imu", ...) of the normalized final trajectory poses."""

    @abstractmethod
    def version_info(self) -> Dict[str, Any]:
        """Pinned versions / commit hashes / image digests of the estimator."""

    @abstractmethod
    def prepare(self, ctx: RunContext) -> None:
        """Write native configuration files into ctx.algo_dir."""

    @abstractmethod
    def command(self, ctx: RunContext) -> List[str]:
        """Exact command line to execute (saved verbatim)."""

    @abstractmethod
    def collect(self, ctx: RunContext, returncode: Optional[int], timed_out: bool) -> AlgorithmOutput:
        """Inspect raw outputs, classify the outcome and write normalized outputs."""
