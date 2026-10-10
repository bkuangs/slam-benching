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


@dataclass
class AlgorithmOutput:
    status: RunStatus
    message: str = ""
    trajectory: Optional[Path] = None          # normalized TUM file (see slambench.trajectory)
    estimate_frame: str = ""                   # logical sensor frame of trajectory poses
    frame_diagnostics: Optional[Path] = None   # normalized CSV, FRAME_DIAGNOSTICS_COLUMNS
    diagnostics_available: Dict[str, bool] = field(default_factory=dict)
    keyframe_only: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


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
        """Logical sensor frame ("cam0", "imu", ...) of the normalized trajectory poses."""

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
