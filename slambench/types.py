"""Shared data types for dataset adapters, algorithm adapters and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class SensorMode(str, Enum):
    MONO = "mono"
    MONO_INERTIAL = "mono_inertial"
    STEREO = "stereo"
    STEREO_INERTIAL = "stereo_inertial"
    LIDAR = "lidar"
    LIDAR_INERTIAL = "lidar_inertial"

    @property
    def is_inertial(self) -> bool:
        return self.value.endswith("_inertial")

    @property
    def is_scale_ambiguous(self) -> bool:
        """Pure monocular is only defined up to scale."""
        return self is SensorMode.MONO


# Logical sensor streams each mode needs. Dataset adapters map these names to
# their own topics/files.
REQUIRED_STREAMS: Dict[SensorMode, set] = {
    SensorMode.MONO: {"cam0"},
    SensorMode.MONO_INERTIAL: {"cam0", "imu"},
    SensorMode.STEREO: {"cam0", "cam1"},
    SensorMode.STEREO_INERTIAL: {"cam0", "cam1", "imu"},
    SensorMode.LIDAR: {"lidar"},
    SensorMode.LIDAR_INERTIAL: {"lidar", "imu"},
}


class RunStatus(str, Enum):
    SUCCESS = "success"
    CONFIG_REJECTED = "config_rejected"
    PREPARE_FAILED = "prepare_failed"
    INIT_FAILURE = "init_failure"
    CRASH = "crash"
    TIMEOUT = "timeout"
    MISSING_OUTPUT = "missing_output"
    EVAL_FAILED = "eval_failed"


@dataclass
class StreamInfo:
    name: str               # logical name: cam0, cam1, imu, lidar
    kind: str               # camera | imu | lidar
    source: str             # topic or file path in the original dataset
    count: int
    rate_hz: float
    first_ns: int
    last_ns: int
    frame_id: str = ""


@dataclass
class CameraIntrinsics:
    model: str                      # "pinhole" | "kannala_brandt"
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: List[float]         # model-specific coefficients
    rectified: bool                 # images already undistorted/rectified
    frame_id: str = ""


@dataclass
class StereoCalibration:
    left: CameraIntrinsics
    right: Optional[CameraIntrinsics]   # None for single-camera datasets (mono modes only)
    T_left_right: np.ndarray        # 4x4, pose of right camera in left camera frame (p_left = T * p_right)
    baseline_m: float
    source: str                     # provenance string

    def to_dict(self) -> dict:
        return {"left": asdict(self.left), "right": asdict(self.right) if self.right else None,
                "T_left_right": np.asarray(self.T_left_right).tolist(),
                "baseline_m": self.baseline_m, "source": self.source}


@dataclass
class ImuNoise:
    """Continuous-time IMU noise model as used by ORB-SLAM3/Kalibr."""
    gyro_noise_density: float       # rad/s/sqrt(Hz)
    accel_noise_density: float      # m/s^2/sqrt(Hz)
    gyro_random_walk: float         # rad/s^2/sqrt(Hz)
    accel_random_walk: float        # m/s^3/sqrt(Hz)
    source: str
    verified: bool = False          # only verified parameters may be used for a run


@dataclass
class ImuCalibration:
    T_cam0_imu: np.ndarray          # 4x4, pose of IMU in cam0 frame (p_cam0 = T * p_imu)
    rate_hz: float
    source: str
    noise: Optional[ImuNoise] = None
    time_offset_s: Optional[float] = None   # t_imu = t_cam + offset; None = unknown
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"T_cam0_imu": np.asarray(self.T_cam0_imu).tolist(), "rate_hz": self.rate_hz,
                "source": self.source, "noise": asdict(self.noise) if self.noise else None,
                "time_offset_s": self.time_offset_s, "notes": self.notes}


@dataclass
class GroundTruth:
    """Ground-truth poses of `body_frame` expressed in `reference_frame`."""
    timestamps_ns: np.ndarray       # int64, sorted, unique
    positions: np.ndarray           # (N,3) metres
    quaternions_xyzw: Optional[np.ndarray]  # (N,4) or None for position-only GT
    reference_frame: str
    body_frame: str
    source: str
    notes: List[str] = field(default_factory=list)

    @property
    def has_orientation(self) -> bool:
        return self.quaternions_xyzw is not None


@dataclass
class SequenceInfo:
    dataset: str
    sequence: str
    path: str
    duration_s: float
    streams: Dict[str, StreamInfo]
    has_ground_truth: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedInput:
    """Estimator-ready offline input written by a dataset adapter.

    Layout (the "slambench offline" format, see docs/adapter_interface.md):
      frames.csv     timestamp_ns,left_path,right_path (paths relative to directory)
      imu.csv        timestamp_ns,wx,wy,wz,ax,ay,az    (inertial modes only)
      manifest.json  provenance, conversions, counts
    """
    directory: Path
    mode: SensorMode
    n_frames: int
    n_imu: int
    manifest: Dict[str, Any]

    @property
    def frames_csv(self) -> Path:
        return self.directory / "frames.csv"

    @property
    def imu_csv(self) -> Path:
        return self.directory / "imu.csv"


class ValidationError(Exception):
    """Raised when a dataset/algorithm/mode combination is not runnable."""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))
