"""Dataset adapter interface.

A dataset adapter owns *all* dataset-specific knowledge: file layout, topic
names, timestamp conventions, calibration provenance, ground-truth frames.
The experiment runner and the evaluator only talk to this interface.

See docs/adapter_interface.md for the contract in prose.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..types import (REQUIRED_STREAMS, GroundTruth, ImuCalibration, PreparedInput,
                     SensorMode, SequenceInfo, StereoCalibration)


class DatasetAdapter(ABC):
    #: registry name used in experiment configs (``dataset.name``)
    name: str = ""

    #: set True once the adapter provides LiDAR calibration and input conversion
    supports_lidar: bool = False

    def __init__(self, root: Path, sequence: str, options: Optional[Dict[str, Any]] = None):
        self.root = Path(root).expanduser()
        self.sequence = sequence
        self.options = dict(options or {})

    # ---- metadata -------------------------------------------------------
    @abstractmethod
    def sequence_info(self) -> SequenceInfo:
        """Sequence metadata incl. every available logical stream (cam0, cam1, imu, lidar)."""

    @abstractmethod
    def stereo_calibration(self) -> Optional[StereoCalibration]:
        """Calibration matching the images handed to the estimator, or None."""

    @abstractmethod
    def imu_calibration(self) -> Optional[ImuCalibration]:
        """Camera-IMU extrinsics and noise model, or None if unavailable."""

    @abstractmethod
    def ground_truth(self) -> Optional[GroundTruth]:
        """Ground truth, kept strictly separate from estimator inputs."""

    @abstractmethod
    def T_gt_body_sensor(self, sensor_frame: str) -> Optional[np.ndarray]:
        """4x4 pose of a logical sensor frame ("cam0", "imu") in the GT body frame.

        Used by the evaluator to express ground truth in the same physical frame
        as the estimate. Returns None if the relation is unknown.
        """

    def provenance(self) -> Dict[str, Any]:
        """Dataset-specific provenance to store in run metadata (paths, sizes, checksums...)."""
        return {"root": str(self.root), "sequence": self.sequence}

    # ---- validation -----------------------------------------------------
    def validate(self, mode: SensorMode) -> List[str]:
        """Return a list of problems preventing `mode` from being run on this sequence."""
        problems: List[str] = []
        try:
            info = self.sequence_info()
        except Exception as exc:  # noqa: BLE001 - report any failure to read the dataset
            return [f"cannot read sequence {self.sequence!r} under {self.root}: {exc}"]
        missing = REQUIRED_STREAMS[mode] - set(info.streams)
        if missing:
            problems.append(f"sequence {self.sequence!r} lacks streams {sorted(missing)} "
                            f"required by mode {mode.value!r} (available: {sorted(info.streams)})")
            return problems
        if {"cam0", "cam1"} <= REQUIRED_STREAMS[mode] and self.stereo_calibration() is None:
            problems.append("no stereo calibration available for this sequence")
        if "lidar" in REQUIRED_STREAMS[mode]:
            if not getattr(self, "supports_lidar", False):
                problems.append(f"dataset adapter {self.name!r} does not implement LiDAR calibration/"
                                "conversion yet (LiDAR modes deferred)")
            return problems
        if mode.is_inertial:
            imu = self.imu_calibration()
            if imu is None:
                problems.append("no camera-IMU calibration available")
            elif imu.noise is None:
                problems.append("IMU noise parameters (noise densities and random walks) are not "
                                "available for this dataset; supply verified values via "
                                "dataset.options.imu_noise with a source")
            elif not imu.noise.verified:
                problems.append(f"IMU noise parameters are not verified (source: {imu.noise.source}); "
                                "set dataset.options.imu_noise.verified: true only after checking them")
        return problems

    # ---- input preparation ----------------------------------------------
    @abstractmethod
    def prepare_offline(self, mode: SensorMode, out_dir: Path) -> PreparedInput:
        """Write estimator inputs in the slambench offline format into `out_dir`.

        Must preserve original timestamps, must not include ground truth, and
        must record every conversion in the manifest.
        """

    def preparation_key(self, mode: SensorMode) -> str:
        """Stable key of everything that influences prepare_offline() output (for caching)."""
        payload = {"adapter": self.name, "adapter_version": getattr(self, "adapter_version", "1"),
                   "sequence": self.sequence, "mode": mode.value,
                   "options": {k: v for k, v in sorted(self.options.items())
                               if k in getattr(self, "input_affecting_options", ())}}
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def prepare_cached(self, mode: SensorMode, cache_root: Path) -> PreparedInput:
        """prepare_offline() with a cache keyed by preparation_key()."""
        key = self.preparation_key(mode)
        out = Path(cache_root) / self.name / self.sequence / f"{mode.value}_{key}"
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("complete") and manifest.get("preparation_key") == key:
                return PreparedInput(out, mode, manifest["n_frames"], manifest.get("n_imu", 0), manifest)
        tmp = out.with_name(out.name + ".partial")
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        prepared = self.prepare_offline(mode, tmp)
        prepared.manifest.update({"complete": True, "preparation_key": key})
        (tmp / "manifest.json").write_text(json.dumps(prepared.manifest, indent=2, default=str))
        if out.exists():
            import shutil
            shutil.rmtree(out)
        tmp.rename(out)
        return PreparedInput(out, mode, prepared.n_frames, prepared.n_imu, prepared.manifest)


def write_frames_csv(path: Path, rows) -> int:
    n = 0
    with open(path, "w") as f:
        f.write("timestamp_ns,left_path,right_path\n")
        for t, left, right in rows:
            f.write(f"{int(t)},{left},{right or ''}\n")
            n += 1
    return n


def write_imu_csv(path: Path, t_ns: np.ndarray, gyro: np.ndarray, acc: np.ndarray) -> int:
    with open(path, "w") as f:
        f.write("timestamp_ns,wx,wy,wz,ax,ay,az\n")
        for t, w, a in zip(t_ns, gyro, acc):
            f.write(f"{int(t)},{w[0]:.9g},{w[1]:.9g},{w[2]:.9g},{a[0]:.9g},{a[1]:.9g},{a[2]:.9g}\n")
    return len(t_ns)
