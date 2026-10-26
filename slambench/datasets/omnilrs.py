"""OmniLRS (Isaac Sim) rendered stereo sequences, written by sim/omnilrs/render_stereo.py.

Layout (one directory per rendered sequence, ``<root>/<sequence>/``):
    cam0/<ns>.png, cam1/<ns>.png   rectified BGR PNG stereo pair (rendered pinhole, zero distortion)
    depth0/<ns>.png                left z-depth, uint16 mm (GT-only; never handed to estimators)
    frames.csv                     timestamp_ns,left_path,right_path
    gt_cam0.tum                    exact left optical-frame pose per frame (TUM, Lunalab world frame)
    calib.json                     intrinsics / baseline the renderer actually used
    dem.npy, dem.json, rocks.json  terrain truth for map-quality evaluation

The camera intrinsics and stereo extrinsics are exact by construction (they define the render), and GT is the
render pose itself, so GT body frame == cam0.
"""

from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import (CameraIntrinsics, GroundTruth, ImuCalibration, PreparedInput, SensorMode, SequenceInfo,
                     StereoCalibration, StreamInfo)
from .base import DatasetAdapter, write_frames_csv


class OmniLrsAdapter(DatasetAdapter):
    name = "omnilrs"
    adapter_version = "1"
    input_affecting_options = ()

    @property
    def seq_dir(self) -> Path:
        return self.root / self.sequence

    @lru_cache(maxsize=None)
    def _calib(self) -> dict:
        return json.loads((self.seq_dir / "calib.json").read_text())

    @lru_cache(maxsize=None)
    def _frames(self):
        rows = []
        with open(self.seq_dir / "frames.csv") as f:
            next(f)
            for line in f:
                t, left, right = line.strip().split(",")
                rows.append((int(t), left, right))
        return rows

    def source_files(self):
        return [self.seq_dir / "cam0", self.seq_dir / "cam1", self.seq_dir / "frames.csv",
                self.seq_dir / "calib.json"]

    def provenance(self):
        p = super().provenance()
        p["render"] = {k: v for k, v in self._calib().items() if k in ("source", "renderer", "rt_subframes",
                                                                        "lighting", "rocks")}
        dem = self.seq_dir / "dem.json"
        if dem.exists():
            p["terrain"] = json.loads(dem.read_text()).get("terrain")
        return p

    def sequence_info(self) -> SequenceInfo:
        if not (self.seq_dir / "frames.csv").exists():
            raise FileNotFoundError(f"{self.seq_dir / 'frames.csv'} not found")
        t = [r[0] for r in self._frames()]
        rate = (len(t) - 1) / ((t[-1] - t[0]) * 1e-9) if len(t) > 1 else 0.0
        streams = {cam: StreamInfo(cam, "camera", str(self.seq_dir / cam), len(t), rate, t[0], t[-1], cam)
                   for cam in ("cam0", "cam1")}
        return SequenceInfo(self.name, self.sequence, str(self.seq_dir), (t[-1] - t[0]) * 1e-9, streams,
                            (self.seq_dir / "gt_cam0.tum").exists(),
                            notes=["rendered in OmniLRS Lunalab (Isaac Sim); GT is the exact render pose"])

    def stereo_calibration(self) -> Optional[StereoCalibration]:
        c = self._calib()

        def intr(frame):
            return CameraIntrinsics("pinhole", int(c["width"]), int(c["height"]), float(c["fx"]), float(c["fy"]),
                                    float(c["cx"]), float(c["cy"]), [0.0, 0.0, 0.0, 0.0], rectified=True,
                                    frame_id=frame)

        T = np.eye(4)
        T[0, 3] = float(c["baseline_m"])
        return StereoCalibration(intr("cam0"), intr("cam1"), T, float(c["baseline_m"]),
                                 source=f"{self.seq_dir / 'calib.json'} (render parameters, exact)")

    def imu_calibration(self) -> Optional[ImuCalibration]:
        return None

    def ground_truth(self) -> Optional[GroundTruth]:
        path = self.seq_dir / "gt_cam0.tum"
        if not path.exists():
            return None
        d = np.loadtxt(path, ndmin=2)
        # TUM seconds are written with 9 decimals from the integer ns stamps; recover them exactly.
        t = np.array([r[0] for r in self._frames()], dtype=np.int64)
        if len(t) != len(d) or np.max(np.abs(t * 1e-9 - d[:, 0])) > 1e-6:
            t = np.round(d[:, 0] * 1e9).astype(np.int64)
        return GroundTruth(t, d[:, 1:4], d[:, 4:8], reference_frame="lunalab_world", body_frame="cam0",
                           source=str(path), notes=["render poses (exact, noise-free)"])

    def T_gt_body_sensor(self, sensor_frame: str) -> Optional[np.ndarray]:
        if sensor_frame == "cam0":
            return np.eye(4)
        if sensor_frame == "cam1":
            T = np.eye(4)
            T[0, 3] = float(self._calib()["baseline_m"])
            return T
        return None

    def prepare_offline(self, mode: SensorMode, out_dir: Path) -> PreparedInput:
        out_dir = Path(out_dir)
        stereo = mode in (SensorMode.STEREO, SensorMode.STEREO_INERTIAL)
        cams = ("cam0", "cam1") if stereo else ("cam0",)
        linked = copied = 0
        for cam in cams:
            (out_dir / cam).mkdir(parents=True, exist_ok=True)
        rows = []
        for t, left, right in self._frames():
            for rel in (left, right) if stereo else (left,):
                src, dst = self.seq_dir / rel, out_dir / rel
                try:
                    os.link(src, dst)  # same filesystem: no extra space
                    linked += 1
                except OSError:
                    shutil.copy2(src, dst)
                    copied += 1
            rows.append((t, left, right if stereo else ""))
        n = write_frames_csv(out_dir / "frames.csv", rows)
        manifest = {"dataset": self.name, "sequence": self.sequence, "mode": mode.value,
                    "source": str(self.seq_dir), "n_frames": n, "n_imu": 0,
                    "timestamps": "render timestamps (copied from the source trajectory), unchanged",
                    "images": "rendered BGR PNG, hard-linked (or copied) unchanged",
                    "hardlinked_files": linked, "copied_files": copied, "preprocessing": []}
        return PreparedInput(out_dir, mode, n, 0, manifest)
