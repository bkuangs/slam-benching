"""TUM-VI (exported EuRoC/ASL format, 512x512 16-bit-derived) adapter.

Used as the *supported example dataset* to verify the ORB-SLAM3 installation
(ORB-SLAM3 ships TUM-VI example configurations).

Expected layout (as extracted from dataset-<seq>_512_16.tar):
    <root>/dataset-<seq>_512_16/mav0/{cam0,cam1}/data.csv + data/*.png
    <root>/dataset-<seq>_512_16/mav0/imu0/data.csv
    <root>/dataset-<seq>_512_16/mav0/mocap0/data.csv   (ground truth, IMU frame)
    <root>/dataset-<seq>_512_16/dso/camchain.yaml      (Kalibr calibration)
    <root>/dataset-<seq>_512_16/dso/imu_config.yaml    (IMU noise, published by TUM)

Facts used (from the files themselves):
* camchain: equidistant (Kannala-Brandt) fisheye, T_cam_imu = p_cam <- p_imu,
  cam1.T_cn_cnm1 = T_cam1_cam0.
* mocap0/data.csv: p_RS_R / q_RS = pose of the IMU (S) in the mocap frame (R);
  quaternion order w,x,y,z.
* imu0: gyro rad/s then accel m/s^2.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..trajectory import invert_T
from ..types import (CameraIntrinsics, GroundTruth, ImuCalibration, ImuNoise, PreparedInput,
                     SensorMode, SequenceInfo, StereoCalibration, StreamInfo)
from .base import DatasetAdapter, write_frames_csv, write_imu_csv


def _read_csv_rows(path: Path):
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            yield row


class TumViAdapter(DatasetAdapter):
    name = "tumvi"
    adapter_version = "2"
    input_affecting_options = ()

    @property
    def seq_dir(self) -> Path:
        d = self.root / self.sequence
        if not d.exists() and (self.root / f"dataset-{self.sequence}_512_16").exists():
            d = self.root / f"dataset-{self.sequence}_512_16"
        return d

    @property
    def mav0(self) -> Path:
        return self.seq_dir / "mav0"

    @lru_cache(maxsize=None)
    def _camchain(self) -> dict:
        return yaml.safe_load((self.seq_dir / "dso" / "camchain.yaml").read_text())

    @lru_cache(maxsize=None)
    def _cam_rows(self, cam: str):
        return [(int(r[0]), r[1].strip()) for r in _read_csv_rows(self.mav0 / cam / "data.csv")]

    @lru_cache(maxsize=None)
    def _imu(self):
        rows = np.array([[float(x) for x in r] for r in _read_csv_rows(self.mav0 / "imu0" / "data.csv")])
        t = np.array([int(r[0]) for r in _read_csv_rows(self.mav0 / "imu0" / "data.csv")], dtype=np.int64)
        return t, rows[:, 1:4], rows[:, 4:7]

    def source_files(self):
        return [self.mav0 / "cam0", self.mav0 / "cam1", self.mav0 / "imu0" / "data.csv"]

    def sequence_info(self) -> SequenceInfo:
        if not self.mav0.exists():
            raise FileNotFoundError(f"{self.mav0} not found")
        streams = {}
        for cam in ("cam0", "cam1"):
            rows = self._cam_rows(cam)
            t = [r[0] for r in rows]
            streams[cam] = StreamInfo(cam, "camera", str(self.mav0 / cam), len(t),
                                      (len(t) - 1) / ((t[-1] - t[0]) * 1e-9), t[0], t[-1])
        ti, _, _ = self._imu()
        streams["imu"] = StreamInfo("imu", "imu", str(self.mav0 / "imu0"), len(ti),
                                    (len(ti) - 1) / ((ti[-1] - ti[0]) * 1e-9), int(ti[0]), int(ti[-1]))
        t0 = streams["cam0"]
        return SequenceInfo(self.name, self.sequence, str(self.seq_dir),
                            (t0.last_ns - t0.first_ns) * 1e-9, streams,
                            (self.mav0 / "mocap0" / "data.csv").exists())

    def _intrinsics(self, cam: str) -> CameraIntrinsics:
        c = self._camchain()[cam]
        if c["distortion_model"] != "equidistant":
            raise ValueError(f"unsupported distortion model {c['distortion_model']}")
        fx, fy, cx, cy = c["intrinsics"]
        w, h = c["resolution"]
        return CameraIntrinsics("kannala_brandt", w, h, fx, fy, cx, cy,
                                list(c["distortion_coeffs"]), rectified=False, frame_id=cam)

    def stereo_calibration(self) -> Optional[StereoCalibration]:
        cc = self._camchain()
        T_c1_c0 = np.array(cc["cam1"]["T_cn_cnm1"], dtype=float)
        T_c0_c1 = invert_T(T_c1_c0)
        return StereoCalibration(self._intrinsics("cam0"), self._intrinsics("cam1"), T_c0_c1,
                                 float(np.linalg.norm(T_c0_c1[:3, 3])),
                                 source=str(self.seq_dir / "dso" / "camchain.yaml"))

    def imu_calibration(self) -> Optional[ImuCalibration]:
        cc = self._camchain()
        T_c0_i = np.array(cc["cam0"]["T_cam_imu"], dtype=float)
        cfg_path = self.seq_dir / "dso" / "imu_config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        noise = ImuNoise(gyro_noise_density=float(cfg["gyroscope_noise_density"]),
                         accel_noise_density=float(cfg["accelerometer_noise_density"]),
                         gyro_random_walk=float(cfg["gyroscope_random_walk"]),
                         accel_random_walk=float(cfg["accelerometer_random_walk"]),
                         source=f"{cfg_path} (TUM-VI published, inflated values)", verified=True)
        return ImuCalibration(T_c0_i, float(cfg.get("update_rate", 200.0)),
                              source=str(self.seq_dir / "dso" / "camchain.yaml"), noise=noise,
                              time_offset_s=0.0,
                              notes=["TUM-VI exported timestamps are already time-offset corrected"])

    def ground_truth(self) -> Optional[GroundTruth]:
        path = self.mav0 / "mocap0" / "data.csv"
        if not path.exists():
            return None
        rows = list(_read_csv_rows(path))
        t = np.array([int(r[0]) for r in rows], dtype=np.int64)
        v = np.array([[float(x) for x in r[1:8]] for r in rows])
        q_xyzw = v[:, [4, 5, 6, 3]]
        order = np.argsort(t, kind="stable")
        t, v, q_xyzw = t[order], v[order], q_xyzw[order]
        keep = np.r_[True, np.diff(t) > 0]
        return GroundTruth(t[keep], v[keep, 0:3], q_xyzw[keep], reference_frame="mocap_world",
                           body_frame="imu", source=str(path),
                           notes=["TUM-VI mocap0 gives the IMU pose (hand-eye calibrated by TUM)"])

    def T_gt_body_sensor(self, sensor_frame: str) -> Optional[np.ndarray]:
        if sensor_frame == "imu":
            return np.eye(4)
        if sensor_frame == "cam0":
            return invert_T(np.array(self._camchain()["cam0"]["T_cam_imu"], dtype=float))
        return None

    def prepare_offline(self, mode: SensorMode, out_dir: Path) -> PreparedInput:
        import cv2
        from concurrent.futures import ThreadPoolExecutor

        out_dir = Path(out_dir)
        stereo = mode in (SensorMode.STEREO, SensorMode.STEREO_INERTIAL)
        depths = set()

        def convert(cam, fname):
            img = cv2.imread(str(self.mav0 / cam / "data" / fname), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"cannot read {cam}/{fname}")
            depths.add(str(img.dtype))
            if img.dtype == np.uint16:
                # Identical to cv::imread(..., IMREAD_GRAYSCALE) used by ORB-SLAM3's TUM-VI example.
                img = (img >> 8).astype(np.uint8)
            if not cv2.imwrite(str(out_dir / cam / fname), img):
                raise IOError(f"cannot write {cam}/{fname}")

        cams = ("cam0", "cam1") if stereo else ("cam0",)
        with ThreadPoolExecutor(8) as pool:
            jobs = []
            for cam in cams:
                (out_dir / cam).mkdir(parents=True, exist_ok=True)
                jobs += [pool.submit(convert, cam, f) for _, f in self._cam_rows(cam)]
            for j in jobs:
                j.result()
        left = {t: f for t, f in self._cam_rows("cam0")}
        right = {t: f for t, f in self._cam_rows("cam1")} if stereo else None
        rows, unpaired = [], 0
        for t in sorted(left):
            if right is not None:
                if t not in right:
                    unpaired += 1
                    continue
                rows.append((t, f"cam0/{left[t]}", f"cam1/{right[t]}"))
            else:
                rows.append((t, f"cam0/{left[t]}", ""))
        n = write_frames_csv(out_dir / "frames.csv", rows)
        n_imu = 0
        if mode.is_inertial:
            t, w, a = self._imu()
            n_imu = write_imu_csv(out_dir / "imu.csv", t, w, a)
        manifest = {
            "dataset": self.name, "sequence": self.sequence, "mode": mode.value,
            "source": str(self.seq_dir), "n_frames": n, "n_imu": n_imu,
            "unpaired_left_frames_dropped": unpaired,
            "timestamps": "original EuRoC-format nanosecond timestamps (unchanged)",
            "source_image_dtypes": sorted(depths),
            "images": "512x512 PNG; uint16 -> uint8 by right shift of 8 bits (bit-identical to "
                      "cv::imread IMREAD_GRAYSCALE as in ORB-SLAM3's TUM-VI example); no other processing",
            "conversions": ["uint16_to_uint8_rshift8"] if "uint16" in depths else [],
            "preprocessing": [],
        }
        return PreparedInput(out_dir, mode, n, n_imu, manifest)
