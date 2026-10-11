"""SubT-MRS (ICCV'23 SLAM Challenge, superodometry.com) extracted-folder adapter.

Currently configured for the SmokeRoom sequence (``SubT_MRS_Hawkins_Smoke_Handheld``,
RC7 handheld payload, Hawkins, visually degraded by smoke). Every fact below was
checked on the published files (see docs/datasets/subtmrs.md):

* Folder release ``SubT_MRS_Hawkins_Smoke_Handheld.zip`` extracts to
  ``SmokeRoom_Folder/{cam_0,thermal,imu}``. ``cam_0/<i>.png`` (640x480, 8-bit RGB,
  ~24.5 Hz) pairs with line ``i`` of ``cam_0/timestamps.txt`` (integer ns).
  ``thermal`` is 640x512 16-bit (not used as estimator input yet).
  There is only ONE RGB camera -> mono / mono_inertial only.
* ``imu/imu_data.csv``: ``timestamp, q_x, q_y, q_z, q_w, ang_vel_{x,y,z}, lin_acc_{x,y,z}``,
  Epson M-G365 at 200 Hz, rad/s and m/s^2 (gravity included). The quaternion is an
  on-board orientation estimate and is ignored.
* Intrinsics YAML: ``rgb_camera`` uses the MEI unified omnidirectional model
  (xi=1.58, radial-tangential k1,k2,p1,p2, gamma1/2, u0/v0). ORB-SLAM3 supports only
  pinhole and Kannala-Brandt, so prepare_offline() resamples every frame into an ideal
  (distortion-free) virtual pinhole camera with the *same orientation and centre* as the
  original camera, which leaves the extrinsics unchanged.
* Extrinsics YAML: ``rgb_camera_to_imu`` is T_imu_cam (p_imu = T * p_cam): the optical
  axis maps to IMU +x (forward), optical x (right) to IMU -y, optical y (down) to IMU -z,
  and the camera sits 0.195 m ahead of the IMU. (``thermal_camera_to_imu`` has a bottom
  row of zeros in the published file and is not used.)
* Ground truth (separate release ``SubT_MRS_Hawkins_Smoke_Room_GT_IMU.zip``):
  ``SubT_MRS_Smoke_Room/integrated_odom/odometry_data.csv``
  ``timestamp, x, y, z, q_x, q_y, q_z, q_w`` = IMU pose at 200 Hz. It is LiDAR-based
  (Super Odometry), i.e. reference trajectory rather than motion capture.
* No IMU noise densities / random walks are published -> inertial modes are rejected
  unless verified values are supplied via ``dataset.options.imu_noise``.
* Sensors are hardware-synchronised (PPS/FPGA, <= 3 ms per the dataset hardware notes).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from ..trajectory import invert_T
from ..types import (CameraIntrinsics, GroundTruth, ImuCalibration, ImuNoise, PreparedInput,
                     SensorMode, SequenceInfo, StereoCalibration, StreamInfo)
from .base import DatasetAdapter, write_frames_csv, write_imu_csv

#: Default file layout per sequence, relative to ``<root>/<sequence>``.
SEQUENCES: Dict[str, Dict[str, str]] = {
    "SubT_MRS_Hawkins_Smoke_Handheld": {
        "data_dir": "SmokeRoom_Folder",
        "gt_csv": "SubT_MRS_Smoke_Room/integrated_odom/odometry_data.csv",
        "intrinsics_yaml": "SubT_MRS_Hawkins_Smoke_Handheld_Intrinsics.yaml",
        "extrinsics_yaml": "SubT_MRS_Hawkins_Smoke_Handheld_Extrinsics.yaml",
    },
}

DEFAULT_OPTIONS: Dict[str, Any] = {
    "data_dir": None,             # override any SEQUENCES path (relative to <root>/<sequence> or absolute)
    "gt_csv": None,
    "intrinsics_yaml": None,
    "extrinsics_yaml": None,
    # Virtual pinhole camera the MEI images are resampled into.
    "pinhole_width": 640,
    "pinhole_height": 480,
    "pinhole_focal": None,        # px, fx = fy; None -> mean(gamma1, gamma2) / (1 + xi)
    "interpolation": "linear",    # linear | cubic
    "imu_noise": None,            # dict(ImuNoise fields) with source + verified
    "time_offset_s": 0.0,         # t_imu = t_cam + offset (hardware synchronised)
}


# ---- MEI (unified omnidirectional) camera model ----------------------------
def mei_project(P: np.ndarray, xi: float, k1: float, k2: float, p1: float, p2: float,
                gamma1: float, gamma2: float, u0: float, v0: float) -> np.ndarray:
    """Project 3D points (..., 3) in the camera frame to pixels (..., 2).

    Same equations as CamOdoCal ``CataCamera::spaceToPlane`` / Kalibr ``omni-radtan``.
    Points with a non-positive denominator are returned as NaN.
    """
    P = np.asarray(P, dtype=float)
    z = P[..., 2] + xi * np.linalg.norm(P, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(z > 0, P[..., 0] / z, np.nan)
        y = np.where(z > 0, P[..., 1] / z, np.nan)
    r2 = x * x + y * y
    rad = k1 * r2 + k2 * r2 * r2
    xd = x + x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y + y * rad + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    return np.stack([gamma1 * xd + u0, gamma2 * yd + v0], axis=-1)


def pinhole_to_mei_maps(mei: Dict[str, float], width: int, height: int, f: float):
    """Remap tables (map_x, map_y float32) sampling an MEI image for a virtual pinhole camera
    (fx = fy = f, principal point at the image centre, no distortion, same orientation)."""
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rays = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], axis=-1)
    uv = mei_project(rays, **mei)
    return uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)


def _read_matrix(d: dict) -> np.ndarray:
    return np.array(d["data"], dtype=float).reshape(int(d["rows"]), int(d["cols"]))


def _sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


class SubtMrsAdapter(DatasetAdapter):
    name = "subtmrs"
    adapter_version = "1"
    input_affecting_options = ("pinhole_width", "pinhole_height", "pinhole_focal", "interpolation")

    def __init__(self, root, sequence, options=None):
        merged = dict(DEFAULT_OPTIONS)
        merged.update(options or {})
        super().__init__(root, sequence, merged)
        if self.options["interpolation"] not in ("linear", "cubic"):
            raise ValueError("subtmrs: interpolation must be 'linear' or 'cubic'")

    # ---- paths -------------------------------------------------------------
    @property
    def seq_dir(self) -> Path:
        return self.root / self.sequence

    def _path(self, key: str) -> Path:
        rel = self.options.get(key) or SEQUENCES.get(self.sequence, {}).get(key)
        if rel is None:
            raise KeyError(f"subtmrs: unknown sequence {self.sequence!r}; set dataset.options.{key} "
                           f"(known sequences: {sorted(SEQUENCES)})")
        p = Path(rel).expanduser()
        return p if p.is_absolute() else self.seq_dir / p

    @property
    def data_dir(self) -> Path:
        return self._path("data_dir")

    # ---- raw files ---------------------------------------------------------
    @lru_cache(maxsize=None)
    def _cam_rows(self):
        ts_path = self.data_dir / "cam_0" / "timestamps.txt"
        t = np.loadtxt(ts_path, dtype=np.int64, ndmin=1)
        return t, [f"{i}.png" for i in range(len(t))]

    @lru_cache(maxsize=None)
    def _imu(self):
        path = self.data_dir / "imu" / "imu_data.csv"
        t = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)
        v = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(5, 11), ndmin=2)
        return t, v[:, 0:3], v[:, 3:6]

    @lru_cache(maxsize=None)
    def _intrinsics_yaml(self) -> dict:
        return yaml.safe_load(self._path("intrinsics_yaml").read_text())

    @lru_cache(maxsize=None)
    def _extrinsics_yaml(self) -> dict:
        return yaml.safe_load(self._path("extrinsics_yaml").read_text())

    def mei_params(self) -> Dict[str, float]:
        c = self._intrinsics_yaml()["rgb_camera"]
        if str(c.get("model_type", "")).upper() != "MEI":
            raise ValueError(f"rgb_camera model {c.get('model_type')!r} is not MEI")
        d, p = c["distortion_parameters"], c["projection_parameters"]
        return {"xi": float(c["mirror_parameters"]["xi"]), "k1": float(d["k1"]), "k2": float(d["k2"]),
                "p1": float(d["p1"]), "p2": float(d["p2"]), "gamma1": float(p["gamma1"]),
                "gamma2": float(p["gamma2"]), "u0": float(p["u0"]), "v0": float(p["v0"])}

    def source_image_size(self):
        c = self._intrinsics_yaml()["rgb_camera"]
        return int(c["image_width"]), int(c["image_height"])

    def pinhole_focal(self) -> float:
        f = self.options.get("pinhole_focal")
        if f is not None:
            return float(f)
        m = self.mei_params()
        return 0.5 * (m["gamma1"] + m["gamma2"]) / (1.0 + m["xi"])

    def T_imu_cam0(self) -> np.ndarray:
        T = _read_matrix(self._extrinsics_yaml()["rgb_camera_to_imu"])
        R = T[:3, :3]
        if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError("rgb_camera_to_imu is not a homogeneous 4x4 transform")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
            raise ValueError("rgb_camera_to_imu rotation is not orthonormal")
        return T

    # ---- metadata ----------------------------------------------------------
    def provenance(self) -> Dict[str, Any]:
        files = {k: self._path(k) for k in ("intrinsics_yaml", "extrinsics_yaml", "gt_csv")}
        return {"root": str(self.root), "sequence": self.sequence, "data_dir": str(self.data_dir),
                "files": {k: {"path": str(p), "sha1": _sha1(p) if p.exists() else None}
                          for k, p in files.items()},
                "options": dict(self.options)}

    def sequence_info(self) -> SequenceInfo:
        if not (self.data_dir / "cam_0" / "timestamps.txt").exists():
            raise FileNotFoundError(f"{self.data_dir}/cam_0/timestamps.txt not found")
        streams = {}
        t, _ = self._cam_rows()
        streams["cam0"] = StreamInfo("cam0", "camera", str(self.data_dir / "cam_0"), len(t),
                                     (len(t) - 1) / ((t[-1] - t[0]) * 1e-9), int(t[0]), int(t[-1]),
                                     "rgb_camera")
        if (self.data_dir / "imu" / "imu_data.csv").exists():
            ti, _, _ = self._imu()
            streams["imu"] = StreamInfo("imu", "imu", str(self.data_dir / "imu"), len(ti),
                                        (len(ti) - 1) / ((ti[-1] - ti[0]) * 1e-9), int(ti[0]), int(ti[-1]),
                                        "imu")
        notes = ["single RGB camera (no stereo pair); MEI images are resampled to a virtual pinhole",
                 "thermal camera present in the release but not exposed as an estimator stream yet"]
        return SequenceInfo(self.name, self.sequence, str(self.seq_dir), (t[-1] - t[0]) * 1e-9,
                            streams, self._path("gt_csv").exists(), notes)

    def stereo_calibration(self) -> Optional[StereoCalibration]:
        w, h = int(self.options["pinhole_width"]), int(self.options["pinhole_height"])
        f = self.pinhole_focal()
        left = CameraIntrinsics("pinhole", w, h, f, f, (w - 1) / 2.0, (h - 1) / 2.0, [0.0, 0.0, 0.0, 0.0],
                                rectified=True, frame_id="cam0")
        return StereoCalibration(left, None, np.eye(4), 0.0,
                                 source=f"virtual pinhole resampled from MEI model in "
                                        f"{self._path('intrinsics_yaml')}")

    def imu_calibration(self) -> Optional[ImuCalibration]:
        if not (self.data_dir / "imu" / "imu_data.csv").exists():
            return None
        noise = None
        cfg = self.options.get("imu_noise")
        if cfg:
            noise = ImuNoise(**{k: cfg[k] for k in ("gyro_noise_density", "accel_noise_density",
                                                     "gyro_random_walk", "accel_random_walk", "source")},
                             verified=bool(cfg.get("verified", False)))
        info = self.sequence_info().streams["imu"]
        return ImuCalibration(invert_T(self.T_imu_cam0()), info.rate_hz,
                              source=f"{self._path('extrinsics_yaml')} rgb_camera_to_imu (= T_imu_cam)",
                              noise=noise, time_offset_s=float(self.options["time_offset_s"]),
                              notes=["IMU noise densities and bias random walks are not published with "
                                     "SubT-MRS",
                                     "sensors are hardware synchronised (<= 3 ms per dataset notes); "
                                     "offset assumed 0 unless dataset.options.time_offset_s is set"])

    @lru_cache(maxsize=None)
    def ground_truth(self) -> Optional[GroundTruth]:
        path = self._path("gt_csv")
        if not path.exists():
            return None
        t = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)
        v = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, 8), ndmin=2)
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        keep = np.r_[True, np.diff(t) > 0]
        return GroundTruth(t[keep], v[keep, 0:3], v[keep, 3:7], reference_frame="subtmrs_gt_world",
                           body_frame="imu", source=str(path),
                           notes=["SubT-MRS 'GT_IMU' integrated odometry: IMU pose from the LiDAR-based "
                                  "reference pipeline (not motion capture)",
                                  f"{int((~keep).sum())} duplicate/unsorted samples dropped"])

    def T_gt_body_sensor(self, sensor_frame: str) -> Optional[np.ndarray]:
        if sensor_frame == "imu":
            return np.eye(4)
        if sensor_frame == "cam0":
            return self.T_imu_cam0()
        return None

    # ---- input preparation ---------------------------------------------------
    def preparation_key(self, mode: SensorMode) -> str:
        base = super().preparation_key(mode)
        calib = _sha1(self._path("intrinsics_yaml"))
        return hashlib.sha1(f"{base}:{calib}".encode()).hexdigest()[:12]

    def prepare_offline(self, mode: SensorMode, out_dir: Path) -> PreparedInput:
        import cv2
        from concurrent.futures import ThreadPoolExecutor

        if mode not in (SensorMode.MONO, SensorMode.MONO_INERTIAL):
            raise ValueError(f"subtmrs provides a single camera; mode {mode.value} is not supported")
        out_dir = Path(out_dir)
        (out_dir / "cam0").mkdir(parents=True, exist_ok=True)
        cam = self.stereo_calibration().left
        mei = self.mei_params()
        src_w, src_h = self.source_image_size()
        map_x, map_y = pinhole_to_mei_maps(mei, cam.width, cam.height, cam.fx)
        inside = (map_x >= 0) & (map_x <= src_w - 1) & (map_y >= 0) & (map_y <= src_h - 1)
        interp = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}[self.options["interpolation"]]
        mx, my = np.nan_to_num(map_x, nan=-1.0), np.nan_to_num(map_y, nan=-1.0)

        t, files = self._cam_rows()
        src_dir = self.data_dir / "cam_0"
        missing = [f for f in files if not (src_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cam_0 images listed in timestamps.txt are missing "
                                    f"(first: {missing[0]})")
        shapes = set()

        def convert(fname):
            img = cv2.imread(str(src_dir / fname), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"cannot read cam_0/{fname}")
            if img.dtype != np.uint8:
                raise ValueError(f"cam_0/{fname} is {img.dtype}, expected 8-bit")
            shapes.add(img.shape)
            if img.shape[:2] != (src_h, src_w):
                raise ValueError(f"cam_0/{fname} is {img.shape[1]}x{img.shape[0]}, calibration is "
                                 f"{src_w}x{src_h}")
            out = cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if not cv2.imwrite(str(out_dir / "cam0" / fname), out):
                raise IOError(f"cannot write cam0/{fname}")

        with ThreadPoolExecutor(8) as pool:
            for j in [pool.submit(convert, f) for f in files]:
                j.result()

        order = np.argsort(t, kind="stable")
        rows, dropped = [], 0
        last = None
        for i in order:
            if last is not None and t[i] <= last:
                dropped += 1
                continue
            rows.append((int(t[i]), f"cam0/{files[i]}", ""))
            last = t[i]
        n = write_frames_csv(out_dir / "frames.csv", rows)

        n_imu, imu_dropped = 0, 0
        if mode.is_inertial:
            ti, w, a = self._imu()
            o = np.argsort(ti, kind="stable")
            ti, w, a = ti[o], w[o], a[o]
            keep = np.r_[True, np.diff(ti) > 0]
            imu_dropped = int((~keep).sum())
            n_imu = write_imu_csv(out_dir / "imu.csv", ti[keep], w[keep], a[keep])

        ts = np.array([r[0] for r in rows], dtype=np.int64)
        dt = np.diff(ts) * 1e-9
        manifest = {
            "dataset": self.name, "sequence": self.sequence, "mode": mode.value,
            "source": str(self.data_dir), "n_frames": n, "n_imu": n_imu,
            "frames_dropped_non_increasing_timestamp": dropped,
            "imu_samples_dropped_non_increasing_timestamp": imu_dropped,
            "frame_dt_s": {"median": float(np.median(dt)), "max": float(dt.max())} if len(dt) else {},
            "timestamps": "original cam_0/timestamps.txt and imu_data.csv nanosecond stamps (unchanged)",
            "source_image_shapes": sorted(str(s) for s in shapes),
            "images": (f"MEI {src_w}x{src_h} -> virtual pinhole {cam.width}x{cam.height}, "
                       f"fx=fy={cam.fx:.6g}, cx={cam.cx}, cy={cam.cy}, zero distortion, same camera "
                       f"orientation; cv2.remap {self.options['interpolation']}, out-of-image pixels black; "
                       "colour order unchanged (BGR as read by OpenCV)"),
            "conversions": ["mei_to_pinhole_remap"],
            "mei_parameters": mei,
            "pinhole_valid_pixel_fraction": float(inside.mean()),
            "imu_columns_used": "ang_vel_xyz (rad/s), lin_acc_xyz (m/s^2); orientation quaternion ignored",
            "preprocessing": [],
        }
        return PreparedInput(out_dir, mode, n, n_imu, manifest)
