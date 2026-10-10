"""LunaPolaris (University of Luxembourg, SnT-SpaceR) ROS 2 bag adapter.

Every fact below was verified on sequence_01 (see docs/datasets/lunapolaris.md):

* Each sequence is a ROS 2 sqlite3 bag directory ``<root>/sequence_XX`` with
  ``metadata.yaml`` and ``sequence_XX.db3`` (Zenodo records, one per sequence).
* Stereo: ZED2 ``/leo01/zed2/{left,right}/image_rect_color`` (sensor_msgs/Image,
  640x360, bgra8, ~14.6 Hz). Images are already rectified by the ZED SDK; the
  matching ``camera_info`` has zero distortion, identity R, and
  P_right[0,3] = -fx * baseline. Left/right header stamps are identical.
* The Zenodo stereo calibration JSON (record 16728838) is an independent
  checkerboard calibration with a *different* rectification (fx=268.49) and
  translation in checkerboard-square units (T_x=-1.1995 squares of 0.1 m).
  It does not describe the recorded rectified images, so it is only used as a
  cross-check of the baseline, never as estimator input.
* IMU: ``/leo01/zed2/imu/data_raw`` and ``/imu/data`` (frame zed2_imu_link).
  Camera-IMU extrinsic: ``/leo01/zed2/left_cam_imu_transform``
  (zed2_left_camera_frame -> zed2_imu_link, ROS body convention, not optical).
  No IMU noise densities/random walks are published -> inertial modes are
  rejected unless verified values are supplied explicitly.
* Ground truth: OptiTrack via VRPN. ``/vrpn_mocap/ZED2_camera_marker/rectified/pose``
  equals the ``/tf`` world->vrpn_mocap/ZED2_camera_marker transform that the
  ``/tf_static`` tree attaches to the robot, and is Z-up. The raw ``/pose`` topic is a
  cyclic axis permutation (Y-up) of it whose body axes do *not* match the tf tree.
  Marker->left optical frame comes from /tf_static (nominal URDF values; the chain
  is internally consistent with optitrack_marker_link to <0.5 mm).
* LiDAR: Livox ``/leo01/livox/lidar`` (PointCloud2) + ``/leo01/livox/imu``.
"""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from ..tf import StaticTfTree
from ..trajectory import T_from_quat, invert_T, make_T
from ..types import (CameraIntrinsics, GroundTruth, ImuCalibration, ImuNoise, PreparedInput,
                     SensorMode, SequenceInfo, StereoCalibration, StreamInfo)
from .base import DatasetAdapter, write_frames_csv, write_imu_csv

DEFAULT_OPTIONS: Dict[str, Any] = {
    "left_topic": "/leo01/zed2/left/image_rect_color",
    "right_topic": "/leo01/zed2/right/image_rect_color",
    "left_info_topic": "/leo01/zed2/left/camera_info",
    "right_info_topic": "/leo01/zed2/right/camera_info",
    "imu_topic": "/leo01/zed2/imu/data_raw",
    "cam_imu_topic": "/leo01/zed2/left_cam_imu_transform",
    "lidar_topic": "/leo01/livox/lidar",
    "lidar_imu_topic": "/leo01/livox/imu",
    "gt_topic": "/vrpn_mocap/ZED2_camera_marker/rectified/pose",
    "gt_body_frame": "vrpn_mocap/ZED2_camera_marker",
    "cam0_frame": "zed2_left_camera_optical_frame",
    "calibration_source": "bag_camera_info",
    "image_format": "png_bgr",        # lossless; alpha channel (bgra8) dropped
    "imu_noise": None,                # dict(ImuNoise fields) with source + verified
    "zenodo_calibration_json": None,  # optional path; cross-check only
}


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _decode_image(msg) -> np.ndarray:
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = msg.encoding
    ch = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3, "mono8": 1}.get(enc)
    if ch is None:
        raise ValueError(f"unsupported image encoding {enc}")
    buf = np.asarray(msg.data, dtype=np.uint8).reshape(h, step)[:, : w * ch].reshape(h, w, ch)
    if enc == "bgra8":
        return buf[:, :, :3], buf[:, :, 3]
    if enc == "rgba8":
        return buf[:, :, 2::-1], buf[:, :, 3]
    if enc == "rgb8":
        return buf[:, :, ::-1], None
    return (buf[:, :, 0] if ch == 1 else buf), None


class LunaPolarisAdapter(DatasetAdapter):
    name = "lunapolaris"
    adapter_version = "1"
    input_affecting_options = ("left_topic", "right_topic", "imu_topic", "image_format")

    def __init__(self, root, sequence, options=None):
        merged = dict(DEFAULT_OPTIONS)
        merged.update(options or {})
        super().__init__(root, sequence, merged)
        if self.options["calibration_source"] != "bag_camera_info":
            raise ValueError("lunapolaris: only calibration_source='bag_camera_info' is valid for the "
                             "recorded rectified images (the Zenodo JSON uses a different rectification)")

    # ---- bag access ------------------------------------------------------
    @property
    def bag_dir(self) -> Path:
        return self.root / self.sequence

    def _reader(self):
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
        return AnyReader([self.bag_dir], default_typestore=get_typestore(Stores.ROS2_HUMBLE))

    @lru_cache(maxsize=None)
    def _metadata(self) -> dict:
        return yaml.safe_load((self.bag_dir / "metadata.yaml").read_text())["rosbag2_bagfile_information"]

    def _topic_counts(self) -> Dict[str, int]:
        return {t["topic_metadata"]["name"]: int(t["message_count"])
                for t in self._metadata()["topics_with_message_count"]}

    def _iter(self, topic: str):
        with self._reader() as r:
            conns = [c for c in r.connections if c.topic == topic]
            if not conns:
                raise KeyError(f"topic {topic} not in bag {self.bag_dir}")
            for c, t, raw in r.messages(connections=conns):
                yield t, r.deserialize(raw, c.msgtype)

    def _first(self, topic: str):
        for t, m in self._iter(topic):
            return m
        raise ValueError(f"topic {topic} has no messages")

    # ---- metadata --------------------------------------------------------
    def provenance(self) -> Dict[str, Any]:
        md = self._metadata()
        db = [self.bag_dir / f["path"] for f in md["files"]]
        return {"root": str(self.root), "sequence": self.sequence, "bag_dir": str(self.bag_dir),
                "bag_files": [{"path": str(p), "size_bytes": p.stat().st_size if p.exists() else None}
                              for p in db],
                "bag_start_ns": md["starting_time"]["nanoseconds_since_epoch"],
                "bag_duration_ns": md["duration"]["nanoseconds"],
                "message_count": md["message_count"],
                "options": {k: v for k, v in self.options.items()}}

    def sequence_info(self) -> SequenceInfo:
        if not (self.bag_dir / "metadata.yaml").exists():
            raise FileNotFoundError(f"{self.bag_dir}/metadata.yaml not found")
        md = self._metadata()
        counts = self._topic_counts()
        start = int(md["starting_time"]["nanoseconds_since_epoch"])
        dur = int(md["duration"]["nanoseconds"])
        o = self.options
        logical = {"cam0": (o["left_topic"], "camera"), "cam1": (o["right_topic"], "camera"),
                   "imu": (o["imu_topic"], "imu"), "lidar": (o["lidar_topic"], "lidar")}
        streams = {}
        for name, (topic, kind) in logical.items():
            n = counts.get(topic, 0)
            if n > 0:
                # rate/first/last are bag-level approximations; exact stamps are read at preparation
                streams[name] = StreamInfo(name, kind, topic, n, n / (dur * 1e-9), start, start + dur)
        notes = ["stream rate/first/last are derived from bag metadata (receive time), "
                 "not header stamps",
                 "lidar_imu (/leo01/livox/imu) is separate from the ZED2 IMU used for visual-inertial"]
        return SequenceInfo(self.name, self.sequence, str(self.bag_dir), dur * 1e-9, streams,
                            counts.get(o["gt_topic"], 0) > 0, notes)

    @lru_cache(maxsize=None)
    def _camera_info(self, topic: str) -> dict:
        """First camera_info, plus a check that it never changes during the sequence."""
        first, n, changed = None, 0, 0
        for _, m in self._iter(topic):
            key = (int(m.width), int(m.height), tuple(np.round(m.k, 6)), tuple(np.round(m.p, 6)),
                   tuple(np.round(m.d, 9)), tuple(np.round(m.r, 9)), m.distortion_model, m.header.frame_id)
            if first is None:
                first = key
            elif key != first:
                changed += 1
            n += 1
        w, h, k, p, d, r, model, frame = first
        return {"width": w, "height": h, "K": list(k), "P": list(p), "D": list(d), "R": list(r),
                "distortion_model": model, "frame_id": frame, "n_messages": n, "n_changed": changed}

    @lru_cache(maxsize=None)
    def stereo_calibration(self) -> Optional[StereoCalibration]:
        li = self._camera_info(self.options["left_info_topic"])
        ri = self._camera_info(self.options["right_info_topic"])
        problems = []
        for side, ci in (("left", li), ("right", ri)):
            if ci["n_changed"]:
                problems.append(f"{side} camera_info changes during the sequence ({ci['n_changed']} msgs)")
            if any(abs(x) > 0 for x in ci["D"]):
                problems.append(f"{side} camera_info has non-zero distortion; images are not rectified")
            if not np.allclose(np.reshape(ci["R"], (3, 3)), np.eye(3)):
                problems.append(f"{side} camera_info R is not identity")
        if problems:
            raise ValueError("; ".join(problems))
        P_l, P_r = np.reshape(li["P"], (3, 4)), np.reshape(ri["P"], (3, 4))
        if not np.allclose(P_l[:, :3], P_r[:, :3]):
            raise ValueError("left/right rectified projection matrices differ")
        fx = P_l[0, 0]
        baseline = -P_r[0, 3] / fx
        if baseline <= 0:
            raise ValueError(f"non-positive stereo baseline {baseline}")

        def intr(ci, P):
            return CameraIntrinsics("pinhole", ci["width"], ci["height"], P[0, 0], P[1, 1], P[0, 2],
                                    P[1, 2], [0.0, 0.0, 0.0, 0.0], rectified=True, frame_id=ci["frame_id"])

        src = (f"bag camera_info {self.options['left_info_topic']}, {self.options['right_info_topic']} "
               f"(ZED SDK rectification, constant over {li['n_messages']}/{ri['n_messages']} msgs)")
        return StereoCalibration(intr(li, P_l), intr(ri, P_r), make_T(np.eye(3), [baseline, 0, 0]),
                                 float(baseline), src)

    def zenodo_crosscheck(self) -> Optional[dict]:
        path = self.options.get("zenodo_calibration_json")
        if not path:
            cand = self.root / "calibration_results.json"
            path = cand if cand.exists() else None
        if not path:
            return None
        j = json.loads(Path(path).read_text())
        sq = 0.1  # 10 cm checkerboard squares (Zenodo record 16728838 description)
        b_json = abs(float(j["T"][0][0])) * sq
        st = self.stereo_calibration()
        return {"json": str(path), "json_baseline_m": b_json, "bag_baseline_m": st.baseline_m,
                "relative_difference": abs(b_json - st.baseline_m) / st.baseline_m,
                "json_rectified_fx": j["left_P"][0][0], "bag_rectified_fx": st.left.fx,
                "note": "JSON uses its own rectification; intrinsics are not interchangeable"}

    @lru_cache(maxsize=None)
    def static_tf(self) -> StaticTfTree:
        tree = StaticTfTree()
        for _, m in self._iter("/tf_static"):
            for tr in m.transforms:
                v, q = tr.transform.translation, tr.transform.rotation
                tree.add(tr.header.frame_id, tr.child_frame_id,
                         T_from_quat([v.x, v.y, v.z], [q.x, q.y, q.z, q.w]))
        return tree

    @lru_cache(maxsize=None)
    def _T_cam0_imu(self):
        Ts, frames = [], set()
        for i, (_, m) in enumerate(self._iter(self.options["cam_imu_topic"])):
            v, q = m.transform.translation, m.transform.rotation
            Ts.append(T_from_quat([v.x, v.y, v.z], [q.x, q.y, q.z, q.w]))
            frames.add((m.header.frame_id, m.child_frame_id))
            if i >= 2000:
                break
        Ts = np.array(Ts)
        spread = float(np.abs(Ts - Ts[0]).max())
        (parent, child), = frames
        T_parent_imu = Ts[0]
        T_parent_cam0 = self.static_tf().lookup(parent, self.options["cam0_frame"])
        if T_parent_cam0 is None:
            raise ValueError(f"no static transform {parent} -> {self.options['cam0_frame']}")
        return invert_T(T_parent_cam0) @ T_parent_imu, parent, child, spread, len(Ts)

    def imu_calibration(self) -> Optional[ImuCalibration]:
        if self._topic_counts().get(self.options["imu_topic"], 0) == 0:
            return None
        T, parent, child, spread, n = self._T_cam0_imu()
        noise = None
        cfg = self.options.get("imu_noise")
        if cfg:
            noise = ImuNoise(**{k: cfg[k] for k in ("gyro_noise_density", "accel_noise_density",
                                                     "gyro_random_walk", "accel_random_walk", "source")},
                             verified=bool(cfg.get("verified", False)))
        info = self.sequence_info().streams.get("imu")
        return ImuCalibration(
            T, info.rate_hz if info else float("nan"),
            source=f"{self.options['cam_imu_topic']} ({parent}->{child}, max element spread over "
                   f"{n} msgs {spread:.2e}) composed with /tf_static {parent}->{self.options['cam0_frame']}",
            noise=noise, time_offset_s=None,
            notes=["IMU noise densities and bias random walks are not published with LunaPolaris; "
                   "sensor_msgs/Imu covariances are per-sample variances and contain no random walk",
                   "camera-IMU time offset is not published (ZED2 stamps are SDK-synchronised)"])

    @lru_cache(maxsize=None)
    def ground_truth(self) -> Optional[GroundTruth]:
        topic = self.options["gt_topic"]
        if self._topic_counts().get(topic, 0) == 0:
            return None
        t, p, q, frames = [], [], [], set()
        for _, m in self._iter(topic):
            t.append(_stamp_ns(m.header))
            p.append([m.pose.position.x, m.pose.position.y, m.pose.position.z])
            o = m.pose.orientation
            q.append([o.x, o.y, o.z, o.w])
            frames.add(m.header.frame_id)
        t, p, q = np.array(t, dtype=np.int64), np.array(p), np.array(q)
        order = np.argsort(t, kind="stable")
        t, p, q = t[order], p[order], q[order]
        keep = np.r_[True, np.diff(t) > 0]
        notes = [f"{int((~keep).sum())} duplicate-stamp samples dropped"]
        if "rectified" in topic:
            notes.append("rectified topic: Z-up world, body axes consistent with /tf_static tree")
        return GroundTruth(t[keep], p[keep], q[keep], reference_frame="/".join(sorted(frames)),
                           body_frame=self.options["gt_body_frame"], source=f"bag topic {topic}",
                           notes=notes)

    def T_gt_body_sensor(self, sensor_frame: str) -> Optional[np.ndarray]:
        tree = self.static_tf()
        T_body_cam0 = tree.lookup(self.options["gt_body_frame"], self.options["cam0_frame"])
        if T_body_cam0 is None:
            return None
        if sensor_frame == "cam0":
            return T_body_cam0
        if sensor_frame == "imu":
            return T_body_cam0 @ self._T_cam0_imu()[0]
        return None

    def validate(self, mode: SensorMode):
        problems = super().validate(mode)
        if not problems and mode in (SensorMode.STEREO, SensorMode.STEREO_INERTIAL):
            try:
                self.stereo_calibration()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"stereo calibration invalid: {exc}")
        return problems

    # ---- conversion ------------------------------------------------------
    def prepare_offline(self, mode: SensorMode, out_dir: Path) -> PreparedInput:
        import cv2

        if mode not in (SensorMode.STEREO, SensorMode.STEREO_INERTIAL, SensorMode.MONO,
                        SensorMode.MONO_INERTIAL):
            raise ValueError(f"offline image conversion does not apply to mode {mode.value}")
        if self.options["image_format"] != "png_bgr":
            raise ValueError("only image_format=png_bgr is implemented")
        out_dir = Path(out_dir)
        stereo = mode in (SensorMode.STEREO, SensorMode.STEREO_INERTIAL)
        (out_dir / "cam0").mkdir(parents=True, exist_ok=True)
        if stereo:
            (out_dir / "cam1").mkdir(exist_ok=True)
        lt, rt = self.options["left_topic"], self.options["right_topic"]
        topics = [lt, rt] if stereo else [lt]
        pending: Dict[int, Dict[str, Any]] = {}
        written = {}
        stats = {"left_msgs": 0, "right_msgs": 0, "duplicate_stamps": 0, "non_opaque_alpha_frames": 0,
                 "encodings": set(), "sizes": set(), "bag_minus_header_ns": []}

        def save(path, img):
            if not cv2.imwrite(str(path), img):
                raise IOError(f"failed to write {path}")

        with ThreadPoolExecutor(max_workers=8) as pool, self._reader() as r:
            conns = [c for c in r.connections if c.topic in topics]
            futures = deque()
            for c, t_bag, raw in r.messages(connections=conns):
                m = r.deserialize(raw, c.msgtype)
                side = "cam0" if c.topic == lt else "cam1"
                stats["left_msgs" if side == "cam0" else "right_msgs"] += 1
                ts = _stamp_ns(m.header)
                img, alpha = _decode_image(m)
                stats["encodings"].add(m.encoding)
                stats["sizes"].add((int(m.width), int(m.height)))
                if alpha is not None and alpha.min() < 255:
                    stats["non_opaque_alpha_frames"] += 1
                if side == "cam0" and len(stats["bag_minus_header_ns"]) < 5000:
                    stats["bag_minus_header_ns"].append(t_bag - ts)
                slot = pending.setdefault(ts, {})
                if side in slot or (ts in written and side in written[ts]):
                    stats["duplicate_stamps"] += 1
                    continue
                rel = f"{side}/{ts}.png"
                futures.append(pool.submit(save, out_dir / rel, img.copy()))
                while len(futures) > 64:  # bound memory held by queued images
                    futures.popleft().result()
                slot[side] = rel
                written.setdefault(ts, set()).add(side)
            for f in futures:
                f.result()

        rows, unpaired = [], {"cam0_only": 0, "cam1_only": 0}
        for ts in sorted(pending):
            s = pending[ts]
            if stereo:
                if "cam0" in s and "cam1" in s:
                    rows.append((ts, s["cam0"], s["cam1"]))
                else:
                    unpaired["cam0_only" if "cam0" in s else "cam1_only"] += 1
            elif "cam0" in s:
                rows.append((ts, s["cam0"], ""))
        n = write_frames_csv(out_dir / "frames.csv", rows)

        n_imu, imu_stats = 0, {}
        if mode.is_inertial:
            t, w, a = [], [], []
            for _, m in self._iter(self.options["imu_topic"]):
                t.append(_stamp_ns(m.header))
                w.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
                a.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
            t, w, a = np.array(t, dtype=np.int64), np.array(w), np.array(a)
            order = np.argsort(t, kind="stable")
            t, w, a = t[order], w[order], a[order]
            keep = np.r_[True, np.diff(t) > 0]
            n_imu = write_imu_csv(out_dir / "imu.csv", t[keep], w[keep], a[keep])
            dt = np.diff(t[keep]) * 1e-9
            imu_stats = {"imu_topic": self.options["imu_topic"], "imu_dropped_duplicates": int((~keep).sum()),
                         "imu_dt_median_s": float(np.median(dt)), "imu_dt_max_s": float(dt.max())}

        lag = np.array(stats.pop("bag_minus_header_ns"), dtype=float) * 1e-9
        frame_dt = np.diff([r[0] for r in rows]) * 1e-9 if len(rows) > 1 else np.array([np.nan])
        manifest = {
            "dataset": self.name, "sequence": self.sequence, "mode": mode.value,
            "source": str(self.bag_dir), "n_frames": n, "n_imu": n_imu,
            "topics": topics, "left_msgs": stats["left_msgs"], "right_msgs": stats["right_msgs"],
            "unpaired_frames_dropped": unpaired, "duplicate_stamps_dropped": stats["duplicate_stamps"],
            "encodings": sorted(stats["encodings"]), "sizes": sorted(stats["sizes"]),
            "non_opaque_alpha_frames": stats["non_opaque_alpha_frames"],
            "timestamps": "sensor_msgs/Image header.stamp (ZED SDK capture time), integer ns, unchanged; "
                          "left/right paired by exact stamp equality",
            "bag_receive_minus_header_s": {"median": float(np.median(lag)), "max": float(lag.max())}
            if len(lag) else None,
            "frame_dt_s": {"median": float(np.nanmedian(frame_dt)), "max": float(np.nanmax(frame_dt))},
            "images": "lossless PNG; bgra8 -> bgr8 by dropping the alpha channel (no other processing)",
            "preprocessing": [],
            **imu_stats,
        }
        return PreparedInput(out_dir, mode, n, n_imu, manifest)
