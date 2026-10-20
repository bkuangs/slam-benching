"""FAST-LIO2 adapter (pinned hku-mars/FAST_LIO 7cc4175, LiDAR-inertial, run inside an Apptainer image).

The image is built from third_party/fastlio/fastlio.def. FAST-LIO's ``laserMapping`` node is patched
(third_party/fastlio/patches) to read the slambench offline LiDAR format instead of ROS topics
(third_party/fastlio/slambench_offline.hpp): every scan and IMU sample is handed to FAST-LIO's own
callbacks in data-time order, at ``playback_rate`` x real time (1.0 = as a live sensor; scans queue
when processing is slower, none are dropped). A ROS master is started inside the container only
because the node needs one for its parameters and (unused) publishers.

Raw outputs (algorithm/):

* frame_log.csv      one row per input scan: status, LiDAR pose right after the scan's iterated
                     Kalman filter update (world <- LiDAR), IMU pose, extrinsic, velocity, biases,
                     point counts and timing
* fastlio_map.ply    voxel centroids (``map_voxel`` m) of every processed scan's motion-compensated
                     points placed at that scan's updated pose (FAST-LIO's ikd-tree map is
                     downsampled to ``filter_size_map`` and is not exported)
* driver_status.json, driver_timing.json, params.yaml

FAST-LIO has no loop closure or later optimization: the pose emitted after each scan is final, so
the online and final trajectories are identical (both are written; one world frame).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..trajectory import Trajectory, write_tum
from ..types import RunStatus, SensorMode
from .base import (FRAME_DIAGNOSTICS_COLUMNS, AlgorithmAdapter, AlgorithmOutput, RunContext,
                   TrajectoryOutput)

FASTLIO_COMMIT = "7cc4175de6f8ba2edf34bab02a42195b141027e9"
IKDTREE_COMMIT = "e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4"
DEFAULT_CONTAINER = "~/scratch/slam-bench-work/containers/fastlio_7cc4175.sif"
DRIVER_SOURCES = ("slambench_offline.hpp",)
DRIVER_SOURCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "fastlio"
BINARY = "/opt/catkin_ws/devel/lib/fast_lio/fastlio_mapping"
POSE_COLUMNS = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
MAP_NAME = "fastlio_map.ply"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "playback_rate": 1.0,       # 1.0 = real-time input, 0 = as fast as FAST-LIO consumes it
    # preprocess (Livox CustomMsg path, lidar_type 1)
    "scan_line": 6,             # points with line >= scan_line are dropped (Livox HAP: 6 lines)
    "blind": 0.5,               # m; nearer points dropped (robot body). FAST-LIO's Avia config: 4 m
    "point_filter_num": 3,      # keep every n-th point
    # mapping
    "filter_size_surf": 0.2,    # m; scan downsampling voxel (Avia config 0.5; LunaPolaris is a ~10 m arena)
    "filter_size_map": 0.2,     # m; ikd-tree map voxel
    "max_iteration": 3,         # iterated EKF iterations
    "cube_side_length": 1000.0, # m; local map cube
    "det_range": 450.0,         # m
    "fov_degree": 120.0,        # Livox HAP: 120 deg horizontal
    # IMU noise: FAST-LIO defaults (not calibrated for the ZED2 IMU)
    "acc_cov": 0.1, "gyr_cov": 0.1, "b_acc_cov": 0.0001, "b_gyr_cov": 0.0001,
    "extrinsic_est_en": False,  # True: estimate the LiDAR->IMU extrinsic online
    "time_offset_lidar_to_imu": None,  # s; None = dataset lidar_calibration time_offset_s (or 0)
    # slambench map export
    "map_voxel": 0.02,          # m; fastlio_map.ply voxel (0 = no map)
    "map_min_count": 1,         # minimum points per exported voxel
}


def driver_source_sha1() -> Optional[str]:
    """sha1 of the offline header and FAST-LIO patches in this repository; must equal the image's
    driver_source_sha1 (fastlio.def concatenates the same files in the same order)."""
    h = hashlib.sha1()
    try:
        files = [DRIVER_SOURCE_DIR / n for n in DRIVER_SOURCES]
        files += sorted((DRIVER_SOURCE_DIR / "patches").glob("*.patch"), key=lambda p: p.name)
        for p in files:
            h.update(p.read_bytes())
    except OSError:
        return None
    return h.hexdigest()


def _float(v: Optional[str]) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


class FastLioAdapter(AlgorithmAdapter):
    name = "fastlio"
    supported_modes = frozenset({SensorMode.LIDAR_INERTIAL})

    # ---- configuration ---------------------------------------------------
    @property
    def container(self) -> Path:
        p = self.options.get("container") or os.environ.get("SLAMBENCH_FASTLIO_SIF") or DEFAULT_CONTAINER
        return Path(p).expanduser()

    @property
    def apptainer(self) -> str:
        return self.options.get("apptainer") or shutil.which("apptainer") or "apptainer"

    def resolve_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(settings) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"unknown fastlio settings: {sorted(unknown)}")
        out = dict(DEFAULT_SETTINGS)
        out.update(settings)
        for k in ("blind", "filter_size_surf", "filter_size_map", "map_voxel", "playback_rate",
                  "acc_cov", "gyr_cov", "b_acc_cov", "b_gyr_cov"):
            if not float(out[k]) >= 0:
                raise ValueError(f"{k} must be >= 0, got {out[k]}")
        for k in ("filter_size_surf", "filter_size_map", "cube_side_length", "det_range", "fov_degree"):
            if not float(out[k]) > 0:
                raise ValueError(f"{k} must be > 0, got {out[k]}")
        for k in ("scan_line", "point_filter_num", "max_iteration", "map_min_count"):
            v = out[k]
            if isinstance(v, bool) or int(v) != v or int(v) < 1:
                raise ValueError(f"{k} must be an integer >= 1, got {v}")
        return out

    def validate(self, mode, dataset, settings) -> List[str]:
        problems = super().validate(mode, dataset, settings)
        if problems:
            return problems
        if not self.container.exists():
            problems.append(f"FAST-LIO container not found: {self.container} "
                            "(build with third_party/fastlio/build_container.sh)")
        if not getattr(dataset, "supports_lidar", False) or not hasattr(dataset, "lidar_calibration"):
            return problems + [f"dataset {dataset.name!r} provides no LiDAR calibration"]
        try:
            calib = dataset.lidar_calibration()
            T = np.asarray(calib["T_imu_lidar"], dtype=float)
            if T.shape != (4, 4) or not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6):
                problems.append("lidar_calibration T_imu_lidar is not a rigid 4x4 transform")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"cannot read LiDAR calibration: {exc}")
        return problems

    def estimate_frame(self, mode: SensorMode) -> str:
        return "lidar"

    # ---- versions ----------------------------------------------------------
    def _sif_sha256(self) -> Optional[str]:
        sif = self.container
        if not sif.exists():
            return None
        st = sif.stat()
        cache = sif.with_name(sif.name + ".sha256.json")
        if cache.exists():
            c = json.loads(cache.read_text())
            if c.get("size") == st.st_size and c.get("mtime") == st.st_mtime:
                return c["sha256"]
        h = hashlib.sha256()
        with open(sif, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        digest = h.hexdigest()
        try:
            cache.write_text(json.dumps({"size": st.st_size, "mtime": st.st_mtime, "sha256": digest}))
        except OSError:
            pass
        return digest

    def version_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"expected_fastlio_commit": FASTLIO_COMMIT,
                                "expected_ikdtree_commit": IKDTREE_COMMIT,
                                "container": str(self.container), "container_sha256": self._sif_sha256()}
        try:
            info["apptainer"] = subprocess.run([self.apptainer, "--version"], capture_output=True,
                                               text=True, timeout=60).stdout.strip()
            out = subprocess.run([self.apptainer, "exec", str(self.container), "cat",
                                  "/opt/slambench/versions.txt"], capture_output=True, text=True,
                                 timeout=120)
            info["container_versions"] = dict(l.split("=", 1) for l in out.stdout.splitlines() if "=" in l)
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)
        cv = info.get("container_versions", {})
        commit = cv.get("fastlio_commit")
        if commit and commit != FASTLIO_COMMIT:
            raise RuntimeError(f"container FAST-LIO commit {commit} != pinned {FASTLIO_COMMIT}")
        if cv.get("ikdtree_commit") and cv["ikdtree_commit"] != IKDTREE_COMMIT:
            raise RuntimeError(f"container ikd-Tree commit {cv['ikdtree_commit']} != pinned {IKDTREE_COMMIT}")
        info["expected_driver_source_sha1"] = expected = driver_source_sha1()
        found = cv.get("driver_source_sha1")
        if commit and expected and found != expected:
            raise RuntimeError(f"container driver source {found or '(not recorded)'} differs from this "
                               f"repository's driver {expected}; rebuild the image with "
                               "third_party/fastlio/build_container.sh")
        return info

    # ---- native configuration ---------------------------------------------
    def parameters(self, ctx: RunContext) -> Dict[str, Any]:
        """FAST-LIO ROS parameters (global names, as its launch files load them)."""
        s = ctx.settings
        calib = ctx.dataset.lidar_calibration()
        T = np.asarray(calib["T_imu_lidar"], dtype=float)
        offset = s["time_offset_lidar_to_imu"]
        if offset is None:
            offset = calib.get("time_offset_s") or 0.0
        return {
            "common": {"lid_topic": "/slambench/lidar", "imu_topic": "/slambench/imu",
                       "time_sync_en": False, "time_offset_lidar_to_imu": float(offset)},
            "preprocess": {"lidar_type": 1, "scan_line": int(s["scan_line"]), "blind": float(s["blind"]),
                           "timestamp_unit": 3, "scan_rate": 10},
            "mapping": {"acc_cov": float(s["acc_cov"]), "gyr_cov": float(s["gyr_cov"]),
                        "b_acc_cov": float(s["b_acc_cov"]), "b_gyr_cov": float(s["b_gyr_cov"]),
                        "fov_degree": float(s["fov_degree"]), "det_range": float(s["det_range"]),
                        "extrinsic_est_en": bool(s["extrinsic_est_en"]),
                        "extrinsic_T": [float(x) for x in T[:3, 3]],
                        "extrinsic_R": [float(x) for x in T[:3, :3].reshape(-1)]},   # row major
            "publish": {"path_en": False, "scan_publish_en": False, "dense_publish_en": False,
                        "scan_bodyframe_pub_en": False},
            "pcd_save": {"pcd_save_en": False, "interval": -1},
            "point_filter_num": int(s["point_filter_num"]),
            "max_iteration": int(s["max_iteration"]),
            "filter_size_corner": float(s["filter_size_surf"]),
            "filter_size_surf": float(s["filter_size_surf"]),
            "filter_size_map": float(s["filter_size_map"]),
            "cube_side_length": float(s["cube_side_length"]),
            "feature_extract_enable": False,
            "runtime_pos_log_enable": False,
            "slambench": {"input_dir": str(ctx.prepared.directory.resolve()),
                          "output_dir": str(ctx.algo_dir.resolve()),
                          "playback_rate": float(s["playback_rate"]),
                          "map_voxel": float(s["map_voxel"]), "map_min_count": int(s["map_min_count"])},
        }

    def prepare(self, ctx: RunContext) -> None:
        ctx.algo_dir.mkdir(parents=True, exist_ok=True)
        calib = ctx.dataset.lidar_calibration()
        text = ("# Generated by slam-benching: FAST-LIO ROS parameters (loaded with rosparam at /).\n"
                f"# extrinsic (LiDAR pose in the IMU frame): {calib.get('source', '')}\n"
                + yaml.safe_dump(self.parameters(ctx), sort_keys=True, default_flow_style=None))
        (ctx.algo_dir / "params.yaml").write_text(text)

    def _binds(self, ctx: RunContext) -> List[str]:
        paths = {ctx.prepared.directory.resolve(), ctx.algo_dir.resolve()}
        for p in ctx.prepared.directory.iterdir():
            if p.is_symlink():
                paths.add(p.resolve())
        out = []
        for p in sorted(paths):
            if not any(str(p).startswith(str(q) + os.sep) for q in paths if q != p):
                out.append(str(p))
        return out

    def command(self, ctx: RunContext) -> List[str]:
        # A private ROS master on a free port, the parameters, then FAST-LIO; the node's exit code
        # is the run's exit code.
        d = shlex.quote(str(ctx.algo_dir.resolve()))
        script = (
            ". /opt/ros/noetic/setup.sh; . /opt/catkin_ws/devel/setup.sh; "
            "PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind((\"127.0.0.1\", 0)); "
            "print(s.getsockname()[1])'); "
            "export ROS_MASTER_URI=http://127.0.0.1:$PORT ROS_IP=127.0.0.1 "
            f"ROS_HOME={d}/ros_home ROS_LOG_DIR={d}/ros_home/log; "
            f"rosmaster --core -p $PORT > {d}/rosmaster.log 2>&1 & MASTER=$!; "
            "for i in $(seq 100); do rosparam list >/dev/null 2>&1 && break; sleep 0.2; done; "
            f"rosparam load {d}/params.yaml / || {{ kill $MASTER; exit 4; }}; "
            f"{BINARY} __name:=laserMapping; rc=$?; "
            "kill $MASTER 2>/dev/null; wait $MASTER 2>/dev/null; exit $rc")
        cmd = [self.apptainer, "exec", "--cleanenv"]
        for b in self._binds(ctx):
            cmd += ["-B", b]
        return cmd + [str(self.container), "/bin/bash", "-c", script]

    # ---- outputs -----------------------------------------------------------
    @staticmethod
    def read_frame_log(path: Path):
        """frame_log.csv -> (rows, n_malformed); a row's pose is None unless valid."""
        rows: List[Dict[str, Any]] = []
        malformed = 0
        if not path.exists():
            return rows, malformed
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    row = {"frame_idx": int(r["frame_idx"]), "timestamp_ns": int(r["timestamp_ns"]),
                           "status": r["status"], "pose_valid": int(r["pose_valid"]) == 1,
                           "track_time_s": float(r["track_time_s"] or 0.0),
                           "lidar_end_time": _float(r.get("lidar_end_time")),
                           "n_imu": int(r.get("n_imu") or 0),
                           "n_points_raw": int(r.get("n_points_raw") or 0),
                           "n_effective": int(r.get("n_effective") or 0),
                           "feed_wall_s": _float(r.get("feed_wall_s")),
                           "done_wall_s": _float(r.get("done_wall_s"))}
                except (TypeError, ValueError, KeyError):
                    malformed += 1
                    continue
                pose = None
                if row["pose_valid"]:
                    vals = [_float(r.get(k)) for k in POSE_COLUMNS]
                    if all(v is not None for v in vals):
                        v = np.array(vals)
                        if abs(np.linalg.norm(v[3:]) - 1.0) < 1e-3:
                            pose = v
                row["pose"] = pose
                rows.append(row)
        return rows, malformed

    def collect(self, ctx: RunContext, returncode: Optional[int], timed_out: bool) -> AlgorithmOutput:
        d = ctx.algo_dir
        frames_ts = np.loadtxt(ctx.prepared.frames_csv, delimiter=",", skiprows=1, usecols=0,
                               dtype=np.int64, ndmin=1)
        extra: Dict[str, Any] = {}
        driver_status: Dict[str, Any] = {}
        if (d / "driver_status.json").exists():
            try:
                driver_status = json.loads((d / "driver_status.json").read_text())
            except json.JSONDecodeError as exc:
                extra["driver_status_unreadable"] = str(exc)
        extra["driver_status"] = driver_status
        if (d / "driver_timing.json").exists():
            try:
                extra["driver_timing"] = json.loads((d / "driver_timing.json").read_text())
            except ValueError as exc:
                extra["driver_timing_unreadable"] = str(exc)

        rows, n_malformed = self.read_frame_log(d / "frame_log.csv")
        posed = [r for r in rows if r["pose"] is not None]
        statuses: Dict[str, int] = {}
        for r in rows:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        end_diff = [abs(r["lidar_end_time"] * 1e9 - r["timestamp_ns"]) * 1e-9 for r in posed
                    if r["lidar_end_time"] is not None]
        lag = [r["done_wall_s"] - r["feed_wall_s"] for r in posed
               if r["done_wall_s"] is not None and r["feed_wall_s"] is not None]
        extra.update({"n_frame_log_rows_malformed": n_malformed, "n_frames_input": int(len(frames_ts)),
                      "n_frames_logged": len(rows), "n_frames_tracking_ok": len(posed),
                      "status_counts": statuses,
                      # FAST-LIO's scan end time vs the frame timestamp (must be ~0: poses are stamped
                      # with the frame timestamp)
                      "max_abs_lidar_end_minus_frame_ts_s": max(end_diff) if end_diff else None,
                      "n_effective_points_median": float(np.median([r["n_effective"] for r in posed]))
                      if posed else None,
                      # input-to-result latency (wall) incl. queueing behind slower scans
                      "latency_s": {"median": float(np.median(lag)), "p95": float(np.percentile(lag, 95)),
                                    "max": float(np.max(lag))} if lag else None})

        trajectories: List[TrajectoryOutput] = []
        traj_path = None
        if posed:
            P = np.array([r["pose"] for r in posed])
            ts = np.array([r["timestamp_ns"] for r in posed], dtype=np.int64)
            for kind, name, desc in (
                    ("online", "trajectory_online.tum",
                     "LiDAR pose right after each scan's iterated EKF update"),
                    ("final", "trajectory.tum",
                     "same as online: FAST-LIO has no loop closure or later optimization")):
                traj = Trajectory(ts, P[:, :3], P[:, 3:], frame="lidar",
                                  meta={"source": str(d / "frame_log.csv"), "estimator": "FAST-LIO2",
                                        "content": "world <- LiDAR at the scan end time (frame timestamp)"})
                path = ctx.run_dir / name
                write_tum(traj, path)
                trajectories.append(TrajectoryOutput(kind, path, "lidar", desc, coordinate_sessions="single"))
                if kind == "final":
                    traj_path = path
        posed_ts = {r["timestamp_ns"] for r in posed}

        diag_path = None
        if rows:
            diag_path = ctx.run_dir / "frame_diagnostics.csv"
            with open(diag_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(FRAME_DIAGNOSTICS_COLUMNS)
                for r in rows:
                    ok = r["pose"] is not None
                    w.writerow([r["timestamp_ns"], r["status"], int(ok), 0, r["track_time_s"],
                                int(r["timestamp_ns"] in posed_ts), 0])

        artifacts: Dict[str, Path] = {}
        for key, name in (("lidar_map", MAP_NAME), ("frame_log", "frame_log.csv")):
            if (d / name).exists() and (d / name).stat().st_size > 0:
                artifacts[key] = d / name
        avail = {"tracking_state": bool(rows), "map_id": False, "tracked_features": False,
                 "sparse_map": False, "coordinate_session": bool(rows), "per_frame_time": bool(rows),
                 "online_pose": bool(posed)}
        common = dict(trajectory=traj_path, estimate_frame="lidar", frame_diagnostics=diag_path,
                      diagnostics_available=avail, extra=extra, trajectories=trajectories,
                      artifacts=artifacts)
        if timed_out:
            return AlgorithmOutput(RunStatus.TIMEOUT, "execution timed out", **common)
        if returncode is None or returncode != 0:
            sig = f" (signal {-returncode})" if returncode is not None and returncode < 0 else ""
            msg = f"FAST-LIO exited with code {returncode}{sig}"
            if driver_status.get("status") and driver_status["status"] != "completed":
                msg += f"; driver status {driver_status['status']}: {driver_status.get('message', '')}"
            return AlgorithmOutput(RunStatus.CRASH, msg, **common)
        if not rows:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, "frame_log.csv missing", **common)
        if driver_status.get("status") != "completed":
            return AlgorithmOutput(RunStatus.CRASH, f"driver status {driver_status.get('status')}: "
                                   f"{driver_status.get('message', '')}", **common)
        if not posed:
            return AlgorithmOutput(RunStatus.INIT_FAILURE,
                                   f"no scan produced a pose in {len(rows)} frames ({statuses})", **common)
        msg = ""
        if ctx.settings["map_voxel"] > 0 and "lidar_map" not in artifacts:
            msg = f"{MAP_NAME} missing"
        return AlgorithmOutput(RunStatus.SUCCESS, msg, **common)
