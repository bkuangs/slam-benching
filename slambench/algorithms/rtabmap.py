"""RTAB-Map adapter (pinned tag 0.23.8, stereo only, run inside an Apptainer image).

The image is built from third_party/rtabmap/rtabmap.def and contains a headless driver
(third_party/rtabmap/slambench_rtabmap_driver.cc) that runs RTAB-Map's stereo visual odometry and
graph SLAM on the slambench offline format, then ``rtabmap-export`` builds the maps. Raw outputs:

* frame_log.csv   per-frame odometry/SLAM diagnostics, online cam0 pose (map correction x
                  odometry) and raw cam0 odometry pose
* nodes.csv       map nodes with their odometry pose and final optimized pose
* links.csv       constraints of the final graph (neighbor, loop closure, proximity, ...)
* rtabmap.db      RTAB-Map database = reusable localization map (images, visual words, graph,
                  per-node local occupancy grids)
* rtabmap_cloud.ply, rtabmap_octomap.bt, rtabmap_octomap.ply
                  dense coloured cloud and 3D OctoMap (occupied / free / unknown) assembled from
                  the final optimized poses (``rtabmap-export --opt 2``)

The final trajectory has a pose for every frame with valid odometry: the frame's odometry pose
re-anchored on the most recent earlier node of the final graph,
``T_final = T_opt(node) * T_odom(node)^-1 * T_odom(frame)`` (the same reference-keyframe
construction ORB-SLAM3 uses for its frame trajectory).
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
from scipy.spatial.transform import Rotation

from ..trajectory import Trajectory, invert_T, make_T, write_tum
from ..types import RunStatus, SensorMode
from .base import (FRAME_DIAGNOSTICS_COLUMNS, AlgorithmAdapter, AlgorithmOutput, RunContext,
                   TrajectoryOutput)

RTABMAP_COMMIT = "d069becc724905d6465a8758c40dae88e13bb9e8"   # tag 0.23.8
RTABMAP_VERSION = "0.23.8"
DEFAULT_CONTAINER = "~/scratch/slam-bench-work/containers/rtabmap_0.23.8.sif"
DRIVER_SOURCES = ("slambench_rtabmap_driver.cc",)
DRIVER_SOURCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "rtabmap"
POSE_COLUMNS = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
ODOM_COLUMNS = tuple("odom_" + c for c in POSE_COLUMNS)
EXPORT_NAME = "rtabmap"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "detection_rate_hz": 1.0,      # map update rate on data time (Rtabmap/DetectionRate); 0 = every frame
    "vis_max_features": 1000,      # Vis/MaxFeatures (odometry + loop-closure registration)
    # Stereo/OpticalFlow: RTAB-Map's default KLT stereo correspondences gave a ~30 deg/m pitch drift on
    # LunaPolaris (forward motion absorbed as rotation); False = descriptor/SSD stereo matching, which
    # matched ORB-SLAM3's motion direction and orientation.
    "stereo_optical_flow": False,
    "kp_max_features": 500,        # Kp/MaxFeatures (bag-of-words loop-closure detection)
    "loop_closure": True,          # False -> Kp/MaxFeatures=-1 and no proximity detection
    "grid_cell_size": 0.05,        # Grid/CellSize (m), local grids and OctoMap resolution
    "grid_range_max": 5.0,         # Grid/RangeMax (m) of stereo points used for the grids
    "grid_ray_tracing": True,      # Grid/RayTracing: mark free space between camera and obstacles
    "grid_ground_is_obstacle": True,  # Grid/GroundIsObstacle: no ground segmentation (aerial use)
    "playback_rate": 1.0,          # 1.0 = real-time throttled input, 0 = as fast as possible
    "log_features": True,          # odometry keypoints + status per frame (features.bin)
    "export_cloud": True,          # rtabmap_cloud.ply
    "export_octomap": True,        # rtabmap_octomap.bt / .ply
    "cloud_voxel": 0.02,           # rtabmap-export --voxel (m)
    "cloud_max_range": 5.0,        # rtabmap-export --max_range (m)
    "cloud_decimation": 2,         # rtabmap-export --decimation (image decimation)
    "extra_parameters": {},        # raw RTAB-Map parameters, e.g. {"Odom/Strategy": 1}
    # Map reuse: path of an earlier RTAB-Map *mapping* run directory. The run then localizes against
    # a copy of that run's rtabmap.db (Mem/IncrementalMemory=false, Mem/InitWMWithAllNodes=true)
    # instead of building a map; no final trajectory and no map export are produced.
    "map_run": None,
}

# Parameters allowed to differ between a localization run and the mapping run whose map it loads.
_LOCALIZATION_PARAMS = {"Mem/IncrementalMemory": "false", "Mem/InitWMWithAllNodes": "true"}

# Parameters the adapter sets from named settings or relies on; extra_parameters must not override them.
_MANAGED = {"Rtabmap/DetectionRate", "Vis/MaxFeatures", "Stereo/OpticalFlow", "Kp/MaxFeatures",
            "RGBD/ProximityBySpace",
            "Grid/CellSize", "Grid/RangeMax", "Grid/RayTracing", "Grid/GroundIsObstacle", "Grid/3D",
            "Grid/Sensor", "RGBD/CreateOccupancyGrid", "Mem/IncrementalMemory", "Mem/BinDataKept",
            "Mem/InitWMWithAllNodes"}
# Unsupported because the final trajectory construction assumes continuous odometry.
_FORBIDDEN = {"Odom/ResetCountdown": "odometry resets would break the final-trajectory construction",
              "RGBD/NewMapOdomChangeDistance": "new maps on odometry jumps are not handled yet"}

# Optical rotation is applied in the driver; poses in all files are cam0 (left optical) poses.


def driver_source_sha1() -> Optional[str]:
    """sha1 of the driver source in this repository; must equal the image's driver_source_sha1."""
    h = hashlib.sha1()
    try:
        for n in DRIVER_SOURCES:
            h.update((DRIVER_SOURCE_DIR / n).read_bytes())
    except OSError:
        return None
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _param_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _pose_T(v: np.ndarray) -> np.ndarray:
    return make_T(Rotation.from_quat(v[3:]).as_matrix(), v[:3])


def _parse_pose(r: Dict[str, str], cols) -> Optional[np.ndarray]:
    vals = [r.get(k) for k in cols]
    if any(v in (None, "") for v in vals):
        return None
    try:
        v = np.array([float(x) for x in vals])
    except ValueError:
        return None
    if not np.all(np.isfinite(v)) or abs(np.linalg.norm(v[3:]) - 1.0) > 1e-3:
        return None
    return v


class RtabmapAdapter(AlgorithmAdapter):
    name = "rtabmap"
    supported_modes = frozenset({SensorMode.STEREO})

    # ---- configuration ---------------------------------------------------
    @property
    def container(self) -> Path:
        p = self.options.get("container") or os.environ.get("SLAMBENCH_RTABMAP_SIF") or DEFAULT_CONTAINER
        return Path(p).expanduser()

    @property
    def apptainer(self) -> str:
        return self.options.get("apptainer") or shutil.which("apptainer") or "apptainer"

    def resolve_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(settings) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"unknown rtabmap settings: {sorted(unknown)}")
        out = dict(DEFAULT_SETTINGS)
        out.update(settings)
        out["extra_parameters"] = dict(out["extra_parameters"] or {})
        for k in out["extra_parameters"]:
            if not isinstance(k, str) or "/" not in k:
                raise ValueError(f"extra_parameters keys must be RTAB-Map names like 'Odom/Strategy', got {k!r}")
            if k in _MANAGED:
                raise ValueError(f"extra_parameters must not set {k}; use the adapter setting instead")
            if k in _FORBIDDEN:
                raise ValueError(f"extra_parameters must not set {k}: {_FORBIDDEN[k]}")
        for key in ("grid_cell_size", "grid_range_max", "cloud_voxel", "cloud_max_range"):
            if not float(out[key]) >= 0:
                raise ValueError(f"{key} must be >= 0, got {out[key]}")
        if not float(out["grid_cell_size"]) > 0:
            raise ValueError("grid_cell_size must be > 0")
        if not float(out["detection_rate_hz"]) >= 0:
            raise ValueError(f"detection_rate_hz must be >= 0, got {out['detection_rate_hz']}")
        d = out["cloud_decimation"]
        if isinstance(d, bool) or int(d) != d or int(d) < 1:
            raise ValueError(f"cloud_decimation must be an integer >= 1, got {d}")
        if out["map_run"] is not None:
            out["map_run"] = str(Path(out["map_run"]).expanduser().resolve())
        return out

    # ---- map reuse ---------------------------------------------------------
    @staticmethod
    def localizing(settings: Dict[str, Any]) -> bool:
        return settings.get("map_run") is not None

    @staticmethod
    def map_database(settings: Dict[str, Any]) -> Path:
        return Path(settings["map_run"]) / "algorithm" / "rtabmap.db"

    @staticmethod
    def _read_params_file(path: Path) -> Dict[str, str]:
        out = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    def _map_reuse_problems(self, settings: Dict[str, Any]) -> List[str]:
        run = Path(settings["map_run"])
        problems = []
        for f in ("algorithm/rtabmap.db", "algorithm/rtabmap_params.txt", "algorithm/nodes.csv",
                  "config.resolved.yaml"):
            if not (run / f).is_file():
                problems.append(f"map_run {run} has no {f} (not a completed RTAB-Map mapping run?)")
        if problems:
            return problems
        map_params = self._read_params_file(run / "algorithm" / "rtabmap_params.txt")
        if map_params.get("Mem/IncrementalMemory") != "true":
            problems.append(f"map_run {run} is not a mapping run (Mem/IncrementalMemory != true)")
        mine = self.parameters(settings)
        diff = sorted(k for k in set(mine) | set(map_params)
                      if k not in _LOCALIZATION_PARAMS and mine.get(k) != map_params.get(k))
        if diff:
            problems.append("localization parameters must equal the mapping run's (features, "
                            "vocabulary and registration must match the map); differing: "
                            + ", ".join(f"{k}: map={map_params.get(k)} here={mine.get(k)}" for k in diff))
        return problems

    def validate(self, mode, dataset, settings) -> List[str]:
        problems = super().validate(mode, dataset, settings)
        if problems:
            return problems
        if not self.container.exists():
            problems.append(f"RTAB-Map container not found: {self.container} "
                            "(build with third_party/rtabmap/build_container.sh)")
        try:
            calib = dataset.stereo_calibration()
        except Exception as exc:  # noqa: BLE001
            return problems + [f"cannot read camera calibration: {exc}"]
        if calib is None or calib.right is None:
            return problems + ["RTAB-Map stereo needs a stereo camera calibration"]
        if self.localizing(settings):
            problems += self._map_reuse_problems(settings)
        for cam in (calib.left, calib.right):
            if cam.model != "pinhole" or not cam.rectified or any(abs(float(k)) > 0 for k in cam.distortion):
                problems.append("RTAB-Map adapter needs rectified, undistorted pinhole stereo images "
                                f"(got {cam.model}, rectified={cam.rectified})")
                break
        else:
            L, R = calib.left, calib.right
            if (L.fx, L.fy, L.cx, L.cy) != (R.fx, R.fy, R.cx, R.cy):
                problems.append("rectified stereo pair must share one intrinsic matrix")
        return problems

    def estimate_frame(self, mode: SensorMode) -> str:
        return "cam0"

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
        info: Dict[str, Any] = {"expected_rtabmap_commit": RTABMAP_COMMIT,
                                "expected_rtabmap_version": RTABMAP_VERSION,
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
        commit = info.get("container_versions", {}).get("rtabmap_commit")
        if commit and commit != RTABMAP_COMMIT:
            raise RuntimeError(f"container RTAB-Map commit {commit} != pinned {RTABMAP_COMMIT}")
        info["expected_driver_source_sha1"] = expected = driver_source_sha1()
        found = info.get("container_versions", {}).get("driver_source_sha1")
        if commit and expected and found != expected:
            raise RuntimeError(f"container driver source {found or '(not recorded)'} differs from this "
                               f"repository's driver {expected}; rebuild the image with "
                               "third_party/rtabmap/build_container.sh")
        return info

    # ---- native configuration ---------------------------------------------
    def parameters(self, settings: Dict[str, Any]) -> Dict[str, str]:
        """RTAB-Map parameters passed to Odometry and Rtabmap (everything else = RTAB-Map defaults)."""
        s = settings
        p: Dict[str, Any] = {
            "Rtabmap/DetectionRate": float(s["detection_rate_hz"]),  # recorded in the DB; driver applies it
            "Vis/MaxFeatures": int(s["vis_max_features"]),
            "Stereo/OpticalFlow": bool(s["stereo_optical_flow"]),
            "Kp/MaxFeatures": int(s["kp_max_features"]) if s["loop_closure"] else -1,
            "Mem/IncrementalMemory": True,
            "Mem/BinDataKept": True,               # keep images in the DB (reusable map, dense export)
            "RGBD/CreateOccupancyGrid": True,      # per-node local grids, needed for the OctoMap
            "Grid/Sensor": 1,                      # from stereo depth
            "Grid/3D": True,
            "Grid/RayTracing": bool(s["grid_ray_tracing"]),
            "Grid/GroundIsObstacle": bool(s["grid_ground_is_obstacle"]),
            "Grid/CellSize": float(s["grid_cell_size"]),
            "Grid/RangeMax": float(s["grid_range_max"]),
        }
        if not s["loop_closure"]:
            p["RGBD/ProximityBySpace"] = False
        if self.localizing(s):
            p.update(_LOCALIZATION_PARAMS)
        p.update(s["extra_parameters"])
        return {k: _param_value(v) for k, v in p.items()}

    def prepare(self, ctx: RunContext) -> None:
        ctx.algo_dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Generated by slam-benching; one RTAB-Map parameter per line (others = defaults)."]
        lines += [f"{k}={v}" for k, v in sorted(self.parameters(ctx.settings).items())]
        (ctx.algo_dir / "rtabmap_params.txt").write_text("\n".join(lines) + "\n")
        if self.localizing(ctx.settings):
            src = self.map_database(ctx.settings)
            info = {"map_run": ctx.settings["map_run"], "map_database": str(src),
                    "map_database_sha256_before": _file_sha256(src)}
            dst = ctx.algo_dir / "rtabmap.db"
            shutil.copyfile(src, dst)
            if _file_sha256(dst) != info["map_database_sha256_before"]:
                raise RuntimeError(f"copy of {src} differs from the original")
            (ctx.algo_dir / "map_source.json").write_text(json.dumps(info, indent=2) + "\n")

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

    def driver_args(self, ctx: RunContext) -> List[str]:
        s = ctx.settings
        L = ctx.dataset.stereo_calibration().left
        b = ctx.dataset.stereo_calibration().baseline_m
        cam = ",".join(repr(float(x)) for x in (L.fx, L.fy, L.cx, L.cy, b)) + f",{int(L.width)},{int(L.height)}"
        args = ["/opt/slambench/bin/rtabmap_driver", str((ctx.algo_dir / "rtabmap_params.txt").resolve()),
                str(ctx.prepared.directory.resolve()), str(ctx.algo_dir.resolve()),
                "--camera", cam, "--detection-rate", repr(float(s["detection_rate_hz"])),
                "--playback-rate", str(float(s["playback_rate"]))]
        if s["log_features"]:
            args.append("--log-features")
        if self.localizing(s):
            args.append("--localize")
        return args

    def export_args(self, ctx: RunContext) -> Optional[List[str]]:
        s = ctx.settings
        if self.localizing(s) or not (s["export_cloud"] or s["export_octomap"]):
            return None
        d = str(ctx.algo_dir.resolve())
        args = ["rtabmap-export", "--output_dir", d, "--output", EXPORT_NAME, "--opt", "2"]
        if s["export_cloud"]:
            args += ["--cloud", "--voxel", repr(float(s["cloud_voxel"])),
                     "--max_range", repr(float(s["cloud_max_range"])),
                     "--decimation", str(int(s["cloud_decimation"]))]
        if s["export_octomap"]:
            args.append("--octomap")
        return args + [d + "/rtabmap.db"]

    def command(self, ctx: RunContext) -> List[str]:
        # The driver's exit code is the run's exit code; the map export runs only after a clean
        # driver exit and records its own exit code (export_rc.txt) and log (export.log).
        d = shlex.quote(str(ctx.algo_dir.resolve()))
        script = shlex.join(self.driver_args(ctx)) + "; rc=$?"
        exp = self.export_args(ctx)
        if exp:
            script += (f"; if [ $rc -eq 0 ]; then {shlex.join(exp)} > {d}/export.log 2>&1; "
                       f"echo $? > {d}/export_rc.txt; fi")
        script += "; exit $rc"
        cmd = [self.apptainer, "exec", "--cleanenv"]
        for b in self._binds(ctx):
            cmd += ["-B", b]
        return cmd + [str(self.container), "/bin/bash", "-c", script]

    # ---- outputs -----------------------------------------------------------
    @staticmethod
    def read_frame_log(path: Path):
        """Parse frame_log.csv -> (rows, n_malformed); a row's pose/odom is None unless valid."""
        rows: List[Dict[str, Any]] = []
        malformed = 0
        if not path.exists():
            return rows, malformed
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    row = {"frame_idx": int(r["frame_idx"]), "timestamp_ns": int(r["timestamp_ns"]),
                           "odom_ok": int(r["odom_ok"]) == 1, "map_id": int(r["map_id"]),
                           "node_id": int(r["node_id"]), "coord_session": int(r["coord_session"]),
                           "slam_update": int(r["slam_update"]) == 1,
                           "loop_closure_id": int(r["loop_closure_id"]),
                           "proximity_id": int(r["proximity_id"]),
                           "features": int(r["features"]), "inliers": int(r["inliers"]),
                           "track_time_s": float(r["track_time_s"])}
                except (TypeError, ValueError, KeyError):
                    malformed += 1
                    continue
                try:  # absent in logs of drivers older than the map-reuse mode
                    row.update({"localized": int(r.get("localized") or 0) == 1,
                                "loc_event": int(r.get("loc_event") or 0) == 1,
                                "loc_rejected": int(r.get("loc_rejected") or 0) == 1,
                                "loc_correction_m": float(r.get("loc_correction_m") or 0.0),
                                "slam_time_s": float(r.get("slam_time_s") or 0.0)})
                except ValueError:
                    malformed += 1
                    continue
                row["pose"] = _parse_pose(r, POSE_COLUMNS) if row["odom_ok"] else None
                row["odom"] = _parse_pose(r, ODOM_COLUMNS) if row["odom_ok"] else None
                rows.append(row)
        return rows, malformed

    @staticmethod
    def read_nodes(path: Path) -> List[Dict[str, Any]]:
        nodes = []
        if not path.exists():
            return nodes
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    n = {"id": int(r["node_id"]), "map_id": int(r["map_id"]),
                         "timestamp_ns": int(r["timestamp_ns"]), "in_final": int(r["in_final"]) == 1}
                except (TypeError, ValueError, KeyError):
                    continue
                n["odom"] = _parse_pose(r, ODOM_COLUMNS)
                n["opt"] = _parse_pose(r, POSE_COLUMNS) if n["in_final"] else None
                nodes.append(n)
        return nodes

    @staticmethod
    def final_poses(log_rows, nodes):
        """Final cam0 pose for each frame with valid odometry, re-anchored on the latest earlier
        final-graph node (the first one for frames before it) of the same map.
        Returns (timestamps_ns, 4x4 poses)."""
        anchors = sorted((n for n in nodes if n["in_final"] and n["opt"] is not None and n["odom"] is not None),
                         key=lambda n: n["timestamp_ns"])
        if not anchors:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 4, 4))
        final_maps = {n["map_id"] for n in anchors}
        a_ts = np.array([n["timestamp_ns"] for n in anchors], dtype=np.int64)
        corr = [_pose_T(n["opt"]) @ invert_T(_pose_T(n["odom"])) for n in anchors]
        ts, poses = [], []
        for r in log_rows:
            if r["odom"] is None or (r["map_id"] >= 0 and r["map_id"] not in final_maps):
                continue
            k = max(int(np.searchsorted(a_ts, r["timestamp_ns"], side="right")) - 1, 0)
            ts.append(r["timestamp_ns"])
            poses.append(corr[k] @ _pose_T(r["odom"]))
        return np.array(ts, dtype=np.int64), np.array(poses).reshape(-1, 4, 4)

    @staticmethod
    def read_pose_csv(path: Path) -> Dict[int, np.ndarray]:
        out = {}
        if not path.exists():
            return out
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                p = _parse_pose(r, POSE_COLUMNS)
                if p is not None and r.get("node_id"):
                    out[int(r["node_id"])] = p
        return out

    def _map_frame_check(self, ctx: RunContext) -> Dict[str, Any]:
        """Loaded map node poses vs the mapping run's final optimized nodes: the localization frame
        must be the mapping run's final frame for its map->GT transform to apply."""
        loaded = self.read_pose_csv(ctx.algo_dir / "map_loaded_nodes.csv")
        mapping = {n["id"]: n["opt"] for n in self.read_nodes(Path(ctx.settings["map_run"]) / "algorithm" / "nodes.csv")
                   if n["in_final"] and n["opt"] is not None}
        common = sorted(set(loaded) & set(mapping))
        out: Dict[str, Any] = {"n_loaded_nodes": len(loaded), "n_mapping_final_nodes": len(mapping),
                               "n_common": len(common)}
        if common:
            dt = [float(np.linalg.norm(loaded[k][:3] - mapping[k][:3])) for k in common]
            dr = [float(np.degrees((Rotation.from_quat(loaded[k][3:]).inv()
                                    * Rotation.from_quat(mapping[k][3:])).magnitude())) for k in common]
            out.update({"max_translation_diff_m": max(dt), "max_rotation_diff_deg": max(dr)})
            out["same_frame"] = bool(len(common) == len(loaded) == len(mapping)
                                     and max(dt) < 1e-3 and max(dr) < 0.05)
        else:
            out["same_frame"] = False
        return out

    def _collect_localization(self, ctx, returncode, timed_out, frames_ts, extra, driver_status):
        d = ctx.algo_dir
        log_rows, n_malformed = self.read_frame_log(d / "frame_log.csv")
        extra["n_frame_log_rows_malformed"] = n_malformed
        src = json.loads((d / "map_source.json").read_text()) if (d / "map_source.json").exists() else {}
        orig = Path(src["map_database"]) if src.get("map_database") else self.map_database(ctx.settings)
        src["map_database_sha256_after"] = _file_sha256(orig) if orig.exists() else None
        src["map_database_unchanged"] = (src["map_database_sha256_after"] is not None and
                                         src["map_database_sha256_after"] == src.get("map_database_sha256_before"))
        load = json.loads((d / "map_load.json").read_text()) if (d / "map_load.json").exists() else {}
        n_ok = sum(r["odom_ok"] for r in log_rows)
        loc = [r for r in log_rows if r["pose"] is not None and r["localized"]]
        events = [r for r in log_rows if r["loc_event"]]
        extra["map_reuse"] = {
            **src, "map_load": load, "map_frame_check": self._map_frame_check(ctx),
            "n_localization_events": len(events),
            "n_rejected_hypotheses": sum(r["loc_rejected"] for r in log_rows),
            "first_localization_frame_idx": next((r["frame_idx"] for r in log_rows if r["localized"]), None),
            "n_localized_poses": len(loc),
            "n_prior_only_poses": sum(r["pose"] is not None and not r["localized"] for r in log_rows)}
        extra.update({"n_frames_input": int(len(frames_ts)), "n_frames_logged": len(log_rows),
                      "n_frames_tracking_ok": n_ok, "n_online_poses": len(loc)})

        trajectories: List[TrajectoryOutput] = []
        if loc:
            P = np.array([r["pose"] for r in loc])
            otraj = Trajectory(np.array([r["timestamp_ns"] for r in loc], dtype=np.int64), P[:, :3], P[:, 3:],
                               frame="cam0",
                               meta={"source": str(d / "frame_log.csv"), "estimator": "RTAB-Map localization",
                                     "content": "map correction x odometry in the loaded map's frame, frames "
                                                "from the first accepted localization on; left camera frame"})
            online_path = ctx.run_dir / "trajectory_online.tum"
            write_tum(otraj, online_path)
            trajectories.append(TrajectoryOutput(
                "online", online_path, "cam0",
                "pose in the saved map's frame after each frame (localization mode), from the first "
                "accepted localization on", coordinate_sessions="single"))

        diag_path = None
        if log_rows:
            diag_path = ctx.run_dir / "frame_diagnostics.csv"
            with open(diag_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(FRAME_DIAGNOSTICS_COLUMNS)
                for r in log_rows:
                    state = ("lost" if not r["odom_ok"] else "localized" if r["localized"] else "not_localized")
                    w.writerow([r["timestamp_ns"], state, int(r["odom_ok"] and r["localized"] and r["pose"] is not None),
                                0, r["track_time_s"], 0, 0])

        artifacts = {k: d / n for k, n in (("tracked_features", "features.bin"),
                                             ("map_loaded_nodes", "map_loaded_nodes.csv"))
                     if (d / n).exists() and (d / n).stat().st_size > 0}
        avail = {"tracking_state": bool(log_rows), "map_id": False,
                 "tracked_features": "tracked_features" in artifacts, "sparse_map": False,
                 "coordinate_session": bool(log_rows), "per_frame_time": bool(log_rows),
                 "online_pose": bool(loc)}
        common = dict(frame_diagnostics=diag_path, diagnostics_available=avail, extra=extra,
                      trajectories=trajectories, artifacts=artifacts)
        mr = extra["map_reuse"]
        if timed_out:
            return AlgorithmOutput(RunStatus.TIMEOUT, "execution timed out", **common)
        if returncode is None or returncode != 0:
            msg = f"driver exited with code {returncode}"
            if driver_status.get("status") and driver_status["status"] != "completed":
                msg += f"; driver status {driver_status['status']}: {driver_status.get('message', '')}"
            return AlgorithmOutput(RunStatus.CRASH, msg, **common)
        if not log_rows:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, "frame_log.csv missing", **common)
        if not mr["map_database_unchanged"]:
            return AlgorithmOutput(RunStatus.CRASH, "the mapping run's database changed during the run", **common)
        if not mr["map_frame_check"].get("same_frame"):
            return AlgorithmOutput(RunStatus.CRASH, "loaded map poses differ from the mapping run's final "
                                   f"graph: {mr['map_frame_check']}", **common)
        if not loc:
            return AlgorithmOutput(RunStatus.INIT_FAILURE,
                                   f"never localized against the map in {len(log_rows)} frames "
                                   f"(odometry valid in {n_ok})", **common)
        return AlgorithmOutput(RunStatus.SUCCESS, "", **common)

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
        if self.localizing(ctx.settings):
            return self._collect_localization(ctx, returncode, timed_out, frames_ts, extra, driver_status)

        log_rows, n_malformed = self.read_frame_log(d / "frame_log.csv")
        nodes = self.read_nodes(d / "nodes.csv")
        n_ok = sum(r["odom_ok"] for r in log_rows)
        trajectories: List[TrajectoryOutput] = []

        online = [r for r in log_rows if r["pose"] is not None]
        extra.update({"n_frame_log_rows_malformed": n_malformed, "n_online_poses": len(online),
                      "n_ok_frames_without_valid_online_pose": n_ok - len(online)})
        if online:
            P = np.array([r["pose"] for r in online])
            otraj = Trajectory(np.array([r["timestamp_ns"] for r in online], dtype=np.int64), P[:, :3], P[:, 3:],
                               frame="cam0",
                               meta={"source": str(d / "frame_log.csv"), "estimator": "RTAB-Map",
                                     "content": "map correction x odometry right after each frame "
                                                "(frames with valid odometry); left camera frame"})
            online_path = ctx.run_dir / "trajectory_online.tum"
            write_tum(otraj, online_path)
            trajectories.append(TrajectoryOutput(
                "online", online_path, "cam0",
                "odometry corrected by the latest graph optimization at that moment (what a robot "
                "would use: map->odom x odom->camera)", coordinate_sessions="per_frame"))

        traj_path, final_ts = None, set()
        fts, fposes = self.final_poses(log_rows, nodes)
        if len(fts):
            q = Rotation.from_matrix(fposes[:, :3, :3]).as_quat()
            traj = Trajectory(fts, fposes[:, :3, 3], q, frame="cam0",
                              meta={"source": str(d / "nodes.csv"), "estimator": "RTAB-Map",
                                    "content": "frame odometry re-anchored on the final optimized graph"})
            traj_path = ctx.run_dir / "trajectory.tum"
            write_tum(traj, traj_path)
            final_ts = set(fts.tolist())
            trajectories.append(TrajectoryOutput(
                "final", traj_path, "cam0",
                "after shutdown: T_opt(node) T_odom(node)^-1 T_odom(frame) with the latest earlier node "
                "of the final graph", coordinate_sessions="single"))

        diag_path = None
        if log_rows:
            diag_path = ctx.run_dir / "frame_diagnostics.csv"
            seen_ok = False
            with open(diag_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(FRAME_DIAGNOSTICS_COLUMNS)
                for r in log_rows:
                    seen_ok |= r["odom_ok"]
                    state = "ok" if r["odom_ok"] else ("lost" if seen_ok else "not_initialized")
                    w.writerow([r["timestamp_ns"], state, int(r["pose"] is not None),
                                r["map_id"] if r["map_id"] >= 0 else "", r["track_time_s"],
                                int(r["timestamp_ns"] in final_ts), r["coord_session"]])

        artifacts: Dict[str, Path] = {}
        for key, name in (("tracked_features", "features.bin"), ("rtabmap_db", "rtabmap.db"),
                          ("map_nodes", "nodes.csv"), ("map_links", "links.csv"),
                          ("dense_cloud", f"{EXPORT_NAME}_cloud.ply"),
                          ("octomap", f"{EXPORT_NAME}_octomap.bt"),
                          ("octomap_cloud", f"{EXPORT_NAME}_octomap.ply"),
                          ("map_export_log", "export.log")):
            if (d / name).exists() and (d / name).stat().st_size > 0:
                artifacts[key] = d / name
        export: Dict[str, Any] = {"requested": self.export_args(ctx) is not None}
        if (d / "export_rc.txt").exists():
            try:
                export["returncode"] = int((d / "export_rc.txt").read_text().strip())
            except ValueError:
                export["returncode"] = None
        missing = [k for k, on in (("dense_cloud", ctx.settings["export_cloud"]),
                                   ("octomap", ctx.settings["export_octomap"])) if on and k not in artifacts]
        export["missing"] = missing
        extra["map_export"] = export

        links = {}
        if (d / "links.csv").exists():
            with open(d / "links.csv", newline="") as f:
                for r in csv.DictReader(f):
                    links[r["type"]] = links.get(r["type"], 0) + 1
        extra["graph"] = {"n_nodes_added": len(nodes), "n_nodes_final": sum(n["in_final"] for n in nodes),
                          "map_ids": sorted({n["map_id"] for n in nodes}), "links_by_type": links,
                          "n_loop_closure_detections": sum(r["loop_closure_id"] > 0 for r in log_rows),
                          "n_proximity_detections": sum(r["proximity_id"] > 0 for r in log_rows),
                          "n_slam_updates": sum(r["slam_update"] for r in log_rows),
                          "n_coordinate_sessions": len({r["coord_session"] for r in log_rows})}
        avail = {"tracking_state": bool(log_rows), "map_id": bool(log_rows),
                 "tracked_features": "tracked_features" in artifacts, "sparse_map": False,
                 "coordinate_session": bool(log_rows), "per_frame_time": bool(log_rows),
                 "online_pose": bool(online)}
        extra.update({"n_frames_input": int(len(frames_ts)), "n_frames_logged": len(log_rows),
                      "n_frames_tracking_ok": n_ok, "n_final_poses": len(final_ts)})

        common = dict(trajectory=traj_path, estimate_frame="cam0", frame_diagnostics=diag_path,
                      diagnostics_available=avail, extra=extra, trajectories=trajectories,
                      artifacts=artifacts)
        if timed_out:
            return AlgorithmOutput(RunStatus.TIMEOUT, "execution timed out", **common)
        if returncode is None or returncode != 0:
            sig = f" (signal {-returncode})" if returncode is not None and returncode < 0 else ""
            msg = f"driver exited with code {returncode}{sig}"
            if driver_status.get("status") and driver_status["status"] != "completed":
                msg += f"; driver status {driver_status['status']}: {driver_status.get('message', '')}"
            return AlgorithmOutput(RunStatus.CRASH, msg, **common)
        if not log_rows:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, "frame_log.csv missing", **common)
        if n_ok == 0:
            return AlgorithmOutput(RunStatus.INIT_FAILURE,
                                   f"odometry never produced a pose in {len(log_rows)} frames", **common)
        if traj_path is None:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, "no final graph nodes (nodes.csv)", **common)
        msg = ""
        if export["requested"] and (missing or export.get("returncode") not in (0,)):
            msg = f"map export incomplete (missing {missing}, rc {export.get('returncode')}; see export.log)"
        return AlgorithmOutput(RunStatus.SUCCESS, msg, **common)
