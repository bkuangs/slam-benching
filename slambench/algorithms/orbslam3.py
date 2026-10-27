"""ORB-SLAM3 adapter (pinned commit 4452a3c, run inside an Apptainer image).

The image is built from third_party/orbslam3/orbslam3.def and contains a
headless driver (third_party/orbslam3/slambench_driver.cc) that reads the
slambench offline format and writes:

* frame_log.csv         per-frame tracking state, active map id, per-frame
                        tracking time and online pose
* trajectory_euroc.txt  System::SaveTrajectoryEuRoC - final optimized poses of
                        all non-LOST frames whose reference keyframe is in the
                        *largest* map (camera frame for visual modes, IMU/body
                        frame for inertial modes). Timestamps in ns (float).
* keyframes_euroc.txt   System::SaveKeyFrameTrajectoryEuRoC
* place_recognition.csv every loop/merge check of ORB-SLAM3's place recognition: candidates and
                        the stage at which each was rejected (patch 0005); frame_log.csv also
                        records the active map's keyframe / map-point count per frame

ORB-SLAM3 writes RECENTLY_LOST frames into trajectory_euroc.txt (only LOST is
excluded). The normalized trajectory therefore keeps all saved poses, and the
frame diagnostics mark which ones were tracked OK so the evaluator can
restrict accuracy to valid samples.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..trajectory import Trajectory, invert_T, write_tum
from ..types import RunStatus, SensorMode
from .base import (FRAME_DIAGNOSTICS_COLUMNS, AlgorithmAdapter, AlgorithmOutput, RunContext,
                   TrajectoryOutput)

ORBSLAM3_COMMIT = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
DEFAULT_CONTAINER = "~/scratch/slam-bench-work/containers/orbslam3_4452a3c.sif"

TRACKING_STATES = {-1: "system_not_ready", 0: "no_images_yet", 1: "not_initialized", 2: "ok",
                   3: "recently_lost", 4: "lost", 5: "ok_klt"}
OK_STATES = {2, 5}
POSE_COLUMNS = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
DRIVER_SOURCES = ("slambench_driver.cc", "slambench_preprocess.h")
DRIVER_SOURCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "orbslam3"


def driver_source_sha1() -> Optional[str]:
    """sha1 of the driver sources and ORB-SLAM3 patches in this repository; must equal the image's
    driver_source_sha1 (orbslam3.def concatenates the same files in the same order)."""
    h = hashlib.sha1()
    try:
        files = [DRIVER_SOURCE_DIR / n for n in DRIVER_SOURCES]
        files += sorted((DRIVER_SOURCE_DIR / "patches").glob("*.patch"), key=lambda p: p.name)
        for f in files:
            h.update(f.read_bytes())
    except OSError:
        return None
    return h.hexdigest()

DEFAULT_SETTINGS: Dict[str, Any] = {
    "orb_n_features": 1200,
    "orb_scale_factor": 1.2,
    "orb_n_levels": 8,
    "orb_ini_th_fast": 20,
    "orb_min_th_fast": 7,
    "stereo_th_depth": 40.0,
    "loop_closure": True,
    "camera_fps": None,            # None -> rounded median frame rate of the prepared input
    "clahe": False,                # preprocessing; disabled by default
    "clahe_clip_limit": 3.0,       # CLAHE clip limit (lower = gentler); used only with clahe: true
    "clahe_tile_grid": 8,          # CLAHE tile grid N x N; used only with clahe: true
    "playback_rate": 1.0,          # 1.0 = real-time throttled input, 0 = as fast as possible
    "imu_insert_kfs_when_lost": None,  # None -> ORB-SLAM3 default
    "log_features": True,          # per-frame keypoints + map-point status (features.bin, ~10 KB/frame)
}


def _fmt(x) -> str:
    return repr(float(x))


def _matrix(name: str, M: np.ndarray) -> str:
    vals = ", ".join(f"{float(v):.12g}" for v in np.asarray(M, dtype=float).reshape(-1))
    return (f"{name}: !!opencv-matrix\n  rows: {M.shape[0]}\n  cols: {M.shape[1]}\n  dt: f\n"
            f"  data: [{vals}]\n")


class OrbSlam3Adapter(AlgorithmAdapter):
    name = "orbslam3"
    estimator_label = "ORB-SLAM3"   # Trajectory meta "estimator" written by collect()
    supported_modes = frozenset({SensorMode.MONO, SensorMode.MONO_INERTIAL,
                                 SensorMode.STEREO, SensorMode.STEREO_INERTIAL})

    # ---- configuration ---------------------------------------------------
    @property
    def container(self) -> Path:
        p = self.options.get("container") or os.environ.get("SLAMBENCH_ORBSLAM3_SIF") or DEFAULT_CONTAINER
        return Path(p).expanduser()

    @property
    def apptainer(self) -> str:
        return self.options.get("apptainer") or shutil.which("apptainer") or "apptainer"

    def resolve_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(settings) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"unknown orbslam3 settings: {sorted(unknown)}")
        out = dict(DEFAULT_SETTINGS)
        out.update(settings)
        if not float(out["clahe_clip_limit"]) > 0:
            raise ValueError(f"clahe_clip_limit must be > 0, got {out['clahe_clip_limit']}")
        if int(out["clahe_tile_grid"]) != out["clahe_tile_grid"] or int(out["clahe_tile_grid"]) < 1:
            raise ValueError(f"clahe_tile_grid must be an integer >= 1, got {out['clahe_tile_grid']}")
        return out

    def validate(self, mode, dataset, settings) -> List[str]:
        problems = super().validate(mode, dataset, settings)
        if problems:
            return problems
        if not self.container.exists():
            problems.append(f"ORB-SLAM3 container not found: {self.container} "
                            "(build with third_party/orbslam3/build_container.sh)")
        stereo = mode in (SensorMode.STEREO, SensorMode.STEREO_INERTIAL)
        try:
            calib = dataset.stereo_calibration()
        except Exception as exc:  # noqa: BLE001
            return problems + [f"cannot read camera calibration: {exc}"]
        if calib is None:
            return problems + ["dataset provides no camera calibration"]
        if stereo and calib.right is None:
            return problems + ["dataset provides a single camera only; stereo modes need two"]
        for cam in ([calib.left, calib.right] if stereo else [calib.left]):
            if cam.model not in ("pinhole", "kannala_brandt"):
                problems.append(f"camera model {cam.model} not supported by ORB-SLAM3 adapter")
        if stereo and calib.left.model == "pinhole" and not calib.left.rectified:
            problems.append("unrectified pinhole stereo is not implemented in the adapter yet")
        return problems

    def estimate_frame(self, mode: SensorMode) -> str:
        return "imu" if mode.is_inertial else "cam0"

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
        info: Dict[str, Any] = {"expected_orbslam3_commit": ORBSLAM3_COMMIT,
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
        commit = info.get("container_versions", {}).get("orbslam3_commit")
        if commit and commit != ORBSLAM3_COMMIT:
            raise RuntimeError(f"container ORB-SLAM3 commit {commit} != pinned {ORBSLAM3_COMMIT}")
        info["expected_driver_source_sha1"] = expected = driver_source_sha1()
        found = info.get("container_versions", {}).get("driver_source_sha1")
        # Only judge a readable image (a missing image is reported by validate()).
        if commit and expected and found != expected:
            raise RuntimeError(f"container driver source {found or '(not recorded)'} differs from this "
                               f"repository's driver {expected}; rebuild the image with "
                               "third_party/orbslam3/build_container.sh")
        return info

    # ---- settings file -----------------------------------------------------
    def _settings_yaml(self, ctx: RunContext) -> str:
        s, mode, ds = ctx.settings, ctx.mode, ctx.dataset
        calib = ds.stereo_calibration()
        L, R = calib.left, calib.right
        stereo = mode in (SensorMode.STEREO, SensorMode.STEREO_INERTIAL)
        fps = s["camera_fps"]
        if fps is None:
            dt = ctx.prepared.manifest.get("frame_dt_s", {}).get("median")
            if dt is None:
                ts = np.loadtxt(ctx.prepared.frames_csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
                dt = float(np.median(np.diff(ts))) * 1e-9
            fps = int(round(1.0 / dt))
        lines = ["%YAML:1.0", "# Generated by slam-benching from dataset calibration; do not edit.",
                 'File.version: "1.0"']
        if L.model == "pinhole" and L.rectified and stereo:
            lines += ['Camera.type: "Rectified"']
        elif L.model == "pinhole":
            lines += ['Camera.type: "PinHole"']
        else:
            lines += ['Camera.type: "KannalaBrandt8"']
        cams = [("Camera1", L)] + ([("Camera2", R)] if stereo and not (L.model == "pinhole" and L.rectified) else [])
        for prefix, c in cams:
            lines += [f"{prefix}.fx: {_fmt(c.fx)}", f"{prefix}.fy: {_fmt(c.fy)}",
                      f"{prefix}.cx: {_fmt(c.cx)}", f"{prefix}.cy: {_fmt(c.cy)}"]
            if c.model == "kannala_brandt":
                for i, k in enumerate(c.distortion[:4]):
                    lines.append(f"{prefix}.k{i + 1}: {_fmt(k)}")
            elif c.model == "pinhole" and not (stereo and c.rectified):
                k1, k2, p1, p2 = (list(c.distortion) + [0, 0, 0, 0])[:4]
                lines += [f"{prefix}.k1: {_fmt(k1)}", f"{prefix}.k2: {_fmt(k2)}",
                          f"{prefix}.p1: {_fmt(p1)}", f"{prefix}.p2: {_fmt(p2)}"]
        lines += [f"Camera.width: {L.width}", f"Camera.height: {L.height}",
                  f"Camera.fps: {fps}", "Camera.RGB: 0"]
        yaml_text = "\n".join(lines) + "\n"
        if stereo:
            if L.model == "pinhole" and L.rectified:
                yaml_text += f"Stereo.b: {_fmt(calib.baseline_m)}\n"
            else:
                yaml_text += _matrix("Stereo.T_c1_c2", calib.T_left_right)
                if L.model == "kannala_brandt":
                    # Key names follow src/Settings.cc (upstream TUM-VI.yaml uses "Camera." which aborts).
                    yaml_text += (f"Camera1.overlappingBegin: 0\nCamera1.overlappingEnd: {L.width - 1}\n"
                                  f"Camera2.overlappingBegin: 0\nCamera2.overlappingEnd: {R.width - 1}\n")
            yaml_text += f"Stereo.ThDepth: {_fmt(s['stereo_th_depth'])}\n"
        if mode.is_inertial:
            imu = ds.imu_calibration()
            n = imu.noise
            yaml_text += _matrix("IMU.T_b_c1", invert_T(imu.T_cam0_imu))
            yaml_text += (f"IMU.NoiseGyro: {_fmt(n.gyro_noise_density)}\nIMU.NoiseAcc: {_fmt(n.accel_noise_density)}\n"
                          f"IMU.GyroWalk: {_fmt(n.gyro_random_walk)}\nIMU.AccWalk: {_fmt(n.accel_random_walk)}\n"
                          f"IMU.Frequency: {_fmt(imu.rate_hz)}\n")
            if s["imu_insert_kfs_when_lost"] is not None:
                yaml_text += f"IMU.InsertKFsWhenLost: {int(bool(s['imu_insert_kfs_when_lost']))}\n"
        yaml_text += (f"ORBextractor.nFeatures: {int(s['orb_n_features'])}\n"
                      f"ORBextractor.scaleFactor: {_fmt(s['orb_scale_factor'])}\n"
                      f"ORBextractor.nLevels: {int(s['orb_n_levels'])}\n"
                      f"ORBextractor.iniThFAST: {int(s['orb_ini_th_fast'])}\n"
                      f"ORBextractor.minThFAST: {int(s['orb_min_th_fast'])}\n"
                      f"loopClosing: {1 if s['loop_closure'] else 0}\n")
        # Viewer parameters are parsed even though the driver disables the viewer.
        yaml_text += ("Viewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 0.9\n"
                      "Viewer.PointSize: 2.0\nViewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3.0\n"
                      "Viewer.ViewpointX: 0.0\nViewer.ViewpointY: -0.7\nViewer.ViewpointZ: -1.8\n"
                      "Viewer.ViewpointF: 500.0\n")
        return yaml_text

    def prepare(self, ctx: RunContext) -> None:
        ctx.algo_dir.mkdir(parents=True, exist_ok=True)
        (ctx.algo_dir / "orbslam3_settings.yaml").write_text(self._settings_yaml(ctx))

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
        s = ctx.settings
        cmd = [self.apptainer, "exec", "--cleanenv"]
        for b in self._binds(ctx):
            cmd += ["-B", b]
        cmd += [str(self.container), "/opt/slambench/bin/orbslam3_driver",
                "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt",
                str((ctx.algo_dir / "orbslam3_settings.yaml").resolve()),
                ctx.mode.value, str(ctx.prepared.directory.resolve()), str(ctx.algo_dir.resolve()),
                "--playback-rate", str(float(s["playback_rate"]))]
        if s["clahe"]:
            cmd += ["--clahe", "--clahe-clip", repr(float(s["clahe_clip_limit"])),
                    "--clahe-tile", str(int(s["clahe_tile_grid"]))]
        if s["log_features"]:
            cmd.append("--log-features")
        return cmd

    # ---- outputs -----------------------------------------------------------
    @staticmethod
    def read_frame_log(path: Path):
        """Parse the driver's frame_log.csv. Returns (rows, n_malformed).

        Rows whose index/timestamp/state cannot be parsed (e.g. a line truncated by a crash) are
        counted and skipped. A row keeps ``pose=None`` unless all seven pose fields are finite and
        the quaternion is unit length.
        """
        rows: List[Dict[str, Any]] = []
        malformed = 0
        if not path.exists():
            return rows, malformed
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    row = {"frame_idx": int(r["frame_idx"]), "timestamp_ns": int(r["timestamp_ns"]),
                           "state": int(r["state"]),
                           "coord_session": (int(r["coord_session"]) if r.get("coord_session") not in (None, "")
                                             else None),
                           "map_id": int(r["map_id"]) if r.get("map_id") not in (None, "") else -1,
                           "track_time_s": (float(r["track_time_s"]) if r.get("track_time_s") not in (None, "")
                                            else float("nan"))}
                except (TypeError, ValueError, KeyError):
                    malformed += 1
                    continue
                pose = None
                vals = [r.get(k) for k in POSE_COLUMNS]
                if all(v not in (None, "") for v in vals):
                    try:
                        v = np.array([float(x) for x in vals])
                        if np.all(np.isfinite(v)) and abs(np.linalg.norm(v[3:]) - 1.0) < 1e-3:
                            pose = v
                    except ValueError:
                        pass
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
        stdout = (ctx.run_dir / "stdout.log")
        if stdout.exists():
            text = stdout.read_text(errors="replace")
            m = re.findall(r"There are (\d+) maps in the atlas", text)
            if m:
                extra["n_maps_in_atlas"] = int(m[-1])
            extra["map_kf_counts"] = {int(a): int(b) for a, b in re.findall(r"Map (\d+) has (\d+) KFs", text)}

        # Per-frame diagnostics and online poses from the driver log.
        log_rows, n_malformed = self.read_frame_log(d / "frame_log.csv")
        states = np.array([r["state"] for r in log_rows], dtype=int)
        n_ok = int(np.isin(states, list(OK_STATES)).sum()) if len(states) else 0
        trajectories: List[TrajectoryOutput] = []

        online = [r for r in log_rows if r["state"] in OK_STATES and r["pose"] is not None]
        extra["n_frame_log_rows_malformed"] = n_malformed
        extra["n_online_poses"] = len(online)
        extra["n_ok_frames_without_valid_online_pose"] = n_ok - len(online)
        if online:
            P = np.array([r["pose"] for r in online])
            otraj = Trajectory(np.array([r["timestamp_ns"] for r in online], dtype=np.int64), P[:, :3], P[:, 3:],
                               frame="cam0",
                               meta={"source": str(d / "frame_log.csv"), "estimator": self.estimator_label,
                                     "content": "online pose returned by Track*() immediately after each "
                                                "frame, tracking-OK frames only; camera frame; expressed "
                                                "in the map active at that moment (see map_id)"})
            online_path = ctx.run_dir / "trajectory_online.tum"
            write_tum(otraj, online_path)
            trajectories.append(TrajectoryOutput(
                "online", online_path, "cam0",
                "pose available right after each frame was tracked (may include loop closures that "
                "had already happened, never later ones)", coordinate_sessions="per_frame"))

        # Final trajectory -> normalized TUM, with timestamps snapped to the exact input stamps.
        traj_path, final_ts, final_problem = None, set(), None
        raw = d / "trajectory_euroc.txt"
        if raw.exists() and raw.stat().st_size > 0:
            data = np.loadtxt(raw, ndmin=2)
            t_float_ns = data[:, 0]
            idx = np.clip(np.searchsorted(frames_ts, t_float_ns), 1, len(frames_ts) - 1)
            cand = np.stack([frames_ts[idx - 1], frames_ts[idx]])
            best = cand[np.argmin(np.abs(cand - t_float_ns[None]), axis=0), np.arange(len(idx))]
            err = np.abs(best - t_float_ns)
            extra["timestamp_snap_max_abs_ns"] = float(err.max())
            if err.max() > 1000:  # double-precision seconds round-trip is accurate to ~0.3 us
                final_problem = f"final trajectory timestamps do not match input frames (max {err.max()} ns)"
            else:
                traj = Trajectory(best, data[:, 1:4], data[:, 4:8], frame=self.estimate_frame(ctx.mode),
                                  meta={"source": str(raw), "estimator": self.estimator_label,
                                        "content": "final optimized poses, largest map, non-LOST frames"})
                traj_path = ctx.run_dir / "trajectory.tum"
                write_tum(traj, traj_path)
                final_ts = set(best.tolist())
                trajectories.append(TrajectoryOutput(
                    "final", traj_path, self.estimate_frame(ctx.mode),
                    "System::SaveTrajectoryEuRoC after shutdown: all later optimization applied; "
                    "largest map only", coordinate_sessions="single"))

        diag_path = None
        if log_rows:
            diag_path = ctx.run_dir / "frame_diagnostics.csv"
            with open(diag_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(FRAME_DIAGNOSTICS_COLUMNS)
                for r in log_rows:
                    st = r["state"]
                    tt = "" if np.isnan(r["track_time_s"]) else r["track_time_s"]
                    w.writerow([r["timestamp_ns"], TRACKING_STATES.get(st, str(st)), int(st in OK_STATES),
                                r["map_id"] if r["map_id"] >= 0 else "", tt, int(r["timestamp_ns"] in final_ts),
                                "" if r["coord_session"] is None else r["coord_session"]])
        artifacts = {}
        for key, name in (("tracked_features", "features.bin"), ("sparse_map_points", "map_points.csv"),
                          ("sparse_map_keyframes", "keyframes.csv"),
                          ("place_recognition", "place_recognition.csv")):
            if (d / name).exists() and (d / name).stat().st_size > 0:
                artifacts[key] = d / name
        if "sparse_map_points" in artifacts:
            try:
                from ..artifacts import read_sparse_map
                mp = read_sparse_map(artifacts["sparse_map_points"])
                extra["sparse_map"] = {"n_points": int(len(mp["xyz"])),
                                       "n_points_in_final_frame": int(mp["in_final"].sum()),
                                       "n_maps": int(len(set(mp["map_id"].tolist())))}
            except Exception as exc:  # noqa: BLE001 - optional artifact; never fail collection
                extra["sparse_map"] = {"error": f"{type(exc).__name__}: {exc}"}
        avail = {"tracking_state": bool(log_rows), "map_id": bool(log_rows),
                 "tracked_features": "tracked_features" in artifacts,
                 "sparse_map": "sparse_map_points" in artifacts,
                 "coordinate_session": any(r["coord_session"] is not None for r in log_rows),
                 "per_frame_time": bool(log_rows), "online_pose": bool(online)}
        extra.update({"n_frames_input": int(len(frames_ts)), "n_frames_logged": len(log_rows),
                      "n_frames_tracking_ok": n_ok, "n_final_poses": len(final_ts)})

        common = dict(trajectory=traj_path, estimate_frame=self.estimate_frame(ctx.mode),
                      frame_diagnostics=diag_path, diagnostics_available=avail, extra=extra,
                      trajectories=trajectories, artifacts=artifacts)
        if timed_out:
            return AlgorithmOutput(RunStatus.TIMEOUT, "execution timed out", **common)
        if returncode is None or returncode != 0:
            sig = f" (signal {-returncode})" if returncode is not None and returncode < 0 else ""
            msg = f"driver exited with code {returncode}{sig}"
            if driver_status.get("status") and driver_status["status"] != "completed":
                msg += f"; driver status {driver_status['status']}: {driver_status.get('message', '')}"
            return AlgorithmOutput(RunStatus.CRASH, msg, **common)
        if log_rows and n_ok == 0:
            return AlgorithmOutput(RunStatus.INIT_FAILURE,
                                   f"tracking never reached OK in {len(log_rows)} frames", **common)
        if final_problem:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, final_problem, **common)
        if traj_path is None:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, "trajectory_euroc.txt missing or empty", **common)
        if not log_rows:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, "frame_log.csv missing", **common)
        return AlgorithmOutput(RunStatus.SUCCESS, "", **common)
