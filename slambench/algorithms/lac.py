"""LAC SLAM adapter: Stanford NAV Lab Lunar Autonomy Challenge stack (pinned 26c3a2b), stereo, GPU.

Dai et al., "Full Stack Navigation, Mapping, and Planning for the Lunar Autonomy Challenge"
(arXiv 2603.17232). The image is built from third_party/lac/lac.def and runs
third_party/lac/slambench_lac_driver.py, which feeds the slambench offline stereo format to the
LAC frontend loop (SuperPoint + LightGlue stereo / frame-to-frame matching, PnP RANSAC visual
odometry) and its GTSAM pose-graph backend (keyframe loop closures + Levenberg-Marquardt).
LAC hard-codes ``.cuda()``, so the command uses ``apptainer exec --nv`` and needs a CUDA GPU.

Deviations from the LAC agent (documented in docs/results/lac_lunapolaris.md):
no semantic segmentation (weights not public; all points labelled ground, no rock mapping), no
IMU (PnP failure -> identity odometry, as LAC does at this commit), time-based keyframes, and the
simulator camera constants replaced by the dataset calibration.

Raw outputs (algorithm/): frame_log.csv (online cam0 pose per frame), trajectory_final.csv
(backend poses after the run), map_points.ply (backend.project_point_map), loop_closures.csv,
driver_status.json (incl. GPU name and peak GPU memory), lac_settings.json.

The online pose is the backend's newest pose right after each frame (includes loop closures found
up to then); the final trajectory is the backend's pose values at the end of the run. LAC only
optimizes the graph when a loop closure is added, so poses after the last loop closure are
identical in both.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..trajectory import Trajectory, write_tum
from ..types import RunStatus, SensorMode
from .base import (FRAME_DIAGNOSTICS_COLUMNS, AlgorithmAdapter, AlgorithmOutput, RunContext,
                   TrajectoryOutput)

LAC_COMMIT = "26c3a2b2a7407e0b2d2e8e28f7cb074961c23f36"
LIGHTGLUE_COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
DEFAULT_CONTAINER = "~/scratch/slam-bench-work/containers/lac_26c3a2b.sif"
DRIVER_SOURCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "lac"
DRIVER = "/opt/slambench/slambench_lac_driver.py"
POSE_COLUMNS = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
MAP_NAME = "map_points.ply"

#: driver frame_log.csv states
STATES = {0: "init", 1: "vo", 2: "vo_failed_identity", 3: "reinitialized"}
OK_STATES = {0, 1}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "playback_rate": 1.0,          # 1.0 = real-time input (frames queue, none dropped), 0 = as fast as possible
    "keyframe_interval_s": 1.0,    # LAC: every 20 simulator steps at 20 Hz = 1 s
    # LAC configs/config.json "loop_closure"
    "loop_closure": True,
    "lc_min_score": 0.5,
    "lc_min_matches": 100,
    "lc_distance_threshold_m": 1.0,
    "lc_angle_threshold_deg": 5.0,
    "max_frames": None,            # debugging: process only the first N frames
    # feed every Nth prepared frame (diagnostic: N-times larger inter-frame motion at the same scene);
    # implemented in the adapter (run-local frames.csv + links to the prepared images), not the driver
    "frame_stride": 1,
    # OMP/MKL threads inside the container; None = library default (one per core, which
    # spins ~7 cores for no latency gain on the frontend, see docs/results/lac_lunapolaris.md)
    "cpu_threads": 1,
    # stereo-match filters (null = LAC behaviour): LAC keeps every LightGlue stereo match with
    # depth <= the caller's max depth (15 m for VO tracking, lac.params.MAX_DEPTH = 56.5 m for
    # backend keyframes; tuned for the simulator's 0.162 m / 914 px rig)
    "max_depth_m": None,
    "min_disparity_px": None,
    "max_row_diff_px": None,
    # "previous" = LAC frame-to-frame PnP; "keyframe" = PnP against a reference frame kept until the
    # median keypoint motion reaches vo_ref_max_flow_px or < vo_ref_min_match_frac of it matches
    "vo_reference": "previous",
    "vo_ref_max_flow_px": 10.0,
    "vo_ref_min_match_frac": 0.5,
}


def driver_source_sha1() -> Optional[str]:
    """sha1 of the driver and import stubs in this repository; must equal the image's
    driver_source_sha1 (lac.def concatenates the same files in the same order)."""
    h = hashlib.sha1()
    try:
        files = [DRIVER_SOURCE_DIR / "slambench_lac_driver.py"]
        files += sorted((DRIVER_SOURCE_DIR / "stubs").glob("*/__init__.py"),
                        key=lambda p: str(p.relative_to(DRIVER_SOURCE_DIR)))
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


def _read_pose(r: Dict[str, str]) -> Optional[np.ndarray]:
    vals = [_float(r.get(k)) for k in POSE_COLUMNS]
    if any(v is None for v in vals):
        return None
    v = np.array(vals)
    return v if abs(np.linalg.norm(v[3:]) - 1.0) < 1e-3 else None


class LacAdapter(AlgorithmAdapter):
    name = "lac"
    supported_modes = frozenset({SensorMode.STEREO})

    # ---- configuration ---------------------------------------------------
    @property
    def container(self) -> Path:
        p = self.options.get("container") or os.environ.get("SLAMBENCH_LAC_SIF") or DEFAULT_CONTAINER
        return Path(p).expanduser()

    @property
    def apptainer(self) -> str:
        return self.options.get("apptainer") or shutil.which("apptainer") or "apptainer"

    def resolve_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(settings) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"unknown lac settings: {sorted(unknown)}")
        out = dict(DEFAULT_SETTINGS)
        out.update(settings)
        if not float(out["playback_rate"]) >= 0:
            raise ValueError(f"playback_rate must be >= 0, got {out['playback_rate']}")
        if out["vo_reference"] not in ("previous", "keyframe"):
            raise ValueError(f"vo_reference must be 'previous' or 'keyframe', got {out['vo_reference']!r}")
        if not float(out["vo_ref_max_flow_px"]) > 0 or not 0 < float(out["vo_ref_min_match_frac"]) <= 1:
            raise ValueError("vo_ref_max_flow_px must be > 0 and vo_ref_min_match_frac in (0, 1]")
        for k in ("max_depth_m", "max_row_diff_px"):
            if out[k] is not None and not float(out[k]) > 0:
                raise ValueError(f"{k} must be > 0 or null, got {out[k]}")
        if out["min_disparity_px"] is not None and not float(out["min_disparity_px"]) >= 0:
            raise ValueError(f"min_disparity_px must be >= 0 or null, got {out['min_disparity_px']}")
        for k in ("keyframe_interval_s", "lc_distance_threshold_m", "lc_angle_threshold_deg"):
            if not float(out[k]) > 0:
                raise ValueError(f"{k} must be > 0, got {out[k]}")
        if not 0 <= float(out["lc_min_score"]) <= 1:
            raise ValueError(f"lc_min_score must be in [0, 1], got {out['lc_min_score']}")
        for k in ("lc_min_matches", "max_frames", "cpu_threads", "frame_stride"):
            v = out[k]
            if v is None and k in ("max_frames", "cpu_threads"):
                continue
            if isinstance(v, bool) or int(v) != v or int(v) < 1:
                raise ValueError(f"{k} must be an integer >= 1, got {v}")
        return out

    def validate(self, mode, dataset, settings) -> List[str]:
        problems = super().validate(mode, dataset, settings)
        if problems:
            return problems
        if not self.container.exists():
            problems.append(f"LAC container not found: {self.container} "
                            "(build with third_party/lac/build_container.sh)")
        try:
            calib = dataset.stereo_calibration()
        except Exception as exc:  # noqa: BLE001
            return problems + [f"cannot read camera calibration: {exc}"]
        if calib is None or calib.right is None:
            return problems + ["LAC needs a stereo camera calibration"]
        L, R = calib.left, calib.right
        if L.model != "pinhole" or not L.rectified or R.model != "pinhole" or not R.rectified:
            problems.append("LAC needs rectified pinhole stereo (it computes depth from horizontal disparity)")
        elif (abs(L.fx - R.fx) > 1e-6 or abs(L.fy - R.fy) > 1e-6 or abs(L.cx - R.cx) > 1e-6
              or abs(L.cy - R.cy) > 1e-6):
            problems.append("LAC needs identical left/right rectified intrinsics")
        if not calib.baseline_m or calib.baseline_m <= 0:
            problems.append("stereo baseline must be > 0")
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
        info: Dict[str, Any] = {"expected_lac_commit": LAC_COMMIT,
                                "expected_lightglue_commit": LIGHTGLUE_COMMIT,
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
        commit = cv.get("lac_commit")
        if commit and commit != LAC_COMMIT:
            raise RuntimeError(f"container LAC commit {commit} != pinned {LAC_COMMIT}")
        if cv.get("lightglue_commit") and cv["lightglue_commit"] != LIGHTGLUE_COMMIT:
            raise RuntimeError(f"container LightGlue commit {cv['lightglue_commit']} != pinned {LIGHTGLUE_COMMIT}")
        info["expected_driver_source_sha1"] = expected = driver_source_sha1()
        found = cv.get("driver_source_sha1")
        if commit and expected and found != expected:
            raise RuntimeError(f"container driver source {found or '(not recorded)'} differs from this "
                               f"repository's driver {expected}; rebuild the image with "
                               "third_party/lac/build_container.sh")
        return info

    # ---- native configuration ---------------------------------------------
    def driver_settings(self, ctx: RunContext) -> Dict[str, Any]:
        s = ctx.settings
        calib = ctx.dataset.stereo_calibration()
        L = calib.left
        return {
            "camera": {"width": int(L.width), "height": int(L.height), "fx": float(L.fx),
                       "fy": float(L.fy), "cx": float(L.cx), "cy": float(L.cy),
                       "baseline_m": float(calib.baseline_m)},
            "playback_rate": float(s["playback_rate"]),
            "keyframe_interval_s": float(s["keyframe_interval_s"]),
            "loop_closure": {"enabled": bool(s["loop_closure"]), "min_score": float(s["lc_min_score"]),
                             "min_matches": int(s["lc_min_matches"]),
                             "distance_threshold_m": float(s["lc_distance_threshold_m"]),
                             "angle_threshold_deg": float(s["lc_angle_threshold_deg"])},
            "max_frames": None if s["max_frames"] is None else int(s["max_frames"]),
            "stereo": {k: None if s[k] is None else float(s[k])
                       for k in ("max_depth_m", "min_disparity_px", "max_row_diff_px")},
            "vo": {"reference": s["vo_reference"], "max_flow_px": float(s["vo_ref_max_flow_px"]),
                   "min_match_frac": float(s["vo_ref_min_match_frac"])},
        }

    def _stride(self, ctx: RunContext) -> int:
        return int(ctx.settings.get("frame_stride") or 1)

    def input_dir(self, ctx: RunContext) -> Path:
        """Directory handed to the driver: the prepared input, or a strided view of it."""
        return ctx.prepared.directory if self._stride(ctx) == 1 else ctx.algo_dir / "input"

    def _write_strided_input(self, ctx: RunContext) -> None:
        n, src, dst = self._stride(ctx), ctx.prepared.directory.resolve(), self.input_dir(ctx)
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_dir() and not (dst / p.name).exists():
                (dst / p.name).symlink_to(p.resolve())
        with open(src / "frames.csv") as f:
            lines = f.read().splitlines()
        (dst / "frames.csv").write_text("\n".join([lines[0]] + lines[1::n]) + "\n")

    def prepare(self, ctx: RunContext) -> None:
        ctx.algo_dir.mkdir(parents=True, exist_ok=True)
        if self._stride(ctx) > 1:
            self._write_strided_input(ctx)
        (ctx.algo_dir / "lac_settings.json").write_text(json.dumps(self.driver_settings(ctx), indent=2))

    def _binds(self, ctx: RunContext) -> List[str]:
        paths = {ctx.prepared.directory.resolve(), ctx.algo_dir.resolve()}
        for d in {ctx.prepared.directory, self.input_dir(ctx)}:
            for p in d.iterdir():
                if p.is_symlink():
                    paths.add(p.resolve())
        out = []
        for p in sorted(paths):
            if not any(str(p).startswith(str(q) + os.sep) for q in paths if q != p):
                out.append(str(p))
        return out

    def command(self, ctx: RunContext) -> List[str]:
        cmd = [self.apptainer, "exec", "--nv", "--cleanenv"]
        n = ctx.settings.get("cpu_threads")
        if n is not None:
            cmd += ["--env", f"OMP_NUM_THREADS={int(n)},MKL_NUM_THREADS={int(n)}"]
        for b in self._binds(ctx):
            cmd += ["-B", b]
        return cmd + [str(self.container), "python", DRIVER,
                      str((ctx.algo_dir / "lac_settings.json").resolve()),
                      str(self.input_dir(ctx).resolve()), str(ctx.algo_dir.resolve())]

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
                           "state": int(r["state"]), "keyframe": int(r["keyframe"]) == 1,
                           "n_tracked": int(r["n_tracked"]), "n_new_points": int(r["n_new_points"]),
                           "n_loop_closures": int(r["n_loop_closures"]),
                           "load_time_s": float(r.get("load_time_s") or "nan"),
                           "track_time_s": float(r["track_time_s"]),
                           "backend_time_s": float(r["backend_time_s"]),
                           "frame_time_s": float(r["frame_time_s"])}
                except (TypeError, ValueError, KeyError):
                    malformed += 1
                    continue
                row["pose"] = _read_pose(r)
                rows.append(row)
        return rows, malformed

    @staticmethod
    def read_final(path: Path):
        ts, poses = [], []
        if not path.exists():
            return np.zeros(0, dtype=np.int64), np.zeros((0, 7))
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                p = _read_pose(r)
                try:
                    t = int(r["timestamp_ns"])
                except (TypeError, ValueError, KeyError):
                    continue
                if p is not None:
                    ts.append(t)
                    poses.append(p)
        return np.array(ts, dtype=np.int64), np.array(poses).reshape(-1, 7)

    def collect(self, ctx: RunContext, returncode: Optional[int], timed_out: bool) -> AlgorithmOutput:
        d = ctx.algo_dir
        frames_ts = np.loadtxt(self.input_dir(ctx) / "frames.csv", delimiter=",", skiprows=1, usecols=0,
                               dtype=np.int64, ndmin=1)
        extra: Dict[str, Any] = {}
        driver_status: Dict[str, Any] = {}
        if (d / "driver_status.json").exists():
            try:
                driver_status = json.loads((d / "driver_status.json").read_text())
            except json.JSONDecodeError as exc:
                extra["driver_status_unreadable"] = str(exc)
        extra["driver_status"] = {k: v for k, v in driver_status.items() if k != "settings"}

        rows, n_malformed = self.read_frame_log(d / "frame_log.csv")
        states: Dict[str, int] = {}
        for r in rows:
            name = STATES.get(r["state"], str(r["state"]))
            states[name] = states.get(name, 0) + 1
        n_ok = sum(1 for r in rows if r["state"] in OK_STATES)

        def stats(key):
            v = np.array([r[key] for r in rows if r["state"] != 0])
            v = v[np.isfinite(v)]
            if not len(v):
                return None
            return {"median": float(np.median(v)), "mean": float(np.mean(v)),
                    "p95": float(np.percentile(v, 95)), "max": float(np.max(v)), "sum": float(np.sum(v))}

        extra.update({"n_frame_log_rows_malformed": n_malformed, "n_frames_input": int(len(frames_ts)),
                      "frame_stride": self._stride(ctx),
                      "n_frames_logged": len(rows), "n_frames_tracking_ok": n_ok, "state_counts": states,
                      "n_keyframes": sum(1 for r in rows if r["keyframe"]),
                      "n_loop_closures": rows[-1]["n_loop_closures"] if rows else 0,
                      "frontend_time_s": stats("track_time_s"), "backend_time_s": stats("backend_time_s"),
                      "frame_time_s": stats("frame_time_s"), "image_load_time_s": stats("load_time_s"),
                      "stereo_filter": driver_status.get("stereo_filter"),
                      "n_vo_reference_switches": driver_status.get("n_vo_reference_switches"),
                      "n_tracked_median": float(np.median([r["n_tracked"] for r in rows])) if rows else None,
                      "gpu": driver_status.get("gpu")})

        trajectories: List[TrajectoryOutput] = []
        online = [r for r in rows if r["pose"] is not None]
        if online:
            P = np.array([r["pose"] for r in online])
            otraj = Trajectory(np.array([r["timestamp_ns"] for r in online], dtype=np.int64), P[:, :3],
                               P[:, 3:], frame="cam0",
                               meta={"source": str(d / "frame_log.csv"), "estimator": "LAC",
                                     "content": "backend's newest pose right after each frame (left camera, "
                                                "OpenCV optical axes); VO-failure frames keep the previous pose"})
            online_path = ctx.run_dir / "trajectory_online.tum"
            write_tum(otraj, online_path)
            trajectories.append(TrajectoryOutput(
                "online", online_path, "cam0",
                "backend pose right after each frame (includes loop closures found up to then)",
                coordinate_sessions="single"))

        traj_path = None
        fts, fposes = self.read_final(d / "trajectory_final.csv")
        final_problem = None
        if len(fts):
            if not np.isin(fts, frames_ts).all():
                final_problem = "final trajectory timestamps do not match input frames"
            else:
                traj = Trajectory(fts, fposes[:, :3], fposes[:, 3:], frame="cam0",
                                  meta={"source": str(d / "trajectory_final.csv"), "estimator": "LAC",
                                        "content": "GTSAM pose values after the run, left camera"})
                traj_path = ctx.run_dir / "trajectory.tum"
                write_tum(traj, traj_path)
                trajectories.append(TrajectoryOutput(
                    "final", traj_path, "cam0",
                    "Backend.get_trajectory() after the run (last loop-closure optimization applied)",
                    coordinate_sessions="single"))
        final_set = set(fts.tolist())

        diag_path = None
        if rows:
            diag_path = ctx.run_dir / "frame_diagnostics.csv"
            with open(diag_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(FRAME_DIAGNOSTICS_COLUMNS)
                for r in rows:
                    w.writerow([r["timestamp_ns"], STATES.get(r["state"], str(r["state"])),
                                int(r["state"] in OK_STATES), 0, r["frame_time_s"],
                                int(r["timestamp_ns"] in final_set), 0])

        artifacts: Dict[str, Path] = {}
        for key, name in (("sparse_map_ply", MAP_NAME), ("loop_closures", "loop_closures.csv"),
                          ("frame_log", "frame_log.csv")):
            if (d / name).exists() and (d / name).stat().st_size > 0:
                artifacts[key] = d / name
        avail = {"tracking_state": bool(rows), "map_id": False, "tracked_features": False,
                 "sparse_map": "sparse_map_ply" in artifacts, "coordinate_session": bool(rows),
                 "per_frame_time": bool(rows), "online_pose": bool(online)}
        common = dict(trajectory=traj_path, estimate_frame="cam0", frame_diagnostics=diag_path,
                      diagnostics_available=avail, extra=extra, trajectories=trajectories,
                      artifacts=artifacts)
        if timed_out:
            return AlgorithmOutput(RunStatus.TIMEOUT, "execution timed out", **common)
        if returncode is None or returncode != 0:
            sig = f" (signal {-returncode})" if returncode is not None and returncode < 0 else ""
            msg = f"LAC driver exited with code {returncode}{sig}"
            if driver_status.get("status") and driver_status["status"] != "completed":
                msg += f"; driver status {driver_status['status']}: {driver_status.get('message', '')}"
            return AlgorithmOutput(RunStatus.CRASH, msg, **common)
        if not rows:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, "frame_log.csv missing", **common)
        if driver_status.get("status") != "completed":
            return AlgorithmOutput(RunStatus.CRASH, f"driver status {driver_status.get('status')}: "
                                   f"{driver_status.get('message', '')}", **common)
        if n_ok <= 1:
            return AlgorithmOutput(RunStatus.INIT_FAILURE,
                                   f"visual odometry never succeeded in {len(rows)} frames ({states})", **common)
        if final_problem:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, final_problem, **common)
        if traj_path is None:
            return AlgorithmOutput(RunStatus.MISSING_OUTPUT, "trajectory_final.csv missing or empty", **common)
        return AlgorithmOutput(RunStatus.SUCCESS, "" if MAP_NAME in {p.name for p in artifacts.values()}
                               else f"{MAP_NAME} missing", **common)
