"""SP_SLAM3 adapter: ORB-SLAM3 with SuperPoint features (pinned 9439971, Apptainer, CUDA GPU).

SP_SLAM3 (https://github.com/fthbng77/SP_SLAM3) is a fork of pre-v1.0 ORB-SLAM3 whose extractor and
matcher are replaced by SuperPoint (TorchScript, keypoints + 256-d float descriptors), an L2
descriptor matcher (frame-to-frame tracking primarily through LightGlue) and a DBoW3 SuperPoint
vocabulary. The System API, tracking states and trajectory writers are ORB-SLAM3's, so this adapter
reuses the ORB-SLAM3 adapter and its driver (third_party/spslam3/slambench_spslam3_driver.cc, a port of
the ORB-SLAM3 driver) and produces the same outputs (frame_log.csv, trajectory_euroc.txt,
keyframes_euroc.txt, map_points.csv / keyframes.csv, optional features.bin), except
place_recognition.csv, which the SP_SLAM3 build does not log.

Differences from the ORB-SLAM3 adapter:

* stereo only (rectified pinhole input, as produced by the slambench preparation);
* the settings file uses the old (pre-v1.0) ORB-SLAM3 format SP_SLAM3 parses (``Camera.bf``,
  ``ThDepth``); ``ORBextractor.iniThFAST`` / ``minThFAST`` are SuperPoint keypoint-confidence
  thresholds (SuperPoint runs at ``nLevels`` image scales; 1 is the SP_SLAM3 recommendation);
* loop closing cannot be disabled (SP_SLAM3 has no ``loopClosing`` switch), so only
  ``loop_closure: true`` is accepted;
* ``apptainer exec --nv``: SuperPoint and LightGlue run on the GPU through LibTorch (CUDA 12.1);
  without a GPU they silently fall back to the CPU, which driver_status.json records as
  ``cuda_available: false``.

Map reuse (``save_map`` / ``load_map`` / ``reuse_mode``, see orbslam3_map_reuse) works as for
ORB-SLAM3; patch 0006 re-enables the fork's disabled atlas save/load (and fixes its serialization).
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..types import SensorMode
from .base import RunContext
from . import orbslam3_map_reuse as mr
from .orbslam3 import OrbSlam3Adapter, _fmt


def _positional(x) -> str:
    """Shortest round-trip float in positional notation (1e-05 -> 0.00001) with a decimal point."""
    text = format(Decimal(repr(float(x))), "f")
    return text if "." in text else text + ".0"

SPSLAM3_COMMIT = "9439971f73f571a06d55bee0c1f0bd2541a381cb"
DEFAULT_CONTAINER = "~/scratch/slam-bench-work/containers/spslam3_9439971.sif"
VOCABULARY = "/opt/SP_SLAM3/Vocabulary/superpoint_voc.dbow3"
LIGHTGLUE_MODEL = "/opt/SP_SLAM3/lightglue.pt"
DRIVER_SOURCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "spslam3"
DRIVER_SOURCES = (DRIVER_SOURCE_DIR / "slambench_spslam3_driver.cc",
                  DRIVER_SOURCE_DIR.parent / "orbslam3" / "slambench_preprocess.h")
SETTINGS_FILE = "spslam3_settings.yaml"
# Images built from an older driver / patch set that still produce identical results for configs
# without map reuse (the driver and patch 0006 only add behaviour behind the map-reuse settings).
LEGACY_DRIVER_SOURCE_SHA1 = {
    "463cc0adb1f40a28ef3c26bb22d320dea6089a30": "spslam3_9439971.sif: before map reuse (patch 0006)",
}


def driver_source_sha1() -> Optional[str]:
    """sha1 of the driver sources and SP_SLAM3 patches in this repository; must equal the image's
    driver_source_sha1 (spslam3.def concatenates the same files in the same order)."""
    h = hashlib.sha1()
    try:
        files = list(DRIVER_SOURCES)
        files += sorted((DRIVER_SOURCE_DIR / "patches").glob("*.patch"), key=lambda p: p.name)
        for f in files:
            h.update(f.read_bytes())
    except OSError:
        return None
    return h.hexdigest()


DEFAULT_SETTINGS: Dict[str, Any] = {
    "sp_n_features": 1200,         # SuperPoint keypoints per image (same budget as the ORB-SLAM3 configs)
    "sp_scale_factor": 1.2,        # pyramid scale factor (only used with sp_n_levels > 1)
    "sp_n_levels": 1,              # SuperPoint pyramid levels (SP_SLAM3 recommends 1)
    "sp_ini_th": 0.155,            # SuperPoint confidence threshold (ORBextractor.iniThFAST)
    "sp_min_th": 0.055,            # fallback confidence threshold (ORBextractor.minThFAST)
    "sp_fp16": False,              # SuperPoint FP16 inference
    "lightglue": True,             # LightGlue frame-to-frame matching (else SPmatcher L2 projection search)
    "lightglue_fp16": False,
    "stereo_th_depth": 40.0,       # close/far point threshold in baselines (ThDepth)
    "loop_closure": True,          # SP_SLAM3 always runs loop closing; only True is accepted
    "camera_fps": None,            # None -> rounded median frame rate of the prepared input
    "clahe": False,
    "clahe_clip_limit": 3.0,
    "clahe_tile_grid": 8,
    "playback_rate": 1.0,          # 1.0 = real-time throttled input, 0 = as fast as possible
    "log_features": True,          # per-frame keypoints + map-point status (features.bin)
}


class SpSlam3Adapter(OrbSlam3Adapter):
    name = "spslam3"
    estimator_label = "SP_SLAM3"
    supported_modes = frozenset({SensorMode.STEREO})

    @property
    def container(self) -> Path:
        p = self.options.get("container") or os.environ.get("SLAMBENCH_SPSLAM3_SIF") or DEFAULT_CONTAINER
        return Path(p).expanduser()

    def resolve_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(settings) - set(DEFAULT_SETTINGS) - set(mr.MAP_REUSE_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown spslam3 settings: {sorted(unknown)}")
        out = dict(DEFAULT_SETTINGS)
        # Map-reuse keys are only added to the resolved settings when used (keeps old configs identical).
        if set(settings) & set(mr.MAP_REUSE_DEFAULTS):
            out.update(mr.MAP_REUSE_DEFAULTS)
        out.update(settings)
        if set(settings) & set(mr.MAP_REUSE_DEFAULTS):
            mr.resolve_map_reuse_settings(out)
        if out["loop_closure"] is not True:
            raise ValueError("SP_SLAM3 cannot disable loop closing; loop_closure must be true")
        for k in ("sp_n_features", "sp_n_levels", "clahe_tile_grid"):
            if isinstance(out[k], bool) or int(out[k]) != out[k] or int(out[k]) < 1:
                raise ValueError(f"{k} must be an integer >= 1, got {out[k]}")
        if not float(out["sp_scale_factor"]) > 1.0:
            raise ValueError(f"sp_scale_factor must be > 1, got {out['sp_scale_factor']}")
        if not 0.0 < float(out["sp_min_th"]) <= float(out["sp_ini_th"]) < 1.0:
            raise ValueError("SuperPoint thresholds must satisfy 0 < sp_min_th <= sp_ini_th < 1, got "
                             f"{out['sp_min_th']} / {out['sp_ini_th']}")
        if not float(out["stereo_th_depth"]) > 0:
            raise ValueError(f"stereo_th_depth must be > 0, got {out['stereo_th_depth']}")
        if not float(out["clahe_clip_limit"]) > 0:
            raise ValueError(f"clahe_clip_limit must be > 0, got {out['clahe_clip_limit']}")
        if not float(out["playback_rate"]) >= 0:
            raise ValueError(f"playback_rate must be >= 0, got {out['playback_rate']}")
        return out

    def validate(self, mode, dataset, settings) -> List[str]:
        problems = super(OrbSlam3Adapter, self).validate(mode, dataset, settings)
        if problems:
            return problems
        if not self.container.exists():
            problems.append(f"SP_SLAM3 container not found: {self.container} "
                            "(build with third_party/spslam3/build_container.sh)")
        try:
            calib = dataset.stereo_calibration()
        except Exception as exc:  # noqa: BLE001
            return problems + [f"cannot read camera calibration: {exc}"]
        if calib is None or calib.right is None:
            return problems + ["SP_SLAM3 adapter needs a stereo camera calibration"]
        if calib.left.model != "pinhole" or not calib.left.rectified:
            problems.append("SP_SLAM3 adapter supports rectified pinhole stereo only "
                            f"(got {calib.left.model}, rectified={calib.left.rectified})")
        problems += self._map_reuse_validate(mode, settings)
        return problems

    def estimate_frame(self, mode: SensorMode) -> str:
        return "cam0"

    def version_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"expected_spslam3_commit": SPSLAM3_COMMIT,
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
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
            if gpu.returncode == 0:
                info["host_gpus"] = [l.strip() for l in gpu.stdout.splitlines() if l.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
        commit = info.get("container_versions", {}).get("spslam3_commit")
        if commit and commit != SPSLAM3_COMMIT:
            raise RuntimeError(f"container SP_SLAM3 commit {commit} != pinned {SPSLAM3_COMMIT}")
        info["expected_driver_source_sha1"] = expected = driver_source_sha1()
        found = info.get("container_versions", {}).get("driver_source_sha1")
        self._legacy_driver = None
        if commit and expected and found != expected and found in LEGACY_DRIVER_SOURCE_SHA1:
            self._legacy_driver = info["legacy_driver_source"] = LEGACY_DRIVER_SOURCE_SHA1[found]
        elif commit and expected and found != expected:
            raise RuntimeError(f"container driver source {found or '(not recorded)'} differs from this "
                               f"repository's driver {expected}; rebuild the image with "
                               "third_party/spslam3/build_container.sh")
        return info

    def _settings_yaml(self, ctx: RunContext) -> str:
        s, ds = ctx.settings, ctx.dataset
        calib = ds.stereo_calibration()
        L = calib.left
        fps = s["camera_fps"]
        if fps is None:
            dt = ctx.prepared.manifest.get("frame_dt_s", {}).get("median")
            if dt is None:
                ts = np.loadtxt(ctx.prepared.frames_csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
                dt = float(np.median(np.diff(ts))) * 1e-9
            fps = int(round(1.0 / dt))
        # Old (pre-v1.0) ORB-SLAM3 settings format. Values checked with FileNode::isReal() must be
        # written with a decimal point; a missing ORBextractor key leaves the extractor unset.
        lines = ["%YAML:1.0", "# Generated by slam-benching from dataset calibration; do not edit.",
                 "# Rectified stereo: zero distortion, Camera.bf = fx * baseline.",
                 'Camera.type: "PinHole"',
                 f"Camera.fx: {_fmt(L.fx)}", f"Camera.fy: {_fmt(L.fy)}",
                 f"Camera.cx: {_fmt(L.cx)}", f"Camera.cy: {_fmt(L.cy)}",
                 "Camera.k1: 0.0", "Camera.k2: 0.0", "Camera.p1: 0.0", "Camera.p2: 0.0",
                 f"Camera.width: {L.width}", f"Camera.height: {L.height}",
                 f"Camera.fps: {fps}", "Camera.RGB: 0",
                 f"Camera.bf: {_fmt(L.fx * calib.baseline_m)}",
                 f"ThDepth: {_fmt(s['stereo_th_depth'])}",
                 f"ORBextractor.nFeatures: {int(s['sp_n_features'])}",
                 f"ORBextractor.scaleFactor: {_fmt(s['sp_scale_factor'])}",
                 f"ORBextractor.nLevels: {int(s['sp_n_levels'])}",
                 f"ORBextractor.iniThFAST: {_positional(s['sp_ini_th'])}",
                 f"ORBextractor.minThFAST: {_positional(s['sp_min_th'])}",
                 f"SuperPoint.useFP16: {int(bool(s['sp_fp16']))}"]
        if s["lightglue"]:
            lines += [f'LightGlue.model_path: "{LIGHTGLUE_MODEL}"',
                      f"LightGlue.useFP16: {int(bool(s['lightglue_fp16']))}"]
        lines += ["Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9",
                  "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0",
                  "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -1.8",
                  "Viewer.ViewpointF: 500.0"]
        lines += mr.settings_yaml_lines(s, ctx.algo_dir)
        return "\n".join(lines) + "\n"

    def prepare(self, ctx: RunContext) -> None:
        ctx.algo_dir.mkdir(parents=True, exist_ok=True)
        (ctx.algo_dir / SETTINGS_FILE).write_text(self._settings_yaml(ctx))
        mr.prepare(ctx.settings, ctx.algo_dir)

    def command(self, ctx: RunContext) -> List[str]:
        s = ctx.settings
        cmd = [self.apptainer, "exec", "--nv", "--cleanenv"]
        for b in self._binds(ctx):
            cmd += ["-B", b]
        cmd += mr.bind(s)
        cmd += [str(self.container), "/opt/slambench/bin/spslam3_driver", VOCABULARY,
                str((ctx.algo_dir / SETTINGS_FILE).resolve()),
                ctx.mode.value, str(ctx.prepared.directory.resolve()), str(ctx.algo_dir.resolve()),
                "--playback-rate", str(float(s["playback_rate"]))]
        if s["clahe"]:
            cmd += ["--clahe", "--clahe-clip", repr(float(s["clahe_clip_limit"])),
                    "--clahe-tile", str(int(s["clahe_tile_grid"]))]
        if s["log_features"]:
            cmd.append("--log-features")
        return cmd + mr.command_args(s)
