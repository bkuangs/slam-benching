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

Learned relocalization (patch 0007; all off by default, and with the defaults the settings file is
unchanged):

* ``global_descriptors: true``  compute a MixVPR global descriptor (4096-d, /opt/SP_SLAM3/mixvpr.pt,
  exported by export_mixvpr_slambench.py) for every keyframe and store it in the atlas (session A);
* ``learned_reloc: true``  session B (``load_map``, ``reuse_mode: relocalize``, a map saved with
  ``global_descriptors: true``): relocalization candidates are the ``learned_reloc_topk`` keyframes
  most similar to the frame's MixVPR descriptor, verified by LightGlue matches to their map points
  (``learned_reloc_verifier: lightglue``; ``bow``: SP_SLAM3's SearchByBoW, an ablation) + MLPnP RANSAC
  + pose optimization, accepted with >= ``learned_reloc_min_inliers`` inliers, and only declared once
  ``learned_reloc_consistent_frames`` consecutive frames were accepted at mutually consistent poses.
  Every attempt is logged to ``algorithm/learned_reloc.csv``. ``learned_reloc_recover_descriptors``
  (default true) hands LightGlue unit-norm descriptors: upstream SPDetector::computeDescriptors
  normalizes each descriptor channel over the image's keypoints instead of each descriptor, which
  leaves an image-dependent per-channel scale that LightGlue cannot match across lighting. The
  spslam3_9439971_r4.sif image always behaves as ``false``.

Relocalization monitor (patch 0009; off by default): ``reloc_monitor: true`` (needs ``learned_reloc``)
re-verifies every ``reloc_monitor_period``-th map-tracked frame of session B with the learned
relocalization's verification (candidates: the MixVPR top-K plus the K keyframes nearest to the tracked
pose; no consistency gate). When the verified pose differs from the
tracked one by more than ``reloc_monitor_max_dev_m`` m or ``reloc_monitor_max_dev_deg`` deg, the frame is
reset to the verified pose (counted as a relocalization) or declared lost. ``reloc_monitor_max_vo_s`` /
``reloc_monitor_max_unverified_s`` (0 = off) declare a frame lost after that long in VO-only tracking /
without a verified pose. Every check is logged to ``algorithm/reloc_monitor.csv``.
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
MIXVPR_MODEL = "/opt/SP_SLAM3/mixvpr.pt"
LEARNED_RELOC_LOG = "learned_reloc.csv"
DRIVER_SOURCE_DIR = Path(__file__).resolve().parents[2] / "third_party" / "spslam3"
DRIVER_SOURCES = (DRIVER_SOURCE_DIR / "slambench_spslam3_driver.cc",
                  DRIVER_SOURCE_DIR.parent / "orbslam3" / "slambench_preprocess.h")
SETTINGS_FILE = "spslam3_settings.yaml"
# Images built from an older driver / patch set that still produce identical results for configs
# without map reuse (the driver and patch 0006 only add behaviour behind the map-reuse settings).
LEGACY_DRIVER_SOURCE_SHA1 = {
    "463cc0adb1f40a28ef3c26bb22d320dea6089a30": "spslam3_9439971.sif: before map reuse (patch 0006)",
    "3fc07c4b60ad6e1c5564bfa178fd55fe19e831c6": "spslam3_9439971_r3.sif: before learned relocalization (patch 0007)",
    "af171357f101d83cf7599de2665aa54b6ed77050": "spslam3_9439971_r4.sif: patch 0007 without the unit-norm "
                                                "descriptor recovery of the LightGlue verifier",
    "9be049ce57cce22a2e59d986c38c09fc1f0c4fca": "spslam3_9439971_r5.sif: before the descriptor fix, keyframe "
                                                "density and map anchoring settings (patch 0008)",
    "c9516337cbad1d13b52bb9afbf7e24c6d637be36": "spslam3_9439971_r6.sif: before the relocalization monitor "
                                                "(patch 0009)",
}
# Legacy images that already have map reuse (patch 0006); they only lack the learned-relocalization settings.
MAP_REUSE_LEGACY_DRIVER_SOURCE_SHA1 = {"3fc07c4b60ad6e1c5564bfa178fd55fe19e831c6",
                                       "af171357f101d83cf7599de2665aa54b6ed77050",
                                       "9be049ce57cce22a2e59d986c38c09fc1f0c4fca",
                                       "c9516337cbad1d13b52bb9afbf7e24c6d637be36"}
# Legacy images with the current learned relocalization (patch 0007); they only lack the patch-0008 settings.
LEARNED_RELOC_LEGACY_DRIVER_SOURCE_SHA1 = {"9be049ce57cce22a2e59d986c38c09fc1f0c4fca",
                                           "c9516337cbad1d13b52bb9afbf7e24c6d637be36"}
# Legacy images with patch 0008; they only lack the relocalization monitor (patch 0009).
SP_FIX_LEGACY_DRIVER_SOURCE_SHA1 = {"c9516337cbad1d13b52bb9afbf7e24c6d637be36"}
# Legacy images with learned relocalization that always behave as learned_reloc_recover_descriptors: false.
LEARNED_RELOC_NO_RECOVERY_DRIVER_SOURCE_SHA1 = {"af171357f101d83cf7599de2665aa54b6ed77050"}


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


# Learned relocalization (patch 0007); added to the resolved settings only when used.
LEARNED_RELOC_DEFAULTS: Dict[str, Any] = {
    "global_descriptors": False,          # MixVPR descriptor per keyframe, stored in the atlas
    "learned_reloc": False,               # MixVPR top-K + verification relocalization (session B)
    "learned_reloc_topk": 5,
    "learned_reloc_min_inliers": 50,
    "learned_reloc_consistent_frames": 3,
    "learned_reloc_verifier": "lightglue",  # lightglue | bow (ablation: SP_SLAM3's SearchByBoW)
    # LightGlue sees unit-norm descriptors recovered from SP_SLAM3's per-channel-normalized ones
    # (upstream SPDetector::computeDescriptors normalizes over keypoints, not over channels)
    "learned_reloc_recover_descriptors": True,
}
LEARNED_RELOC_VERIFIERS = ("lightglue", "bow")


# Descriptor fix, keyframe density and map anchoring (patch 0008); added to the resolved settings only
# when used. None = upstream SP_SLAM3 behaviour.
SP_FIX_DEFAULTS: Dict[str, Any] = {
    # Unit-norm SuperPoint descriptors (upstream normalizes every channel over the image's keypoints)
    "sp_unit_descriptors": False,
    # SPmatcher L2 thresholds; None -> upstream 0.25 / 0.55, or with sp_unit_descriptors the
    # equal-recall unit-descriptor values (scripts/sp_descriptor_threshold_calibration.py)
    "sp_th_low": None,
    "sp_th_high": None,
    # Scale of unit descriptors before DBoW3 quantization (vocabulary trained on per-channel-normalized
    # descriptors of norm ~sqrt(256 / keypoints)); None -> 1, or sqrt(256 / sp_n_features) with
    # sp_unit_descriptors
    "sp_bow_descriptor_scale": None,
    # Keyframe insertion (Tracking::NeedNewKeyFrame / CreateNewKeyFrame); None -> upstream 100 / 70 / 0.75 / 100
    "kf_close_tracked": None,
    "kf_close_non_tracked": None,
    "kf_ref_ratio": None,
    "kf_max_points": None,
    # Map anchoring of visual-odometry frames in localization mode (see patch 0008, Tracking.h)
    "map_anchor": False,
    "map_anchor_k": 3,
    "map_anchor_min_inliers": 30,
    "map_anchor_period": 1,
    "map_anchor_max_kf_dist": 5.0,
    "map_anchor_max_kf_angle_deg": 45.0,
    "map_anchor_max_correction": 2.0,
    "map_anchor_max_correction_deg": 15.0,
}
# Unit-norm equivalents of TH_LOW 0.25 / TH_HIGH 0.55 (scripts/sp_descriptor_threshold_calibration.py on
# LunaPolaris seq01 stereo and temporal LightGlue matches): TH_LOW at equal true-match recall (0.516 for
# both pair types); TH_HIGH 0.55 accepts every true match and every nearest impostor, so it is scaled by
# the median per-channel-normalized descriptor norm (0.463).
SP_UNIT_TH_LOW = 0.52
SP_UNIT_TH_HIGH = 1.19
MAP_ANCHOR_LOG = "map_anchor.csv"

# Relocalization monitor (patch 0009); added to the resolved settings only when used.
RELOC_MONITOR_DEFAULTS: Dict[str, Any] = {
    "reloc_monitor": False,
    "reloc_monitor_period": 15,            # re-verify every N-th map-tracked frame
    "reloc_monitor_max_dev_m": 3.0,        # verified vs tracked pose: reset above this deviation
    "reloc_monitor_max_dev_deg": 3.0,
    "reloc_monitor_max_vo_s": 0.0,         # declare lost after this long VO-only (0 = off)
    "reloc_monitor_max_unverified_s": 0.0,  # declare lost after this long without a verified pose (0 = off)
}
RELOC_MONITOR_LOG = "reloc_monitor.csv"


def sp_fix_effective(s: Dict[str, Any]) -> Dict[str, Any]:
    """Thresholds / BoW scale actually used (None = key not written, upstream value)."""
    unit = bool(s.get("sp_unit_descriptors"))
    th_low = s.get("sp_th_low", None)
    th_high = s.get("sp_th_high", None)
    bow = s.get("sp_bow_descriptor_scale", None)
    if unit:
        th_low = SP_UNIT_TH_LOW if th_low is None else th_low
        th_high = SP_UNIT_TH_HIGH if th_high is None else th_high
        bow = (256.0 / float(s["sp_n_features"])) ** 0.5 if bow is None else bow
    return {"th_low": th_low, "th_high": th_high, "bow_scale": bow}


def resolve_sp_fix_settings(out: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the patch-0008 settings of a resolved settings dict (in place)."""
    for k in ("sp_unit_descriptors", "map_anchor"):
        if not isinstance(out[k], bool):
            raise ValueError(f"{k} must be true or false, got {out[k]!r}")
    for k in ("sp_th_low", "sp_th_high", "sp_bow_descriptor_scale", "kf_ref_ratio",
              "map_anchor_max_kf_dist", "map_anchor_max_kf_angle_deg", "map_anchor_max_correction",
              "map_anchor_max_correction_deg"):
        v = out[k]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0):
            raise ValueError(f"{k} must be a number > 0, got {v!r}")
    for k in ("kf_close_tracked", "kf_close_non_tracked", "kf_max_points", "map_anchor_k",
              "map_anchor_min_inliers", "map_anchor_period"):
        v = out[k]
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < (0 if k.startswith("kf_") else 1)):
            raise ValueError(f"{k} must be an integer >= {0 if k.startswith('kf_') else 1}, got {v!r}")
    eff = sp_fix_effective(out)
    if eff["th_low"] is not None and eff["th_high"] is not None and not eff["th_low"] < eff["th_high"]:
        raise ValueError(f"sp_th_low must be < sp_th_high, got {eff['th_low']} / {eff['th_high']}")
    if out["map_anchor"]:
        if out.get("load_map") is None or out.get("reuse_mode") != "relocalize":
            raise ValueError("map_anchor needs load_map with reuse_mode: relocalize")
        if not out["lightglue"]:
            raise ValueError("map_anchor needs lightglue: true")
    if out["map_anchor_min_inliers"] < 15:
        raise ValueError("map_anchor_min_inliers must be >= 15 (TrackLocalMap's own minimum)")
    return out


def sp_fix_yaml_lines(s: Dict[str, Any], algo_dir: Path) -> List[str]:
    """SuperPoint.* / SPmatcher.* / Tracking.kf* / MapAnchor.* keys of the settings file; empty when unused."""
    lines = []
    eff = sp_fix_effective(s)
    if s.get("sp_unit_descriptors"):
        lines.append("SuperPoint.unitDescriptors: 1")
    if eff["bow_scale"] is not None:
        lines.append(f"SuperPoint.bowDescriptorScale: {_positional(eff['bow_scale'])}")
    if eff["th_low"] is not None:
        lines.append(f"SPmatcher.thLow: {_positional(eff['th_low'])}")
    if eff["th_high"] is not None:
        lines.append(f"SPmatcher.thHigh: {_positional(eff['th_high'])}")
    for k, key in (("kf_close_tracked", "kfCloseTracked"), ("kf_close_non_tracked", "kfCloseNonTracked"),
                   ("kf_max_points", "kfMaxPoints")):
        if s.get(k) is not None:
            lines.append(f"Tracking.{key}: {int(s[k])}")
    if s.get("kf_ref_ratio") is not None:
        lines.append(f"Tracking.kfRefRatio: {_positional(s['kf_ref_ratio'])}")
    if s.get("map_anchor"):
        lines += ["MapAnchor.enable: 1",
                  f"MapAnchor.k: {int(s['map_anchor_k'])}",
                  f"MapAnchor.minInliers: {int(s['map_anchor_min_inliers'])}",
                  f"MapAnchor.period: {int(s['map_anchor_period'])}",
                  f"MapAnchor.maxKFDist: {_positional(s['map_anchor_max_kf_dist'])}",
                  f"MapAnchor.maxKFAngleDeg: {_positional(s['map_anchor_max_kf_angle_deg'])}",
                  f"MapAnchor.maxCorrection: {_positional(s['map_anchor_max_correction'])}",
                  f"MapAnchor.maxCorrectionDeg: {_positional(s['map_anchor_max_correction_deg'])}",
                  f'MapAnchor.logFile: "{(Path(algo_dir).resolve() / MAP_ANCHOR_LOG)}"']
    return lines


def summarize_map_anchor(path: Path) -> Dict[str, Any]:
    """Summary of the driver's per-attempt map-anchoring log (map_anchor.csv)."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Any] = {"n_attempts": len(rows), "n_accepted": sum(r["accepted"] == "1" for r in rows)}
    if rows:
        msv = np.array([float(r["ms"]) for r in rows])
        out["ms"] = {"median": float(np.median(msv)), "p95": float(np.percentile(msv, 95))}
        acc = [r for r in rows if r["accepted"] == "1"]
        if acc:
            c = np.array([float(r["correction_m"]) for r in acc])
            li = np.array([float(r["local_inliers"]) for r in acc])
            out["accepted_correction_m"] = {"median": float(np.median(c)), "p95": float(np.percentile(c, 95))}
            out["accepted_local_inliers"] = {"median": float(np.median(li))}
    return out


def resolve_reloc_monitor_settings(out: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the relocalization-monitor settings of a resolved settings dict (in place)."""
    if not isinstance(out["reloc_monitor"], bool):
        raise ValueError(f"reloc_monitor must be true or false, got {out['reloc_monitor']!r}")
    v = out["reloc_monitor_period"]
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"reloc_monitor_period must be an integer >= 1, got {v!r}")
    for k in ("reloc_monitor_max_dev_m", "reloc_monitor_max_dev_deg", "reloc_monitor_max_vo_s",
              "reloc_monitor_max_unverified_s"):
        v, positive = out[k], "_dev_" in k
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (v > 0 if positive else v >= 0):
            raise ValueError(f"{k} must be a number {'> 0' if positive else '>= 0'}, got {v!r}")
    if out["reloc_monitor"] and not out.get("learned_reloc"):
        raise ValueError("reloc_monitor needs learned_reloc: true")
    return out


def reloc_monitor_yaml_lines(s: Dict[str, Any], algo_dir: Path) -> List[str]:
    """RelocMonitor.* keys of the settings file; empty when unused."""
    if not s.get("reloc_monitor"):
        return []
    return ["RelocMonitor.enable: 1",
            f"RelocMonitor.period: {int(s['reloc_monitor_period'])}",
            f"RelocMonitor.maxDevM: {_positional(s['reloc_monitor_max_dev_m'])}",
            f"RelocMonitor.maxDevDeg: {_positional(s['reloc_monitor_max_dev_deg'])}",
            f"RelocMonitor.maxVOs: {_positional(s['reloc_monitor_max_vo_s'])}",
            f"RelocMonitor.maxUnverifiedS: {_positional(s['reloc_monitor_max_unverified_s'])}",
            f'RelocMonitor.logFile: "{(Path(algo_dir).resolve() / RELOC_MONITOR_LOG)}"']


def summarize_reloc_monitor(path: Path) -> Dict[str, Any]:
    """Summary of the driver's relocalization-monitor log (reloc_monitor.csv)."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    checks = [r for r in rows if r["trigger"].startswith("period")]
    # verified: 0, or the candidate source of the verified pose (vpr = MixVPR retrieval, prior = nearest
    # keyframes to the tracked pose)
    verified = [r for r in checks if r["verified"] != "0"]
    out: Dict[str, Any] = {"n_rows": len(rows), "n_checks": len(checks), "n_verified": len(verified),
                           "n_verified_by_prior": sum(r["verified"] == "prior" for r in verified)}
    for d in ("agree", "unverified", "reset", "reset_lost", "lost"):
        out[f"n_{d}"] = sum(r["decision"] == d for r in rows)
    dev = np.array([float(r["dev_m"]) for r in verified])
    if dev.size:
        out["verified_dev_m"] = {"median": float(np.median(dev)), "p90": float(np.percentile(dev, 90)),
                                 "max": float(dev.max())}
    if checks:
        msv = np.array([float(r["ms"]) for r in checks])
        out["check_ms"] = {"median": float(np.median(msv)), "p95": float(np.percentile(msv, 95))}
    return out


def resolve_learned_reloc_settings(out: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the learned-relocalization settings of a resolved settings dict (in place)."""
    for k in ("global_descriptors", "learned_reloc", "learned_reloc_recover_descriptors"):
        if not isinstance(out[k], bool):
            raise ValueError(f"{k} must be true or false, got {out[k]!r}")
    for k in ("learned_reloc_topk", "learned_reloc_min_inliers", "learned_reloc_consistent_frames"):
        if isinstance(out[k], bool) or not isinstance(out[k], int) or out[k] < 1:
            raise ValueError(f"{k} must be an integer >= 1, got {out[k]!r}")
    if out["learned_reloc_min_inliers"] < 10:
        raise ValueError("learned_reloc_min_inliers must be >= 10 (pose optimization needs 10 inliers)")
    if out["learned_reloc_verifier"] not in LEARNED_RELOC_VERIFIERS:
        raise ValueError(f"learned_reloc_verifier must be one of {LEARNED_RELOC_VERIFIERS}, "
                         f"got {out['learned_reloc_verifier']!r}")
    if out["learned_reloc"]:
        if out.get("load_map") is None or out.get("reuse_mode") != "relocalize":
            raise ValueError("learned_reloc needs load_map with reuse_mode: relocalize")
        if out["learned_reloc_verifier"] == "lightglue" and not out["lightglue"]:
            raise ValueError("learned_reloc_verifier: lightglue needs lightglue: true")
    return out


def learned_reloc_yaml_lines(s: Dict[str, Any], algo_dir: Path) -> List[str]:
    """GlobalDescriptor.* / LearnedReloc.* keys of the settings file; empty when unused."""
    lines = []
    if s.get("global_descriptors") or s.get("learned_reloc"):
        lines.append(f'GlobalDescriptor.model_path: "{MIXVPR_MODEL}"')
    if s.get("learned_reloc"):
        lines += ["LearnedReloc.enable: 1",
                  f"LearnedReloc.topK: {int(s['learned_reloc_topk'])}",
                  f"LearnedReloc.minInliers: {int(s['learned_reloc_min_inliers'])}",
                  f"LearnedReloc.consistentFrames: {int(s['learned_reloc_consistent_frames'])}",
                  f'LearnedReloc.verifier: "{s["learned_reloc_verifier"]}"',
                  f"LearnedReloc.recoverDescriptors: {int(bool(s['learned_reloc_recover_descriptors']))}",
                  f'LearnedReloc.logFile: "{(Path(algo_dir).resolve() / LEARNED_RELOC_LOG)}"']
    return lines


def summarize_learned_reloc(path: Path) -> Dict[str, Any]:
    """Summary of the driver's per-attempt learned-relocalization log (learned_reloc.csv)."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Any] = {"n_attempts": len(rows)}
    if not rows:
        return out

    def col(k):
        return np.array([float(r[k]) for r in rows if r.get(k) not in (None, "")])

    acc = [r for r in rows if r["accepted"] == "1"]
    conf = [r for r in rows if r["confirmed"] == "1"]
    out.update({
        "n_attempts_lost": sum(r["mode"] == "lost" for r in rows),
        "n_accepted_single_frame": len(acc), "n_confirmed": len(conf),
        "first_confirmed_timestamp_s": float(conf[0]["timestamp"]) if conf else None,
        "n_database_keyframes": int(rows[0]["n_db"]),
        "n_with_verified_candidate": sum(int(r["best_rank"]) >= 0 for r in rows),
    })
    for k in ("desc_ms", "retrieval_ms", "verify_ms", "total_ms"):
        v = col(k)
        out[k] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
    best = [int(r["best_inliers"]) for r in rows]
    out["best_inliers"] = {"median": float(np.median(best)), "max": int(max(best))}
    sims = [float(r["cand_sims"].split(";")[0]) for r in rows if r["cand_sims"]]
    if sims:
        out["top1_similarity"] = {"median": float(np.median(sims)), "max": float(max(sims))}
    return out


class SpSlam3Adapter(OrbSlam3Adapter):
    name = "spslam3"
    estimator_label = "SP_SLAM3"
    supported_modes = frozenset({SensorMode.STEREO})

    @property
    def container(self) -> Path:
        p = self.options.get("container") or os.environ.get("SLAMBENCH_SPSLAM3_SIF") or DEFAULT_CONTAINER
        return Path(p).expanduser()

    def resolve_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        unknown = (set(settings) - set(DEFAULT_SETTINGS) - set(mr.MAP_REUSE_DEFAULTS)
                   - set(LEARNED_RELOC_DEFAULTS) - set(SP_FIX_DEFAULTS) - set(RELOC_MONITOR_DEFAULTS))
        if unknown:
            raise ValueError(f"unknown spslam3 settings: {sorted(unknown)}")
        out = dict(DEFAULT_SETTINGS)
        # Map-reuse / learned-relocalization keys are only added to the resolved settings when used
        # (keeps old configs identical).
        if set(settings) & set(mr.MAP_REUSE_DEFAULTS):
            out.update(mr.MAP_REUSE_DEFAULTS)
        if set(settings) & set(LEARNED_RELOC_DEFAULTS):
            out.update(LEARNED_RELOC_DEFAULTS)
        if set(settings) & set(SP_FIX_DEFAULTS):
            out.update(SP_FIX_DEFAULTS)
        if set(settings) & set(RELOC_MONITOR_DEFAULTS):
            out.update(RELOC_MONITOR_DEFAULTS)
        out.update(settings)
        if set(settings) & set(mr.MAP_REUSE_DEFAULTS):
            mr.resolve_map_reuse_settings(out)
        if set(settings) & set(LEARNED_RELOC_DEFAULTS):
            resolve_learned_reloc_settings(out)
        if set(settings) & set(SP_FIX_DEFAULTS):
            resolve_sp_fix_settings(out)
        if set(settings) & set(RELOC_MONITOR_DEFAULTS):
            resolve_reloc_monitor_settings(out)
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
        problems += self._learned_reloc_validate(settings)
        problems += self._sp_fix_validate(settings)
        return problems

    def _sp_fix_validate(self, settings) -> List[str]:
        problems = []
        legacy = getattr(self, "_legacy_driver", None)
        used = [k for k in SP_FIX_DEFAULTS if k in settings and settings[k] != SP_FIX_DEFAULTS[k]]
        if used and legacy and getattr(self, "_legacy_driver_sha1", None) not in SP_FIX_LEGACY_DRIVER_SOURCE_SHA1:
            problems.append(f"{', '.join(used)} need an image built with patch 0008; {self.container} is "
                            f"{legacy}")
        used = [k for k in RELOC_MONITOR_DEFAULTS if k in settings and settings[k] != RELOC_MONITOR_DEFAULTS[k]]
        if used and legacy:
            problems.append(f"{', '.join(used)} need an image built with patch 0009; {self.container} is "
                            f"{legacy}")
        return problems

    def _map_reuse_validate(self, mode, settings) -> List[str]:
        legacy = getattr(self, "_legacy_driver", None)
        if legacy and getattr(self, "_legacy_driver_sha1", None) in MAP_REUSE_LEGACY_DRIVER_SOURCE_SHA1:
            self._legacy_driver = None  # this legacy image has map reuse
            try:
                return super()._map_reuse_validate(mode, settings)
            finally:
                self._legacy_driver = legacy
        return super()._map_reuse_validate(mode, settings)

    def _learned_reloc_validate(self, settings) -> List[str]:
        if not (settings.get("global_descriptors") or settings.get("learned_reloc")):
            return []
        problems = []
        legacy_sha1 = getattr(self, "_legacy_driver_sha1", None)
        if legacy_sha1 in LEARNED_RELOC_NO_RECOVERY_DRIVER_SOURCE_SHA1:
            if (settings.get("learned_reloc") and settings.get("learned_reloc_verifier") == "lightglue"
                    and settings.get("learned_reloc_recover_descriptors")):
                problems.append("learned_reloc_recover_descriptors: true needs an image with the current "
                                f"patch 0007; {self.container} is {self._legacy_driver} "
                                "(set it to false to reproduce that image's runs)")
        elif getattr(self, "_legacy_driver", None) and legacy_sha1 not in LEARNED_RELOC_LEGACY_DRIVER_SOURCE_SHA1:
            problems.append("global_descriptors / learned_reloc need an image built with patch 0007; "
                            f"{self.container} is {self._legacy_driver}")
        if settings.get("learned_reloc") and mr.loading(settings):
            run = mr.mapping_run_dir(mr.atlas_to_load(settings))
            if run is not None:
                import yaml
                cfg = yaml.safe_load((run / "config.resolved.yaml").read_text())
                if not (cfg["algorithm"].get("settings") or {}).get("global_descriptors"):
                    problems.append(f"learned_reloc: the map run {run} was not made with "
                                    "global_descriptors: true (its keyframes have no MixVPR descriptor)")
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
        self._legacy_driver = self._legacy_driver_sha1 = None
        if commit and expected and found != expected and found in LEGACY_DRIVER_SOURCE_SHA1:
            self._legacy_driver = info["legacy_driver_source"] = LEGACY_DRIVER_SOURCE_SHA1[found]
            self._legacy_driver_sha1 = found
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
        lines += learned_reloc_yaml_lines(s, ctx.algo_dir)
        lines += sp_fix_yaml_lines(s, ctx.algo_dir)
        lines += reloc_monitor_yaml_lines(s, ctx.algo_dir)
        return "\n".join(lines) + "\n"

    def prepare(self, ctx: RunContext) -> None:
        ctx.algo_dir.mkdir(parents=True, exist_ok=True)
        (ctx.algo_dir / SETTINGS_FILE).write_text(self._settings_yaml(ctx))
        mr.prepare(ctx.settings, ctx.algo_dir)

    def collect(self, ctx: RunContext, returncode, timed_out):
        out = super().collect(ctx, returncode, timed_out)
        log = ctx.algo_dir / LEARNED_RELOC_LOG
        if ctx.settings.get("learned_reloc") and log.exists():
            out.artifacts["learned_reloc"] = log
            try:
                out.extra["learned_reloc"] = summarize_learned_reloc(log)
            except Exception as exc:  # noqa: BLE001 - diagnostics only; never fail collection
                out.extra["learned_reloc"] = {"error": f"{type(exc).__name__}: {exc}"}
        log = ctx.algo_dir / MAP_ANCHOR_LOG
        if ctx.settings.get("map_anchor") and log.exists():
            out.artifacts["map_anchor"] = log
            try:
                out.extra["map_anchor"] = summarize_map_anchor(log)
            except Exception as exc:  # noqa: BLE001 - diagnostics only; never fail collection
                out.extra["map_anchor"] = {"error": f"{type(exc).__name__}: {exc}"}
        log = ctx.algo_dir / RELOC_MONITOR_LOG
        if ctx.settings.get("reloc_monitor") and log.exists():
            out.artifacts["reloc_monitor"] = log
            try:
                out.extra["reloc_monitor"] = summarize_reloc_monitor(log)
            except Exception as exc:  # noqa: BLE001 - diagnostics only; never fail collection
                out.extra["reloc_monitor"] = {"error": f"{type(exc).__name__}: {exc}"}
        return out

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
