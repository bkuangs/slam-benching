"""Map reuse (atlas save / load) for the ORB-SLAM3-family adapters (orbslam3, spslam3).

Settings (all optional; with their defaults the generated settings file and command line are
byte-identical to a run without map reuse):

* ``save_map: true``      session A: the atlas is saved at shutdown to ``algorithm/atlas.osa``
* ``load_map: <path>``    session B: load the atlas of a mapping run (its run directory, its
                          ``algorithm`` directory or the ``.osa`` file itself)
* ``reuse_mode``          ``relocalize`` (default): localization only against the loaded map
                          (patched tracking: starts lost, relocalizes whenever lost, never creates a
                          map or keyframe); ``merge``: stock multi-session mapping, a new map is
                          started and ORB-SLAM3's place recognition may merge it into the loaded map

The driver writes ``loaded_map.json`` (loaded map size, load time, ``T_final_map``),
``loaded_map/keyframes.csv`` (the loaded keyframes, for the map-frame check) and three extra
frame_log.csv columns (``in_loaded_map``, ``relocalized``, ``vo_only``).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

MAP_REUSE_DEFAULTS: Dict[str, Any] = {
    "save_map": False,       # save the atlas to algorithm/atlas.osa at shutdown
    "load_map": None,        # mapping run dir / its algorithm dir / .osa file to load
    "reuse_mode": "relocalize",  # relocalize | merge (used only with load_map)
}
REUSE_MODES = ("relocalize", "merge")
ATLAS_BASENAME = "atlas"     # ORB-SLAM3 appends ".osa"
OK_STATES = {2, 5}


def resolve_map_reuse_settings(out: Dict[str, Any]) -> Dict[str, Any]:
    """Validate / normalize the map-reuse settings in a resolved settings dict (in place)."""
    if not isinstance(out["save_map"], bool):
        raise ValueError(f"save_map must be true or false, got {out['save_map']!r}")
    if out["reuse_mode"] not in REUSE_MODES:
        raise ValueError(f"reuse_mode must be one of {REUSE_MODES}, got {out['reuse_mode']!r}")
    if out["load_map"] is not None:
        out["load_map"] = str(_latest_mapping_run(Path(str(out["load_map"])).expanduser().resolve()))
        if out["save_map"] and out["reuse_mode"] == "relocalize":
            raise ValueError("save_map with reuse_mode: relocalize would save the unchanged loaded map; "
                             "use reuse_mode: merge to save the multi-session atlas")
    return out


def _latest_mapping_run(p: Path) -> Path:
    """A config's output directory (runs/<config name>, holding run-id subdirectories) resolves to its
    latest successful run that saved an atlas, so a query config can name the mapping config's output
    directory before that run exists (the resolved run is recorded in config.resolved.yaml)."""
    if p.suffix == ".osa" or not p.is_dir() or (p / "algorithm").is_dir() or (p / f"{ATLAS_BASENAME}.osa").exists():
        return p
    for run in sorted((c for c in p.iterdir() if c.is_dir()), reverse=True):
        try:
            ok = json.loads((run / "status.json").read_text()).get("status") == "success"
        except (OSError, ValueError):
            ok = False
        if ok and (run / "algorithm" / f"{ATLAS_BASENAME}.osa").is_file():
            return run
    return p


def loading(settings: Dict[str, Any]) -> bool:
    return settings.get("load_map") is not None


def uses_map_reuse(settings: Dict[str, Any]) -> bool:
    return bool(settings.get("save_map")) or loading(settings)


def atlas_to_load(settings: Dict[str, Any]) -> Path:
    """The .osa file `load_map` refers to (it may not exist; see map_reuse_problems)."""
    p = Path(settings["load_map"])
    if p.suffix == ".osa":
        return p
    if (p / "algorithm").is_dir():
        p = p / "algorithm"
    return p / f"{ATLAS_BASENAME}.osa"


def mapping_run_dir(atlas: Path) -> Optional[Path]:
    """Run directory of the mapping run that saved `atlas` (None if it is not inside one)."""
    run = atlas.parent.parent
    return run if atlas.parent.name == "algorithm" and (run / "config.resolved.yaml").exists() else None


def settings_yaml_lines(settings: Dict[str, Any], algo_dir: Path) -> List[str]:
    """System.* atlas keys for the settings file; empty without map reuse."""
    lines = []
    if loading(settings):
        lines.append(f'System.LoadAtlasFromFile: "{str(atlas_to_load(settings).resolve())[:-len(".osa")]}"')
    if settings.get("save_map"):
        lines.append(f'System.SaveAtlasToFile: "{(Path(algo_dir).resolve() / ATLAS_BASENAME)}"')
    return lines


def command_args(settings: Dict[str, Any]) -> List[str]:
    return ["--relocalize-only"] if loading(settings) and settings["reuse_mode"] == "relocalize" else []


def bind(settings: Dict[str, Any]) -> List[str]:
    """Read-only bind of the directory holding the atlas to load."""
    if not loading(settings):
        return []
    d = atlas_to_load(settings).resolve().parent
    return ["-B", f"{d}:{d}:ro"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# Settings that may differ between the mapping run and a session reusing its map.
_FREE_SETTINGS = {"save_map", "load_map", "reuse_mode", "playback_rate", "log_features", "clahe",
                  "clahe_clip_limit", "clahe_tile_grid", "camera_fps", "loop_closure"}


def map_reuse_problems(algorithm: str, settings: Dict[str, Any]) -> List[str]:
    if not loading(settings):
        return []
    atlas = atlas_to_load(settings)
    if not atlas.is_file():
        return [f"load_map: atlas {atlas} not found (map with save_map: true first)"]
    run = mapping_run_dir(atlas)
    if run is None:
        return []
    import yaml
    cfg = yaml.safe_load((run / "config.resolved.yaml").read_text())
    if cfg["algorithm"]["name"] != algorithm:
        return [f"load_map: atlas was saved by {cfg['algorithm']['name']}, not {algorithm}"]
    theirs = cfg["algorithm"].get("settings") or {}
    diff = sorted(k for k in set(theirs) | set(settings)
                  if k not in _FREE_SETTINGS and theirs.get(k) != settings.get(k))
    if diff:
        return ["load_map: feature/stereo settings must equal the mapping run's; differing: "
                + ", ".join(f"{k}: map={theirs.get(k)} here={settings.get(k)}" for k in diff)]
    return []


def prepare(settings: Dict[str, Any], algo_dir: Path) -> None:
    if not loading(settings):
        return
    atlas = atlas_to_load(settings)
    run = mapping_run_dir(atlas)
    info = {"load_map": settings["load_map"], "atlas": str(atlas), "map_run": str(run) if run else None,
            "reuse_mode": settings["reuse_mode"], "atlas_sha256_before": file_sha256(atlas),
            "atlas_size_mb": atlas.stat().st_size / 1e6}
    (algo_dir / "map_source.json").write_text(json.dumps(info, indent=2) + "\n")


def _read_keyframes(path: Path) -> Dict[int, np.ndarray]:
    out = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("in_final_frame") == "1":
                out[int(r["keyframe_id"])] = np.array([float(r[k]) for k in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")])
    return out


def map_frame_check(loaded_kfs: Path, mapping_kfs: Path) -> Dict[str, Any]:
    """Loaded keyframes (expressed by the driver in the saved session's final frame) vs the mapping
    run's final keyframes: equal => the mapping run's map->GT transform applies to this run."""
    loaded, mapping = _read_keyframes(loaded_kfs), _read_keyframes(mapping_kfs)
    common = sorted(set(loaded) & set(mapping))
    out: Dict[str, Any] = {"n_loaded_keyframes": len(loaded), "n_mapping_final_keyframes": len(mapping),
                           "n_common": len(common)}
    if common:
        dt = [float(np.linalg.norm(loaded[k][:3] - mapping[k][:3])) for k in common]
        dr = [float(np.degrees((Rotation.from_quat(loaded[k][3:]).inv()
                                * Rotation.from_quat(mapping[k][3:])).magnitude())) for k in common]
        out.update({"max_translation_diff_m": max(dt), "max_rotation_diff_deg": max(dr)})
        out["same_frame"] = bool(len(common) == len(loaded) == len(mapping) and max(dt) < 1e-3 and max(dr) < 0.05)
    else:
        out["same_frame"] = False
    return out


def _T(pose7: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose7[3:]).as_matrix()
    T[:3, 3] = pose7[:3]
    return T


def map_reuse_rows(log_rows: List[Dict[str, Any]], T_final_map: np.ndarray, mode: str) -> List[Dict[str, Any]]:
    """frame_log rows -> rows for evaluation.map_reuse.map_reuse_metrics.

    A frame is localized when it was tracked (state OK) in the loaded map (relocalize mode: always
    the active map; merge mode: only after the new map was merged into it). Its pose is mapped to
    the saved session's final frame with T_final_map. Other frames get no pose (not localized), so
    poses of a separate pre-merge map are never scored.
    """
    out, was_in = [], False
    for r in log_rows:
        in_map = bool(r.get("in_loaded_map"))
        loc = r["state"] in OK_STATES and r["pose"] is not None and in_map
        pose = None
        if loc:
            T = T_final_map @ _T(r["pose"])
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            pose = np.r_[T[:3, 3], q]
        event = bool(r.get("relocalized")) if mode == "relocalize" else (in_map and not was_in)
        was_in = in_map
        out.append({"frame_idx": r["frame_idx"], "timestamp_ns": r["timestamp_ns"], "pose": pose,
                    "localized": loc, "loc_event": bool(event and loc), "loc_correction_m": None,
                    "loc_rejected": 0, "track_time_s": r["track_time_s"], "slam_update": False,
                    "slam_time_s": 0.0, "vo_only": bool(r.get("vo_only")), "in_loaded_map": in_map,
                    "state": r["state"]})
    return out


def read_loaded_map(algo_dir: Path) -> Dict[str, Any]:
    p = Path(algo_dir) / "loaded_map.json"
    return json.loads(p.read_text()) if p.exists() else {}


def collect_map_reuse(settings: Dict[str, Any], algo_dir: Path, log_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """extra['map_reuse'] (loading runs) / extra['saved_map'] (saving runs) entries."""
    d = Path(algo_dir)
    res: Dict[str, Any] = {}
    if settings.get("save_map"):
        atlas = d / f"{ATLAS_BASENAME}.osa"
        kf = _read_keyframes(d / "keyframes.csv")
        n_mp = None
        if (d / "map_points.csv").exists():
            with open(d / "map_points.csv", newline="") as f:
                n_mp = sum(r.get("in_final_frame") == "1" for r in csv.DictReader(f))
        res["saved_map"] = {"atlas": str(atlas), "exists": atlas.is_file(),
                            "atlas_size_mb": atlas.stat().st_size / 1e6 if atlas.is_file() else None,
                            "atlas_sha256": file_sha256(atlas) if atlas.is_file() else None,
                            "n_keyframes_final_map": len(kf), "n_map_points_final_map": n_mp}
    if not loading(settings):
        return res
    src = json.loads((d / "map_source.json").read_text()) if (d / "map_source.json").exists() else {}
    atlas = Path(src.get("atlas") or atlas_to_load(settings))
    src["atlas_sha256_after"] = file_sha256(atlas) if atlas.exists() else None
    src["atlas_unchanged"] = (src["atlas_sha256_after"] is not None
                              and src["atlas_sha256_after"] == src.get("atlas_sha256_before"))
    load = read_loaded_map(d)
    run = Path(src["map_run"]) if src.get("map_run") else None
    check = (map_frame_check(d / "loaded_map" / "keyframes.csv", run / "algorithm" / "keyframes.csv")
             if run else {"same_frame": None, "note": "atlas not inside a run directory; not checked"})
    mode = settings["reuse_mode"]
    rows = map_reuse_rows(log_rows, np.array(load["T_final_map"]) if load.get("T_final_map") else np.eye(4), mode)
    loc = [r for r in rows if r["localized"]]
    first_in = next((r for r in rows if r["in_loaded_map"]), None)
    res["map_reuse"] = {
        **src, "map_load": {**load, "init_time_s": load.get("init_time_s")}, "map_frame_check": check,
        "reuse_mode": mode,
        "n_relocalizations": sum(bool(r.get("relocalized")) for r in log_rows),
        "first_localization_frame_idx": loc[0]["frame_idx"] if loc else None,
        "n_localized_poses": len(loc),
        "n_vo_only_poses": sum(r["vo_only"] for r in loc),
        "merged_into_loaded_map": (first_in is not None) if mode == "merge" else None,
        "merge_frame_idx": first_in["frame_idx"] if mode == "merge" and first_in else None}
    return res
