"""Rerun replay of a map-reuse (localization) run, drawn in the saved map's fixed frame.

Everything is placed in the ground-truth world with the *mapping run's* single map->GT transform
(``fixed_alignment_from``), exactly as ``map_reuse_metrics`` scores it; nothing is fitted on the
query run, so a wrong localization stays visibly wrong.

Supported: RTAB-Map localization runs, and ORB-SLAM3 / SP_SLAM3 runs that loaded an atlas
(``load_map``; relocalize or merge mode).

3D: the saved map (RTAB-Map: dense cloud and node path; ORB-SLAM3 family: sparse map points and
keyframe path, both exported by the mapping run), the query's ground truth, the query's localized
poses coloured by fixed-frame error, accepted localizations (with a line to the matched map node /
keyframe when it is logged), rejected hypotheses and (SP_SLAM3) accepted map anchors. Images: the
query camera and the map image the vehicle should be recognizing (the map node / keyframe it matched
at a localization or anchor, otherwise the one closest to its true pose). Plots: fixed-frame error,
localized / odometry state, and per-frame counts (RTAB-Map: features; SP_SLAM3 learned
relocalization: verification inliers per attempt and anchor inliers).
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .rerun_export import FEATURE_COLORS, GT_COLOR, PREFETCH, _encode, _load_features, _load_run

MAP_CLOUD_MAX_POINTS = 400_000
MAP_NODE_COLOR = [255, 170, 60]
PRIOR_COLOR = [120, 120, 200]
EVENT_COLOR = [0, 200, 255]
ANCHOR_COLOR = [230, 60, 230]
REJECT_COLOR = [230, 40, 40]
ERR_BANDS = [(0.1, [40, 190, 60]), (0.25, [200, 200, 0]), (0.5, [255, 140, 0]), (np.inf, [230, 40, 40])]


def _err_color(e: float) -> List[int]:
    if not np.isfinite(e):
        return [160, 160, 160]
    return next(c for thr, c in ERR_BANDS if e < thr)


def read_ply_vertices(path: Path, max_points: int = 0, seed: int = 0):
    """(xyz float64 Nx3, rgb uint8 Nx3 or None) from a binary little-endian or ASCII PLY."""
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1",
             "uint8": "u1", "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2",
             "int16": "<i2", "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, n, props, elem = None, 0, [], None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                elem = tok[1]
                if elem == "vertex":
                    n = int(tok[2])
            elif tok[0] == "property" and elem == "vertex":
                if tok[1] == "list":
                    raise ValueError(f"{path}: list properties on vertices are not supported")
                props.append((tok[2], types[tok[1]]))
            elif tok[0] == "end_header":
                break
        if fmt == "binary_little_endian":
            data = np.fromfile(f, dtype=np.dtype(props), count=n)  # vertex block comes first
        elif fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, ndmin=2)
            data = {name: data[:, k] for k, (name, _) in enumerate(props)}
        else:
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
    xyz = np.stack([np.asarray(data[k], dtype=float) for k in ("x", "y", "z")], axis=1)
    names = [p[0] for p in props]
    rgb = (np.stack([np.asarray(data[k]) for k in ("red", "green", "blue")], axis=1).astype(np.uint8)
           if all(k in names for k in ("red", "green", "blue")) else None)
    ok = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[ok], (rgb[ok] if rgb is not None else None)
    if max_points and len(xyz) > max_points:
        sel = np.sort(np.random.default_rng(seed).choice(len(xyz), max_points, replace=False))
        xyz, rgb = xyz[sel], (rgb[sel] if rgb is not None else None)
    return xyz, rgb


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """xyzw quaternions (N,4) -> rotation matrices (N,3,3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def _map_nodes(map_run: Path) -> Dict[str, Any]:
    """Mapping run's final map nodes: ids, timestamps, cam0 positions and optical axes (map frame)."""
    ids, ts, P, Q = [], [], [], []
    with open(map_run / "algorithm" / "nodes.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["in_final"]) != 1:
                continue
            ids.append(int(r["node_id"]))
            ts.append(int(r["timestamp_ns"]))
            P.append([float(r[k]) for k in ("tx", "ty", "tz")])
            Q.append([float(r[k]) for k in ("qx", "qy", "qz", "qw")])
    P, Q = np.array(P, dtype=float).reshape(-1, 3), np.array(Q, dtype=float).reshape(-1, 4)
    return {"id": np.array(ids, dtype=np.int64), "t_ns": np.array(ts, dtype=np.int64), "pos": P,
            "z_axis": _quat_to_R(Q)[:, :, 2] if len(Q) else np.zeros((0, 3))}


def _map_keyframes(map_run: Path) -> Dict[str, Any]:
    """ORB-SLAM3-family mapping run's final keyframes (``keyframes.csv``, cam0 poses in the final frame)."""
    ids, ts, P, Q = [], [], [], []
    with open(map_run / "algorithm" / "keyframes.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["in_final_frame"]) != 1:
                continue
            ids.append(int(r["keyframe_id"]))
            ts.append(int(r["timestamp_ns"]))
            P.append([float(r[k]) for k in ("tx", "ty", "tz")])
            Q.append([float(r[k]) for k in ("qx", "qy", "qz", "qw")])
    P, Q = np.array(P, dtype=float).reshape(-1, 3), np.array(Q, dtype=float).reshape(-1, 4)
    return {"id": np.array(ids, dtype=np.int64), "t_ns": np.array(ts, dtype=np.int64), "pos": P,
            "z_axis": _quat_to_R(Q)[:, :, 2] if len(Q) else np.zeros((0, 3))}


def _map_points(path: Path) -> np.ndarray:
    """Sparse map points of the final frame (``map_points.csv``), Nx3."""
    xyz = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["in_final_frame"]) == 1:
                xyz.append([float(r["x"]), float(r["y"]), float(r["z"])])
    return np.array(xyz, dtype=float).reshape(-1, 3)


def _nearest(sorted_ts: np.ndarray, t_ns: np.ndarray, tol_ns: int) -> np.ndarray:
    """Index of the nearest entry of ``sorted_ts`` for each of ``t_ns`` (-1 if farther than ``tol_ns``)."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    if len(sorted_ts) == 0:
        return np.full(len(t_ns), -1, dtype=np.int64)
    j = np.searchsorted(sorted_ts, t_ns)
    lo, hi = np.clip(j - 1, 0, len(sorted_ts) - 1), np.clip(j, 0, len(sorted_ts) - 1)
    j = np.where(np.abs(sorted_ts[lo] - t_ns) <= np.abs(sorted_ts[hi] - t_ns), lo, hi)
    return np.where(np.abs(sorted_ts[j] - t_ns) <= tol_ns, j, -1)


def _map_images(map_run: Path, node_ts: np.ndarray) -> List[Optional[Path]]:
    """Left image of each map node in the mapping run's prepared input (None if unavailable).

    Matched by nearest timestamp within 1 ms (ORB-SLAM3 keyframe stamps pass through a double)."""
    pj = map_run / "prepared_input.json"
    if not pj.exists():
        return [None] * len(node_ts)
    prep = Path(json.loads(pj.read_text())["directory"])
    ts, files = [], []
    try:
        with open(prep / "frames.csv") as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split(",")
                ts.append(int(c[0]))
                files.append(prep / c[1])
    except OSError:
        return [None] * len(node_ts)
    order = np.argsort(np.array(ts, dtype=np.int64))
    sts = np.array(ts, dtype=np.int64)[order]
    return [files[order[j]] if j >= 0 else None for j in _nearest(sts, node_ts, 1_000_000).tolist()]


def _rtabmap_source(run_dir: Path, map_run: Path) -> Dict[str, Any]:
    """Map and normalized per-frame rows of an RTAB-Map localization run."""
    from ..algorithms.rtabmap import RtabmapAdapter
    rows, _ = RtabmapAdapter.read_frame_log(run_dir / "algorithm" / "frame_log.csv")
    for r in rows:
        r["match_id"] = r["loop_closure_id"] or r["proximity_id"]
        kind = "loop closure" if r["loop_closure_id"] else "proximity"
        r["event_text"] = (f"({kind}) against map node {r['match_id']}: correction "
                           f"{r['loc_correction_m']:.2f} m") if r["loc_event"] else ""
    cloud, rgb = None, None
    cloud_path = map_run / "algorithm" / "rtabmap_cloud.ply"
    if cloud_path.exists():
        cloud, rgb = read_ply_vertices(cloud_path, MAP_CLOUD_MAX_POINTS)
    lt = np.array([r["timestamp_ns"] for r in rows], dtype=np.int64)
    return {"nodes": _map_nodes(map_run), "node_name": "map node", "cloud": cloud, "cloud_rgb": rgb,
            "cloud_desc": "dense cloud", "rows": rows,
            "state": [("odometry_ok", "odometry OK (x0.9)", lt, [0.9 * r["odom_ok"] for r in rows], [42, 161, 152])],
            "counts_name": "odometry features",
            "counts": [("features", "keypoints", lt, [r["features"] for r in rows], FEATURE_COLORS[0], "line"),
                       ("inliers", "odometry inliers", lt, [r["inliers"] for r in rows], FEATURE_COLORS[1], "line")],
            "anchors": [], "rejected_text": "localization hypothesis rejected (graph-consistency check)",
            "notes": []}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _sec_to_ns(x: str) -> int:
    """'1750084256.971692562' -> ns (the decimal string, not a double: keeps sub-µs digits)."""
    a, _, b = x.strip().partition(".")
    return int(a) * 1_000_000_000 + int((b + "000000000")[:9])


def _orbslam3_source(run_dir: Path, map_run: Path, mr_info: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map and normalized per-frame rows of an ORB-SLAM3 / SP_SLAM3 run that loaded an atlas.

    Rows and poses come from ``orbslam3_map_reuse.map_reuse_rows`` (with ``T_final_map``), exactly as
    the evaluator builds them. SP_SLAM3 learned relocalization logs the verified map keyframe of each
    accepted relocalization (``learned_reloc.csv``) and accepted map anchors (``map_anchor.csv``);
    stock relocalization does not log the matched keyframe."""
    from ..algorithms import orbslam3_map_reuse as omr
    from ..algorithms.orbslam3 import OrbSlam3Adapter
    algo = run_dir / "algorithm"
    log_rows, _ = OrbSlam3Adapter.read_frame_log(algo / "frame_log.csv")
    load = omr.read_loaded_map(algo)
    T = np.array(load["T_final_map"]) if load.get("T_final_map") else np.eye(4)
    mode = mr_info.get("reuse_mode", "relocalize")
    rows = omr.map_reuse_rows(log_rows, T, mode)
    ts = np.array([r["timestamp_ns"] for r in rows], dtype=np.int64)
    settings = (cfg.get("algorithm") or {}).get("settings") or {}
    gate = int(settings.get("learned_reloc_min_inliers", 50))

    # learned relocalization: verified keyframe of each confirmed relocalization, inliers per attempt
    lr = _read_csv(algo / "learned_reloc.csv")
    lr_t = np.array([_sec_to_ns(r["timestamp"]) for r in lr], dtype=np.int64)
    lr_fi = _nearest(ts, lr_t, 1_000_000)
    match = {}
    for r, k in zip(lr, lr_fi.tolist()):
        if k >= 0 and r["confirmed"] == "1" and int(r["best_kf_id"]) >= 0:
            match[k] = (int(r["best_kf_id"]),
                        f"(learned relocalization: MixVPR rank {int(r['best_rank']) + 1} of "
                        f"{len(r['cand_kf_ids'].split(';'))}, {r['best_inliers']} inliers) against map "
                        f"keyframe {r['best_kf_id']}")
    for k, r in enumerate(rows):
        m = match.get(k)
        r["match_id"] = m[0] if m else None
        r["event_text"] = m[1] if m else ("(relocalization; matched keyframe not logged)" if mode == "relocalize"
                                          else "(map merge into the loaded map)")
    an = _read_csv(algo / "map_anchor.csv")
    an_t = np.array([_sec_to_ns(r["timestamp"]) for r in an], dtype=np.int64)
    anchors = []
    for r, k in zip(an, _nearest(ts, an_t, 1_000_000).tolist()):
        if r["accepted"] == "1" and k >= 0:
            kf = [int(x) for x in r["kf_ids"].split(";") if x]
            anchors.append((int(ts[k]), kf[0] if kf else None,
                            f"map anchor to keyframes {r['kf_ids'].replace(';', ', ')}: {r['pose_inliers']} pose / "
                            f"{r['local_inliers']} local-map inliers, correction {float(r['correction_m']):.2f} m / "
                            f"{float(r['correction_deg']):.1f} deg"))
    counts = []
    if lr:
        best = np.array([max(0, int(r["best_inliers"])) for r in lr], dtype=float)
        counts += [("reloc_best_inliers", "learned reloc: best final inliers per attempt", lr_t, best,
                    FEATURE_COLORS[0], "points"),
                   ("reloc_gate", f"{gate}-inlier gate", ts[[0, -1]] if len(ts) else lr_t[:0], [gate, gate],
                    [230, 40, 40], "line")]
    if an:
        counts.append(("anchor_local_inliers", "map anchor: local-map inliers per attempt", an_t,
                       [float(r["local_inliers"]) for r in an], FEATURE_COLORS[1], "points"))
    notes = []
    if not lr and mode == "relocalize":
        notes.append("stock DBoW relocalization: the matched map keyframe is not logged, so the map image "
                     "shows the keyframe nearest the query's true pose")
    mp = map_run / "algorithm" / "map_points.csv"
    return {"nodes": _map_keyframes(map_run), "node_name": "map keyframe",
            "cloud": _map_points(mp) if mp.exists() else None, "cloud_rgb": None,
            "cloud_desc": "sparse map points", "rows": rows,
            "state": [("tracking_ok", "tracking OK (x0.9)", ts,
                       [0.9 * (r["state"] in omr.OK_STATES) for r in rows], [42, 161, 152]),
                      ("vo_only", "VO-only (x0.8)", ts, [0.8 * r["vo_only"] for r in rows], [200, 120, 220])],
            "counts_name": "relocalization / anchor inliers" if counts else "counts",
            "counts": counts, "anchors": anchors, "rejected_text": "", "notes": notes}


def _ground_truth_cam0(cfg, fixed, log):
    """Query ground truth as cam0 poses in the GT world (as the evaluator builds it) or None."""
    from ..runner import make_dataset
    from ..trajectory import Trajectory
    try:
        dataset = make_dataset(cfg)
        gt = dataset.ground_truth()
        T_bs = dataset.T_gt_body_sensor("cam0")
    except Exception as exc:  # noqa: BLE001 - the replay still shows the map and the query images
        log(f"[view] ground truth unavailable ({type(exc).__name__}: {exc})")
        return None
    if gt is None or T_bs is None or not gt.has_orientation:
        log("[view] ground truth cannot be expressed in cam0")
        return None
    if fixed.get("T_gt_body_trajectory_frame") is not None and \
            not np.allclose(fixed["T_gt_body_trajectory_frame"], T_bs, atol=1e-9):
        log("[view] GT body->cam0 differs from the mapping run's; ground truth not shown")
        return None
    return Trajectory(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, gt.body_frame).transformed(T_right=T_bs)


def _blueprint(counts_name: str = "odometry features"):
    import rerun.blueprint as rrb
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(rrb.Spatial2DView(origin="/camera/query", name="query camera (left)"),
                         rrb.Spatial2DView(origin="/camera/map", name="saved map: image it should recognize"),
                         rrb.TextLogView(origin="/events", name="localization events"),
                         rrb.TextDocumentView(origin="/info", name="run"), row_shares=[3, 3, 2, 2]),
            rrb.Vertical(rrb.Spatial3DView(origin="/world", name="saved map's fixed frame (GT world)"),
                         rrb.TimeSeriesView(origin="/plots/error", name="fixed-frame position error [m]"),
                         rrb.TimeSeriesView(origin="/plots/state", name="state"),
                         rrb.TimeSeriesView(origin="/plots/counts", name=counts_name),
                         row_shares=[5, 2, 1, 1]),
            column_shares=[2, 3]))


def _info(run, mr_info, fixed, metrics, n_cloud, gt_ok, src) -> str:
    cfg, st = run["cfg"], run["status"]
    m = metrics or {}
    i, e, c = m.get("initialization") or {}, m.get("localization_events") or {}, m.get("coverage") or {}
    comp, err = m.get("computation") or {}, (m.get("error_fixed_frame") or {}).get("translation_m") or {}
    L = [f"# {cfg.get('name', '')}",
         f"**Map**: `{fixed.get('mapping_sequence')}` (run `{fixed.get('mapping_run_id')}`, mapping ATE "
         f"{(fixed.get('mapping_final_ate_rmse_m') or float('nan')):.3f} m, "
         f"{len(src['nodes']['id'])} {src['node_name']}s). "
         f"**Query**: `{cfg['dataset']['sequence']}`.",
         f"* status **{st.get('status')}**: {st.get('message', '')}", ""]
    if i.get("localized"):
        L.append(f"* first localization at {i['data_time_s']:.1f} s (frame {i['frame_idx']}), error "
                 f"{(i.get('error_at_first_localization_m') or float('nan')):.3f} m")
    else:
        L.append("* **never localized** against the map")
    if c:
        cl = c.get("correctly_localized_fraction") or {}
        L.append(f"* localized {100 * c.get('localized_fraction', 0):.1f}% of the time; correctly (<0.25 m) "
                 f"{100 * cl.get('0.25m', 0):.1f}%")
    if e:
        L.append(f"* {e.get('n_events')} accepted localizations, {e.get('n_false_relocalizations')} false, "
                 f"{e.get('n_rejected_hypotheses')} rejected hypotheses")
    if err.get("rmse") is not None:
        L.append(f"* fixed-frame error of localized poses: RMSE {err['rmse']:.3f} m")
    if (m.get("vo_only") or {}).get("fraction_of_localized") is not None:
        L.append(f"* VO-only (too few map matches) {100 * m['vo_only']['fraction_of_localized']:.1f}% of localized "
                 "frames")
    if src["anchors"]:
        L.append(f"* {len(src['anchors'])} accepted map anchors")
    if comp:
        L.append(f"* map load {comp.get('map_load_s', float('nan')):.0f} s, "
                 f"{1e3 * (comp.get('per_frame_time_s') or {}).get('mean', float('nan')):.0f} ms/frame, "
                 f"real-time factor {comp.get('real_time_factor', float('nan')):.2f}")
    L += ["", "## How to read it",
          "All geometry is in the ground-truth world via the **mapping run's** map->GT transform; nothing is "
          "fitted on this run, so a wrong localization stays visibly wrong.",
          f"* map: {src['cloud_desc']} ({n_cloud} points) and orange {src['node_name']} path of the mapping run",
          "* grey: query ground truth" + ("" if gt_ok else " (unavailable)"),
          "* query poses after localization, coloured by error: green < 0.1 m, yellow < 0.25 m, "
          "orange < 0.5 m, red >= 0.5 m; blue-grey marker = pose from the prior only (not yet localized)",
          f"* cyan spheres + lines: accepted localizations and the {src['node_name']} they matched (if logged); "
          "red crosses: rejected hypotheses (at the query's true position); magenta: accepted map anchors",
          f"* map image: the {src['node_name']} matched at the last localization or anchor if recent (< 2 s), "
          f"otherwise the {src['node_name']} nearest the query's true pose and viewing direction"]
    L += [f"* {n}" for n in src["notes"]]
    return "\n".join(L)


def export_map_reuse(run_dir: Path, out_path: Optional[Path] = None, stride: int = 2, jpeg_quality: int = 75,
                     max_width: int = 640, images: bool = True, log=print) -> Dict[str, Any]:
    import rerun as rr

    from ..evaluation.map_reuse import fixed_alignment_from
    from ..evaluation.metrics import associate

    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "replay.rrd"
    stride = max(1, int(stride))
    run = _load_run(run_dir)
    out = json.loads((run_dir / "algorithm_output.json").read_text())
    mr_info = out["extra"]["map_reuse"]
    map_run = Path(mr_info["map_run"])
    fixed = fixed_alignment_from(map_run)
    R, t = fixed["R"], fixed["t"]
    mp = run_dir / "eval" / "metrics.json"
    metrics = json.loads(mp.read_text()).get("map_reuse") if mp.exists() else None

    ts = run["frame_ts"]
    t0 = int(ts[0])

    def rel(x):
        return (np.asarray(x, dtype=np.int64) - t0) * 1e-9

    def frame_of(x):
        return np.clip(np.searchsorted(ts, x), 0, len(ts) - 1)

    if run["cfg"]["algorithm"]["name"] in ("orbslam3", "spslam3"):
        src = _orbslam3_source(run_dir, map_run, mr_info, run["cfg"])
    else:
        src = _rtabmap_source(run_dir, map_run)
    rows = src["rows"]

    rec = rr.RecordingStream("slambench", recording_id=str(run["status"].get("run_id") or run_dir.name))
    rec.save(str(out_path), default_blueprint=_blueprint(src["counts_name"]))
    rec.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def series(path, t_ns, values, style):
        if len(t_ns) == 0:
            return
        rec.log(path, style, static=True)
        rec.send_columns(path, indexes=[rr.TimeColumn("time", duration=rel(t_ns)),
                                        rr.TimeColumn("frame", sequence=frame_of(t_ns))],
                         columns=rr.Scalars.columns(scalars=np.asarray(values, dtype=float)))

    # ---- the saved map, in the GT world
    nodes = src["nodes"]
    node_world = nodes["pos"] @ R.T + t
    node_axis = nodes["z_axis"] @ R.T
    node_idx = {int(k): j for j, k in enumerate(nodes["id"].tolist())}
    rec.log("/world/map/nodes", rr.LineStrips3D([node_world], colors=[MAP_NODE_COLOR], radii=0.008), static=True)
    rec.log("/world/map/node_points", rr.Points3D(node_world, colors=[MAP_NODE_COLOR], radii=0.02,
                                                  labels=[f"{src['node_name']} {k}" for k in nodes["id"].tolist()]),
            static=True)
    n_cloud = 0
    if src["cloud"] is not None and len(src["cloud"]):
        xyz = src["cloud"] @ R.T + t
        if src["cloud_rgb"] is not None:
            rec.log("/world/map/cloud", rr.Points3D(xyz, colors=src["cloud_rgb"], radii=0.006), static=True)
        else:  # sparse map points: coloured by height in the GT world
            z = xyz[:, 2]
            u = np.clip((z - np.percentile(z, 2)) / max(1e-6, np.ptp(np.percentile(z, [2, 98]))), 0, 1)
            col = (np.stack([0.35 + 0.65 * u, 0.35 + 0.45 * u, 0.9 - 0.6 * u], 1) * 255).astype(np.uint8)
            rec.log("/world/map/cloud", rr.Points3D(xyz, colors=col, radii=0.01), static=True)
        n_cloud = len(xyz)
    map_imgs = _map_images(map_run, nodes["t_ns"])

    # ---- query: poses in the fixed frame, GT, errors
    gt = _ground_truth_cam0(run["cfg"], fixed, log)
    pose_rows = [r for r in rows if r["pose"] is not None]
    p_t = np.array([r["timestamp_ns"] for r in pose_rows], dtype=np.int64)
    p_world = (np.array([r["pose"][:3] for r in pose_rows]).reshape(-1, 3) @ R.T + t)
    p_loc = np.array([r["localized"] for r in pose_rows], dtype=bool)
    err = np.full(len(pose_rows), np.nan)
    gt_by_ts: Dict[int, Any] = {}
    if gt is not None:
        rec.log("/world/query/ground_truth", rr.LineStrips3D([gt.positions], colors=[GT_COLOR], radii=0.01),
                static=True)
        ei, gi, _ = associate(p_t, gt.timestamps_ns, 0.02)
        err[ei] = np.linalg.norm(p_world[ei] - gt.positions[gi], axis=1)
        fi, gfi, _ = associate(ts, gt.timestamps_ns, 0.05)
        gaxis = gt.poses()[:, :3, 2]
        for a, b in zip(fi.tolist(), gfi.tolist()):
            gt_by_ts[int(ts[a])] = (gt.positions[b], gaxis[b])
    if p_loc.any():
        rec.log("/world/query/localized", rr.Points3D(p_world[p_loc], colors=[_err_color(e) for e in err[p_loc]],
                                                      radii=0.012), static=True)
        k = p_loc & np.isfinite(err)
        series("/plots/error/localized", p_t[k], err[k],
               rr.SeriesPoints(colors=[[40, 190, 60]], markers=["circle"], marker_sizes=[1.5],
                               names=["localized (fixed frame)"]))
    if (~p_loc).any() and np.isfinite(err[~p_loc]).any():
        k = ~p_loc & np.isfinite(err)
        series("/plots/error/prior_only", p_t[k], err[k],
               rr.SeriesPoints(colors=[PRIOR_COLOR], markers=["circle"], marker_sizes=[1.0],
                               names=["not localized (prior only)"]))
    for thr, col in ERR_BANDS[:-1]:
        series(f"/plots/error/threshold_{thr}", ts[[0, -1]], [thr, thr],
               rr.SeriesLines(colors=[col], names=[f"{thr} m"], widths=[0.5]))

    # ---- state plots
    lt = np.array([r["timestamp_ns"] for r in rows], dtype=np.int64)
    series("/plots/state/localized", lt, [float(r["localized"]) for r in rows],
           rr.SeriesLines(colors=[[40, 190, 60]], names=["localized"]))
    for key, name, st_t, vals, col in src["state"]:
        series(f"/plots/state/{key}", st_t, vals, rr.SeriesLines(colors=[col], names=[name]))
    ev = [r for r in rows if r["loc_event"]]
    rj = [r for r in rows if r["loc_rejected"]]
    series("/plots/state/accepted", np.array([r["timestamp_ns"] for r in ev], dtype=np.int64), [1.1] * len(ev),
           rr.SeriesPoints(colors=[EVENT_COLOR], markers=["diamond"], marker_sizes=[4], names=["accepted"]))
    series("/plots/state/rejected", np.array([r["timestamp_ns"] for r in rj], dtype=np.int64), [1.2] * len(rj),
           rr.SeriesPoints(colors=[REJECT_COLOR], markers=["cross"], marker_sizes=[4], names=["rejected"]))
    an_t = np.array([a[0] for a in src["anchors"]], dtype=np.int64)
    series("/plots/state/anchor", an_t, [1.0] * len(an_t),
           rr.SeriesPoints(colors=[ANCHOR_COLOR], markers=["up"], marker_sizes=[3], names=["map anchor"]))
    for key, name, c_t, vals, col, kind in src["counts"]:
        style = (rr.SeriesLines(colors=[col], names=[name]) if kind == "line" else
                 rr.SeriesPoints(colors=[col], markers=["circle"], marker_sizes=[1.5], names=[name]))
        series(f"/plots/counts/{key}", c_t, vals, style)

    # ---- events in 3D and as text
    pos_by_ts = dict(zip(p_t.tolist(), p_world))
    err_by_ts = dict(zip(p_t.tolist(), err.tolist()))
    ev_pts, ev_lab, ev_lines = [], [], []
    rj_pts = []
    first_done = False
    for r in rows:
        tt = r["timestamp_ns"]
        msgs = []
        if r["loc_event"]:
            matched = r["match_id"]
            e = err_by_ts.get(tt, np.nan)
            txt = (f"{'FIRST ' if not first_done else ''}localization {r['event_text']}, error after "
                   + (f"{e:.3f} m" if np.isfinite(e) else "n/a"))
            msgs.append(("/events/accepted", "WARN" if np.isfinite(e) and e >= 0.5 else "INFO", txt))
            first_done = True
            if tt in pos_by_ts:
                ev_pts.append(pos_by_ts[tt])
                ev_lab.append(f"{rel(tt):.0f} s")
                if matched in node_idx:
                    ev_lines.append([pos_by_ts[tt], node_world[node_idx[matched]]])
        if r["loc_rejected"]:
            msgs.append(("/events/rejected", "DEBUG", src["rejected_text"]))
            g = gt_by_ts.get(tt)
            if g is not None:
                rj_pts.append(g[0])
        if msgs:
            rec.set_time("time", duration=float(rel(tt)))
            rec.set_time("frame", sequence=int(frame_of(tt)))
            for path, lvl, m in msgs:
                rec.log(path, rr.TextLog(m, level=lvl))
    if ev_pts:
        rec.log("/world/query/accepted", rr.Points3D(ev_pts, colors=[EVENT_COLOR], radii=0.05, labels=ev_lab),
                static=True)
    if ev_lines:
        rec.log("/world/query/accepted_match", rr.LineStrips3D(ev_lines, colors=[EVENT_COLOR], radii=0.006),
                static=True)
    if rj_pts:
        rec.log("/world/query/rejected", rr.Points3D(rj_pts, colors=[REJECT_COLOR], radii=0.04), static=True)
    an_pts = []
    for tt, kf, txt in src["anchors"]:
        e = err_by_ts.get(tt, np.nan)
        rec.set_time("time", duration=float(rel(tt)))
        rec.set_time("frame", sequence=int(frame_of(tt)))
        rec.log("/events/anchor", rr.TextLog(txt + ", error after " + (f"{e:.3f} m" if np.isfinite(e) else "n/a"),
                                             level="DEBUG"))
        if tt in pos_by_ts:
            an_pts.append(pos_by_ts[tt])
    if an_pts:
        rec.log("/world/query/anchors", rr.Points3D(an_pts, colors=[ANCHOR_COLOR], radii=0.03), static=True)
    anchor_by_ts = {tt: (kf, txt) for tt, kf, txt in src["anchors"]}

    # ---- per frame: current pose, GT, error line, query and map images
    features = _load_features(run, [])
    img_idx = list(range(0, len(run["frames"]), stride)) if images else []
    pool = ThreadPoolExecutor(8) if img_idx else None
    pending: Dict[int, Any] = {}

    def submit(n):
        for j in img_idx[n:n + 1]:
            pending[j] = pool.submit(_encode, run["prep_dir"] / run["frames"][j][1], jpeg_quality, max_width)

    for n in range(min(PREFETCH, len(img_idx))):
        submit(n)
    next_n = min(PREFETCH, len(img_idx))
    map_cache: Dict[int, Any] = {}
    last_map_node, last_match = None, (None, -np.inf, "")
    row_by_ts = {r["timestamp_ns"]: r for r in rows}
    n_img = 0
    for i, (tt, _, _) in enumerate(run["frames"]):
        rec.set_time("time", duration=float(rel(tt)))
        rec.set_time("frame", sequence=i)
        r = row_by_ts.get(tt)
        g = gt_by_ts.get(tt)
        p = pos_by_ts.get(tt)
        if p is not None:
            loc = bool(r and r["localized"])
            e = err_by_ts.get(tt, np.nan)
            rec.log("/world/query/now", rr.Points3D([p], colors=[_err_color(e) if loc else PRIOR_COLOR], radii=0.08,
                                                     labels=["localized" if loc else "prior only"]))
            if g is not None:
                rec.log("/world/query/now_error", rr.LineStrips3D([[p, g[0]]], colors=[[230, 40, 40]], radii=0.005))
        else:
            rec.log("/world/query/now", rr.Clear(recursive=False))
            rec.log("/world/query/now_error", rr.Clear(recursive=False))
        if g is not None:
            rec.log("/world/query/ground_truth_now", rr.Points3D([g[0]], colors=[GT_COLOR], radii=0.07))
        if r is not None and r["loc_event"] and r["match_id"] in node_idx:
            last_match = (node_idx[r["match_id"]], rel(tt), "matched by the last accepted localization")
        if tt in anchor_by_ts and anchor_by_ts[tt][0] in node_idx:
            last_match = (node_idx[anchor_by_ts[tt][0]], rel(tt), "matched by the last map anchor")
        if i in pending:
            res = pending.pop(i).result()
            if res is not None:
                data, scale = res
                rec.log("/camera/query", rr.EncodedImage(contents=data, media_type="image/jpeg"))
                n_img += 1
                f = features.get(tt) if features else None
                if f is not None and len(f[0]):
                    rec.log("/camera/query/features", rr.Points2D(
                        f[0] * scale, colors=[FEATURE_COLORS.get(int(x), [255, 255, 255]) for x in f[1]], radii=2.0))
                elif features:
                    rec.log("/camera/query/features", rr.Clear(recursive=False))
            submit(next_n)
            next_n += 1
            # map image: the recently matched node, else the node nearest the query's true view
            j, why = None, ""
            if last_match[0] is not None and rel(tt) - last_match[1] < 2.0:
                j, why = last_match[0], last_match[2]
            elif g is not None and len(node_world):
                d = np.linalg.norm(node_world - g[0], axis=1) + 1.0 * (1.0 - node_axis @ g[1])
                j = int(np.argmin(d))
                why = f"nearest to the query's true pose ({np.linalg.norm(node_world[j] - g[0]):.2f} m away)"
            if j is not None and j != last_map_node and map_imgs[j] is not None:
                if j not in map_cache:
                    map_cache[j] = _encode(map_imgs[j], jpeg_quality, max_width)
                if map_cache[j] is not None:
                    rec.log("/camera/map", rr.EncodedImage(contents=map_cache[j][0], media_type="image/jpeg"))
                    rec.log("/camera/map/label", rr.Points2D([[6, 10]], radii=0.1, colors=[[255, 255, 0]],
                                                             labels=[f"{src['node_name']} {int(nodes['id'][j])}: {why}"]))
                last_map_node = j
    if pool is not None:
        pool.shutdown()

    rec.log("/info", rr.TextDocument(_info(run, mr_info, fixed, metrics, n_cloud, gt is not None, src),
                                     media_type=rr.MediaType.MARKDOWN), static=True)
    rec.flush()
    rec.disconnect()
    return {"out": str(out_path), "view": "map_reuse", "n_frames": int(len(ts)), "n_images": n_img,
            "map_run": str(map_run), "map_nodes": int(len(node_world)), "map_cloud_points": n_cloud,
            "n_localized_poses": int(p_loc.sum()), "n_accepted": len(ev), "n_rejected": len(rj),
            "n_anchors": len(src["anchors"]),
            "ground_truth": gt is not None, "size_mb": round(out_path.stat().st_size / 2 ** 20, 2),
            "rerun_version": rr.__version__}
