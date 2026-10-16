"""Rerun replay of a map-reuse (localization) run, drawn in the saved map's fixed frame.

Everything is placed in the ground-truth world with the *mapping run's* single map->GT transform
(``fixed_alignment_from``), exactly as ``map_reuse_metrics`` scores it; nothing is fitted on the
query run, so a wrong localization stays visibly wrong.

3D: the saved map (dense cloud and node path of the mapping run), the query's ground truth, the
query's localized poses coloured by fixed-frame error, accepted localizations (with a line to the
matched map node) and rejected hypotheses. Images: the query camera and the map image the vehicle
should be recognizing (the map node it matched at a localization, otherwise the node closest to its
true pose). Plots: fixed-frame error, localized / odometry state, features, timing.
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


def _map_images(map_run: Path, node_ts: np.ndarray) -> List[Optional[Path]]:
    """Left image of each map node in the mapping run's prepared input (None if unavailable)."""
    pj = map_run / "prepared_input.json"
    if not pj.exists():
        return [None] * len(node_ts)
    prep = Path(json.loads(pj.read_text())["directory"])
    by_ts = {}
    try:
        with open(prep / "frames.csv") as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split(",")
                by_ts[int(c[0])] = prep / c[1]
    except OSError:
        return [None] * len(node_ts)
    return [by_ts.get(int(t)) for t in node_ts.tolist()]


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


def _blueprint():
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
                         rrb.TimeSeriesView(origin="/plots/features", name="odometry features"),
                         row_shares=[5, 2, 1, 1]),
            column_shares=[2, 3]))


def _info(run, mr_info, fixed, metrics, n_cloud, gt_ok) -> str:
    cfg, st = run["cfg"], run["status"]
    m = metrics or {}
    i, e, c = m.get("initialization") or {}, m.get("localization_events") or {}, m.get("coverage") or {}
    comp, err = m.get("computation") or {}, (m.get("error_fixed_frame") or {}).get("translation_m") or {}
    L = [f"# {cfg.get('name', '')}",
         f"**Map**: `{fixed.get('mapping_sequence')}` (run `{fixed.get('mapping_run_id')}`, mapping ATE "
         f"{(fixed.get('mapping_final_ate_rmse_m') or float('nan')):.3f} m, "
         f"{(mr_info.get('map_load') or {}).get('n_map_nodes')} nodes). "
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
    if comp:
        L.append(f"* map load {comp.get('map_load_s', float('nan')):.0f} s, "
                 f"{1e3 * (comp.get('per_frame_time_s') or {}).get('mean', float('nan')):.0f} ms/frame, "
                 f"real-time factor {comp.get('real_time_factor', float('nan')):.2f}")
    L += ["", "## How to read it",
          "All geometry is in the ground-truth world via the **mapping run's** map->GT transform; nothing is "
          "fitted on this run, so a wrong localization stays visibly wrong.",
          f"* map: dense cloud ({n_cloud} points) and orange node path of the mapping run",
          "* grey: query ground truth" + ("" if gt_ok else " (unavailable)"),
          "* query poses after localization, coloured by error: green < 0.1 m, yellow < 0.25 m, "
          "orange < 0.5 m, red >= 0.5 m; blue-grey marker = pose from the prior only (not yet localized)",
          "* cyan spheres + lines: accepted localizations and the map node they matched; red crosses: "
          "rejected hypotheses (at the query's true position)",
          "* map image: the node matched at the last localization if recent (< 2 s), otherwise the map "
          "node nearest the query's true pose and viewing direction"]
    return "\n".join(L)


def export_map_reuse(run_dir: Path, out_path: Optional[Path] = None, stride: int = 2, jpeg_quality: int = 75,
                     max_width: int = 640, images: bool = True, log=print) -> Dict[str, Any]:
    import rerun as rr

    from ..algorithms.rtabmap import RtabmapAdapter
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

    rec = rr.RecordingStream("slambench", recording_id=str(run["status"].get("run_id") or run_dir.name))
    rec.save(str(out_path), default_blueprint=_blueprint())
    rec.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def series(path, t_ns, values, style):
        if len(t_ns) == 0:
            return
        rec.log(path, style, static=True)
        rec.send_columns(path, indexes=[rr.TimeColumn("time", duration=rel(t_ns)),
                                        rr.TimeColumn("frame", sequence=frame_of(t_ns))],
                         columns=rr.Scalars.columns(scalars=np.asarray(values, dtype=float)))

    # ---- the saved map, in the GT world
    nodes = _map_nodes(map_run)
    node_world = nodes["pos"] @ R.T + t
    node_axis = nodes["z_axis"] @ R.T
    node_idx = {int(k): j for j, k in enumerate(nodes["id"].tolist())}
    rec.log("/world/map/nodes", rr.LineStrips3D([node_world], colors=[MAP_NODE_COLOR], radii=0.008), static=True)
    rec.log("/world/map/node_points", rr.Points3D(node_world, colors=[MAP_NODE_COLOR], radii=0.02,
                                                  labels=[f"map node {k}" for k in nodes["id"].tolist()]),
            static=True)
    n_cloud = 0
    cloud_path = map_run / "algorithm" / "rtabmap_cloud.ply"
    if cloud_path.exists():
        xyz, rgb = read_ply_vertices(cloud_path, MAP_CLOUD_MAX_POINTS)
        rec.log("/world/map/cloud", rr.Points3D(xyz @ R.T + t, colors=rgb, radii=0.006), static=True)
        n_cloud = len(xyz)
    map_imgs = _map_images(map_run, nodes["t_ns"])

    # ---- query: poses in the fixed frame, GT, errors
    rows, _ = RtabmapAdapter.read_frame_log(run_dir / "algorithm" / "frame_log.csv")
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
    series("/plots/state/odometry_ok", lt, [0.9 * r["odom_ok"] for r in rows],
           rr.SeriesLines(colors=[[42, 161, 152]], names=["odometry OK (x0.9)"]))
    ev = [r for r in rows if r["loc_event"]]
    rj = [r for r in rows if r["loc_rejected"]]
    series("/plots/state/accepted", np.array([r["timestamp_ns"] for r in ev], dtype=np.int64), [1.1] * len(ev),
           rr.SeriesPoints(colors=[EVENT_COLOR], markers=["diamond"], marker_sizes=[4], names=["accepted"]))
    series("/plots/state/rejected", np.array([r["timestamp_ns"] for r in rj], dtype=np.int64), [1.2] * len(rj),
           rr.SeriesPoints(colors=[REJECT_COLOR], markers=["cross"], marker_sizes=[4], names=["rejected"]))
    series("/plots/features/features", lt, [r["features"] for r in rows],
           rr.SeriesLines(colors=[FEATURE_COLORS[0]], names=["keypoints"]))
    series("/plots/features/inliers", lt, [r["inliers"] for r in rows],
           rr.SeriesLines(colors=[FEATURE_COLORS[1]], names=["odometry inliers"]))

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
            matched = r["loop_closure_id"] or r["proximity_id"]
            e = err_by_ts.get(tt, np.nan)
            kind = "loop closure" if r["loop_closure_id"] else "proximity"
            txt = (f"{'FIRST ' if not first_done else ''}localization ({kind}) against map node {matched}: "
                   f"correction {r['loc_correction_m']:.2f} m, error after "
                   + (f"{e:.3f} m" if np.isfinite(e) else "n/a"))
            msgs.append(("/events/accepted", "WARN" if np.isfinite(e) and e >= 0.5 else "INFO", txt))
            first_done = True
            if tt in pos_by_ts:
                ev_pts.append(pos_by_ts[tt])
                ev_lab.append(f"{rel(tt):.0f} s")
                if matched in node_idx:
                    ev_lines.append([pos_by_ts[tt], node_world[node_idx[matched]]])
        if r["loc_rejected"]:
            msgs.append(("/events/rejected", "DEBUG", "localization hypothesis rejected (graph-consistency check)"))
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
    last_map_node, last_match = None, (None, -np.inf)
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
        if r is not None and r["loc_event"]:
            m = r["loop_closure_id"] or r["proximity_id"]
            if m in node_idx:
                last_match = (node_idx[m], rel(tt))
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
                j, why = last_match[0], "matched by the last accepted localization"
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
                                                             labels=[f"map node {int(nodes['id'][j])}: {why}"]))
                last_map_node = j
    if pool is not None:
        pool.shutdown()

    rec.log("/info", rr.TextDocument(_info(run, mr_info, fixed, metrics, n_cloud, gt is not None),
                                     media_type=rr.MediaType.MARKDOWN), static=True)
    rec.flush()
    rec.disconnect()
    return {"out": str(out_path), "view": "map_reuse", "n_frames": int(len(ts)), "n_images": n_img,
            "map_run": str(map_run), "map_nodes": int(len(node_world)), "map_cloud_points": n_cloud,
            "n_localized_poses": int(p_loc.sum()), "n_accepted": len(ev), "n_rejected": len(rj),
            "ground_truth": gt is not None, "size_mb": round(out_path.stat().st_size / 2 ** 20, 2),
            "rerun_version": rr.__version__}
