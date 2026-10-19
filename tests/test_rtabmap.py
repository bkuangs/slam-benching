"""RTAB-Map adapter tests: settings, native parameters, command and output collection (no image)."""

import csv
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from slambench.algorithms import ALGORITHMS
from slambench.algorithms.base import RunContext
from slambench.algorithms.rtabmap import ODOM_COLUMNS, POSE_COLUMNS, RtabmapAdapter
from slambench.trajectory import make_T, read_tum
from slambench.types import RunStatus, SensorMode

from synthetic import FakeDataset, sample

LOG_COLS = ["frame_idx", "timestamp_ns", "odom_ok", "features", "inliers", "matches", "local_map_size",
            "keyframe_added", "slam_update", "node_added", "node_id", "map_id", "loop_closure_id",
            "proximity_id", "wm_size", "odom_time_s", "slam_time_s", "track_time_s", "coord_session",
            *POSE_COLUMNS, *ODOM_COLUMNS]
NODE_COLS = ["node_id", "map_id", "timestamp_ns", "in_final", *ODOM_COLUMNS, *POSE_COLUMNS]


def _vec(T):
    return list(T[:3, 3]) + list(Rotation.from_matrix(T[:3, :3]).as_quat())


def _ctx(tmp_path, settings=None, algo="algo"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds = FakeDataset(tmp_path, "s1")
    (tmp_path / "prep").mkdir(exist_ok=True)
    prep = ds.prepare_offline(SensorMode.STEREO, tmp_path / "prep")
    a = RtabmapAdapter({"container": str(tmp_path / "x.sif")})
    return a, ds, RunContext(tmp_path, tmp_path / algo, ds, SensorMode.STEREO, prep,
                             a.resolve_settings(settings or {}))


def _write_outputs(ctx, ds, lost=(100, 130), unfinal_node=None):
    """Odometry = D * GT (constant drift D); final graph nodes every 15 frames at their GT pose."""
    d = ctx.algo_dir
    d.mkdir(parents=True, exist_ok=True)
    gt = sample(ds.gt, ds.frames).poses()
    D = make_T(Rotation.from_euler("z", 0.3).as_matrix(), [1.0, -2.0, 0.5])
    with open(d / "frame_log.csv", "w", newline="") as f, open(d / "nodes.csv", "w", newline="") as g:
        w, wn = csv.writer(f), csv.writer(g)
        w.writerow(LOG_COLS)
        wn.writerow(NODE_COLS)
        node = 0
        for i, t in enumerate(ds.frames):
            ok = not (lost[0] <= i < lost[1])
            odom = D @ gt[i]
            added = ok and i % 15 == 0
            if added:
                node += 1
                fin = node != unfinal_node
                wn.writerow([node, 0, t, int(fin), *_vec(odom), *(_vec(gt[i]) if fin else [""] * 7)])
            pose = _vec(odom) + _vec(odom) if ok else [""] * 14
            w.writerow([i, t, int(ok), 500, 120, 150, 800, int(added), int(added), int(added), node, 0,
                        7 if i == 200 else 0, 0, node, 0.01, 0.02 if added else 0, 0.03, 0, *pose])
    with open(d / "links.csv", "w") as f:
        f.write("from,to,type\n1,2,Neighbor\n2,3,Neighbor\n20,1,GlobalClosure\n")
    (d / "driver_status.json").write_text(json.dumps({"status": "completed", "message": ""}))


def test_registered_and_rejects_non_stereo_and_bad_settings(tmp_path):
    assert ALGORITHMS["rtabmap"] is RtabmapAdapter
    a, ds, _ = _ctx(tmp_path)
    s = a.resolve_settings({})
    assert "does not support sensor mode 'mono'" in a.validate(SensorMode.MONO, ds, s)[0]
    assert any("container not found" in p for p in a.validate(SensorMode.STEREO, ds, s))
    for bad in ({"nope": 1}, {"extra_parameters": {"Odom/ResetCountdown": 1}},
                {"extra_parameters": {"Grid/CellSize": 0.1}}, {"extra_parameters": {"NoSlash": 1}},
                {"cloud_decimation": 0}, {"grid_cell_size": 0}, {"detection_rate_hz": -1},
                {"occupancy_grid": False}):  # the OctoMap export (on by default) needs the grids
        with pytest.raises(ValueError):
            a.resolve_settings(bad)


def test_parameters_command_and_export(tmp_path):
    a, ds, ctx = _ctx(tmp_path, {"extra_parameters": {"Odom/Strategy": 1, "Vis/MinInliers": 25}})
    a.prepare(ctx)
    text = (ctx.algo_dir / "rtabmap_params.txt").read_text()
    for line in ("RGBD/CreateOccupancyGrid=true", "Grid/RayTracing=true", "Grid/3D=true", "Grid/CellSize=0.05",
                 "Kp/MaxFeatures=500", "Stereo/OpticalFlow=false", "Odom/Strategy=1", "Vis/MinInliers=25", "Mem/IncrementalMemory=true"):
        assert line in text.splitlines(), line
    cmd = a.command(ctx)
    assert cmd[-3:-1] == ["/bin/bash", "-c"]
    script = cmd[-1]
    assert "--camera 260.0,260.0,320.0,180.0,0.12,640,360" in script
    assert "--detection-rate 1.0" in script and "--playback-rate 1.0" in script and "--log-features" in script
    assert "rtabmap-export" in script and "--opt 2" in script and "--octomap" in script and "--cloud" in script
    assert script.endswith("exit $rc")
    ctx.settings.update(a.resolve_settings({"loop_closure": False, "export_cloud": False,
                                            "export_octomap": False}))
    assert "Kp/MaxFeatures=-1" in [f"{k}={v}" for k, v in a.parameters(ctx.settings).items()]
    assert a.parameters(ctx.settings)["RGBD/ProximityBySpace"] == "false"
    assert "rtabmap-export" not in a.command(ctx)[-1]
    assert "--threaded-map-update" not in a.command(ctx)[-1]


def test_threaded_map_update_and_no_occupancy_grid(tmp_path):
    a, ds, ctx = _ctx(tmp_path, {"threaded_map_update": True, "occupancy_grid": False,
                                 "export_octomap": False})
    p = a.parameters(ctx.settings)
    assert p["RGBD/CreateOccupancyGrid"] == "false"
    assert not any(k.startswith("Grid/") for k in p)
    script = a.command(ctx)[-1]
    assert "--threaded-map-update" in script
    assert "rtabmap-export" in script and "--cloud" in script and "--octomap" not in script


def test_collect_final_trajectory_reanchors_odometry_on_graph(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    _write_outputs(ctx, ds, unfinal_node=3)
    (ctx.algo_dir / "export_rc.txt").write_text("0\n")
    for n in ("rtabmap.db", "rtabmap_cloud.ply", "rtabmap_octomap.bt"):
        (ctx.algo_dir / n).write_bytes(b"x")
    out = a.collect(ctx, 0, False)
    assert out.status == RunStatus.SUCCESS and out.message == ""
    n = len(ds.frames)
    fin = read_tum(tmp_path / "trajectory.tum")
    assert len(fin) == n - 30
    gt = sample(ds.gt, fin.timestamps_ns)
    assert np.abs(fin.positions - gt.positions).max() < 1e-6     # drift removed by the graph
    onl = read_tum(tmp_path / "trajectory_online.tum")
    assert len(onl) == n - 30 and np.abs(onl.positions - gt.positions).max() > 0.5
    rows = list(csv.DictReader(open(tmp_path / "frame_diagnostics.csv")))
    assert [r["tracking_state"] for r in rows[99:101]] == ["ok", "lost"]
    assert rows[100]["tracking_ok"] == "0" and rows[100]["in_final_trajectory"] == "0"
    assert rows[0]["coordinate_session"] == "0" and rows[0]["map_id"] == "0"
    g = out.extra["graph"]
    assert g["n_nodes_final"] == g["n_nodes_added"] - 1 and g["n_loop_closure_detections"] == 1
    assert g["links_by_type"] == {"Neighbor": 2, "GlobalClosure": 1}
    assert {"rtabmap_db", "dense_cloud", "octomap", "map_nodes", "map_links"} <= set(out.artifacts)
    assert [t.kind for t in out.trajectories] == ["online", "final"]


def test_collect_flags_missing_export_and_failures(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    _write_outputs(ctx, ds)
    (ctx.algo_dir / "export_rc.txt").write_text("1\n")
    out = a.collect(ctx, 0, False)
    assert out.status == RunStatus.SUCCESS and "map export incomplete" in out.message
    assert out.extra["map_export"]["missing"] == ["dense_cloud", "octomap"]
    assert a.collect(ctx, -9, False).status == RunStatus.CRASH
    assert a.collect(ctx, None, True).status == RunStatus.TIMEOUT
    _write_outputs(ctx, ds, lost=(0, 10 ** 6))
    assert a.collect(ctx, 0, False).status == RunStatus.INIT_FAILURE


def _mapping_run(tmp_path):
    """A fake completed mapping run: outputs of _write_outputs plus params, config and eval."""
    a, ds, ctx = _ctx(tmp_path / "map", algo="algorithm")
    a.prepare(ctx)
    _write_outputs(ctx, ds)
    (ctx.algo_dir / "rtabmap.db").write_bytes(b"map database")
    (tmp_path / "map" / "config.resolved.yaml").write_text("name: m\n")
    A = make_T(Rotation.from_euler("z", 0.2).as_matrix(), [1.0, 2.0, 0.0])   # map -> GT
    (tmp_path / "map" / "eval").mkdir()
    (tmp_path / "map" / "eval" / "metrics.json").write_text(json.dumps({
        "run_id": "m", "sequence": "s1", "accuracy": {"final": {
            "sessions": [{"alignment": {"type": "se3", "R": A[:3, :3].tolist(), "t": A[:3, 3].tolist()}}],
            "ate": {"translation_m": {"rmse": 0.01}}, "ground_truth": {}}}}))
    return tmp_path / "map", ds, A


def _write_localization(ctx, ds, map_run, A, first=50, false_at=150):
    """Prior-only (wrong) poses before `first`, correct map-frame poses after, a 2 m wrong
    relocalization at `false_at` (corrected at false_at + 20)."""
    d = ctx.algo_dir
    with open(d / "map_loaded_nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", *POSE_COLUMNS])
        for r in csv.DictReader(open(map_run / "algorithm" / "nodes.csv")):
            w.writerow([r["node_id"], *[r[c] for c in POSE_COLUMNS]])
    gt = sample(ds.gt, ds.frames).poses()
    Ainv = np.linalg.inv(A)
    with open(d / "frame_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLS + ["localized", "loc_event", "loc_rejected", "loc_correction_m"])
        for i, t in enumerate(ds.frames):
            P = Ainv @ gt[i]
            if i < first:
                P = make_T(np.eye(3), [3.0, 0, 0]) @ P
            if false_at <= i < false_at + 20:
                P = make_T(np.eye(3), [0, 2.0, 0]) @ P
            ev = i in (first, false_at, false_at + 20, 250)
            w.writerow([i, t, 1, 500, 120, 150, 800, 0, int(i % 15 == 0), 0, 0, -1, 7 if ev else 0, 0, 10,
                        0.01, 0.2 if i % 15 == 0 else 0, 0.03, 0, *_vec(P), *_vec(gt[i]),
                        int(i >= first), int(ev), int(i == 10), 0.5 if ev else 0])
    (d / "map_load.json").write_text(json.dumps({"init_time_s": 1.5, "n_map_nodes": 3}))
    (d / "driver_status.json").write_text(json.dumps({"status": "completed", "message": ""}))


def test_localization_mode_settings_and_command(tmp_path):
    map_run, _, _ = _mapping_run(tmp_path)
    a, ds, ctx = _ctx(tmp_path / "loc", {"map_run": str(map_run)})
    assert a.parameters(ctx.settings)["Mem/IncrementalMemory"] == "false"
    assert a.parameters(ctx.settings)["Mem/InitWMWithAllNodes"] == "true"
    assert a._map_reuse_problems(ctx.settings) == []
    a.prepare(ctx)
    assert (ctx.algo_dir / "rtabmap.db").read_bytes() == b"map database"
    script = a.command(ctx)[-1]
    assert "--localize" in script and "rtabmap-export" not in script
    bad = a.resolve_settings({"map_run": str(map_run), "vis_max_features": 500})
    assert "Vis/MaxFeatures" in a._map_reuse_problems(bad)[0]
    assert "no algorithm/rtabmap.db" in a._map_reuse_problems(a.resolve_settings({"map_run": str(tmp_path)}))[0]


def test_localization_collect_and_fixed_frame_metrics(tmp_path):
    from slambench.evaluation.map_reuse import fixed_alignment_from, map_reuse_metrics
    map_run, _, A = _mapping_run(tmp_path)
    a, ds, ctx = _ctx(tmp_path / "loc", {"map_run": str(map_run)})
    a.prepare(ctx)
    _write_localization(ctx, ds, map_run, A)
    out = a.collect(ctx, 0, False)
    assert out.status == RunStatus.SUCCESS, out.message
    mr = out.extra["map_reuse"]
    assert mr["map_database_unchanged"] and mr["map_frame_check"]["same_frame"]
    assert mr["first_localization_frame_idx"] == 50 and mr["n_localization_events"] == 4
    assert [t.kind for t in out.trajectories] == ["online"]
    assert out.trajectories[0].coordinate_sessions == "single"
    assert len(read_tum(tmp_path / "loc" / "trajectory_online.tum")) == len(ds.frames) - 50
    rows = list(csv.DictReader(open(tmp_path / "loc" / "frame_diagnostics.csv")))
    assert rows[0]["tracking_state"] == "not_localized" and rows[50]["tracking_ok"] == "1"

    # a changed map database or a different loaded map frame invalidates the run
    (map_run / "algorithm" / "rtabmap.db").write_bytes(b"changed")
    assert a.collect(ctx, 0, False).status == RunStatus.CRASH
    (map_run / "algorithm" / "rtabmap.db").write_bytes(b"map database")

    log, _ = a.read_frame_log(ctx.algo_dir / "frame_log.csv")
    gt = sample(ds.gt, ds.gt.timestamps_ns)
    fixed = fixed_alignment_from(map_run)
    m = map_reuse_metrics(log, np.array(ds.frames), gt, fixed, {}, {"wall_time_s": 9.0},
                          mr, 0.02, tmp_path)
    ini = m["initialization"]
    assert ini["frame_idx"] == 50 and ini["error_at_first_localization_m"] < 1e-6
    ev = m["localization_events"]
    assert ev["n_events"] == 4 and ev["n_false_relocalizations"] == 1 and ev["n_rejected_hypotheses"] == 1
    assert [e["false_relocalization"] for e in ev["events"]] == [False, True, False, False]
    n = len(ds.frames)
    cov = m["coverage"]
    assert abs(cov["localized_fraction"] - (n - 50) / n) < 0.01
    assert abs(cov["correctly_localized_fraction"]["0.1m"] - (n - 70) / n) < 0.01
    assert m["error_fixed_frame"]["translation_m"]["max"] > 1.9
    assert m["error_prior_only"]["translation_m"]["min"] > 2.9
    assert m["computation"]["map_load_s"] == 1.5 and m["computation"]["n_map_updates"] > 0
    assert (tmp_path / "map_reuse_frames.csv").exists()
