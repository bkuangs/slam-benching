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


def _ctx(tmp_path, settings=None):
    ds = FakeDataset(tmp_path, "s1")
    (tmp_path / "prep").mkdir(exist_ok=True)
    prep = ds.prepare_offline(SensorMode.STEREO, tmp_path / "prep")
    a = RtabmapAdapter({"container": str(tmp_path / "x.sif")})
    return a, ds, RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.STEREO, prep,
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
                {"cloud_decimation": 0}, {"grid_cell_size": 0}, {"detection_rate_hz": -1}):
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
