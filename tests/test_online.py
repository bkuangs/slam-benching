"""Online vs final trajectory evaluation through the real ORB-SLAM3 output normalization.

A fake ORB-SLAM3 writes driver-format files (frame_log.csv with online poses, trajectory_euroc.txt,
driver_status.json); everything after execution (OrbSlam3Adapter.collect, evaluation, summary) is
the production code path.
"""

import csv
import json
import os
import signal
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from slambench import runner
from slambench.algorithms import ALGORITHMS
from slambench.algorithms.base import AlgorithmAdapter, RunContext
from slambench.algorithms.orbslam3 import OrbSlam3Adapter
from slambench.config import DEFAULTS, deep_merge, validate_config
from slambench.datasets import DATASETS
from slambench.evaluation.metrics import evaluate_sessions
from slambench.evaluation.summary import write_summary
from slambench.trajectory import Trajectory, invert_T
from slambench.types import SensorMode

from synthetic import FakeDataset, make_T, random_world, sample  # noqa: F401

# Driver log before coord_session existed (evaluator falls back to map ids) and the current one.
LOG_HEADER = "frame_idx,timestamp_ns,state,map_id,map_changed,track_time_s,tx,ty,tz,qx,qy,qz,qw\n"
LOG_HEADER_SESSIONS = ("frame_idx,timestamp_ns,state,map_id,map_changed,track_time_s,coord_session,"
                       "tx,ty,tz,qx,qy,qz,qw\n")


def _world_poses(gt, ts, T_world):
    """Camera poses (cam0 == GT body for FakeDataset) expressed in another world frame."""
    return sample(gt, ts).transformed(T_left=T_world)


def _row(i, t, state, map_id, pose=None, session=None):
    s = f"{i},{t},{state},{map_id},0,0.010000" + ("" if session is None else f",{session}")
    if pose is None:
        return s + ",,,,,,,\n"
    p, q = pose
    return s + "," + ",".join(f"{v:.9f}" for v in [*p, *q]) + "\n"


def write_scenario(scenario, gt, frames, algo_dir):
    """Driver outputs for a scenario; returns the exit code the fake driver should produce."""
    algo_dir.mkdir(parents=True, exist_ok=True)
    W1, W2 = random_world(1), random_world(2)
    n = len(frames)
    rows, final, header = [], None, LOG_HEADER
    if scenario == "drift":
        # Online poses drift linearly (0.05 m/s along world x); the final export is exact.
        on = _world_poses(gt, frames, W1)
        t_rel = (frames - frames[0]) * 1e-9
        drift = np.c_[0.05 * t_rel, np.zeros(n), np.zeros(n)] @ W1[:3, :3].T
        for i, t in enumerate(frames):
            rows.append(_row(i, t, 2, 0, (on.positions[i] + drift[i], on.quaternions_xyzw[i])))
        final = (frames, _world_poses(gt, frames, W1))
    elif scenario == "map_reset":
        # Exact poses, but map 1 (second half) has an unrelated world frame; no tracking gap.
        h = n // 2
        a, b = _world_poses(gt, frames[:h], W1), _world_poses(gt, frames[h:], W2)
        for i, t in enumerate(frames):
            src, k = (a, i) if i < h else (b, i - h)
            rows.append(_row(i, t, 2, 0 if i < h else 1, (src.positions[k], src.quaternions_xyzw[k])))
        final = (frames[h:], b)
    elif scenario == "merge":
        # ORB-SLAM3 map merge: the map id stays 0 but the world frame changes at frame h; the merge
        # completed while frame h was tracked, so the driver marks that frame ambiguous (-1).
        header = LOG_HEADER_SESSIONS
        h = n // 2
        a, b = _world_poses(gt, frames[:h], W1), _world_poses(gt, frames[h:], W2)
        for i, t in enumerate(frames):
            src, k = (a, i) if i < h else (b, i - h)
            sess = 0 if i < h else (-1 if i == h else 1)
            rows.append(_row(i, t, 2, 0, (src.positions[k], src.quaternions_xyzw[k]), session=sess))
        final = (frames, _world_poses(gt, frames, W2))
    elif scenario == "crash":
        # 60 % of the frames tracked exactly, then a segfault mid-write: no final trajectory.
        k = int(0.6 * n)
        on = _world_poses(gt, frames[:k], W1)
        for i in range(k):
            rows.append(_row(i, frames[i], 2, 0, (on.positions[i], on.quaternions_xyzw[i])))
        rows.append(f"{k},{frames[k]}")  # line truncated by the crash
    with open(algo_dir / "frame_log.csv", "w") as f:
        f.write(header + "".join(rows))
    if scenario == "drift":
        write_map_and_features(algo_dir, frames, random_world(1))
    if final is not None:
        ts, tr = final
        with open(algo_dir / "trajectory_euroc.txt", "w") as f:
            for t, p, q in zip(ts, tr.positions, tr.quaternions_xyzw):
                f.write(f"{float(t):.6f} " + " ".join(f"{v:.9f}" for v in [*p, *q]) + "\n")
        (algo_dir / "driver_status.json").write_text(json.dumps({"status": "completed"}))
        return 0
    return -signal.SIGSEGV


MAP_POINTS_GT = np.random.default_rng(5).uniform(-6, 6, size=(40, 3))   # true positions (GT frame)


def write_map_and_features(algo_dir, frames, T_world):
    """Driver map/feature artifacts: map points in the final trajectory's world frame (T_world)."""
    import struct
    P = MAP_POINTS_GT @ T_world[:3, :3].T + T_world[:3, 3]
    with open(algo_dir / "map_points.csv", "w") as f:
        f.write("map_id,point_id,x,y,z,n_observations,in_final_frame\n")
        for i, p in enumerate(P):
            f.write(f"0,{i},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},3,1\n")
        f.write("1,999,100.0,100.0,100.0,2,0\n")           # another map: no common frame
    with open(algo_dir / "keyframes.csv", "w") as f:
        f.write("map_id,keyframe_id,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,in_final_frame\n")
        f.write(f"0,0,{frames[0]},{P[0][0]},{P[0][1]},{P[0][2]},0,0,0,1,1\n")
    with open(algo_dir / "features.bin", "wb") as f:
        f.write(b"SBFEAT01")
        for i, t in enumerate(frames[:20]):
            st = [1] * 5 + [2] * 2 + [0] * 3
            f.write(struct.pack("<qii", int(t), i, len(st)))
            for k, v in enumerate(st):
                f.write(struct.pack("<ffB", 10.0 + k, 20.0 + i, v))
        f.write(struct.pack("<qii", int(frames[20]), 20, 10) + b"\x00" * 7)   # truncated by a "crash"


class ScenarioOrbSlam3(OrbSlam3Adapter):
    """ORB-SLAM3 adapter whose 'driver' replays a scenario (real collect())."""
    name = "orbslam3_scenario"

    def validate(self, mode, dataset, settings):
        return AlgorithmAdapter.validate(self, mode, dataset, settings)

    def version_info(self):
        return {"fake": "scenario"}

    def prepare(self, ctx):
        frames = np.loadtxt(ctx.prepared.frames_csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
        self._code = write_scenario(self.options["scenario"], ctx.dataset.gt, frames, ctx.algo_dir)

    def command(self, ctx):
        if self._code < 0:
            return [sys.executable, "-c", f"import os; os.kill(os.getpid(), {-self._code})"]
        return [sys.executable, "-c", f"raise SystemExit({self._code})"]


@pytest.fixture(autouse=True)
def register(monkeypatch):
    monkeypatch.setitem(DATASETS, "fake", FakeDataset)
    monkeypatch.setitem(ALGORITHMS, ScenarioOrbSlam3.name, ScenarioOrbSlam3)


def run(tmp_path, scenario, **eval_over):
    ev = {"plots": True, "association_max_dt_s": 0.005, "alignment": "se3",
          "rpe": [{"delta": 1, "unit": "frames"}, {"delta": 1.0, "unit": "s"}]}
    ev.update(eval_over)
    cfg = {"name": scenario, "output_root": str(tmp_path / "runs"), "work_dir": str(tmp_path / "work"),
           "dataset": {"name": "fake", "root": str(tmp_path), "sequence": "s1"},
           "algorithm": {"name": ScenarioOrbSlam3.name, "mode": "stereo", "options": {"scenario": scenario}},
           "execution": {"timeout_s": 60, "memory_poll_s": 0.05}, "evaluation": ev}
    d = runner.run_experiment(validate_config(deep_merge(DEFAULTS, cfg)), log=lambda *_: None)
    st = json.loads((d / "status.json").read_text())
    m = json.loads((d / "eval" / "metrics.json").read_text()) if (d / "eval" / "metrics.json").exists() else None
    return d, st, m


def test_drifting_online_and_corrected_final_differ(tmp_path):
    d, st, m = run(tmp_path, "drift")
    assert st["status"] == "success" and st["eval_status"] == "ok" and st["stage"] == "done"
    on, fi = m["accuracy"]["online"], m["accuracy"]["final"]
    assert on["trajectory"]["kind"] == "online" and fi["trajectory"]["kind"] == "final"
    assert "aligned benchmark accuracy" in on["label"]
    assert fi["ate"]["translation_m"]["rmse"] < 1e-5
    assert on["ate"]["translation_m"]["rmse"] > 0.05
    assert on["association"]["n_matched"] == fi["association"]["n_matched"] == 300
    for f in ("trajectory_online.tum", "trajectory.tum", "eval/trajectory_online.png", "eval/trajectory.png"):
        assert (d / f).exists(), f
    rows = list(csv.DictReader(open(write_summary([d], tmp_path / "s.csv"))))
    r = rows[0]
    assert float(r["online_ate_trans_rmse_m"]) > 0.05 > float(r["final_ate_trans_rmse_m"])
    assert float(r["online_rpe_1s_trans_rmse_m"]) > float(r["final_rpe_1s_trans_rmse_m"])
    assert r["online_rpe_1frames_pairs"] == r["final_rpe_1frames_pairs"] == "299"


def test_map_reset_sessions_aligned_separately_and_rpe_not_across_maps(tmp_path):
    d, st, m = run(tmp_path, "map_reset")
    assert st["status"] == "success"
    on = m["accuracy"]["online"]
    assert on["n_sessions"] == on["n_sessions_pooled"] == 2
    assert [s["session"]["map_ids"] for s in on["sessions"]] == [[0], [1]]
    assert "map ids used instead" in on["session_note"]
    assert [s["session"]["n_poses"] for s in on["sessions"]] == [150, 150]
    assert on["sessions"][0]["session"]["span_fraction_of_input"] == pytest.approx(149 / 299)
    assert on["ate"]["translation_m"]["rmse"] < 1e-5
    # frame-to-frame RPE: 149 pairs per map, never the 1 pair across the map change
    assert on["rpe"][0]["n_pairs"] == 298 and on["rpe"][0]["pairs_within_one_map_session_only"]
    assert on["rpe"][0]["translation_m"]["rmse"] < 1e-5
    # Concatenating both maps under one alignment would be badly wrong.
    from slambench.trajectory import read_tum
    online = read_tum(d / "trajectory_online.tum")
    gt = FakeDataset(tmp_path, "s1").gt
    wrong = evaluate_sessions(online, gt, {"association_max_dt_s": 0.005}, "single")
    assert wrong["ate"]["translation_m"]["rmse"] > 1.0
    assert wrong["n_sessions"] == 1
    # final export holds only the largest map (one coordinate frame)
    assert m["accuracy"]["final"]["n_sessions"] == 1
    assert m["coverage"]["n_maps"] == 2


def test_crash_keeps_online_prefix_evaluation(tmp_path):
    d, st, m = run(tmp_path, "crash")
    assert st["status"] == "crash" and "signal 11" in st["message"]
    assert st["estimator_outcome"].startswith("terminated by signal 11")
    assert st["eval_status"] == "ok"
    out = json.loads((d / "algorithm_output.json").read_text())
    assert out["extra"]["n_frame_log_rows_malformed"] == 1 and out["extra"]["n_online_poses"] == 180
    assert [t["kind"] for t in out["trajectories"]] == ["online"]
    assert m["accuracy"]["final"]["available"] is False
    on = m["accuracy"]["online"]
    assert on["association"]["n_matched"] == 180
    assert on["ate"]["translation_m"]["rmse"] < 1e-5
    assert on["evaluated_coverage"]["evaluated_duration_fraction"] == pytest.approx(179 / 299)
    r = list(csv.DictReader(open(write_summary([d], tmp_path / "s.csv"))))[0]
    assert r["status"] == "crash" and r["online_ate_trans_rmse_m"] != "" and r["final_ate_trans_rmse_m"] == ""
    assert "no final trajectory" in r["final_unavailable_reason"]


def test_small_sessions_are_reported_but_not_pooled(tmp_path):
    d, st, m = run(tmp_path, "map_reset", session_min_matched=151)
    on = m["accuracy"]["online"]
    assert on["n_sessions"] == 2 and on["n_sessions_pooled"] == 0
    assert on["ate"]["available"] is False and on["n_poses_in_unpooled_sessions"] == 300
    assert "session_min_matched" in on["sessions"][0]["excluded_reason"]


def test_online_and_final_frames_differ_for_inertial_modes(tmp_path):
    ds = FakeDataset(tmp_path, "s1")
    (tmp_path / "prep").mkdir()
    prep = ds.prepare_offline(SensorMode.STEREO_INERTIAL, tmp_path / "prep")
    algo = OrbSlam3Adapter({"container": str(tmp_path / "none.sif")})
    ctx = RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.STEREO_INERTIAL, prep, algo.resolve_settings({}))
    write_scenario("drift", ds.gt, ds.frames, ctx.algo_dir)
    out = algo.collect(ctx, 0, False)
    frames = {t.kind: (t.frame, t.coordinate_sessions) for t in out.all_trajectories()}
    assert frames == {"online": ("cam0", "per_frame"), "final": ("imu", "single")}


def test_frame_log_rows_without_valid_pose_are_not_online_poses(tmp_path):
    p = tmp_path / "frame_log.csv"
    p.write_text(LOG_HEADER + _row(0, 10, 2, 0, ([0, 0, 0], [0, 0, 0, 1]))
                 + "1,20,2,0,0,0.01,1.2\n"                       # cut inside the pose
                 + _row(2, 30, 2, 0, ([0, 0, np.nan], [0, 0, 0, 1]))
                 + _row(3, 40, 2, 0, ([0, 0, 0], [0, 0, 0, 2]))  # not a unit quaternion
                 + _row(4, 50, 4, 0))
    rows, bad = OrbSlam3Adapter.read_frame_log(p)
    assert bad == 0 and [r["pose"] is not None for r in rows] == [True, False, False, False, False]


def test_frame_change_inside_one_map_id_starts_a_new_session(tmp_path):
    d, st, m = run(tmp_path, "merge")
    assert st["status"] == "success"
    on = m["accuracy"]["online"]
    assert on["trajectory"]["coordinate_sessions"] == "per_frame" and "session_note" not in on
    assert [(s["session"]["session_id"], s["session"]["map_ids"], s["session"]["n_poses"])
            for s in on["sessions"]] == [(0, [0], 150), (1, [0], 149)]
    assert on["n_poses_ambiguous_frame"] == 1
    assert on["ate"]["translation_m"]["rmse"] < 1e-5
    assert on["rpe"][0]["n_pairs"] == 149 + 148
    diag = (d / "frame_diagnostics.csv").read_text().splitlines()
    assert diag[0].endswith(",coordinate_session") and diag[151].endswith(",-1")
    # Splitting by map id alone (one id here) would mix both frames.
    from slambench.evaluation.metrics import FrameDiagnostics
    from slambench.trajectory import read_tum
    by_map = evaluate_sessions(read_tum(d / "trajectory_online.tum"), FakeDataset(tmp_path, "s1").gt,
                               {"association_max_dt_s": 0.005}, "map_id",
                               FrameDiagnostics.read(d / "frame_diagnostics.csv"))
    assert by_map["n_sessions"] == 1 and by_map["ate"]["translation_m"]["rmse"] > 1.0
