"""Evaluator checks with known trajectory errors and missing tracking intervals."""

import numpy as np
import pytest

from slambench.evaluation.metrics import (FrameDiagnostics, associate, coverage_from_states,
                                          coverage_proxy_from_output, evaluate_trajectory)
from slambench.trajectory import Trajectory, read_tum, write_tum

from synthetic import T0, circle_gt, frame_times, random_world, sample, write_diag


def _alternating_z(est: Trajectory, mag: float) -> Trajectory:
    s = np.where(np.arange(len(est)) % 2 == 0, mag, -mag)
    return Trajectory(est.timestamps_ns, est.positions + np.c_[0 * s, 0 * s, s], est.quaternions_xyzw, est.frame)


def test_perfect_estimate_in_other_world_frame_has_zero_ate():
    gt = circle_gt()
    est = sample(gt, frame_times()).transformed(T_left=random_world())
    r = evaluate_trajectory(est, gt, {"alignment": "se3"})
    assert r["ate"]["translation_m"]["rmse"] < 1e-6
    assert r["ate"]["rotation_deg"]["rmse"] < 1e-4
    assert r["alignment"]["type"] == "se3" and r["alignment"]["scale"] == 1.0


def test_known_translation_error_recovered():
    gt = circle_gt()
    est = _alternating_z(sample(gt, frame_times()), 0.10).transformed(T_left=random_world(1))
    r = evaluate_trajectory(est, gt, {"alignment": "se3"})
    assert r["ate"]["translation_m"]["rmse"] == pytest.approx(0.10, abs=2e-3)
    r_none = evaluate_trajectory(est, gt, {"alignment": "none"})
    assert r_none["ate"]["translation_m"]["rmse"] > 1.0  # world frames differ without alignment


def test_sim3_is_explicit_and_labelled():
    gt = circle_gt()
    est = sample(gt, frame_times())
    est = Trajectory(est.timestamps_ns, 0.5 * est.positions, est.quaternions_xyzw, "cam0")
    se3 = evaluate_trajectory(est, gt, {"alignment": "se3"}, scale_ambiguous=True)
    sim3 = evaluate_trajectory(est, gt, {"alignment": "sim3"}, scale_ambiguous=True)
    assert se3["ate"]["translation_m"]["rmse"] > 1.0
    assert "warning" in se3["alignment"]
    assert sim3["ate"]["translation_m"]["rmse"] < 1e-6
    assert sim3["alignment"]["scale"] == pytest.approx(2.0, rel=1e-6)
    assert "Sim(3)" in sim3["alignment"]["label"]


def test_association_tolerance_rejects_far_samples():
    gt = circle_gt()
    est = sample(gt, frame_times())
    # +16 ms puts every sample >=16 ms away from all 30 Hz GT samples
    shifted = Trajectory(est.timestamps_ns + 16_000_000, est.positions, est.quaternions_xyzw)
    ei, gi, dt = associate(shifted.timestamps_ns, gt.timestamps_ns, 0.010)
    assert len(ei) == 0
    r = evaluate_trajectory(shifted, gt, {"association_max_dt_s": 0.010})
    assert r["ate"]["available"] is False


def test_lost_interval_excluded_and_reported():
    """Estimator outputs (bad) poses while lost; they must be excluded, and the loss reported."""
    gt = circle_gt()
    ts = frame_times()
    t_rel = (ts - T0) * 1e-9
    lost = (t_rel >= 10.0) & (t_rel < 15.0)
    est = sample(gt, ts)
    est.positions[lost] += 3.0  # garbage poses during the loss (e.g. extrapolation)
    est = est.transformed(T_left=random_world(2))
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "diag.csv"
        write_diag(p, ts, ~lost)
        diag = FrameDiagnostics.read(p)
    r = evaluate_trajectory(est, gt, {"rpe": [{"delta": 1.0, "unit": "s"}]}, diag, ts)
    assert r["n_output_poses_excluded_not_tracking_ok"] == int(lost.sum())
    assert r["ate"]["translation_m"]["rmse"] < 1e-6
    assert r["evaluated_coverage"]["n_segments"] == 2
    # evaluated duration excludes the 5 s gap (and the one frame period bridging it)
    assert r["evaluated_coverage"]["evaluated_duration_s"] == pytest.approx(60.0 - 5.0 - 2 / 15.0, abs=0.07)
    # no RPE pair spans the gap
    rp = r["rpe"][0]
    assert rp["translation_m"]["max"] < 1e-6
    cov = coverage_from_states(diag, ts)
    assert cov["n_tracking_loss_events"] == 1
    assert cov["loss_events"][0]["recovery_duration_s"] == pytest.approx(5.0, abs=1 / 15)
    assert cov["n_unrecovered"] == 0
    assert cov["frame_coverage"] == pytest.approx(1 - lost.mean())
    assert cov["time_coverage"] == pytest.approx(55.0 / 60.0, abs=0.01)


def test_missing_output_is_not_interpolated():
    gt = circle_gt()
    ts = frame_times()
    t_rel = (ts - T0) * 1e-9
    keep = ~((t_rel >= 20.0) & (t_rel < 30.0))
    est = sample(gt, ts[keep])
    r = evaluate_trajectory(est, gt, {"segment_max_gap_s": 0.5}, None, ts)
    assert r["association"]["n_matched"] == int(keep.sum())
    assert r["evaluated_coverage"]["n_segments"] == 2
    assert r["evaluated_coverage"]["evaluated_duration_fraction"] < 0.85
    assert r["evaluated_coverage"]["matched_frame_fraction"] == pytest.approx(keep.mean())


def test_unrecovered_loss_and_map_change():
    ts = frame_times(10.0)
    n = len(ts)
    ok = np.ones(n, bool)
    ok[30:40] = False          # recovered, in a new map
    ok[120:] = False           # never recovered
    maps = np.where(np.arange(n) < 40, 0, 1)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "diag.csv"
        write_diag(p, ts, ok, maps)
        diag = FrameDiagnostics.read(p)
    cov = coverage_from_states(diag, ts)
    assert cov["n_tracking_loss_events"] == 2
    assert cov["n_unrecovered"] == 1
    assert cov["n_recovered_in_new_map"] == 1
    assert cov["n_maps"] == 2


def test_position_only_ground_truth_marks_rotation_unavailable():
    gt = circle_gt()
    gt_pos_only = Trajectory(gt.timestamps_ns, gt.positions, None)
    est = sample(gt, frame_times())
    r = evaluate_trajectory(est, gt_pos_only, {"rpe": [{"delta": 1.0, "unit": "m"}]})
    assert r["ate"]["rotation_deg"]["available"] is False
    assert r["rpe"][0]["rotation_deg"]["available"] is False
    assert r["rpe"][0]["translation_m"]["rmse"] < 1e-6


def test_rpe_records_interval_and_units():
    gt = circle_gt()
    est = _alternating_z(sample(gt, frame_times()), 0.05)
    r = evaluate_trajectory(est, gt, {"rpe": [{"delta": 2.0, "unit": "m"}, {"delta": 5, "unit": "frames"}]})
    assert [(x["delta"], x["delta_unit"]) for x in r["rpe"]] == [(2.0, "m"), (5.0, "frames")]
    assert r["rpe"][0]["n_pairs"] > 0
    with pytest.raises(ValueError):
        evaluate_trajectory(est, gt, {"rpe": [{"delta": 1, "unit": "furlongs"}]})


def test_proxy_coverage_is_labelled():
    ts = frame_times()
    est = sample(circle_gt(), ts[::10])
    p = coverage_proxy_from_output(est, ts, keyframe_only=True)
    assert p["source"].startswith("PROXY")
    assert "keyframe-only" in p["warning"]
    assert p["n_tracking_loss_events"]["available"] is False


def test_tum_roundtrip_preserves_ns(tmp_path):
    gt = circle_gt(2.0)
    gt = Trajectory(gt.timestamps_ns + 123, gt.positions, gt.quaternions_xyzw, "cam0")
    write_tum(gt, tmp_path / "t.tum")
    back = read_tum(tmp_path / "t.tum")
    assert np.array_equal(back.timestamps_ns, gt.timestamps_ns)
    assert back.frame == "cam0"
    assert np.allclose(back.positions, gt.positions, atol=1e-9)
