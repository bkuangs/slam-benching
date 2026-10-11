"""Runner, configuration rejection and batch summary tests (fake estimator, no external deps)."""

import csv
import json

import numpy as np
import pytest
import yaml

from slambench import runner
from slambench.algorithms import ALGORITHMS
from slambench.algorithms.orbslam3 import OrbSlam3Adapter
from slambench.config import DEFAULTS, deep_merge, load_batch, validate_config
from slambench.datasets import DATASETS
from slambench.evaluation.summary import write_summary
from slambench.types import SensorMode

from synthetic import FakeAlgorithm, FakeDataset


@pytest.fixture(autouse=True)
def register_fakes(monkeypatch):
    monkeypatch.setitem(DATASETS, "fake", FakeDataset)
    monkeypatch.setitem(ALGORITHMS, "fake_algo", FakeAlgorithm)


def _cfg(tmp_path, **over):
    c = {"name": "t", "output_root": str(tmp_path / "runs"), "work_dir": str(tmp_path / "work"),
         "dataset": {"name": "fake", "root": str(tmp_path), "sequence": "s1"},
         "algorithm": {"name": "fake_algo", "mode": "stereo"},
         "execution": {"timeout_s": 60, "memory_poll_s": 0.05},
         "evaluation": {"plots": True, "association_max_dt_s": 0.02}}
    c = deep_merge(c, over)
    return validate_config(deep_merge(DEFAULTS, c))


def test_successful_run_records_metadata_and_metrics(tmp_path):
    d = runner.run_experiment(_cfg(tmp_path), log=lambda *_: None)
    st = json.loads((d / "status.json").read_text())
    assert st["status"] == "success" and st["eval_status"] == "ok"
    for f in ("config.resolved.yaml", "environment.json", "command.txt", "stdout.log", "stderr.log",
              "dataset_info.json", "prepared_input.json", "algorithm_output.json", "trajectory.tum",
              "eval/metrics.json", "eval/trajectory.png", "eval/error_and_tracking.png"):
        assert (d / f).exists(), f
    m = json.loads((d / "eval" / "metrics.json").read_text())
    assert m["accuracy"]["final"]["ate"]["translation_m"]["rmse"] == pytest.approx(0.05, abs=2e-3)
    assert m["coverage"]["source"] == "estimator_tracking_state"
    assert m["coverage"]["frame_coverage"] == 1.0
    env = json.loads((d / "environment.json").read_text())
    assert env["algorithm"] == {"fake": "1.0"} and "git" in env
    assert st["peak_rss_mb"] is not None and st["wall_time_s"] > 0


def test_incompatible_sensor_mode_rejected_clearly(tmp_path):
    d = runner.run_experiment(_cfg(tmp_path, algorithm={"mode": "stereo_inertial"}), log=lambda *_: None)
    st = json.loads((d / "status.json").read_text())
    assert st["status"] == "config_rejected"
    assert any("IMU" in p for p in st["problems"])
    assert any("does not support sensor mode 'stereo_inertial'" in p for p in st["problems"])
    assert not (d / "command.txt").exists()


def test_orbslam3_rejects_lidar_and_missing_container(tmp_path):
    ds = FakeDataset(tmp_path, "s1")
    a = OrbSlam3Adapter({"container": str(tmp_path / "nope.sif")})
    s = a.resolve_settings({})
    assert "does not support sensor mode 'lidar_inertial'" in a.validate(SensorMode.LIDAR_INERTIAL, ds, s)[0]
    assert any("container not found" in p for p in a.validate(SensorMode.STEREO, ds, s))
    with pytest.raises(ValueError):
        a.resolve_settings({"not_a_setting": 1})


def test_orbslam3_settings_file_for_rectified_stereo(tmp_path):
    from slambench.algorithms.base import RunContext
    from slambench.types import PreparedInput
    ds = FakeDataset(tmp_path, "s1")
    (tmp_path / "prep").mkdir()
    prepared = ds.prepare_offline(SensorMode.STEREO, tmp_path / "prep")
    a = OrbSlam3Adapter({"container": str(tmp_path / "x.sif")})
    ctx = RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.STEREO, prepared,
                     a.resolve_settings({"loop_closure": False}))
    a.prepare(ctx)
    text = (tmp_path / "algo" / "orbslam3_settings.yaml").read_text()
    assert text.startswith("%YAML:1.0")
    assert 'Camera.type: "Rectified"' in text and "Stereo.b: 0.12" in text
    assert "Camera.fps: 15" in text and "loopClosing: 0" in text
    cmd = a.command(ctx)
    assert cmd[-2:] == ["--playback-rate", "1.0"] and "--clahe" not in cmd


def test_crash_is_reported_and_batch_summary_lists_failures(tmp_path):
    ok = runner.run_experiment(_cfg(tmp_path), log=lambda *_: None)
    bad = runner.run_experiment(_cfg(tmp_path, name="crash", algorithm={"options": {"behaviour": "crash"}}),
                                log=lambda *_: None)
    rej = runner.run_experiment(_cfg(tmp_path, name="rej", algorithm={"mode": "mono"}), log=lambda *_: None)
    st = json.loads((bad / "status.json").read_text())
    assert st["status"] == "crash" and st["returncode"] != 0
    out = write_summary([ok, bad, rej], tmp_path / "summary.csv")
    rows = list(csv.DictReader(open(out)))
    assert [r["status"] for r in rows] == ["success", "crash", "config_rejected"]
    assert float(rows[0]["final_ate_trans_rmse_m"]) == pytest.approx(0.05, abs=2e-3)
    assert rows[1]["final_ate_trans_rmse_m"] == ""


def test_timeout_is_reported(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(FakeAlgorithm, "command",
                        lambda self, ctx: [sys.executable, "-c", "import time; time.sleep(30)"])
    d = runner.run_experiment(_cfg(tmp_path, execution={"timeout_s": 1}), log=lambda *_: None)
    st = json.loads((d / "status.json").read_text())
    assert st["timed_out"] is True and st["status"] == "timeout"


def test_batch_file_merges_defaults(tmp_path):
    exp = tmp_path / "e.yaml"
    exp.write_text(yaml.safe_dump({"dataset": {"name": "fake", "root": "/x", "sequence": "s"},
                                   "algorithm": {"name": "fake_algo", "mode": "stereo"}}))
    b = tmp_path / "b.yaml"
    b.write_text(yaml.safe_dump({"defaults": {"output_root": "/o"},
                                 "experiments": ["e.yaml", {"name": "inline", "dataset": {"name": "fake",
                                                 "root": "/y", "sequence": "s"},
                                                 "algorithm": {"name": "fake_algo", "mode": "mono"}}]}))
    cfgs = load_batch(b)
    assert [c["output_root"] for c in cfgs] == ["/o", "/o"]
    assert cfgs[1]["algorithm"]["mode"] == "mono"


def test_orbslam3_settings_file_for_fisheye_stereo(tmp_path):
    from slambench.algorithms.base import RunContext
    from slambench.types import CameraIntrinsics, StereoCalibration
    from slambench.trajectory import make_T
    ds = FakeDataset(tmp_path, "s1")
    kb = CameraIntrinsics("kannala_brandt", 512, 512, 190.0, 190.0, 255.0, 256.0, [0.1, 0.2, 0.3, 0.4], False)
    ds.stereo_calibration = lambda: StereoCalibration(kb, kb, make_T(np.eye(3), [0.1, 0, 0]), 0.1, "fake")
    (tmp_path / "prep").mkdir()
    prepared = ds.prepare_offline(SensorMode.STEREO, tmp_path / "prep")
    a = OrbSlam3Adapter({"container": str(tmp_path / "x.sif")})
    ctx = RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.STEREO, prepared, a.resolve_settings({}))
    a.prepare(ctx)
    text = (tmp_path / "algo" / "orbslam3_settings.yaml").read_text()
    for key in ('Camera.type: "KannalaBrandt8"', "Camera1.k4: 0.4", "Camera2.fx", "Stereo.T_c1_c2",
                "Camera1.overlappingBegin: 0", "Camera1.overlappingEnd: 511", "Camera2.overlappingEnd: 511"):
        assert key in text, key


def test_lidar_modes_rejected_until_adapter_supports_them(tmp_path):
    ds = FakeDataset(tmp_path, "s1")
    probs = ds.validate(SensorMode.LIDAR_INERTIAL)
    assert any("lacks streams ['lidar']" in p for p in probs)
    ds.sequence_info = lambda: type("I", (), {"streams": {"lidar": 1, "imu": 1}})()
    assert any("LiDAR modes deferred" in p for p in ds.validate(SensorMode.LIDAR_INERTIAL))
