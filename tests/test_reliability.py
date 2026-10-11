"""Experiment bookkeeping: every run leaves a status, a stage and (if needed) error details."""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from slambench import runner
from slambench.algorithms import ALGORITHMS
from slambench.algorithms.base import AlgorithmOutput
from slambench.algorithms.orbslam3 import OrbSlam3Adapter
from slambench.cli import run_batch
from slambench.config import DEFAULTS, deep_merge, validate_config
from slambench.datasets import DATASETS
from slambench.evaluation.summary import write_summary
from slambench.types import RunStatus

from synthetic import FakeAlgorithm, FakeDataset


@pytest.fixture(autouse=True)
def register_fakes(monkeypatch):
    monkeypatch.setitem(DATASETS, "fake", FakeDataset)
    monkeypatch.setitem(ALGORITHMS, "fake_algo", FakeAlgorithm)


def _cfg(tmp_path, validate=True, **over):
    c = {"name": "t", "output_root": str(tmp_path / "runs"), "work_dir": str(tmp_path / "work"),
         "dataset": {"name": "fake", "root": str(tmp_path), "sequence": "s1"},
         "algorithm": {"name": "fake_algo", "mode": "stereo"},
         "execution": {"timeout_s": 60, "memory_poll_s": 0.05},
         "evaluation": {"plots": False, "association_max_dt_s": 0.02}}
    c = deep_merge(deep_merge(DEFAULTS, c), over)
    return validate_config(c) if validate else c


def _status(d):
    return json.loads((Path(d) / "status.json").read_text())


def test_invalid_config_gets_run_dir_status_and_summary_row(tmp_path):
    bad = _cfg(tmp_path, validate=False, name=None, dataset={"name": "fake", "root": None})
    bad["_source"] = "/x/broken_experiment.yaml"
    entries, failed = run_batch([bad, _cfg(tmp_path)], log=lambda *_: None)
    assert failed == 1 and len(entries) == 2
    st = _status(entries[0])
    assert st["status"] == "config_rejected" and st["stage"] == "resolve"
    assert "dataset.root" in st["message"] and st["error"]["type"] == "ValueError"
    err = json.loads((entries[0] / "error.json").read_text())
    assert "Traceback" in err["traceback"]
    assert _status(entries[1])["status"] == "success"  # batch continued
    rows = list(csv.DictReader(open(write_summary(entries, tmp_path / "s.csv"))))
    assert [r["status"] for r in rows] == ["config_rejected", "success"]
    assert rows[0]["error_stage"] == "resolve" and rows[0]["config_source"] == "/x/broken_experiment.yaml"


def test_unknown_dataset_is_rejected_with_status(tmp_path):
    d = runner.run_experiment(_cfg(tmp_path, dataset={"name": "nope"}), log=lambda *_: None)
    st = _status(d)
    assert st["status"] == "config_rejected" and "unknown dataset" in st["message"]


def test_collection_error_is_distinguished_from_estimator_crash(tmp_path, monkeypatch):
    def broken_collect(self, ctx, rc, timed_out):
        raise ValueError("cannot parse estimator output")
    monkeypatch.setattr(FakeAlgorithm, "collect", broken_collect)
    d = runner.run_experiment(_cfg(tmp_path), log=lambda *_: None)
    st = _status(d)
    assert st["status"] == "collect_failed" and st["returncode"] == 0
    assert st["estimator_outcome"] == "exited with code 0" and "exited with code 0" in st["message"]
    assert st["error"]["stage"] == "collect" and st["stage"] == "done"
    assert st["eval_status"].startswith("skipped")
    row = list(csv.DictReader(open(write_summary([d], tmp_path / "s.csv"))))[0]
    assert row["status"] == "collect_failed" and row["error_stage"] == "collect"


def test_estimator_crash_is_not_a_harness_error(tmp_path):
    d = runner.run_experiment(_cfg(tmp_path, algorithm={"options": {"behaviour": "crash"}}), log=lambda *_: None)
    st = _status(d)
    assert st["status"] == "crash" and "error" not in st and st["stage"] == "done"


def test_missing_executable_is_a_harness_error(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeAlgorithm, "command", lambda self, ctx: [str(tmp_path / "no-such-binary")])
    d = runner.run_experiment(_cfg(tmp_path), log=lambda *_: None)
    st = _status(d)
    assert st["status"] == "harness_error" and st["error"]["stage"] == "execute"
    assert (d / "error.json").exists()


def test_unexpected_exception_still_writes_final_status(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment_info", lambda: (_ for _ in ()).throw(OSError("disk on fire")))
    d = runner.run_experiment(_cfg(tmp_path), log=lambda *_: None)
    st = _status(d)
    assert st["status"] == "harness_error" and st["error"]["stage"] == "environment"
    assert "disk on fire" in st["message"] and "harness_finished" in st


def test_status_is_written_before_and_during_execution(tmp_path, monkeypatch):
    code = "import json; s = json.load(open('status.json')); print(s['status'], s['stage'])"
    monkeypatch.setattr(FakeAlgorithm, "command", lambda self, ctx: [sys.executable, "-c", code])
    monkeypatch.setattr(FakeAlgorithm, "collect", lambda self, ctx, rc, to: (_ for _ in ()).throw(KeyError("x")))
    d = runner.run_experiment(_cfg(tmp_path), log=lambda *_: None)
    assert (d / "stdout.log").read_text().split() == ["running", "execute"]


def test_memory_monitor_attach_failure_keeps_estimator_result(tmp_path, monkeypatch):
    import psutil

    def gone(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil, "Process", gone)
    d = runner.run_experiment(_cfg(tmp_path), log=lambda *_: None)
    st = _status(d)
    assert st["status"] == "success" and st["returncode"] == 0
    assert st["peak_rss_mb"] is None and "could not attach memory monitor" in st["peak_rss_unavailable_reason"]


def test_memory_sampling_failure_keeps_estimator_result(tmp_path, monkeypatch):
    import psutil

    class Flaky:
        def __init__(self, pid):
            pass

        def children(self, recursive=False):
            raise RuntimeError("procfs unavailable")
    monkeypatch.setattr(psutil, "Process", Flaky)
    monkeypatch.setattr(FakeAlgorithm, "command",
                        lambda self, ctx: [sys.executable, "-c", "import time; time.sleep(0.3); open('poses.txt','w')"])
    monkeypatch.setattr(FakeAlgorithm, "collect",
                        lambda self, ctx, rc, to: AlgorithmOutput(RunStatus.MISSING_OUTPUT, f"rc={rc}"))
    d = runner.run_experiment(_cfg(tmp_path), log=lambda *_: None)
    st = _status(d)
    assert st["returncode"] == 0 and st["message"] == "rc=0" and st["peak_rss_mb"] is None
    assert "memory sampling stopped" in st["memory_monitor"]["errors"][0]


def test_execute_kills_estimator_if_harness_is_interrupted(tmp_path, monkeypatch):
    import psutil
    started = {}

    class Interrupting:
        def __init__(self, pid):
            started["pid"] = pid

        def children(self, recursive=False):
            raise KeyboardInterrupt
    monkeypatch.setattr(psutil, "Process", Interrupting)
    with pytest.raises(KeyboardInterrupt):
        runner.execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path,
                       tmp_path / "o", tmp_path / "e", 60, 0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(started["pid"], 0)


# ---- CLAHE inside the estimator image ----------------------------------------
def _sif_with_check():
    algo = OrbSlam3Adapter({})
    if not algo.container.exists() or not shutil.which(algo.apptainer):
        return None, None
    r = subprocess.run([algo.apptainer, "exec", "--cleanenv", str(algo.container), "test", "-x",
                        "/opt/slambench/bin/slambench_preprocess_check"], capture_output=True)
    return (algo, algo.container) if r.returncode == 0 else (None, None)


def test_clahe_changes_color_input_and_rejects_unsupported(tmp_path):
    import cv2
    algo, sif = _sif_with_check()
    if sif is None:
        pytest.skip("ORB-SLAM3 image with slambench_preprocess_check not available")
    rng = np.random.default_rng(0)
    base = (np.linspace(80, 140, 320)[None, :] * np.ones((240, 1))).astype(np.uint8)
    col = np.dstack([base, base // 2 + rng.integers(0, 20, base.shape).astype(np.uint8), base])
    cv2.imwrite(str(tmp_path / "c.png"), col)
    cv2.imwrite(str(tmp_path / "u16.png"), base.astype(np.uint16) * 200)

    def check(name, *flags):
        return subprocess.run([algo.apptainer, "exec", "--cleanenv", "-B", str(tmp_path), str(sif),
                               "/opt/slambench/bin/slambench_preprocess_check", str(tmp_path / name),
                               str(tmp_path / f"out_{name}"), *flags], capture_output=True, text=True)
    r = check("c.png", "--clahe")
    assert r.returncode == 0 and "color_to_gray=1 clahe=1" in r.stdout, r.stderr
    out = cv2.imread(str(tmp_path / "out_c.png"), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    assert out.ndim == 2 and np.abs(out.astype(int) - gray).mean() > 5
    assert np.array_equal(out, cv2.createCLAHE(3.0, (8, 8)).apply(gray))
    r = check("c.png")
    assert r.returncode == 0 and np.array_equal(cv2.imread(str(tmp_path / "out_c.png")), col)
    r = check("u16.png", "--clahe")
    assert r.returncode == 3 and "not 8-bit" in r.stderr
