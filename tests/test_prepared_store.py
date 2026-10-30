"""Prepared-input location: reuse of persistent caches, node-local preparation in Slurm jobs, and the
run-directory copy of frames.csv that evaluation reads once node-local inputs are gone."""

import json
import shutil

import pytest

from slambench.evaluation.evaluate import _input_timestamps
from slambench.prepared_store import (INPUT_FRAMES_COPY, keep_input_frames, node_local_root, prepare_input)
from slambench.types import SensorMode

from synthetic import FakeDataset

STEREO = SensorMode.STEREO


def cfg(tmp_path, choice="auto"):
    return {"work_dir": str(tmp_path / "work"), "prepared_dir": choice}


def job_env(tmp_path):
    return {"SLURM_JOB_ID": "123", "TMPDIR": str(tmp_path / "node")}


def test_auto_reuses_persistent_cache_else_prepares_node_local_in_jobs(tmp_path):
    ds = FakeDataset(tmp_path, "seq")
    prep, loc = prepare_input(ds, STEREO, cfg(tmp_path), env=job_env(tmp_path))
    assert loc == "node_local" and prep.directory.is_relative_to(tmp_path / "node")
    assert node_local_root(job_env(tmp_path)).name.endswith("-123")

    persistent, loc = prepare_input(ds, STEREO, cfg(tmp_path, "work_dir"), env=job_env(tmp_path))
    assert loc == "work_dir" and persistent.directory.is_relative_to(tmp_path / "work" / "prepared")
    again, loc = prepare_input(ds, STEREO, cfg(tmp_path), env=job_env(tmp_path))
    assert loc == "work_dir (existing cache)" and again.directory == persistent.directory


def test_auto_outside_slurm_uses_work_dir_and_node_local_requires_a_job(tmp_path):
    ds = FakeDataset(tmp_path, "seq")
    prep, loc = prepare_input(ds, STEREO, cfg(tmp_path), env={})
    assert loc == "work_dir" and prep.directory.is_relative_to(tmp_path / "work")
    with pytest.raises(ValueError):
        prepare_input(ds, STEREO, cfg(tmp_path, "node_local"), env={})


def test_input_timestamps_survive_removal_of_node_local_input(tmp_path):
    ds = FakeDataset(tmp_path, "seq")
    prep, _ = prepare_input(ds, STEREO, cfg(tmp_path), env=job_env(tmp_path))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    keep_input_frames(prep, run_dir)
    (run_dir / "prepared_input.json").write_text(json.dumps({"directory": str(prep.directory)}))
    expected = _input_timestamps(run_dir)
    shutil.rmtree(prep.directory)
    assert (run_dir / INPUT_FRAMES_COPY).exists()
    assert len(expected) == prep.n_frames and (_input_timestamps(run_dir) == expected).all()
