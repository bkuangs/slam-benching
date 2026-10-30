"""Where prepared (converted) estimator inputs are cached.

`prepared_dir` (top-level config key):
* ``auto`` (default): reuse a complete cache under ``<work_dir>/prepared`` if one exists; otherwise,
  inside a Slurm job, prepare on the compute node's local disk (`node_local`), else under
  ``<work_dir>/prepared``.
* ``work_dir``: always ``<work_dir>/prepared`` (persistent, shared between jobs, counts against scratch).
* ``node_local``: ``$SLURM_TMPDIR`` / ``$TMPDIR`` / ``/tmp`` + ``slambench-prepared-<user>-<job id>``.
  Per job, removed when the slambench process exits; requires a Slurm job.
* any other string: a directory path.

Node-local inputs are gone after the job, so the runner copies ``frames.csv`` into the run directory
(``input_frames.csv``); evaluation and viewers read input timestamps from there.
"""

from __future__ import annotations

import atexit
import getpass
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import expand
from .types import PreparedInput, SensorMode

INPUT_FRAMES_COPY = "input_frames.csv"
_cleanup_registered = set()


def persistent_root(cfg: Dict[str, Any]) -> Path:
    return expand(cfg["work_dir"]) / "prepared"


def node_local_root(env=None) -> Optional[Path]:
    """Per-job directory on the compute node's local disk, or None outside a Slurm job."""
    env = os.environ if env is None else env
    job = env.get("SLURM_JOB_ID")
    if not job:
        return None
    base = Path(env.get("SLURM_TMPDIR") or env.get("TMPDIR") or "/tmp")
    return base / f"slambench-prepared-{getpass.getuser()}-{job}"


def _remove_at_exit(root: Path) -> None:
    if root not in _cleanup_registered:
        _cleanup_registered.add(root)
        atexit.register(shutil.rmtree, root, True)


def prepare_input(dataset, mode: SensorMode, cfg: Dict[str, Any], env=None) -> Tuple[PreparedInput, str]:
    """(prepared input, location label) according to cfg['prepared_dir']."""
    choice = cfg.get("prepared_dir") or "auto"
    pers = persistent_root(cfg)
    if choice == "work_dir":
        return dataset.prepare_cached(mode, pers), "work_dir"
    if choice in ("auto", "node_local"):
        if choice == "auto":
            hit = dataset.cached_preparation(mode, pers)
            if hit is not None:
                return hit, "work_dir (existing cache)"
        root = node_local_root(env)
        if root is None:
            if choice == "node_local":
                raise ValueError("prepared_dir: node_local needs a Slurm job (SLURM_JOB_ID is not set)")
            return dataset.prepare_cached(mode, pers), "work_dir"
        _remove_at_exit(root)
        return dataset.prepare_cached(mode, root), "node_local"
    return dataset.prepare_cached(mode, expand(choice)), "path"


def keep_input_frames(prepared: PreparedInput, run_dir: Path) -> None:
    shutil.copyfile(Path(prepared.directory) / "frames.csv", Path(run_dir) / INPUT_FRAMES_COPY)


def input_frames_csv(run_dir: Path, prepared_directory: str) -> Path:
    """The run's copy of frames.csv if present (node-local inputs), else the prepared one."""
    copy = Path(run_dir) / INPUT_FRAMES_COPY
    return copy if copy.exists() else Path(prepared_directory) / "frames.csv"
