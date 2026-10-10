"""Experiment runner: validate -> prepare -> execute -> collect -> evaluate."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import resource
import shlex
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .algorithms import ALGORITHMS
from .algorithms.base import RunContext
from .config import expand
from .datasets import DATASETS
from .types import RunStatus, SensorMode

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_dataset(cfg: Dict[str, Any]):
    d = cfg["dataset"]
    if d["name"] not in DATASETS:
        raise ValueError(f"unknown dataset {d['name']!r} (known: {sorted(DATASETS)})")
    return DATASETS[d["name"]](expand(d["root"]), d["sequence"], d.get("options") or {})


def make_algorithm(cfg: Dict[str, Any]):
    a = cfg["algorithm"]
    if a["name"] not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {a['name']!r} (known: {sorted(ALGORITHMS)})")
    return ALGORITHMS[a["name"]](a.get("options") or {})


def _git_info() -> Dict[str, Any]:
    def g(*args):
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = g("status", "--porcelain")
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(status), "status": status,
            "diff": g("diff", "HEAD") if status else None}


def environment_info() -> Dict[str, Any]:
    pkgs = {}
    for name in ("numpy", "scipy", "evo", "rosbags", "yaml", "cv2", "matplotlib", "psutil"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", getattr(mod, "version", "unknown"))
        except Exception as exc:  # noqa: BLE001
            pkgs[name] = f"unavailable: {exc}"
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    from . import __version__
    return {"slambench_version": __version__, "git": _git_info(), "python": sys.version,
            "packages": pkgs, "hostname": platform.node(), "platform": platform.platform(),
            "cpu_model": cpu, "n_cpus": os.cpu_count(),
            "slurm": {k: v for k, v in os.environ.items() if k in (
                "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
                "SLURM_JOB_PARTITION")}}


def execute(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path, timeout_s: float,
            poll_s: float) -> Dict[str, Any]:
    """Run `cmd`, sampling the RSS of the whole process tree; kill the group on timeout."""
    import psutil

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    started = dt.datetime.now().isoformat(timespec="seconds")
    peak, samples, timed_out = 0, 0, False
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, start_new_session=True)
        root = psutil.Process(proc.pid)
        while True:
            try:
                rc = proc.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            try:
                procs = [root] + root.children(recursive=True)
                rss = 0
                for p in procs:
                    try:
                        rss += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
                samples += 1
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() - t0 > timeout_s:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    rc = proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
                break
    wall = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"returncode": rc, "timed_out": timed_out, "wall_time_s": wall, "started": started,
            "finished": dt.datetime.now().isoformat(timespec="seconds"),
            "peak_rss_mb": peak / 2 ** 20 if samples else None,
            "peak_rss_method": f"sum of RSS over the process tree sampled every {poll_s}s ({samples} samples)",
            "max_single_process_rss_mb_rusage": after.ru_maxrss / 1024,
            "cpu_user_s": after.ru_utime - before.ru_utime,
            "cpu_system_s": after.ru_stime - before.ru_stime}


def _new_run_dir(cfg: Dict[str, Any]) -> Tuple[str, Path]:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{uuid.uuid4().hex[:6]}"
    d = expand(cfg["output_root"]) / cfg["name"] / run_id
    d.mkdir(parents=True, exist_ok=False)
    return run_id, d


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def run_experiment(cfg: Dict[str, Any], evaluate: bool = True, log=print) -> Path:
    """Run one fully-merged experiment config. Always returns a run directory with status.json."""
    run_id, run_dir = _new_run_dir(cfg)
    log(f"[slambench] run {run_id} -> {run_dir}")
    status: Dict[str, Any] = {"run_id": run_id, "status": None, "message": "",
                              "config_summary": {
                                  "dataset": cfg["dataset"]["name"], "sequence": cfg["dataset"]["sequence"],
                                  "algorithm": cfg["algorithm"]["name"], "mode": cfg["algorithm"]["mode"]}}

    def finish(st: RunStatus, msg: str = "", **extra) -> Path:
        status.update({"status": st.value, "message": msg, **extra})
        _write_json(run_dir / "status.json", status)
        log(f"[slambench] {run_id}: {st.value} {msg}")
        return run_dir

    # 1. resolve
    try:
        mode = SensorMode(cfg["algorithm"]["mode"])
        dataset = make_dataset(cfg)
        algo = make_algorithm(cfg)
        cfg["algorithm"]["settings"] = algo.resolve_settings(cfg["algorithm"].get("settings") or {})
        from .evaluation.metrics import DEFAULT_EVAL
        from .config import deep_merge
        cfg["evaluation"] = deep_merge(DEFAULT_EVAL, cfg.get("evaluation") or {})
    except Exception as exc:  # noqa: BLE001
        (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        return finish(RunStatus.CONFIG_REJECTED, f"invalid configuration: {exc}")
    s = cfg["algorithm"]["settings"]
    status["config_summary"].update({"loop_closure": s.get("loop_closure"),
                                     "playback_rate": s.get("playback_rate")})
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # 2. environment + versions
    env = environment_info()
    try:
        env["algorithm"] = algo.version_info()
    except Exception as exc:  # noqa: BLE001
        env["algorithm"] = {"error": str(exc)}
        _write_json(run_dir / "environment.json", env)
        return finish(RunStatus.CONFIG_REJECTED, f"estimator version check failed: {exc}")
    _write_json(run_dir / "environment.json", env)

    # 3. validation (sensor/calibration compatibility)
    problems = dataset.validate(mode) + algo.validate(mode, dataset, s)
    if problems:
        return finish(RunStatus.CONFIG_REJECTED, "incompatible configuration: " + " | ".join(problems),
                      problems=problems)

    # 4. dataset info + input preparation
    try:
        info: Dict[str, Any] = {"sequence": dataset.sequence_info().to_dict(),
                                "provenance": dataset.provenance()}
        st = dataset.stereo_calibration()
        info["stereo_calibration"] = st.to_dict() if st else None
        if mode.is_inertial:
            imu = dataset.imu_calibration()
            info["imu_calibration"] = imu.to_dict() if imu else None
        if hasattr(dataset, "zenodo_crosscheck"):
            info["calibration_crosscheck"] = dataset.zenodo_crosscheck()
        _write_json(run_dir / "dataset_info.json", info)
        log(f"[slambench] preparing {mode.value} input for {dataset.name}/{dataset.sequence} ...")
        t0 = time.monotonic()
        prepared = dataset.prepare_cached(mode, expand(cfg["work_dir"]) / "prepared")
        _write_json(run_dir / "prepared_input.json",
                    {"directory": str(prepared.directory), "n_frames": prepared.n_frames,
                     "n_imu": prepared.n_imu, "manifest": prepared.manifest,
                     "prepare_or_cache_time_s": time.monotonic() - t0})
    except Exception as exc:  # noqa: BLE001
        (run_dir / "prepare_error.txt").write_text(traceback.format_exc())
        return finish(RunStatus.PREPARE_FAILED, f"{type(exc).__name__}: {exc}")

    # 5. estimator configuration + command
    ctx = RunContext(run_dir, run_dir / "algorithm", dataset, mode, prepared, s)
    try:
        algo.prepare(ctx)
        cmd = algo.command(ctx)
    except Exception as exc:  # noqa: BLE001
        (run_dir / "prepare_error.txt").write_text(traceback.format_exc())
        return finish(RunStatus.PREPARE_FAILED, f"{type(exc).__name__}: {exc}")
    (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")

    # 6. execute
    ex = cfg["execution"]
    log(f"[slambench] executing: {shlex.join(cmd)}")
    res = execute(cmd, run_dir, run_dir / "stdout.log", run_dir / "stderr.log",
                  float(ex["timeout_s"]), float(ex["memory_poll_s"]))
    status.update(res)

    # 7. collect + normalize
    out = algo.collect(ctx, res["returncode"], res["timed_out"])
    _write_json(run_dir / "algorithm_output.json",
                {"status": out.status.value, "message": out.message,
                 "trajectory": str(out.trajectory) if out.trajectory else None,
                 "estimate_frame": out.estimate_frame,
                 "frame_diagnostics": str(out.frame_diagnostics) if out.frame_diagnostics else None,
                 "diagnostics_available": out.diagnostics_available, "keyframe_only": out.keyframe_only,
                 "extra": out.extra})
    finish(out.status, out.message)

    # 8. evaluate (also for failed runs that left diagnostics, so failures are characterised)
    if evaluate and cfg["evaluation"].get("enabled", True):
        try:
            from .evaluation import evaluate_run
            evaluate_run(run_dir, dataset)
            status["eval_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            (run_dir / "eval").mkdir(exist_ok=True)
            (run_dir / "eval" / "error.txt").write_text(traceback.format_exc())
            status["eval_status"] = f"{RunStatus.EVAL_FAILED.value}: {type(exc).__name__}: {exc}"
        _write_json(run_dir / "status.json", status)
        log(f"[slambench] evaluation: {status['eval_status']}")
    return run_dir
