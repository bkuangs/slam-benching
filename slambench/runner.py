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


def _kill_group(proc: subprocess.Popen) -> Optional[int]:
    """SIGTERM, then SIGKILL, the estimator's process group; returns its exit code."""
    for sig, wait_s in ((signal.SIGTERM, 20), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            return proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            continue
    return proc.poll()


def execute(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path, timeout_s: float,
            poll_s: float) -> Dict[str, Any]:
    """Run `cmd`, sampling the RSS of the whole process tree; kill the group on timeout.

    Memory monitoring is best effort: if it cannot attach (e.g. the process already exited) or
    fails while sampling, the estimator keeps running, its exit code is still collected and the
    memory figures are reported as unavailable with the reason.
    """
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    started = dt.datetime.now().isoformat(timespec="seconds")
    peak, samples, timed_out = 0, 0, False
    monitor_errors: List[str] = []
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - psutil is a dependency
        psutil, root = None, None
        monitor_errors.append(f"psutil unavailable: {exc}")
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, start_new_session=True)
        try:
            root = None
            if psutil is not None:
                try:
                    root = psutil.Process(proc.pid)
                except Exception as exc:  # noqa: BLE001 - e.g. NoSuchProcess for an instant exit
                    monitor_errors.append(f"could not attach memory monitor: {type(exc).__name__}: {exc}")
            while True:
                try:
                    rc = proc.wait(timeout=poll_s)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if root is not None:
                    try:
                        rss = 0
                        for p in [root] + root.children(recursive=True):
                            try:
                                rss += p.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                pass
                        peak = max(peak, rss)
                        samples += 1
                    except psutil.NoSuchProcess:
                        pass
                    except Exception as exc:  # noqa: BLE001 - stop monitoring, keep the run
                        monitor_errors.append(f"memory sampling stopped: {type(exc).__name__}: {exc}")
                        root = None
                if time.monotonic() - t0 > timeout_s:
                    timed_out = True
                    rc = _kill_group(proc)
                    break
        except BaseException:
            _kill_group(proc)  # never leave the estimator running when the harness fails
            raise
    wall = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    res = {"returncode": rc, "timed_out": timed_out, "wall_time_s": wall, "started": started,
           "finished": dt.datetime.now().isoformat(timespec="seconds"),
           "peak_rss_mb": peak / 2 ** 20 if samples else None,
           "peak_rss_method": f"sum of RSS over the process tree sampled every {poll_s}s ({samples} samples)",
           "max_single_process_rss_mb_rusage": after.ru_maxrss / 1024,
           "cpu_user_s": after.ru_utime - before.ru_utime,
           "cpu_system_s": after.ru_stime - before.ru_stime,
           "memory_monitor": {"samples": samples, "errors": monitor_errors}}
    if not samples:
        res["peak_rss_unavailable_reason"] = (
            "; ".join(monitor_errors) or "estimator exited before the first memory sample")
    return res


def estimator_outcome(returncode: Optional[int], timed_out: bool) -> str:
    if timed_out:
        return "timed out (killed by slambench)"
    if returncode is None:
        return "exit code unknown"
    if returncode < 0:
        try:
            return f"terminated by signal {-returncode} ({signal.Signals(-returncode).name})"
        except ValueError:
            return f"terminated by signal {-returncode}"
    return f"exited with code {returncode}"


def _safe_name(cfg: Dict[str, Any]) -> str:
    if cfg.get("name"):
        return str(cfg["name"])
    d, a = cfg.get("dataset") or {}, cfg.get("algorithm") or {}
    parts = [d.get("name"), d.get("sequence"), a.get("name"), a.get("mode")]
    if all(parts):
        return "_".join(str(p) for p in parts)
    return Path(str(cfg.get("_source") or "unnamed")).stem.split("#")[0] or "unnamed"


def _new_run_dir(cfg: Dict[str, Any]) -> Tuple[str, Path]:
    from .config import DEFAULTS
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{uuid.uuid4().hex[:6]}"
    d = expand(cfg.get("output_root") or DEFAULTS["output_root"]) / _safe_name(cfg) / run_id
    d.mkdir(parents=True, exist_ok=False)
    return run_id, d


def _write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (a reader never sees a half-written status file)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _dump_config(run_dir: Path, cfg: Dict[str, Any]) -> None:
    try:
        (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:  # noqa: BLE001 - not YAML-serialisable; keep a readable copy
        (run_dir / "config.resolved.yaml").write_text(json.dumps(cfg, indent=2, default=str))


#: Pipeline stages recorded in status.json["stage"] (in order).
STAGES = ("resolve", "environment", "validate", "dataset_info", "prepare_input", "algorithm_prepare",
          "execute", "collect", "evaluate", "done")


def run_experiment(cfg: Dict[str, Any], evaluate: bool = True, log=print) -> Path:
    """Run one experiment config. Always returns a run directory whose status.json is final.

    status.json is written as soon as the run directory exists (status "running") and updated at
    every stage, so even a run killed externally shows how far it got. Every failure path writes a
    terminal status; unexpected exceptions also write error.json (stage, type, message, traceback).
    """
    run_id, run_dir = _new_run_dir(cfg)
    log(f"[slambench] run {run_id} -> {run_dir}")
    d, a = cfg.get("dataset") or {}, cfg.get("algorithm") or {}
    status: Dict[str, Any] = {
        "run_id": run_id, "status": "running", "stage": None, "message": "",
        "harness_started": dt.datetime.now().isoformat(timespec="seconds"),
        "config_summary": {"experiment": _safe_name(cfg), "dataset": d.get("name"), "sequence": d.get("sequence"),
                           "algorithm": a.get("name"), "mode": a.get("mode"), "source": cfg.get("_source")}}

    def stage(name: str) -> None:
        status["stage"] = name
        _write_json(run_dir / "status.json", status)

    def finish(st: RunStatus, msg: str = "", exc: Optional[BaseException] = None, **extra) -> Path:
        status.update({"status": st.value, "message": msg, **extra})
        if exc is not None:
            err = {"stage": status["stage"], "type": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc()}
            _write_json(run_dir / "error.json", err)
            status["error"] = {k: err[k] for k in ("stage", "type", "message")}
        status["harness_finished"] = dt.datetime.now().isoformat(timespec="seconds")
        _write_json(run_dir / "status.json", status)
        log(f"[slambench] {run_id}: {st.value} {msg}")
        return run_dir

    stage("resolve")
    try:
        return _run_stages(cfg, evaluate, log, run_dir, status, stage, finish)
    except Exception as exc:  # noqa: BLE001 - last resort: any unexpected harness failure
        return finish(RunStatus.HARNESS_ERROR, f"unexpected error in stage {status['stage']!r}: "
                                               f"{type(exc).__name__}: {exc}", exc=exc)


def _run_stages(cfg, evaluate, log, run_dir, status, stage, finish) -> Path:
    from .config import deep_merge, validate_config
    from .evaluation.metrics import DEFAULT_EVAL

    # 1. resolve
    try:
        cfg = validate_config(cfg)
        mode = SensorMode(cfg["algorithm"]["mode"])
        dataset = make_dataset(cfg)
        algo = make_algorithm(cfg)
        cfg["algorithm"]["settings"] = algo.resolve_settings(cfg["algorithm"].get("settings") or {})
        cfg["evaluation"] = deep_merge(DEFAULT_EVAL, cfg.get("evaluation") or {})
    except Exception as exc:  # noqa: BLE001
        _dump_config(run_dir, cfg)
        return finish(RunStatus.CONFIG_REJECTED, f"invalid configuration: {type(exc).__name__}: {exc}", exc=exc)
    s = cfg["algorithm"]["settings"]
    status["config_summary"].update({"loop_closure": s.get("loop_closure"), "clahe": s.get("clahe"),
                                     "playback_rate": s.get("playback_rate")})
    _dump_config(run_dir, cfg)

    # 2. environment + versions
    stage("environment")
    env = environment_info()
    try:
        env["algorithm"] = algo.version_info()
    except Exception as exc:  # noqa: BLE001
        env["algorithm"] = {"error": str(exc)}
        _write_json(run_dir / "environment.json", env)
        return finish(RunStatus.CONFIG_REJECTED, f"estimator version check failed: {exc}", exc=exc)
    _write_json(run_dir / "environment.json", env)

    # 3. validation (sensor/calibration compatibility)
    stage("validate")
    try:
        problems = dataset.validate(mode) + algo.validate(mode, dataset, s)
    except Exception as exc:  # noqa: BLE001 - e.g. unreadable calibration
        return finish(RunStatus.CONFIG_REJECTED, f"validation failed: {type(exc).__name__}: {exc}", exc=exc)
    if problems:
        return finish(RunStatus.CONFIG_REJECTED, "incompatible configuration: " + " | ".join(problems),
                      problems=problems)

    # 4. dataset info + input preparation
    try:
        stage("dataset_info")
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
        stage("prepare_input")
        log(f"[slambench] preparing {mode.value} input for {dataset.name}/{dataset.sequence} ...")
        t0 = time.monotonic()
        prepared = dataset.prepare_cached(mode, expand(cfg["work_dir"]) / "prepared")
        _write_json(run_dir / "prepared_input.json",
                    {"directory": str(prepared.directory), "n_frames": prepared.n_frames,
                     "n_imu": prepared.n_imu, "manifest": prepared.manifest,
                     "prepare_or_cache_time_s": time.monotonic() - t0})
    except Exception as exc:  # noqa: BLE001
        (run_dir / "prepare_error.txt").write_text(traceback.format_exc())
        return finish(RunStatus.PREPARE_FAILED, f"{type(exc).__name__}: {exc}", exc=exc)

    # 5. estimator configuration + command
    stage("algorithm_prepare")
    ctx = RunContext(run_dir, run_dir / "algorithm", dataset, mode, prepared, s)
    try:
        algo.prepare(ctx)
        cmd = algo.command(ctx)
    except Exception as exc:  # noqa: BLE001
        (run_dir / "prepare_error.txt").write_text(traceback.format_exc())
        return finish(RunStatus.PREPARE_FAILED, f"{type(exc).__name__}: {exc}", exc=exc)
    (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")

    # 6. execute
    stage("execute")
    ex = cfg["execution"]
    log(f"[slambench] executing: {shlex.join(cmd)}")
    try:
        res = execute(cmd, run_dir, run_dir / "stdout.log", run_dir / "stderr.log",
                      float(ex["timeout_s"]), float(ex["memory_poll_s"]))
    except Exception as exc:  # noqa: BLE001 - e.g. executable not found: estimator never ran
        return finish(RunStatus.HARNESS_ERROR, f"could not execute estimator: {type(exc).__name__}: {exc}",
                      exc=exc)
    status.update(res)
    status["estimator_outcome"] = estimator_outcome(res["returncode"], res["timed_out"])

    # 7. collect + normalize (a parsing failure is the harness's, not the estimator's)
    stage("collect")
    try:
        out = algo.collect(ctx, res["returncode"], res["timed_out"])
        _write_json(run_dir / "algorithm_output.json", out.to_dict())
    except Exception as exc:  # noqa: BLE001
        finish(RunStatus.COLLECT_FAILED,
               f"could not collect estimator outputs: {type(exc).__name__}: {exc} "
               f"(estimator {status['estimator_outcome']})", exc=exc,
               eval_status="skipped: outputs could not be collected")
        stage("done")
        return run_dir
    finish(out.status, out.message)

    # 8. evaluate (also for failed runs that left diagnostics or online poses)
    if evaluate and cfg["evaluation"].get("enabled", True):
        stage("evaluate")
        try:
            from .evaluation import evaluate_run
            evaluate_run(run_dir, dataset)
            status["eval_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            (run_dir / "eval").mkdir(exist_ok=True)
            (run_dir / "eval" / "error.txt").write_text(traceback.format_exc())
            status["eval_status"] = f"{RunStatus.EVAL_FAILED.value}: {type(exc).__name__}: {exc}"
        log(f"[slambench] evaluation: {status['eval_status']}")
    else:
        status["eval_status"] = "disabled"
    stage("done")
    return run_dir
