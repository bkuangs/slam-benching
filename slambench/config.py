"""Experiment configuration loading, merging and resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "name": None,
    "output_root": "~/scratch/slam-bench-work/runs",
    "work_dir": "~/scratch/slam-bench-work",
    "dataset": {"name": None, "root": None, "sequence": None, "options": {}},
    "algorithm": {"name": None, "mode": None, "options": {}, "settings": {}},
    "execution": {"timeout_s": 7200, "memory_poll_s": 0.5},
    "evaluation": {},
}


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"override must be key=value, got {it!r}")
        k, v = it.split("=", 1)
        set_dotted(out, k, yaml.safe_load(v))
    return out


def load_config(path: Path, overrides: Optional[Dict[str, Any]] = None,
                validate: bool = True) -> Dict[str, Any]:
    """Load one experiment. With validate=False, missing keys are left for run_experiment() to
    report as a config_rejected run (so the failure still produces a run directory)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "experiments" in raw:
        raise ValueError(f"{path} is a batch file; use `slambench batch`")
    cfg = deep_merge(DEFAULTS, raw)
    cfg = deep_merge(cfg, overrides or {})
    cfg["_source"] = str(Path(path).resolve())
    return validate_config(cfg) if validate else cfg


def load_batch(path: Path, overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    defaults = raw.get("defaults") or {}
    cfgs = []
    for item in raw.get("experiments") or []:
        if isinstance(item, str):
            p = (path.parent / item) if not Path(item).is_absolute() else Path(item)
            exp = yaml.safe_load(p.read_text()) or {}
            src = str(p.resolve())
        else:
            exp, src = item, f"{path.resolve()}#inline"
        cfg = deep_merge(deep_merge(DEFAULTS, defaults), exp)
        cfg = deep_merge(cfg, overrides or {})
        cfg["_source"] = src
        cfgs.append(cfg)
    return cfgs


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    missing = [k for k in ("dataset.name", "dataset.root", "dataset.sequence", "algorithm.name",
                           "algorithm.mode")
               if cfg[k.split(".")[0]].get(k.split(".")[1]) in (None, "")]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if not cfg.get("name"):
        cfg["name"] = (f"{cfg['dataset']['name']}_{cfg['dataset']['sequence']}_"
                       f"{cfg['algorithm']['name']}_{cfg['algorithm']['mode']}")
    return cfg


def expand(p: str) -> Path:
    return Path(p).expanduser()
