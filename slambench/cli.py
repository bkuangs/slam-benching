"""Command line interface: `slambench <command>` (or `python -m slambench`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _cmd_run(args) -> int:
    from .config import load_config, parse_overrides
    from .evaluation.summary import write_summary
    from .runner import run_experiment
    cfg = load_config(Path(args.config), parse_overrides(args.set), validate=False)
    run_dir = run_experiment(cfg, evaluate=not args.no_eval)
    st = json.loads((run_dir / "status.json").read_text())
    write_summary([run_dir], run_dir / "summary.csv")
    print(json.dumps({k: st.get(k) for k in ("run_id", "status", "message", "eval_status")}, indent=2))
    return 0 if st["status"] == "success" else 1


def _cmd_batch(args) -> int:
    import datetime as dt
    from .config import expand, load_batch, load_config, parse_overrides
    from .evaluation.summary import write_summary
    from .runner import run_experiment
    ov = parse_overrides(args.set)
    cfgs = []
    for p in args.configs:
        raw = yaml.safe_load(Path(p).read_text()) or {}
        cfgs += load_batch(Path(p), ov) if "experiments" in raw else [load_config(Path(p), ov, validate=False)]
    entries, failed = run_batch(cfgs, evaluate=not args.no_eval)
    out = Path(args.summary) if args.summary else (
        expand(cfgs[0].get("output_root") or "~/scratch/slam-bench-work/runs")
        / f"batch_{dt.datetime.now():%Y%m%d-%H%M%S}.csv")
    write_summary(entries, out)
    print(f"[slambench] {len(entries)} experiments, {failed} not successful; summary: {out}")
    return 0 if failed == 0 else 1


def run_batch(cfgs, evaluate: bool = True, log=print):
    """Run every config; one summary entry per config, even if a run could not be created."""
    from .runner import _safe_name, run_experiment
    entries, failed = [], 0
    for i, cfg in enumerate(cfgs):
        log(f"[slambench] batch {i + 1}/{len(cfgs)}: {_safe_name(cfg)}")
        try:
            d = run_experiment(cfg, evaluate=evaluate, log=log)
        except Exception as exc:  # noqa: BLE001 - e.g. run directory cannot be created
            log(f"[slambench] experiment could not start: {exc}")
            entries.append({"run_id": "", "experiment": _safe_name(cfg), "status": "harness_error",
                            "stage": "create_run_dir",
                            "message": f"run directory could not be created: {type(exc).__name__}: {exc}",
                            "config_source": cfg.get("_source", "")})
            failed += 1
            continue
        entries.append(d)
        failed += json.loads((d / "status.json").read_text()).get("status") != "success"
    return entries, failed


def _cmd_evaluate(args) -> int:
    from .evaluation import evaluate_run
    from .config import parse_overrides
    m = evaluate_run(Path(args.run_dir), eval_cfg=parse_overrides(args.set).get("evaluation"))
    acc = m.get("accuracy", {})
    print(json.dumps({**{f"{k}_ate": (a.get("ate") if isinstance(a, dict) and "ate" in a else a)
                         for k, a in acc.items()},
                      "coverage_source": m.get("coverage", {}).get("source")}, indent=2, default=str))
    return 0


def _cmd_view(args) -> int:
    try:
        from .viz.rerun_export import export_run
    except ImportError as exc:  # pragma: no cover
        print(f"viewer needs the rerun SDK: pip install -e '.[viz]' ({exc})")
        return 2
    rep = export_run(Path(args.run_dir), Path(args.output) if args.output else None, stride=args.stride,
                     jpeg_quality=args.jpeg_quality, max_width=args.max_width, images=not args.no_images)
    print(json.dumps(rep, indent=2, default=str))
    print(f"open locally with: pip install rerun-sdk=={rep['rerun_version']} && rerun {Path(rep['out']).name}")
    return 0


def _cmd_summarize(args) -> int:
    from .evaluation.summary import find_runs, write_summary
    runs = find_runs(Path(args.root))
    out = write_summary(runs, Path(args.output) if args.output else Path(args.root) / "summary.csv")
    print(f"{len(runs)} runs -> {out}")
    return 0


def _cmd_inspect(args) -> int:
    from .config import load_config, parse_overrides
    from .runner import make_algorithm, make_dataset
    from .types import SensorMode
    cfg = load_config(Path(args.config), parse_overrides(args.set))
    ds = make_dataset(cfg)
    algo = make_algorithm(cfg)
    info = ds.sequence_info()
    print(json.dumps(info.to_dict(), indent=2, default=str))
    st = ds.stereo_calibration()
    print("stereo calibration:", json.dumps(st.to_dict() if st else None, indent=2, default=str))
    gt = ds.ground_truth()
    if gt is not None:
        print(f"ground truth: {len(gt.timestamps_ns)} poses, body={gt.body_frame}, ref={gt.reference_frame}, "
              f"orientation={gt.has_orientation}, source={gt.source}")
    for m in SensorMode:
        settings = algo.resolve_settings(cfg["algorithm"].get("settings") or {})
        problems = ds.validate(m) + algo.validate(m, ds, settings)
        print(f"  {m.value:16s} {'OK' if not problems else 'REJECTED: ' + ' | '.join(problems)}")
    return 0


def _cmd_prepare(args) -> int:
    from .config import expand, load_config, parse_overrides
    from .runner import make_dataset
    from .types import SensorMode
    cfg = load_config(Path(args.config), parse_overrides(args.set))
    ds = make_dataset(cfg)
    mode = SensorMode(cfg["algorithm"]["mode"])
    problems = ds.validate(mode)
    if problems:
        print("REJECTED: " + " | ".join(problems))
        return 1
    p = ds.prepare_cached(mode, expand(cfg["work_dir"]) / "prepared")
    print(json.dumps({"directory": str(p.directory), "n_frames": p.n_frames, "n_imu": p.n_imu}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="slambench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    p = add("run", _cmd_run, "run one experiment config")
    p.add_argument("config")
    p.add_argument("--set", action="append", help="override, e.g. algorithm.settings.loop_closure=false")
    p.add_argument("--no-eval", action="store_true")
    p = add("batch", _cmd_batch, "run several configs or a batch file; writes a summary CSV")
    p.add_argument("configs", nargs="+")
    p.add_argument("--set", action="append")
    p.add_argument("--summary", help="output CSV path")
    p.add_argument("--no-eval", action="store_true")
    p = add("evaluate", _cmd_evaluate, "(re-)evaluate a run directory")
    p.add_argument("run_dir")
    p.add_argument("--set", action="append", help="e.g. evaluation.alignment=sim3")
    p = add("view", _cmd_view, "export a run directory to a Rerun replay file (.rrd)")
    p.add_argument("run_dir")
    p.add_argument("-o", "--output", help="default: <run_dir>/replay.rrd")
    p.add_argument("--stride", type=int, default=2, help="log every N-th image (poses/diagnostics: all frames)")
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--max-width", type=int, default=640, help="downscale wider images (0 = keep)")
    p.add_argument("--no-images", action="store_true")
    p = add("summarize", _cmd_summarize, "aggregate all runs under a directory into a CSV")
    p.add_argument("root")
    p.add_argument("-o", "--output")
    p = add("inspect", _cmd_inspect, "show dataset streams/calibration and per-mode compatibility")
    p.add_argument("config")
    p.add_argument("--set", action="append")
    p = add("prepare", _cmd_prepare, "only convert/prepare estimator input for a config")
    p.add_argument("config")
    p.add_argument("--set", action="append")
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
