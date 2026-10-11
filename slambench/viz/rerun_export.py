"""Export a completed run directory to a Rerun recording (.rrd) for synchronized replay.

Panels: camera images, aligned 3D trajectories (ground truth, final, one path per online coordinate
session), tracking events, and time series (position error, tracking state, tracking time).

Positions are drawn exactly as the evaluator scored them: each coordinate session is aligned on its
own (``evaluate_sessions``), sessions are separate paths, and no path is drawn across a session
boundary or a tracking gap. The alignment uses the whole session's ground truth, so the 3D view is
*aligned benchmark accuracy*, not what the vehicle knew at that instant. If accuracy cannot be
evaluated (no ground truth, dataset unavailable), raw estimator poses are drawn per session
instead, each in its own unaligned frame, and the run panel says so.

Open the result with the matching viewer version: ``pip install rerun-sdk==<version>`` then
``rerun file.rrd``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

GT_COLOR = [150, 150, 150]
FINAL_COLOR = [220, 50, 47]
SESSION_COLORS = [[38, 139, 210], [133, 153, 0], [108, 113, 196], [181, 137, 0], [42, 161, 152],
                  [211, 54, 130], [203, 75, 22]]
BAD_STATES = {"lost", "recently_lost"}
PREFETCH = 64


def _session_color(k: int) -> List[int]:
    return SESSION_COLORS[k % len(SESSION_COLORS)]


def _load_run(run_dir: Path) -> Dict[str, Any]:
    from ..algorithms.base import trajectories_from_output_json
    from ..evaluation.metrics import FrameDiagnostics

    run = {"dir": run_dir, "cfg": yaml.safe_load((run_dir / "config.resolved.yaml").read_text()),
           "status": json.loads((run_dir / "status.json").read_text())}
    prep_json = run_dir / "prepared_input.json"
    if not prep_json.exists():
        raise FileNotFoundError(f"{prep_json} missing: the run never prepared its input "
                                f"(status {run['status'].get('status')}); nothing to replay")
    run["prep_dir"] = Path(json.loads(prep_json.read_text())["directory"])
    rows = []
    with open(run["prep_dir"] / "frames.csv") as f:
        next(f)
        for line in f:
            t, left, right = (line.rstrip("\n").split(",") + ["", ""])[:3]
            rows.append((int(t), left, right))
    run["frames"] = rows
    run["frame_ts"] = np.array([r[0] for r in rows], dtype=np.int64)
    out_path = run_dir / "algorithm_output.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    run["trajectories"] = {t.kind: t for t in trajectories_from_output_json(out)}
    diag_path = out.get("frame_diagnostics")
    run["diag"] = FrameDiagnostics.read(Path(diag_path)) if diag_path and Path(diag_path).exists() else None
    return run


def _aligned_accuracy(run: Dict[str, Any], log) -> Dict[str, Any]:
    """Re-run the evaluator in memory to get per-session aligned series (not stored on disk)."""
    from ..evaluation.evaluate import _evaluate_kind
    from ..evaluation.metrics import DEFAULT_EVAL
    from ..runner import make_dataset
    from ..types import SensorMode

    res: Dict[str, Any] = {"gt": {}, "acc": {}, "note": None}
    try:
        dataset = make_dataset(run["cfg"])
        gt = dataset.ground_truth()
    except Exception as exc:  # noqa: BLE001 - the viewer still shows raw poses
        res["note"] = f"dataset unavailable ({type(exc).__name__}: {exc}); raw unaligned poses shown"
        log(f"[view] {res['note']}")
        return res
    ecfg = dict(DEFAULT_EVAL)
    ecfg.update(run["cfg"].get("evaluation") or {})
    mode = SensorMode(run["cfg"]["algorithm"]["mode"])
    for kind, t in run["trajectories"].items():
        acc, gt_pos = _evaluate_kind(t, dataset, gt, ecfg, run["diag"], run["frame_ts"], mode)
        res["acc"][kind] = acc
        if gt_pos is not None:
            res["gt"][kind] = gt_pos
    return res


def _sessions_to_draw(run: Dict[str, Any], aligned: Dict[str, Any], kind: str):
    """[dict(label, t_ns, pos, gt_pos, err, segment, aligned)] for one trajectory kind."""
    acc = aligned["acc"].get(kind) or {}
    out = []
    for s in acc.get("sessions") or []:
        sv = s.get("_series")
        if sv is not None:
            out.append({"label": s["session"], "t_ns": sv["t_ns"], "pos": sv["est_pos"], "gt_pos": sv["gt_pos"],
                        "err": sv["trans_err"], "segment": sv["segment"], "aligned": True})
    if out or kind not in run["trajectories"]:
        return out
    # No aligned series (no GT, unknown extrinsic, too few matches): raw poses per session.
    from ..evaluation.metrics import split_sessions
    from ..trajectory import read_tum
    t = run["trajectories"][kind]
    if not t.path.exists():
        return out
    traj = read_tum(t.path).sorted_unique()
    split, _, _ = split_sessions(traj, t.coordinate_sessions, run["diag"])
    for label, idx in split:
        out.append({"label": label, "t_ns": traj.timestamps_ns[idx], "pos": traj.positions[idx],
                    "gt_pos": None, "err": None, "segment": None, "aligned": False})
    return out


def _encode(path: Path, quality: int, max_width: int) -> Optional[bytes]:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if max_width and img.shape[1] > max_width:
        h = int(round(img.shape[0] * max_width / img.shape[1]))
        img = cv2.resize(img, (max_width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes() if ok else None


def _info_markdown(run: Dict[str, Any], aligned: Dict[str, Any], drawn: Dict[str, list], stride: int,
                   n_missing: int) -> str:
    st, cfg = run["status"], run["cfg"]
    lines = [f"# {cfg.get('name', '')}",
             f"**{cfg['algorithm']['name']} {cfg['algorithm']['mode']}** on "
             f"{cfg['dataset']['name']}/{cfg['dataset']['sequence']}", "",
             f"* run `{st.get('run_id')}`: status **{st.get('status')}** (stage {st.get('stage')}) "
             f"{st.get('message', '')}",
             f"* estimator: {st.get('estimator_outcome', 'n/a')}; evaluation: {st.get('eval_status', 'n/a')}",
             f"* images: every {stride} frame(s), JPEG preview" + (f"; {n_missing} missing" if n_missing else ""),
             "", "## Trajectories",
             "Grey = ground truth, red = final, one colour per online coordinate session. Each session "
             "is aligned to ground truth on its own using the whole session (including later ground "
             "truth): **aligned benchmark accuracy**, not what the vehicle knew at that instant. Paths "
             "are never joined across a session boundary or a tracking gap.", ""]
    for kind in ("online", "final"):
        acc = aligned["acc"].get(kind)
        if kind not in run["trajectories"]:
            lines.append(f"* **{kind}**: not produced by this run")
            continue
        if drawn.get(kind) and not drawn[kind][0]["aligned"]:
            lines.append(f"* **{kind}**: raw estimator poses, NOT aligned (each session in its own frame)")
        if not acc or acc.get("available") is False or acc.get("ate", {}).get("available") is False:
            reason = (acc or {}).get("reason") or (acc or {}).get("ate", {}).get("reason", "no evaluation")
            lines.append(f"* **{kind}** accuracy unavailable: {reason}")
            continue
        a = acc["ate"]["translation_m"]
        lines.append(f"* **{kind}**: ATE RMSE {a['rmse']:.3f} m over {a['n']} samples, "
                     f"{acc.get('n_sessions_pooled', 1)}/{acc.get('n_sessions', 1)} sessions pooled "
                     f"({acc['alignment']['label']})")
        for s in acc.get("sessions", []):
            i = s["session"]
            e = (s.get("ate") or {}).get("translation_m") or {}
            lines.append(f"  * session {i['session_id']} (maps {i['map_ids']}): {i['n_poses']} poses from "
                         f"{(i['first_rel_s'] or 0):.1f} s for {i['span_s']:.1f} s; "
                         + (f"ATE {e['rmse']:.3f} m" if e.get("rmse") is not None else "not aligned")
                         + ("" if s.get("included_in_pooled") else " (not pooled)"))
    if aligned.get("note"):
        lines += ["", f"_{aligned['note']}_"]
    return "\n".join(lines)


def _blueprint(stereo: bool):
    import rerun.blueprint as rrb
    cam = (rrb.Horizontal(rrb.Spatial2DView(origin="/camera/left", name="left"),
                          rrb.Spatial2DView(origin="/camera/right", name="right")) if stereo
           else rrb.Spatial2DView(origin="/camera/left", name="camera"))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(cam, rrb.TextLogView(origin="/events", name="tracking events"),
                         rrb.TextDocumentView(origin="/info", name="run"), row_shares=[3, 2, 2]),
            rrb.Vertical(rrb.Spatial3DView(origin="/world", name="trajectories (aligned per session)"),
                         rrb.TimeSeriesView(origin="/plots/error", name="position error [m]"),
                         rrb.TimeSeriesView(origin="/plots/tracking", name="tracking"),
                         rrb.TimeSeriesView(origin="/plots/timing", name="tracking time [ms]"),
                         row_shares=[4, 2, 1, 1]),
            column_shares=[2, 3]))


def export_run(run_dir: Path, out_path: Optional[Path] = None, stride: int = 2, jpeg_quality: int = 75,
               max_width: int = 640, images: bool = True, log=print) -> Dict[str, Any]:
    """Write `out_path` (default <run_dir>/replay.rrd); returns a small report."""
    import rerun as rr

    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "replay.rrd"
    stride = max(1, int(stride))
    run = _load_run(run_dir)
    ts = run["frame_ts"]
    if len(ts) == 0:
        raise ValueError("prepared input has no frames")
    t0 = int(ts[0])

    def rel(t):
        return (np.asarray(t, dtype=np.int64) - t0) * 1e-9

    def frame_of(t):
        return np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)

    stereo = any(r[2] for r in run["frames"])
    aligned = _aligned_accuracy(run, log)

    rec = rr.RecordingStream("slambench", recording_id=str(run["status"].get("run_id") or run_dir.name))
    rec.save(str(out_path), default_blueprint=_blueprint(stereo))
    rec.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def series(path, t_ns, values, style):
        if len(t_ns) == 0:
            return
        rec.log(path, style, static=True)
        rec.send_columns(path, indexes=[rr.TimeColumn("time", duration=rel(t_ns)),
                                        rr.TimeColumn("frame", sequence=frame_of(t_ns))],
                         columns=rr.Scalars.columns(scalars=np.asarray(values, dtype=float)))

    def points(color, name):
        return rr.SeriesPoints(colors=[color], markers=["circle"], marker_sizes=[1.5], names=[name])

    # ---- 3D: static paths per session (split at tracking gaps), start markers, error series ----
    gt_any = aligned["gt"].get("online", aligned["gt"].get("final"))
    if gt_any is not None:
        rec.log("/world/ground_truth/path", rr.LineStrips3D([gt_any], colors=[GT_COLOR], radii=0.01), static=True)
    drawn = {kind: _sessions_to_draw(run, aligned, kind) for kind in ("final", "online")}
    current: Dict[str, Dict[int, Any]] = {}
    for kind, sessions in drawn.items():
        cur: Dict[int, Any] = {}
        for k, s in enumerate(sessions):
            color = FINAL_COLOR if kind == "final" else _session_color(k)
            sid = s["label"].get("session_id", k)
            base = f"/world/{kind}" + (f"/session_{sid}" if kind == "online" else "")
            pos, seg = s["pos"], s["segment"]
            strips = [pos[seg == g] for g in np.unique(seg)] if seg is not None else [pos]
            name = f"online session {sid} (maps {s['label'].get('map_ids')})" if kind == "online" else "final"
            if not s["aligned"]:
                name += " [raw, unaligned]"
            rec.log(f"{base}/path", rr.LineStrips3D([p for p in strips if len(p) > 1], colors=[color],
                                                     radii=0.012), static=True)
            rec.log(f"{base}/start", rr.Points3D([pos[0]], colors=[color], radii=0.06,
                                                 labels=[f"{name} starts @ {rel(s['t_ns'][0]):.1f} s"]),
                    static=True)
            for i, t in enumerate(s["t_ns"].tolist()):
                cur[t] = (pos[i], color, None if s["gt_pos"] is None else s["gt_pos"][i])
            if s["err"] is not None:
                path = f"/plots/error/online/session_{sid}" if kind == "online" else "/plots/error/final"
                series(path, s["t_ns"], s["err"], points(color, f"online s{sid}" if kind == "online" else "final"))
        current[kind] = cur

    # ---- tracking diagnostics and events ----
    diag = run["diag"]
    if diag is not None:
        series("/plots/tracking/tracking_ok", diag.timestamps_ns, diag.tracking_ok.astype(float),
               rr.SeriesLines(colors=[[42, 161, 152]], names=["tracking OK"]))
        cs = diag.coordinate_session
        if cs is not None and np.all(cs > -999):
            series("/plots/tracking/coordinate_session", diag.timestamps_ns, cs,
                   rr.SeriesLines(colors=[[108, 113, 196]], names=["coordinate session (-1 = ambiguous)"]))
        series("/plots/tracking/map_id", diag.timestamps_ns, diag.map_id,
               rr.SeriesLines(colors=[[181, 137, 0]], names=["map id"]))
        good = ~np.isnan(diag.track_time_s)
        series("/plots/timing/track_time_ms", diag.timestamps_ns[good], diag.track_time_s[good] * 1e3,
               rr.SeriesLines(colors=[[38, 139, 210]], names=["tracking time"]))
        if len(ts) > 1:
            period = float(np.median(np.diff(ts))) * 1e-6
            series("/plots/timing/frame_period_ms", ts[[0, -1]], [period, period],
                   rr.SeriesLines(colors=[[0, 0, 0]], names=["frame period"]))
        _log_events(rec, rr, diag, rel, frame_of)

    # ---- per-frame current-pose markers and images, in time order ----
    img_idx = list(range(0, len(run["frames"]), stride)) if images else []
    pool = ThreadPoolExecutor(8) if img_idx else None
    pending: Dict[int, Any] = {}

    def submit(n):
        for j in img_idx[n:n + 1]:
            fr = run["frames"][j]
            pending[j] = [("left", pool.submit(_encode, run["prep_dir"] / fr[1], jpeg_quality, max_width))]
            if fr[2]:
                pending[j].append(("right", pool.submit(_encode, run["prep_dir"] / fr[2], jpeg_quality,
                                                        max_width)))

    for n in range(min(PREFETCH, len(img_idx))):
        submit(n)
    next_n = min(PREFETCH, len(img_idx))
    n_img = n_missing = 0
    for i, (t, _, _) in enumerate(run["frames"]):
        rec.set_time("time", duration=float(rel(t)))
        rec.set_time("frame", sequence=i)
        gt_now = None
        for kind, cur in current.items():
            p = cur.get(t)
            if p is None:
                rec.log(f"/world/{kind}_now", rr.Clear(recursive=False))
            else:
                rec.log(f"/world/{kind}_now", rr.Points3D([p[0]], colors=[p[1]], radii=0.08))
                gt_now = p[2] if gt_now is None else gt_now
        if gt_now is not None:
            rec.log("/world/ground_truth_now", rr.Points3D([gt_now], colors=[GT_COLOR], radii=0.08))
        else:
            rec.log("/world/ground_truth_now", rr.Clear(recursive=False))
        if i in pending:
            for side, fut in pending.pop(i):
                data = fut.result()
                if data is None:
                    n_missing += 1
                    continue
                rec.log(f"/camera/{side}", rr.EncodedImage(contents=data, media_type="image/jpeg"))
                n_img += 1
            submit(next_n)
            next_n += 1
    if pool is not None:
        pool.shutdown()

    rec.log("/info", rr.TextDocument(_info_markdown(run, aligned, drawn, stride, n_missing),
                                     media_type=rr.MediaType.MARKDOWN), static=True)
    rec.flush()
    rec.disconnect()
    return {"out": str(out_path), "n_frames": int(len(ts)), "n_images": n_img, "n_images_missing": n_missing,
            "sessions": {k: [s["label"].get("session_id") for s in v] for k, v in drawn.items()},
            "aligned": {k: bool(v) and v[0]["aligned"] for k, v in drawn.items()},
            "size_mb": round(out_path.stat().st_size / 2 ** 20, 2), "rerun_version": rr.__version__,
            "note": aligned.get("note")}


def _log_events(rec, rr, diag, rel, frame_of) -> None:
    """Text events: tracking state changes and coordinate-session changes."""
    prev_state, prev_sess = None, None
    cs = diag.coordinate_session
    for i, t in enumerate(diag.timestamps_ns.tolist()):
        msgs = []  # one entity per event type: same-time logs to one entity would shadow each other
        state = str(diag.state[i])
        if state != prev_state:
            msgs.append(("/events/tracking", "WARN" if state in BAD_STATES else "INFO",
                         f"tracking: {prev_state or 'start'} -> {state}"))
            prev_state = state
        if cs is not None and cs[i] > -999 and cs[i] != prev_sess:
            if cs[i] == -1:
                msgs.append(("/events/session", "WARN",
                             "coordinate frame changed while this frame was tracked (pose excluded)"))
            else:
                msgs.append(("/events/session", "WARN" if prev_sess is not None else "INFO",
                             f"coordinate session {cs[i]} (map {diag.map_id[i]}) begins"))
            prev_sess = int(cs[i])
        if msgs:
            rec.set_time("time", duration=float(rel(t)))
            rec.set_time("frame", sequence=int(frame_of(t)))
            for path, lvl, msg in msgs:
                rec.log(path, rr.TextLog(msg, level=lvl))
