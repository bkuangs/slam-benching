#!/usr/bin/env python3
"""slam-benching offline driver for the Stanford NAV Lab Lunar Autonomy Challenge (LAC) SLAM stack.

Runs the LAC stereo visual odometry (SuperPoint + LightGlue features, PnP RANSAC) and the GTSAM
pose-graph backend with keyframe loop closures (lac.slam.semantic_feature_tracker /
lac.slam.backend at the pinned commit) on the slam-benching offline stereo format. The per-frame
loop mirrors lac.slam.frontend.Frontend.process_frame with these differences (see docs):

* no semantic segmentation (U-Net weights are not public): every keypoint gets the GROUND label;
  rock detection / rock mapping are therefore not run;
* no IMU: when PnP fails the odometry is the identity with the IMU noise model, which is exactly
  what Frontend does at this commit (its IMU estimate is overwritten with np.eye(4));
* keyframes are time based (every ``keyframe_interval_s`` of data, LAC: every 20 simulator steps =
  1 s) instead of step based;
* simulator constants in lac.params (intrinsics from a 70 deg FOV, 1280x720, 0.162 m baseline) are
  replaced by the dataset calibration before any other lac module is imported;
* optional stereo-match filters (settings "stereo"): a max depth other than LAC's, a minimum
  disparity and a maximum row difference between the left and right keypoint. With the defaults
  (all null) FeatureTracker.process_stereo behaves exactly like LAC's;
* optional keyframe-referenced VO (settings "vo": reference "keyframe"): PnP against a reference
  frame that is kept until the keypoints moved max_flow_px or lost too many matches, instead of
  LAC's frame-to-frame PnP ("previous", the default). Map points: as LAC, each frame's stereo points
  that matched no tracked point.

Usage: slambench_lac_driver.py SETTINGS_JSON PREPARED_DIR OUT_DIR

Outputs in OUT_DIR:
  frame_log.csv         per frame: state, VO matches, timings, online cam0 (OpenCV optical) pose
  trajectory_final.csv  backend poses after the run (cam0 optical frame), one per processed frame
  map_points.ply        backend.project_point_map(): new stereo points of every frame, world frame
  loop_closures.csv     accepted loop closures (keyframe pose index pairs, matches)
  driver_status.json    status, counts, GPU name and peak GPU memory, timing summary
"""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

#: LAC "camera" axes (x forward, y left, z up) from OpenCV optical axes (x right, y down, z forward):
#: p_camera = OPENCV_TO_CAMERA_ACTIVE @ p_opencv (lac.utils.geometry.OPENCV_TO_CAMERA_ACTIVE).
OPENCV_TO_CAMERA_ACTIVE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

STATE_INIT, STATE_VO, STATE_VO_FAILED, STATE_REINIT = 0, 1, 2, 3
GROUND_LABEL = 3  # lac.perception.segmentation.SemanticClasses.GROUND


def rover_to_opencv_pose(T_world_rover: np.ndarray) -> np.ndarray:
    """LAC rover pose -> pose of the left camera's OpenCV optical frame. The driver puts the
    FrontLeft camera at the rover origin with zero angles, so rover axes == LAC camera axes."""
    T_rover_cv = np.eye(4)
    T_rover_cv[:3, :3] = OPENCV_TO_CAMERA_ACTIVE
    return np.asarray(T_world_rover, dtype=float) @ T_rover_cv


def mat_to_tq(T: np.ndarray):
    """4x4 -> (t, q xyzw)."""
    from scipy.spatial.transform import Rotation
    R = np.asarray(T[:3, :3], dtype=float)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return np.asarray(T[:3, 3], dtype=float), Rotation.from_matrix(R).as_quat()


def stereo_filter(kp_left, kp_right, fb: float, max_depth: float, min_disparity_px=None,
                  max_row_diff_px=None):
    """LAC stereo depth (fx * baseline / disparity) and keep mask for matched keypoints (N x 2,
    numpy or torch). LAC keeps every match with depth <= max_depth; the optional filters also
    drop disparities <= min_disparity_px and |row difference| > max_row_diff_px."""
    disp = kp_left[:, 0] - kp_right[:, 0]
    depths = fb / (disp + 1e-8)
    keep = ~(depths > max_depth)
    if min_disparity_px is not None:
        keep = keep & (disp > min_disparity_px)
    if max_row_diff_px is not None:
        keep = keep & (abs(kp_left[:, 1] - kp_right[:, 1]) <= max_row_diff_px)
    return keep, depths


def install_stereo_filter(cfg: dict) -> dict:
    """Replace FeatureTracker.process_stereo (same signature and outputs) with one that applies
    stereo_filter; returns the effective settings. max_depth_m null keeps the caller's max_depth
    (LAC: 15 m in SemanticFeatureTracker, lac.params.MAX_DEPTH = 56.5 m for keyframes)."""
    import lac.slam.feature_tracker as F
    eff = {"max_depth_m": None if cfg.get("max_depth_m") is None else float(cfg["max_depth_m"]),
           "min_disparity_px": cfg.get("min_disparity_px"),
           "max_row_diff_px": cfg.get("max_row_diff_px")}

    def process_stereo(self, left_image, right_image, max_matches=None, min_score=None,
                       max_depth=F.MAX_DEPTH, return_matched_feats=False):
        feats_left = self.extract_feats(left_image)
        feats_right = self.extract_feats(right_image)
        matches = self.match_feats(feats_left, feats_right, max_matches, min_score)
        md = eff["max_depth_m"] if eff["max_depth_m"] is not None else max_depth
        keep, depths = stereo_filter(feats_left["keypoints"][0][matches[:, 0]],
                                     feats_right["keypoints"][0][matches[:, 1]],
                                     F.FL_X * F.STEREO_BASELINE, md,
                                     eff["min_disparity_px"], eff["max_row_diff_px"])
        matches, depths = matches[keep], depths[keep]
        if return_matched_feats:
            return (F.prune_features(feats_left, matches[:, 0]),
                    F.prune_features(feats_right, matches[:, 1]), matches, depths)
        return feats_left, feats_right, matches, depths

    F.FeatureTracker.process_stereo = process_stereo
    return eff


def reference_switch_due(flow_px: float, n_matches: int, n_ref: int, max_flow_px: float,
                         min_match_frac: float) -> bool:
    """Keyframe-referenced VO: start a new reference once the median keypoint motion since the
    reference reaches max_flow_px or fewer than min_match_frac of its points still match."""
    return not np.isfinite(flow_px) or flow_px >= max_flow_px or n_matches < min_match_frac * n_ref


class FramePoints:
    """TrackedPoints stand-in for Backend.update on frames that keep the old reference: the
    current frame's stereo points (current frame coordinates); as in update_tracks, points that
    matched a tracked point are not new (lengths 1), the others are (lengths 0)."""

    def __init__(self, points_local, labels, matched_idxs):
        self.points_local, self.labels = points_local, labels
        self.lengths = np.zeros(len(points_local), dtype=int)
        self.lengths[np.asarray(matched_idxs, dtype=int)] = 1


def track_against_reference(tracker, left, right, labels, max_flow_px, min_match_frac):
    """SemanticFeatureTracker.track_pnp, except that PnP is solved against the reference frame's
    points (not the previous frame's) and the tracks are only replaced when a new reference is due.
    Returns (T_ref_cur or None, switched, n_matches, flow_px, points for Backend.update)."""
    import lac.slam.semantic_feature_tracker as S
    feats_left, _, _, depths = tracker.process_stereo(left, right, max_matches=S.MAX_STEREO_MATCHES,
                                                      max_depth=S.MAX_DEPTH, return_matched_feats=True)
    tp = tracker.tracked_points
    mf = tracker.match_feats(tp.feats, feats_left).cpu().numpy()
    p2 = feats_left["keypoints"][0][mf[:, 1]].cpu().numpy()
    T = S.solve_vision_pnp(tp.points_local[mf[:, 0]], p2)
    flow = float(np.median(np.linalg.norm(p2 - tp.points[mf[:, 0]], axis=1))) if len(mf) else float("nan")
    switched = T is None or reference_switch_due(flow, len(mf), len(tp.ids), max_flow_px, min_match_frac)
    if switched:
        tracker.update_tracks(feats_left, depths, mf, labels)
        pts = tracker.tracked_points
    else:
        kp = feats_left["keypoints"][0].cpu().numpy()
        px = np.round(kp).astype(int)
        pts = FramePoints(tracker.project_stereo(np.eye(4), kp, depths, cam_name=tracker.cam),
                          labels[px[:, 1], px[:, 0]], mf[:, 1])
    return T, switched, int(len(mf)), flow, pts


def keyframe_due(t_ns: int, last_kf_ns, interval_s: float) -> bool:
    return last_kf_ns is None or (t_ns - last_kf_ns) * 1e-9 >= interval_s - 1e-6


def patch_lac_params(cam: dict) -> None:
    """Replace simulator constants with the dataset calibration. Must run before any other lac
    module is imported (they copy these names at import time)."""
    import lac.params as P
    K = np.array([[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]])
    P.IMG_WIDTH, P.IMG_HEIGHT = int(cam["width"]), int(cam["height"])
    P.FL_X, P.FL_Y = float(cam["fx"]), float(cam["fy"])
    P.STEREO_BASELINE = float(cam["baseline_m"])
    P.CAMERA_INTRINSICS = K
    P.IMG_FOV_RAD = float(2.0 * np.arctan(cam["width"] / (2.0 * cam["fx"])))
    fl = P.GEOMETRY_DICT["rover"]["cameras"]["FrontLeft"]
    for k in ("x", "y", "z", "roll", "pitch", "yaw"):
        fl[k] = 0.0
    fl["width"], fl["height"] = int(cam["width"]), int(cam["height"])

    import lac.perception.vision as V
    import lac.perception.depth as D

    def get_camera_intrinsics(cam_name, camera_config=None):  # noqa: ARG001 - single calibrated camera
        return K.copy()

    def calc_camera_intrinsics(w, h):  # noqa: ARG001
        return K.copy()

    V.get_camera_intrinsics = get_camera_intrinsics
    V.calc_camera_intrinsics = calc_camera_intrinsics
    D.get_camera_intrinsics = get_camera_intrinsics


def write_ply(path: Path, xyz: np.ndarray, labels: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<i4")])
    rec["x"], rec["y"], rec["z"], rec["label"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], labels
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\n"
                 "property float x\nproperty float y\nproperty float z\nproperty int label\n"
                 "end_header\n").encode())
        f.write(rec.tobytes())


def main(argv) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    settings = json.loads(Path(argv[1]).read_text())
    prepared, out = Path(argv[2]), Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    status = {"status": "starting", "message": ""}

    def write_status():
        (out / "driver_status.json").write_text(json.dumps(status, indent=2))

    write_status()
    cam = settings["camera"]
    patch_lac_params(cam)

    import cv2
    import torch
    from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
    from lac.slam.backend import Backend
    from gtsam.symbol_shorthand import X

    status["stereo_filter"] = install_stereo_filter(settings.get("stereo") or {})

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        status.update(status="error", message="CUDA is not available (LAC hard-codes .cuda())")
        write_status()
        return 3
    dev = torch.cuda.current_device()
    status["gpu"] = {"name": torch.cuda.get_device_name(dev),
                     "capability": list(torch.cuda.get_device_capability(dev)),
                     "torch": torch.__version__, "cuda": torch.version.cuda}

    with open(prepared / "frames.csv", newline="") as f:
        frames = [(int(r["timestamp_ns"]), prepared / r["left_path"], prepared / r["right_path"])
                  for r in csv.DictReader(f)]
    max_frames = settings.get("max_frames")
    if max_frames:
        frames = frames[: int(max_frames)]
    if not frames:
        status.update(status="error", message="no frames")
        write_status()
        return 4

    lc = settings["loop_closure"]
    lc_config = {k: lc[k] for k in ("min_score", "min_matches", "distance_threshold_m", "angle_threshold_deg")}
    if not lc.get("enabled", True):
        lc_config["distance_threshold_m"] = -1.0  # no keyframe is ever within a negative distance
    kf_interval = float(settings["keyframe_interval_s"])
    playback = float(settings.get("playback_rate", 0.0))
    vo = settings.get("vo") or {}
    vo_ref = vo.get("reference", "previous")
    if vo_ref not in ("previous", "keyframe"):
        status.update(status="error", message=f"unknown vo.reference {vo_ref!r}")
        write_status()
        return 2
    status["vo"] = vo

    def load(p):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"cannot read {p}")
        return img

    def sync():
        torch.cuda.synchronize(dev)

    tracker = SemanticFeatureTracker({"FrontLeft": {"width": cam["width"], "height": cam["height"]}})
    labels = np.full((int(cam["height"]), int(cam["width"])), GROUND_LABEL, dtype=np.int64)

    pose_ts = []           # backend pose index -> timestamp_ns
    log_f = open(out / "frame_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["frame_idx", "timestamp_ns", "state", "keyframe", "n_tracked", "n_new_points",
                  "n_loop_closures", "ref_switch", "load_time_s", "track_time_s", "backend_time_s", "frame_time_s",
                  "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
    n_state = {STATE_INIT: 0, STATE_VO: 0, STATE_VO_FAILED: 0, STATE_REINIT: 0}
    errors = []
    t_wall0 = time.perf_counter()
    backend = None
    last_kf_ns = None
    T_ref_prev = np.eye(4)  # previous frame's pose in the VO reference frame (vo_ref == "keyframe")
    n_ref_switches = 0
    status["status"] = "running"
    write_status()
    try:
        for i, (t_ns, lp, rp) in enumerate(frames):
            if playback > 0.0:
                target = (t_ns - frames[0][0]) * 1e-9 / playback
                elapsed = time.perf_counter() - t_wall0
                if target > elapsed:
                    time.sleep(target - elapsed)
            tl = time.perf_counter()
            left, right = load(lp), load(rp)
            t0 = time.perf_counter()
            n_new, switched = 0, True
            if backend is None:
                tracker.initialize(left, right, labels)
                sync()
                t1 = time.perf_counter()
                backend = Backend(np.eye(4), tracker, lc_config)
                # Backend.__init__ adds pose 0 but no map points; Frontend/nav_agent do the same.
                pose_ts.append(t_ns)
                state, kf = STATE_INIT, False
                t2 = time.perf_counter()
            else:
                state = STATE_VO
                try:
                    if vo_ref == "previous":
                        odometry = tracker.track_pnp(left, right, labels)
                    else:
                        T_ref_cur, switched, _, _, frame_pts = track_against_reference(
                            tracker, left, right, labels, float(vo["max_flow_px"]),
                            float(vo["min_match_frac"]))
                        odometry = None if T_ref_cur is None else np.linalg.inv(T_ref_prev) @ T_ref_cur
                        T_ref_prev = np.eye(4) if switched else T_ref_cur
                except Exception as exc:  # noqa: BLE001 - e.g. no features in a dark frame
                    errors.append(f"frame {i}: track_pnp {type(exc).__name__}: {exc}")
                    tracker.initialize(left, right, labels)
                    # initialize() sizes `lengths` to MAX_STEREO_MATCHES, not to the number of points
                    tp = tracker.tracked_points
                    tp.lengths = np.zeros(len(tp.ids), dtype=int)
                    odometry, state, switched, T_ref_prev = None, STATE_REINIT, True, np.eye(4)
                n_ref_switches += int(switched)
                sync()
                t1 = time.perf_counter()
                source = "VO"
                if odometry is None:
                    odometry, source = np.eye(4), "IMU"
                    if state == STATE_VO:
                        state = STATE_VO_FAILED
                kf = keyframe_due(t_ns, last_kf_ns, kf_interval)
                if kf:
                    last_kf_ns = t_ns
                tp = tracker.tracked_points if switched else frame_pts
                n_new = int((tp.lengths == 0).sum())
                backend.update({"step": i, "FrontLeft": left, "FrontRight": right,
                                "odometry": odometry, "odometry_source": source,
                                "tracked_points": tp, "keyframe": kf})
                sync()
                t2 = time.perf_counter()
                pose_ts.append(t_ns)
            n_state[state] += 1
            T = rover_to_opencv_pose(backend.values.atPose3(X(backend.pose_idx - 1)).matrix())
            t, q = mat_to_tq(T)
            tp = tracker.tracked_points
            log.writerow([i, t_ns, state, int(kf), int(len(tp.ids)), n_new, len(backend.loop_closures),
                          int(switched), f"{t0 - tl:.6f}", f"{t1 - t0:.6f}", f"{t2 - t1:.6f}", f"{t2 - t0:.6f}",
                          *(f"{v:.9f}" for v in t), *(f"{v:.9f}" for v in q)])
            if i % 500 == 0:
                log_f.flush()
                print(f"[lac] frame {i}/{len(frames)} state={state} lc={len(backend.loop_closures)} "
                      f"t={t2 - t0:.3f}s", flush=True)
        status["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        status.update(status="error", message=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        log_f.close()
    status["wall_time_s"] = time.perf_counter() - t_wall0

    if backend is not None:
        traj = backend.get_trajectory()
        with open(out / "trajectory_final.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
            for t_ns, Tr in zip(pose_ts, traj):
                t, q = mat_to_tq(rover_to_opencv_pose(Tr))
                w.writerow([t_ns, *(f"{v:.9f}" for v in t), *(f"{v:.9f}" for v in q)])
        try:
            if backend.point_map:
                cloud = backend.project_point_map()
                write_ply(out / "map_points.ply", cloud.points, cloud.labels)
                status["n_map_points"] = int(len(cloud.points))
        except Exception as exc:  # noqa: BLE001
            status["map_export_error"] = f"{type(exc).__name__}: {exc}"
        with open(out / "loop_closures.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["from_pose_idx", "to_pose_idx", "from_timestamp_ns", "to_timestamp_ns", "n_matches"])
            for (a, b), m in zip(backend.loop_closures, backend.loop_closures_matches):
                w.writerow([a, b, pose_ts[a], pose_ts[b], m])
        status["n_loop_closures"] = len(backend.loop_closures)
        status["n_keyframes"] = len(backend.keyframe_data)
        status["n_poses"] = len(traj)
    status["n_frames_input"] = len(frames)
    status["n_frames_by_state"] = {"init": n_state[STATE_INIT], "vo": n_state[STATE_VO],
                                   "vo_failed_identity": n_state[STATE_VO_FAILED],
                                   "reinitialized": n_state[STATE_REINIT]}
    status["n_errors"] = len(errors)
    status["n_vo_reference_switches"] = n_ref_switches
    status["errors_first"] = errors[:20]
    status["gpu"]["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(dev))
    status["gpu"]["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(dev))
    status["settings"] = settings
    write_status()
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
