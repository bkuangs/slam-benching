#!/usr/bin/env python3
"""Write VPR benchmark sessions (vpr/sessions.py format) for the OmniLRS aerial renders and LunaPolaris.

Sim: poses from gt_cam0.tum by row index (the ~1.8e9 s TUM stamps cannot resolve ns, all four lighting
renders share the trajectory frame-for-frame). Real: mocap GT (same topic/body->cam0 transform as the
slambench evaluation) expressed in the rectified left camera frame, nearest mocap sample within 20 ms of
each image stamp (association_max_dt_s of the LunaPolaris configs); images from the slambench prepared
stereo input (prepared if missing; seq11 with right_row_shift_px -1.63).

Run with the slambench venv (needs rosbags, no torch):
    python scripts/vpr_prepare_sessions.py --out /storage/ice1/3/6/bkuang7/slam-bench-work/vpr/sessions
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vpr.sessions import read_frames_csv, write_session  # noqa: E402

SIM_CONDITIONS = ["nominal", "low_sun", "dim", "glare"]
REAL = {"seq01": ("sequence_01", 0.0), "seq02": ("sequence_02", 0.0), "seq11": ("sequence_11", -1.63)}
GT_TOPIC = "/vrpn_mocap/ZED2_camera_marker/rectified/pose"


def sim_sessions(renders, out):
    for c in SIM_CONDITIONS:
        root = os.path.join(renders, f"lunaryard_aerial_{c}")
        t, left, _ = read_frames_csv(root)
        gt = np.loadtxt(os.path.join(root, "gt_cam0.tum"))
        assert len(gt) == len(left), (root, len(gt), len(left))
        T = np.tile(np.eye(4), (len(gt), 1, 1))
        T[:, :3, :3] = Rotation.from_quat(gt[:, 4:8]).as_matrix()
        T[:, :3, 3] = gt[:, 1:4]
        cal = json.load(open(os.path.join(root, "calib.json")))
        meta = {"name": f"sim_{c}", "kind": "sim", "image_root": root, "depth_dir": os.path.join(root, "depth0"),
                "fx": cal["fx"], "fy": cal["fy"], "cx": cal["cx"], "cy": cal["cy"], "baseline_m": cal["baseline_m"],
                "width": cal["width"], "height": cal["height"], "lighting": c,
                "pose_source": "gt_cam0.tum by row index (OpenCV optical cam0->world)"}
        write_session(os.path.join(out, meta["name"]), meta, T, np.ones(len(T), bool), t)
        print(meta["name"], len(T), "frames")


def real_sessions(root, work, out):
    from slambench.datasets.lunapolaris import LunaPolarisAdapter
    from slambench.evaluation.metrics import associate
    from slambench.trajectory import Trajectory
    from slambench.types import SensorMode

    for name, (seq, shift) in REAL.items():
        ds = LunaPolarisAdapter(Path(root), seq, {"gt_topic": GT_TOPIC, "right_row_shift_px": shift})
        prep = ds.prepare_cached(SensorMode.STEREO, Path(work) / "prepared")
        t, left, _ = read_frames_csv(os.path.realpath(prep.directory))
        gt = ds.ground_truth()
        T_bs = ds.T_gt_body_sensor("cam0")
        P = Trajectory(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw).transformed(T_right=T_bs).poses()
        ii, jj, _ = associate(t, gt.timestamps_ns, 0.02)
        T = np.tile(np.eye(4), (len(t), 1, 1))
        valid = np.zeros(len(t), bool)
        T[ii], valid[ii] = P[jj], True
        st = ds.stereo_calibration()
        meta = {"name": f"real_{name}", "kind": "real", "image_root": os.path.realpath(prep.directory),
                "fx": st.left.fx, "fy": st.left.fy, "cx": st.left.cx, "cy": st.left.cy, "baseline_m": st.baseline_m,
                "width": st.left.width, "height": st.left.height, "sequence": seq, "right_row_shift_px": shift,
                "pose_source": f"{GT_TOPIC} @ T_body_cam0 from /tf_static, nearest sample within 20 ms",
                "T_gt_body_cam0": T_bs.tolist()}
        write_session(os.path.join(out, meta["name"]), meta, T, valid, t)
        print(meta["name"], len(T), "frames", int(valid.sum()), "with GT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--renders", default="/storage/ice1/3/6/bkuang7/omnilrs/renders")
    ap.add_argument("--lunapolaris", default=os.path.expanduser("~/scratch/LunaPolaris"))
    ap.add_argument("--work", default=os.path.expanduser("~/scratch/slam-bench-work"),
                    help="slambench work_dir (prepared inputs are cached under work/prepared)")
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args()
    sim_sessions(a.renders, a.out)
    if not a.skip_real:
        real_sessions(a.lunapolaris, a.work, a.out)


if __name__ == "__main__":
    main()
