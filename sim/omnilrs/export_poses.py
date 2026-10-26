"""Export a real LunaPolaris camera trajectory as render poses for the OmniLRS Lunalab stereo renderer.

Writes an .npz with the real left-camera frame timestamps (restricted to GT coverage), the GT left-camera
(OpenCV optical frame) poses interpolated at those timestamps, and the ZED2 rectified stereo calibration.
Run in the slam-benching venv:
    python sim/omnilrs/export_poses.py configs/lunapolaris_seq01_orbslam3_stereo.yaml poses.npz [--max-s 60]
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from slambench.runner import make_dataset
from slambench.trajectory import Trajectory
from slambench.types import SensorMode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--max-s", type=float, default=None, help="keep only the first N seconds of frames")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    ds = make_dataset(cfg)
    gt = ds.ground_truth()
    g = Trajectory(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, gt.body_frame).transformed(
        T_right=ds.T_gt_body_sensor("cam0"))
    calib = ds.stereo_calibration()

    prepared = ds.prepare_cached(SensorMode.STEREO, Path(cfg["work_dir"]).expanduser() / "prepared")
    frames = np.genfromtxt(Path(prepared.directory) / "frames.csv", delimiter=",", skip_header=1,
                           usecols=0, dtype=np.int64)

    tg = g.timestamps_ns
    t = frames[(frames >= tg[0]) & (frames <= tg[-1])]
    if a.max_s is not None:
        t = t[t <= t[0] + int(a.max_s * 1e9)]
    t = t[:: a.stride]
    # Drop frames inside GT dropouts (>0.1 s between the bracketing GT samples).
    j = np.clip(np.searchsorted(tg, t), 1, len(tg) - 1)
    t = t[(tg[j] - tg[j - 1]) < 100_000_000]

    tgs = (tg - tg[0]) * 1e-9
    ts = (t - tg[0]) * 1e-9
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = Slerp(tgs, Rotation.from_quat(g.quaternions_xyzw))(ts).as_matrix()
    T[:, :3, 3] = np.stack([np.interp(ts, tgs, g.positions[:, k]) for k in range(3)], axis=1)

    L = calib.left
    np.savez(a.out, timestamps_ns=t, T_world_cam0=T, width=L.width, height=L.height, fx=L.fx, fy=L.fy,
             cx=L.cx, cy=L.cy, baseline_m=calib.baseline_m,
             source=f"{cfg['dataset']['name']}/{cfg['dataset']['sequence']} GT cam0 (OpenCV optical frame)")
    print(f"{len(t)} frames over {(t[-1] - t[0]) * 1e-9:.1f} s, fx={L.fx:.3f} baseline={calib.baseline_m:.5f} "
          f"-> {a.out}")


if __name__ == "__main__":
    main()
