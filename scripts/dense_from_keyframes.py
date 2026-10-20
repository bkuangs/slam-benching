#!/usr/bin/env python3
"""Dense map from an ORB-SLAM3 stereo run: per-keyframe stereo depth fused into a TSDF.

ORB-SLAM3 only keeps a sparse map. This post-processes a finished run: for every keyframe of the final
map it computes a disparity image with OpenCV SGBM (left-right checked), converts it to depth with the
run's rectified calibration and integrates it into an Open3D TSDF volume at the keyframe's final
(loop-closed, bundle-adjusted) pose. Outputs in <run>/dense/ (or --out):

  mesh.ply          triangle mesh (marching cubes on the TSDF), vertex colors from the left images
  cloud.ply         TSDF surface points
  dense_stats.json  timing (per-keyframe depth and fusion time, CPU), sizes and a consistency check:
                    distance from each sparse ORB-SLAM3 map point to the mesh

Usage: python scripts/dense_from_keyframes.py <orbslam3 run dir> [--voxel 0.05] [--max-depth 5]
Needs opencv-python and open3d.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd


def load_calib(settings_yaml: Path):
    fs = cv2.FileStorage(str(settings_yaml), cv2.FILE_STORAGE_READ)
    get = lambda k: fs.getNode(k).real()
    calib = dict(fx=get("Camera1.fx"), fy=get("Camera1.fy"), cx=get("Camera1.cx"), cy=get("Camera1.cy"),
                 w=int(get("Camera.width")), h=int(get("Camera.height")), b=get("Stereo.b"))
    fs.release()
    if calib["b"] <= 0:
        raise SystemExit(f"{settings_yaml}: no stereo baseline (Stereo.b); stereo runs only")
    return calib


def pose_matrix(row) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([row.qw, row.qx, row.qy, row.qz])
    T[:3, 3] = [row.tx, row.ty, row.tz]
    return T


def gt_keyframe_poses(run: Path, t_ns: np.ndarray):
    """Ground-truth T_world_cam0 at the keyframe times (LunaPolaris) and which keyframes have ground truth."""
    import yaml
    from slambench.datasets.lunapolaris import LunaPolarisAdapter
    from slambench.evaluation.map_quality import PoseInterpolator
    ds = yaml.safe_load((run / "config.resolved.yaml").read_text())["dataset"]
    if ds["name"] != "lunapolaris":
        raise SystemExit("--gt-poses is implemented for LunaPolaris only")
    ad = LunaPolarisAdapter(Path(ds["root"]).expanduser(), ds["sequence"], ds.get("options") or {})
    gt = ad.ground_truth()
    pose = PoseInterpolator(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, ad.T_gt_body_sensor("cam0"))
    keep = pose.valid(t_ns)
    return pose.poses(t_ns[keep]), keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, help="output directory (default <run_dir>/dense)")
    ap.add_argument("--voxel", type=float, default=0.05, help="TSDF voxel size, m (default 0.05)")
    ap.add_argument("--max-depth", type=float, default=5.0, help="ignore depth beyond this, m (default 5)")
    ap.add_argument("--num-disp", type=int, default=64, help="SGBM disparity range, multiple of 16")
    ap.add_argument("--block", type=int, default=5, help="SGBM block size (odd)")
    ap.add_argument("--uniqueness", type=int, default=10, help="SGBM uniquenessRatio (higher = stricter)")
    ap.add_argument("--gt-poses", action="store_true",
                    help="fuse at ground-truth camera poses (LunaPolaris) instead of ORB-SLAM3's; the mesh is "
                         "then in the ground-truth world (isolates depth error from pose error)")
    args = ap.parse_args()

    run = args.run_dir.resolve()
    alg = run / "algorithm"
    out = (args.out or run / "dense").resolve()
    out.mkdir(parents=True, exist_ok=True)
    calib = load_calib(alg / "orbslam3_settings.yaml")
    prepared = Path(json.loads((run / "prepared_input.json").read_text())["directory"])
    frames = pd.read_csv(prepared / "frames.csv")
    stamps = frames.timestamp_ns.to_numpy(np.int64)

    kfs = pd.read_csv(alg / "keyframes.csv")
    kfs = kfs[kfs.in_final_frame == 1].sort_values("timestamp_ns").reset_index(drop=True)
    # keyframes.csv stamps went through a double; match each to the nearest input frame
    idx = np.clip(np.searchsorted(stamps, kfs.timestamp_ns.to_numpy(np.int64)), 1, len(stamps) - 1)
    idx -= (kfs.timestamp_ns.to_numpy(np.int64) - stamps[idx - 1]) < (stamps[idx] - kfs.timestamp_ns.to_numpy(np.int64))
    match_err_ms = np.abs(stamps[idx] - kfs.timestamp_ns.to_numpy(np.int64)).max() / 1e6
    if match_err_ms > 5:
        raise SystemExit(f"keyframe/frame timestamp mismatch up to {match_err_ms:.1f} ms")

    if args.gt_poses:
        T_wc, keep = gt_keyframe_poses(run, stamps[idx])
        kfs, idx = kfs[keep].reset_index(drop=True), idx[keep]
    else:
        T_wc = np.array([pose_matrix(r) for r in kfs.itertuples()])

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=args.num_disp, blockSize=args.block,
        P1=8 * 3 * args.block ** 2, P2=32 * 3 * args.block ** 2, disp12MaxDiff=1,
        uniquenessRatio=args.uniqueness, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM)
    intr = o3d.camera.PinholeCameraIntrinsic(calib["w"], calib["h"], calib["fx"], calib["fy"],
                                             calib["cx"], calib["cy"])
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=4 * args.voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    fb = calib["fx"] * calib["b"]
    min_disp = fb / args.max_depth

    t_load, t_depth, t_fuse, valid_frac = [], [], [], []
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for k, fi in enumerate(idx):
        t0 = time.perf_counter()
        left = cv2.imread(str(prepared / frames.left_path[fi]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(prepared / frames.right_path[fi]), cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        disp = sgbm.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
        depth = np.where(disp > min_disp, fb / np.maximum(disp, 1e-6), 0.0).astype(np.float32)
        t2 = time.perf_counter()
        valid_frac.append(float((depth > 0).mean()))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(left[:, :, ::-1])), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=args.max_depth, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(T_wc[k]))  # extrinsic = T_cw
        t3 = time.perf_counter()
        t_load.append(t1 - t0); t_depth.append(t2 - t1); t_fuse.append(t3 - t2)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(idx)} keyframes", flush=True)
    t4 = time.perf_counter()
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud = vol.extract_point_cloud()
    t_extract = time.perf_counter() - t4
    cpu_s, wall_s = time.process_time() - cpu0, time.perf_counter() - wall0
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(out / "cloud.ply"), cloud)

    # Consistency: distance from each sparse map point (triangulated / bundle-adjusted by ORB-SLAM3)
    # to the fused surface, for points inside the depth range of some keyframe (ORB-SLAM3 poses only;
    # with --gt-poses the sparse map is in another frame).
    mp = pd.read_csv(alg / "map_points.csv")
    mp = mp[mp.in_final_frame == 1][["x", "y", "z"]].to_numpy()
    sparse_check = None
    if not args.gt_poses:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        kf_pos = T_wc[:, :3, 3]
        near = np.min(np.linalg.norm(mp[:, None, :] - kf_pos[None, ::5, :], axis=2), axis=1) <= args.max_depth
        dist = scene.compute_distance(o3d.core.Tensor(mp[near], dtype=o3d.core.Dtype.Float32)).numpy()
        sparse_check = {"total": int(len(mp)), "within_depth_range": int(near.sum()),
                        "dist_to_mesh_m": {"median": float(np.median(dist)),
                                           "p90": float(np.percentile(dist, 90)),
                                           "frac_within_voxel": float((dist <= args.voxel).mean())}}

    ms = lambda a: {"mean": 1e3 * float(np.mean(a)), "p95": 1e3 * float(np.percentile(a, 95))}
    stats = {
        "run_dir": str(run), "n_keyframes": int(len(idx)), "voxel_m": args.voxel, "max_depth_m": args.max_depth,
        "pose_source": "ground_truth" if args.gt_poses else "orbslam3",
        "sgbm": {"num_disparities": args.num_disp, "block_size": args.block, "uniqueness_ratio": args.uniqueness},
        "valid_depth_pixel_fraction_mean": float(np.mean(valid_frac)),
        "per_keyframe_ms": {"image_load": ms(t_load), "stereo_depth": ms(t_depth), "tsdf_integrate": ms(t_fuse)},
        "mesh_extract_s": t_extract, "total_wall_s": wall_s, "total_process_cpu_s": cpu_s,
        "opencv_threads": cv2.getNumThreads(),
        "mesh": {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
                 "file_mb": os.path.getsize(out / "mesh.ply") / 1e6},
        "cloud_points": len(cloud.points),
        "sparse_points": sparse_check,
    }
    (out / "dense_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
