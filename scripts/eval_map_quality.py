#!/usr/bin/env python3
"""Geometric map quality on LunaPolaris against a LiDAR reference map.

Reference map: every --scan-stride-th Livox scan (/leo01/livox/lidar), each point placed in the
motion-capture world with the ground-truth pose interpolated at the point's own timestamp (motion
compensation) and the /tf_static marker->livox extrinsic (nominal URDF value), merged into a --voxel
grid; voxels hit fewer than --min-hits times are dropped (noise, people walking by). Alternate scans
also go into two half maps whose disagreement is reported as the reference's own noise floor.
Cached in <out>/<sequence>/reference_*.npz.

Estimated maps evaluated (each if present):
  orb_sparse         ORB-SLAM3 sparse map points (map_points.csv, final map)
  orb_tsdf[:<dir>]   mesh from scripts/dense_from_keyframes.py (<orb run>/<dir>/mesh.ply, sampled); with
                     --poses-tum it is fused at another run's (e.g. RTAB-Map's) poses
  rtab_cloud         RTAB-Map dense cloud export (rtabmap_cloud.ply)
  rtab_octomap       RTAB-Map OctoMap occupied cell centres (rtabmap_octomap.ply, free cells removed)
  lio_map            FAST-LIO voxel map (fastlio_map.ply; built from the same Livox scans as the reference,
                     but placed with FAST-LIO's own poses instead of motion capture)
Each is moved into the ground-truth world from the sensor poses it was built from (keyframes, nodes, a
dense dir's poses.csv, or FAST-LIO's per-scan LiDAR poses against ground truth in the LiDAR frame), in two ways, so trajectory errors show up as map errors:
  aligned_by_trajectory   SE3 best aligning those camera positions to ground truth (as ATE does); a
                          nearly planar path leaves the tilt poorly constrained
  aligned_by_orientation  rotation from the camera orientations (chordal mean), translation from positions A second score after ICP against the reference (point-to-plane, 10 cm) separates local
geometry from that global misplacement. Maps built from ground-truth poses (dense_from_keyframes.py
--gt-poses) are already in the ground-truth world and are not aligned.

Metrics (slambench/evaluation/map_quality.py), all maps voxelised to --voxel first:
  precision@tau  estimated points within tau of the reference, counting only estimated points inside the
                 space the LiDAR covered (25 cm cells, dilated by one); `in_reference_region` gives the share
  recall@tau     reference points the camera could see (GT camera poses at ~1 Hz, depth <= --max-depth,
                 z-buffer against occlusion) that have an estimated point within tau
  fscore@tau     harmonic mean
  recall_all@tau recall over every reference voxel (not only camera-visible ones), e.g. for LiDAR maps

Usage: python scripts/eval_map_quality.py --orb <orbslam3 run> --rtab <rtabmap run> [--lio <fastlio run>]
                                          [--dense-dir dense ...]
Needs open3d.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from slambench.datasets.lunapolaris import LunaPolarisAdapter
from slambench.evaluation.map_quality import (align_by_orientation, PoseInterpolator, VoxelAccumulator, region_mask, score,
                                              umeyama_se3, visible_mask, voxel_downsample)

LIDAR_TOPIC = "/leo01/livox/lidar"
LIDAR_FRAME = "leo01/livox_frame"


def adapter_for(run: Path) -> LunaPolarisAdapter:
    ds = yaml.safe_load((run / "config.resolved.yaml").read_text())["dataset"]
    if ds["name"] != "lunapolaris":
        raise SystemExit(f"{run}: dataset {ds['name']}; this evaluation needs LunaPolaris (LiDAR + mocap)")
    return LunaPolarisAdapter(Path(ds["root"]).expanduser(), ds["sequence"], ds.get("options") or {})


def frame_times(run: Path) -> np.ndarray:
    prepared = Path(json.loads((run / "prepared_input.json").read_text())["directory"])
    return pd.read_csv(prepared / "frames.csv").timestamp_ns.to_numpy(np.int64)


def build_reference(ad: LunaPolarisAdapter, t_range, args, cache: Path):
    if cache.exists():
        z = np.load(cache)
        return z["points"], json.loads(str(z["info"]))
    gt = ad.ground_truth()
    T_ml = ad.static_tf().lookup(ad.options["gt_body_frame"], LIDAR_FRAME)
    if T_ml is None:
        raise SystemExit(f"no /tf_static path {ad.options['gt_body_frame']} -> {LIDAR_FRAME}")
    pose = PoseInterpolator(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, T_ml)
    halves = [VoxelAccumulator(args.voxel), VoxelAccumulator(args.voxel)]
    n_scans = n_used = n_pts = 0
    t0 = time.time()
    for i, (_, m) in enumerate(ad._iter(LIDAR_TOPIC)):
        n_scans += 1
        if i % args.scan_stride:
            continue
        off = {f.name: f.offset for f in m.fields}
        dt = np.dtype({"names": ["x", "y", "z", "t"], "formats": ["<f4", "<f4", "<f4", "<f8"],
                       "offsets": [off["x"], off["y"], off["z"], off["timestamp"]], "itemsize": m.point_step})
        a = np.frombuffer(bytes(m.data), dt, count=m.width * m.height)
        xyz = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(float)
        t = a["t"].astype(np.int64)
        r = np.linalg.norm(xyz, axis=1)
        keep = np.isfinite(r) & (r > args.lidar_min_range) & (r < args.lidar_max_range)
        keep &= (t >= t_range[0]) & (t <= t_range[1])
        if keep.sum() < 100:
            continue
        xyz, t = xyz[keep], t[keep]
        keep = pose.valid(t)
        xyz, t = xyz[keep], t[keep]
        if len(t) < 100:
            continue
        # motion compensation: one ground-truth pose per 10 ms slice of the 100 ms scan
        edges = np.linspace(t.min(), t.max() + 1, 11)
        k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, 9)
        T = pose.poses(((edges[:-1] + edges[1:]) / 2).astype(np.int64))
        world = np.einsum("nij,nj->ni", T[k, :3, :3], xyz) + T[k, :3, 3]
        halves[n_used % 2].add(world)
        n_used += 1
        n_pts += len(world)
        if n_used % 500 == 0:
            print(f"  reference: {n_used} scans, {n_pts / 1e6:.0f}M points, {time.time() - t0:.0f}s", flush=True)
    a_pts, _ = halves[0].result(min_count=max(1, args.min_hits // 2))
    b_pts, _ = halves[1].result(min_count=max(1, args.min_hits // 2))
    d_ab = np.r_[cKDTree(b_pts).query(a_pts)[0], cKDTree(a_pts).query(b_pts)[0]]
    halves[0].merge(halves[1])
    points, counts = halves[0].result(min_count=args.min_hits)
    info = {"lidar_scans_in_bag": n_scans, "scans_used": n_used, "points_used": n_pts,
            "voxel_m": args.voxel, "min_hits": args.min_hits, "n_voxels": int(len(points)),
            "T_marker_lidar": T_ml.tolist(), "extrinsic_source": "/tf_static (nominal URDF)",
            "half_map_distance_m": {"median": float(np.median(d_ab)), "p90": float(np.percentile(d_ab, 90)),
                                    "within_2cm": float(np.mean(d_ab <= 0.02))},
            "build_s": time.time() - t0}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, points=points, info=json.dumps(info))
    return points, info


def read_ply_points(path: Path):
    pc = o3d.io.read_point_cloud(str(path))
    return np.asarray(pc.points), (np.asarray(pc.colors) if pc.has_colors() else None)


def load_maps(orb: Path, rtab: Path, lio: Path, dense_dirs, voxel):
    """name -> (points, trajectory) where trajectory = (t_ns, positions, quaternions xyzw[, sensor frame])
    of the sensor poses the map was built from (camera unless a frame is given), or None if the map is
    already in the ground-truth world."""
    def traj_of(df):
        return (df.timestamp_ns.to_numpy(np.int64), df[["tx", "ty", "tz"]].to_numpy(),
                df[["qx", "qy", "qz", "qw"]].to_numpy())

    maps = {}
    if orb:
        kf = pd.read_csv(orb / "algorithm" / "keyframes.csv")
        kf = kf[kf.in_final_frame == 1]
        traj = traj_of(kf)
        mp = pd.read_csv(orb / "algorithm" / "map_points.csv")
        maps["orb_sparse"] = (mp[mp.in_final_frame == 1][["x", "y", "z"]].to_numpy(), traj)
        for d in dense_dirs:
            mesh_path = orb / d / "mesh.ply"
            if not mesh_path.exists():
                print(f"  skip {mesh_path} (missing)")
                continue
            stats = json.loads((orb / d / "dense_stats.json").read_text())
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            n = max(1000, int(mesh.get_surface_area() / voxel ** 2))
            pts = np.asarray(mesh.sample_points_uniformly(n).points)
            name = "orb_tsdf" if d == "dense" else f"orb_tsdf:{d}"
            if stats.get("pose_source") == "ground_truth":
                maps[name] = (pts, None)
            elif (orb / d / "poses.csv").exists():
                maps[name] = (pts, traj_of(pd.read_csv(orb / d / "poses.csv")))
            elif stats.get("pose_source", "orbslam3") == "orbslam3":
                maps[name] = (pts, traj)
            else:
                print(f"  skip {mesh_path} (no poses.csv for pose source {stats['pose_source']})")
    if rtab:
        nodes = pd.read_csv(rtab / "algorithm" / "nodes.csv")
        nodes = nodes[nodes.in_final == 1]
        traj = traj_of(nodes)
        p = rtab / "algorithm" / "rtabmap_cloud.ply"
        if p.exists():
            maps["rtab_cloud"] = (read_ply_points(p)[0], traj)
        p = rtab / "algorithm" / "rtabmap_octomap.ply"
        if p.exists():
            pts, col = read_ply_points(p)
            # rtabmap-export writes all OctoMap leaves: occupied in image colour, free cells white
            occ = ~np.all(col > 0.999, axis=1) if col is not None else np.ones(len(pts), bool)
            maps["rtab_octomap"] = (pts[occ], traj)
    if lio:
        log = pd.read_csv(lio / "algorithm" / "frame_log.csv")
        log = log[log.pose_valid == 1]
        p = lio / "algorithm" / "fastlio_map.ply"
        if p.exists():
            maps["lio_map"] = (read_ply_points(p)[0], (*traj_of(log), "lidar"))
        else:
            print(f"  skip {p} (missing)")
    return maps


def icp_refine(src: np.ndarray, ref_pcd, max_dist: float):
    s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    r = o3d.pipelines.registration.registration_icp(
        s, ref_pcd, max_dist, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    T = r.transformation
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    # T's translation is the shift of the world origin, which a small rotation inflates far from the map;
    # report how far the map's own points actually move.
    moved = np.linalg.norm(src @ (T[:3, :3] - np.eye(3)).T + T[:3, 3], axis=1)
    return T, {"translation_m": float(np.linalg.norm(T[:3, 3])), "rotation_deg": float(ang),
               "point_shift_m": {"median": float(np.median(moved)), "max": float(moved.max())},
               "T": T.tolist(), "fitness": float(r.fitness), "inlier_rmse_m": float(r.inlier_rmse)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orb", type=Path, help="ORB-SLAM3 run directory")
    ap.add_argument("--rtab", type=Path, help="RTAB-Map run directory (with dense export)")
    ap.add_argument("--lio", type=Path, help="FAST-LIO run directory (fastlio_map.ply)")
    ap.add_argument("--dense-dir", action="append", default=None,
                    help="dense_from_keyframes.py output dir(s) inside the ORB-SLAM3 run (default: dense)")
    ap.add_argument("--out", type=Path, default=Path("~/scratch/slam-bench-work/map_quality").expanduser())
    ap.add_argument("--voxel", type=float, default=0.02, help="reference / evaluation voxel (m)")
    ap.add_argument("--min-hits", type=int, default=4, help="LiDAR points needed per reference voxel")
    ap.add_argument("--scan-stride", type=int, default=1)
    ap.add_argument("--lidar-min-range", type=float, default=0.5, help="m; drops the rover itself")
    ap.add_argument("--lidar-max-range", type=float, default=10.0)
    ap.add_argument("--max-depth", type=float, default=5.0, help="camera depth counted as visible (m)")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10])
    args = ap.parse_args()
    if not (args.orb or args.rtab or args.lio):
        ap.error("give --orb, --rtab and/or --lio")
    dense_dirs = args.dense_dir or ["dense"]
    runs = [r.resolve() for r in (args.orb, args.rtab, args.lio) if r]
    ad = adapter_for(runs[0])
    seq = yaml.safe_load((runs[0] / "config.resolved.yaml").read_text())["dataset"]["sequence"]
    out = args.out / seq
    out.mkdir(parents=True, exist_ok=True)

    ft = frame_times(runs[0])
    tag = f"v{args.voxel}_h{args.min_hits}_s{args.scan_stride}_r{args.lidar_min_range}-{args.lidar_max_range}"
    print(f"{seq}: reference map", flush=True)
    ref, ref_info = build_reference(ad, (ft[0], ft[-1]), args, out / f"reference_{tag}.npz")
    print(f"  {len(ref):,} voxels; half-map distance median {ref_info['half_map_distance_m']['median'] * 100:.1f} cm")

    gt = ad.ground_truth()
    cam = PoseInterpolator(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw, ad.T_gt_body_sensor("cam0"))
    sensor_pose = {"cam0": cam,
                   "lidar": PoseInterpolator(gt.timestamps_ns, gt.positions, gt.quaternions_xyzw,
                                             ad.T_gt_body_sensor("lidar"))}
    left = ad.stereo_calibration().left
    vis_cache = out / f"visible_{tag}_d{args.max_depth}.npy"
    if vis_cache.exists():
        vis = np.load(vis_cache)
    else:
        ts = ft[::15]
        ts = ts[cam.valid(ts)]
        print(f"  visibility from {len(ts)} ground-truth camera poses", flush=True)
        vis = visible_mask(ref, cam.poses(ts), left.fx, left.fy, left.cx, left.cy, left.width, left.height,
                           args.max_depth)
        np.save(vis_cache, vis)
    print(f"  {vis.sum():,} reference voxels visible to the camera ({vis.mean() * 100:.0f}%)")

    ref_tree = cKDTree(ref)
    ref_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref))
    ref_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    results = {"sequence": seq, "orb_run": str(args.orb.resolve()) if args.orb else None,
               "rtab_run": str(args.rtab.resolve()) if args.rtab else None,
               "lio_run": str(args.lio.resolve()) if args.lio else None,
               "reference": ref_info, "visible_fraction": float(vis.mean()), "maps": {}}
    aligned = {}
    all_ref = np.ones(len(ref), bool)
    for name, (raw, traj) in load_maps(args.orb and args.orb.resolve(), args.rtab and args.rtab.resolve(),
                                       args.lio and args.lio.resolve(), dense_dirs, args.voxel).items():
        entry = {}
        if traj is not None:
            t_ns, pos, quat = traj[:3]
            interp = sensor_pose[traj[3] if len(traj) > 3 else "cam0"]
            ok = interp.valid(t_ns)
            T_gt = interp.poses(t_ns[ok])
            R_est = Rotation.from_quat(quat[ok]).as_matrix()
            fits = {"aligned_by_trajectory": umeyama_se3(pos[ok], T_gt[:, :3, 3]),
                    "aligned_by_orientation": align_by_orientation(R_est, pos[ok], T_gt[:, :3, :3], T_gt[:, :3, 3])}
            entry["alignment"] = {"pairs": int(ok.sum()), "sensor_frame": traj[3] if len(traj) > 3 else "cam0"}
            for key, (R, tr) in fits.items():
                res = np.linalg.norm(T_gt[:, :3, 3] - (pos[ok] @ R.T + tr), axis=1)
                rot_res = Rotation.from_matrix(np.einsum("ij,njk->nik", R, R_est)).inv() * \
                    Rotation.from_matrix(T_gt[:, :3, :3])
                entry["alignment"][key] = {"trajectory_rmse_m": float(np.sqrt(np.mean(res ** 2))),
                                           "orientation_rms_deg": float(np.degrees(
                                               np.sqrt(np.mean(rot_res.magnitude() ** 2))))}
            # how differently the two fits orient the map's "up" axis (tilt only, ignores heading)
            up = [R.T @ np.array([0.0, 0.0, 1.0]) for R, _ in fits.values()]
            entry["alignment"]["tilt_difference_deg"] = float(np.degrees(np.arccos(np.clip(up[0] @ up[1], -1, 1))))
        else:
            fits = {"aligned_by_trajectory": (np.eye(3), np.zeros(3))}
            entry["alignment"] = {"pairs": 0, "note": "built with ground-truth poses; no alignment"}
        for key, (R, tr) in fits.items():
            pts = voxel_downsample(raw @ R.T + tr, args.voxel)
            inreg = region_mask(pts, ref)
            pts = pts[inreg]
            s = score(pts, ref, vis, args.taus, ref_tree)
            s["in_reference_region"] = float(inreg.mean())
            s.update({k.replace("recall_", "recall_all_"): v for k, v in
                      score(pts, ref, all_ref, args.taus, ref_tree).items() if k.startswith("recall_")})
            T, icp = icp_refine(pts, ref_pcd, 0.10)
            c = pts.mean(0)
            icp["shift_at_map_centroid_m"] = (T[:3, :3] @ c + T[:3, 3] - c).tolist()
            s["icp"] = icp
            s["after_icp"] = score(pts @ T[:3, :3].T + T[:3, 3], ref, vis, args.taus, ref_tree)
            entry[key] = s
            if key == "aligned_by_trajectory":
                aligned[name] = pts
            prf = "  ".join(f"P/R/F@{k} {s['precision_' + k]:.2f}/{s['recall_' + k]:.2f}/{s['fscore_' + k]:.2f}"
                            for k in (f"{round(tau * 100)}cm" for tau in args.taus))
            dz = icp["shift_at_map_centroid_m"][2]
            print(f"  {name:22s} {key.split('_')[-1]:11s} {s['n_points']:>9,} pts  acc med "
                  f"{s['accuracy_median_m'] * 100:5.1f} cm  {prf}  | ICP dz {dz * 100:+.1f} cm "
                  f"{icp['rotation_deg']:.2f} deg", flush=True)
        if "tilt_difference_deg" in entry["alignment"]:
            print(f"  {'':22s} tilt difference between the two fits {entry['alignment']['tilt_difference_deg']:.2f} deg")
        results["maps"][name] = entry
    (out / "map_quality.json").write_text(json.dumps(results, indent=2))
    plot(ref, vis, aligned, ref_tree, out / "map_quality.png", seq)
    print(f"wrote {out / 'map_quality.json'}, {out / 'map_quality.png'}")


def plot(ref, vis, maps, ref_tree, path, seq):
    n = 1 + len(maps)
    cols = 3
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5.4 * rows), sharex=True, sharey=True, squeeze=False)
    axs = axs.ravel()
    lo, hi = np.percentile(ref[vis][:, :2], 0.5, axis=0) - 0.5, np.percentile(ref[vis][:, :2], 99.5, axis=0) + 0.5
    axs[0].scatter(ref[~vis, 0], ref[~vis, 1], s=0.2, c="0.85", rasterized=True)
    axs[0].scatter(ref[vis, 0], ref[vis, 1], s=0.2, c="0.2", rasterized=True)
    axs[0].set_title(f"LiDAR reference\n({len(ref):,} voxels; dark = seen by camera)", fontsize=10)
    sc = None
    for a, (name, pts) in zip(axs[1:], maps.items()):
        d = ref_tree.query(pts)[0]
        o = np.argsort(d)[::-1]
        sc = a.scatter(pts[o, 0], pts[o, 1], s=0.3, c=d[o], cmap="turbo", vmin=0, vmax=0.2, rasterized=True)
        a.set_title(f"{name}\ndistance to reference", fontsize=10)
    for a in axs[n:]:
        a.axis("off")
    for a in axs[:n]:
        a.set_aspect("equal")
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(lo[1], hi[1])
    if sc is not None:
        fig.colorbar(sc, ax=axs[:n].tolist(), shrink=0.6, label="distance to LiDAR reference (m)")
    fig.suptitle(f"LunaPolaris {seq}, top-down in the motion-capture world (maps aligned by trajectory)")
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
