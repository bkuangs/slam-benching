# FAST-LIO2 (LiDAR-inertial) on LunaPolaris seq01 / 02 / 11

FAST-LIO2 (`fastlio` adapter, `lidar_inertial` mode: Livox HAP + ZED2 IMU) compared with the stereo
baselines ORB-SLAM3 and RTAB-Map (`*_stereo_lean`, median of 3 repeats). FAST-LIO is a **single run**
per sequence. All runs used the same node type (Xeon Gold 6226, 8 CPUs, 32 GB).

Configs: `configs/lunapolaris_seq{01,02,11}_fastlio_lidar_inertial.yaml`.

## Trajectory accuracy and robustness

Final trajectories, SE3-aligned to motion capture. FAST-LIO poses are evaluated in the LiDAR frame and
the stereo methods in the left camera frame (each against ground truth mapped into that sensor frame).

| Seq | Method | ATE RMSE (m) | RPE 1 m (m / °) | Frame coverage | Loss events |
|---|---|---|---|---|---|
| 01 | FAST-LIO2 | 0.066 | 0.073 / 2.91 | 0.992 | 0 |
| 01 | ORB-SLAM3 | **0.031** | 0.061 / 2.94 | 1.000 | 0 |
| 01 | RTAB-Map | 0.067 | 0.060 / 3.04 | 1.000 | 0 |
| 02 | FAST-LIO2 | **0.051** | 0.064 / 2.91 | 0.995 | 0 |
| 02 | ORB-SLAM3 | 0.059 | 0.063 / 2.92 | 1.000 | 0 |
| 02 | RTAB-Map | 0.057 | 0.059 / 2.95 | 1.000 | 0 |
| 11 | FAST-LIO2 | **0.061** | 0.071 / 2.97 | 0.999 | 0 |
| 11 | ORB-SLAM3 | 0.081 | 0.063 / 2.80 | 1.000 | 0 |
| 11 | RTAB-Map | 0.082 | 0.060 / 2.85 | 1.000 | 0 |

- FAST-LIO coverage is below 1 only because of the first 1–7 s, while it waits for IMU initialisation
  (`initializing` status: 58 / 38 / 8 scans). It has no failures after that.
- ATE rotation RMSE is about 4.6° for every method. Of that, 4.3° is the constant offset `diag_const_rot_offset`,
  which is the same for FAST-LIO, ORB-SLAM3 and RTAB-Map (3.9–4.4°). That points to the ground-truth
  marker→sensor extrinsics (nominal URDF values), not to any estimator.
- The RPE is roughly the same for all three methods. RPE at 1 m is dominated by the same rotation offset
  and by ground-truth noise.

## Resource usage

| Seq | Method | Input rate | Per-frame time mean / p95 (ms) | CPU per frame (ms) | CPU per s of data | Mid-run cores | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|
| 01 | FAST-LIO2 | 8.7 Hz | 7.7 / 10.1 | 29 | 0.25 | 0.23 | 416 |
| 01 | ORB-SLAM3 | 14.6 Hz | 24.0 / 31.7 | 77 | 1.12 | 0.77 | 1121 |
| 01 | RTAB-Map | 14.6 Hz | 59.0 / 67.2 | 82 | 1.19 | 0.62 | 514 |
| 02 | FAST-LIO2 | 8.7 Hz | 8.3 / 10.9 | 32 | 0.27 | 0.26 | 403 |
| 02 | ORB-SLAM3 | 14.6 Hz | 24.5 / 29.9 | 72 | 1.05 | 0.72 | 970 |
| 02 | RTAB-Map | 14.6 Hz | 60.5 / 72.4 | 84 | 1.24 | 0.63 | 521 |
| 11 | FAST-LIO2 | 8.7 Hz | 8.0 / 10.3 | 31 | 0.26 | 0.23 | 520 |
| 11 | ORB-SLAM3 | 14.6 Hz | 24.6 / 29.8 | 68 | 1.01 | 0.69 | 677 |
| 11 | RTAB-Map | 14.6 Hz | 64.3 / 74.3 | 89 | 1.32 | 0.64 | 496 |

- **CPU per second of data** is the fair comparison, because the frame rates differ (8.7 Hz LiDAR
  scans vs 14.6 Hz stereo pairs). On that measure FAST-LIO2 uses about 4× less CPU than either stereo method.
- **Per-frame time** for FAST-LIO is the IEKF update plus the map increment (`track_time_s`). It does
  not include reading a scan from disk. CPU, RSS and core counts are for the whole process tree, which
  includes the in-container `rosmaster` (about 40–90 MB RSS).
- **About a third of FAST-LIO's CPU** (~60 s of ~220 s) goes into tracking. The rest is scan decoding
  and the 2 cm map-export accumulation (`map_accumulation_time_total_s` 17–23 s). The map export is
  needed for the map-quality evaluation below, but it is not part of FAST-LIO itself.
- **Latency:** the input-to-result latency is about 10 ms in the median. On seq11 it peaks at 13 s,
  around scans 169–173. That spike is a scratch-filesystem read stall (`read_time_s` up to 3.2 s); it
  is not caused by the estimator, whose `track_time_s` stays at or below 14 ms.

## Map quality (LiDAR reference)

Method: `scripts/eval_map_quality.py --orb … --rtab … --lio …`, with a 2 cm voxel grid and the
LiDAR reference rebuilt from motion-capture poses. The ORB-SLAM3 and RTAB-Map rows reproduce the
earlier map-quality run exactly. Every map is placed in the ground-truth world by aligning its
trajectory (the same SE3 alignment as ATE). Columns:

- **P/R/F**: precision, recall and F-score at 10 cm.
- **Recall-all**: recall over every reference voxel. Plain recall counts only voxels the camera could see.
- **After ICP**: scores after an extra point-to-plane ICP against the reference. This separates local
  geometry from global misplacement.

| Seq | Map | Points | Acc. median | P / R / F @10 cm | F @5 cm | Recall-all @10 cm | ICP Δz | After ICP: F @10 / F @5 |
|---|---|---|---|---|---|---|---|---|
| 01 | lio_map | 1.12 M | **1.8 cm** | 0.91 / 0.53 / **0.67** | **0.47** | **0.44** | −21 cm | **0.92 / 0.74** |
| 01 | orb_tsdf | 411 k | 5.4 cm | 0.66 / 0.49 / 0.56 | 0.31 | 0.43 | −12 cm | 0.74 / 0.52 |
| 01 | rtab_cloud | 629 k | 8.3 cm | 0.58 / 0.40 / 0.47 | 0.25 | 0.28 | −29 cm | 0.75 / 0.57 |
| 02 | lio_map | 1.15 M | **2.0 cm** | 0.89 / 0.54 / **0.68** | **0.48** | **0.45** | −21 cm | **0.93 / 0.77** |
| 02 | orb_tsdf | 340 k | 7.0 cm | 0.61 / 0.39 / 0.48 | 0.24 | 0.31 | −17 cm | 0.68 / 0.45 |
| 02 | rtab_cloud | 745 k | 13.7 cm | 0.35 / 0.29 / 0.32 | 0.12 | 0.20 | −31 cm | 0.71 / 0.55 |
| 11 | lio_map | 1.20 M | **2.1 cm** | 0.90 / 0.54 / **0.68** | **0.47** | **0.45** | −21 cm | **0.93 / 0.76** |
| 11 | orb_tsdf | 408 k | 6.6 cm | 0.60 / 0.54 / 0.57 | 0.34 | 0.46 | −12 cm | 0.66 / 0.45 |
| 11 | rtab_cloud | 525 k | 12.7 cm | 0.40 / 0.26 / 0.31 | 0.13 | 0.16 | −34 cm | 0.65 / 0.49 |

ORB-SLAM3 sparse points and the RTAB-Map OctoMap are in
`~/scratch/slam-bench-work/map_quality_lio/sequence_*/map_quality.json`.

### Reading these numbers

- **The LIO comparison is not independent.** The reference is built from the same Livox scans that
  FAST-LIO maps. Only the poses differ: motion capture for the reference, FAST-LIO's own poses for
  the LIO map. So the `lio_map` scores measure FAST-LIO's pose and placement error, not how well the
  sensor captures geometry. The stereo maps are scored against a different sensor. Read the camera and
  LiDAR rows as "how close to the LiDAR reference", not as a ranking of sensors.
- **Every map needs a −12 to −34 cm vertical ICP correction**, with only 0.3–3° of rotation. For
  FAST-LIO the correction is 21 cm on all three sequences, even though its ATE is about 6 cm. Most of
  this is therefore probably a reference-side bias: the nominal marker→Livox rotation is likely off by
  a few degrees. That is the same ~4.3° rotation offset seen in ATE above, and with the HAP pitched
  down, it shifts the reference points vertically. Pre-ICP numbers include that bias; post-ICP numbers
  show local consistency.
- **Recall of the LIO map stays near 0.54 before ICP.** That ceiling follows from the vertical offset;
  after ICP, recall reaches 0.85–0.89 (camera-visible voxels).
- **Recall-all** applies only before ICP; it is not computed after ICP.

## Caveats

- FAST-LIO is a single run per sequence. ORB-SLAM3 and RTAB-Map numbers are medians of 3.
- IMU noise parameters are FAST-LIO defaults, not calibrated for the ZED2 IMU. The LiDAR–IMU
  extrinsic comes from `/tf_static` and is not estimated online (`extrinsic_est_en: false`).
- The ZED2 IMU (~98 Hz) is used instead of the Livox IMU, because the Livox IMU is heavily
  decimated in these bags. See `docs/datasets/lunapolaris.md`.
- Offline driver: scans are fed at recorded pace, so wall time roughly equals sequence length. The
  latency figures include time spent queued behind slower scans.
