# Milestone 1 results: ORB-SLAM3 stereo × LunaPolaris sequence_01

Date: 2026-10-10. Host: PACE ICE `ice-cpu` nodes (8 cores, Slurm). Each run's `environment.json`
records the framework commit. Run outputs live in `~/scratch/slam-bench-work/runs/` (outside git)
and are aggregated in `runs/all_runs_summary.csv`.

## Estimator build (pinned)
`third_party/orbslam3/orbslam3.def` builds `orbslam3_4452a3c.sif` (sha256 `af3bf9ac…a9bf1`) with
ORB-SLAM3 `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`, Pangolin v0.6 (`dd801d24`), OpenCV 4.2.0,
Eigen 3.3.7, Boost 1.71, GCC 9.4 and Ubuntu 20.04.6. The image's
`/opt/slambench/orbslam3_applied.diff` records every change from upstream:
1. `find_package(OpenCV 4.4)` → `4.2`, because Ubuntu 20.04 ships 4.2.
2. `0001-expose-current-map-id.patch` adds `System::GetCurrentMapId()` for per-frame diagnostics.
3. `0002-fix-rectified-stereo-settings-print.patch` fixes an upstream segfault. For
   `Camera.type: "Rectified"` stereo, `Settings` prints `originalCalib2_`, which is never
   initialised. The first LunaPolaris run found this and was recorded as `crash` (signal 11).

## 1. Installation check on a supported dataset (TUM-VI room1, stereo)
| run | image | ATE trans RMSE | ATE rot RMSE | RPE 1 m trans / rot | frames OK | losses | p95 track time / period |
|---|---|---|---|---|---|---|---|
| `20261010-160107_3de946` | final | 0.0116 m | 0.45° | 0.0131 m / 0.60° | 100 % | 0 | 48.3 / 50 ms |
| `20261010-154329_cfce7f` | before patch 0002 | 0.0147 m | 0.44° | 0.0146 m / 0.59° | 100 % | 0 | 49.5 / 50 ms |

2762 of 2821 frames matched the 120 Hz mocap within 5 ms, covering 98.6 % of the duration. The two
runs differ by 0.3 cm (1.2 vs 1.5 cm) because multi-threaded ORB-SLAM3 is not deterministic, so
comparisons need repeated runs.

Two failures were found and fixed during verification. Both stay in the summary as visible runs:
* **`crash` (exit 255).** ORB-SLAM3's `Settings.cc` requires `Camera1.overlappingBegin`, but its own
  `Examples/Stereo/TUM-VI.yaml` uses `Camera.overlappingBegin`. The settings generator now follows
  `Settings.cc`.
* **"success" with 1410 loss events and 50 % coverage.** TUM-VI `512_16` PNGs are 16-bit. The
  driver read them unchanged, so ORB found no features. The adapter now converts them with `>> 8`,
  which is bit-identical to the `IMREAD_GRAYSCALE` read in ORB-SLAM3's example, and records the
  conversion. The driver now rejects non-8-bit input.

## 2. First pairing: LunaPolaris sequence_01, stereo (run `20261010-160107_35bc03`)
Config: `configs/lunapolaris_seq01_orbslam3_stereo.yaml` (1200 ORB features, loop closure on,
real-time playback, no preprocessing). Input: 13 194 stereo pairs, after dropping 3 left-only and
39 right-only frames. Calibration comes from the bag's `camera_info` (fx 263.27, baseline 0.11983 m).

| metric | value |
|---|---|
| status | success (exit 0), 1 map, 833 keyframes |
| tracking coverage (estimator states) | 13 194/13 194 frames OK over 904.8 s, 0 loss events, first OK at frame 0 |
| final trajectory | 13 194 poses (all frames) in the left camera optical frame |
| association | 12 516 matched (94.9 % of frames) within 20 ms, mean \|Δt\| 8.4 ms, 12 segments (mocap gaps) |
| evaluated duration | 889.2 s (98.3 % of input duration) |
| ATE translation (SE(3)) | **RMSE 0.0311 m**, median 0.0281 m, max 0.0578 m |
| ATE rotation | RMSE 4.27° (see caveat) |
| RPE 1 s | 0.0035 m / 0.87° RMSE (12 407 pairs) |
| RPE 1 m (GT path length) | 0.061 m / 2.95° RMSE (10 880 pairs) (see caveat) |
| per-frame tracking time | mean 22.4 ms, p95 25.0 ms vs 66.7 ms frame period (0 % of frames over) |
| tracking-thread throughput | 44.6 fps (offline, single sequence) |
| wall time / CPU | 1212 s wall (real-time throttled: ≈ 905 s sequence plus startup and saving), 921 s CPU user |
| peak RSS | 1.09 GB (process tree, sampled every 0.5 s) |

**Caveat (GT extrinsic):**
* **What dominates the rotation error:** a near-constant 4.1° offset in the body frame, mostly a
  tilt about the camera x-axis. Removing it would leave 1.18° RMSE.
* **Where it comes from:** the nominal `/tf_static` marker→camera chain, which uses URDF values
  rather than a hand–eye calibration.
* **How it is handled:** it is reported as
  `accuracy.diagnostics.constant_body_rotation_offset` and is **not** corrected.
* **Effect on RPE:** it inflates translational RPE over 1 m by about 1 m·sin(4.1°) ≈ 7 cm.
* **Effect on ATE:** translation ATE after SE(3) alignment is only weakly affected, because the
  lever arm is 7 cm.
* **Consequence:** don't compare rotation ATE or per-metre RPE across datasets until the extrinsic
  is verified.

## 3. Rejections (sensor/configuration compatibility)
Every experiment in `configs/rejection_examples.yaml` is rejected before execution
(`config_rejected`) and listed in the summary:
* LunaPolaris stereo_inertial: no IMU noise densities or random walks are published.
* LunaPolaris lidar_inertial with ORB-SLAM3: ORB-SLAM3 doesn't support the mode, and the LiDAR
  adapter isn't implemented yet.
* TUM-VI lidar: the dataset has no LiDAR stream.

## 4. Evaluator checks
`pytest -q` runs 21 tests on synthetic ground truth. They check:
* recovery of a known 0.10 m error under an arbitrary world transform;
* explicitly labelled Sim(3) recovering a scale of 2.0;
* association tolerance;
* exclusion of poses written while lost;
* loss and recovery durations;
* that RPE pairs never span gaps and missing output is never interpolated;
* unrecovered losses and map changes;
* position-only ground truth;
* RPE units;
* proxy labelling;
* the constant-extrinsic diagnostic;
* runner metadata, timeout and crash classification, and batch summaries that include failed runs.

## Remaining blockers
* **Stereo-inertial on LunaPolaris:** the IMU noise model is missing. It needs an Allan-variance
  calibration of the ZED2 IMU, or manufacturer values with provenance. The ZED2 IMU stream is also
  irregular: stamps lie on a 2.5 ms (400 Hz) grid but are published at about 98 Hz on average, with a
  median spacing of 12.5 ms and gaps up to 0.64 s. No camera–IMU time offset is published.
* **GT→camera extrinsic:** the rotation differs by about 4° from the nominal tf chain (see above).
* **Camera–mocap latency:** unknown, so no correction is applied.
* **LiDAR modes:** these need a LiDAR estimator adapter and LunaPolaris LiDAR/LiDAR-IMU
  calibration. `/tf_static` has only nominal Livox extrinsics.
* **Bag read speed:** each run reads metadata and GT from the 33.8 GB sqlite bag with full table
  scans, which take minutes. Caching the derived calibration and GT would speed up re-evaluation.
