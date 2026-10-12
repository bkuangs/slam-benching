# SubT-MRS SmokeRoom: facts used by the adapter

Source: SubT-MRS dataset (AirLab, CMU), released for the ICCV'23 SLAM Challenge
(<https://superodometry.com/datasets>, challenge repo
<https://github.com/yuanjun-gao/ICCV2023_SLAM_Challenge>). Licence CC BY 4.0.
SmokeRoom is a bonus sequence of the visual-inertial and sensor-fusion tracks
(submission name `SubT_MRS_Hawkins_Smoke_Handheld`): Hawkins site, RC7 handheld payload,
visually degraded by smoke, 104.8 m, ~419 s. Adapter: `slambench/datasets/subtmrs.py`
(`dataset.name: subtmrs`). The calibration and ground-truth formats below were verified on
2026-10-10. Folder layout, image formats and timestamps were checked on the remote folder zip
through HTTP range reads.

## Download
`sbatch scripts/download_subtmrs_smokeroom.sbatch [~/scratch/datasets/subtmrs]` fetches and extracts:

| file | Google Drive id | content |
|---|---|---|
| `SubT_MRS_Hawkins_Smoke_Handheld.zip` (8.6 GB) | `140UJgOCa-2itHUZQfwOK1zISD8XRGw51` | folder release → `SmokeRoom_Folder/` |
| `SubT_MRS_Hawkins_Smoke_Room_GT_IMU.zip` (13 MB) | `1YFm6F-c5NI84zKNqyCWA-_c-KABSKF4y` | reference trajectory → `SubT_MRS_Smoke_Room/` |
| `SubT_MRS_Hawkins_Smoke_Handheld_Intrinsics.yaml` | `1Q0JiqiIgGZ-7DZKZDNJ68F7rpC2rrLtT` | RGB (MEI) + thermal (pinhole) intrinsics |
| `SubT_MRS_Hawkins_Smoke_Handheld_Extrinsics.yaml` | `1HjWlRVQQvgrFGlgRxcczRt92Xy_P-5Ij` | camera→IMU extrinsics |

A 10 GB ROS 1 bag release (`1B7UFXWiXHQ_FsDiV_XJYjRQ4lhHGj-0m`) exists as well; the adapter uses the
folder release. Expected layout (`dataset.root` = `~/scratch/datasets/subtmrs`):
```
SubT_MRS_Hawkins_Smoke_Handheld/
  SmokeRoom_Folder/cam_0/{0..10289}.png, timestamps.txt
  SmokeRoom_Folder/thermal/{...}.png, timestamps.txt
  SmokeRoom_Folder/imu/imu_data.csv
  SubT_MRS_Smoke_Room/integrated_odom/odometry_data.csv
  SubT_MRS_Smoke_Room/imu/imu_data.csv          (identical to SmokeRoom_Folder/imu)
  SubT_MRS_Hawkins_Smoke_Handheld_{Intrinsics,Extrinsics}.yaml
```
Every path can be overridden with `dataset.options.{data_dir,gt_csv,intrinsics_yaml,extrinsics_yaml}`
(relative to `<root>/<sequence>` or absolute). This also lets the adapter read other SubT-MRS
sequences that use the same folder format.

## Streams
| logical | source | format | count | rate |
|---|---|---|---|---|
| cam0 | `cam_0/<i>.png` + line `i` of `cam_0/timestamps.txt` (ns) | 640x480 8-bit RGB PNG | 10 290 | ~24.5 Hz |
| imu | `imu/imu_data.csv` | `timestamp, q_x..q_w, ang_vel_xyz (rad/s), lin_acc_xyz (m/s², gravity incl.)` | 83 768 | 200 Hz |
| (thermal) | `thermal/` | 640x512 16-bit PNG | 12 510 | 30 Hz |

The release has only **one RGB camera**, so stereo modes are rejected. The IMU quaternion columns
hold an on-board orientation estimate and are ignored. The thermal camera is not yet an estimator
stream: it would need a 16→8-bit conversion, which is preprocessing, and its published extrinsic
has a bottom row of zeros.
The RGB image shows the RC car body and the shadow it casts with its own light at the bottom of
every frame. Both are static in the image. Option `crop_bottom_px` (default 0, recorded under
`preprocessing`) removes those rows from the pinhole image. fx, fy, cx and cy stay unchanged, so
the model is still exact. Edge persistence over 355 frames puts the car outline and shadow below
row ~345 of the 640x480 virtual image, so the SmokeRoom configs use `crop_bottom_px: 140` (640x340).
Without the crop, ORB-SLAM3 mono reported 72 % of frames as tracked but estimated almost no
motion (Sim(3) scale 61.8, ATE RMSE 7.3 m; run `20261010-225344_2e6404`).

## Camera model → virtual pinhole
`rgb_camera` uses the MEI unified omnidirectional model (`xi = 1.583`, radial-tangential
`k1, k2, p1, p2`, `gamma1 = 758.3`, `gamma2 = 676.6`, `u0 = 318.3`, `v0 = 239.8`). ORB-SLAM3 accepts
only pinhole and Kannala-Brandt models. `prepare_offline()` therefore resamples every frame into an
ideal pinhole camera with `cv2.remap` (bilinear by default). The projection follows CamOdoCal
`CataCamera::spaceToPlane` / Kalibr `omni-radtan`.
* Defaults: 640x480, `fx = fy = mean(gamma1, gamma2)/(1+xi) ≈ 277.8 px` (keeps centre resolution),
  principal point at the image centre, zero distortion. FOV ≈ 98°x82°, and every output pixel
  samples inside the source image.
* The virtual camera has the same orientation and optical centre as the original one, so the
  extrinsics are unchanged.
* Options `pinhole_width`, `pinhole_height`, `pinhole_focal`, `interpolation` change the prepared
  input. They are part of the cache key together with the intrinsics file hash. The manifest
  records the conversion (`mei_to_pinhole_remap`), the MEI parameters and the valid-pixel fraction.

## Extrinsics and frames
`rgb_camera_to_imu` is **T_imu_cam** (p_imu = T · p_cam). Its axes confirm this: the optical axis maps
to IMU +x (forward), optical x (right) to IMU −y, optical y (down) to IMU −z, and the camera sits
0.195 m ahead of the IMU. The IMU frame is x forward, y left, z up (challenge convention).
`T_cam0_imu = inv(T_imu_cam)`; `T_gt_body_sensor("cam0") = T_imu_cam`.

## Ground truth
`integrated_odom/odometry_data.csv`: `timestamp (ns), x, y, z, q_x, q_y, q_z, q_w` = pose of the
**IMU** at 200 Hz (83 667 samples, starting at the origin about 0.47 s after the first frame). It is
the challenge's LiDAR-based reference trajectory, not motion capture, so report accuracy against it
as "reference trajectory". Use `association_max_dt_s: 0.005`.

## IMU noise and timing
No noise densities or bias random walks are published. The dataset notes say an Allan-variance
calibration was done but give no values. Inertial modes are rejected until
`dataset.options.imu_noise` is supplied with a source and `verified: true`. The sensors are
hardware-synchronised (PPS and FPGA, ≤ 3 ms between sensors), so `time_offset_s` defaults to 0.
A non-zero `time_offset_s` is rejected for inertial modes: slambench does not shift IMU timestamps
yet, so recording an offset would misdescribe the inputs actually used.

## Configs
* `configs/subtmrs_smokeroom_orbslam3_mono.yaml`: ORB-SLAM3 mono, Sim(3) alignment.
* `configs/subtmrs_smokeroom_orbslam3_mono_clahe.yaml`: the same with `clahe: true` (grayscale +
  CLAHE(3.0, 8x8) in the estimator image), for comparing contrast enhancement under smoke. Other
  strengths: `--set algorithm.settings.clahe_clip_limit=1.5 --set name=..._clahe_clip1.5`.
  Results of repeated runs: `docs/results/smokeroom_clahe_repeats.md`.
* `configs/subtmrs_smokeroom_orbslam3_mono_inertial.yaml`: rejected until IMU noise is verified.
