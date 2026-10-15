# LunaPolaris: verified facts used by the adapter

Source: Zenodo, one record per sequence (sequence_01 = record 15737938; calibration images =
record 16728838). Licence CC BY 4.0, "© 2025 University of Luxembourg — Database by Dave van der
Meer, SnT – SpaceR". Everything below was read from `sequence_01` itself on 2026-10-10
(`slambench inspect configs/lunapolaris_seq01_orbslam3_stereo.yaml` reproduces most of it).

## Container format
ROS 2 bag, sqlite3 storage, CDR serialization (`sequence_XX/metadata.yaml` + `sequence_XX.db3`).
Read without ROS through `rosbags` (pure Python). Sequence 01: 912.1 s bag duration,
494 324 messages, 33.8 GB, md5 `be6629e277e209de6fb9e479ffd1ec98`.

## Streams (sequence_01)
| logical | topic | type | count | rate | frame_id |
|---|---|---|---|---|---|
| cam0 | `/leo01/zed2/left/image_rect_color` | Image bgra8 640x360 | 13 197 | 14.6 Hz | zed2_left_camera_optical_frame |
| cam1 | `/leo01/zed2/right/image_rect_color` | Image bgra8 640x360 | 13 233 | 14.6 Hz | zed2_right_camera_optical_frame |
| imu | `/leo01/zed2/imu/data_raw` (also `/imu/data`) | Imu | 88 963 | ~98 Hz | zed2_imu_link |
| lidar | `/leo01/livox/lidar` | PointCloud2 | 7 904 | 8.7 Hz | livox_frame |
| (lidar IMU) | `/leo01/livox/imu` | Imu | 16 207 | 17.8 Hz | livox_frame |
| GT | `/vrpn_mocap/ZED2_camera_marker/rectified/pose` | PoseStamped | 27 610 | ~30 Hz | world |

Also present: camera_info (left/right), atm_press, imu/mag, joint_states, /tf, /tf_static,
`/leo01/zed2/left_cam_imu_transform`, robot_description. Camera, IMU and mocap start ~2.6–7.3 s
after the bag starts; max inter-message gaps (bag time) are 2 s for images and 3.4 s for mocap.

## Timestamps
* Images use `header.stamp` (ZED SDK capture time). Left/right stamps are identical, so the adapter
  pairs frames by exact stamp equality and drops unpaired frames (counts in `manifest.json`).
* Bag receive time is ~0.14 s after the header stamp; header stamps are used everywhere, bag time
  never. Values are integer nanoseconds end-to-end.
* Mocap uses its own `header.stamp`. No camera–mocap latency calibration is published; none is
  applied.

## Calibration
* The images are **already rectified** (`image_rect_color`). The matching `camera_info` messages
  (constant for the whole sequence) give fx = fy = 263.270, cx = 327.288, cy = 177.833, zero
  distortion, identity R, and right P[0,3] = −31.547 → baseline 0.11983 m. This is the calibration
  passed to estimators (`calibration_source: bag_camera_info`).
* The Zenodo JSON (`calibration_results.json`) is an independent checkerboard calibration with its
  own rectification (rectified fx = 268.49) whose translation is in checkerboard squares
  (T_x = −1.1995 squares × 0.10 m = 0.11995 m). Its baseline agrees with the bag to 0.10 %, but its
  intrinsics do not describe the recorded images, so it is only a cross-check
  (`dataset_info.json → calibration_crosscheck`).
* Camera–IMU extrinsic: `/leo01/zed2/left_cam_imu_transform` (zed2_left_camera_frame →
  zed2_imu_link, constant), composed with /tf_static to the optical frame.
* **IMU noise densities and bias random walks are not published.** `sensor_msgs/Imu` covariances
  are per-sample variances without random walk. Inertial modes are rejected until verified values
  are supplied (`dataset.options.imu_noise` with `source` and `verified: true`).
* Camera–IMU time offset: not published.
* ZED2 IMU timing (header stamps): on a 2.5 ms (400 Hz) grid but irregularly published, ~98 Hz on
  average, median spacing 12.5 ms, max gap 0.64 s; `/imu/data` and `/imu/data_raw` share stamps.

## Ground truth and frames
* OptiTrack/VRPN pose of rigid body `ZED2_camera_marker`, position + orientation, metres.
* `/rectified/pose` equals the `/tf` world → `vrpn_mocap/ZED2_camera_marker` transform, and with it
  the rover base z-axis points world-up (mean z-axis component 0.998). The raw `/pose` topic is a
  cyclic axis permutation (Y-up world) whose body axes are *not* those of the tf tree. The adapter
  uses `/rectified/pose`.
* `/tf_static` connects `vrpn_mocap/ZED2_camera_marker` → `leo01/base_footprint` → … →
  `zed2_left_camera_optical_frame`. Lever arm marker → left optical: 0.070 m. The chain is
  internally consistent (marker vs. `optitrack_marker_link` differ by 0.4 mm / 0.008°), but these
  are nominal URDF values, not a hand–eye calibration. The ORB-SLAM3 stereo run on sequence_01
  shows a near-constant 4.1° rotation (mostly about the camera x-axis) between the estimated and the
  GT-derived camera frame, so the nominal camera tilt is likely off by about this amount. It is
  reported as a diagnostic and not corrected (see docs/results/milestone1.md).
* GT never enters estimator inputs; the evaluator converts it to the estimate frame.

## Conversion to the slambench offline format (`prepare_offline`)
* PNG, lossless; bgra8 → bgr8 by dropping alpha (alpha opacity is checked and counted).
* No resizing, colour conversion, equalisation or undistortion. `preprocessing: []` in manifest,
  except for the optional row shift below.
* **sequence_11 stereo row misalignment.** Its "rectified" images are not row-aligned: features in the
  right image sit a constant 1.65 px lower than in the left (fit dy = y_left − y_right = −1.647 px,
  x/y-dependent terms < 0.2 px over the image; sequence_01: −0.13 px). Its camera_info has identical
  left/right P rows, so the offset is undocumented. Same-row stereo matchers (RTAB-Map block
  matching) then find almost no depth and odometry is lost from the start; ORB-SLAM3 tolerates it.
  `dataset.options.right_row_shift_px` (px, negative = up; default 0) shifts every right image by a
  bilinear warp at preparation, is part of the cache key, and is recorded in the manifest
  `preprocessing`. The seq11 configs set −1.63. Measure other sequences with
  `scripts/measure_stereo_row_offset.py <prepared_stereo_dir>` and use the printed constant term.
* Cached under `<work_dir>/prepared/lunapolaris/<sequence>/<mode>_<key>/`.
