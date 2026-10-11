# Evaluation methodology

Implemented in `slambench/evaluation/metrics.py`; settings live under `evaluation:` in the
experiment config and are copied into `eval/metrics.json → settings`.

| setting | default | meaning |
|---|---|---|
| `alignment` | `se3` | `se3` rigid Umeyama (metric stereo / inertial), `sim3` scale-corrected (explicit opt-in, labelled "Sim(3) scale-corrected"; for scale-ambiguous monocular), `none` |
| `association_max_dt_s` | 0.02 | max \|t_est − t_gt\| for nearest-neighbour association |
| `segment_max_gap_s` | 0.5 | consecutive matched samples farther apart start a new segment |
| `rpe` | `[{delta: 1, unit: m}, {delta: 1, unit: s}]` | RPE intervals; units `m` (GT path length), `s`, `frames` |
| `valid_states` | `tracking_ok` | evaluate only poses from frames the estimator reports as tracked OK (`all_output` to disable) |
| `session_min_matched` | 20 | map sessions with fewer matched samples are listed but not pooled (their alignment would overfit) |

## Online and final trajectories
A run can declare two trajectories (`algorithm_output.json → trajectories`), and each is evaluated
separately in its own sensor frame (`accuracy.online`, `accuracy.final`):

| kind | file | question it answers |
|---|---|---|
| `online` | `trajectory_online.tum` | How accurate was the pose available when this image was processed? |
| `final` | `trajectory.tum` | How accurate is the reconstruction after later mapping and loop-closure corrections? |

* ORB-SLAM3 `online` = the camera pose returned by `Track*()` right after each tracking-OK frame
  (`frame_log.csv`, original timestamps; `cam0` frame in every mode). It already contains loop
  closures that happened *before* that frame, never later ones, so it is not pure odometry.
  `final` = `SaveTrajectoryEuRoC` after shutdown (`imu` frame in inertial modes, largest map only).
  The two therefore use different GT-body→sensor transforms.
* **Coordinate sessions.** An online pose is expressed in the world frame of the map active when
  it was emitted, and that frame can change: a new map after tracking loss, a **map merge** (in
  ORB-SLAM3 the merged map takes over the old map *id*, so ids alone miss it), or an in-place
  re-expression (IMU initialisation / inertial merge rescale and rotate the map). The driver
  therefore logs a per-frame `coord_session` (frame_diagnostics `coordinate_session`) that changes
  whenever the active Map object, its id, its origin keyframe or that keyframe's pose changes; loop
  closure and bundle adjustment keep the origin keyframe fixed, so they do not start a session. The
  map frame is read before and after each tracking call; if it changed in between, the frame is
  marked `-1` (ambiguous) and excluded (`n_poses_ambiguous_frame`).
  Each session is associated, aligned and scored on its own; nothing is concatenated under one
  alignment, and RPE never pairs samples from different sessions. `accuracy.<kind>.sessions[]` lists
  each session (session id, map ids, poses, span, fraction of the input duration, its own alignment
  and metrics). The top-level ATE/RPE pool the per-sample/per-pair errors of sessions with at least
  `session_min_matched` matches. Runs from before `coord_session` existed fall back to map ids with a
  `session_note`. `final` uses one session (`coordinate_sessions: single`).
* **Failed runs.** Evaluation also runs for crashed or timed-out runs: the online prefix written
  before a crash is scored while the run keeps its failed status (`final` is then unavailable).
* **Label.** All accuracy numbers are *aligned benchmark accuracy*: the alignment is fitted over
  each whole session with all of its ground truth, including samples after a pose was emitted. They
  describe trajectory consistency, not the causal error a vehicle saw at that instant.
* Summary columns are prefixed `online_` / `final_`; RPE columns are named by interval, e.g.
  `online_rpe_1s_trans_rmse_m`, `final_rpe_1m_rot_rmse_deg`.

## Pipeline
1. **Frames.** Ground truth gives the pose of the GT body (e.g. mocap marker) in the mocap world.
   It is converted to the estimate's sensor frame: `T_world_sensor = T_world_body · T_body_sensor`
   with `T_body_sensor` from the dataset adapter (LunaPolaris: /tf_static chain; TUM-VI:
   camchain `T_cam_imu`). If the relation is unknown, accuracy is reported as unavailable.
2. **Valid samples.** With estimator tracking states, only poses from `tracking_ok` frames count;
   the number of excluded output poses is reported (ORB-SLAM3 writes RECENTLY_LOST frames).
3. **Association.** Nearest GT sample within `association_max_dt_s`; no interpolation of
   either trajectory; |Δt| statistics are reported.
4. **Alignment** over all matched samples (evo Umeyama). Rotation and scale are reported.
5. **ATE.** Translation error norm (m): RMSE/mean/median/std/min/max. Rotation error (deg) only
   if both trajectories have orientation; otherwise `{"available": false, "reason": ...}`.
6. **Segments.** Break at time gaps > `segment_max_gap_s`, at any non-OK frame between two
   samples, and at estimator map changes.
7. **RPE.** For each sample i, the first j in the *same segment* that is `delta` away (GT path
   length, time, or frames). Error `E = (T_gt,i⁻¹ T_gt,j)⁻¹ (T_est,i⁻¹ T_est,j)`; translation (m)
   and rotation (deg). Pairs never span a tracking gap. Position-only GT falls back to the
   world-frame displacement difference and marks rotation unavailable.
8. **Evaluated coverage** next to every accuracy number: matched samples, segments, evaluated
   duration (sum of segment spans) and its fraction of the input duration, fraction of input frames
   with any GT nearby.

## Diagnostics (`accuracy.diagnostics`)
* `constant_body_rotation_offset`: the mean rotation between the GT-derived and estimated sensor
  frames, plus the rotation RMSE that would remain without it. It is never applied. A large offset
  with a small residual points to a GT-body→sensor extrinsic error. Such an error dominates rotation
  ATE and inflates translational RPE by about delta·sin(angle).

## Tracking coverage (`coverage`)
From estimator tracking states (`source: estimator_tracking_state`):
* `frame_coverage` = tracked-OK frames / input frames; `time_coverage` = time owned by OK frames /
  sequence duration (frame i owns [t_i, t_{i+1})).
* `time_to_first_ok_s` (initialisation), `n_tracking_loss_events` (OK → not OK after
  initialisation), per-event recovery duration (loss → next OK), `n_unrecovered`,
  `n_recovered_in_new_map`, `n_maps`, and the fraction of frames present in the final trajectory.

If an estimator has no tracking state, a **proxy** (`source: PROXY_output_availability`) is
reported with a warning; keyframe-only outputs are flagged. Path-length ratios are never used
as completion.

## Compute cost (`timing`)
* `wall_time_s` of the estimator process, `peak_rss_mb` (sum of RSS over the process tree,
  sampled; method string recorded), CPU user/system time.
* `per_frame_time_s`: wall time of each tracking call (tracking thread only; mapping and loop
  closing run concurrently in ORB-SLAM3).
* `realtime`: fraction of frames whose tracking time exceeds the recorded frame period, and whether
  the p95 fits within it. With `playback_rate: 1.0` the input is throttled to real time, so wall
  time ≈ sequence duration and is *not* a throughput number; `tracking_throughput_fps`
  (= frames / Σ tracking time) is the offline throughput of the tracking thread.

## Not implemented / reported as unavailable
* Orientation metrics with position-only GT.
* Camera–mocap time-offset estimation (none is applied; LunaPolaris publishes none).
* Camera–IMU time offsets: a configured non-zero offset is rejected (not applied to inputs yet).
* Causal (un-aligned / prefix-aligned) online error; online accuracy is aligned benchmark accuracy.
* Per-frame time for estimators that do not expose it.
