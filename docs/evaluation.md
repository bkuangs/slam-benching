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
* Per-frame time for estimators that do not expose it.
