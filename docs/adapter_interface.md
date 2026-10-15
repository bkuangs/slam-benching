# Adapter interface

The framework separates three concerns. Only adapters contain dataset- or estimator-specific code.

```
config.yaml ──► runner ──► DatasetAdapter.validate/prepare ──► PreparedInput (offline format)
                      └──► AlgorithmAdapter.validate/prepare/command/collect ──► trajectory.tum
                                                                              frame_diagnostics.csv
                      └──► evaluation.evaluate_run (uses DatasetAdapter.ground_truth / T_gt_body_sensor)
```

Register new adapters in `slambench/datasets/__init__.py::DATASETS` and
`slambench/algorithms/__init__.py::ALGORITHMS`.

## Sensor modes
`SensorMode` (slambench/types.py): `mono`, `mono_inertial`, `stereo`, `stereo_inertial`, `lidar`,
`lidar_inertial`. Each mode needs a fixed set of logical streams (`REQUIRED_STREAMS`): `cam0`,
`cam1`, `imu`, `lidar`. Dataset adapters map these names to their own topics/files.

## DatasetAdapter (`slambench/datasets/base.py`)
Constructor: `(root: Path, sequence: str, options: dict)`; `options` comes from `dataset.options`.

| method | contract |
|---|---|
| `sequence_info() -> SequenceInfo` | streams actually present (logical name → `StreamInfo`), duration, GT availability. |
| `stereo_calibration() -> StereoCalibration \| None` | calibration that matches the images handed to the estimator (`rectified` flag, model, `T_left_right`, baseline, provenance string). Single-camera datasets set `right=None` (mono modes only). |
| `imu_calibration() -> ImuCalibration \| None` | `T_cam0_imu`, rate, `ImuNoise` (with `source` and `verified`), time offset (`None` = unknown). A non-zero offset is rejected by `validate()` because slambench does not apply offsets to inputs yet. |
| `ground_truth() -> GroundTruth \| None` | poses of `body_frame` in `reference_frame`, integer-ns stamps, `quaternions_xyzw=None` for position-only GT. Never written into estimator inputs. |
| `T_gt_body_sensor(frame) -> 4x4 \| None` | pose of logical sensor frame (`cam0`, `imu`) in the GT body frame; `None` if unknown (accuracy is then reported as unavailable). |
| `validate(mode) -> [str]` | human-readable reasons the mode cannot run (missing streams, calibration, unverified IMU noise). Base class implements the generic checks. |
| `prepare_offline(mode, out_dir) -> PreparedInput` | write the offline format (below) with original timestamps; record every conversion in the manifest; `preprocessing: []` unless explicitly enabled. |
| `provenance() -> dict` | paths, sizes, checksums… stored in `dataset_info.json`. |
| `source_files() -> [Path]` | raw data and calibration files/directories that determine `prepare_offline()` output; fingerprinted into the cache key. |

`prepare_cached()` wraps `prepare_offline()` with a cache keyed by `preparation_key()`: adapter
name/version, the resolved dataset root, sequence, mode, `input_affecting_options` and a fingerprint
of every `source_files()` entry (content hash for files ≤ 4 MB, size + mtime for larger files and
for every file inside a directory). The inputs are recorded in `manifest.json →
preparation_inputs`, so a different or updated copy of a sequence never reuses stale inputs.

### slambench offline format
```
<dir>/frames.csv     timestamp_ns,left_path,right_path      (paths relative to <dir>; right empty for mono)
<dir>/imu.csv        timestamp_ns,wx,wy,wz,ax,ay,az         (rad/s, m/s^2; inertial modes)
<dir>/manifest.json  provenance, counts, dropped samples, conversions, preprocessing list
```
Direct bag replay is also allowed: an algorithm adapter may ignore the offline format and consume
`dataset.root/sequence` itself (declare this in the adapter); the dataset adapter still provides
calibration and GT.

## AlgorithmAdapter (`slambench/algorithms/base.py`)
Constructor: `(options: dict)` from `algorithm.options` (installation paths, container image…).
`algorithm.settings` are estimator parameters and are resolved/validated by `resolve_settings()`.

| member | contract |
|---|---|
| `name`, `supported_modes` | registry name, set of `SensorMode`. |
| `resolve_settings(settings)` | fill defaults, reject unknown keys; result is saved in `config.resolved.yaml`. |
| `validate(mode, dataset, settings) -> [str]` | unsupported mode, missing installation, unsupported camera model… |
| `estimate_frame(mode) -> str` | logical frame of output poses (`cam0`, `imu`). |
| `version_info() -> dict` | pinned commit, image digest, library versions; raise if the installed version differs from the pin. |
| `prepare(ctx)` | write native config into `ctx.algo_dir` from dataset calibration + settings. |
| `command(ctx) -> [str]` | exact argv (saved to `command.txt`). |
| `collect(ctx, returncode, timed_out) -> AlgorithmOutput` | classify outcome (`success`, `init_failure`, `crash`, `timeout`, `missing_output`), write the normalized trajectories and diagnostics. Should not raise on partial outputs (e.g. after a crash); an exception is recorded as `collect_failed`. |

### Normalized outputs
* `trajectory.tum` (+ `trajectory.json` sidecar): TUM format, `timestamp tx ty tz qx qy qz qw`,
  seconds with 9 decimals from the original integer-ns stamps; `T_world_sensor` of
  `estimate_frame`; metres; only poses the estimator produced (no interpolation).
* `frame_diagnostics.csv`: `timestamp_ns, tracking_state, tracking_ok, map_id, track_time_s,
  in_final_trajectory, coordinate_session`. Empty fields mean *unavailable*, never zero. If an estimator exposes no
  tracking state, do not write the file: the evaluator then reports an explicitly labelled
  output-availability **proxy** instead of coverage.
* `AlgorithmOutput.trajectories`: one `TrajectoryOutput` per normalized trajectory with `kind`
  (`online` = pose emitted right after each frame; `final` = exported after shutdown), `path`,
  `frame` (logical sensor frame; online and final may differ), `coordinate_sessions` (`single`;
  `per_frame` when each pose is in the world frame given per frame by the diagnostics column
  `coordinate_session`, -1 = ambiguous; `map_id` to use map ids) and `keyframe_only`.
  The legacy `trajectory`/`estimate_frame` fields are read as the final trajectory.
* `AlgorithmOutput.artifacts` (optional): `tracked_features` (`features.bin`: per-frame keypoints in
  input-image pixels with status unmatched / inlier / outlier), `sparse_map_points`
  (`map_points.csv`) and `sparse_map_keyframes` (`keyframes.csv`); rows with `in_final_frame=1` are
  in the final trajectory's world frame. Formats: `slambench/artifacts.py`. Used by the replay viewer.
  ORB-SLAM3 also writes `place_recognition` (`place_recognition.csv`: every loop/merge check, its
  candidates and the stage at which each was rejected; columns documented in patch 0005), and its
  `frame_log.csv` records the active map's keyframe and map-point count per frame.
  RTAB-Map (stereo) writes map artifacts instead of a sparse map: `rtabmap_db` (`rtabmap.db`, the
  reusable localization map: images, visual words, pose graph, per-node local occupancy grids),
  `map_nodes` (`nodes.csv`: node id, map id, stamp, cam0 odometry pose and final optimized pose),
  `map_links` (`links.csv`: final graph constraints and their type), `dense_cloud`
  (`rtabmap_cloud.ply`), `octomap` (`rtabmap_octomap.bt`: 3D occupancy with occupied / free /
  unknown space, built from ray-traced stereo local grids) and `octomap_cloud`
  (`rtabmap_octomap.ply`), all assembled with the final optimized poses by `rtabmap-export --opt 2`.
  Its final trajectory gives every frame with valid odometry a pose,
  `T_opt(node) T_odom(node)^-1 T_odom(frame)` with the latest earlier final-graph node; its online
  trajectory is RTAB-Map's map correction times odometry at that frame, and `coordinate_session`
  changes when the optimized graph's root node changes (new or joined maps). The adapter sets
  `Stereo/OpticalFlow=false` by default (setting `stereo_optical_flow`): with RTAB-Map's default KLT
  stereo correspondences, LunaPolaris odometry absorbed forward motion as ~30°/m of pitch drift.
  **Map reuse** (setting `map_run: <mapping run dir>`): the run copies that run's `rtabmap.db`,
  checks that all RTAB-Map parameters equal the mapping run's except `Mem/IncrementalMemory=false`
  and `Mem/InitWMWithAllNodes=true`, and the driver (`--localize`) localizes against the map in a
  fresh process without saving it back. Only an online trajectory (`coordinate_sessions: single`,
  the map's frame) is produced, starting at the first accepted loop-closure / proximity link;
  earlier frames carry only RTAB-Map's prior (last pose saved in the map) and have tracking state
  `not_localized`. `extra.map_reuse` records the map database hash before/after (the run fails if
  it changed) and `map_frame_check`, which compares the loaded node poses with the mapping run's
  final graph (the run fails unless they are identical, since evaluation relies on that frame).
* `AlgorithmOutput.diagnostics_available` states which diagnostics exist.

## Run directory
`<output_root>/<name>/<YYYYmmdd-HHMMSS>_<id>/`

| file | content |
|---|---|
| `config.resolved.yaml` | merged config incl. all algorithm defaults |
| `environment.json` | framework git commit/dirty diff, Python packages, host/CPU, Slurm job, estimator `version_info()` |
| `dataset_info.json` | streams, calibration used, provenance, calibration cross-checks |
| `prepared_input.json` | prepared directory + manifest |
| `command.txt`, `stdout.log`, `stderr.log` | exact command and logs |
| `status.json` | written at creation (`status: running`) and updated per `stage`; final status, message, `estimator_outcome`, return code, timeout flag, wall time, peak RSS (or `peak_rss_unavailable_reason`), CPU time, `eval_status`, `error` |
| `error.json` | for harness-side failures: stage, exception type, message, traceback |
| `algorithm/` | native config and raw estimator outputs |
| `algorithm_output.json`, `trajectory.tum` (final), `trajectory_online.tum`, `frame_diagnostics.csv` | normalized outputs |
| `eval/metrics.json`, `eval/*.png` | metrics and plots |
| `summary.csv` | one-row summary (also produced for batches) |

## Run status and stages
Stages (`status.json → stage`): `resolve`, `environment`, `validate`, `dataset_info`,
`prepare_input`, `algorithm_prepare`, `execute`, `collect`, `evaluate`, `done`.

| status | meaning |
|---|---|
| `success`, `init_failure`, `crash`, `timeout`, `missing_output` | estimator outcome, classified by the algorithm adapter |
| `config_rejected` | invalid config, failed version check, or incompatible dataset/mode/calibration |
| `prepare_failed` | input conversion or native configuration failed |
| `collect_failed` | the estimator finished (see `estimator_outcome`) but slambench could not parse its outputs |
| `harness_error` | unexpected slambench failure (e.g. estimator executable missing) |

An estimator segfault is `crash`; a Python error while parsing its output is `collect_failed`.
Memory monitoring is best effort and never changes the outcome. A run killed from outside keeps
`status: running` with the last stage. Batches run every experiment and write one summary row each.

## Adding a dataset or an algorithm
* Dataset: subclass `DatasetAdapter`, implement the methods above (including `source_files()`),
  register it, add a config.
  Run `slambench inspect <config>` to see streams, calibration and per-mode compatibility.
* Algorithm: subclass `AlgorithmAdapter`, pin the estimator (container/commit), emit the
  normalized outputs, register it. The runner and evaluator need no changes.
