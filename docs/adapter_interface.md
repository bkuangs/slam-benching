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
| `stereo_calibration() -> StereoCalibration \| None` | calibration that matches the images handed to the estimator (`rectified` flag, model, `T_left_right`, baseline, provenance string). |
| `imu_calibration() -> ImuCalibration \| None` | `T_cam0_imu`, rate, `ImuNoise` (with `source` and `verified`), time offset (`None` = unknown). |
| `ground_truth() -> GroundTruth \| None` | poses of `body_frame` in `reference_frame`, integer-ns stamps, `quaternions_xyzw=None` for position-only GT. Never written into estimator inputs. |
| `T_gt_body_sensor(frame) -> 4x4 \| None` | pose of logical sensor frame (`cam0`, `imu`) in the GT body frame; `None` if unknown (accuracy is then reported as unavailable). |
| `validate(mode) -> [str]` | human-readable reasons the mode cannot run (missing streams, calibration, unverified IMU noise). Base class implements the generic checks. |
| `prepare_offline(mode, out_dir) -> PreparedInput` | write the offline format (below) with original timestamps; record every conversion in the manifest; `preprocessing: []` unless explicitly enabled. |
| `provenance() -> dict` | paths, sizes, checksums… stored in `dataset_info.json`. |

`prepare_cached()` wraps `prepare_offline()` with a cache keyed by
`preparation_key()` (adapter name/version, sequence, mode and `input_affecting_options`).

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
| `collect(ctx, returncode, timed_out) -> AlgorithmOutput` | classify outcome (`success`, `init_failure`, `crash`, `timeout`, `missing_output`), write the normalized trajectory and diagnostics. |

### Normalized outputs
* `trajectory.tum` (+ `trajectory.json` sidecar): TUM format, `timestamp tx ty tz qx qy qz qw`,
  seconds with 9 decimals from the original integer-ns stamps; `T_world_sensor` of
  `estimate_frame`; metres; only poses the estimator produced (no interpolation).
* `frame_diagnostics.csv`: `timestamp_ns, tracking_state, tracking_ok, map_id, track_time_s,
  in_final_trajectory`. Empty fields mean *unavailable*, never zero. If an estimator exposes no
  tracking state, do not write the file: the evaluator then reports an explicitly labelled
  output-availability **proxy** instead of coverage.
* `AlgorithmOutput.diagnostics_available` states which diagnostics exist;
  `keyframe_only=True` marks keyframe-only trajectories.

## Run directory
`<output_root>/<name>/<YYYYmmdd-HHMMSS>_<id>/`

| file | content |
|---|---|
| `config.resolved.yaml` | merged config incl. all algorithm defaults |
| `environment.json` | framework git commit/dirty diff, Python packages, host/CPU, Slurm job, estimator `version_info()` |
| `dataset_info.json` | streams, calibration used, provenance, calibration cross-checks |
| `prepared_input.json` | prepared directory + manifest |
| `command.txt`, `stdout.log`, `stderr.log` | exact command and logs |
| `status.json` | status, message, return code, timeout flag, wall time, peak RSS, CPU time, `eval_status` |
| `algorithm/` | native config and raw estimator outputs |
| `algorithm_output.json`, `trajectory.tum`, `frame_diagnostics.csv` | normalized outputs |
| `eval/metrics.json`, `eval/*.png` | metrics and plots |
| `summary.csv` | one-row summary (also produced for batches) |

## Adding a dataset or an algorithm
* Dataset: subclass `DatasetAdapter`, implement the methods above, register it, add a config.
  Run `slambench inspect <config>` to see streams, calibration and per-mode compatibility.
* Algorithm: subclass `AlgorithmAdapter`, pin the estimator (container/commit), emit the
  normalized outputs, register it. The runner and evaluator need no changes.
