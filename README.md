# slam-benching

Reusable experiment and evaluation framework for comparing SLAM sensor suites (monocular+IMU,
stereo+IMU, LiDAR+IMU) under challenging terrain, lighting and obscurants, including compute cost.
Existing estimators are reused unmodified (or with recorded minimal patches) behind adapters; this
repository handles dataset adaptation, execution, output normalisation and evaluation.

**Status (first milestone):** ORB-SLAM3 (pinned) × LunaPolaris sequence_01 runs end to end in
stereo mode: 100 % of frames tracked and ATE RMSE 0.031 m (SE(3)). See
`docs/results/milestone1.md` for full results and caveats.
Stereo-inertial is implemented in the adapters but rejected for LunaPolaris until verified IMU
noise parameters exist (see `docs/datasets/lunapolaris.md`). The estimator installation is
verified on TUM-VI `room1`.

```
slambench/
  types.py, trajectory.py, tf.py      shared types, normalized trajectory format, static TF tree
  datasets/   base.py                 DatasetAdapter interface + offline format writers
              lunapolaris.py          ROS 2 bag adapter (rosbags, no ROS install needed)
              tumvi.py                TUM-VI (EuRoC/ASL export) adapter, used for installation checks
              subtmrs.py              SubT-MRS folder release (SmokeRoom; MEI -> virtual pinhole, mono only)
  algorithms/ base.py                 AlgorithmAdapter interface + normalized diagnostics columns
              orbslam3.py             ORB-SLAM3 via Apptainer image
  runner.py, config.py, cli.py        experiment execution, YAML configs, CLI
  evaluation/ metrics.py, evaluate.py, plots.py, summary.py
third_party/orbslam3/                 pinned container definition, headless driver, patch
configs/                              example experiment and batch configs
scripts/slurm_slambench.sbatch        run any slambench command on a PACE compute node
scripts/download_subtmrs_smokeroom.sbatch  fetch SubT-MRS SmokeRoom data + calibration + GT
docs/                                 adapter interface, evaluation methodology, dataset notes
tests/                                evaluator / runner tests with synthetic ground truth
```

## Setup (PACE ICE; adapt paths elsewhere)
Raw data, converted inputs, containers and run outputs stay outside the repository, by default in
`~/scratch/slam-bench-work` (`work_dir`, `output_root` in configs).

```bash
# 1. Python environment (Python >= 3.9)
python3 -m venv ~/scratch/slam-bench-work/venv
source ~/scratch/slam-bench-work/venv/bin/activate
pip install -e '.[test]'
pytest -q                                   # evaluator/runner checks, no data needed

# 2. ORB-SLAM3 image (pinned commit 4452a3c, Pangolin v0.6, Ubuntu 20.04; ~12 min on 8 cores)
#    Needs apptainer with --fakeroot. Run on a compute node:
sbatch -p ice-cpu -c 8 --mem=32G -t 02:00:00 third_party/orbslam3/build_container.sh
#    -> ~/scratch/slam-bench-work/containers/orbslam3_4452a3c.sif (versions in /opt/slambench/versions.txt)

# 3. Data
#    LunaPolaris sequence_01 (33.8 GB, Zenodo record 15737938) -> ~/scratch/LunaPolaris/sequence_01/
#      metadata.yaml + sequence_01.db3; optional calibration_results.json (record 16728838) one level up.
#    TUM-VI room1 (installation check, 1.7 GB):
#      https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar
#      -> ~/scratch/datasets/tumvi/dataset-room1_512_16/
#    SubT-MRS SmokeRoom (ICCV'23 SLAM Challenge, 8.6 GB; mono / mono_inertial only):
#      sbatch scripts/download_subtmrs_smokeroom.sbatch
#      -> ~/scratch/datasets/subtmrs/SubT_MRS_Hawkins_Smoke_Handheld/  (see docs/datasets/subtmrs.md)
```

Large downloads were killed on PACE login nodes; run them as Slurm jobs.

## Running
```bash
slambench inspect configs/lunapolaris_seq01_orbslam3_stereo.yaml   # streams, calibration, per-mode compatibility
sbatch -c 8 --mem=32G -t 03:00:00 scripts/slurm_slambench.sbatch \
       run configs/lunapolaris_seq01_orbslam3_stereo.yaml           # convert (cached), run, evaluate
sbatch scripts/slurm_slambench.sbatch batch configs/batch_example.yaml   # several configs + summary CSV
sbatch scripts/slurm_slambench.sbatch batch configs/rejection_examples.yaml  # incompatible configs -> config_rejected
slambench evaluate <run_dir> --set evaluation.alignment=sim3      # re-evaluate with other settings
slambench summarize ~/scratch/slam-bench-work/runs -o summary.csv # aggregate all runs (failed ones too)
```
Any config value can be overridden: `--set algorithm.settings.loop_closure=false`.

### Experiment config
```yaml
name: lunapolaris_seq01_orbslam3_stereo
output_root: ~/scratch/slam-bench-work/runs
work_dir: ~/scratch/slam-bench-work            # prepared-input cache
dataset:   {name: lunapolaris, root: ~/scratch/LunaPolaris, sequence: sequence_01, options: {}}
algorithm:
  name: orbslam3
  mode: stereo                                 # mono | mono_inertial | stereo | stereo_inertial | lidar | lidar_inertial
  options:  {container: ~/scratch/slam-bench-work/containers/orbslam3_4452a3c.sif}
  settings: {orb_n_features: 1200, loop_closure: true, clahe: false, playback_rate: 1.0}
execution:  {timeout_s: 3600}
evaluation: {alignment: se3, association_max_dt_s: 0.02, rpe: [{delta: 1.0, unit: m}, {delta: 1.0, unit: s}]}
```
Preprocessing (e.g. `clahe`) is off by default and recorded when enabled. Loop closure is an
explicit setting recorded in every summary row; keep it constant within a comparison.
`playback_rate: 1.0` feeds frames at the recorded rate (estimator threads behave as online);
`0` feeds as fast as possible.

### What a run directory contains
`config.resolved.yaml`, `environment.json` (git commit + diff, packages, host, Slurm job,
container sha256 and library versions), `dataset_info.json`, `prepared_input.json`,
`command.txt`, `stdout.log`, `stderr.log`, `status.json` (status, stage, estimator outcome,
return code, wall time, peak RSS, CPU time; written from the start), `error.json` (harness
failures), `algorithm/` (native config + raw outputs), `trajectory.tum` (final),
`trajectory_online.tum` (pose available at each frame),
`frame_diagnostics.csv`, `eval/metrics.json`, `eval/trajectory.png`,
`eval/error_and_tracking.png`, `summary.csv`. Status values: `success`, `config_rejected`,
`prepare_failed`, `init_failure`, `crash`, `timeout`, `missing_output` (estimator outcomes), plus
`collect_failed` and `harness_error` (slambench-side failures); evaluation problems are reported
separately in `eval_status`. Online and final trajectories are evaluated separately
(`online_*` / `final_*` summary columns; see `docs/evaluation.md`).

## Documentation
* `docs/adapter_interface.md` — dataset/algorithm adapter contracts, offline input format,
  normalized outputs, how to add a dataset or estimator.
* `docs/evaluation.md` — association, frames, alignment, ATE/RPE, coverage, timing.
* `docs/datasets/lunapolaris.md` — verified LunaPolaris topics, timestamps, calibration, frames.
* `docs/datasets/subtmrs.md` — SubT-MRS SmokeRoom layout, MEI→pinhole conversion, extrinsics, GT.
* `docs/results/milestone1.md` — what ran, what was verified, blockers.

## Licences
LunaPolaris: CC BY 4.0, © 2025 University of Luxembourg — Database by Dave van der Meer,
SnT – SpaceR. ORB-SLAM3: GPLv3 (built inside the container, not redistributed here).
TUM-VI: CC BY 4.0. SubT-MRS: CC BY 4.0 (AirLab, CMU).
