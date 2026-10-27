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
              omnilrs.py              OmniLRS/Isaac Sim rendered stereo sequences (exact calibration + GT)
  viz/        rerun_export.py         run directory -> Rerun replay (.rrd)
              map_reuse_export.py     replay of a localization run in the saved map's fixed frame
  algorithms/ base.py                 AlgorithmAdapter interface + normalized diagnostics columns
              orbslam3.py             ORB-SLAM3 via Apptainer image
              rtabmap.py              RTAB-Map 0.23.8 (stereo) via Apptainer image; also exports maps
              fastlio.py              FAST-LIO2 (LiDAR-inertial) via Apptainer image; voxel map export
              lac.py                  Stanford LAC SLAM (SuperPoint/LightGlue stereo VO + GTSAM), GPU image
  runner.py, config.py, cli.py        experiment execution, YAML configs, CLI
  evaluation/ metrics.py, evaluate.py, plots.py, summary.py
third_party/orbslam3/                 pinned container definition, headless driver, patch
third_party/rtabmap/                  pinned container definition, headless stereo driver
third_party/fastlio/                  pinned container definition, offline-input patch for laserMapping
third_party/lac/                      pinned container definition, offline Python driver for the LAC stack
sim/omnilrs/                          render LunaPolaris trajectories in the OmniLRS Lunalab (Isaac Sim, GPU)
configs/                              example experiment and batch configs
scripts/slurm_slambench.sbatch        run any slambench command on a PACE compute node
scripts/download_subtmrs_smokeroom.sbatch  fetch SubT-MRS SmokeRoom data + calibration + GT
scripts/compare_variants.py           repeated runs of config variants: outages, map continuity, shared-timestamp error
scripts/lac_stereo_depth_probe.py     dump LAC's stereo keypoint matches (runs in the LAC container, GPU)
scripts/stereo_depth_bias.py          stereo depth bias vs simulator GT depth or the LiDAR reference map
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
#    RTAB-Map image (tag 0.23.8 on the pinned introlab3it/rtabmap:jammy-deps digest; optional):
sbatch -p ice-cpu -c 16 --mem=48G -t 03:00:00 --constraint=gold6226 third_party/rtabmap/build_container.sh
#    -> ~/scratch/slam-bench-work/containers/rtabmap_0.23.8.sif
#    FAST-LIO2 image (hku-mars/FAST_LIO 7cc4175 on the pinned ros:noetic-perception digest; optional):
sbatch -p ice-cpu -c 8 --mem=16G -t 01:30:00 --constraint=gold6226 third_party/fastlio/build_container.sh
#    -> ~/scratch/slam-bench-work/containers/fastlio_7cc4175.sif
#    LAC SLAM image (Stanford-NavLab/lunar_autonomy_challenge 26c3a2b on pinned pytorch 2.3.1/CUDA 12.1;
#    optional; runs need a GPU node, e.g. sbatch -p ice-gpu --gres=gpu:v100:1 ...):
sbatch -p ice-cpu -c 8 --mem=32G -t 01:30:00 third_party/lac/build_container.sh
#    -> ~/scratch/slam-bench-work/containers/lac_26c3a2b.sif

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
slambench view <run_dir>                                          # replay file <run_dir>/replay.rrd
```

### Replay viewer
`pip install -e '.[viz]'`, then `slambench view <run_dir> [--stride 2] [--no-images]` writes
`<run_dir>/replay.rrd` (SmokeRoom, stride 2: ~110 MB, ~1.5 min). Copy it to a machine with a display
and open it with the **same** Rerun version: `pip install rerun-sdk==0.26.2 && rerun replay.rrd`.
Play, pause and scrub on the `time` (seconds since the first frame) or `frame` timeline:
* camera images (every `stride`-th frame, JPEG preview of the prepared estimator input) with the
  features ORB-SLAM3 tracked: green = matched to a map point, orange = matched but rejected as
  outlier, blue = no map point (ORB-SLAM3 setting `log_features`, on by default);
* 3D: ground truth (grey), final trajectory (red) with its sparse map (orange points, dark-red
  keyframes; the map ORB-SLAM3 exported at shutdown, placed with the final trajectory's alignment),
  one colour per online coordinate session, each with a labelled start marker; paths break at tracking gaps and session boundaries, and moving
  markers show the current pose (cleared while there is none);
* position error per online session and for the final trajectory, tracking OK / coordinate session /
  map id, keypoints / inliers / outliers per frame, tracking time vs frame period, and a log of
  tracking-state and session changes;
* a run panel with status and per-session accuracy.

Positions are aligned exactly as the evaluator scores them (each session on its own, using the whole
session's ground truth): aligned benchmark accuracy, not what the vehicle knew at that instant. If
ground truth or the dataset is unavailable, raw poses are drawn per session and labelled unaligned.
Only the map exported at shutdown is shown (points of other, unmerged maps have no common frame and
are counted, not drawn); map growth over time is not recorded. Monocular maps carry the same per-
session scale estimate as the trajectory.
Any config value can be overridden: `--set algorithm.settings.loop_closure=false`.

### Map reuse (RTAB-Map)
Run a mapping config (e.g. `configs/lunapolaris_seq01_rtabmap_stereo.yaml`), put its run directory
into a `configs/lunapolaris_mapreuse_*` config as `algorithm.settings.map_run`, then run that config:
it localizes another sequence against a copy of the saved map in a fresh process and scores it in
the map's fixed frame (`map_reuse` in `eval/metrics.json`; see docs/evaluation.md).
`slambench view <localization run>` writes a map-reuse replay instead of the aligned one: the saved
map (mapping run's dense cloud and node path), the query's ground truth and its localized poses
coloured by fixed-frame error, all placed with the mapping run's map->GT transform (nothing fitted on
the query run); accepted localizations with a line to the matched map node, rejected hypotheses, the
query camera next to the map image it should recognize (the matched node, else the map node nearest
its true pose and viewing direction), and fixed-frame error / localized state over time.

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
Preprocessing (e.g. `clahe`) is off by default and recorded when enabled; its strength is set by
`clahe_clip_limit` (default 3.0, lower = gentler) and `clahe_tile_grid` (default 8). Every summary row
carries the `experiment` name and the `clahe` / `clahe_clip_limit` settings, so preprocessing variants of the same estimator (e.g.
`subtmrs_smokeroom_orbslam3_mono` vs `..._mono_clahe`) are separate, comparable entries. Loop closure is an
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
* `docs/datasets/omnilrs.md` — simulated Lunalab stereo: PACE/Isaac Sim setup, render pipeline, layout.
* `docs/results/milestone1.md` — what ran, what was verified, blockers.
* `docs/results/lac_lunapolaris.md` — LAC SLAM (SuperPoint/LightGlue + GTSAM) vs ORB-SLAM3,
  RTAB-Map and FAST-LIO2 on seq01/02/11: accuracy, resources, map quality, drift diagnosis.

## Licences
LunaPolaris: CC BY 4.0, © 2025 University of Luxembourg — Database by Dave van der Meer,
SnT – SpaceR. ORB-SLAM3: GPLv3 (built inside the container, not redistributed here). RTAB-Map: BSD-3-Clause
(built inside the container). FAST-LIO: GPLv2 (built inside the container; only a patch is kept here).
LAC (Stanford NAV Lab): MIT; LightGlue: Apache-2.0; SuperPoint weights: Magic Leap non-commercial
licence (all fetched inside the container, not redistributed here).
TUM-VI: CC BY 4.0. SubT-MRS: CC BY 4.0 (AirLab, CMU). OmniLRS: BSD-3-Clause; Isaac Sim: NVIDIA licence
(container pulled from NGC, not redistributed here).
