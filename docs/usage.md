# Usage reference

Detailed setup, running and output reference. The overview is in the [README](../README.md); findings are in [research_summary](research_summary.md).

## Repository layout
```
slambench/
  types.py, trajectory.py, tf.py      shared types, normalized trajectory format, static TF tree
  datasets/   base.py                 DatasetAdapter interface + offline format writers
              lunapolaris.py          ROS 2 bag adapter (rosbags, no ROS install needed)
              tumvi.py                TUM-VI (EuRoC/ASL export) adapter, used for installation checks
              omnilrs.py              OmniLRS/Isaac Sim rendered stereo sequences (exact calibration + GT)
  viz/        rerun_export.py         run directory -> Rerun replay (.rrd)
              map_reuse_export.py     replay of a localization run in the saved map's fixed frame
  algorithms/ base.py                 AlgorithmAdapter interface + normalized diagnostics columns
              orbslam3.py             ORB-SLAM3 via Apptainer image
              orbslam3_map_reuse.py   atlas save / load (map reuse) for ORB-SLAM3 and SP_SLAM3
              rtabmap.py              RTAB-Map 0.23.8 (stereo) via Apptainer image; also exports maps
              fastlio.py              FAST-LIO2 (LiDAR-inertial) via Apptainer image; voxel map export
              lac.py                  Stanford LAC SLAM (SuperPoint/LightGlue stereo VO + GTSAM), GPU image
              spslam3.py              SP_SLAM3 (ORB-SLAM3 + SuperPoint/LightGlue, stereo), GPU image
  runner.py, config.py, cli.py        experiment execution, YAML configs, CLI
  evaluation/ metrics.py, evaluate.py, plots.py, summary.py
third_party/orbslam3/                 pinned container definition, headless driver, patch
third_party/rtabmap/                  pinned container definition, headless stereo driver
third_party/fastlio/                  pinned container definition, offline-input patch for laserMapping
third_party/lac/                      pinned container definition, offline Python driver for the LAC stack
third_party/spslam3/                  pinned container definition, headless driver (port of ORB-SLAM3's), patches
sim/omnilrs/                          render LunaPolaris trajectories in the OmniLRS Lunalab, aerial surveys in the Lunaryard (Isaac Sim)
configs/                              one YAML per experiment (layout: configs/README.md); batch files at top level
  lunapolaris/, lunapolaris/map_reuse/ real LunaPolaris runs; map-reuse session B
  omnilrs/{lunalab,lunaryard,largescale,map_reuse}/  simulated runs by OmniLRS environment; session B
  tumvi/                              estimator installation check
scripts/slurm_slambench.sbatch        run any slambench command on a PACE compute node
scripts/compare_variants.py           repeated runs of config variants: outages, map continuity, shared-timestamp error
scripts/lac_stereo_depth_probe.py     dump LAC's stereo keypoint matches (runs in the LAC container, GPU)
scripts/stereo_depth_bias.py          stereo depth bias vs simulator GT depth or the LiDAR reference map
scripts/flare_headroom_probe.py       flare-in-view score per frame vs stereo/temporal matches and 1-s local drift
scripts/lidar_degeneracy_probe.py     LiDAR point-to-plane conditioning per scan: full vs ground-only crops
scripts/fastlio_divergence_report.py  FAST-LIO2 divergence onset (speed, n_effective) and pre-divergence ATE
vpr/, scripts/vpr_*.py                place recognition benchmark: AnyLoc/MixVPR/CosPlace vs DBoW2 retrieval + LightGlue/PnP
                                      relocalization (docs/results/place_recognition.md)
docs/                                 adapter interface, evaluation methodology, dataset notes
tests/                                evaluator / runner tests with synthetic ground truth
```

## Setup (PACE ICE; adapt paths elsewhere)
Raw data, converted inputs, containers and run outputs stay outside the repository, by default in
`~/scratch/slam-bench-work` (`work_dir`, `output_root` in configs).
Converted inputs (`prepared_dir`, default `auto`): an existing complete cache in `<work_dir>/prepared` is
reused; otherwise a Slurm job converts into the compute node's local disk (`$TMPDIR`, ~1.4 TB on ICE),
deleted when the job ends, so new variants cost no scratch quota. Set `prepared_dir: work_dir` (or use
`slambench prepare`) for a persistent cache, e.g. inputs read by analysis scripts or many repeats.

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
#    SP_SLAM3 image (fthbng77/SP_SLAM3 9439971, LibTorch 2.3.0+cu121, Ubuntu 20.04; optional; runs need
#    a CUDA GPU node, e.g. sbatch -p coc-gpu -q coc-ice --gres=gpu:v100:1 ...):
sbatch -p ice-cpu -c 8 --mem=32G -t 02:00:00 third_party/spslam3/build_container.sh
#    -> ~/scratch/slam-bench-work/containers/spslam3_9439971.sif

# 3. Data
#    LunaPolaris sequence_01 (33.8 GB, Zenodo record 15737938) -> ~/scratch/LunaPolaris/sequence_01/
#      metadata.yaml + sequence_01.db3; optional calibration_results.json (record 16728838) one level up.
#    TUM-VI room1 (installation check, 1.7 GB):
#      https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar
#      -> ~/scratch/datasets/tumvi/dataset-room1_512_16/
```

Large downloads were killed on PACE login nodes; run them as Slurm jobs.

## Running
```bash
slambench inspect configs/lunapolaris/lunapolaris_seq01_orbslam3_stereo.yaml   # streams, calibration, per-mode compatibility
sbatch -c 8 --mem=32G -t 03:00:00 scripts/slurm_slambench.sbatch \
       run configs/lunapolaris/lunapolaris_seq01_orbslam3_stereo.yaml           # convert (cached), run, evaluate
sbatch scripts/slurm_slambench.sbatch batch configs/batch_example.yaml   # several configs + summary CSV
sbatch scripts/slurm_slambench.sbatch batch configs/rejection_examples.yaml  # incompatible configs -> config_rejected
slambench evaluate <run_dir> --set evaluation.alignment=sim3      # re-evaluate with other settings
slambench summarize ~/scratch/slam-bench-work/runs -o summary.csv # aggregate all runs (failed ones too)
slambench view <run_dir>                                          # replay file <run_dir>/replay.rrd
```

### Replay viewer
`pip install -e '.[viz]'`, then `slambench view <run_dir> [--stride 2] [--no-images]` writes
`<run_dir>/replay.rrd`. Copy it to a machine with a display
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
Run a mapping config (e.g. `configs/lunapolaris/lunapolaris_seq01_rtabmap_stereo.yaml`), put its run directory
into a `configs/lunapolaris/map_reuse/lunapolaris_mapreuse_*` config as `algorithm.settings.map_run`, then run that config:
it localizes another sequence against a copy of the saved map in a fresh process and scores it in
the map's fixed frame (`map_reuse` in `eval/metrics.json`; see docs/evaluation.md).
`slambench view <localization run>` writes a map-reuse replay instead of the aligned one: the saved
map (mapping run's dense cloud and node path), the query's ground truth and its localized poses
coloured by fixed-frame error, all placed with the mapping run's map->GT transform (nothing fitted on
the query run); accepted localizations with a line to the matched map node, rejected hypotheses, the
query camera next to the map image it should recognize (the matched node, else the map node nearest
its true pose and viewing direction), and fixed-frame error / localized state over time.

### Map reuse (ORB-SLAM3, SP_SLAM3)
Run a mapping config with `save_map: true` (e.g. `configs/lunapolaris/lunapolaris_seq01_orbslam3_stereo_map.yaml`).
Then run a `configs/**/*mapreuse*_{orbslam3,spslam3}_{relocalize,merge}.yaml` config. Its `load_map`
names the mapping config's output directory, so the latest successful mapping run is used, and it
localizes another sequence or lighting in that atlas. Submit it with
`sbatch --dependency=afterok:<mapping job>`. Method and results: docs/results/map_reuse.md.

### Experiment config
```yaml
name: lunapolaris_seq01_orbslam3_stereo
output_root: ~/scratch/slam-bench-work/runs
work_dir: ~/scratch/slam-bench-work            # persistent prepared-input cache (<work_dir>/prepared)
prepared_dir: auto                             # auto | work_dir | node_local | <path>
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
`<experiment>` vs `<experiment>_clahe`) are separate, comparable entries. Loop closure is an
explicit setting recorded in every summary row; keep it constant within a comparison.
`playback_rate: 1.0` feeds frames at the recorded rate (estimator threads behave as online);
`0` feeds as fast as possible.

### What a run directory contains
`config.resolved.yaml`, `environment.json` (git commit + diff, packages, host, Slurm job,
container sha256 and library versions), `dataset_info.json`, `prepared_input.json` (cache
`directory` and `location`), `input_frames.csv` (copy of the input frame list),
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
