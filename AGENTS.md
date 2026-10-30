# Notes for agents working in this repository

**Start with [`docs/research_summary.md`](docs/research_summary.md).** It covers the research question, the
current argument, the evidence and the open work. Update it when a result changes the story.

## Environment (Georgia Tech PACE ICE)

* Python venv: `source ~/scratch/slam-bench-work/venv/bin/activate`. Tests: `pytest -q` (no data needed).
  The venv has no torch; torch work runs inside the LAC image (below).
* Apptainer: `export PATH=/storage/ice1/3/6/bkuang7/install-dir/bin:$PATH`. `~/scratch` is a symlink that
  is **not visible inside Apptainer**, so use the real path `/storage/ice1/3/6/bkuang7/...` in binds and
  configs.
* Images: `/storage/ice1/3/6/bkuang7/slam-bench-work/containers/*.sif`. The map-reuse configs pin
  `orbslam3_4452a3c_driver3.sif` and `spslam3_9439971_r3.sif` explicitly.
* Torch / LightGlue / SuperPoint / ALIKED / DISK: `lac_26c3a2b.sif` with
  `--env TORCH_HOME=/storage/ice1/3/6/bkuang7/slam-bench-work/torch_home --bind /storage/ice1/3/6/bkuang7`.
  The login node is far too slow for these models; use a GPU node.

## Running experiments

* One run: `sbatch [slurm opts] scripts/slurm_slambench.sbatch run configs/<name>.yaml`. Output goes to
  `~/scratch/slam-bench-work/runs/<name>/<run id>/` (`summary.csv`, `eval/metrics.json`,
  `algorithm/driver_status.json`).
* CPU estimators (ORB-SLAM3, RTAB-Map, FAST-LIO2): `-p ice-cpu --constraint=gold6226`, so timing is
  comparable.
* GPU: request `--gres=gpu:v100:1` or `gpu:l40s:1` (with `-p coc-gpu -q coc-ice`). **Never bare `gpu:1`**:
  it can land on an AMD MI210. Isaac Sim (OmniLRS rendering) works only on `gpu:rtx_6000:1`.
* Scratch quota is 300 GB and nearly full. Converted inputs go to node-local disk by default
  (`prepared_dir: auto`, README "Setup"); only `slambench prepare` / `prepared_dir: work_dir` write
  persistent caches to scratch. Don't launch several full LunaPolaris LiDAR conversions (~8 GB each) into
  `work_dir`.
* Map-reuse session B depends on session A: `sbatch --dependency=afterok:<A job> ...`.
* Accuracy comparisons need repeats (median of 3): multi-threaded ORB-SLAM3 is not deterministic.

## Conventions

* Every experiment is a YAML in `configs/`; result pages in `docs/results/` cite their configs and run ids.
* Datasets, prepared inputs, run outputs and `*.sif` images never go in git.
* On LunaPolaris, compare **translation** error. Rotation error carries a constant ~4.2° GT extrinsic offset.
