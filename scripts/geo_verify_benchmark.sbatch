#!/bin/bash
# Geometric verification benchmark (scripts/geo_verify_benchmark.py). The first argument is the stage:
#   match    (GPU, LAC image):  sbatch -p coc-gpu -q coc-ice --gres=gpu:l40s:1 scripts/geo_verify_benchmark.sbatch match --suite real
#   lidar    (CPU, venv; reads the LunaPolaris bags directly, writes only small voxelized scans):
#            sbatch -p ice-cpu scripts/geo_verify_benchmark.sbatch lidar
#   geometry (CPU, LAC image):  sbatch -p ice-cpu -c 24 --mem=96G -t 8:00:00 scripts/geo_verify_benchmark.sbatch geometry --suite real --workers 24
#   summary  (any):             scripts/geo_verify_benchmark.sbatch summary
# Outputs go to /storage/ice1/3/6/bkuang7/slam-bench-work/vpr/geo_verify (a few hundred MB, mostly LiDAR scans).
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o geo_verify-%j.out
set -euo pipefail
export PATH=/storage/ice1/3/6/bkuang7/install-dir/bin:$PATH
W=/storage/ice1/3/6/bkuang7/slam-bench-work
S=$W/containers/lac_26c3a2b.sif
cd "${SLURM_SUBMIT_DIR:-.}"
stage=$1
case $stage in
  match)
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    apptainer exec --nv --env TORCH_HOME=$W/torch_home --env OMP_NUM_THREADS=8 --bind /storage/ice1/3/6/bkuang7 $S \
        python -W ignore scripts/geo_verify_benchmark.py "$@" ;;
  geometry|summary)
    apptainer exec --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
        --bind /storage/ice1/3/6/bkuang7 $S python -W ignore scripts/geo_verify_benchmark.py "$@" ;;
  lidar)
    source $W/venv/bin/activate
    python scripts/geo_verify_benchmark.py "$@" ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
