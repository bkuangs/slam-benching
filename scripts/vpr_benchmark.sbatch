#!/bin/bash
# VPR benchmark (scripts/vpr_benchmark.py) inside the LAC image. ViT-G/14 needs ~6 GB; an L40S is fastest.
#   sbatch scripts/vpr_benchmark.sbatch --suite sim  --out /storage/ice1/3/6/bkuang7/slam-bench-work/vpr/results/sim
#   sbatch scripts/vpr_benchmark.sbatch --suite real --out /storage/ice1/3/6/bkuang7/slam-bench-work/vpr/results/real
# Sessions first: python scripts/vpr_prepare_sessions.py --out /storage/ice1/3/6/bkuang7/slam-bench-work/vpr/sessions
#SBATCH -p coc-gpu
#SBATCH -q coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o vpr_benchmark-%j.out
export PATH=/storage/ice1/3/6/bkuang7/install-dir/bin:$PATH
W=/storage/ice1/3/6/bkuang7/slam-bench-work
S=$W/containers/lac_26c3a2b.sif
cd "${SLURM_SUBMIT_DIR:-.}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv --env TORCH_HOME=$W/torch_home --env OMP_NUM_THREADS=8 --bind /storage/ice1/3/6/bkuang7 $S \
    python -W ignore scripts/vpr_benchmark.py "$@"
