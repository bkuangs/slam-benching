#!/bin/bash
#SBATCH -p coc-gpu
#SBATCH -q coc-ice
#SBATCH --gres=gpu:v100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o cross_lighting_reloc-%j.out
export PATH=/storage/ice1/3/6/bkuang7/install-dir/bin:$PATH
S=/storage/ice1/3/6/bkuang7/slam-bench-work/containers/lac_26c3a2b.sif
TH=/storage/ice1/3/6/bkuang7/slam-bench-work/torch_home
cd "${SLURM_SUBMIT_DIR:-.}"
apptainer exec --nv --env TORCH_HOME=$TH --bind /storage/ice1/3/6/bkuang7 $S python -W ignore sim/omnilrs/cross_lighting_reloc.py "$@"
