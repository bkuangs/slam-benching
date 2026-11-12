#!/bin/bash
# Classical-vs-learned relocalization grid on LunaPolaris (scripts/vpr_benchmark.py; docs/results/place_recognition.md):
# prepares the stereo inputs and sessions on node-local $TMPDIR (scratch is nearly full), then runs the benchmark.
#   sbatch scripts/vpr_grid.sbatch real0309 seq03 seq09 [extra vpr_benchmark.py args]
#   sbatch scripts/vpr_grid.sbatch real seq01 seq02 seq11
# Needs vpr_weights/superpoint_voc.dbow3 (scripts/dbow3_decompress.sh).
#SBATCH -p coc-gpu
#SBATCH -q coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /home/hice1/%u/scratch/slam-bench-work/logs/vpr_grid-%j.out
set -euo pipefail
SUITE=$1; shift
REAL=()
while [ $# -gt 0 ] && [[ $1 == seq* ]]; do REAL+=("$1"); shift; done
export PATH=/storage/ice1/3/6/bkuang7/install-dir/bin:$PATH
W=/storage/ice1/3/6/bkuang7/slam-bench-work
S=$W/containers/lac_26c3a2b.sif
L=${TMPDIR:-/tmp}/vpr_grid_$SLURM_JOB_ID
mkdir -p "$L"
trap 'rm -rf "$L"' EXIT
cd "${SLURM_SUBMIT_DIR:-.}"
df -h "$L" | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source $W/venv/bin/activate
python scripts/vpr_prepare_sessions.py --out "$L/sessions" --work "$L" --skip-sim --real "${REAL[@]}"
deactivate
apptainer exec --nv --env TORCH_HOME=$W/torch_home --env OMP_NUM_THREADS=8 --bind /storage/ice1/3/6/bkuang7 \
    --bind "$L" $S python -W ignore scripts/vpr_benchmark.py --suite "$SUITE" --sessions "$L/sessions" \
    --out "$W/vpr/results/${SUITE}_grid" "$@"
