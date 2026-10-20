#!/bin/bash
# Build the pinned FAST-LIO2 Apptainer image used by slam-benching.
#
# Usage: third_party/fastlio/build_container.sh [OUTPUT_SIF]
# Default output: $SLAMBENCH_CONTAINERS/fastlio_7cc4175.sif
# (SLAMBENCH_CONTAINERS defaults to ~/scratch/slam-bench-work/containers).
# On PACE, run it on a compute node, e.g.:
#   sbatch -p ice-cpu -c 8 --mem=16G -t 01:00:00 --constraint=gold6226 third_party/fastlio/build_container.sh
# FAST-LIO enables OpenMP with MP_PROC_NUM=3 when the build machine has more than 4 cores (recorded
# as mp_proc_num in /opt/slambench/versions.txt).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/third_party/fastlio/fastlio.def" ]; then
  HERE="${SLURM_SUBMIT_DIR}/third_party/fastlio"
fi
OUT_DIR="${SLAMBENCH_CONTAINERS:-$HOME/scratch/slam-bench-work/containers}"
OUT="${1:-$OUT_DIR/fastlio_7cc4175.sif}"
mkdir -p "$(dirname "$OUT")"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$HOME/scratch/slam-bench-work/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/scratch/slam-bench-work/cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
export SLAMBENCH_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"
cd "$HERE"
apptainer build --fakeroot --force "$OUT" fastlio.def
apptainer test "$OUT"
echo "Built $OUT"
