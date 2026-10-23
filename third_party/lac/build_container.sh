#!/bin/bash
# Build the pinned LAC SLAM (Stanford NAV Lab) Apptainer image used by slam-benching.
#
# Usage: third_party/lac/build_container.sh [OUTPUT_SIF]
# Default output: $SLAMBENCH_CONTAINERS/lac_26c3a2b.sif
# (SLAMBENCH_CONTAINERS defaults to ~/scratch/slam-bench-work/containers).
# On PACE, run it on a compute node, e.g.:
#   sbatch -p ice-cpu -c 8 --mem=32G -t 01:00:00 third_party/lac/build_container.sh
# (CPU node is fine: the image only needs a GPU at run time, via `apptainer exec --nv`.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/third_party/lac/lac.def" ]; then
  HERE="${SLURM_SUBMIT_DIR}/third_party/lac"
fi
OUT_DIR="${SLAMBENCH_CONTAINERS:-$HOME/scratch/slam-bench-work/containers}"
OUT="${1:-$OUT_DIR/lac_26c3a2b.sif}"
mkdir -p "$(dirname "$OUT")"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$HOME/scratch/slam-bench-work/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/scratch/slam-bench-work/cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
export SLAMBENCH_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"
cd "$HERE"
apptainer build --fakeroot --force "$OUT" lac.def
apptainer test "$OUT"
echo "Built $OUT"
