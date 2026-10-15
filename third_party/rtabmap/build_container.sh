#!/bin/bash
# Build the pinned RTAB-Map Apptainer image used by slam-benching.
#
# Usage: third_party/rtabmap/build_container.sh [OUTPUT_SIF]
# Default output: $SLAMBENCH_CONTAINERS/rtabmap_0.23.8.sif
# (SLAMBENCH_CONTAINERS defaults to ~/scratch/slam-bench-work/containers).
# On PACE, run it on a compute node, e.g.:
#   sbatch -p ice-cpu -c 8 --mem=32G -t 03:00:00 --constraint=gold6226 third_party/rtabmap/build_container.sh
# RTAB-Map and the driver are compiled without -march=native, but run benchmarks on one node type
# (--constraint=gold6226) so timing stays comparable with the ORB-SLAM3 runs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/third_party/rtabmap/rtabmap.def" ]; then
  HERE="${SLURM_SUBMIT_DIR}/third_party/rtabmap"
fi
OUT_DIR="${SLAMBENCH_CONTAINERS:-$HOME/scratch/slam-bench-work/containers}"
OUT="${1:-$OUT_DIR/rtabmap_0.23.8.sif}"
mkdir -p "$(dirname "$OUT")"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$HOME/scratch/slam-bench-work/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/scratch/slam-bench-work/cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
export SLAMBENCH_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"
cd "$HERE"
apptainer build --fakeroot --force "$OUT" rtabmap.def
apptainer test "$OUT"
echo "Built $OUT"
