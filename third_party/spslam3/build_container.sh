#!/bin/bash
# Build the pinned SP_SLAM3 Apptainer image used by slam-benching.
#
# Usage: third_party/spslam3/build_container.sh [OUTPUT_SIF]
# Default output: $SLAMBENCH_CONTAINERS/spslam3_9439971.sif
# (SLAMBENCH_CONTAINERS defaults to ~/scratch/slam-bench-work/containers).
# On PACE, run it on a compute node (no GPU needed to build), e.g.:
#   sbatch -p ice-cpu -c 8 --mem=32G -t 02:00:00 third_party/spslam3/build_container.sh
# Unlike the ORB-SLAM3 image, SP_SLAM3 is compiled without -march=native, so the image runs on
# any x86-64 GPU node (run it with `apptainer exec --nv`; the adapter does this).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/third_party/spslam3/spslam3.def" ]; then
  HERE="${SLURM_SUBMIT_DIR}/third_party/spslam3"
fi
OUT_DIR="${SLAMBENCH_CONTAINERS:-$HOME/scratch/slam-bench-work/containers}"
OUT="${1:-$OUT_DIR/spslam3_9439971.sif}"
mkdir -p "$(dirname "$OUT")"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$HOME/scratch/slam-bench-work/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/scratch/slam-bench-work/cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
export SLAMBENCH_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"
# MixVPR TorchScript model baked into the image (learned relocalization, patch 0007); export it
# first with export_mixvpr_slambench.py (see its docstring). Staged next to the definition for
# %files and removed afterwards.
MIXVPR_MODEL="${SLAMBENCH_MIXVPR_MODEL:-$HOME/scratch/slam-bench-work/vpr_weights/mixvpr_slambench.pt}"
if [ ! -f "$MIXVPR_MODEL" ]; then
  echo "MixVPR TorchScript model not found: $MIXVPR_MODEL (set SLAMBENCH_MIXVPR_MODEL;" \
       "export it with third_party/spslam3/export_mixvpr_slambench.py)" >&2
  exit 1
fi
cd "$HERE"
cp "$MIXVPR_MODEL" mixvpr_slambench.pt
trap 'rm -f "$HERE/mixvpr_slambench.pt"' EXIT
apptainer build --fakeroot --force "$OUT" spslam3.def
apptainer test "$OUT"
echo "Built $OUT"
