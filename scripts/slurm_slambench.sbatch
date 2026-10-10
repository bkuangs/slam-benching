#!/bin/bash
# Run a slambench command on a compute node.
#   sbatch scripts/slurm_slambench.sbatch run configs/lunapolaris_seq01_orbslam3_stereo.yaml
#   sbatch scripts/slurm_slambench.sbatch batch configs/batch_example.yaml
# Resources can be overridden on the sbatch command line (-c, --mem, -t, -p).
# The submitting shell's environment (PATH incl. apptainer) is inherited by default.
#SBATCH --job-name=slambench
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slambench-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
VENV="${SLAMBENCH_VENV:-$HOME/scratch/slam-bench-work/venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
if [ -n "${SLAMBENCH_APPTAINER_DIR:-}" ]; then export PATH="$SLAMBENCH_APPTAINER_DIR:$PATH"; fi
command -v apptainer >/dev/null || { echo "apptainer not on PATH" >&2; exit 2; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"
exec python -m slambench "$@"
