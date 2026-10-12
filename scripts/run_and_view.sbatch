#!/bin/bash
# Run one experiment config, then export its replay (<run_dir>/replay.rrd).
#   sbatch scripts/run_and_view.sbatch configs/<experiment>.yaml [extra `slambench view` args]
#SBATCH --job-name=slambench-view
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slambench-view-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source "${SLAMBENCH_VENV:-$HOME/scratch/slam-bench-work/venv}/bin/activate"
if [ -n "${SLAMBENCH_APPTAINER_DIR:-}" ]; then export PATH="$SLAMBENCH_APPTAINER_DIR:$PATH"; fi
CFG="$1"; shift
python -m slambench run "$CFG"
NAME=$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CFG")
OUT=$(python -c "import sys,yaml,os; print(os.path.expanduser(yaml.safe_load(open(sys.argv[1])).get('output_root','~/scratch/slam-bench-work/runs')))" "$CFG")
RUN_DIR=$(ls -d "$OUT/$NAME"/*/ | tail -1)
python -m slambench view "$RUN_DIR" "$@"
