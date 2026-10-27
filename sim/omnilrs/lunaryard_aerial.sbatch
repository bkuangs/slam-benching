#!/bin/bash
# End-to-end Lunaryard aerial lighting sweep on one RTX 6000 node (see docs/datasets/omnilrs.md):
#   1. build the Lunaryard 80 m scene once to dump its DEM (--dem-only)
#   2. aerial_poses.py on that DEM (slambench venv on the host)
#   3. a --preview render of every lighting condition (check these first)
#   4. the full renders, <renders>/lunaryard_aerial_<condition>/
#   sbatch -p coc-gpu -q coc-ice --gres=gpu:rtx_6000:1 sim/omnilrs/lunaryard_aerial.sbatch
# STAGES="dem poses preview full" selects stages; CONDITIONS is passed on to render_lunaryard_sweep.sbatch.
#SBATCH -c 6 --mem=48G -t 05:00:00 -J lunaryard_aerial
set -euo pipefail
O=${OMNILRS_WORK:-/storage/ice1/3/6/bkuang7/omnilrs}
REPO=${SLAMBENCH_REPO:-$HOME/dev/slam-benching}
VENV=${SLAMBENCH_VENV:-$HOME/scratch/slam-bench-work/venv}
P=$O/poses; R=$O/renders; S=$REPO/sim/omnilrs
STAGES=${STAGES:-dem poses preview full}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$P" "$R"
if [[ $STAGES == *dem* ]]; then
  python "$S/aerial_poses.py" "$P/lunaryard_aerial_flat.npz"           # placeholder trajectory, flat terrain
  bash "$S/render.sbatch" "$P/lunaryard_aerial_flat.npz" "$R/lunaryard_dem" --env lunaryard --dem-only
fi
if [[ $STAGES == *poses* ]]; then
  python "$S/aerial_poses.py" "$P/lunaryard_aerial.npz" --dem "$R/lunaryard_dem/dem.npy"
fi
if [[ $STAGES == *preview* ]]; then
  bash "$S/render_lunaryard_sweep.sbatch" "$P/lunaryard_aerial.npz" "$R/prev_lunaryard_aerial" --preview 12
fi
if [[ $STAGES == *full* ]]; then
  bash "$S/render_lunaryard_sweep.sbatch" "$P/lunaryard_aerial.npz" "$R/lunaryard_aerial"
fi
