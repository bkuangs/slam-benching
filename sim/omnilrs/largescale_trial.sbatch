#!/bin/bash
# OmniLRS LargeScale (real LOLA Site20 DEM + streamed procedural high-res terrain) aerial stereo trial on one RTX 6000
# node (docs/results/largescale_trial.md):
#   1. aerial_poses.py terrain-following trajectories at 30 m and 60 m AGL straight from the DEM asset (host venv)
#   2. render_stereo.py --env largescale for each, static sun (az 180, el 45)
#   sbatch -p coc-gpu -q coc-ice --gres=gpu:rtx_6000:1 sim/omnilrs/largescale_trial.sbatch
# BASELINE=0.5 renders a wider stereo baseline (default ZED2 0.11983 m); use with a SUFFIX such as _b050.
# ALTS="30 60" selects the trajectories; extra args are passed on to render_stereo.py (e.g. --preview 12).
# v2 (non-repeating ground, docs/results/largescale_trial.md): LOOK="$LOOK_V2" SUFFIX=_v2 sbatch ... where
#   LOOK_V2="--sun-azimuth 180 --sun-elevation 45 --terrain-shader bombed --albedo-brightness 1.0 --bright-amp 0.3"
#SBATCH -c 12 --mem=64G -t 03:00:00 -J ls_trial
set -euo pipefail
O=${OMNILRS_WORK:-/storage/ice1/3/6/bkuang7/omnilrs}
REPO=${SLAMBENCH_REPO:-$HOME/dev/slam-benching}
VENV=${SLAMBENCH_VENV:-$HOME/scratch/slam-bench-work/venv}
P=$O/poses; R=${OUT_ROOT:-$O/renders/largescale_trial}; S=$REPO/sim/omnilrs
DEM=$O/OmniLRS/assets/Terrains/SouthPole/Site20_final_adj_5mpp_surf
# LunarRegolith8k (the LargeScale cfg material) has albedo_add 1, i.e. white without auto-exposure.
LOOK=${LOOK:---sun-azimuth 180 --sun-elevation 45 --albedo-add 0 --albedo-brightness 0.6}
declare -A TRAJ=(
  [30]="--altitude-m 30 --speed 5 --legs 2 --spacing 30 --alt-perturb-m 1.0"
  [60]="--altitude-m 60 --speed 8 --legs 2 --spacing 60 --alt-perturb-m 2.0"
)
# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$P" "$R"
for alt in ${ALTS:-30 60}; do
  # shellcheck disable=SC2086
  python "$S/aerial_poses.py" "$P/ls_aerial_${alt}m${SUFFIX:-}.npz" --largescale-dem "$DEM" --start-xy 2800 -2200 \
    --terrain-follow --duration-s 30 --edge-margin 100 ${TRAJ[$alt]} \
    ${BASELINE:+--baseline-m $BASELINE}
  echo "=== ${alt} m AGL: $LOOK $*"
  # shellcheck disable=SC2086
  bash "$S/render.sbatch" "$P/ls_aerial_${alt}m${SUFFIX:-}.npz" "$R/ls_aerial_${alt}m${SUFFIX:-}" --env largescale --ls-start 2800 -2200 \
    $LOOK "$@"
done
