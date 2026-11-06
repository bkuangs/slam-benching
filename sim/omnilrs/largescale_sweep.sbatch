#!/bin/bash
# OmniLRS LargeScale (Site20) map-reuse lighting sweep on one RTX 6000 node (docs/results/largescale_trial.md,
# "Map reuse across lighting"): the query pass of the 30/60 m AGL survey, rendered under the lighting conditions of
# the Lunaryard aerial sweep (sim/omnilrs/render_lunaryard_sweep.sbatch), with the v2 ground shader and a 0.5 m
# stereo baseline. The query pass flies the map pass's lawnmower shifted by (0.2, 0.25) x leg spacing with
# different perturbations (seed 1). The map pass (nominal) is largescale_trial/ls_aerial_<alt>m_b050 from
# largescale_trial.sbatch (BASELINE=0.5 SUFFIX=_b050).
#   ALT=60 sbatch -p coc-gpu -q coc-ice --gres=gpu:rtx_6000:1 sim/omnilrs/largescale_sweep.sbatch
# writes renders/largescale_sweep/ls<ALT>_query_<condition>/. CONDITIONS="nominal glare" restricts the set; PASS=map
# renders the map trajectory instead (ls<ALT>_map_<condition>, e.g. same-trajectory queries).
#SBATCH -c 8 --mem=48G -t 03:00:00 -J ls_sweep
set -euo pipefail
O=${OMNILRS_WORK:-/storage/ice1/3/6/bkuang7/omnilrs}
REPO=${SLAMBENCH_REPO:-$HOME/dev/slam-benching}
VENV=${SLAMBENCH_VENV:-$HOME/scratch/slam-bench-work/venv}
P=$O/poses; R=${OUT_ROOT:-$O/renders/largescale_sweep}; S=$REPO/sim/omnilrs
DEM=$O/OmniLRS/assets/Terrains/SouthPole/Site20_final_adj_5mpp_surf
ALT=${ALT:-60}; PASS=${PASS:-query}
LOOK="--terrain-shader bombed --albedo-brightness 1.0 --bright-amp 0.3"
# Lunaryard sweep definitions (azimuth 180 = sun toward -y, across the +/-x legs; 270 = ahead on the +x legs).
# low_sun is raised from 7 to 13 deg: the Site20 ground under the survey slopes ~6 deg down toward +y (away from an
# azimuth-180 sun), so 7 deg leaves 34 % of it in DEM shadow and the rest at grazing incidence (median cos 0.02);
# 13 deg gives the same 7 deg incidence on the mean local terrain plane as Lunaryard's 7 deg on flat ground.
declare -A LIGHT=(
  [nominal]="--sun-azimuth 180 --sun-elevation 45"
  [low_sun]="--sun-azimuth 180 --sun-elevation 13"
  [dim]="--sun-azimuth 180 --sun-elevation 45 --sun-intensity 175"
  [glare]="--sun-azimuth 270 --sun-elevation 10 --lens-flares on"
)
declare -A TRAJ=(
  [30]="--altitude-m 30 --speed 5 --legs 2 --spacing 30 --alt-perturb-m 1.0"
  [60]="--altitude-m 60 --speed 8 --legs 2 --spacing 60 --alt-perturb-m 2.0"
)
declare -A QUERY=([30]="--offset-xy 6 7.5 --seed 1" [60]="--offset-xy 12 15 --seed 1")
# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$P" "$R"
POSES=$P/ls_aerial_${ALT}m_b050_${PASS}.npz
# shellcheck disable=SC2086
python "$S/aerial_poses.py" "$POSES" --largescale-dem "$DEM" --start-xy 2800 -2200 --terrain-follow --duration-s 30 \
  --edge-margin 100 --baseline-m 0.5 ${TRAJ[$ALT]} $([[ $PASS == query ]] && echo "${QUERY[$ALT]}")
for c in ${CONDITIONS:-nominal low_sun dim glare}; do
  echo "=== ${ALT} m ${PASS} $c: ${LIGHT[$c]} $LOOK $*"
  # shellcheck disable=SC2086
  bash "$S/render.sbatch" "$POSES" "$R/ls${ALT}_${PASS}_$c" --env largescale --ls-start 2800 -2200 ${LIGHT[$c]} $LOOK "$@"
done
