#!/bin/bash
# Render the same trajectory in the OmniLRS Lunaryard (80 m, procedural craters, seed 42) under the lighting
# conditions of the aerial lighting sweep, one after the other in a single job (RTX 6000 node only):
#   sbatch -p coc-gpu -q coc-ice --gres=gpu:rtx_6000:1 sim/omnilrs/render_lunaryard_sweep.sbatch \
#          <poses.npz> <out_prefix> [extra render_stereo.py args, e.g. --preview 8]
# writes <out_prefix>_<condition>/ for each condition. CONDITIONS="nominal dim" restricts the set.
#SBATCH -c 6 --mem=48G -t 08:00:00 -J omnilrs_sweep
set -euo pipefail
REPO=${SLAMBENCH_REPO:-$HOME/dev/slam-benching}
POSES=$1; PREFIX=$2; shift 2
declare -A LIGHT=(
  [nominal]="--sun-azimuth 180 --sun-elevation 45"
  [low_sun]="--sun-azimuth 180 --sun-elevation 7"
  [dim]="--sun-azimuth 180 --sun-elevation 45 --sun-intensity 175"
  [glare]="--sun-azimuth 270 --sun-elevation 10 --lens-flares on"
)
# Terrain look used for the sweep: the cfg's Basalt (GravelStones) tiles every 0.2 m at albedo 0.3, which renders as a
# featureless grey from 5 m altitude; 5 m tiles at albedo 0.6 give a fine regolith-like texture, plus 300 rocks.
TERRAIN=${TERRAIN:---texture-scale 0.2 --albedo-brightness 0.6 --rocks 300}
for c in ${CONDITIONS:-nominal low_sun dim glare}; do
  echo "=== $c: ${LIGHT[$c]} $TERRAIN $*"
  # shellcheck disable=SC2086
  bash "$REPO/sim/omnilrs/render.sbatch" "$POSES" "${PREFIX}_$c" --env lunaryard ${LIGHT[$c]} $TERRAIN "$@"
done
