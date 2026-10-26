#!/bin/bash
# Render an OmniLRS Lunalab stereo sequence on a PACE RTX 6000 node (Isaac Sim 5.0.0 crashes on driver 615
# L40S/A40 nodes; the RTX 6000 nodes run driver 580).
#   sbatch -p coc-gpu -q coc-ice --gres=gpu:rtx_6000:1 sim/omnilrs/render.sbatch <poses.npz> <out_dir> [render_stereo.py args]
#SBATCH -c 6 --mem=48G -t 04:00:00 -J omnilrs_render
set -euo pipefail
O=${OMNILRS_WORK:-/storage/ice1/3/6/bkuang7/omnilrs}   # SIF, OmniLRS clone, pydeps, caches (real path, not ~/scratch)
REPO=${SLAMBENCH_REPO:-$HOME/dev/slam-benching}
export PATH=/storage/ice1/3/6/bkuang7/install-dir/bin:$PATH
POSES=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
mkdir -p "$OUT"; C=$O/isaac_cache; K=$OUT/.kit; mkdir -p $K/data $K/logs
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
apptainer exec --nv --cleanenv --writable-tmpfs \
  --env ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,OMNI_KIT_ACCEPT_EULA=YES,PYTHONPATH=$O/pydeps:$O/OmniLRS \
  -B $C/kit:/isaac-sim/kit/cache -B $K/data:/isaac-sim/kit/data -B $K/logs:/isaac-sim/kit/logs \
  -B $C/ov:$HOME/.cache/ov -B $C/pip:$HOME/.cache/pip -B $C/glcache:$HOME/.cache/nvidia/GLCache \
  -B $C/computecache:$HOME/.nv/ComputeCache -B $C/logs:$HOME/.nvidia-omniverse/logs \
  -B $C/data:$HOME/.local/share/ov/data -B $C/documents:$HOME/Documents \
  -B $O -B "$(dirname "$POSES")" -B "$(dirname "$OUT")" -B "$REPO" --pwd $O/OmniLRS \
  $O/containers/isaac-sim_5.0.0.sif /isaac-sim/python.sh "$REPO/sim/omnilrs/render_stereo.py" "$POSES" "$OUT" "$@"
