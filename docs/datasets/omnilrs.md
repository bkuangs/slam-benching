# OmniLRS simulated Lunalab stereo

Synthetic stereo sequences rendered in the [OmniLRS](https://github.com/OmniLRS/OmniLRS) Lunalab (Isaac Sim
digital twin of the University of Luxembourg LunaLab, where LunaPolaris was recorded). The camera replays a
real LunaPolaris GT trajectory, so sim and real runs share the same motion and can be compared frame by frame.
Calibration and GT are exact by construction.

## PACE setup (verified 2026-10-26)
* Isaac Sim 5.0.0 (the version OmniLRS pins) from NGC: `apptainer pull isaac-sim_5.0.0.sif
  docker://nvcr.io/nvidia/isaac-sim:5.0.0` (7.1 GB; pull with `APPTAINER_TMPDIR` on node-local `/tmp`).
  The host glibc (RHEL 9, 2.34) is too old for the pip wheels, so the container is required.
* **GPU: RTX 6000 nodes only** (`-p coc-gpu -q coc-ice --gres=gpu:rtx_6000:1`, driver 580). On the L40S and A40
  nodes (driver 615.71) the RTX renderer segfaults in `librtx.scenedb.plugin.so` at startup, for both Isaac Sim
  5.0.0 and 5.1.0 and with every renderer/multi-GPU/DLSS setting tried. Only 2 RTX 6000 nodes (4 GPUs each).
* OmniLRS clone with LFS assets (6.1 GB) and its extra Python deps (`pandas zfpy omegaconf hydra-core skyfield`,
  installed `--no-deps --target pydeps` with the container's Python so Isaac's numpy is kept). ROS 2 is not
  needed: the renderer drives the Lunalab environment classes directly.
* `~/scratch` is a symlink that is not visible inside Apptainer; `sim/omnilrs/render.sbatch` uses real paths
  (`OMNILRS_WORK`, default `/storage/ice1/3/6/bkuang7/omnilrs`).

## Pipeline
```bash
# 1. GT camera poses at the real frame times + ZED2 calibration (slambench venv)
python sim/omnilrs/export_poses.py configs/lunapolaris_seq01_orbslam3_stereo.yaml poses/seq01_60s.npz --max-s 60
# 2. Render (RTX 6000 node, ~0.42 s per stereo frame + ~90 s scene setup)
sbatch -p coc-gpu -q coc-ice --gres=gpu:rtx_6000:1 sim/omnilrs/render.sbatch poses/seq01_60s.npz \
       renders/seq01_60s_norocks --rocks 0
# 3. Run any stereo estimator on it
sbatch -c 8 --mem=32G -t 01:00:00 scripts/slurm_slambench.sbatch run configs/omnilrs_seq01_60s_orbslam3_stereo.yaml
```
`render_stereo.py` options: `--terrain` (`Terrain_lidar01`/`Terrain_lidar02`, LiDAR scans of the real lab floor,
1 cm grid), `--rocks N` (OmniLRS rock instancer; 0 disables), `--projector on|off`, `--projector-pos x y z`,
`--projector-intensity`, `--room-lights on|off`, `--rt-subframes`, `--preview N` (N frames spread over the
sequence), `--max-frames`.

## Scene and camera
* World frame = Lunalab terrain frame: DEM corner at the origin, x across the 6.5 m width, y along the 10 m
  length. The mocap frame is used as is; with `Terrain_lidar01` the seq01 camera stays 0.38 m (σ 2.7 cm) above
  the DEM, consistent with the rover camera height, but the exact mocap-to-lab registration is not verified.
* Rectified pinhole stereo with the LunaPolaris ZED2 `fx = fy = 263.270`, 640×360, baseline 0.11983 m. The
  principal point is the image centre (320, 180) rather than the real (327.3, 177.8): USD aperture offsets are
  left at zero, and `calib.json` records what was rendered.
* Renderer: RayTracedLighting, 4 RT subframes per frame, no rover body, no motion blur, no sensor noise.

## Output layout (`<root>/<sequence>/`)
| file | content |
|---|---|
| `cam0/`, `cam1/` | `<ns>.png` BGR, ready for the slambench offline format (the adapter hard-links them) |
| `depth0/` | left z-depth, uint16 mm (GT only) |
| `frames.csv` | `timestamp_ns,left_path,right_path` |
| `gt_cam0.tum` | left optical-frame pose per frame |
| `calib.json` | rendered intrinsics, baseline, lighting/rock settings |
| `dem.npy`, `dem.json` | terrain height map (`dem[iy, ix]` at `(ix, iy) * res`) |
| `rocks.json` | rock instance positions/orientations/scales |

Storage: ~0.6 MB per frame (RGB pair + depth); a full seq01 replay (13,108 frames) is ~8 GB and ~95 min.

## First result (seq01, first 60 s, no rocks)
ORB-SLAM3 stereo (same settings as the real runs), SE(3)-aligned over the same 60 s (1.9 m path):

| input | ATE (online) | ATE (final) | end-displacement ratio |
|---|---|---|---|
| real LunaPolaris seq01 | 0.0085 m | 0.0083 m | 0.989 |
| OmniLRS replay | 0.0176 m | 0.0141 m | 0.989 |

The simulated floor is smoother and brighter than the real gravel (fewer, weaker corners), which is the most
visible sim-to-real difference so far.
