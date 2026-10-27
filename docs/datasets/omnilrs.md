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

LAC on the same replay (`configs/omnilrs_seq01_60s_lac_stereo{,_kfvo}.yaml`, CUDA GPU) matches ORB-SLAM3
in simulation (kfvo: 0.012 m, ratio 0.987) but not on the real recording (0.025 m, 0.953). `depth0/` serves as
the exact reference for `scripts/stereo_depth_bias.py --gt-depth`. See `docs/results/lac_lunapolaris.md`.

## Aerial Lunaryard lighting sweep
A drone-like stereo survey over the OmniLRS **Lunaryard 80 m** (procedural DEM with craters, seed 42, 5 cm grid)
under four lighting conditions, with the same trajectory, terrain and rocks in all four. It asks whether learned
features (SP_SLAM3: SuperPoint + LightGlue) beat ORB under harsh lighting and faster aerial motion. Results:
[`docs/results/lunaryard_aerial_lighting.md`](../results/lunaryard_aerial_lighting.md).

### Reproduce
```bash
# DEM dump, poses on that DEM, 12-frame previews per condition, then the 4 full renders (one RTX 6000 job, ~1.5 h)
sbatch -p coc-gpu -q coc-ice --gres=gpu:rtx_6000:1 sim/omnilrs/lunaryard_aerial.sbatch      # STAGES="dem poses preview full"
# 12 estimator runs (ORB-SLAM3 on ice-cpu; SP_SLAM3 / LAC need a CUDA GPU)
for c in nominal low_sun dim glare; do
  sbatch scripts/slurm_slambench.sbatch run configs/omnilrs_lunaryard_aerial_${c}_orbslam3_stereo.yaml
  for e in spslam3_stereo lac_stereo_kfvo; do
    sbatch -p coc-gpu -q coc-ice --gres=gpu:v100:1 -c 8 --mem=32G -t 3:00:00 \
           scripts/slurm_slambench.sbatch run configs/omnilrs_lunaryard_aerial_${c}_${e}.yaml
  done
done
```
The renders are written to `renders/lunaryard_aerial_<condition>/` (layout as above; ~1.1–1.7 GB each).
`render_stereo.py --env lunaryard` options: `--yard-size 20|40|80`, `--seed`, `--sun-azimuth`, `--sun-elevation`,
`--sun-intensity` (the distant light is set directly and the stellar engine is disabled, so the lighting is
static), `--terrain-material Basalt|Sand|LunarRegolith8k`, `--texture-scale`, `--albedo-brightness`, `--rocks N`,
`--lens-flares on|off`, `--dem-only`, `--preview N`. `calib.json` records the environment, terrain material,
rock count, lighting (requested and measured sun direction, exposure/tonemapper settings) and whether the sun
disk is ever inside the left image. Lunaryard renders take ~0.35 s per stereo frame plus ~90 s scene setup.

### Settings
* **Trajectory** (`sim/omnilrs/aerial_poses.py`): lawnmower survey of 5 legs × 35.4 m along ±x, 10 m apart,
  joined by 5 m-radius semicircular turns at a constant yaw rate (0.4 rad/s). It covers x 17–63 m, y 20–60 m,
  so the camera is ≥ 10 m from every edge and the image footprint stays on the terrain. The drone flies at
  2 m/s (240 m in 120 s), 15 Hz, 1800 frames. Altitude is the DEM mean + 5 m (AGL 4.5–5.7 m, mean 5.0, stored per frame as
  `agl_m`). The camera looks along the direction of travel, pitched 60° down, and has smooth sinusoidal
  perturbations of ±0.3 m altitude and ±2° pitch/roll/yaw. Camera: the ZED2-like rectified 640×360 pinhole with
  `fx = fy = 263.27` and baseline 0.11983 m used for the LunaPolaris replays. Legs 0/2/4 fly +x and legs 1/3 fly −x.
* **Lighting** (`sim/omnilrs/render_lunaryard_sweep.sbatch`). Sun azimuth 180° puts the sun toward −y, across
  the legs; azimuth 270° puts it toward +x, i.e. ahead of the camera on legs 0/2/4.

  | condition | sun az / el | intensity | look |
  |---|---|---|---|
  | `nominal` | 180° / 45° | 1750 (cfg default) | evenly lit, short shadows, mean grey ≈ 100 |
  | `low_sun` | 180° / 7° | 1750 | grazing light, crater interiors and long rock shadows pitch black, mean grey ≈ 30 |
  | `dim` | 180° / 45° | 175 (÷10) | underexposed, mean grey ≈ 13 |
  | `glare` | 270° / 10° | 1750, lens flares on | sun ahead on +x legs (backlit, shadows toward the camera, bright forward-scatter sheen at the top of the frame), behind on −x legs; mean grey ≈ 55 |

  Exposure is fixed (auto-exposure `/rtx/post/histogram/enabled` off; tonemapper and camera f-stop/ISO at their
  defaults), so `dim` really is darker. **The sun disk is never in the image.** With a 60° pitch the top image row
  looks ≥ 24° below the horizon, and for `glare` the sun is ≥ 34° from the top-centre ray. Lens flares are
  enabled for `glare` but have nothing to flare from.
* **Terrain look.** The Lunaryard cfg's `Basalt` material (GravelStones, `texture_scale` 5 = texture repeating
  every 0.2 m, albedo 0.3) renders as a featureless grey at 5 m AGL. A nominal preview gave 4–24 ORB keypoints
  per frame (FAST 20), with only soft crater shading. The OmniLRS MDLs bake their parameters in (`material X(*)`),
  so `--texture-scale` / `--albedo-brightness` write a modified copy of the MDL next to the original and bind
  that. The sweep uses `--texture-scale 0.2 --albedo-brightness 0.6` (the gravel texture repeats every 5 m and
  reads as fine regolith from 5 m) and `--rocks 300` instead of the default 10, giving ~1000 ORB keypoints per
  nominal frame. Override this with `TERRAIN="..."`.
* **Estimators.** Settings are identical to the real LunaPolaris configs: `lunapolaris_seq01_orbslam3_stereo_lean`,
  `lunapolaris_seq01_spslam3_stereo` (SuperPoint thresholds 1e-5/1e-6, 1200 keypoints, LightGlue) and
  `lunapolaris_seq01_lac_stereo_kfvo`. Only the dataset block differs.

### Results (mean [min, max] over 3 runs for ORB-SLAM3 / SP_SLAM3, 1 run for LAC)
| lighting | ORB-SLAM3 ATE SE3 | SP_SLAM3 ATE SE3 | LAC kfvo ATE SE3 |
|---|---|---|---|
| nominal | **0.126** [0.117, 0.137] | 0.181 [0.135, 0.243] | 0.589 |
| low_sun | **0.175** [0.164, 0.192] | 0.249 [0.187, 0.284] | 0.600 |
| dim | **0.169** [0.151, 0.179] | 0.199 [0.138, 0.232] | 0.616 |
| glare | **0.152** [0.127, 0.191] | 0.387 [0.316, 0.469] | 0.691 |

Every successful run tracked 100% of the frames in one map, with no tracking loss. One ORB-SLAM3 nominal run
aborted at 22% of the sequence with the known `Sophus::SO3::exp` NaN abort; it was re-run. The full table
(Sim3, RPE, scale, frame time) and the interpretation are in the results file.

### Caveats
* No sensor noise, motion blur, rolling shutter, vignetting or auto-exposure. Real underexposed images would be
  noisy, so `dim` is optimistic, especially for FAST (ORB-SLAM3 falls back to its FAST threshold of 7 here).
* The terrain texture is a modified OmniLRS material, not a calibrated lunar regolith: it repeats every 5 m,
  and the 300 rocks come from OmniLRS's `lunalab_rocks` set. The DEM relief is gentle (craters, no large slopes).
* The sun is never in view, so `glare` tests backlit, low-sun scenes with specular sheen, not a sun disk or flare.
* The principal point is the image centre. The timestamps are synthetic (15 Hz from 1.8e18 ns). There is no IMU.
* SP_SLAM3 needs ~160 ms per frame on a V100, 2.4× the 67 ms frame period. Playback is throttled, not dropping
  frames, so it processed every frame but each run took 2.7× real time.
