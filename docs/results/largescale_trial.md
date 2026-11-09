# OmniLRS LargeScale aerial trial (Site20 DEM + streamed procedural terrain)

Trial of OmniLRS's `LargeScale` environment as an alternative to the 80 m procedural Lunaryard terrain
(aerial survey at 5 m AGL; the ground texture repeats every 5 m). LargeScale uses a real LOLA south-pole
DEM (Site20, 5 m/px, 16 × 16 km) and adds a procedural high-resolution layer (2.5 cm/px, craters and rocks)
that streams around the camera.

**Result:** the pipeline works end to end:
* renders at 30 m and 60 m AGL;
* no block popping;
* depth agrees with AGL;
* ORB-SLAM3 stereo tracks 100 % of frames at 30 m (it fails at 60 m).

Issues found:
* The high-resolution texture visibly tiled at 60 m. **This is fixed in v2** (section "v2: non-repeating
  ground"): a world-space, texture-bombed regolith MDL with large-scale albedo noise. No 2 m periodicity
  remains, image std rose from 5–7.5 to ~12, and ORB-SLAM3 now tracks all of the 60 m run.
* Building a new 50 m high-resolution block stalls the render for ~11–15 s, so throughput depends on the
  flight path.
* At 60 m, stereo scale is biased (~0.8) because the 0.12 m baseline gives sub-pixel disparity. **A wider
  rendered baseline fixes this** (section "Stereo baseline test"): with 0.5 m, ORB-SLAM3 reaches 0.53 m
  ATE at 60 m, against 11 m with 0.12 m.

v1 below refers to the first renders (OmniPBR `LunarRegolith8k`, job 6815226). Their images were deleted
after the v2 comparison; their metadata (`calib.json`, `terrain.csv`, GT) is kept.

## What was done

* **DEM:** downloaded only `Site20_final_adj_5mpp_surf` (41 MB tif) and wrote `dem.npy` (float32
  3200 × 3200, the raw raster) and `dem.yaml` (`center_coordinates` lon/lat, `pixel_size` [5, −5], `size`)
  to `OmniLRS/assets/Terrains/SouthPole/Site20_final_adj_5mpp_surf/`. This is the same output as
  OmniLRS's `scripts/extract_dems_override.sh` (in the OmniLRS checkout, not this repo), written with
  rasterio, which is now in the venv. The tif was then
  deleted (40 MB on disk).
* **Renderer:** `render_stereo.py --env largescale`.
  * Start position: `--ls-start`, default (2800, −2200) m from the DEM centre.
  * Lighting: fixed sun from `--sun-azimuth/--sun-elevation` (stellar engine off, as in Lunaryard).
  * Terrain streaming: before every frame, `LSTM.update_visual_mesh(cam xy)` runs, then
    `--mesh-settle` (1) extra `app.update()`. The call blocks until the HR blocks and the clipmap
    interpolation are done.
  * The assert `update_every_n_meters × texture_scale ∈ ℤ` keeps the texture fixed to the ground. The
    clipmap UVs are mesh-local, and the mesh moves in 2 m steps.
  * Outputs:
    * GT and image outputs have the same format as the other envs.
    * `depth0` is in **10 mm units** (`calib.json` `depth0_unit_mm`), because 1 mm overflows uint16 past
      65 m.
    * `dem.npy` is an LR-DEM crop in the local frame.
    * New per-frame `terrain.csv`: terrain z, AGL, mesh update time, render time.
* **Poses:** `aerial_poses.py --largescale-dem <dir> --terrain-follow`. The lawnmower is flown at
  constant AGL over a Gaussian-low-passed (2 s) DEM profile. The frame mapping is in
  `sim/omnilrs/largescale_dem.py`, tested in `tests/test_largescale_dem.py`. HR minus LR terrain height
  within 150 m of the start: mean 0.39 m, max 1.04 m, so the frames agree.
* **Trajectories** (30 s, 15 Hz, 450 frames, pitch 60°, the same as the Lunaryard survey):

  | | AGL | speed | path | legs × spacing | max footprint distance |
  |---|---|---|---|---|---|
  | `ls_aerial_30m` | 29.1–31.1 m | 5 m/s | 150 m | 2 × 30 m | 104 m |
  | `ls_aerial_60m` | 58.3–62.2 m | 8 m/s | 241 m | 2 × 60 m | 207 m |

  Pitch 60° was kept so the views match the existing survey. The top image edge is 26° below the
  horizon, so there is no sky. The 60 m footprint reaches beyond the fine clipmap (±~123 m) into the
  coarse mesh. No seam is visible there.
* **Look:** `--sun-azimuth 180 --sun-elevation 45 --albedo-add 0 --albedo-brightness 0.6`. The material
  default `LunarRegolith8k` has `albedo_add: 1` and renders pure white with auto-exposure off (test job
  6815063, mean 229).

Job `sim/omnilrs/largescale_trial.sbatch`: **6815226** (rtx_6000, both altitudes, 15 min). Test run with
30 frames: 6815063. Renders are in `/storage/ice1/3/6/bkuang7/omnilrs/renders/largescale_trial/`.

## Timings (Quadro RTX 6000, v1)

| | 30 m | 60 m |
|---|---|---|
| startup (Isaac boot + LargeScale build, ~155–200 s) | 228 s | 196 s |
| render only (`render_s`, 4 RT subframes, stereo + depth) | 0.30 s/frame | 0.33 s/frame |
| mesh updates (every 2 m) | 65, typically 0.3–0.5 s | 112 |
| 50 m block shifts (new HR blocks) | 3 × 11.5–13.1 s | 5 × 10.4–13.3 s (+2 × 2.7 s) |
| mean over the run | 0.43 s/frame | 0.55 s/frame |

So with LargeScale, a 30 s survey costs about 7.5 min of GPU time. Block shifts dominate at higher speed.
This is still about 2× slower per frame than Lunaryard, which needs no terrain updates.

## Image and geometry checks (v1)

* **Depth vs AGL:** at the image centre (pitch 60°) the median depth is 34.5 m at 29.5 m AGL (expected
  AGL / sin 60° = 34.1 m) and 69.7 m at 59.4 m AGL (expected 68.6 m). Valid depth on 100 % of pixels.
* **Popping:** none found. The mean absolute difference between consecutive frames is the same on
  mesh-update frames and block-shift frames as on other frames: 30 m 2.9 vs 2.6, 60 m 1.8 vs 1.7 (8-bit
  grey). Visually there are no half-loaded blocks. `update_visual_mesh` blocks until the data is ready.
* **Exposure:** mean grey 155, constant (154–157). Contrast is low (in-image std 7.5 at 30 m, 5.0 at 60 m;
  Lunaryard nominal is ~17). Most of the contrast comes from the procedural craters (0.25–2.5 m). Rocks are
  sub-pixel. The 5 m DEM relief appears only as soft shading.
* **Texture tiling:**
  * The LunarRegolith8k albedo is nearly flat, but its normal map has a 2 m period (`texture_scale` 0.5).
  * At 30 m it shows as faint regular banding, visible on a contrast-stretched crop.
  * At 60 m it is a clearly visible 2-D grid of ~7 px period across the whole image. It is fixed to the
    ground, so it is a genuine repeating ground texture, a stronger version of the Lunaryard 5 m
    problem.
* Previews are in the session folder only (not in git): contact sheets of 12 frames per run, a single
  frame, a contrast-stretched crop and a depth image for each altitude, and a Site20 hillshade. The v2
  previews (`ls30v2_*`, `ls60v2_*`, `ls60_vs_v2_*`) are there too.

## SLAM sanity check (v1; ORB-SLAM3 stereo, settings as for the Lunaryard aerial survey)

| config | run | tracked | maps | final ATE RMSE (SE3) | RPE 1 s |
|---|---|---|---|---|---|
| `configs/omnilrs_largescale_aerial_30m_orbslam3_stereo.yaml` | 20261104-181118_b0001b | 100 % | 1 | 0.52 m (max 1.20) on 150 m | 0.25 m |
| `configs/omnilrs_largescale_aerial_60m_orbslam3_stereo.yaml` | 20261104-181509_7c11db | 89 % (2 losses) | 2 | 23.9 m (max 40.3) on 241 m | 8.1 m |

These are the first runs; the repeats are in the v2 table.

The stereo baseline is the ZED2's 0.12 m. That gives ~0.9 px disparity at 34 m range and ~0.45 px at
70 m, so stereo depth is weak at these altitudes:
* 30 m tracks well: 0.52 m ATE, 0.35 % of the path.
* 60 m loses tracking twice and splits the map. The low texture contrast and the repeating grid probably
  contribute.

Do not compare these numbers with the 5 m Lunaryard results.

## v2: non-repeating ground (`--terrain-shader bombed`)

`sim/omnilrs/regolith_mdl.py` generates an MDL material that `render_stereo.py` binds to both clipmap
meshes instead of OmniPBR. The approach worked in the material itself, so no offline texture was needed.
* **World-space projection:** lookups use the world-space shading position plus the LargeScale
  `starting_position`. The pattern is therefore fixed to global DEM coordinates. It does not depend on
  the mesh UVs or the 2 m mesh-update step (the texture/update-step assert is skipped for this shader).
* **Texture bombing:**
  * A 1 m lattice; every lattice point gets a hashed random rotation and UV offset into the same
    LunarRegolith8k diffuse, normal and AO maps (2 m period).
  * The 4 surrounding lattice points are blended with smoothstep weights.
  * Diffuse, normal and AO deviations use a variance-preserving blend:
    mean + Σ wᵢ(Tᵢ − mean) / √Σ wᵢ² (Heitz & Neyret 2018). Texture means were measured on the 8k maps.
  * Each tangent-space normal sample is rotated back to world xy.
  * Lookups are made relative to the lattice point to keep float precision at km-scale coordinates.
* **Albedo variation:**
  * 4-octave value noise, wavelengths 48 / 23 / 11 / 5.3 m, gain 0.7.
  * Normalized in numpy (`fbm_np`, mirrors the MDL hash) to std = amp/2, clamped at ±1.5 amp.
  * Trial setting `--bright-amp 0.3`: σ = 15 %, so ±30 % ≈ 2σ, slightly above the ±10–20 % first
    suggested.
  * Tone mapping compresses albedo contrast (pixel value ∝ albedo^~0.6). On a test plane, ±20 % only
    reached std 8 and ±30 % reached 11–12.
* **Look:**
  * The BSDF is Lambertian, which is close to OmniPBR at roughness 1.
  * `--albedo-brightness 1.0`. The Lambertian MDL renders darker than OmniPBR at the same value
    (plane test: mean 88 vs 163 at 0.6).
  * Mean grey is 112–116 (Lunaryard nominal is ~100). The texture's slight brown tint is kept.
* **Not done:**
  * Crater-correlated darkening. Craters are mesh geometry, not material inputs, so this would need a
    crater mask.
  * The v2 MDL ignores `--albedo-add` and `--texture-scale`.

Job **6824611** (`LOOK="--sun-azimuth 180 --sun-elevation 45 --terrain-shader bombed --albedo-brightness
1.0 --bright-amp 0.3" SUFFIX=_v2 ALTS="60 30" sbatch ... largescale_trial.sbatch`) rendered the same
trajectories with the same sun to `ls_aerial_{30,60}m_v2/`.

**Periodicity check** (session script `ls_grid.py`, not in the repo):
* 12 frames per run are orthorectified to a 0.5 m world-xy grid with GT depth and pose (range < 150 m),
  over a 32 × 32 m patch.
* The patch is high-passed (σ 2 m) and normalized.
* Metrics are the autocorrelation at the 2 m lag minus the mean at the 1 m and 3 m lags, and the FFT
  power at (±0.5, 0) / (0, ±0.5) cycles/m divided by the median power on that frequency ring.
* Control rings at 0.37 and 0.63 cycles/m give the noise floor.

| run | autocorr excess at 2 m | FFT peak / ring median at 0.5 cyc/m | control 0.37 / 0.63 | image mean | image std (in-frame) |
|---|---|---|---|---|---|
| 30 m v1 | 0.068 | **41.0** | 10.4 / 16.5 | 155 | 7.5 |
| 30 m v2 | 0.023 | 13.8 | 10.7 / 12.4 | 112 | 12.2 |
| 60 m v1 | 0.191 | **128.1** | 8.6 / 12.8 | 156 | 5.0 |
| 60 m v2 | 0.013 | 6.1 | 5.6 / 9.3 | 116 | 11.7 |

* The 2 m peak drops to the noise floor in both v2 runs, and the grid is no longer visible.
* Albedo blotches of 5–50 m are fixed to the ground and the craters remain.
* Nothing clips: no pixel reaches 254.
* Std rose to ~12 but is still below Lunaryard's ~17. A larger `--bright-amp` or a lower sun would close
  the gap.
* Popping: consecutive-frame differences are the same on update and non-update frames (60 m: 2.67 vs
  2.62).
* Depth vs AGL is unchanged (34.5 / 69.7 m at the centre).

**Timing** (same node type): render 0.29–0.30 s/frame (v1 0.30–0.33). The run mean is 0.43 s/frame
at 30 m and 0.54 s/frame at 60 m (v1 0.43 / 0.55). Block-shift stalls are 11.6–15.4 s (v1 10.4–13.3 s),
which is within node-to-node variation. The shader costs nothing measurable.

**ORB-SLAM3 stereo** (same settings, 3 runs each on `ice-cpu` gold6226, medians). Run ids are in
`~/scratch/slam-bench-work/runs/omnilrs_largescale_aerial_{30m,60m}{,_v2}_orbslam3_stereo/`.

| | tracked | losses | maps | final ATE SE3 (median, range) | Sim3 ATE / scale | RPE 1 s |
|---|---|---|---|---|---|---|
| 30 m v1 | 100 % | 0 | 1 | 0.54 m (0.52–0.66) | 0.24 m / 0.98 | 0.25 m |
| 30 m v2 | 100 % | 0 | 1 | 0.61 m (0.50–0.65) | **0.12 m** / 0.98 | **0.16 m** |
| 60 m v1 | 79–89 % | 2–4 | 2–3 | 23.9 m (23.4–26.1) | 7.8–8.6 m / ~3.9 (largest map) | 8.0 m |
| 60 m v2 | **100 %** | **0** | **1** | 11.0 m (10.7–25.8) | 0.48 / 0.51 / 23 m; scale 0.79 | 2.1 m |

* **30 m:** v2 halves the Sim3 error and the 1 s RPE. The SE3 ATE is dominated by a 2 % scale error
  and is unchanged.
* **60 m:** v2 stops the tracking losses and map splits. In 2 of 3 runs the shape is right (Sim3 0.5 m),
  but stereo scale is 0.79. With ~0.45 px disparity, the stereo depth bias is the remaining error, not
  the texture. The third run diverged.

v2 SLAM configs: `configs/omnilrs_largescale_aerial_{30m,60m}_v2_orbslam3_stereo.yaml`. The v1 configs
can no longer run because their images were deleted; neither can the v2 ones since the later cleanup
(see "Disk" under "Map reuse across lighting"). Such configs are marked `# HISTORICAL` at the top.

## Stereo baseline test at 60 m (and 30 m)

**Question:** is the 0.12 m ZED2 baseline what limits ORB-SLAM3 at 60 m AGL? **Yes.**
With a 0.5 m or 1.0 m baseline, ORB-SLAM3 is as accurate at 60 m as it was at 30 m: the 0.79 scale bias
and the diverging run both go away.

**Setup**
* Same as v2: the same 60 m trajectory (left-camera poses identical to v2), the same fixed sun
  (az 180°, el 45°), and the bombed ground shader.
* Only the right camera moved. `aerial_poses.py --baseline-m` sets the baseline, and the sbatch passes
  it through `BASELINE`:
  `LOOK="$LOOK_V2" BASELINE=0.5 SUFFIX=_b050 ALTS=60 sbatch ... -c 8 --mem=48G sim/omnilrs/largescale_trial.sbatch`.
* Render jobs: 6831796 (60 m, 0.5 m), 6831797 (60 m, 1.0 m), 6831798 (30 m, 0.5 m), each on one
  RTX 6000 with 8 CPUs.

**Checks**
* `calib.json` `baseline_m` (0.5 / 1.0) flows through the dataset loader to ORB-SLAM3 `Stereo.b`.
* The left images match v2 (mean abs difference 0.09 grey levels).
* Warping the right image onto the left with GT depth and the 0.5 m baseline gives 0.58 grey levels of
  mean error. With 0 / 0.12 / 1.0 m the error is 1.6–2.0.

**Expected disparity** fx·b/Z at the centre-pixel depth (fx 263.27 px):

| baseline | 60 m (Z ≈ 69.9 m) | 30 m (Z ≈ 34.6 m) |
|---|---|---|
| 0.12 m | 0.45 px | 0.91 px |
| 0.5 m | 1.88 px | 3.8 px |
| 1.0 m | 3.77 px | 7.6 px |

**`Stereo.ThDepth`** (close/far threshold in baselines).
* The wide-baseline configs use ThDepth = 100 m / b: 200 for 0.5 m and 100 for 1.0 m. Points closer
  than 100 m, which is most of the image (median depth 70 m), are "close", so each keyframe creates
  stereo map points from them.
* Two controls at 60 m, 3 runs each:
  * 0.12 m with the same 100 m split (ThDepth 834.5) is much worse than ThDepth 40. Sub-pixel-disparity
    points then become close points: ATE 26.4 m, Sim3 26.0 m. So the 0.12 m reference keeps ThDepth 40.
  * 0.5 m with ThDepth 40 (every point "far") still works well: ATE 0.99 m, Sim3 0.37 m / scale 0.979.
    Most of the gain therefore comes from the baseline, and the 100 m split adds a bit more.

**ORB-SLAM3 stereo** (3 runs each on `ice-cpu` gold6226; medians, with the range of the SE3 ATE).
Raw ATE is the final SE3-aligned ATE. Sim3 aligns scale too (session script `ls_sim3.py`, not in the repo; Umeyama). The 1 s drift is
RPE translation over 1 s. Run ids are under
`~/scratch/slam-bench-work/runs/omnilrs_largescale_aerial_<cfg>_orbslam3_stereo/`.

| AGL | baseline (ThDepth) | tracked | losses | maps | raw ATE | Sim3 ATE / scale | 1 s drift |
|---|---|---|---|---|---|---|---|
| 60 m | 0.12 m (40), v2 reference | 100 % | 0 | 1 | 11.0 m (10.7–25.8) | 0.51 m / 0.79 (one run 23 m / 0.75) | 2.1 m |
| 60 m | 0.12 m (834.5), control | 100 % | 0 | 1 | 26.4 m (26.0–27.9) | 26.0 m / 0.94 | 6.6 m |
| 60 m | 0.5 m (40), control | 100 % | 0 | 1 | 0.99 m (0.96–1.10) | 0.37 m / 0.979 | 0.36 m |
| 60 m | **0.5 m (200)** | 100 % | 0 | 1 | **0.53 m (0.46–0.61)** | **0.27 m / 0.989** | **0.28 m** |
| 60 m | **1.0 m (100)** | 100 % | 0 | 1 | 0.76 m (0.69–0.81) | 0.39 m / 0.985 | 0.29 m |
| 30 m | 0.12 m (40), v2 reference | 100 % | 0 | 1 | 0.61 m (0.50–0.65) | 0.12 m / 0.977 | 0.16 m |
| 30 m | 0.5 m (200) | 100 % | 0 | 1 | **0.31 m (0.29–0.41)** | 0.17 m / 0.990 | 0.14 m |

**Findings**
* **60 m:**
  * A 0.5 m baseline cuts raw ATE about 20× (11 m to 0.53 m) and the 1 s drift about 7× (2.1 m to
    0.28 m). The scale error drops from 21 % to about 1 %, and all 3 runs agree.
  * 1.0 m is not better than 0.5 m (0.76 m ATE). The remaining ~1–1.5 % scale error and sub-metre
    shape error are not disparity-limited.
* **30 m:**
  * 0.5 m halves raw ATE (0.61 m to 0.31 m), mainly through scale (0.977 to 0.990).
  * The Sim3 shape error is unchanged within the spread (0.12 vs 0.17 m).
* **Rendering** is unchanged: 0.42–0.53 s/frame over the run, with 474–552 s per job including startup.
  The baseline does not affect render cost.

**Disk:** the three new renders take 895 MB (292–302 MB each). The v2 0.12 m renders are kept as the
reference.

**Configs:** `configs/omnilrs_largescale_aerial_{60m_b050,60m_b100,30m_b050}_orbslam3_stereo.yaml`, plus
the controls `..._60m_b012_th100m_...` and `..._60m_b050_th40_...`.

**Preview:** `ls60b_stereo.png` (session files) shows left and right images for frame 200 at each
baseline.

## Map reuse across lighting (60 m and 30 m, 0.5 m baseline)

Repeats the Lunaryard aerial map-reuse experiment ([map_reuse.md](map_reuse.md), sim tables;
[place_recognition.md](place_recognition.md)) on LargeScale. It uses the v2 shader, a 0.5 m baseline and
`Stereo.ThDepth` 200 (100 m / b).

### Setup
* **Lighting:** the four Lunaryard sweep conditions (`sim/omnilrs/render_lunaryard_sweep.sbatch`):
  * nominal az 180 / el 45;
  * low_sun az 180 / **el 13** (Lunaryard: 7);
  * dim az 180 / el 45 at intensity 175;
  * glare az 270 / el 10 with lens flares.
  * Azimuth 180 puts the sun across the ±x survey legs, and 270 ahead on the +x legs, as on Lunaryard,
    so no azimuth change was needed.
  * **Adaptation:** the Site20 ground under the survey slopes ~6° down toward +y, away from an
    azimuth-180 sun. At el 7 a DEM ray-march puts 34 % of the area in shadow, and the rest at grazing
    incidence (median cos 0.025, nearly black). El 13 gives 0.3 % shadow and the same 7° incidence on
    the mean terrain plane that Lunaryard's el 7 gives on flat ground.
  * The sun is out of view at the 60° camera pitch, so glare is a low, frontal, shadow-flipping sun.
    No flare artifacts are visible.
* **Passes** (`sim/omnilrs/largescale_sweep.sbatch`, jobs 6842727 for 60 m and 6842728 for 30 m):
  * **Map:** the existing nominal b050 renders (`largescale_trial/ls_aerial_{60,30}m_b050`).
  * **Query:** the same lawnmower, with the centre shifted by (0.2, 0.25) × leg spacing (`--offset-xy 12
    15` at 60 m, `6 7.5` at 30 m) and other perturbations (`--seed 1`). It is rendered under all four
    lightings.
  * The query flies 15–19 m (60 m) or 8–9 m (30 m) from the map path. That is about 12° of viewpoint
    change at the 70 m / 35 m centre range. Every query view centre lies within 25 m of a map view
    centre with a similar heading.
  * Unlike Lunaryard (identical trajectory), each query therefore has a viewpoint change as well as a
    lighting change. The **same** row (the map images themselves) is the sanity check and the error
    floor; **nominal** isolates the viewpoint change.
* **Image statistics** (left image, mean / std of 8-bit grey):

  | condition | LargeScale 60 m | LargeScale 30 m | Lunaryard 5 m |
  |---|---|---|---|
  | nominal | 116 / 11.9 | 112 / 12.0 | 102 / 16.9 |
  | low_sun | 32 / 8.8 | 31 / 10.1 | 30 / 20.7 |
  | dim | 16 / 3.0 | 15 / 2.9 | 13 / 3.4 |
  | glare | 73 / 11.1 | 69 / 12.4 | 55 / 25.5 |

  Nothing saturates. Render time is 189–236 s per 450 frames per condition, 430–531 s with startup.
* **Estimators:** the Lunaryard map-reuse settings, except ThDepth:
  * ORB-SLAM3 with DBoW2 (`orbslam3_4452a3c_driver3.sif`);
  * SP_SLAM3 r3 with DBoW3 (`spslam3_9439971_r3.sif`, map `..._spslam3_stereo_map`);
  * SP_SLAM3 r6 in the final recommended configuration (`spslam3_9439971_r6.sif`): unit descriptors +
    `map_anchor`, learned relocalization with the LightGlue verifier, `consistent_frames 1`, and map
    `..._spslam3_stereo_map_gd_unitdesc`.
  * Session B is `reuse_mode: relocalize`, scored in the session-A map frame with the session-A
    map→GT transform (`slambench/evaluation/map_reuse.py`).
  * Each cell is the median of 3 session-B runs against the latest of 3 maps. Maps and queries are
    each run 3×.
* **Thresholds:** the standard false-relocalization threshold (0.5 m / 15°) is below the map's own
  fixed-frame error at 60 m (see the same rows). The tables therefore also give an altitude-scaled
  wrong-place check: an event or frame counts at 2 m (60 m) or 1 m (30 m).

### Mapping (session A, nominal, median of 3)

| AGL | estimator | raw ATE (SE3) | Sim3 ATE / scale | KFs | atlas |
|---|---|---|---|---|---|
| 60 m | ORB-SLAM3 | **0.46 m** [0.44–0.80] | 0.32 m / 0.992 | 130 | 22 MB |
| 60 m | SP_SLAM3 r3 | 2.47 m [2.15–2.81] | 0.77 m / 1.059 | 88 | 132 MB |
| 60 m | SP_SLAM3 r6 (gd, unitdesc) | 1.32 m [1.29–1.79] | 0.57 m / 1.029 | 87 | 133 MB |
| 30 m | ORB-SLAM3 | 0.38 m [0.29–0.40] | 0.19 m / 0.991 | 35 | 8 MB |
| 30 m | SP_SLAM3 r3 | 0.31 m [0.23–0.33] | 0.24 m / 1.005 | 30 | 46 MB |
| 30 m | SP_SLAM3 r6 (gd, unitdesc) | **0.21 m** [0.19–0.29] | 0.18 m / 1.006 | 50 | 77 MB |

All runs track 100 % in one map. At 60 m SP_SLAM3 **overestimates scale by 3–6 %**. ORB does not (0.8 %
under). At 1.9 px centre disparity, SuperPoint's keypoint localization is not precise enough for stereo
depth. Session B uses the latest map:
* 60 m: ORB `20261106-091833_e6929b` (0.80 m), r3 `..._7f7471` (2.47 m), r6 `..._622c5f` (1.32 m);
* 30 m: ORB `..._a7d229` (0.29 m), r3 `..._44e929` (0.23 m), r6 `..._5c6d87` (0.29 m).

These map errors set the floor of the session-B errors.

### Relocalization (session B, map = nominal, median of 3)

Columns:
* localized: fraction of input frames localized in the map; brackets give the min–max over the 3 runs;
* ≤ thr: fraction localized and within that error;
* first loc: data time to the first localization;
* false reloc: relocalization events beyond 0.5 m / 15° and beyond the altitude-scaled threshold, then
  the total number of events, summed over 3 runs;
* median / p90: fixed-frame translation error of localized frames.

**60 m AGL**

| estimator | query | localized | ≤0.5 m | ≤2 m | first loc | false @0.5 m / @2 m / events | median | p90 | ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| ORB-SLAM3 | same | 1.000 | 0.020 | 1.000 | 0.0 s | 3 / 0 / 3 | 0.79 m | 0.99 m | 22 |
| ORB-SLAM3 | nominal | 0.898 | 0.051 | 0.896 | 3.1 s | 3 / 3 / 3 | 0.76 m | 1.07 m | 22 |
| ORB-SLAM3 | low_sun | **0** (init_failure) | 0 | 0 | – | – | – | – | 22 |
| ORB-SLAM3 | dim | **0** (init_failure) | 0 | 0 | – | – | – | – | 8 |
| ORB-SLAM3 | glare | **0** (init_failure) | 0 | 0 | – | – | – | – | 22 |
| SP_SLAM3 r3 | same | 1.000 | 0.000 | 0.420 | 0.0 s | 3 / 3 / 3 | 2.23 m | 3.34 m | 196 |
| SP_SLAM3 r3 | nominal | 0.907 [0.84–0.96] | 0.004 | 0.216 | 2.8 s | 3 / 0 / 3 | 2.76 m | 4.16 m | 192 |
| SP_SLAM3 r3 | low_sun | 0.000 (2 frames) | 0 | 0 | 1.9 s | 2 / 0 / 2 | 1.04 m | 1.29 m | 145 |
| SP_SLAM3 r3 | dim | 0.889 [0.15–0.92] | 0.011 | 0.202 | 3.3 s | 4 / 2 / 4 | 2.81 m | 4.26 m | 191 |
| SP_SLAM3 r3 | glare | **0** (init_failure) | 0 | 0 | – | – | – | – | 142 |
| SP_SLAM3 r6 | same | 1.000 | 0.013 | 0.960 | 0.0 s | 3 / 0 / 3 | 1.14 m | 1.85 m | 202 |
| SP_SLAM3 r6 | nominal | 1.000 | 0.067 | 0.802 | 0.0 s | 3 / 0 / 3 | 1.45 m | 2.40 m | 193 |
| SP_SLAM3 r6 | low_sun | 0.873 [0.86–0.94] | 0.024 | 0.236 | 0.0 s | 94 / 3 / 115 | 3.71 m [2.7–23.6] | 22.7 m | 234 |
| SP_SLAM3 r6 | dim | 1.000 | 0.062 | 0.796 | 0.0 s | 3 / 0 / 3 | 1.45 m | 2.34 m | 192 |
| SP_SLAM3 r6 | glare | 0.789 [0.22–0.83] | 0.016 | 0.167 | 0.0 s | 202 / 22 / 222 | 5.07 m [1.3–6.4] | 23.7 m | 242 |

**30 m AGL**

| estimator | query | localized | ≤0.5 m | ≤1 m | first loc | false @0.5 m / @1 m / events | median | p90 | ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| ORB-SLAM3 | same | 1.000 | 0.947 | 1.000 | 0.0 s | 0 / 0 / 3 | 0.24 m | 0.41 m | 23 |
| ORB-SLAM3 | nominal | 0.960 | 0.960 | 0.960 | 1.2 s | 0 / 0 / 3 | 0.26 m | 0.40 m | 22 |
| ORB-SLAM3 | low_sun | **0** (init_failure) | 0 | 0 | – | – | – | – | 22 |
| ORB-SLAM3 | dim | **0** (init_failure) | 0 | 0 | – | – | – | – | 12 |
| ORB-SLAM3 | glare | **0** (init_failure) | 0 | 0 | – | – | – | – | 23 |
| SP_SLAM3 r3 | same | 0.882 | 0.882 | 0.882 | 0.0 s | 0 / 0 / 105 | 0.16 m | 0.34 m | 201 |
| SP_SLAM3 r3 | nominal | 0.467 [0.40–0.49] | 0.464 | 0.467 | 0.5 s | 3 / 0 / 110 | 0.19 m | 0.31 m | 178 |
| SP_SLAM3 r3 | low_sun | **0** (init_failure) | 0 | 0 | – | – | – | – | 142 |
| SP_SLAM3 r3 | dim | 0.191 [0.14–0.20] | 0.191 | 0.191 | 0.3 s | 0 / 0 / 60 | 0.25 m | 0.38 m | 179 |
| SP_SLAM3 r3 | glare | **0** (init_failure) | 0 | 0 | – | – | – | – | 139 |
| SP_SLAM3 r6 | same | 1.000 | 0.944 | 1.000 | 0.0 s | 0 / 0 / 3 | 0.26 m | 0.41 m | 189 |
| SP_SLAM3 r6 | nominal | 1.000 | 0.922 | 1.000 | 0.0 s | 0 / 0 / 3 | 0.31 m | 0.48 m | 188 |
| SP_SLAM3 r6 | low_sun | 0.769 [0.76–0.92] | 0.276 | 0.569 | 0.0 s | 29 / 6 / 241 | 0.64 m | 2.25 m | 237 |
| SP_SLAM3 r6 | dim | 0.998 | 0.920 | 0.998 | 0.0 s | 0 / 0 / 5 | 0.32 m | 0.48 m | 194 |
| SP_SLAM3 r6 | glare | 0.593 [0.59–0.59] | 0.182 | 0.296 | 0.0 s | 41 / 3 / 262 | 1.04 m | 4.40 m | 248 |

Notes:
* `init_failure` means the run never localized in 450 frames. The 60 m SP_SLAM3 r3 low_sun runs
  localized 1–2 frames in total.
* Rows with no brackets have identical values across the 3 runs.
* **No relocalization event is wrong-place.** Across all r6 low_sun and glare runs, the largest error at
  a relocalization event is 3.7 m (median 0.1–1.3 m). Every error above 5 m comes from tracking after a
  correct relocalization:
  * In 2/3 of the 60 m low_sun runs and 2/3 of the 60 m glare runs, relocalization events stop after
    7–11 s.
  * The query then keeps tracking against the map (state OK, 200–400 matches per frame, not VO-only),
    but the error grows steadily to 30–46 m by the end. Example:
    `runs/omnilrs_lsmapreuse_60m_mapnominal_low_sun_spslam3_learnedreloc_unitdesc_anchor_gate1/20261106-095800_707814/eval/map_reuse.png`.
  * This **silent divergence** is not counted as a false relocalization, and it is the dangerous case
    here.
  * At 30 m it is rare: 9 % of the frames of one low_sun run are above 5 m.

### Offline VPR on the LargeScale set

`scripts/vpr_benchmark.py --suite largescale{60,30}`:
* Sessions come from `vpr_prepare_sessions.py --largescale`.
* **Database:** every 5th map-pass frame (90 frames, 2.7 m / 1.7 m apart).
* **Queries:** every 5th frame of each query pass (90 each).
* **Positive:** a database camera within 25 m (60 m AGL) or 12.5 m (30 m AGL) and 30° of the query.
  Chance R@1 is therefore 11–13 %, against 1.7 % on Lunaryard.
* **Success:** < 1 m (60 m) or < 0.5 m (30 m) and < 2°.
* **Gross false positive:** error beyond the positive radius or 10°.

**Recall@1 (%), map = nominal map pass**

| query | 60 m: DBoW2 / AnyLoc / MixVPR / CosPlace | 30 m: same order | Lunaryard 5 m: same order |
|---|---|---|---|
| nominal (offset pass) | 94 / 100 / 98 / 88 | 92 / 99 / 94 / 94 | – |
| low_sun | 22 / 40 / 66 / 20 | 22 / 79 / 44 / 24 | 3 / 85 / 56 / 2 |
| dim | 26 / 93 / 91 / 63 | 23 / 92 / 83 / 71 | 61 / 100 / 99 / 97 |
| glare | 26 / 29 / 17 / 6 | 17 / 27 / 13 / 6 | 2 / 9 / 12 / 4 |

**End-to-end success (%)**: SP+LightGlue verifier, best of top-5, GT depth. The oracle retrieves the
GT-nearest database frame.

| query | 60 m: DBoW2 / AnyLoc / MixVPR / CosPlace / oracle | 30 m: same order |
|---|---|---|
| low_sun | 72 / 73 / **97** / 61 / 98 | 76 / 94 / **96** / 69 / 100 |
| dim | 78 / **100** / 97 / 90 / 100 | 79 / 99 / 99 / **100** / 99 |
| glare | **58** / 47 / 32 / 40 / 98 | **44** / 26 / 40 / 23 / 86 |

**Gross false positives (% of all cross-session queries)**: SP+LG, top-5, GT depth, pooled over the four
query passes. The cell shows ≥ 20 / ≥ 50 inliers.

| retrieval | Lunaryard 5 m | LargeScale 60 m | LargeScale 30 m |
|---|---|---|---|
| DBoW2 | 8.0 / 3.3 | 0.6 / 0.6 | 1.1 / 1.1 |
| AnyLoc | 3.0 / 0.3 | 0.3 / 0.3 | 2.2 / 2.2 |
| MixVPR | 4.9 / 0.8 | 1.1 / 0.8 | 1.9 / 1.9 |
| CosPlace | 5.2 / 0.8 | 1.1 / 0.8 | 0.8 / 0.6 |
| oracle (GT-nearest frame) | – | 0.6 / 0.6 | 1.7 / 1.7 |

* **The texture alias is gone.**
  * The oracle, which always picks the correct place, has the same 0.6–1.7 % gross-FP floor. So almost
    all of the remaining gross false positives are bad PnP poses at the right place, mostly under
    glare at 30 m.
  * Split by whether the chosen frame is a GT positive: DBoW2 and AnyLoc have 0 % gross FP among
    wrong-place picks, and MixVPR and CosPlace have 4–9 % of a small wrong-place set (4–10 % of localized
    queries).
  * On Lunaryard, 55–83 % of the ≥ 50-inlier gross false positives were texture-alias wrong places.
* **DBoW2 retrieval recovers.** It goes from 2–3 % R@1 on Lunaryard to 17–26 %, and end-to-end success
  is 58–78 % (against 15 % on Lunaryard low_sun). The real DEM relief and the larger footprint at
  altitude give BoW distinct words. R@1 is still well below the learned methods, and chance is higher
  here (11–13 %).
* **Glare is still retrieval-limited, as on Lunaryard.** The oracle verifies 86–98 %; the best
  retriever gets 44–58 %.
* **Dim:** AnyLoc and MixVPR stay at 83–93 % R@1, while DBoW2 drops to 23–26 %. This matches the
  ORB-SLAM3 init_failure on dim.
* **Offline stereo depth at 60 m is too noisy for the 1 m success threshold.** With SGBM depth on the
  0.5 m baseline (~70 m range), even the oracle reaches only 20–30 % success. The median localized error
  is 3.5–6 m, and gross FP at ≥ 50 inliers is 6–11 %, which falls to ~1 % at ≥ 200 inliers.
  * At 30 m, stereo depth works: oracle 63 % success, gross FP ≤ 0.3 %.
  * SLAM uses triangulated multi-view map points rather than single-pair SGBM, which is why the SLAM
    relocalization events above stay below 3.7 m.

### Comparison with Lunaryard (5 m AGL, identical trajectory)

| | Lunaryard 5 m | LargeScale 30 m | LargeScale 60 m |
|---|---|---|---|
| ORB-SLAM3 / SP_SLAM3 r3, DBoW | nominal and dim ✓; low_sun and glare init_failure | ORB nominal only; r3 nominal 47 %, dim 19 %, low_sun/glare init_failure | ORB nominal only; r3 nominal 91 %, dim 89 %, low_sun/glare ✗ |
| SP_SLAM3 r6 nominal / dim | 0.999 / – localized, 0.14 m | 1.000 / 0.998, 0.31 m | 1.000 / 1.000, 1.45 m (map floor 1.14 m) |
| SP_SLAM3 r6 low_sun | 0.999, 0.137 m | 0.769, 0.64 m, p90 2.3 m | 0.873, 3.7 m, p90 23 m (divergence) |
| SP_SLAM3 r6 glare | 0.978, 0.204 m, p95 0.83 m | 0.593, 1.04 m, p90 4.4 m | 0.789, 5.1 m, p90 24 m (divergence) |
| wrong-place relocalizations | 0/3 (r6) | 0 events > 5 m | 0 events > 5 m |

The Lunaryard r6 rows are the r6 sim non-regression check in [map_reuse.md](map_reuse.md) (default
gate, single run).

**Takeaways**
* **Cross-lighting relocalization is harder at altitude on real-DEM terrain, not easier.**
  * DBoW relocalization (ORB-SLAM3 and SP_SLAM3 r3) still fails on low_sun and glare, as on Lunaryard.
  * ORB-SLAM3 now also fails on dim. Image std is 3, and FAST finds too few corners at 60 m
    (8–12 ms/frame).
  * SP_SLAM3 r6 still localizes on every condition, but low_sun and glare drop from 0.98–0.999 to
    0.59–0.87 localized.
  * The causes: real DEM relief changes the shadows more than flat procedural ground does, and the query
    adds a 12° viewpoint change.
* **Accuracy at 60 m is map-limited.**
  * With the identical images, error is 0.8 m (ORB) or 1.1 m (r6), because the SP map has a 3 % scale
    error.
  * The 0.5 m false-relocalization threshold from the 5 m study is meaningless here: ORB's same row
    counts 3/3 "false". Use an altitude-scaled threshold.
* **The texture-alias false-positive problem is gone at the SLAM level.** No relocalization event lands
  more than 3.7 m off. Offline, gross FP at ≥ 50 inliers is 0.3–2.2 %, the same as the GT-nearest
  oracle, so wrong-place aliasing is gone (on Lunaryard it was 0.3–3.3 %, mostly texture aliases).
  * **The failure mode moves to silent post-relocalization divergence** at 60 m under low_sun or glare:
    the error grows to 30–46 m while the system reports tracking OK.
  * That needs a consistency monitor: re-verify against the map, or check the error against an
    independent prior. A stricter relocalization gate does not help.
* **30 m is the better operating point for map reuse:** ORB 0.96 localized at 0.26 m across the
  viewpoint change, and r6 ≥ 0.92 within 0.5 m on nominal and dim.

**Configs:**
* Session A: `configs/omnilrs_largescale_aerial_{60,30}m_b050_{orbslam3_stereo_map,spslam3_stereo_map,spslam3_stereo_map_gd_unitdesc}.yaml`.
* Session B:
  `configs/omnilrs_lsmapreuse_{60,30}m_mapnominal_{same,nominal,low_sun,dim,glare}_{orbslam3_relocalize,spslam3_relocalize,spslam3_learnedreloc_unitdesc_anchor_gate1}.yaml`.

**Jobs:**
* Renders 6842727 and 6842728.
* Maps: ORB 6842754–6842756 (60 m) and 6842758–6842760 (30 m); SP 6842757 (60 m) and 6842761 (30 m).
* Session B: ORB 6842957/59/61 (60 m) and 6842963/65/67 (30 m); SP 6842958/60/62 (60 m) and
  6842964/66/68 (30 m). Batch jobs end FAILED whenever a run ends in init_failure, as expected.
* VPR 6844336 (60 m) and 6844337 (30 m), V100. The first submission, 6843959/60, read the 10 mm GT
  depth unit as mm and capped depth at 60 m. Its recall numbers are identical; its end-to-end numbers
  are invalid.

All runs are in `runs/<config>/20261106-09xx–10xx_*`.

**Disk** (net about +1.7 GB):
* query renders 2.1 GB (8 × 260–270 MB);
* the 6 atlases kept, 0.42 GB (the older repeats' atlases were deleted after their metrics were
  recorded);
* session-B runs 0.1 GB;
* the v2 0.12 m and 60 m b100 images deleted (−0.9 GB, metadata kept).

**Previews:** `ls60sweep_contact.png` and `ls30sweep_contact.png` (session files) show 4 frames per
lighting.

## Issues and recommended next steps

1. **Tiling:** fixed by v2 (above). Remaining steps:
   * the sim false-positive rates are re-measured on this terrain: the alias is gone (see "Map reuse
     across lighting");
   * consider a stronger albedo variation or a lower sun if Lunaryard-level contrast is needed.
2. **Block-shift stalls:** for longer surveys, use a smaller HR block or higher `num_workers`, or render
   in two passes. Stalls do not affect image correctness, only throughput.
3. **Stereo at altitude:** confirmed. Render 30–60 m AGL sequences with a 0.5 m baseline
   (`BASELINE=0.5`) and set `Stereo.ThDepth` to about 100 m / b. 1.0 m gives no further gain.
4. **Contrast:** try a lower sun (15–30°) to bring out the DEM relief.
5. **Shutdown:** the process now sets the monitor event and joins with a timeout. Isaac's fast shutdown
   still prints harmless multiprocessing errors.
