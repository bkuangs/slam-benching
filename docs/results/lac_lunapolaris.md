# LAC SLAM (SuperPoint/LightGlue + GTSAM) on LunaPolaris seq01 / 02 / 11

This page reports the Stanford NAV Lab Lunar Autonomy Challenge SLAM stack (`lac` adapter,
`stereo` mode, commit `26c3a2b`) on LunaPolaris seq01, 02 and 11. The stack is SuperPoint +
LightGlue stereo VO with PnP RANSAC, plus a GTSAM pose graph with keyframe loop closures. It is
compared with ORB-SLAM3 and RTAB-Map (`*_stereo_lean`, median of 3) and with FAST-LIO2 (single run).
The comparison numbers are from `docs/results/fastlio_lunapolaris.md`.

There are two LAC variants, and each is a **single run** per sequence:

- **LAC-faithful** (`configs/lunapolaris_seq{01,02,11}_lac_stereo.yaml`): LAC's frontend and backend
  as published, with the deviations listed under [Caveats](#caveats).
- **LAC-kfvo** (`*_lac_stereo_kfvo.yaml`): the same, except `vo_reference: keyframe`. It was
  designed and checked on seq01 only, so seq02 and seq11 are held out (see
  [Drift diagnosis](#drift-diagnosis)).

LAC ran on a Tesla V100 node with 8 CPUs and `cpu_threads: 1`. The other estimators are CPU-only.

## Trajectory accuracy

Final trajectories are SE3-aligned to motion capture, in the left camera frame. "Online" is the pose
LAC reported for each frame at the time, before later loop closures.

| Seq | Method | ATE RMSE (m) | ATE rot (°) | RPE 1 m (m / °) | RPE 1 s (m) | Online ATE (m) | Loop closures | Coverage | Loss events |
|---|---|---|---|---|---|---|---|---|---|
| 01 | LAC-faithful | 1.034 | 21.3 | 0.237 / 4.61 | 0.0192 | 1.033 | 1229 | 1.0 | 0 |
| 01 | LAC-kfvo | 0.162 | 5.1 | 0.110 / 3.11 | 0.0146 | 0.176 | 1250 | 1.0 | 0 |
| 01 | ORB-SLAM3 | **0.031** | | 0.061 / 2.94 | | | | 1.0 | 0 |
| 01 | RTAB-Map | 0.067 | | 0.060 / 3.04 | | | | 1.0 | 0 |
| 01 | FAST-LIO2 | 0.066 | | 0.073 / 2.91 | | | | 0.992 | 0 |
| 02 | LAC-faithful | 0.586 | 12.3 | 0.156 / 3.73 | 0.0175 | 0.591 | 1583 | 1.0 | 0 |
| 02 | LAC-kfvo | 0.193 | 5.4 | 0.099 / 2.95 | 0.0142 | 0.202 | 1549 | 1.0 | 0 |
| 02 | ORB-SLAM3 | 0.059 | | 0.063 / 2.92 | | | | 1.0 | 0 |
| 02 | RTAB-Map | 0.057 | | 0.059 / 2.95 | | | | 1.0 | 0 |
| 02 | FAST-LIO2 | **0.051** | | 0.064 / 2.91 | | | | 0.995 | 0 |
| 11 | LAC-faithful | 0.200 | 4.9 | 0.087 / 2.99 | 0.0093 | 0.200 | 2328 | 1.0 | 0 |
| 11 | LAC-kfvo | 0.168 | 4.1 | 0.085 / 2.85 | 0.0091 | 0.164 | 2258 | 1.0 | 0 |
| 11 | ORB-SLAM3 | 0.081 | | 0.063 / 2.80 | | | | 1.0 | 0 |
| 11 | RTAB-Map | 0.082 | | 0.060 / 2.85 | | | | 1.0 | 0 |
| 11 | FAST-LIO2 | **0.061** | | 0.071 / 2.97 | | | | 0.999 | 0 |

- **LAC never loses tracking.** It has no relocalisation state: when PnP fails it falls back to
  identity odometry. In practice PnP succeeded on every frame.
- **The faithful variant drifts.** The trajectory shows a steady heading drift plus about 0.5 m of
  vertical drift. That is why its ATE is 0.2–1.0 m.
- **Loop closures do not repair that drift.** About 1200–2300 closures are accepted, yet online ATE
  is roughly equal to final ATE. LAC finds loop candidates near the *estimated* pose (within 1 m and
  5°), so once the estimate has drifted it only closes loops that agree with the drift.
- **Keyframe-referenced VO cuts ATE** by 6× on seq01 and 3× on held-out seq02. On seq11 the gain
  is small, because the faithful run already had the least drift there. LAC-kfvo is still 2–5× worse
  than ORB-SLAM3, RTAB-Map and FAST-LIO2.
- **LAC's short-term accuracy is similar to the others.** RPE over 1 s is 9–19 mm, about the same
  as the others' local accuracy. The gap is long-term drift, not local noise.

## Drift diagnosis

### Scale check

The Sim3 scale is not a reliable measure of scale error when the heading drifts. For example,
faithful seq01 gives s = 0.86. Instead, we used a rotation-invariant check:

- Take 5 s and 10 s windows in which ground truth moves more than 0.15 m.
- In each window, divide the estimated displacement length by the ground-truth displacement length.
- Report the median ratio over all windows.

| Run | Median displacement ratio (10 s) |
|---|---|
| ORB-SLAM3 / RTAB-Map, seq01 | 0.985 / 0.989 |
| LAC-faithful, seq01, first 1500 frames | 0.89 |
| LAC-faithful, seq01, loop closure off | ~0.87 |
| LAC-kfvo, seq01, first 1500 frames | 0.97 |
| LAC-kfvo, seq01 / 02 / 11, full | 0.96 / 0.93 / 0.95 |

**LAC's VO underestimates motion.** On the first 1500 frames of seq01, ATE drops from 0.160 m
(faithful) to 0.040 m (kfvo).

### Cause

The rover moves about 0.5 cm per frame, which is less than a pixel of image motion for most
features at 14.6 Hz. LAC runs PnP between consecutive frames. LightGlue-matched SuperPoint keypoints
come from a detector grid and are not refined to sub-pixel positions. As a result, many true
sub-pixel motions are measured as zero, and the chained odometry is biased toward too little motion.
LAC was tuned on a simulator running at 20 Hz with a faster rover and a much wider stereo rig
(fx·B ≈ 148 px·m, vs 31.5 px·m for the ZED2), where per-frame motion is several pixels.

The keyframe variant keeps one reference frame until the median keypoint motion reaches 10 px, or
until fewer than 50% of its points still match. It runs PnP against that reference. Map points are
still added from every frame, as in LAC: each frame's stereo points that matched no tracked point.

### What was ruled out (seq01, `playback_rate` 0)

- **Calibration.** The patched intrinsics and baseline were verified inside the container. The
  images are rectified and the left and right cx are equal.
- **Stereo outliers and depth range.** Each of these changed ATE by at most 8%:
  - a minimum-disparity filter;
  - a 1.5 px row-difference filter;
  - a maximum depth of 10 m or 5 m.
- **Loop closure.** Turning it off makes ATE worse (1.30 m vs 1.03 m), so loop closure is not the
  source of the drift.

The tuning runs (`lunapolaris_seq01_lac_tune_t*`) used an image with a stereo-filter bug: the VO
tracker's 15 m depth limit was replaced by 56.5 m. The bug is now fixed (null `max_depth_m` uses
the caller's value). The faithful runs are not affected: their image predates the filter,
so they ran LAC's unmodified `process_stereo`.

Even with keyframe VO, the motion ratio stays at 0.93–0.96. Some bias remains, either from the
coarse keypoints or from PnP's 1 px RANSAC threshold. Sub-pixel keypoint refinement is the obvious
next step, but it has not been tried.

### Follow-up: frame stride, simulation and stereo depth

**Frame stride** (`frame_stride` setting: feed every Nth frame, so per-frame motion is N times
larger). seq01, full sequence, single runs:

| Run | ATE | Median displacement ratio (5 s) |
|---|---|---|
| ORB-SLAM3 / RTAB-Map (`_lean`) | 0.031 / 0.064 m | 0.985 / 0.988 |
| LAC-faithful, stride 1 / 2 / 4 | 1.034 / 0.451 / 0.205 m | 0.925 / 0.938 / 0.956 |
| LAC-kfvo, stride 1 / 2 / 4 | 0.162 / 0.126 / 0.134 m | 0.956 / 0.953 / 0.955 |

A larger stride removes the faithful variant's extra shrinkage, as the cause above predicts. Both
variants then level off at about 0.955, so per-frame motion does not explain the remaining bias.

**Simulation** (`configs/omnilrs_seq01_60s_lac_stereo{,_kfvo}.yaml`; see `docs/datasets/omnilrs.md`).
These runs replay the first 60 s of seq01 in the OmniLRS Lunalab, with exact calibration and ground truth.
SE(3)-aligned over the same 60 s, using the end-displacement ratio:

| Run | Real ATE / ratio | Sim ATE / ratio |
|---|---|---|
| ORB-SLAM3 | 0.010 m / 0.987 | 0.014 m / 0.989 |
| LAC-kfvo | 0.025 m / 0.953 | **0.012 m / 0.987** |
| LAC-faithful | 0.057 m / 0.938 | 0.021 m / 0.970 |

**In simulation, LAC-kfvo matches ORB-SLAM3.** Its extra motion shrinkage is specific to the real
images. The same camera, motion, keypoint grid and LAC code are used in both cases.

**Stereo depth.** `scripts/lac_stereo_depth_probe.py` runs LAC's own stereo matching (SuperPoint +
LightGlue) on every Nth frame. `scripts/stereo_depth_bias.py` compares the resulting keypoint depths
with a depth reference.

- **Sim vs exact depth:** median ratio 0.999; disparity error +0.04 px. No bias.
- **Real vs the LiDAR reference map:** median ratio 1.08, i.e. LAC's depths are *longer*, not
  shorter. The result is noisy:
  - only 20% of matches have a LiDAR depth within 1.5 px;
  - the spread is wide (IQR about ±15%);
  - the LiDAR-to-camera extrinsic is the nominal URDF.

  Too-long depths would exaggerate motion, so they do not explain the shrinkage either.

SuperPoint keypoints sit on a 0.625 px grid: they are integer pixels at the 1024 px resize, scaled
back to 640 px.

Open causes are real-image effects on matching: sensor noise, the repetitive gravel texture, and
exposure. A next step is adding noise and gravel-like texture to the simulator renders.

## Resource usage

| Seq | Method | Frame time mean / p95 (ms) | Frontend median (ms) | Backend median / max | Image load median (ms) | CPU per frame (ms) | CPU per s of data | Mid-run cores | Peak RSS (MB) | Peak GPU mem (GiB) | Wall / data duration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | LAC-faithful | 90 / 103 | 61 | 0.15 ms / 2.5 s | — | 105 | 1.52 | 0.58 | 1694 | 0.74 | 2.55 |
| 01 | LAC-kfvo | 88 / 103 | 62 | 0.14 ms / 1.9 s | 80 | 102 | 1.48 | 0.58 | 1580 | 0.74 | 2.52 |
| 02 | LAC-faithful | 82 / 96 | 57 | 0.15 ms / 1.9 s | — | 97 | 1.43 | 0.58 | 2259 | 0.84 | 2.46 |
| 02 | LAC-kfvo | 84 / 99 | 60 | 0.15 ms / 1.8 s | 80 | 99 | 1.45 | 0.57 | 1758 | 0.84 | 2.53 |
| 11 | LAC-faithful | 81 / 90 | 52 | 0.15 ms / 1.4 s | — | 96 | 1.42 | 0.58 | 1563 | 1.30 | 2.42 |
| 11 | LAC-kfvo | 83 / 93 | 55 | 0.15 ms / 1.7 s | 79 | 99 | 1.46 | 0.52 | 1742 | 1.30 | 2.64 |

For comparison (from `fastlio_lunapolaris.md`), CPU per second of data is:

- ORB-SLAM3: 1.01–1.12, with a 24 ms mean frame time and 0.7–1.1 GB RSS;
- RTAB-Map: 1.19–1.32, with a 60 ms mean frame time and about 0.5 GB RSS;
- FAST-LIO2: 0.25–0.27.

How to read this table:

- **CPU is about 30% higher than the stereo baselines, and it also uses a GPU.** The GPU runs
  SuperPoint + LightGlue: two stereo matches and one temporal match per frame. Peak allocated GPU
  memory is 0.7–1.3 GiB.
- **The frontend fits the frame rate; the full pipeline does not.** The frontend median of 52–62 ms
  is under the 68 ms frame period. The mean frame time of 81–90 ms is over it, because of backend
  spikes.
- **Backend spikes are full re-optimisations.** The backend median is 0.15 ms. But every accepted
  loop closure triggers a full Levenberg–Marquardt solve of the whole graph, which takes up to
  1.4–2.5 s near the end of a run. Total backend time is about 350 s per run.
- **Wall time is about 2.5× the data duration.** Frames are queued, not dropped. Reading and
  decoding both PNGs takes about 80 ms per frame (median), and that is not part of the frame time:
  - about 29 ms per image is a cold read from network storage;
  - about 12 ms per image is RGB PNG decoding.

  This input path is shared with the other stereo estimators; it is not caused by LAC.
- **`cpu_threads` = 1** sets `OMP_NUM_THREADS` / `MKL_NUM_THREADS`. The library default spun about
  7 cores, with no change in frontend latency: in a test on seq11, wall time was 82.6 s → 54 s,
  CPU user time 380 s → 52 s, and the frontend median stayed at 49 ms.

## Map quality (LiDAR reference)

The method and columns are the same as in `docs/results/fastlio_lunapolaris.md`:

- `scripts/eval_map_quality.py --lac …` with a 2 cm voxel grid;
- the LiDAR reference is built from motion-capture poses;
- each map is placed using its trajectory's SE3 alignment.

`lac_points` is LAC's own map (`Backend.project_point_map()`): sparse stereo SuperPoint points,
each placed at the final pose of the frame where it was first seen. This is not a dense
reconstruction.

| Seq | Map | Points | Acc. median | P / R / F @10 cm | F @5 cm | Recall-all @10 cm | ICP Δz | After ICP: F @10 / F @5 |
|---|---|---|---|---|---|---|---|---|
| 01 | lac_points (LAC-faithful) | 230 k | 8.3 cm | 0.54 / 0.64 / 0.58 | 0.37 | 0.53 | −14 cm | 0.72 / 0.47 |
| 01 | lac_points (LAC-kfvo) | 278 k | 8.8 cm | 0.54 / 0.59 / 0.56 | 0.30 | 0.46 | −25 cm | 0.81 / 0.59 |
| 01 | lio_map (FAST-LIO2) | 1124 k | 1.8 cm | 0.91 / 0.53 / 0.67 | 0.47 | 0.44 | −21 cm | 0.92 / 0.74 |
| 01 | orb_tsdf (ORB-SLAM3) | 410 k | 5.4 cm | 0.66 / 0.49 / 0.56 | 0.31 | 0.43 | −12 cm | 0.74 / 0.52 |
| 01 | rtab_cloud (RTAB-Map) | 629 k | 8.3 cm | 0.58 / 0.40 / 0.47 | 0.25 | 0.28 | −29 cm | 0.75 / 0.57 |
| 02 | lac_points (LAC-faithful) | 286 k | 9.1 cm | 0.53 / 0.57 / 0.55 | 0.32 | 0.45 | −22 cm | 0.78 / 0.56 |
| 02 | lac_points (LAC-kfvo) | 308 k | 9.8 cm | 0.50 / 0.57 / 0.54 | 0.29 | 0.45 | −26 cm | 0.81 / 0.59 |
| 02 | lio_map (FAST-LIO2) | 1145 k | 2.0 cm | 0.89 / 0.54 / 0.68 | 0.48 | 0.45 | −21 cm | 0.93 / 0.77 |
| 02 | orb_tsdf (ORB-SLAM3) | 339 k | 7.0 cm | 0.61 / 0.39 / 0.48 | 0.24 | 0.31 | −17 cm | 0.68 / 0.45 |
| 02 | rtab_cloud (RTAB-Map) | 745 k | 13.7 cm | 0.35 / 0.29 / 0.32 | 0.12 | 0.20 | −31 cm | 0.71 / 0.55 |
| 11 | lac_points (LAC-faithful) | 356 k | 7.4 cm | 0.62 / 0.57 / 0.60 | 0.30 | 0.45 | −24 cm | 0.81 / 0.64 |
| 11 | lac_points (LAC-kfvo) | 378 k | 7.0 cm | 0.65 / 0.60 / 0.62 | 0.32 | 0.47 | −23 cm | 0.81 / 0.64 |
| 11 | lio_map (FAST-LIO2) | 1197 k | 2.1 cm | 0.90 / 0.54 / 0.68 | 0.47 | 0.45 | −21 cm | 0.93 / 0.76 |
| 11 | orb_tsdf (ORB-SLAM3) | 408 k | 6.6 cm | 0.60 / 0.54 / 0.57 | 0.34 | 0.46 | −12 cm | 0.66 / 0.45 |
| 11 | rtab_cloud (RTAB-Map) | 525 k | 12.7 cm | 0.40 / 0.26 / 0.31 | 0.13 | 0.16 | −34 cm | 0.65 / 0.49 |

The baseline rows reproduce `fastlio_lunapolaris.md` exactly. ICP Δz is the vertical shift of the
extra ICP correction, evaluated at the map's centroid. Every map needs a similar −12 to −34 cm,
consistent with the reference-side bias discussed in `fastlio_lunapolaris.md`.

Full results are in `~/scratch/slam-bench-work/map_quality_lac{,_faithful}/sequence_*/map_quality.json`.

- **LAC's points are about as accurate as RTAB-Map's** (7–10 cm median) and less accurate than
  the ORB-SLAM3 TSDF (5–7 cm). Pre-ICP F@10 cm (0.54–0.62) is close to ORB-SLAM3 TSDF and above
  RTAB-Map. After ICP, LAC-kfvo's local geometry (F@10 0.81) matches or beats both stereo baselines,
  but stays below FAST-LIO2 (0.92–0.93).
- **These maps are small.** At 0.2–0.4 M points, LAC's sparse map is about as dense as the ORB-SLAM3
  TSDF. Each frame adds only its stereo points that matched no tracked point,
  about 17–28 per frame.
- **The F-score barely reflects xy drift.** The arena is mostly flat floor, and a horizontally
  shifted floor still lies on the reference floor. The faithful runs therefore score close to the
  baselines despite 0.6–1.0 m of ATE. On seq01, LAC-kfvo's after-ICP F@10 rises from 0.72 to
  0.81, while its pre-ICP F-score barely changes. For landing-site maps,
  a 2.5D height-map metric would be more telling.

## Caveats

- **Single run** per sequence for each LAC variant.
- **No semantic segmentation.** LAC's U-Net weights are not public, so every keypoint gets the
  ground label, and LAC's rock detection and rock mapping are not run.
- **No IMU.** When PnP fails, odometry is the identity with LAC's IMU noise model. This is what LAC's
  frontend does at this commit.
- **Keyframes are time-based**, every 1 s of data. LAC uses every 20 simulator steps, which is also
  1 s.
- **Calibration.** The LunaPolaris calibration replaces LAC's simulator constants (`lac.params`).
- **Different hardware.** LAC runs on a GPU (V100); the other estimators run on CPU only.
- **Licence.** The SuperPoint weights carry Magic Leap's non-commercial licence.
- **Pacing.** Faithful runs were made with an earlier image, from before the stereo filter and the
  `image_load_time_s` / `ref_switch` logging existed. That image runs LAC's code unmodified. All
  full-length runs use real-time pacing (`playback_rate` 1); the diagnosis runs used 0.
- **Single-sequence tuning.** LAC-kfvo was designed on seq01 only. Its reference-switch thresholds
  (10 px, 50%) were not swept.
