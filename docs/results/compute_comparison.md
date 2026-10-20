# Compute cost: ORB-SLAM3 vs RTAB-Map stereo (LunaPolaris)

Configs: `configs/lunapolaris_seq{01,02,11}_{orbslam3,rtabmap}_stereo_lean.yaml`, 3 repeats each,
Slurm `--constraint=gold6226` (Xeon Gold 6226, 8 cores per job), `playback_rate: 1.0`.

## What was made comparable
The first comparison (default configs) was not like-for-like:
* RTAB-Map's per-frame time included its 1 Hz map update, because our driver ran odometry and map
  update one after the other; ORB-SLAM3 reports only its tracking thread (local mapping and loop
  closing run on their own threads). The lean RTAB-Map configs use `threaded_map_update: true`
  (map updates on a worker thread, as rtabmap_ros' separate nodes do), so both report the
  per-frame front end only.
* RTAB-Map built per-node 3D occupancy grids during the run and exported a dense cloud and OctoMap
  after it, inside the measured process tree. Lean configs: `occupancy_grid: false`, no export.
* Both logged per-frame keypoints (`features.bin`); lean configs: `log_features: false`.

Both use binary descriptors (ORB-SLAM3: ORB; RTAB-Map default: GFTT corners + BRIEF), so the
difference is not the descriptor type.

## Results (mean over 3 repeats; [min, max])

| Seq | Estimator | ATE RMSE (m) | Per-frame time mean / p95 (ms) | Frames over 68 ms | CPU per frame (ms) | Peak RSS (MB) |
|---|---|---|---|---|---|---|
| 01 | ORB-SLAM3 | 0.032 | 25 / 33 | 0% | 78 [74, 84] | 1136 |
| 01 | RTAB-Map | 0.067 | 59 / 68 | 6% | 81 [81, 82] | 514 |
| 02 | ORB-SLAM3 | 0.059 | 25 / 32 | 0% | 71 [67, 75] | 974 |
| 02 | RTAB-Map | 0.056 | 62 / 75 | 20% [9, 36] | 85 [81, 90] | 521 |
| 11 | ORB-SLAM3 | 0.081 | 25 / 31 | 0% | 67 [66, 67] | 679 |
| 11 | RTAB-Map | 0.084 | 64 / 74 | 30% [27, 32] | 88 [87, 89] | 495 |

* Per-frame time: ORB-SLAM3 tracking thread; RTAB-Map odometry (map updates on their own thread).
* CPU per frame: user + system CPU of the whole process tree (all threads, incl. mapping / loop
  closing / map updates and image decoding) divided by the number of frames, excluding the first
  25 s (vocabulary / parameter loading). Every RTAB-Map map update was processed (none dropped).
* RTAB-Map's CPU split (`driver_timing.json`): odometry ~58 ms per frame, map updates (graph,
  loop closure) only ~5% of its CPU.

## Conclusions
* ORB-SLAM3's front end is ~2.5x faster per frame (25 vs 59–64 ms) and its p95 is ~55% lower;
  it never exceeded the 15 Hz frame period, RTAB-Map's odometry did on 6–30% of frames.
* Total CPU per frame is close: ORB-SLAM3 used 4–24% less (seq01 / seq02 / seq11: 4% / 16% / 24%).
  ORB-SLAM3 spreads its work over tracking + mapping threads; RTAB-Map's is mostly one
  single-threaded odometry.
* RTAB-Map used about half the memory (~510 vs 680–1140 MB; ORB-SLAM3 holds its vocabulary).
* Accuracy is comparable on seq02 / seq11; ORB-SLAM3 is twice as accurate on seq01.
* Compared with the default configs, threading the map update cut RTAB-Map's p95 from ~150 ms
  to 68–75 ms; dropping grids/export cut its total CPU by ~5–10%.

## Cost of RTAB-Map's dense map
`configs/lunapolaris_seq{01,02,11}_rtabmap_stereo_dense.yaml` are the lean RTAB-Map configs with
per-node 3D occupancy grids (0.05 m cells, 5 m range, from stereo depth) and the post-run dense cloud
+ OctoMap export switched back on. 3 repeats each; means over all 9 runs per variant:

| | Lean (no dense map) | Dense | Change |
|---|---|---|---|
| ATE RMSE (m), seq01 / 02 / 11 | 0.067 / 0.056 / 0.084 | 0.058 / 0.056 / 0.089 | within repeat noise |
| Odometry per frame, mean / p95 (ms) | 61 / 72 | 60 / 70 | none (grids are not on the front end) |
| Map update CPU (ms per 1 Hz update) | 51 | 94 | +85% |
| CPU per frame, whole driver (ms) | 85.5 | 90.0 | +5% |
| Post-run export (cloud + OctoMap) | — | 25 s CPU, 15 s wall | one-off |
| Total CPU incl. export (s) | 1139 | 1223 | +7% |
| Peak RSS (MB) | 510 | 685 | +34% |
| Map database (MB) | 282 | 510–675 | ~2.1x |

* In threaded mode the grids only load the map-update thread (~10% of one core at 1 Hz), so per-frame
  latency is unchanged; no update was dropped. Odometry dominates either way.
* The grids are per-node local maps; the global OctoMap is only assembled by the post-run export.
  A live global map (as rtabmap_ros' map assembler publishes) would add work not measured here.
* Against ORB-SLAM3 (no dense map): RTAB-Map with the dense map uses ~26% more CPU per frame
  (90 vs 72 ms) and ~25% less memory (685 vs 930 MB), at 2.5x the per-frame front-end latency.

## Prototype: dense map on top of ORB-SLAM3
`scripts/dense_from_keyframes.py <orbslam3 run>` (`pip install -e .[dense]`) post-processes a run:
SGBM stereo depth for each keyframe of the final map, fused into a 5 cm TSDF at the keyframe's final
(loop-closed) pose, then marching cubes. Lean ORB-SLAM3 runs, one per sequence, 8-core ice-cpu node:

| Seq | Keyframes | Per keyframe: SGBM / TSDF fusion (ms) | Total CPU | Mesh | Sparse points within 5 cm of mesh (median dist) |
|---|---|---|---|---|---|
| 01 | 819 | 56 / 19 | 108 s | 452k triangles, 21 MB | 86% (2.2 cm) |
| 02 | 544 | 56 / 16 | 67 s | 335k triangles, 15 MB | 89% (1.4 cm) |
| 11 | 174 | 56 / 19 | 23 s | 380k triangles, 18 MB | 85% (1.4 cm) |

![seq01 maps](figures/seq01_dense_maps.png)
Top-down maps of seq01: ORB-SLAM3 sparse points, the keyframe TSDF, RTAB-Map's dense cloud and its
OctoMap (occupied cells by height over free cells), with RTAB-Map aligned to ORB-SLAM3's frame by its
trajectory (`scripts/plot_dense_maps.py`). Same for [seq02](figures/seq02_dense_maps.png) and
[seq11](figures/seq11_dense_maps.png). The height gradient across the floor is the tilt of ORB-SLAM3's
world frame (its first camera pose), not terrain.

* Depth is computed only per keyframe (0.2–0.9 per second here, vs RTAB-Map's 1 Hz grids), so done
  online on a background thread this would cost ~75 ms per keyframe, 1.5–7% of one core on average.
  The image load time (~90 ms per pair from scratch) is excluded; a live system already has the images.
* The sparse map points (triangulated and bundle-adjusted independently of SGBM) agree with the fused
  surface to ~1.5–2 cm, so the keyframe poses and stereo depth are consistent.
* The TSDF surface covers more 10 cm cells than RTAB-Map's exported cloud (48–68k vs 28–39k; ORB
  sparse map: 3–6k), partly because a TSDF surface is a few voxels thick and noisy depth at range
  (e.g. the dark backdrop) adds fringe. Accuracy against ground-truth geometry is measured below.
* Limits: offline (after the run), no online map updates on loop closure, SGBM single-threaded, no
  free-space / occupancy output (the TSDF has it implicitly; an OctoMap export would be next).

## Map quality against a LiDAR reference
`scripts/eval_map_quality.py` builds a reference map from the robot's Livox LiDAR: all ~7,800 scans,
de-skewed per point, placed with motion capture and the `/tf_static` extrinsic, in 2 cm voxels seen
≥4 times. Two half-maps built from alternating scans agree to a median of 0.4 cm (97% within 2 cm),
so the reference is far tighter than the errors measured. Each map is placed in the mocap world by
fitting its own keyframe/node trajectory to ground truth (SE3), then scored Tanks-and-Temples style:
* **precision@τ**: fraction of map points (within the LiDAR's coverage) within τ of the reference;
  **accuracy** = median of those distances.
* **recall@τ**: fraction of reference voxels the camera actually saw (frustum ≤ 5 m, occlusion-tested
  from ground-truth poses; ~57% of the reference) that have a map point within τ.
* **F@τ**: harmonic mean of the two. τ = 5 and 10 cm.

Variants: `orb_tsdf` (SGBM + TSDF at ORB-SLAM3 keyframe poses, as above), `dense_strict` (max depth
3 m, SGBM uniqueness 15), `dense_gt` (same keyframe images fused at ground-truth poses: depth error
only), `dense_rtab` (same keyframe images fused at RTAB-Map's poses from its `trajectory.tum`: compares
the two systems' poses with the depth pipeline held fixed), and RTAB-Map's own exported cloud / OctoMap
(occupied cells). Maps are aligned by trajectory (SE3 on positions) unless noted.

Run 1 of each configuration:

| seq | map | points | accuracy (median) | precision@10cm | recall@10cm | F@10cm | F@5cm |
|---|---|---|---|---|---|---|---|
| 01 | ORB sparse | 19k | 5.2 cm | 0.94 | 0.21 | 0.35 | 0.07 |
| 01 | ORB + TSDF | 411k | 5.4 cm | 0.66 | 0.49 | **0.56** | 0.31 |
| 01 | ORB + TSDF strict | 257k | 5.0 cm | 0.78 | 0.40 | 0.53 | 0.24 |
| 01 | TSDF at GT poses | 398k | 4.5 cm | 0.63 | 0.45 | 0.53 | 0.33 |
| 01 | RTAB-Map cloud | 629k | 8.3 cm | 0.58 | 0.40 | 0.47 | 0.25 |
| 01 | RTAB-Map OctoMap | 81k | 8.1 cm | 0.59 | 0.34 | 0.43 | 0.20 |
| 02 | ORB sparse | 12k | 5.9 cm | 0.88 | 0.13 | 0.23 | 0.04 |
| 02 | ORB + TSDF | 340k | 7.0 cm | 0.61 | 0.39 | **0.48** | 0.24 |
| 02 | ORB + TSDF strict | 241k | 7.1 cm | 0.66 | 0.29 | 0.40 | 0.17 |
| 02 | TSDF at GT poses | 354k | 2.4 cm | 0.69 | 0.56 | 0.62 | 0.42 |
| 02 | RTAB-Map cloud | 745k | 13.7 cm | 0.35 | 0.29 | 0.32 | 0.12 |
| 02 | RTAB-Map OctoMap | 96k | 13.2 cm | 0.36 | 0.23 | 0.28 | 0.08 |
| 11 | ORB sparse | 8k | 5.2 cm | 0.87 | 0.08 | 0.15 | 0.03 |
| 11 | ORB + TSDF | 409k | 6.6 cm | 0.60 | 0.54 | **0.57** | 0.34 |
| 11 | ORB + TSDF strict | 212k | 6.6 cm | 0.64 | 0.35 | 0.45 | 0.22 |
| 11 | TSDF at GT poses | 393k | 3.1 cm | 0.66 | 0.60 | 0.63 | 0.42 |
| 11 | RTAB-Map cloud | 525k | 12.7 cm | 0.40 | 0.26 | 0.31 | 0.13 |
| 11 | RTAB-Map OctoMap | 71k | 12.8 cm | 0.40 | 0.19 | 0.26 | 0.09 |

All 3 repeats (min–max over runs; RTAB-Map run *i* paired with ORB-SLAM3 run *i*):

| seq | map | F@10cm | median error |
|---|---|---|---|
| 01 | ORB + TSDF | 0.56–0.57 | 5.4–5.6 cm |
| 01 | TSDF at RTAB-Map poses | 0.54–0.55 | 5.9–6.1 cm |
| 01 | RTAB-Map cloud | 0.47–0.49 | 8.2–8.3 cm |
| 02 | ORB + TSDF | 0.48–0.49 | 6.8–7.0 cm |
| 02 | TSDF at RTAB-Map poses | 0.47–0.48 | 7.3–7.5 cm |
| 02 | RTAB-Map cloud | 0.32 | 13.5–14.2 cm |
| 11 | ORB + TSDF | 0.56–0.57 | 6.4–6.6 cm |
| 11 | TSDF at RTAB-Map poses | 0.56–0.57 | 6.3–6.8 cm |
| 11 | RTAB-Map cloud | 0.31–0.32 | 12.6–13.1 cm |

![seq01 map quality](figures/seq01_map_quality.png)

* **The map-building step, not the SLAM poses, decides map quality here.** Fusing the same keyframe
  images at RTAB-Map's poses gives almost the same map as at ORB-SLAM3's (F@10cm within 0.02, median
  error within 0.5 cm, all sequences and repeats). RTAB-Map's own cloud is clearly worse (F@10cm
  0.31–0.49 vs 0.47–0.57) because it stacks per-frame stereo points: compared directly with the TSDF
  built at its own poses (same frame, no alignment), it lies 6 cm away at the median and 17–19 cm at
  p90, i.e. it is a thick, noisy shell around the surface, and its floor sits 8–12 cm higher than the
  TSDF's. TSDF fusion averages many views and removes most of that. So the earlier result "ORB-SLAM3 +
  TSDF beats RTAB-Map" is really "TSDF fusion beats RTAB-Map's cloud export"; either SLAM system's
  poses would do.
* **Run-to-run spread is tiny** (≤0.02 F, ≤0.7 cm median error), far smaller than the gaps above.
* **The alignment method doesn't change the picture.** Aligning by camera orientations (rotation from
  the chordal mean of per-pose orientation differences, then translation) instead of positions only
  changes the fitted tilt by 0.2–1.6° and F@10cm by ≤0.04, and every ranking holds. Both fits leave a
  ~4° RMS orientation residual for *both* systems, which a constant error in the nominal
  marker→camera extrinsic would produce; that affects absolute numbers, not the comparison.
* **The sparse ORB-SLAM3 map is accurate but nearly empty**: 86–94% of its points are within 10 cm, but
  it covers only 8–21% of the visible surface.
* **Pose error is about half of the TSDF's error** on seq02/11: fusing the same depth at ground-truth
  poses halves the median error (7 → 2.4 cm, 6.6 → 3.1 cm) and lifts F@10cm to ~0.62. The rest is
  stereo depth noise (and the floor/backdrop fringe visible in the figure). On seq01 the gain is small.
  The poses of the two SLAM systems are about equally good for this.
* **Stricter SGBM didn't help** (one setting tried): slightly higher precision, much lower recall, lower F.
* **Height offsets.** ICP against the (floor-dominated) reference moves RTAB-Map's cloud down 28–34 cm,
  TSDFs at either system's poses down 10–17 cm, and the GT-pose TSDF up 5–10 cm. The cloud's extra
  ~15 cm comes from its raised, noisy floor (above), not from its poses. The remaining offset shared by
  both systems' TSDFs is unexplained (candidates: the nominal camera/LiDAR extrinsics, small tilt
  errors that a flat trajectory can't resolve). After ICP every dense map scores F@10cm 0.63–0.76 with
  ~1–2 cm median error, but that ICP also slides maps 10–25 cm sideways (a flat room barely constrains
  in-plane motion), so post-ICP numbers are an optimistic bound and stay out of the tables.
* Caveats: nominal LiDAR and camera extrinsics; the LiDAR sees ~6 m, so far walls and backdrop are
  outside the scored region; one room; RTAB-Map's cloud export used default settings (no voxel
  filtering, noise removal or depth limits beyond the run config), which could narrow the gap.

## Caveat: input reading did not keep up with real time
Wall time per frame was ~96 ms (ORB-SLAM3) and ~133 ms (RTAB-Map) against the 68 ms frame period:
reading a stereo PNG pair from the shared scratch filesystem takes ~90 ms, so playback ran ~1.4x /
~1.9x slower than real time (repeats started together ran in lockstep). Per-frame compute time and
CPU per frame are unaffected, but each estimator's background work (ORB-SLAM3 local mapping,
RTAB-Map map updates) had more wall time per frame than on a live robot. A true real-time test needs
the input staged on node-local disk or in memory.
