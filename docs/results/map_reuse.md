# Map reuse: ORB-SLAM3 and SP_SLAM3, map in session A, localize in session B

Goal: a drone maps terrain once, and that map is reused later under different lighting. The
feature-level test (`docs/results/lunaryard_aerial_lighting.md`, `sim/omnilrs/cross_lighting_reloc.py`)
found that ORB fails to match across sun-angle changes, while SuperPoint+LightGlue matches more often.
This test runs the complete systems: session A builds and saves an atlas, and session B loads it
and localizes in it.

## Method
* **Session A** (mapping): stereo, normal SLAM, `save_map: true`. The atlas is written to
  `algorithm/atlas.osa` at shutdown.
* **Session B** (query): `load_map: runs/<session A config>`, which resolves to the latest successful
  mapping run. Two modes:
  * `relocalize`: localization only. Patch 0006 starts tracking LOST in the loaded map. Every
    untracked frame runs `Tracking::Relocalization`, which uses DBoW keyframe-database candidates,
    BoW matching, PnP RANSAC and pose optimization with at least 50 inliers. No keyframe or map
    is ever created, and local mapping is stopped. A frame that is not tracked in the loaded map
    has no pose.
  * `merge`: stock ORB-SLAM3 multi-session mapping. B starts a new map, and place recognition may
    merge it into the loaded map. Frames before the merge are in a separate map and are excluded
    (not localized). Loop closure is on.
* **Scoring** (`slambench/evaluation/map_reuse.py`, `docs/evaluation.md`): the **map frame** is
  fixed by session A. One SE(3) `T_gt_map` comes from session A's own final trajectory vs GT,
  over all of A. Session B's poses (`frame_log.csv`, through the driver's `T_final_map`) are
  scored with that transform and are **never re-aligned**. Metrics:
  * localized = fraction of the input duration with a pose in the loaded map;
  * ≤0.5 m = localized with error ≤ 0.5 m. Frames without a GT sample (mocap gaps) cannot count;
  * first loc = data time of the first localization (the merge, in merge mode);
  * error RMSE / median / p95 over localized frames;
  * rotation error median;
  * longest lost streak after the first localization;
  * per-frame tracking time (mean);
  * atlas load time.
* The adapter also checks two things. The loaded keyframes must equal the mapping run's final
  keyframes (`map_frame_check`, max 12 µm for ORB sim). The atlas file must be unchanged
  (sha256 before and after).

### Data
* **Sim:** OmniLRS `lunaryard_aerial_<lighting>`, 120 s / 1800 frames / 15 Hz, at 5 m AGL. The map
  is built from `nominal`; the queries are `nominal`, `low_sun`, `dim` and `glare` (see
  `docs/datasets/omnilrs.md`). **Caveat: every query flies the identical trajectory as the
  map**, so the viewpoint is as easy as possible and only lighting changes. The `nominal`
  query uses identical images (sanity check).
* **Real:** LunaPolaris `sequence_01` (map), queried with `sequence_01` (sanity), `sequence_02`
  and `sequence_11`. These are 900–912 s each, recorded on 3 different days (2025-06-16/17/18).
  * **World frame check** (all three use the same lab mocap, topic
    `/vrpn_mocap/ZED2_camera_marker/rectified/pose`):
    * start positions are within 15 cm (01: (1.92, 2.69, 0.46), 02: (1.77, 2.60, 0.46),
      11: (1.90, 2.70, 0.46));
    * the bounding boxes are equal to 1–2 cm (x 1.64–4.59, y 2.20–8.11, z 0.41–0.51);
    * each sequence's XY nearest-neighbour distance to the others is 8–12 mm median, and 100 %
      of points are within 0.3 m.

    So GT shares one frame, and cross-sequence fixed-frame scoring is valid.
  * **Caveat:** the three runs drive almost the same path. The difference between them is
    mostly scene and lighting, not viewpoint.
  * Image statistics (cam0, 20 samples each):

    | seq | mean grey | p99 | saturated (≥250) | ORB keypoints |
    |---|---|---|---|---|
    | 01 | 137 | 209 | 0.03 % | 882 |
    | 02 | 138 | 219 | 0.11 % | 918 |
    | 11 | 121 | 253 | 7 % | 1157 |

    Seq11 is darker with blown highlights, and it needs the −1.63 px right-row shift (see
    `docs/datasets/lunapolaris.md`).
* Estimator settings are the existing lean stereo settings: ORB 1200 features, SP 1200 features,
  loop closure on, real-time playback. The configs are `configs/*_stereo_map.yaml` (A) and
  `configs/{omnilrs_mapreuse_mapnominal,lunapolaris_mapreuse_map01}_*_{orbslam3,spslam3}_{relocalize,merge}.yaml`
  (B).

### Reproduce
```bash
export PATH=/storage/ice1/3/6/bkuang7/install-dir/bin:$PATH
# session A (ORB-SLAM3 runs on CPU; SP_SLAM3 needs a GPU: -p coc-gpu -q coc-ice --gres=gpu:v100:1)
A=$(sbatch --parsable -p ice-cpu -c 8 --mem=32G -t 3:00:00 scripts/slurm_slambench.sbatch run configs/omnilrs_lunaryard_aerial_nominal_orbslam3_stereo_map.yaml)
# session B, after A succeeded
sbatch -p ice-cpu -c 8 --mem=32G -t 3:00:00 --dependency=afterok:$A scripts/slurm_slambench.sbatch run configs/omnilrs_mapreuse_mapnominal_low_sun_orbslam3_relocalize.yaml
```
Each config's header has its exact command. Results are in `runs/<config>/<run>/eval/metrics.json`
(section `map_reuse`), `extra.map_reuse` / `extra.saved_map` in `algorithm_output.json`, and
`eval/map_reuse_frames.csv`.

Images: `orbslam3_4452a3c_driver3.sif` and `spslam3_9439971_r3.sif` (patch 0006). The SP_SLAM3
mapping runs used `spslam3_9439971_r2.sif`. It has the same atlas format and differs only in code
reachable when an atlas is loaded (see below). r2 has since been deleted, and the configs now
use r3.

## Results

### Session A maps
| map | estimator | ATE RMSE (SE3) | keyframes | map points | atlas | wall |
|---|---|---|---|---|---|---|
| sim nominal (120 s) | ORB-SLAM3 | 0.133 m | 816 | 54 016 | 186 MB | 172 s |
| sim nominal (120 s) | SP_SLAM3 | 0.136 m | 581 | 67 539 | 919 MB | 415 s |
| LunaPolaris seq01 (912 s) | ORB-SLAM3 | 0.032 m | 823 | 25 149 | 186 MB | 1418 s |
| LunaPolaris seq01 (912 s) | SP_SLAM3 | 0.053 m | 266 | 14 556 | 396 MB | 2757 s |

SP atlases are about 7× larger per keyframe (1.6 MB vs 0.23 MB): 256 float32 values (1 KB) per
descriptor versus 32 bytes for ORB. The atlas loads in ≈10 s for ORB and ≈3.7 s for SP.

### Simulation (map = nominal; same trajectory)
| estimator | mode | query | localized | ≤0.5 m | first loc | RMSE | median | p95 | rot med | longest lost | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ORB-SLAM3 | relocalize | nominal | 1.000 | 1.000 | 0.0 s | 0.133 m | 0.135 m | 0.197 m | 0.34° | 0.0 s | 29 |
| ORB-SLAM3 | relocalize | low_sun | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 30 |
| ORB-SLAM3 | relocalize | dim | 1.000 | 1.000 | 0.0 s | 0.133 m | 0.134 m | 0.195 m | 0.35° | 0.0 s | 20 |
| ORB-SLAM3 | relocalize | glare | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 30 |
| ORB-SLAM3 | merge | nominal | 0.985 | 0.985 | 1.8 s | 0.133 m | 0.132 m | 0.199 m | 0.34° | 0.0 s | 30 |
| ORB-SLAM3 | merge | low_sun | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 26 |
| ORB-SLAM3 | merge | dim | 0.986 | 0.986 | 1.7 s | 0.133 m | 0.131 m | 0.202 m | 0.34° | 0.0 s | 22 |
| ORB-SLAM3 | merge | glare | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 27 |
| SP_SLAM3 | relocalize | nominal | 1.000 | 1.000 | 0.0 s | 0.136 m | 0.120 m | 0.234 m | 0.24° | 0.0 s | 160 |
| SP_SLAM3 | relocalize | low_sun | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 161 |
| SP_SLAM3 | relocalize | dim | 0.999 | 0.999 | 0.1 s | 0.136 m | 0.120 m | 0.233 m | 0.24° | 0.0 s | 160 |
| SP_SLAM3 | relocalize | glare | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 161 |
| SP_SLAM3 | merge | nominal | 0.978 | 0.978 | 2.7 s | 0.128 m | 0.118 m | 0.219 m | 0.24° | 0.0 s | 163 |
| SP_SLAM3 | merge | low_sun | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 123 |
| SP_SLAM3 | merge | dim | 0.981 | 0.981 | 2.3 s | 0.130 m | 0.118 m | 0.221 m | 0.24° | 0.0 s | 127 |
| SP_SLAM3 | merge | glare | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 124 |

`init_failure` = never localized in the loaded map: 0 of 1800 frames. In merge mode, all four
estimator × lighting failures still tracked all 1800 frames in their own new map, but it never
merged, so it stays in an unknown frame.

### Real (LunaPolaris, map = seq01)
| estimator | mode | query | localized | ≤0.5 m | first loc | RMSE | median | p95 | rot med | longest lost | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ORB-SLAM3 | relocalize | seq01 | 0.982 | 0.932 | 16.5 s | 0.032 m | 0.029 m | 0.054 m | 4.19° | 0.0 s | 29 |
| ORB-SLAM3 | relocalize | seq02 | 0.773 | 0.736 | 204.0 s | 0.036 m | 0.034 m | 0.056 m | 4.14° | 0.0 s | 28 |
| ORB-SLAM3 | relocalize | seq11 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 29 |
| ORB-SLAM3 | merge | seq01 | 0.974 | 0.924 | 23.7 s | 0.032 m | 0.028 m | 0.054 m | 4.20° | 0.0 s | 28 |
| ORB-SLAM3 | merge | seq02 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 22 |
| ORB-SLAM3 | merge | seq11 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 25 |
| SP_SLAM3 | relocalize | seq01 | 0.990 | 0.778 | 0.0 s | 0.358 m | 0.094 m | 0.838 m | 4.21° | 5.0 s | 139 |
| SP_SLAM3 | relocalize | seq02 | 0.949 | 0.588 | 0.2 s | 0.488 m | 0.320 m | 0.902 m | 4.20° | 37.8 s | 146 |
| SP_SLAM3 | relocalize | seq11 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 133 |
| SP_SLAM3 | merge | seq01 | 0.975 | 0.925 | 22.4 s | 0.051 m | 0.047 m | 0.078 m | 4.26° | 0.0 s | 115 |
| SP_SLAM3 | merge | seq02 | 0.959 | 0.919 | 37.2 s | 0.058 m | 0.053 m | 0.093 m | 4.26° | 0.0 s | 107 |
| SP_SLAM3 | merge | seq11 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 102 |

`init_failure` rows in merge mode (ORB seq02, both seq11) likewise tracked the whole query in
their own new map but never merged into the loaded map.

### Real (LunaPolaris, map = seq11)
Session A seq11 maps: ORB-SLAM3 ATE 0.089 m, SP_SLAM3 ATE 0.098 m (`right_row_shift_px: -1.63`).

| estimator | mode | query | localized | ≤0.5 m | first loc | RMSE | median | p95 | rot med | longest lost | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ORB-SLAM3 | relocalize | seq11 | 0.989 | 0.966 | 10.0 s | 0.089 m | 0.083 m | 0.135 m | 4.23° | 0.0 s | 28 |
| ORB-SLAM3 | relocalize | seq01 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 27 |
| ORB-SLAM3 | merge | seq11 | 0.924 | 0.902 | 68.6 s | 0.087 m | 0.081 m | 0.134 m | 4.22° | 0.0 s | 25 |
| ORB-SLAM3 | merge | seq01 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 25 |
| SP_SLAM3 | relocalize | seq11 | 1.000 | 0.977 | 0.0 s | 0.130 m | 0.095 m | 0.245 m | 3.96° | 0.0 s | 135 |
| SP_SLAM3 | relocalize | seq01 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 128 |
| SP_SLAM3 | merge | seq11 | 0.969 | 0.947 | 27.9 s | 0.096 m | 0.086 m | 0.155 m | 4.02° | 0.0 s | 134 |
| SP_SLAM3 | merge | seq01 | **0** (init_failure) | 0.000 | – | – | – | – | – | – | 127 |

The seq11 → seq11 sanity runs pass: 92–100 % localized, at session A's own accuracy. The seq11
map is therefore usable. Reuse fails in both directions between seq01 and seq11, so the lighting
and scene change, not the map, defeats DBoW place recognition. Learned retrieval addresses this
step: see [place_recognition.md](place_recognition.md), where real 01→11 Recall@1 is 37–39 %
with AnyLoc/MixVPR versus 3 % with DBoW2.

The rotation error on LunaPolaris is about 4° even for the seq01 → seq01 sanity run, while the
translation error is 3 cm. The SE(3) map→GT alignment is fitted on positions, so a constant
marker-to-camera rotation offset in the mocap GT cannot be absorbed and shows up as rotation
error. Session A's own aligned rotation ATE is also 4.2°. Compare the translation errors.

## Findings
1. **Same-lighting sanity checks pass for both systems and both modes.** Sim nominal → nominal
   localizes 98–100 % of frames. LunaPolaris seq01 → seq01 localizes 97–99 %. In merge mode the
   error equals session A's own ATE (sim 0.13 m, ORB LP 3 cm, SP LP 5 cm). This shows that the
   map frame is carried over correctly with no re-alignment on session B.
2. **Dimming (same sun angle, lower intensity) is harmless.** Both systems localize ≥ 98 % with
   the same error as nominal.
3. **A changed sun angle defeats both systems completely.** For low_sun and glare, both
   estimators in both modes localize 0 of 1800 frames, even though the trajectory is identical.
   SuperPoint + LightGlue matched far better at the feature level (65 % / 26 % vs ORB's 12 % / 0 %,
   [lunaryard_aerial_lighting.md](lunaryard_aerial_lighting.md)). That advantage does not
   carry over because SP_SLAM3 uses LightGlue only for map initialization and triangulation.
   Relocalization, loop detection and merge detection use DBoW3 retrieval plus BoW-node
   nearest-neighbour matching (`SearchByBoW`), the same pipeline as ORB-SLAM3. The fork's learned
   `PlaceRecognition` module is not used. ORB place-recognition diagnostics confirm where it
   breaks:
   * low_sun: candidates reach at most 24 BoW matches, with 0 Sim3 inliers.
   * glare: up to 51 BoW matches, but every Sim3 fails.
   * dim, for comparison: 83 BoW matches and 20 inliers, and it merges.

   Shadows move with the sun, so the local appearance the descriptors encode changes.
4. **Real data, different day.**
   * seq02 (similar exposure):
     * SP_SLAM3 relocalizes at 0.2 s (95 % localized) and merges at 37 s (96 %, 5.3 cm median).
     * ORB-SLAM3 needs 204 s to first relocalize (77 %, 3.4 cm median) and never merges.
   * seq11 (darker, 7 % saturated pixels, different day) fails for both systems in both modes,
     in both directions (seq01 map → seq11 and seq11 map → seq01). seq11 → seq11 works (92–100 %).
5. **SP_SLAM3 relocalize-only is inaccurate on real data.** It localizes often (95–99 %), but
   the median error is 0.09–0.32 m and the p95 is about 0.9 m, versus 5 cm in merge mode. In
   relocalize-only mode no local mapping runs, so tracking motion-only-optimizes against the
   sparse SP session-A map (266 keyframes). Errors grow along long stretches between
   relocalizations, up to 1.1 m, and none of the 32 relocalizations was false (> 0.5 m at the
   event). ORB's denser map does not show this (3 cm).
6. **Merge latency.** After a merge, error matches session A. The merge happens about 2 s into
   the sim queries and 22–37 s into the real queries. Earlier frames are in a separate map and
   are excluded.
7. **SP_SLAM3 loop correction corrupts the map.** With loop verification enabled in session B,
   loops between session-A keyframes were closed and the map was distorted. The nominal query
   reached 32.8 m RMSE, and dim reached a p95 of 33 m after a loop at frame 1591 (r2 image runs).
   r3 therefore enables the BoW verification fix for merge candidates only.
8. **Cost.**
   * Tracking time per frame: ORB 20–30 ms; SP 100–160 ms (GPU SuperPoint).
   * Atlas size: SP atlases are about 7× larger per keyframe.

**Implication for lander terrain maps:** neither system can reuse a map built under a different
sun angle. Same-illumination reuse works, and with merging it reaches mapping-session accuracy.
Cross-lighting reuse needs a learned global place recognizer, such as NetVLAD/AnyLoc-style
retrieval. It also needs relocalization that verifies with LightGlue matches instead of BoW-node
matching, or maps with illumination-invariant (e.g. geometric/DEM) features.

## Patches (patch 0006, both estimators) and why
* **Relocalize-only tracking.** Stock ORB-SLAM3 with a loaded atlas and
  `ActivateLocalizationMode` still runs `StereoInitialization` on the first frame and creates a
  new map. With `--relocalize-only`, tracking starts LOST, every untracked frame relocalizes, and
  RECENTLY_LOST extrapolation is disabled. No map, keyframe or reset is ever created.
* **Atlas paths.** Absolute paths are used. Upstream writes and reads `<name>.osa` relative to the
  cwd.
* **Atlas save waits** for local mapping, loop closing and global BA.
* **`MapPoint::PreSave` crash.** Upstream erased observations inside the loop that iterates them,
  so long sessions segfaulted on save. This is now fixed.
* **Driver output.** The driver writes `loaded_map.json` (`T_final_map`, the loaded keyframes) and
  `frame_log.csv` columns `in_loaded_map`, `relocalized` and `vo_only`.
* **SP_SLAM3 serialization.** The fork's disabled atlas save and load are re-enabled. It also
  serializes what the fork never saved: right keypoints, `NLeft`/`NRight`, DBoW3 BoW vectors,
  cameras and map origins. It initializes `bStepByStep`, which was uninitialized and hung tracking
  after a load.
* **SP_SLAM3 upstream bugs fixed** (they are only reachable when an atlas is loaded, so all
  earlier SP results are unchanged):
  1. `Tracking::Relocalization` never incremented `nCandidates`, so SP_SLAM3 **can never
     relocalize**.
  2. The relocalization guided search `SPmatcher::SearchByProjection(Frame, KeyFrame, …)` had two
     faults:
     * It wrote `mvpMapPoints[-1]` when no candidate was found, because `bestDist` starts at 2 and
        is always ≤ the threshold. This caused heap corruption and crashed the dim query
        (`double free or corruption`).
     * It used ORB Hamming thresholds (100 / 64) on L2 distances in [0, 2].

     It now uses the SP thresholds.
  3. `LoopClosing::DetectCommonRegionsFromBoW` projected `mvpLoopMapPoints`, which stay empty
     until a loop has been closed, instead of the candidate's map points. So **BoW loop and merge
     verification can never succeed**: SP_SLAM3 has no loop closure and no merging. This is fixed
     for merge candidates only (see Findings).

## Caveats
* Sim queries fly the identical trajectory as the map, so viewpoint changes are not tested. The
  LunaPolaris sequences also repeat nearly the same path.
* One run per cell. B jobs ran in parallel on shared nodes. Per-frame times are
  tracking-thread times with real-time playback (`docs/evaluation.md`).
* SP_SLAM3 session A ran without loop closure (upstream bug 3 above). Its map therefore has
  uncorrected drift, which is part of A's ATE.
* Merge mode's error excludes the frames before the merge (separate map, unknown frame).
* LunaPolaris rotation errors include the GT marker offset (see above).
