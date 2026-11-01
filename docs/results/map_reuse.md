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

## Learned relocalization (MixVPR + LightGlue) inside SP_SLAM3
The offline study ([place_recognition.md](place_recognition.md)) found that DBoW retrieval is the
step that fails across lighting. It also found that MixVPR top-5 retrieval with SuperPoint +
LightGlue + PnP verification relocalizes 78–95 % of sim nominal→low_sun queries and 32–35 % of
real 01→11 queries. This section tests that pipeline inside running SP_SLAM3, in relocalize mode
(session B starts lost in the loaded map and never creates a map).

### What was built (patch 0007, image `spslam3_9439971_r5.sif`)
* **MixVPR in C++.** `third_party/spslam3/export_mixvpr_slambench.py` exports the official
  GSV-Cities ResNet-50 MixVPR (4096-d) to TorchScript, with the preprocessing inside the model:
  BGR/gray uint8 → RGB, bilinear resize to 320×320, ImageNet normalization, and L2 normalization.
  It is exported with the LAC image's torch 2.3.1, which loads in the driver's LibTorch 2.3.0.
  Against the Python model on real LunaPolaris and sim frames the cosine is ≥ 0.9998. The model
  is baked into the image as `/opt/SP_SLAM3/mixvpr.pt`, like `lightglue.pt`.
* **Descriptor per keyframe** (`global_descriptors: true`, session A). Each new keyframe gets one
  MixVPR descriptor of its left image, stored as `KeyFrame::mGlobalDescriptor`. It is serialized
  in the atlas as KeyFrame boost class version 1, so atlases saved earlier still load (they just
  have no descriptors). This adds 16 KB per keyframe, about 1 % of an SP atlas. Session A was
  rebuilt with image r4 and `global_descriptors: true`
  (`configs/*_spslam3_stereo_map_gd.yaml`):

  | map | ATE RMSE (SE3) | keyframes | map points | atlas |
  |---|---|---|---|---|
  | sim nominal | 0.176 m | 594 | 67 351 | 947 MB |
  | LunaPolaris seq01 | 0.053 m | 273 | 15 065 | 411 MB |
  | LunaPolaris seq11 | 0.090 m | 188 | 10 806 | 284 MB |

  These maps are new; the r3 sim map had ATE 0.136 m. The sim ATE difference is run-to-run
  variance of SP_SLAM3 mapping without loop closure. Compare each learned-reloc row with its own
  map's ATE.
* **Retrieval** (`learned_reloc: true`, session B). `Tracking::Relocalization` calls
  `LearnedRelocalization()` instead of the DBoW3 candidates. It computes the query frame's MixVPR
  descriptor and takes the top-K (5) keyframes of the loaded map by cosine similarity, brute
  force on the GPU.
* **Verification** (`learned_reloc_verifier: lightglue`). For each candidate:
  * LightGlue (the existing TorchScript `lightglue.pt`) matches the frame's SuperPoint keypoints
    to the keyframe's;
  * only matches whose keyframe keypoint has a MapPoint are kept;
  * MLPnP RANSAC runs on those (expected inlier ratio 0.2, 1000 iterations; the stock 0.9 / 300
    rejected most cross-lighting candidates);
  * then the usual `PoseOptimization`, the `SearchByProjection` refinement and re-optimization.

  The best candidate is accepted with ≥ `learned_reloc_min_inliers` (50) inliers.
* **Ablation** `learned_reloc_verifier: bow`: MixVPR candidates with the stock `SearchByBoW`
  verification.
* **Temporal gate** (`learned_reloc_consistent_frames: 3`). An accepted frame only arms a
  pending relocalization. The pose is set, and tracking resumes, when 3 consecutive frames are
  accepted, each within 0.3 m / 5° of the previous accepted pose. A rejected frame resets the
  count. The motion between consecutive frames is ≤ a few cm at 15–30 fps, so the poses are
  compared directly instead of composing tracked motion.
* **Logging.** Every attempt writes one row to `algorithm/learned_reloc.csv`:
  * the top-K keyframe ids and similarities;
  * per candidate: LightGlue matches, matches with a MapPoint, PnP inliers, final inliers, time;
  * accepted / consistent count / confirmed, and the pose;
  * MixVPR, retrieval and verification times.

  `scripts/learned_reloc_diagnostics.py <run or config dir>` scores these rows against GT and
  writes `eval/learned_reloc_diagnostics.json`:
  * retrieval Recall@1 / @K, where a candidate keyframe is positive if its GT position is
    within 2 m / 30° (sim) or 0.5 m / 30° (real) of the query;
  * outcome per attempt: accepted, *retrieval miss* (no positive in the top K), or *verification
    fail* (a positive was retrieved but nothing was accepted);
  * false single-frame accepts and false confirmations (> 0.5 m or > 15°).
* **Upstream descriptor bug found and worked around.** In the first image (r4), LightGlue
  verification returned **0 matches for 90 % of candidates, including GT-positive ones**.
  `SPDetector::computeDescriptors` (`src/SuperPoint.cc`) normalizes the [256 × N] descriptor
  matrix with `torch::norm(desc, 2, 1)`, i.e. each *channel across the keypoints*, not each
  descriptor. SP_SLAM3's descriptors therefore have norm ≈ 0.46 and an image-dependent
  per-channel scale. LightGlue was trained on unit descriptors:
  * same-lighting matches survive (~1200 offline);
  * cross-lighting matches collapse from 360–470 (low_sun) and 26–172 (glare) to 0.

  Patch 0007 recovers the true unit-norm descriptors for LightGlue verification only
  (`learned_reloc_recover_descriptors: true`, the default; image r5). It solves the least-squares
  problem for the per-channel scales (the rows must have unit norm), which is exact offline
  (cos 1.0000), then row-normalizes. Keyframe results are cached. The rest of SP_SLAM3 is left as
  it was so that all earlier SP results stay comparable. This includes tracking, its own LightGlue
  use in initialization and triangulation, and `SearchByBoW`. The bug probably also weakens those
  steps (see Caveats).
* **Merge mode was not implemented.** Learned candidates for `LoopClosing` were left out; this
  section is relocalize mode only.

Configs:
* session A: `configs/*_spslam3_stereo_map_gd.yaml`;
* session B: `configs/*mapreuse*_spslam3_learnedreloc_{lgfix,lg,bow}.yaml`;
  * `lgfix` = LightGlue verification on recovered descriptors (r5);
  * `lg` = LightGlue on the raw descriptors (r4);
  * `bow` = SearchByBoW verification (r4; the BoW path is identical in r5);
* `*_lgfix1f.yaml`: the gate ablation with 1 consistent frame.

```bash
A=$(sbatch --parsable -p coc-gpu -q coc-ice --gres=gpu:v100:1 -c 8 --mem=32G -t 3:00:00 scripts/slurm_slambench.sbatch run configs/lunapolaris_seq01_spslam3_stereo_map_gd.yaml)
sbatch -p coc-gpu -q coc-ice --gres=gpu:v100:1 -c 8 --mem=32G -t 3:00:00 --dependency=afterok:$A scripts/slurm_slambench.sbatch run configs/lunapolaris_mapreuse_map01_loc11_spslam3_learnedreloc_lgfix.yaml
python scripts/learned_reloc_diagnostics.py ~/scratch/slam-bench-work/runs/lunapolaris_mapreuse_map01_loc11_spslam3_learnedreloc_lgfix
```

### Results
The baseline rows are the r3 `relocalize` rows from the tables above (DBoW retrieval, r3 maps,
not rerun). Extra columns:
* *false/ev* = false relocalization events (> 0.5 m or > 15° at the event) / evaluable events;
* *first loc (proc)* = processing time from the start of the run to the first localization.

LunaPolaris rotation includes the ~4.2° GT offset; compare translations.

Simulation (map = nominal):

| variant | query | localized | ≤0.5 m | first loc | RMSE | median | p95 | rot med | longest lost | ms/frame | false/ev | first loc (proc) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (DBoW, r3) | nominal | 1.000 | 1.000 | 0.0 s | 0.136 m | 0.120 m | 0.234 m | 0.24° | 0.0 s | 160 | – | – |
| baseline (DBoW, r3) | dim | 0.999 | 0.999 | 0.1 s | 0.136 m | 0.120 m | 0.233 m | 0.24° | 0.0 s | 160 | – | – |
| baseline (DBoW, r3) | low_sun | **0** | 0.000 | – | – | – | – | – | – | 161 | – | – |
| baseline (DBoW, r3) | glare | **0** | 0.000 | – | – | – | – | – | – | 161 | – | – |
| **MixVPR + LightGlue (r5)** | nominal | 0.999 | 0.999 | 0.1 s | 0.176 m | 0.135 m | 0.311 m | 0.29° | 0.0 s | 148 | 0/1 | 17.1 s |
| **MixVPR + LightGlue (r5)** | dim | 0.999 | 0.999 | 0.1 s | 0.176 m | 0.135 m | 0.312 m | 0.30° | 0.0 s | 117 | 0/1 | 10.2 s |
| **MixVPR + LightGlue (r5)** | low_sun | **0.999** | 0.999 | 0.1 s | 0.176 m | 0.136 m | 0.310 m | 0.33° | 0.0 s | 117 | 0/1 | 16.9 s |
| **MixVPR + LightGlue (r5)** | glare | **0.694** | 0.467 | 2.3 s | 2.130 m | 0.219 m | 5.499 m | 0.98° | 22.3 s | 152 | 0/9 | 27.5 s |
| MixVPR + LightGlue, raw desc. (r4) | nominal | 0.999 | 0.999 | 0.1 s | 0.176 m | 0.135 m | 0.311 m | 0.29° | 0.0 s | 122 | 0/1 | 16.9 s |
| MixVPR + LightGlue, raw desc. (r4) | dim | 0.999 | 0.999 | 0.1 s | 0.176 m | 0.135 m | 0.312 m | 0.30° | 0.0 s | 122 | 0/1 | 10.0 s |
| MixVPR + LightGlue, raw desc. (r4) | low_sun | 0.926 | 0.926 | 8.9 s | 0.162 m | 0.128 m | 0.283 m | 0.32° | 0.0 s | 119 | 0/1 | 38.2 s |
| MixVPR + LightGlue, raw desc. (r4) | glare | 0 | 0.000 | – | – | – | – | – | – | 149 | – | – |
| MixVPR + SearchByBoW (ablation) | nominal | 0.999 | 0.999 | 0.1 s | 0.176 m | 0.136 m | 0.311 m | 0.29° | 0.0 s | 148 | 0/1 | 0.9 s |
| MixVPR + SearchByBoW (ablation) | dim | 0.999 | 0.999 | 0.1 s | 0.176 m | 0.135 m | 0.312 m | 0.30° | 0.0 s | 122 | 0/1 | 0.8 s |
| MixVPR + SearchByBoW (ablation) | low_sun | 0 | 0.000 | – | – | – | – | – | – | 134 | – | – |
| MixVPR + SearchByBoW (ablation) | glare | 0 | 0.000 | – | – | – | – | – | – | 111 | – | – |

Real (LunaPolaris):

| variant | map → query | localized | ≤0.5 m | first loc | RMSE | median | p95 | rot med | longest lost | ms/frame | false/ev | first loc (proc) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (DBoW, r3) | 01 → 01 | 0.990 | 0.778 | 0.0 s | 0.358 m | 0.094 m | 0.838 m | 4.21° | 5.0 s | 139 | – | – |
| baseline (DBoW, r3) | 01 → 02 | 0.949 | 0.588 | 0.2 s | 0.488 m | 0.320 m | 0.902 m | 4.20° | 37.8 s | 146 | – | – |
| baseline (DBoW, r3) | 01 → 11 | **0** | 0.000 | – | – | – | – | – | – | 133 | – | – |
| baseline (DBoW, r3) | 11 → 11 | 1.000 | 0.977 | 0.0 s | 0.130 m | 0.095 m | 0.245 m | 3.96° | 0.0 s | 135 | – | – |
| baseline (DBoW, r3) | 11 → 01 | **0** | 0.000 | – | – | – | – | – | – | 128 | – | – |
| **MixVPR + LightGlue (r5)** | 01 → 01 | 0.999 | 0.804 | 0.2 s | 0.337 m | 0.083 m | 0.806 m | 4.15° | 0.2 s | 136 | 0/8 | 17.1 s |
| **MixVPR + LightGlue (r5)** | 01 → 02 | 0.999 | 0.757 | 0.2 s | 0.376 m | 0.219 m | 0.774 m | 4.47° | 0.4 s | 108 | 0/14 | 15.1 s |
| **MixVPR + LightGlue (r5)** | 01 → 11 | **0.710** | 0.220 | 262.7 s | 0.788 m | 0.669 m | 1.377 m | 16.40° | 0.0 s | 282 | 0/1 | 1039.7 s |
| **MixVPR + LightGlue (r5)** | 11 → 11 | 1.000 | 0.958 | 0.2 s | 0.171 m | 0.111 m | 0.362 m | 3.77° | 0.2 s | 135 | 0/6 | 11.1 s |
| **MixVPR + LightGlue (r5)** | 11 → 01 | **0.321** | 0.139 | 614.1 s | 0.923 m | 0.661 m | 1.579 m | 9.37° | 0.0 s | 182 | 0/1 | 1609.6 s |
| MixVPR + LightGlue, 1-frame gate (r5) | 01 → 11 | 0.708 | 0.389 | 259.1 s | 0.553 m | 0.428 m | 1.125 m | 10.23° | 5.1 s | 281 | 0/4 | 1027.1 s |
| MixVPR + LightGlue, 1-frame gate (r5) | 11 → 01 | **0.753** | 0.292 | 223.4 s | 0.867 m | 0.695 m | 1.605 m | 8.90° | 0.1 s | 284 | 0/4 | 820.0 s |
| MixVPR + LightGlue, raw desc. (r4) | 01 → 01 | 0.999 | 0.647 | 0.2 s | 0.436 m | 0.255 m | 0.807 m | 4.42° | 0.2 s | 113 | 0/9 | 10.2 s |
| MixVPR + LightGlue, raw desc. (r4) | 01 → 02 | 0.998 | 0.741 | 0.2 s | 0.422 m | 0.248 m | 0.866 m | 4.28° | 0.5 s | 110 | 0/15 | 14.8 s |
| MixVPR + LightGlue, raw desc. (r4) | 01 → 11 | 0 | 0.000 | – | – | – | – | – | – | 146 | – | – |
| MixVPR + LightGlue, raw desc. (r4) | 11 → 11 | 1.000 | 0.967 | 0.2 s | 0.167 m | 0.111 m | 0.346 m | 3.78° | 0.0 s | 106 | 0/5 | 6.4 s |
| MixVPR + LightGlue, raw desc. (r4) | 11 → 01 | 0 | 0.000 | – | – | – | – | – | – | 139 | – | – |
| MixVPR + SearchByBoW (ablation) | 01 → 01 | 0.999 | 0.644 | 0.2 s | 0.437 m | 0.253 m | 0.805 m | 4.42° | 0.2 s | 137 | 0/10 | 0.9 s |
| MixVPR + SearchByBoW (ablation) | 01 → 02 | 0.969 | 0.753 | 6.6 s | 0.370 m | 0.223 m | 0.766 m | 4.51° | 19.1 s | 138 | 0/13 | 10.2 s |
| MixVPR + SearchByBoW (ablation) | 01 → 11 | 0 | 0.000 | – | – | – | – | – | – | 128 | – | – |
| MixVPR + SearchByBoW (ablation) | 11 → 11 | 1.000 | 0.965 | 0.2 s | 0.164 m | 0.109 m | 0.340 m | 3.78° | 0.0 s | 135 | 0/5 | 0.7 s |
| MixVPR + SearchByBoW (ablation) | 11 → 01 | 0 | 0.000 | – | – | – | – | – | – | 125 | – | – |

### Per-attempt diagnostics (retrieval vs verification)
Columns:
* attempts = relocalization attempts (frames not tracked);
* R@1 / R@5 = attempts with a GT-positive keyframe at rank 1 / in the top 5;
* retrieval miss / verif. fail = attempts not accepted because no positive was retrieved / a
  positive was retrieved but verification rejected it;
* accept | pos = accepted fraction of the attempts that retrieved a positive;
* false = false single-frame accepts (> 0.5 m or > 15°); confirmed = temporally confirmed
  relocalizations.

| variant | map → query | attempts | R@1 | R@5 | accepted | retrieval miss | verif. fail | accept \| pos | false | confirmed (false) |
|---|---|---|---|---|---|---|---|---|---|---|
| LightGlue (r5) | sim → low_sun | 3 | 1.00 | 1.00 | 3 | 0 | 0 | 1.00 | 0 | 1 (0) |
| LightGlue (r4) | sim → low_sun | 135 | 0.57 | 0.84 | 11 | 22 | 102 | 0.10 | 0 | 1 (0) |
| SearchByBoW | sim → low_sun | 1800 | 0.62 | 0.81 | 0 | 344 | 1456 | 0.00 | 0 | 0 |
| LightGlue (r5) | sim → glare | 885 | 0.14 | 0.32 | 78 | 602 | 205 | 0.28 | 0 | 9 (0) |
| LightGlue (r4) | sim → glare | 1800 | 0.13 | 0.30 | 0 | 1265 | 535 | 0.00 | 0 | 0 |
| SearchByBoW | sim → glare | 1800 | 0.13 | 0.30 | 0 | 1265 | 535 | 0.00 | 0 | 0 |
| LightGlue (r5) | 01 → 02 | 75 | 0.92 | 1.00 | 54 | 0 | 20 | 0.72 | 0 | 14 (0) |
| LightGlue (r4) | 01 → 02 | 215 | 0.98 | 1.00 | 84 | 0 | 127 | 0.40 | 0 | 15 (0) |
| SearchByBoW | 01 → 02 | 1071 | 0.89 | 0.99 | 120 | 10 | 913 | 0.11 | 0 | 14 (0) |
| LightGlue (r5) | 01 → 11 | 12881 | 0.37 | 0.56 | 23 | 5536 | 7022 | < 0.01 | 0 | 1 (0) |
| LightGlue (r4) | 01 → 11 | 13382 | 0.38 | 0.57 | 0 | 5578 | 7498 | 0.00 | 0 | 0 |
| SearchByBoW | 01 → 11 | 13382 | 0.38 | 0.57 | 0 | 5578 | 7498 | 0.00 | 0 | 0 |
| LightGlue (r5) | 11 → 01 | 12893 | 0.26 | 0.42 | 41 | 7137 | 5061 | < 0.01 | 0 | 1 (0) |
| LightGlue (r4) | 11 → 01 | 13194 | 0.28 | 0.43 | 0 | 7137 | 5379 | 0.00 | 0 | 0 |
| SearchByBoW | 11 → 01 | 13194 | 0.28 | 0.43 | 0 | 7140 | 5376 | 0.00 | 0 | 0 |
| LightGlue, 1-frame gate (r5) | 01 → 11 | 12139 | 0.35 | 0.54 | 4 | 5506 | 6343 | < 0.01 | 0 | 4 (0) |
| LightGlue, 1-frame gate (r5) | 11 → 01 | 12106 | 0.24 | 0.38 | 5 | 7140 | 4344 | < 0.01 | 0 | 5 (0) |

The same-lighting runs (sim nominal / dim, 01 → 01, 11 → 11) relocalize on the first 3–34
attempts, with R@5 = 1.00 and every attempt accepted.

Matches per GT-positive candidate (median / p90; r5):

| map → query | LightGlue matches | … on a MapPoint | PnP inliers | final inliers | positives ≥ 50 |
|---|---|---|---|---|---|
| 01 → 01 | 468 / 589 | 181 / 306 | 106 / 206 | 95 / 198 | 123 of 123 |
| 01 → 02 | 235 / 305 | 98 / 130 | 40 / 64 | 49 / 68 | 122 of 251 |
| sim → glare | 90 / 307 | 67 / 230 | 0 / 42 | 0 / 56 | 95 of 551 |
| 11 → 01 | 73 / 176 | 30 / 73 | 11 / 25 | 9 / 32 | 36 of 10 245 |

### Timing (median per attempt; V100 and L40S mixed)
* **MixVPR forward:** 2.5–3.1 ms on an L40S, 5.2–5.7 ms on a V100. **Retrieval** over 188–594
  keyframes: 0.23 ms.
* **LightGlue verification** of 5 candidates (r5):
  * 93–208 ms per attempt, p95 ≈ 270 ms, including the descriptor recovery and PnP;
  * with the raw (r4) descriptors, 58–69 ms, because candidates with 0 matches skip PnP;
  * `SearchByBoW` verification: 20–48 ms.
* **TorchScript warm-up:** the first LightGlue calls take 4–11 s while the TorchScript graph is
  specialized to new input sizes. Data time to the first localization is 0.1–0.2 s for
  same-lighting queries, but processing time is 10–17 s with LightGlue versus 0.7–0.9 s with
  BoW. Playback queues frames rather than dropping them, so the backlog is worked off later. A
  dummy LightGlue call at startup would remove the warm-up; it was not added.
* **Per-frame tracking time:** 108–152 ms when the run is mostly tracked (r3: 128–161 ms). Lost
  stretches cost one attempt per frame (≈ 0.1–0.3 s). With real-time playback, the 01 → 11 run
  (182–281 ms/frame) fell behind and hit the 3600 s execution timeout after 11 334 of 13 382
  frames. It was rerun with a 7200 s timeout (`timeout_s` in the real `lgfix` configs).

### Findings
1. **Sun-angle change in sim is solved for low_sun.** 0 → **99.9 %** localized from frame 2
   (0.1 s). The error equals the nominal query's and session A's ATE (0.176 m RMSE, 0.136 m
   median). Retrieval and verification succeed on the first 3 attempts. After that, ordinary
   tracking against the nominal map holds through all 1800 frames.
2. **Glare is mostly limited by retrieval.** It goes from 0 → 69 % localized, first at 2.3 s,
   with 9 confirmed relocalizations and **0 false**. Each event lands 0.14–0.43 m from GT.
   * 68 % of attempts have no GT-positive keyframe in the top 5 (R@5 0.32, as offline).
   * When a positive is retrieved, LightGlue verification accepts 28 % of attempts.
   * The large RMSE / p95 (2.1 m / 5.5 m) comes from **tracking between relocalizations**, not
     from relocalization. Motion-only tracking against the nominal map under glare drifts to
     6–8 m before it is lost and relocalized again.
3. **Real cross-day reuse.** It goes from 0 to above 0 in both directions, but weakly.
   * 11 → 01: 32 % localized. The first confirmed relocalization is at 614 s; it is correct (0.12 m).
   * 01 → 11: 71 % localized. The first confirmed relocalization is at 263 s (0.06 m), but the
     median error is 0.67 m, and only 22 % of the input is within 0.5 m. The first attempt of
     this run hit the 3600 s timeout at 56 %; the table shows the rerun.

   Retrieval is about as good as offline (R@1 0.26–0.38, R@5 0.42–0.57). **Verification is the
   bottleneck:**
   * On GT-positive candidates in 11 → 01, LightGlue finds a median of 73 matches. Only 30 of them land on
     keyframe keypoints that have a MapPoint. The SP session-A maps are sparse: 10–15 k points
     over 188–273 keyframes.
   * About 11 matches survive PnP, so only 36 of 10 245 positive candidates reach the 50-inlier
     gate. 01 → 11 is similar: a median of 39 MapPoint matches per positive candidate.

   Offline, the database image's keypoints all had stereo depth, which is why the offline
   success rate (32–35 %) is far higher than the in-SLAM accept rate on positives (< 1 %).
   * The 3-frame gate costs a great deal here. 11 → 01 had 41 single-frame accepts, all correct,
     including 22 at 232–262 s, but only one run of 3 consecutive accepts. With the 1-frame gate
     ablation:
     * 11 → 01: first localization at 223 s instead of 614 s, and 75 % localized instead of 32 %;
     * 01 → 11: 4 relocalizations instead of 1, median error 0.43 m instead of 0.67 m;
     * all 9 accepts were correct.

     On this data the 3-frame gate only cost recall. No single-frame accept in any run was false (Finding 6).
   * **After a relocalization, tracking is not anchored to the map.** In the cross-day runs,
     87–95 % of the "localized" frames are `vo_only`: localization-mode visual odometry with too
     few map-point matches. (Same lighting: < 1 %; glare: 27 %.) The pose is carried by
     frame-to-frame VO, which drifts, for example from 0.12 m to 1.6 m over 300 s in 11 → 01.
     Each VO frame also attempts relocalization, which mostly fails verification. This is why
     ≤0.5 m stays at 14–39 %, and why the cross-day runs cost about 280 ms/frame. SP_SLAM3's
     own map matching across days is weak, because it uses the mis-normalized descriptors and
     no local mapping runs.
4. **The BoW-verifier ablation fails everywhere the lighting changes,** even when MixVPR
   retrieves the right keyframe:
   * low_sun: R@5 0.81, but `SearchByBoW` finds a median of 1 match on positives;
   * 01 → 11 / 11 → 01: 0 accepts.

   Learned retrieval alone is not enough. LightGlue verification is required, and with SP_SLAM3
   it only works on correctly normalized descriptors: r4 low_sun reached 93 % only after 9 s,
   and r4 glare and real cross-day stayed at 0.
5. **No regression under the same lighting.** Sim nominal / dim: 99.9 % localized at session A's
   accuracy. 01 → 01, 01 → 02 and 11 → 11: 99.9–100 % localized, slightly more than the r3
   baseline (01 → 02: 94.9 → 99.9 %, longest lost 37.8 s → 0.4 s).

   The real relocalize-mode errors (median 0.08–0.22 m, p95 ≈ 0.8 m on map 01) are the known
   relocalize-only tracking drift, not relocalization error:
   * accepted relocalizations are 4–10 cm from GT (median);
   * the BoW ablation, with the same map, has the same tracking error.
6. **No false relocalizations.** Across all learned-reloc runs, no confirmed relocalization and
   no single-frame accept was > 0.5 m / 15° from GT: 0 / 112 evaluable events, 0 false among
   588 single-frame accepts (116 confirmed).

### Caveats (learned relocalization)
* One run per cell. B jobs ran in parallel on V100 and L40S nodes, so ms/frame and the MixVPR
  times mix GPU types.
* New session-A maps (r4 image, `global_descriptors: true`). The sim map's ATE is 0.176 m versus
  r3's 0.136 m, so compare sim rows against their own map's ATE.
* Relocalize-only tracking runs no local mapping (see Findings 5 above). Its drift dominates the
  real-data errors and the glare errors. It was not changed here.
* The upstream SP descriptor normalization bug is fixed only for the learned-reloc LightGlue
  verifier. SP_SLAM3's tracking, `SearchByBoW`, `SearchByProjection` and its LightGlue
  initialization / triangulation still use the mis-normalized descriptors. The BoW ablation and
  all baselines include the bug. A global fix would likely improve SP_SLAM3's cross-lighting
  tracking, and the glare drift, but would change every SP result. It is left as an open item.
* Positives for the diagnostics are defined by keyframe GT position (2 m / 30° sim,
  0.5 m / 30° real). A "negative" candidate can still yield a correct pose; whether a pose is
  correct is judged only by the 0.5 m / 15° false-relocalization test.
* Merge mode with learned candidates (LoopClosing) was not implemented.
