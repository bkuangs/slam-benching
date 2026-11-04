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
  real-data errors and the glare errors. It was not changed here; map anchoring (r6, next section)
  addresses it.
* The upstream SP descriptor normalization bug is fixed only for the learned-reloc LightGlue
  verifier. SP_SLAM3's tracking, `SearchByBoW`, `SearchByProjection` and its LightGlue
  initialization / triangulation still use the mis-normalized descriptors. The BoW ablation and
  all baselines include the bug. A global fix would likely improve SP_SLAM3's cross-lighting
  tracking, and the glare drift, but would change every SP result. Image r6 (patch 0008) makes the
  global fix available as an opt-in setting (`sp_unit_descriptors`); see the next section.
* Positives for the diagnostics are defined by keyframe GT position (2 m / 30° sim,
  0.5 m / 30° real). A "negative" candidate can still yield a correct pose; whether a pose is
  correct is judged only by the 0.5 m / 15° false-relocalization test.
* Merge mode with learned candidates (LoopClosing) was not implemented.

### Learned relocalization: fixes (r6, patch 0008)
The r5 diagnosis left three problems on real cross-day data:
* verification rarely reaches the 50-inlier gate;
* tracking after a relocalization is mostly VO-only and drifts;
* SP_SLAM3 still uses mis-normalized SuperPoint descriptors outside the r5 verifier.

Patch 0008 (`third_party/spslam3/patches/0008-descriptor-fix-keyframe-density-map-anchoring.patch`,
image `spslam3_9439971_r6.sif`) adds one setting group per fix. Every new setting defaults to off, or
to the upstream value, so existing configs and images behave exactly as before. The adapter rejects
these settings on older images.

* **Fix 1: per-keypoint descriptor normalization everywhere** (`sp_unit_descriptors: true`).
  * `SPDetector::computeDescriptors` L2-normalizes each descriptor instead of each channel. Every
    consumer therefore sees unit descriptors: tracking, `SearchByProjection`, local-mapping
    triangulation, stereo matching, LightGlue initialization / triangulation and the BoW path. The
    r5 recovery in the verifier becomes a no-op.
  * The `SPmatcher` L2 thresholds were tuned for descriptors of norm ≈ 0.46, so they become
    settings (`sp_th_low` / `sp_th_high`). With unit descriptors they default to **0.52 / 1.19**,
    from `scripts/sp_descriptor_threshold_calibration.py`, run on seq01 stereo and temporal
    LightGlue matches:
    * TH_LOW 0.25 maps to 0.516 at equal true-match recall (sim: 0.513–0.518);
    * TH_HIGH 0.55 accepts every true match and every nearest impostor, so it is scaled by the
      median old descriptor norm (0.463).
  * The DBoW3 vocabulary was trained on the old descriptors. Unit descriptors are therefore scaled
    by `sp_bow_descriptor_scale` = √(256 / `sp_n_features`) = 0.46 before quantization, so they
    land at the norm the vocabulary expects. A run with scale 1 (no rescaling) checks how
    sensitive the BoW path is.
  * Fix 1 changes session A, so the maps were rebuilt with it.
* **Fix 2: denser session-A maps** (`kf_close_tracked`, `kf_close_non_tracked`, `kf_ref_ratio`,
  `kf_max_points`; upstream 100 / 70 / 0.75 / 100). These expose the hard-coded constants of
  `Tracking::NeedNewKeyFrame` / `CreateNewKeyFrame`. The variant used here is `kf_ref_ratio: 0.9`
  (insert a keyframe when fewer than 90 % of the reference keyframe's points are tracked) plus
  `kf_max_points: 400` (close stereo points created per keyframe). Variants tried on seq11:
  * `kf_max_points: 400` alone;
  * `kf_ref_ratio: 0.85`;
  * `kf_close_tracked: 150`.

  A 1200-keypoint budget is unchanged: more keypoints would also change tracking and the
  thresholds. LightGlue for triangulation matches is already used by SP_SLAM3 (`lightglue: true`).
* **Fix 3: map anchoring after relocalization** (`map_anchor: true`, option (a); relocalize mode
  only). When a localization-mode frame is VO-only (`mbVO`, too few map-point matches), and was
  not just relocalized, `Tracking::MapAnchor()` runs:
  1. Take the `map_anchor_k` (3) map keyframes nearest to the VO pose, within 5 m and 45° viewing
     angle.
  2. LightGlue-match the frame to each of them (unit descriptors). Take the union of the
     map-point matches with the frame's current VO map-point matches.
  3. Run MLPnP RANSAC if there are ≥ 15 matches; otherwise keep the VO pose.
  4. Run `PoseOptimization`, which needs ≥ 10 inliers, then `TrackLocalMap`.
  5. Accept if the local-map inliers are ≥ `map_anchor_min_inliers` (30) and the correction from
     the VO pose is ≤ 2 m / 15°. Otherwise restore the VO state exactly.

  An accepted frame leaves VO mode, and the following frames track the local map normally. Every
  attempt is logged to `algorithm/map_anchor.csv`. Option (b), periodic re-relocalization, was not
  needed: (a) already re-uses the relocalization verifier's machinery with a pose prior in place
  of retrieval.
* **Diagnostics.** `scripts/learned_reloc_diagnostics.py` now also reports the **gate pass rate on
  GT-positive candidates**: the fraction of positives with final inliers ≥ the gate, plus median /
  p90 LightGlue, map-point, PnP and final inliers (`positive_candidates` in
  `eval/learned_reloc_diagnostics.json`).

Configs (all pin image r6):
* session A: `configs/*_spslam3_stereo_map_gd_unitdesc.yaml` (Fix 1) and
  `configs/lunapolaris_seq{01,11}_spslam3_stereo_map_gd_dense.yaml` (Fixes 1 + 2);
* session B: `configs/*mapreuse*_spslam3_learnedreloc_<variant>.yaml`, where the variant is:
  * `unitdesc` = Fix 1;
  * `dense` = Fixes 1 + 2;
  * `lgfix_anchor` = Fix 3 alone, on the r4 maps;
  * `unitdesc_anchor` = Fixes 1 + 3, the final configuration;
  * `dense_anchor` = all three;
  * `unitdesc_anchor_gate1` = the final configuration with a 1-frame gate.

Run with:

```bash
A=$(sbatch --parsable -p coc-gpu -q coc-ice --gres=gpu:v100:1 -c 8 --mem=32G -t 3:00:00 scripts/slurm_slambench.sbatch run configs/lunapolaris_seq11_spslam3_stereo_map_gd_unitdesc.yaml)
sbatch -p coc-gpu -q coc-ice --gres=gpu:v100:1 -c 8 --mem=32G -t 3:00:00 --dependency=afterok:$A scripts/slurm_slambench.sbatch run configs/lunapolaris_mapreuse_map11_loc01_spslam3_learnedreloc_unitdesc_anchor.yaml
python scripts/learned_reloc_diagnostics.py ~/scratch/slam-bench-work/runs/lunapolaris_mapreuse_map11_loc01_spslam3_learnedreloc_unitdesc_anchor
```

The rows below come from two sources:
* Rows marked *dev* (Fix 1 alone, Fixes 1 + 2, all three) were run during development. They used the r5
  image with the patch-0008 library and driver bound over it. That is the same source as image r6
  (driver source sha1 `c9516337…`), with run outputs in a home-directory dev tree.
* All other r6 rows were run with image r6 and the configs listed above.

The dense maps (1.8–2.0 GB each) were not rebuilt on scratch because of the storage quota.

#### Session-A maps

| map | variant | ATE RMSE (SE3) | keyframes | map points | atlas | ms/frame |
|---|---|---|---|---|---|---|
| seq01 | upstream descriptors (r4, `_map_gd`) | 0.053 m | 273 | 15 065 | 411 MB | 107 |
| seq01 | **Fix 1** (r6, `_map_gd_unitdesc`) | **0.041 m** | 449 | 25 546 | 677 MB | 138 |
| seq01 | Fix 1 (dev) | 0.040 m | 468 | 26 584 | 705 MB | 141 |
| seq01 | Fix 1, BoW scale 1 (dev) | 0.043 m | 478 | 26 724 | 722 MB | 141 |
| seq01 | Fixes 1 + 2 (dev, `_map_gd_dense`) | 0.039 m | 1 216 | 49 644 | 1 810 MB | 161 |
| seq11 | upstream descriptors (r4, `_map_gd`) | 0.090 m | 188 | 10 806 | 284 MB | 101 |
| seq11 | **Fix 1** (r6) | **0.061 m** | 371 | 22 652 | 563 MB | 134 |
| seq11 | Fix 1 (dev) | 0.064 m | 354 | 21 473 | 537 MB | 134 |
| seq11 | Fixes 1 + 2 (dev) | 0.057 m | 1 361 | 59 529 | 2 038 MB | 152 |
| seq11 | Fix 1 + `kf_max_points: 400` only (dev) | 0.070 m | 318 | 19 192 | 482 MB | 134 |
| seq11 | Fix 1 + `kf_max_points: 400`, `kf_ref_ratio: 0.85` (dev) | 0.058 m | 1 295 | 58 681 | 1 941 MB | 152 |
| seq11 | Fix 1 + `kf_close_tracked: 150` (dev) | 0.066 m | 536 | 31 885 | 812 MB | 138 |
| sim nominal | upstream descriptors (r4) | 0.176 m | 594 | 67 351 | 947 MB | 155 |
| sim nominal | **Fix 1** (r6) | **0.151 m** | 596 | 69 316 | 952 MB | 156 |

* **Fix 1 improves mapping, with no regression.** Same-sequence ATE improves against SP_SLAM3's
  own numbers (seq01 0.050 m, seq11 0.091–0.098 m) and the r4 maps: seq01 0.053 → 0.041 m, seq11
  0.090 → 0.061 m, sim 0.176 → 0.151 m. The real maps get 1.6–2× more keyframes and map points, because the unit-norm matcher keeps more
  stereo and LightGlue matches. The sim map's size is unchanged.
* **The BoW-dependent parts do not degrade.** In SP_SLAM3 the DBoW3 vocabulary is used in only
  three places:
  * `TrackReferenceKeyFrame` (`SearchByBoW`);
  * DBoW relocalization;
  * loop / merge detection.

  Local-mapping triangulation uses LightGlue, and neighbour fusion uses projection. Feeding unit
  descriptors to the old vocabulary *without* rescaling (BoW scale 1) costs only 0.003 m of ATE
  (0.043 vs 0.040 m), so tracking is insensitive to the BoW mismatch either way. **Loop closing
  could not be evaluated:** no SP_SLAM3 mapping run here, with or without the fix, detected a loop
  or merged a map (`map_changed` = 0 in every frame log). The baseline had none to lose. The r6 DBoW
  relocalization path was not re-run, since learned relocalization replaces it.
* **Fix 2:** `kf_ref_ratio: 0.9` is what densifies the map (2.6–3.8× the keyframes of Fix 1).
  `kf_max_points: 400` alone adds nothing (318 keyframes), and `kf_close_tracked: 150` gives an
  intermediate map. The dense maps have the best ATE (0.039 / 0.057 m) but are 1.8–2.0 GB per
  sequence.

#### Session B, real (relocalize mode)

Cross-day cells marked "×3" are per-column medians of 3 identical runs, with the range in
brackets. The other cells are single runs. All runs have **0 false relocalizations** (0 events
> 0.5 m / 15°), so that column is omitted. Other columns:
* *VO-only* = fraction of localized frames tracked as localization-mode VO (too few map-point
  matches);
* *gate pass* = GT-positive verification candidates (keyframe within 0.5 m / 30°) with ≥ 50
  final inliers;
* *anchors* = accepted / attempted map anchors.

| variant | map → query | localized | ≤0.5 m | first loc | median | RMSE | p95 | ms/frame | VO-only | gate pass on positives | anchors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r5 | 01 → 01 | 0.999 | 0.804 | 0.2 s | 0.083 m | 0.337 m | 0.806 m | 138 | 0.001 | 123 / 123 | – |
| r5 | 01 → 02 | 0.999 | 0.757 | 0.2 s | 0.219 m | 0.376 m | 0.774 m | 110 | 0.004 | 122 / 251 (49 %) | – |
| r5 | 01 → 11 | 0.710 | 0.220 | 263 s | 0.669 m | 0.788 m | 1.377 m | 278 | **0.947** | 26 / 16 791 (0.15 %) | – |
| r5 | 11 → 11 | 1.000 | 0.958 | 0.2 s | 0.111 m | 0.171 m | 0.362 m | 137 | 0.001 | 75 / 75 | – |
| r5 | 11 → 01 | 0.321 | 0.139 | 614 s | 0.661 m | 0.923 m | 1.579 m | 183 | **0.930** | 36 / 10 245 (0.35 %) | – |
| Fix 1 (dev) | 01 → 01 | 1.000 | 0.727 | 0.2 s | 0.055 m | 0.416 m | 0.887 m | 158 | 0.000 | 30 / 30 | – |
| Fix 1 (dev) | 01 → 02 | 0.999 | 0.540 | 0.2 s | 0.420 m | 0.515 m | 1.019 m | 156 | 0.003 | 117 / 177 (66 %) | – |
| Fix 1 (dev) | 01 → 11 | 0.713 | 0.442 | 260 s | 0.345 m | 0.515 m | 1.074 m | 234 | 0.829 | 87 / 13 544 (0.64 %) | – |
| Fix 1 (dev) | 11 → 11 | 1.000 | 0.968 | 0.2 s | 0.066 m | 0.127 m | 0.312 m | 148 | 0.000 | 30 / 30 | – |
| Fix 1 (dev) | 11 → 01 | 0.745 | 0.290 | 230 s | 0.636 m | 0.802 m | 1.565 m | 233 | 0.822 | 111 / 10 038 (1.1 %) | – |
| Fixes 1 + 2 (dev) | 01 → 01 | 1.000 | 0.949 | 0.2 s | 0.037 m | 0.038 m | 0.056 m | 182 | 0.000 | 15 / 15 | – |
| Fixes 1 + 2 (dev) | 01 → 02 | 0.991 | 0.704 | 7.3 s | 0.385 m | 0.445 m | 0.851 m | 178 | 0.001 | 60 / 67 | – |
| Fixes 1 + 2 (dev) | 01 → 11 | 0.733 | 0.509 | 225 s | 0.327 m | 0.471 m | 0.968 m | 230 | 0.694 | 104 / 15 323 (0.68 %) | – |
| Fixes 1 + 2 (dev) | 11 → 11 | 1.000 | 0.977 | 0.2 s | 0.049 m | 0.057 m | 0.093 m | 179 | 0.000 | 15 / 15 | – |
| Fixes 1 + 2 (dev) | 11 → 01 | 0.903 | 0.457 | 88 s | 0.436 m | 0.710 m | 1.510 m | 224 | 0.623 | 100 / 9 710 (1.0 %) | – |
| Fix 3 alone (r6, r4 maps) | 01 → 11 | 0.711 | 0.703 | 262 s | 0.110 m | 0.139 m | 0.245 m | 250 | 0.395 | 9 / 6 742 (0.13 %) | 74 / 3 835 |
| Fix 3 alone (r6, r4 maps) | 11 → 01 | 0.715 | 0.481 | 258 s | 0.248 m | 0.532 m | 1.294 m | 290 | 0.617 | 11 / 7 178 (0.15 %) | 42 / 5 956 |
| **Fixes 1 + 3 (r6, final)** | 01 → 01 | 1.000 | 0.795 | 0.2 s | 0.042 m | 0.395 m | 1.080 m | 153 | 0.000 | 15 / 15 | 0 / 0 |
| **Fixes 1 + 3 (r6, final)** | 01 → 02 | 0.999 | 0.535 | 0.2 s | 0.422 m | 0.512 m | 0.852 m | 153 | 0.000 | 39 / 41 | 6 / 6 |
| **Fixes 1 + 3 (r6, final)** | 01 → 11 ×3 | **0.690** [0.612–0.708] | 0.682 | 249 s [242–258] | **0.100 m** [0.093–0.111] | 0.154 m | 0.278 m | 215 | **0.327** [0.306–0.351] | 85 / 26 989 (0.31 %) | 85–113 / ~3 000 |
| **Fixes 1 + 3 (r6, final)** | 11 → 11 | 1.000 | 0.951 | 0.2 s | 0.066 m | 0.148 m | 0.396 m | 148 | 0.000 | 35 / 35 | 1 / 1 |
| **Fixes 1 + 3 (r6, final)** | 11 → 01 ×3 | **0.750** [0.643–0.873] | 0.566 | 223 s [66–226] | **0.169 m** [0.167–0.249] | 0.363 m | 0.801 m | 224 | **0.421** [0.382–0.636] | 191 / 17 298 (1.1 %) | 45–60 / 4 300–5 500 |
| Fixes 1 + 2 + 3 (dev) | 01 → 01 | 1.000 | 0.949 | 0.2 s | 0.037 m | 0.038 m | 0.056 m | 182 | 0.000 | 15 / 15 | 0 / 0 |
| Fixes 1 + 2 + 3 (dev) | 01 → 02 | 0.992 | 0.617 | 7.5 s | 0.413 m | 0.497 m | 0.867 m | 178 | 0.000 | 11 / 11 | 2 / 2 |
| Fixes 1 + 2 + 3 (dev) | 01 → 11 ×3 | 0.748 [0.725–0.748] | 0.740 | 228 s [225–228] | 0.116 m [0.108–0.123] | 0.145 m | 0.301 m | 218 | 0.268 [0.258–0.284] | 67 / 22 685 (0.30 %) | 29–34 / ~2 700 |
| Fixes 1 + 2 + 3 (dev) | 11 → 11 | 1.000 | 0.977 | 0.2 s | 0.049 m | 0.057 m | 0.093 m | 178 | 0.000 | 15 / 15 | 0 / 0 |
| Fixes 1 + 2 + 3 (dev) | 11 → 01 ×3 | 0.607 [0.583–0.880] | 0.418 | 210 s [87–210] | 0.313 m [0.252–0.345] | 0.586 m | 1.273 m | 220 | 0.246 [0.197–0.341] | 212 / 26 181 (0.81 %) | 6–9 / 1 500–4 000 |
| **Fixes 1 + 3, 1-frame gate (r6)** | 01 → 11 ×3 | **0.896** [0.887–0.907] | 0.875 | **76 s** [75–102] | **0.095 m** [0.086–0.115] | 0.145 m | 0.317 m | 197 | **0.241** [0.237–0.249] | 6 / 8 237 (0.07 %) | 118–139 / ~3 050 |
| **Fixes 1 + 3, 1-frame gate (r6)** | 11 → 01 ×3 | **0.932** [0.930–0.937] | 0.762 | **60 s** [57–63] | **0.160 m** [0.155–0.191] | 0.321 m | 0.740 m | 212 | **0.359** [0.354–0.389] | 13 / 5 808 (0.22 %) | 56–68 / 4 500–4 900 |

Two runs with the same code as the final configuration also exist. A dev run of Fixes 1 + 3 gave
01 → 11 0.713 localized, 0.106 m median; 11 → 01 0.745, 0.178 m. A run with
`map_anchor_min_inliers: 20` on the dense map (11 → 01) gave 0.766, 0.375 m. Neither is in the
table.

Translation error of map-tracked vs VO-only frames (median):

| variant | 01 → 11: map-tracked | 01 → 11: VO-only | 11 → 01: map-tracked | 11 → 01: VO-only |
|---|---|---|---|---|
| r5 | 0.106 m (5 % of frames) | 0.809 m | 0.158 m (7 %) | 0.720 m |
| Fix 1 (dev) | 0.062 m (17 %) | 0.443 m | 0.126 m (18 %) | 0.781 m |
| Fix 3 alone | 0.085 m (61 %) | 0.154 m | 0.140 m (38 %) | 0.474 m |
| Fixes 1 + 3 (final, 3 runs) | 0.078–0.082 m (65–69 %) | 0.131–0.176 m | 0.084–0.143 m (36–61 %) | 0.37–0.45 m |
| Fixes 1 + 2 + 3 (dev, 3 runs) | 0.099–0.114 m (72–74 %) | 0.119–0.148 m | 0.171–0.249 m (66–80 %) | 0.67–1.16 m |

#### Session B, simulation (final configuration, non-regression check)

| variant | query | localized | ≤0.5 m | first loc | median | RMSE | p95 | ms/frame | VO-only | false/ev | gate pass on positives |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r5 (r4 map, ATE 0.176 m) | nominal | 0.999 | 0.999 | 0.1 s | 0.135 m | 0.176 m | 0.311 m | 148 | – | 0/1 | – |
| r5 | low_sun | 0.999 | 0.999 | 0.1 s | 0.136 m | 0.176 m | 0.310 m | 117 | – | 0/1 | – |
| r5 | glare | 0.694 | 0.467 | 2.3 s | 0.219 m | 2.130 m | 5.499 m | 152 | 0.27 | 0/9 | 95 / 551 (17 %) |
| **Fixes 1 + 3 (r6, map ATE 0.151 m)** | nominal | 0.999 | 0.999 | 0.1 s | 0.138 m | 0.151 m | 0.249 m | 160 | 0.000 | 0/1 | 15 / 15 |
| **Fixes 1 + 3 (r6)** | low_sun | 0.999 | 0.999 | 0.1 s | 0.137 m | 0.151 m | 0.256 m | 159 | 0.000 | 0/1 | 6 / 6 |
| **Fixes 1 + 3 (r6)** | glare | **0.978** | **0.857** | 2.5 s | 0.204 m | **0.448 m** | **0.832 m** | 164 | 0.014 | 0/3 | 21 / 31 (68 %) |

Run ids (`runs/<config>/<run id>`):
* r6 maps:
  * seq01 `20261103-212024_75a05a`;
  * seq11 `20261103-212024_f5cc77`;
  * sim `20261103-212024_2c9e19`.
* Final configuration:
  * 01 → 01 `20261103-221434_7bb941`;
  * 01 → 02 `20261103-221434_7fd196`;
  * 01 → 11 `20261103-221533_c89af4`, `20261103-221605_ec56d8`, `20261103-222234_0a8e27`;
  * 11 → 11 `20261103-221233_90eb1d`;
  * 11 → 01 `20261103-221303_b7f52a`, `20261103-221633_d4bb83`, `20261103-223818_429652`.
* Fix 3 alone:
  * 01 → 11 `20261103-212024_5235b6`;
  * 11 → 01 `20261103-212024_21d66f`.
* Sim:
  * nominal `20261103-212903_4e1ca5`;
  * low_sun `20261103-212903_294d83`;
  * glare `20261103-213627_85d396`.
* Final configuration, 1-frame gate (`..._unitdesc_anchor_gate1`):
  * 01 → 11 `20261103-224640_d00e6b`, `20261104-000332_fdb285`, `20261104-000928_d84ef5`;
  * 11 → 01 `20261103-224709_08a38e`, `20261104-000556_816721`, `20261104-001630_b90b34`.
* The dev rows are not on scratch, and are not kept in the repository. Their session-A atlases
  (0.5–2.0 GB each) were deleted to stay within the storage limits.

#### Findings (r6)
1. **Map anchoring (Fix 3) is the fix that matters for cross-day accuracy.** With it, VO-only frames
   are re-registered to nearby map keyframes:
   * VO-only falls from 93–95 % to 33–42 % of localized frames (final, median of 3);
   * the median fixed-frame error falls 0.669 → **0.100 m** (01 → 11) and 0.661 → **0.169 m**
     (11 → 01);
   * p95 falls 1.38 → 0.28 m and 1.58 → 0.80 m.

   Fix 3 alone, on the r4 maps with r5 descriptors, already gives most of this in 01 → 11 (0.110 m,
   VO-only 0.40). In 11 → 01 it gives less (0.248 m, VO-only 0.62): there, unit descriptors also
   matter, because they double the matches available to the anchor. Only 3–4 % (01 → 11) and
   1 % (11 → 01) of anchor attempts are accepted, but that is enough. Once anchored, frames track
   the map at the same-day error level (0.08 m median on 01 → 11). All r6 runs still have 0 false
   relocalizations; the anchor's 2 m / 15° correction bound and its local-map inlier check keep
   wrong anchors out.
2. **Fix 1 (unit descriptors) helps every stage, but does not by itself fix cross-day reuse.**
   * Mapping ATE improves 14–32 % (above).
   * On GT-positive candidates, LightGlue matches double (73 → 149 median in 11 → 01; 83 → 142 in
     01 → 11), and so do the map-point matches (30 → 72). PnP inliers rise only 11 → 20, and
     final inliers 9 → 24.
   * The gate pass rate on positives rises 0.35 → 1.1 % (11 → 01) and 0.15 → 0.64 % (01 → 11).
   * 11 → 01 localized 0.32 → 0.75, first relocalization at 230 s instead of 614 s.

     **Caveat:** run-to-run variance in recall is large. Fix 3 alone on r5 descriptors also reached
     0.715 in 11 → 01, and identical final-config runs range 0.64–0.87, with first relocalization
     at 66–226 s. The r5 0.32 was probably an unlucky run.
   * Without anchoring the tracking still drifts (VO-only 0.82–0.83, median 0.35–0.64 m).
   * Same-day: the median improves (01 → 01 0.083 → 0.042–0.055 m, 11 → 11 0.111 → 0.066 m), and
     the sim glare run improves a great deal (below).
3. **Denser maps (Fix 2) help same-day, not cross-day.**
   * Same-day accuracy improves most: 01 → 01 RMSE 0.34–0.42 → **0.038 m**, which removes the ~1 m
     excursions that the r5 and Fix 1 runs have on a stretch of seq01. 11 → 11 RMSE 0.13–0.17 →
     0.057 m.
   * The gate pass rate on positives is unchanged (0.68 % vs 0.64 %; 1.0 % vs 1.1 %). More map
     points per keyframe do not survive cross-day PnP: median 19 PnP inliers either way.
   * Combined with anchoring it is *worse* across days than Fixes 1 + 3: 11 → 01 median 0.313 m
     vs 0.169 m, VO-only error 0.67–1.16 m. Anchor attempts on the dense map succeed less often
     (6–9 accepted vs 45–60 per run). Even when a map keyframe within 1 m of the GT pose is among
     the anchor candidates, PnP reaches 10 inliers in 0.4 % of attempts on the dense map vs 4.1 %
     on the Fix 1 map. A likely cause is that the extra keyframes and up to 400 close points per keyframe add weakly
     constrained points, which match less reliably across days. This was not isolated.
   * It costs 3–4× the atlas size, and 20 % more ms/frame in same-day runs.

   It is therefore not in the final configuration.
4. **1-frame vs 3-frame gate on the final configuration.** The 1-frame gate
   (`learned_reloc_consistent_frames: 1`) is better on every cross-day metric, and the 3 repeats
   agree closely:
   * localized 0.69 → **0.90** (01 → 11) and 0.75 → **0.93** (11 → 01);
   * first relocalization 249 → 76 s and 223 → 60 s;
   * median error 0.100 → 0.095 m and 0.169 → 0.160 m; p95 0.28 → 0.32 m and 0.80 → 0.74 m;
   * still **0 false relocalizations**, with 1–5 relocalization events per run.

   With the r5 code the 1-frame gate only raised recall (11 → 01 0.32 → 0.75), and the poses then
   drifted. With anchoring, the extra relocalized frames are as accurate as the rest. The
   3-frame gate costs recall because cross-day verification passes are rare and seldom fall on 3
   consecutive frames. The 50-inlier gate, the 2 m / 15° anchor bound and the local-map check keep
   false relocalizations out even without the consistency check. Error splits: map-tracked
   0.079–0.097 m vs VO-only 0.105–0.196 m (01 → 11); 0.113–0.134 m vs 0.352–0.405 m (11 → 01).
   The gate pass rate on positives is lower (0.07 % / 0.22 %). Most likely the easy stretches now
   localize early, so the remaining verification attempts fall on the hard ones; this was not
   checked.
   **Recommended setting for cross-day reuse: 1-frame gate**. The default stays at 3, so that
   existing configs behave the same.
5. **No regression where things already worked.**
   * Same lighting stays at 99.9–100 % localized, first at 0.2 s, with 0 false relocalizations
     in every r6 run.
   * Sim nominal and low_sun match the map's ATE (0.151 m RMSE).
   * Sim glare improves from 69 % to **97.8 %** localized, RMSE 2.13 → 0.45 m, p95 5.5 → 0.83 m.
     This is mostly Fix 1: VO-only drops from 27 % to 1.4 % and only 28 anchors were attempted. With
     correct descriptors, motion-only tracking against the nominal map holds under glare instead of
     drifting to 6–8 m.
   * ms/frame is 148–160 on same-lighting runs (r5 110–138, because there are more map points to
     project) and 215–224 cross-day (r5 183–278).
6. **One regression: 01 → 02 accuracy, a threshold trade-off.** Every unit-descriptor run of 01 → 02 has median error
   0.39–0.42 m: four variants, all ≥ 0.99 localized, 0 VO-only, 0 false. r5 had 0.219 m (r3
   0.320 m, r4 0.248 m). The error is concentrated in the same route segments (240–480 s, 720–780 s)
   where r5 also reaches 0.4–0.8 m, plus 60–180 s, all while tracking the map. The cause is
   the TH_HIGH equivalent (1.19), which admits more wrong projection matches when tracking a map
   from another session. It was confirmed with dev runs of the final configuration with only the
   session-B `sp_th_high` changed (single runs, r6 image, dev output; the threshold applies to
   projection / tracking matches and to the local-map search):

   | session B `sp_th_high` | 01 → 01 median / RMSE | 01 → 02 median / RMSE | 11 → 11 median / RMSE | 01 → 11 loc / median / p95 / VO-only | 11 → 01 loc / median / p95 / VO-only |
   |---|---|---|---|---|---|
   | 1.19 (default, final) | 0.042 / 0.395 m | 0.422 / 0.512 m | 0.066 / 0.148 m | 0.690 / 0.100 / 0.28 m / 0.33 | 0.750 / 0.169 / 0.80 m / 0.42 |
   | 0.9 | – | 0.127 / 0.277 m | – | – | – |
   | 0.75 | **0.038 / 0.054 m** | **0.100 / 0.197 m** | **0.060 / 0.065 m** | 0.711 / 0.097 / 0.83 m / 0.66 | 0.734 / **0.575** / 1.34 m / **0.88** |

   A stricter threshold removes the 01 → 02 regression and beats r5 (0.219 m), and it also removes
   the same-day excursions (01 → 01 RMSE 0.395 → 0.054 m, as good as the dense map). Across days,
   however, it halves the frames that keep enough map matches: VO-only rises to 0.66 / 0.88, and
   11 → 01 drifts back to a 0.58 m median. Cross-day tracking needs the loose threshold, which
   accepts the weaker matches that survive the lighting change. Same-day reuse is better with the
   strict one. Each run has 0 false relocalizations. **So it is a trade-off, not a free fix:** use
   `sp_th_high: 0.75` for same-lighting reuse, and the default for cross-day reuse. A threshold
   that adapts to the number of tracked map points, for example, could get both; this was not
   tried.

#### Remaining failure diagnosis (real cross-day)
* **Verification across days is still the binding limit on recall.** With correct descriptors,
  LightGlue finds about 140 matches on a GT-positive keyframe and about 70 of them have a map
  point. Only about 20 are PnP inliers, so about 1 % of positives pass the 50-inlier gate. Most
  cross-day 2D–3D matches are wrong, so the limit is the appearance change between the two days
  (low spotlight + robot light vs overhead light on regolith), not map density. Fix 2 tripled the
  points and left the inlier count unchanged. Retrieval is unchanged (R@5 0.25–0.53).
* **Recall is a rare-event process.** Relocalization requires a short window where verification
  passes on consecutive frames. Identical runs differ by up to 2.7 min in the first relocalization
  and by 0.23–0.30 in the localized fraction. Comparisons therefore need repeats, and single dev rows
  are indicative only.
* **11 → 01 remains harder than 01 → 11.** It has more VO-only frames (38–64 %) at about 0.4 m
  error. On the seq11 map, anchor attempts succeed about 1 % of the time vs about 3.5 % on seq01,
  for the same reason as above. The seq11 map's KFs see seq01's regolith under a different light.
* **Remaining options:**
  * a gate on map-point *reprojection consistency* instead of a raw inlier count, accepting
    15–30-inlier PnP solutions that agree with the anchored track;
  * multi-session maps (one per lighting);
  * a learned matcher trained across lighting changes.

#### Caveats (r6)
* **Repeats.** Only the final configuration's cross-day cells and the all-fixes cross-day cells
  are medians of 3. The Fix 1, Fixes 1 + 2 and Fix-3-alone rows are single runs, and recall varies
  a lot between identical runs. Read their localized fraction and first-relocalization time as
  indicative only. The accuracy effects (median / p95 error, VO-only fraction) are large and
  consistent across repeats.
* **Dev rows.** They ran the patch-0008 library and driver bound over image r5, built from the same
  source as r6, so they are not image-r6 runs. All r6 and dev B runs used V100 GPUs; the r5 rows
  mix V100 and L40S.
* **Map variance.** Every fix that changes session A also gives a new map. Some of a row's
  difference is therefore mapping variance; compare each row with its own map's ATE.
* **Fix 2 scope.** Fix 2 explored the keyframe policy only. The keypoint budget (1200, equal to
  ORB-SLAM3's) was kept so that results stay comparable, and because the SuperPoint thresholds were
  chosen to fill it.
* **Fix 3 scope.** Option (b), periodic re-relocalization, was not implemented.
* **Calibration.** The unit-descriptor thresholds were calibrated on seq01. The same calibration on
  sim frames gives the same TH_LOW (0.513–0.518), so the constants are shared.
* **What was not measured.** Loop closing never fires in these SP mapping runs, so the effect of
  the BoW / descriptor change on loop closure is untested. The sim improvement was not ablated per
  fix.
