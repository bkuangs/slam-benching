# Map reuse: condensed summary

Short version of [map_reuse.md](map_reuse.md), which has the full tables, the patch details,
the dev runs and the reproduce commands.

**Question:** can a map built in session A be reused for localization in session B under different
lighting (sim) or on a different day (real)?

## Setup
* **Session A** builds and saves an atlas (stereo SLAM, `save_map: true`).
* **Session B** loads the atlas (`load_map:`) and runs in one of two modes:
  * `relocalize`: localization only. Tracking starts lost, and every untracked frame tries to
    relocalize. No keyframes or maps are created.
  * `merge`: stock ORB-SLAM3 multi-session mapping. B starts a new map that may later merge into
    A's map. Frames before the merge are not counted as localized.
* **Scoring** (`slambench/evaluation/map_reuse.py`): B is scored in **A's fixed map frame**. The
  map→GT transform comes from A's own trajectory, and B is **never re-aligned**.
  * *localized* = fraction of input time with a pose in the loaded map.
  * *≤0.5 m* = fraction localized with error ≤ 0.5 m.
  * *first loc* = data time of the first localization.
  * *false relocalization* = an event > 0.5 m or > 15° from GT.
* **Data**
  * **Sim:** OmniLRS Lunaryard aerial, 5 m AGL, 120 s (1800 frames). Map = `nominal`; queries =
    `nominal`, `dim`, `low_sun`, `glare`. Every query flies the **identical trajectory**, so only
    lighting changes.
  * **Real:** LunaPolaris seq01 / seq02 / seq11, about 900 s each, recorded on 3 different days
    along nearly the same path. All three share one mocap world frame (checked to within 1–2 cm).
    Seq11 is darker, with 7 % saturated pixels.
* **On LunaPolaris, compare translation.** Rotation error includes a constant ~4.2° GT
  marker-offset.

## 1. Stock systems (DBoW2/DBoW3 relocalization)
Patch 0006 added relocalize-only tracking and atlas save/load. It also fixed SP_SLAM3 bugs that
made relocalization, loop closure and merging impossible once an atlas was loaded.

**Sim (map = nominal)**, localized fraction:

| query | ORB-SLAM3 reloc / merge | SP_SLAM3 reloc / merge |
|---|---|---|
| nominal | 1.00 / 0.99 | 1.00 / 0.98 |
| dim | 1.00 / 0.99 | 1.00 / 0.98 |
| low_sun | **0 / 0** | **0 / 0** |
| glare | **0 / 0** | **0 / 0** |

The localized runs had 0.13 m RMSE, equal to the map's own ATE.

**Real**, localized fraction (median error):

| map → query | ORB-SLAM3 reloc | ORB-SLAM3 merge | SP_SLAM3 reloc | SP_SLAM3 merge |
|---|---|---|---|---|
| 01 → 01 | 0.98 (0.029 m) | 0.97 (0.028 m) | 0.99 (0.094 m) | 0.98 (0.047 m) |
| 01 → 02 | 0.77 (0.034 m), first at 204 s | **0** | 0.95 (0.32 m) | 0.96 (0.053 m) |
| 01 → 11 | **0** | **0** | **0** | **0** |
| 11 → 11 | 0.99 (0.083 m) | 0.92 (0.081 m) | 1.00 (0.095 m) | 0.97 (0.086 m) |
| 11 → 01 | **0** | **0** | **0** | **0** |

**Takeaways**
* Same-lighting reuse works and reaches mapping accuracy, so the map frame carries over correctly.
* Dimming is harmless.
* **Any sun-angle change (low_sun, glare) or the seq01↔seq11 day change defeats both systems
  completely.** In merge mode the query still tracks in its own map but never merges.
* The cause is retrieval and verification by bag-of-words. SP_SLAM3 uses the same DBoW +
  `SearchByBoW` pipeline as ORB-SLAM3, so its SuperPoint features don't help here. The shadows
  move with the sun, which changes the local appearance.
* SP_SLAM3 relocalize-only drifts on real data (median 0.09–0.32 m, p95 about 0.9 m) because no
  local mapping runs. Merge mode stays at 5 cm.
* **Cost:** ORB-SLAM3 runs at 20–30 ms/frame and SP_SLAM3 at 100–160 ms/frame. SP atlases are
  about 7× larger per keyframe.

## 2. Learned relocalization in SP_SLAM3 (patch 0007, image r5)
Relocalize mode only; merge mode with learned candidates was not implemented.

**Pipeline per relocalization attempt**
1. **Retrieval:** a MixVPR global descriptor of the query frame is compared with the descriptors
   stored per keyframe in session A. The top 5 keyframes become candidates.
2. **Verification:** LightGlue matches the frame's SuperPoint keypoints to each candidate's. MLPnP
   RANSAC and pose optimization then run, and the frame is accepted with ≥ 50 inliers.
3. **Temporal gate:** the relocalization is confirmed once N consecutive frames are accepted
   (`learned_reloc_consistent_frames`, default 3), each within 0.3 m / 5° of the previous accepted
   pose.

**Upstream bug found:** SP_SLAM3 L2-normalizes the descriptor matrix per *channel*, not per
keypoint. Descriptors have norm ≈ 0.46, and LightGlue returned 0 matches on 90 % of candidates
across lighting changes. r5 recovers unit descriptors for the verifier only.

**Results (r5, 3-frame gate)**

| map → query | localized | ≤0.5 m | first loc | median | false |
|---|---|---|---|---|---|
| sim → low_sun | 0 → **0.999** | 0.999 | 0.1 s | 0.136 m | 0 |
| sim → glare | 0 → **0.694** | 0.467 | 2.3 s | 0.219 m (RMSE 2.13 m) | 0/9 |
| 01 → 11 | 0 → 0.710 | 0.220 | 263 s | 0.669 m | 0 |
| 11 → 01 | 0 → 0.321 | 0.139 | 614 s | 0.661 m | 0 |

* **Glare is limited by retrieval:** R@5 = 0.32, so 68 % of attempts have no correct keyframe in
  the top 5.
* **Real cross-day reuse is limited by verification.** Retrieval works (R@5 0.42–0.57), but under
  1 % of correct candidates reach 50 inliers, because few LightGlue matches land on map points.
* **Tracking between relocalizations drifts:** 87–95 % of the "localized" cross-day frames are
  VO-only (too few map-point matches), which explains the 0.66 m median.
* **Ablation:** MixVPR + stock `SearchByBoW` verification localizes 0 % under every lighting
  change. Learned retrieval alone isn't enough; LightGlue verification is required.

## 3. Fixes (patch 0008, image r6)
All new settings default to off or to upstream values.

* **Fix 1: unit descriptors everywhere** (`sp_unit_descriptors: true`). Matcher thresholds are
  recalibrated to `sp_th_low` / `sp_th_high` = 0.52 / 1.19. Mapping ATE improves 14–32 % (seq01
  0.053 → 0.041 m, seq11 0.090 → 0.061 m, sim 0.176 → 0.151 m).
* **Fix 2: denser keyframes** (`kf_ref_ratio: 0.9`, `kf_max_points: 400`).
  * This helps same-day reuse (01 → 01 RMSE 0.34–0.42 → 0.038 m).
  * It doesn't help, and even hurts, cross-day reuse, and the atlas grows 3–4× (1.8–2.0 GB).
  * **Not adopted.**
* **Fix 3: map anchoring** (`map_anchor: true`). When a frame is VO-only, LightGlue-match it to the
  3 map keyframes nearest its current pose (within 5 m / 45°). Then run PnP and local-map
  tracking. The anchor is accepted with ≥ 30 inliers and a correction ≤ 2 m / 15°; otherwise the VO
  state is kept.

**Real cross-day results** (×3 = median of 3 runs):

| variant | map → query | localized | ≤0.5 m | first loc | median | p95 | VO-only |
|---|---|---|---|---|---|---|---|
| r5 | 01 → 11 | 0.710 | 0.220 | 263 s | 0.669 m | 1.38 m | 0.95 |
| Fix 3 alone | 01 → 11 | 0.711 | 0.703 | 262 s | 0.110 m | 0.25 m | 0.40 |
| Fixes 1 + 3 ×3 | 01 → 11 | 0.690 | 0.682 | 249 s | 0.100 m | 0.28 m | 0.33 |
| **Fixes 1 + 3, 1-frame gate ×3** | 01 → 11 | **0.896** | **0.875** | **76 s** | **0.095 m** | 0.32 m | 0.24 |
| r5 | 11 → 01 | 0.321 | 0.139 | 614 s | 0.661 m | 1.58 m | 0.93 |
| Fix 3 alone | 11 → 01 | 0.715 | 0.481 | 258 s | 0.248 m | 1.29 m | 0.62 |
| Fixes 1 + 3 ×3 | 11 → 01 | 0.750 | 0.566 | 223 s | 0.169 m | 0.80 m | 0.42 |
| **Fixes 1 + 3, 1-frame gate ×3** | 11 → 01 | **0.932** | **0.762** | **60 s** | **0.160 m** | 0.74 m | 0.36 |

**Sim results** (Fixes 1 + 3, 3-frame gate, single runs):
* nominal and low_sun: 0.999 localized, 0.151 m RMSE, equal to the map's ATE.
* glare: **0.978** localized (r5: 0.694), RMSE 2.13 → 0.45 m. This is mostly Fix 1: correct
  descriptors keep tracking on the map under glare.

**Takeaways**
1. **Map anchoring is the fix that matters for cross-day accuracy.**
   * Median error falls 0.67 → 0.10 m and 0.66 → 0.17 m.
   * VO-only frames fall from 93–95 % to 33–42 %.
   * Only 1–4 % of anchor attempts succeed, but that is enough to keep tracking on the map.
2. **Fix 1 helps every stage** (mapping ATE, matches per candidate, the sim glare run), but it
   doesn't fix cross-day reuse on its own.
3. **The 1-frame gate is the recommended cross-day setting.** Cross-day verification passes are
   rare and seldom come 3 in a row, so the 3-frame gate mostly cost recall. The 50-inlier check and
   the anchor bounds keep out false relocalizations. The default stays at 3.
4. **No false relocalizations in any run** (r5 and r6; 0 of 588 single-frame accepts in r5).
5. **Same lighting does not regress** (99.9–100 % localized, first at 0.2 s).
6. **One regression, 01 → 02** (median 0.22 → 0.42 m with unit descriptors). It comes from the
   loose `sp_th_high` (1.19):
   * `sp_th_high: 0.75` fixes it (0.10 m) and the same-day excursions (01 → 01 RMSE 0.054 m).
   * But it breaks cross-day tracking (11 → 01 median 0.58 m, VO-only 0.88).
   * So use 0.75 for same-lighting reuse and the default for cross-day reuse.

## Remaining limits and caveats
* **Verification is the remaining cross-day limit.** A correct keyframe gives about 140
  LightGlue matches, about 70 of them on map points, but only about 20 PnP inliers. Only about 1 %
  of correct candidates pass the 50-inlier check. The limit is the appearance change between the
  days, not map density: Fix 2 tripled the points without changing the inlier count.
* **11 → 01 is harder than 01 → 11:** anchors succeed about 1 % of the time on the seq11 map
  versus about 3.5 % on seq01.
* **Recall varies a lot between identical runs.** First relocalization differs by up to 2.7 min,
  and the localized fraction by 0.23–0.30. Single-run rows are indicative only; accuracy effects
  are consistent across repeats.
* **The viewpoint is easy:** sim queries repeat the map's trajectory exactly, and the real
  sequences repeat nearly the same path.
* **Not tested:**
  * the 1-frame gate on same-day, 01 → 02 and sim queries;
  * merge mode with learned candidates;
  * loop closure with unit descriptors (no SP_SLAM3 mapping run ever closed a loop).
* **Options not tried:**
  * a reprojection-consistency gate that accepts 15–30-inlier solutions;
  * multi-session maps, one per lighting;
  * a matcher trained across lighting changes;
  * an adaptive `sp_th_high`.

## Configs and run ids (final configuration)
Run outputs are under `~/scratch/slam-bench-work/runs/<config>/<run id>/`.

* **Session A maps (r6, Fix 1):** `configs/*_spslam3_stereo_map_gd_unitdesc.yaml`. Run ids: seq01
  `20261103-212024_75a05a`, seq11 `20261103-212024_f5cc77`, sim `20261103-212024_2c9e19`.
* **Fixes 1 + 3:** `configs/*mapreuse*_spslam3_learnedreloc_unitdesc_anchor.yaml`.
  * 01 → 11: `20261103-221533_c89af4`, `20261103-221605_ec56d8`, `20261103-222234_0a8e27`
  * 11 → 01: `20261103-221303_b7f52a`, `20261103-221633_d4bb83`, `20261103-223818_429652`
  * sim: nominal `20261103-212903_4e1ca5`, low_sun `20261103-212903_294d83`, glare
    `20261103-213627_85d396`
* **Fixes 1 + 3, 1-frame gate:** `configs/lunapolaris_mapreuse_map{01_loc11,11_loc01}_spslam3_learnedreloc_unitdesc_anchor_gate1.yaml`.
  * 01 → 11: `20261103-224640_d00e6b`, `20261104-000332_fdb285`, `20261104-000928_d84ef5`
  * 11 → 01: `20261103-224709_08a38e`, `20261104-000556_816721`, `20261104-001630_b90b34`
* **Per-attempt diagnostics:** `python scripts/learned_reloc_diagnostics.py <run or config dir>`.
