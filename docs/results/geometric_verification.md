# Geometric verification of low-inlier relocalization candidates across lighting

Learned relocalization (MixVPR top-5 → SuperPoint+LightGlue → PnP) accepts a candidate only with
≥ 50 PnP inliers. Inside SP_SLAM3 only about 1 % of correct cross-day candidates reach that
([map_reuse](map_reuse.md), claim 10). The hypothesis tested here is as follows. A weak image match
(8–30 inliers) plus agreement in 3D shape between the query and the map should accept more correct
candidates without accepting wrong places, because 3D shape does not change with the sun. This page is
the offline test (Stage 1).

## Short answer

**Geometry helps, but less than hoped, and not uniformly.** On held-out real cross-day pairs, compare
"≥ 50 inliers OR (≥ 8 inliers + geometry)" (rule D) with today's 50-inlier gate (rule A) at the same
gross false-positive rate. All numbers are % of queries.

**Stereo query depth** (what SP_SLAM3 has):

| pair | success, strict (0.1 m / 2°) | success, loose (< 0.25 m) | accepted share of correct candidates |
|---|---|---|---|
| 11→01 | 25.2 → 27.2 | 34.1 → 37.2 | 82 → 88 % |
| 01→11 | 26.6 → 33.3 | 32.3 → 47.2 | 72 → 93 % |

**LiDAR query depth:**

- **Ground-only LiDAR:** strict success 25.2 → 28.1 (11→01) and 26.6 → 34.3 (01→11).
- **Full LiDAR scans:** with the ICP-refined pose, strict success **25.2 → 45.2** and **26.6 → 48.5**.
  The lab walls make this the easy case.

**The ceiling is low offline.** Accepting every candidate with ≥ 8 inliers and no geometry (rule B)
reaches only 29.1 / 35.1 % strict success, at **6.9 / 10.7 % gross FP**. Geometry removes almost all
of those gross false positives and keeps most of the gain. With dense SGBM depth for PnP, the 50-inlier
gate already accepts 72–82 % of the correct candidates. The in-SLAM "1 %" is therefore a property of the
sparse SLAM map, not of the 50-inlier gate as such (see "Offline vs in-SLAM").

**Where geometry fails:**

- **Flat ground (degeneracy).** On ground-only clouds, ICP slides along the floor. The ICP-refined pose
  is worse than PnP for every ground-only and stereo source (e.g. 01→02 lidar_ground: 93.6 → 68.8 %
  strict). A conditioning test is needed: on stereo_ground it rejects 113 correct and 19 wrong-place
  candidates on the tuning pair.
- **Noisy depth on low-relief terrain (sim stereo).** SGBM noise is ±5 % (0.15–0.6 m at 3–12 m). That
  exceeds the terrain relief, so no threshold accepts any sim candidate from stereo geometry.

**Where it works best: rejecting wrong places.** In the sim, with GT depth, the shape check removes
**276 of 278** high-confidence (≥ 50-inlier) wrong-place candidates caused by the 5 m texture repeat
(nominal→dim), and keeps 1454 of 1455 correct ones. It takes all gross false positives to 0 on low_sun
and glare. With full LiDAR, the ICP pose also beats PnP (median 0.08 → 0.03 m).

**Stage 2 (SP_SLAM3 patch 0009) was not run.** The stereo gain is moderate and asymmetric between the
two directions, and the r7 image does not fit the storage budget (see "Stage 2 decision").

## Method

Code: `vpr/geometry.py` (point-to-plane ICP, conditioning, basin test; unit tests in
`tests/test_geometry.py`) and `scripts/geo_verify_benchmark.py` + `.sbatch`. The script has four stages:

| stage | where it runs | what it does |
|---|---|---|
| `match` | GPU, LAC image | MixVPR top-5 from the existing `rank_*.npz` → SP+LightGlue → PnP RANSAC. Same `Extractors`/`localize` as `sim/omnilrs/cross_lighting_reloc.py`; PnP depth as in [place_recognition](place_recognition.md): sim GT `depth0`, real SGBM (num_disp 96). |
| `lidar` | CPU, venv | Livox scans read straight from the bags; no prepared cache (see "LiDAR"). |
| `geometry` | CPU, 24 workers, LAC image | Query and map clouds for every source, then ICP for every candidate on which PnP succeeded. |
| `summary` | CPU | Accept rules, threshold tuning, tables. |

### Sessions, pairs and thresholds

Sessions are the VPR sessions of [place_recognition](place_recognition.md). Success and false-positive
definitions are the same as there:

| | success | gross FP | also reported |
|---|---|---|---|
| sim | < 0.25 m and < 2° | > 2 m or > 10° | – |
| real | < 0.1 m and < 2° | > 0.5 m or > 10° | **loose**: translation < 0.25 m only |

On real data the relative pose error between two LunaPolaris sessions does not carry the constant ~4.2°
GT offset (median rotation errors of correct candidates are ~1°). The loose threshold is reported anyway.

| suite | map → query pairs |
|---|---|
| sim | nominal → dim, low_sun, glare |
| real | 01 → 02, 01 → 11, 11 → 01; same-session sanity 01 → 01, 11 → 11 |

### Query and map 3D

**Query cloud (camera frame)** uses only the query session's own data:

| source | how the query cloud is made |
|---|---|
| sim `depth` | GT `depth0`, every 4th pixel, 0.3–30 m, 0.10 m voxels |
| sim `stereo` | SGBM (num_disp 32) on the query's own stereo pair |
| real `stereo` | SGBM (num_disp 96), every 2nd pixel, 0.3–6 m, 3 cm voxels |
| real `stereo_ground` | as `stereo`, cropped in the rover footprint frame to *h* ≤ 0.5 m and *r* ≤ 3 m, the crop of [lidar_ground_only](../research/lidar_ground_only.md) |
| real `lidar` | Livox scan nearest the image (scan time = midpoint of its point stamps; median \|dt\| 27–30 ms; frames with \|dt\| > 0.1 s, ~7 %, count as missing). Moved into cam0 with `/tf_static` (`T_cam0_lidar`, identical for seq01/02/11); 5 cm voxels |
| real `lidar_ground` | the same scan with the adapter's ground-only crop (`crop_lidar_points`, *h* 0.5 m, *r* 3 m) |

**Map cloud (map world frame):** the candidate keyframe's own cloud plus up to 8 neighbouring map
database frames, from the same source type:

- sim: within 3 m and 45°;
- real: within 1 m and 45°.

**Assumption:** map clouds are placed with the **map session's GT poses**. Session A is the map, so
this stands in for a well-optimized SLAM map. The query side never uses GT.

### Geometric check

1. Transform the query cloud by the PnP pose.
2. Run robust point-to-plane ICP against the map cloud:
   * coarse-to-fine correspondence gates (sim 1.0/0.5/0.25 m, real 0.3/0.15/0.08 m);
   * Huber weights;
   * map normals from 20 nearest neighbours, keeping planar points only.
3. Compute these measures:
   * **fitness:** fraction of query points with a map neighbour within 3× the inlier distance and a
     point-to-plane residual below the inlier distance (sim 0.10 m, real stereo 0.04 m, LiDAR 0.05 m);
   * **RMSE** of the inlier residuals;
   * **correction:** the change from the PnP pose to the ICP pose;
   * **conditioning**, two measures:
     * **Hessian:** smallest translational eigenvalue `cond_t` of the scale-normalized point-to-plane
       Hessian on the inliers (as in `lidar_degeneracy_probe.py`);
     * **basin test:** restart ICP from the result shifted by ±*s* horizontally and ±5° in yaw, and take
       the worst translation spread of the converged poses (sim *s* = 0.5 m, real 0.15 m). This is
       robust to noisy normals, which inflate the Hessian measure on stereo clouds.
4. Accept when fitness ≥ *f*, RMSE ≤ *e*, correction ≤ *c* (m) and ≤ 5°, `cond_t` ≥ *k* and
   basin spread ≤ *b*.

**Accept rules** (inliers from SP+LG PnP RANSAC; "geometry" = the check above):

| rule | accepts |
|---|---|
| A | ≥ 50 inliers (current) |
| B | ≥ floor, no geometry (false-positive control) |
| C | ≥ floor + geometry |
| D | ≥ 50 OR (≥ floor + geometry) |
| E | PnP succeeded (≥ 6) + geometry; best candidate by fitness |

Floors 8, 12, 20 and 30 were swept. Among accepted candidates, A–D pick the one with the most inliers.
"D, ICP pose" is rule D with the ICP pose replacing PnP wherever geometry passed.

### Threshold tuning (no tuning on the test pairs)

The thresholds come from a grid over *f*, *e*, *c*, *k* and *b*. The grid point chosen maximizes accepted
correct low-inlier candidates (floor ≤ n < 50) on the tuning pair, with **zero** wrong-place candidates
accepted. Ties go to the stricter conditioning setting.

The planned splits (sim dim, real 01→02) **cannot tune anything**: neither has a single correct candidate
with 8–49 inliers (every correct dim/02 candidate has ≥ 50). The tuning pairs used instead:

| suite | tuned on | held out |
|---|---|---|
| sim | nominal → low_sun | glare, dim |
| real, fold 1 | 01 → 11 | 11 → 01, 01 → 02, same-session pairs |
| real, fold 2 | 11 → 01 | 01 → 11 |

Only held-out pairs are quoted as results.

## Results

All outputs are in `slam-bench-work/vpr/geo_verify/`: `summary.md`/`.json` (sim, real fold 1) and
`summary_real_fold2.md`/`.json`.

### Where the low-inlier candidates are

Candidate counts (MixVPR top-5, PnP succeeded) by inlier band. "Correct" uses the strict threshold;
"wrong" means gross error.

| pair | 8–19: correct / wrong | 20–49: correct / wrong | ≥ 50: correct / wrong | queries with a correct cand. ≥ 50 / ≥ 8 |
|---|---|---|---|---|
| sim nominal→dim | 0 / 24 | 0 / 23 | 1455 / 278 | 100 / 100 % |
| sim nominal→low_sun | 9 / 268 | 8 / 64 | 655 / 15 | 77.5 / 78.6 % |
| sim nominal→glare | 2 / 390 | 15 / 19 | 100 / 0 | 17.5 / 19.7 % |
| real 01→02 | 0 / 2 | 0 / 5 | 3400 / 27 | 98.5 / 98.5 % |
| real 01→11 | 28 / 300 | 188 / 113 | 547 / 15 | 29.2 / 41.2 % |
| real 11→01 | 14 / 170 | 100 / 51 | 539 / 10 | 28.8 / 34.4 % |

Below 50 inliers, wrong places outnumber correct candidates by about 2× (real cross-day) to 20× (sim). Even a perfect verifier at
floor 8 lifts the share of queries that have a correct candidate only from 29 to 41 % (01→11) and from 29
to 34 % (11→01). In the sim, the low-inlier band is almost entirely wrong places: the lighting change
leaves either many inliers or none.

### Simulation (geometry from GT depth; tuned on low_sun)

Each cell is success % / gross FP % / accepted share of correct candidates % / median translation error
of localized queries (m).

| rule | nominal→low_sun (tuning) | nominal→dim (held out) | nominal→glare (held out) |
|---|---|---|---|
| A: ≥ 50 | 77.5 / 0.8 / 97 / 0.018 | 100.0 / 0.0 / 100 / 0.005 | 17.5 / 0.0 / 85 / 0.121 |
| B: ≥ 8 | 78.3 / 16.7 / 100 / 0.022 | 100.0 / 0.0 / 100 / 0.005 | 19.4 / **52.8** / 100 / 19.8 |
| B: ≥ 20 | 78.3 / 4.2 / 99 / 0.019 | 100.0 / 0.0 / 100 / 0.005 | 19.2 / 3.3 / 98 / 0.151 |
| C: ≥ 8 + geometry | 78.6 / **0.0** / 99 / 0.018 | 100.0 / 0.0 / 100 / 0.005 | 19.4 / **0.0** / 100 / 0.142 |
| D: ≥ 50 or (≥ 8 + geo) | 78.6 / 0.8 / 100 / 0.019 | 100.0 / 0.0 / 100 / 0.005 | 19.4 / 0.0 / 100 / 0.142 |
| E: geometry only | 78.6 / 0.0 / 99 / 0.021 | 100.0 / 0.0 / 100 / 0.006 | 19.4 / 0.0 / 100 / 0.142 |
| D, ICP pose | 78.9 / 0.8 / 100 / **0.002** | 100.0 / 0.0 / 100 / **0.002** | **23.1** / 0.0 / 100 / **0.002** |

Tuned: fitness ≥ 0.8, RMSE ≤ 0.05 m, correction ≤ 1 m, `cond_t` ≥ 0.005, basin ≤ 0.075 m.

* **Texture aliasing.** Rule C (geometry on every candidate) removes the ≥ 50-inlier aliased wrong
  places: 276 of 278 on dim and 15 of 15 on low_sun, while keeping 1454 / 1455 and 651 / 655 of the
  correct ones. Wrong-place RMSE is 0.036 m against 0.003 m for correct places. The terrain's
  shape does not repeat with its 5 m texture. The two that slip through on dim pass every threshold,
  including conditioning.
* **Success** gains only 1–2 points, because only 17 correct candidates per pair sit below 50 inliers.
  The ICP pose fixes the PnP error on glare (median 0.14 → 0.002 m, 17.5 → 23.1 %).
* **Sim stereo fails.** No threshold accepts any candidate on the tuning pair, so rules C and E accept
  nothing, and D reduces to A. SGBM is unbiased (median depth ratio 1.00) but noisy (p10–p90 ±5 %:
  0.15–0.3 m at 3–6 m, 0.4–0.8 m at 6–12 m). ICP corrections of ~0.27 m follow, larger than the terrain
  relief. Correct (0.041 m) and wrong (0.048 m) RMSE overlap. The Hessian conditioning is inflated by
  noisy normals (0.05–0.10 vs 0.004–0.017 for GT depth).

### Real, held-out cross-day pairs

11→01 comes from fold 1 (tuned on 01→11); 01→11 comes from fold 2 (tuned on 11→01). Each cell is
success % / gross FP % / accepted share of correct candidates %.

**Strict (0.1 m / 2°):**

| rule | stereo 11→01 | stereo 01→11 | stereo_ground 11→01 | stereo_ground 01→11 | lidar_ground 11→01 | lidar_ground 01→11 | lidar 11→01 | lidar 01→11 |
|---|---|---|---|---|---|---|---|---|
| A: ≥ 50 | 25.2 / 0.2 / 82 | 26.6 / 0.0 / 72 | 25.2 / 0.2 / 82 | 26.6 / 0.0 / 72 | 25.2 / 0.2 / 82 | 26.6 / 0.0 / 72 | 25.2 / 0.2 / 82 | 26.6 / 0.0 / 72 |
| B: ≥ 8 | 29.1 / 6.9 / 100 | 35.1 / 10.7 / 100 | 29.1 / 6.9 / 100 | 35.1 / 10.7 / 100 | 29.1 / 6.9 / 100 | 35.1 / 10.7 / 100 | 29.1 / 6.9 / 100 | 35.1 / 10.7 / 100 |
| B: ≥ 20 | 29.0 / 1.1 / 98 | 34.2 / 1.4 / 96 | 29.0 / 1.1 / 98 | 34.2 / 1.4 / 96 | 29.0 / 1.1 / 98 | 34.2 / 1.4 / 96 | 29.0 / 1.1 / 98 | 34.2 / 1.4 / 96 |
| C: ≥ 8 + geo | 14.9 / 0.0 / 31 | 28.4 / 0.1 / 80 | 13.5 / 0.0 / 40 | 27.2 / 0.3 / 60 | 23.0 / 0.0 / 70 | 27.2 / 0.6 / 68 | 26.2 / 0.0 / 85 | 32.0 / 0.0 / 85 |
| D: ≥ 50 or (≥ 8 + geo) | 27.2 / 0.2 / 88 | 33.3 / 0.1 / 93 | 26.4 / 0.2 / 87 | 32.7 / 0.2 / 89 | 28.1 / 0.2 / 95 | 34.3 / 0.6 / 92 | 29.2 / 0.2 / 99 | 35.4 / 0.0 / 96 |
| E: geometry only | 14.7 / 0.0 / 31 | 24.3 / 0.1 / 80 | 12.8 / 0.0 / 40 | 26.1 / 0.5 / 60 | 19.8 / 0.0 / 71 | 21.6 / 0.6 / 68 | 21.5 / 0.0 / 85 | 26.2 / 0.0 / 85 |
| D, ICP pose | 25.9 / 0.2 / 88 | 16.5 / 0.0 / 93 | 24.1 / 0.2 / 87 | 21.2 / 0.2 / 89 | 28.2 / 0.4 / 95 | 35.6 / 0.6 / 92 | **45.2** / 0.2 / 99 | **48.5** / 0.0 / 96 |

**Loose (translation < 0.25 m):**

| rule | stereo 11→01 | stereo 01→11 | stereo_ground 11→01 | stereo_ground 01→11 | lidar_ground 11→01 | lidar_ground 01→11 | lidar 11→01 | lidar 01→11 |
|---|---|---|---|---|---|---|---|---|
| A: ≥ 50 | 34.1 / 0.2 / 64 | 32.3 / 0.0 / 57 | 34.1 / 0.2 / 64 | 32.3 / 0.0 / 57 | 34.1 / 0.2 / 64 | 32.3 / 0.0 / 57 | 34.1 / 0.2 / 64 | 32.3 / 0.0 / 57 |
| B: ≥ 8 | 49.1 / 6.9 / 99 | 52.4 / 10.7 / 100 | 49.1 / 6.9 / 99 | 52.4 / 10.7 / 100 | 49.1 / 6.9 / 99 | 52.4 / 10.7 / 100 | 49.1 / 6.9 / 99 | 52.4 / 10.7 / 100 |
| B: ≥ 20 | 45.6 / 1.1 / 89 | 48.8 / 1.4 / 91 | 45.6 / 1.1 / 89 | 48.8 / 1.4 / 91 | 45.6 / 1.1 / 89 | 48.8 / 1.4 / 91 | 45.6 / 1.1 / 89 | 48.8 / 1.4 / 91 |
| C: ≥ 8 + geo | 18.8 / 0.0 / 22 | 39.5 / 0.1 / 69 | 20.1 / 0.0 / 31 | 38.8 / 0.3 / 51 | 36.5 / 0.0 / 63 | 42.9 / 0.6 / 64 | 45.0 / 0.0 / 85 | 47.3 / 0.0 / 83 |
| D: ≥ 50 or (≥ 8 + geo) | 37.2 / 0.2 / 70 | 47.2 / 0.1 / 85 | 37.9 / 0.2 / 71 | 45.2 / 0.2 / 76 | 44.0 / 0.2 / 85 | 50.3 / 0.6 / 86 | 48.6 / 0.2 / 94 | 51.2 / 0.0 / 91 |
| E: geometry only | 18.9 / 0.0 / 22 | 39.2 / 0.1 / 69 | 20.4 / 0.0 / 31 | 38.9 / 0.5 / 51 | 36.8 / 0.0 / 63 | 41.7 / 0.6 / 64 | 44.7 / 0.0 / 85 | 47.5 / 0.0 / 83 |
| D, ICP pose | 37.2 / 0.2 / 70 | 45.4 / 0.0 / 85 | 37.9 / 0.2 / 71 | 45.2 / 0.2 / 76 | 44.4 / 0.4 / 85 | 52.4 / 0.6 / 86 | 49.0 / 0.2 / 94 | 51.7 / 0.0 / 91 |

Gross FP 0.1 % = 1 query; 0.6 % = 5 queries. Floors 12–30 behave like 8 with slightly fewer accepts.
The full sweep is in `summary*.md`.

#### Observations

* **Rule D is the useful form.** Rules C and E need geometry for every candidate, and the shape
  thresholds tuned to reject low-inlier wrong places also reject many good ≥ 50-inlier candidates. For
  stereo on 11→01, C keeps only 31 % of correct candidates.
* **Stereo geometry recovers part of B's gain without B's wrong places**, but not all of it:
  * 11→01: +2.0 points strict / +3.1 loose of a possible +3.9 / +15.0;
  * 01→11: +6.7 / +14.9 of +8.5 / +20.1;
  * gross FP stays at 0.0–0.2 %, against 6.9–10.7 % for B ≥ 8.
  * A plain floor of 20 (B ≥ 20) gives similar success at 1.1–1.4 % gross FP. Geometry buys back that
    last ~1 % of wrong places.
* **Non-gross false positives grow.** Under the strict threshold, FP rises 6–10 % → 10–21 % (correct
  place, 0.1–0.5 m off). The low-inlier PnP poses are less accurate, and ICP against stereo or a ground
  plane does not fix them.
* **Full LiDAR is the only source whose ICP pose beats PnP:** median 0.08 → 0.03 m; 52 → 92 % of
  accepted candidates within the strict threshold. Its query success nearly doubles. The lab walls
  dominate this registration (see [lidar_ground_only](../research/lidar_ground_only.md)), so it says
  little about open terrain.
* **The ground-only crop accepts nearly as many correct candidates** as the full scan (92–95 % vs
  96–99 %). It is less safe: 9 wrong-place candidates are accepted on 01→11 (fold 2), 5 of which become
  gross FP queries (0.6 %). Its ICP pose does not improve on PnP: the floor leaves horizontal translation
  and yaw weakly constrained.

#### Conditioning: how often it rejects

Low-inlier candidates (8–49 inliers) that pass the shape and correction tests, broken down by whether
conditioning then rejects them. Each cell is correct / wrong place.

| source | pair | low-inlier cands | correct / wrong place | pass shape + correction | rejected by conditioning | accepted |
|---|---|---|---|---|---|---|
| stereo | 01→11 (tuning) | 1190 | 216 / 413 | 118 | 0 / 0 | 57 / 0 |
| stereo | 11→01 | 834 | 114 / 221 | 69 | 0 / 0 | 37 / 0 |
| stereo_ground | 01→11 (tuning) | 1190 | 216 / 413 | 474 | **113 / 19** | 71 / 0 |
| stereo_ground | 11→01 | 834 | 114 / 221 | 282 | **61 / 2** | 28 / 0 |
| lidar_ground | 01→11 (tuning) | 1190 | 216 / 413 | 398 | 0 / 0 | 152 / 0 |
| lidar_ground | 11→01 | 834 | 114 / 221 | 272 | 0 / 0 | 81 / 0 |
| lidar | 01→11 (tuning) | 1190 | 216 / 413 | 515 | 0 / 0 | 185 / 0 |
| lidar | 11→01 | 834 | 114 / 221 | 392 | 0 / 0 | 108 / 0 |

* **Where conditioning matters.** The test is essential on the stereo ground crop. Without it, 19 wrong
  places on the tuning pair pass the shape test, because a wrong pose on flat ground still fits the
  floor. The test also removes about half the correct candidates.
* **Where it does not.** On full stereo and full LiDAR, the walls and rocks condition the registration,
  and the shape thresholds alone already separate.
* **The ground-only LiDAR crop.** The ICP correction bound (≤ 0.25 m from PnP) does most of the
  rejecting. The tuned `cond_t` floors (0.005 / 0.01 in the two folds) are weak, and its held-out failures
  (5 gross queries on 01→11) are the flat-ground case that the conditioning tests do not catch at
  that floor.
* **The basin test was selected only in some cases:** stereo_ground in fold 1, LiDAR in fold 2, and
  sim depth. Its
  extra cost (~4× ICP) buys little over the Hessian measure once the correction bound is in place.

### Same lighting and same day (sanity)

Same-day pairs have no correct candidates with 8–49 inliers (01→02: 30 low-inlier candidates, all
inaccurate or wrong; 01→01: 13; 11→11: 49, 4 correct). Rule D is therefore **identical to A** for every
source, and no rule adds a gross false positive.

Each cell is success % / gross FP % / accepted share of correct candidates % / median translation error (m).
All pairs use fold-1 thresholds.

| rule (geometry source) | 01→02 | 01→01 | 11→11 |
|---|---|---|---|
| A, B, D (all sources) | 93.6 / 0.0 / 100 / 0.013 | 95.5 / 0.0 / 100 / 0.004 | 98.0 / 0.0 / 100 / 0.003 |
| C, stereo | 74.8 / 0.0 / 56 / 0.015 | 94.7 / 0.0 / 93 / 0.004 | 95.9 / 0.0 / 91 / 0.003 |
| C, stereo_ground | 68.6 / 0.0 / 44 / 0.019 | 79.2 / 0.0 / 52 / 0.006 | 83.5 / 0.0 / 63 / 0.005 |
| C, lidar | 87.4 / 0.0 / 93 / 0.014 | 88.0 / 0.0 / 92 / 0.004 | 89.1 / 0.0 / 91 / 0.003 |
| C, lidar_ground | 83.1 / 0.0 / 79 / 0.014 | 87.9 / 0.0 / 84 / 0.004 | 89.2 / 0.0 / 89 / 0.003 |
| D + ICP pose, stereo | 86.9 / 0.0 / 100 / 0.043 | 94.7 / 0.0 / 100 / 0.036 | 93.0 / 0.0 / 100 / 0.032 |
| D + ICP pose, stereo_ground | 83.7 / 0.0 / 100 / 0.021 | 90.1 / 0.0 / 100 / 0.010 | 92.3 / 0.0 / 100 / 0.014 |
| D + ICP pose, lidar | 93.7 / 0.0 / 100 / 0.022 | 96.2 / 0.0 / 100 / 0.015 | 97.7 / 0.0 / 100 / 0.013 |
| D + ICP pose, lidar_ground | 68.8 / 0.0 / 100 / 0.048 | 78.1 / 0.0 / 100 / 0.034 | 83.8 / 0.0 / 100 / 0.032 |

When PnP is already good (median 3–13 mm), replacing it with the ICP pose makes it worse for every
source. The worst case is the LiDAR ground crop (3–5 cm), where ICP slides along the floor. Requiring
geometry for every candidate (C) drops 7–56 % of correct same-day candidates. Use geometry only for
candidates below the inlier gate (rule D), and keep the PnP pose. If the ICP pose is used at all, use it
only with well-conditioned (walled or rocky) structure.

### Timing

These are offline costs per query or per candidate (CPU single thread for geometry, V100 for matching).

| step | sim | real |
|---|---|---|
| SP extract + LightGlue match (per candidate) + PnP | 12 + 18 + 3 ms | 12 + 20 + 2 ms |
| SGBM depth for PnP (real) | – | 144 ms |
| query cloud | 19 ms (depth) / 85 ms (stereo) | stereo 264 ms, stereo_ground 175 ms, LiDAR < 1 ms (bag decode not counted) |
| ICP (per candidate) | 260–320 ms | 145–400 ms |
| basin test (per candidate) | ~1.0–1.3 s | 0.6–1.6 s |
| map cloud (per map frame, built once per map) | 0.47 s / 1.2 s | 0.07–1.1 s |

The ICP is a numpy/scipy implementation (open3d is not in the image), so these are upper bounds. For
cheap verification inside SLAM, a sparse version against MapPoints would be needed.

### Offline vs in-SLAM

The offline 50-inlier gate already accepts 72–82 % of correct cross-day candidates, because PnP here uses
dense SGBM depth for every query keypoint. Inside SP_SLAM3, verification matches against the keyframe's
sparse MapPoints, and only ~1 % of correct cross-day candidates pass. So the large in-SLAM headroom does
not appear offline. What the offline test shows:

1. 3D shape separates correct from wrong places when the depth is good (GT, LiDAR, and real stereo at
   indoor ranges).
2. Ground-only geometry needs a conditioning test or a tight PnP-consistency bound.
3. Geometry does not make low-inlier PnP poses more accurate unless the structure is well conditioned
   (walls).

## Stage 2 decision

Stage 2 (SP_SLAM3 patch 0009, image `spslam3_9439971_r7.sif`) was **not run**, for two reasons:

1. **The gain is moderate, not clear-cut, for the source SP_SLAM3 would use.** With stereo, accepted
   correct candidates rise 72 → 93 % on held-out 01→11 but only 82 → 88 % on 11→01. Query success rises
   +2.0 / +6.7 points (strict) at equal gross FP, and non-gross FP rises. In SLAM, the query geometry
   would be sparse stereo keypoints against sparse MapPoints, which is weaker than the dense SGBM tested
   here. The r6 1-frame gate already localizes 90 / 93 % of cross-day frames through tracking and map
   anchoring (claim 10). The remaining in-SLAM benefit would be an earlier first relocalization.
2. **Storage.** An r7 image is ~3.3 GB, larger than the 3 GB scratch budget (scratch at 97.4 %). The
   superseded `spslam3_9439971_r4.sif` cannot be deleted, because 10 configs still reference it
   (`grep -r spslam3_9439971_r4 configs slambench`).

The best in-SLAM version, if space is made: accept a 15–49-inlier candidate when stereo keypoints
(depth from the stereo match) agree with the candidate's and covisible keyframes' MapPoints:

- point-to-point residual RMSE ≤ 2 cm;
- fitness ≥ 0.7;
- correction ≤ 0.25 m;
- a Hessian conditioning floor.

Keep the 50-inlier path unchanged (rule D).

## How to reproduce

```bash
# matching: GPU (V100 or L40S, never bare gpu:1), LAC image, ~15 min per suite
sbatch -p coc-gpu -q coc-ice --gres=gpu:v100:1 -t 2:00:00 scripts/geo_verify_benchmark.sbatch match --suite sim
sbatch -p coc-gpu -q coc-ice --gres=gpu:v100:1 -t 2:00:00 scripts/geo_verify_benchmark.sbatch match --suite real
# LiDAR: CPU, venv, reads the bags directly (~11 min per sequence, ~200 MB output)
sbatch -p ice-cpu -t 2:00:00 scripts/geo_verify_benchmark.sbatch lidar
# geometry: CPU, LAC image (sim ~25 min, real ~1.5 h on 24 cores; real needs the lidar outputs)
sbatch -p ice-cpu -c 24 --mem=64G scripts/geo_verify_benchmark.sbatch geometry --suite sim --workers 24
sbatch -p ice-cpu -c 24 --mem=96G -t 8:00:00 scripts/geo_verify_benchmark.sbatch geometry --suite real --workers 24
# tables (a few minutes; venv or LAC image)
python scripts/geo_verify_benchmark.py summary
python scripts/geo_verify_benchmark.py summary --suites real --tune real:real_seq11:real_seq01 --name summary_real_fold2
```

Outputs are in `slam-bench-work/vpr/geo_verify/` (~0.3 GB in total):

| file | contents |
|---|---|
| `cands_<map>__<query>.npz` | top-5 candidates, inliers, PnP poses and errors |
| `lidar_<session>.npz` | nearest scan per needed frame, full and ground crop, camera frame, float16 |
| `geo_<pair>.npz` | per-candidate ICP fitness, RMSE, correction, conditioning, basin spread, ICP pose error, timing |
| `summary*.md/json` | tables |

## Caveats

* **Map poses are GT.** A SLAM map adds its own error to the map cloud.
* **Sim queries fly the map's path.** Viewpoint change is small, which favours both PnP and ICP.
* **Real LunaPolaris is a small lab.** Walls are within a few metres. "Ground-only" is the honest proxy
  for open terrain, and only the LiDAR ground crop reaches the floor reliably (the camera points ~28°
  down).
* **Tuning is on one pair per fold**, with few correct low-inlier candidates (17 in the sim, 114–216 on
  real data). The thresholds are coarse, and held-out results differ between the folds.
