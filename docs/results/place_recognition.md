# Place recognition across lighting: learned VPR vs DBoW2, and retrieval + verification relocalization

Goal: a drone maps terrain, and the map is later reused under a different sun angle. Earlier tests
showed two things:

- **Feature level** (`docs/results/lunaryard_aerial_lighting.md`): when the map frame is chosen from
  GT, SuperPoint+LightGlue relocalizes across sun-angle changes and ORB does not.
- **System level** (`docs/results/map_reuse.md`): ORB-SLAM3 and SP_SLAM3 relocalize 0% across lighting
  changes, in sim and on LunaPolaris seq01↔seq11. The DBoW2 keyframe database fails before any
  matching is attempted.

This benchmark asks:

1. Do learned global descriptors (AnyLoc, MixVPR) find the right map frame across lighting where
   DBoW2 does not?
2. Does learned retrieval followed by SP+LightGlue + PnP verification relocalize end to end, with
   no oracle? What is its false-positive rate?

## Short answer

**Retrieval is the bottleneck, and learned VPR mostly fixes it.**

- **Sim, nominal map → low_sun query:** Recall@1 is 3% for DBoW2, 85% for AnyLoc and 56% for
  MixVPR.
- **Real, seq01 → seq11 (different lab lighting):** Recall@5 is 15% for DBoW2, 69% for AnyLoc and
  61% for MixVPR.
- **Sim end to end (top-5 + SP+LightGlue + PnP, stereo depth):**
  - Across the non-glare lighting changes: AnyLoc reaches 96% and MixVPR 82%, against 38% for DBoW2
    (oracle 100%).
  - On nominal→low_sun alone: 95% / 78% / 15%.
- **ORB as the verifier fails across lighting even when the frame is correct.** It gets 0–1% with
  oracle retrieval on every sim sun-angle pair and on real seq01↔seq11.
  - So for a lighting-robust relocalizer, both the retrieval and the local features have to
    change.
- **Glare is unsolved by every retrieval method.**
  - Glare changes the sun azimuth, so the shadows flip.
  - Retrieval reaches only 9–31% Recall@1 into or out of glare, and about 3% with glare as the map.
  - Verification itself still works (oracle + SP+LightGlue: 58–98%).
- **Real seq01↔seq11 end to end:** AnyLoc/MixVPR + SP+LightGlue reach 25–35% success at top-5,
  against 41–43% for the oracle and 5% for DBoW2.
  - Most of the gap to 100% is inaccurate poses beyond the strict 0.1 m / 2° threshold, not wrong
    places (see the false-positive analysis).
- **False positives:**
  - With the ≥20-inlier gate, DBoW2 retrieval gives up to 8% gross wrong-place localizations in
    sim. Learned retrieval gives 2.4–4.9%.
  - Raising the gate to **≥50 inliers** (ORB-SLAM3's own relocalization threshold) cuts gross false
    positives to ≤0.8% for the learned methods and costs 1–4 points of success.
  - At ≥100 inliers, gross false positives are ≤0.1% in sim and real.

## Method

### Data and sampling

- **Sim:** OmniLRS `lunaryard_aerial_{nominal,low_sun,dim,glare}`.
  - Every condition flies the same 1800-frame lawnmower trajectory (5 m AGL, 60° pitch, 15 Hz) over
    a ~45 × 40 m area. Only the lighting differs.
  - Images are the rectified left camera, 640×360. Session frames are matched by row index.
  - **Database:** every 5th frame from frame 0 (360 frames, ~0.67 m apart).
  - **Queries:** every 5th frame from frame 2 (360 frames). Each query sits ~0.27 m from the
    nearest database frame, so a same-lighting pair never queries an identical image.
  - All 16 map→query lighting pairs are run.
- **Real:** LunaPolaris stereo sequences 01, 02 and 11. They share one mocap world frame.
  - GT is `/vrpn_mocap/ZED2_camera_marker/rectified/pose` composed with the adapter's
    `T_body_cam0`, so it is the rectified left camera. The nearest mocap sample within 20 ms is
    used; frames without one are dropped.
  - Images are the slambench adapter's rectified inputs. seq11 uses `right_row_shift_px: -1.63`.
  - The rover drives ~30 m in ~15 min (~3 cm/s, often stationary) in a ~3 × 6 m area. All three
    sequences follow almost the same path.
  - **Database:** keyframe-like sampling. A frame is added after ≥0.2 m or ≥15° of motion since the
    last database frame, with a minimum gap of 7 frames (≈0.47 s). This gives 157 / 162 / 159
    database frames for seq01 / 02 / 11.
  - **Queries:** every 15th frame (1 Hz) from frame 7, skipping database frames. This gives
    825 / 840 / 871 queries.
  - Pairs: 01→11, 11→01, 01→02, 02→01, plus same-session shifted sanity checks 11→11 and 01→01.
  - seq11 is the different-lighting session (low spotlight + robot light, more small rocks). seq02
    has the same ceiling light as seq01.

### Positives and success thresholds

**Positives (retrieval).** A database frame within *d* metres and 30° of the query's GT pose counts
as a positive.

- **Sim, *d* = 2 m.**
  - The footprint at 5 m AGL and 60° pitch is ~10 m across, so frames 2 m apart overlap by well
    over 50%.
  - The lawnmower lines are 10 m apart, so the next line is never a positive.
- **Real, *d* = 0.5 m.**
  - The forward-looking camera is ~0.4 m above the floor, and the trajectory spans ~3 × 6 m.
  - Within 0.5 m and 30° the views overlap heavily. The database spacing is ≤0.2 m, so every query
    has several positives.
  - 100% of queries in every pair have at least one positive.

Recall@K counts only queries that have a positive. In this benchmark that is every query.

**Success (end to end).**

- **Sim:** pose error < 0.25 m and < 2°.
- **Real:** pose error < 0.1 m and < 2°. The lab is small, so a 0.25 m error would be a different
  place on the mat.

**False positive (FP):** a query is localized (≥20 PnP inliers) but the pose is outside the success
threshold.

**Gross FP:** a false positive whose error is beyond the positive radius (> 2 m sim / > 0.5 m real)
or above 10°. This is the dangerous wrong-place case for a lander. The remaining false positives are
correct places with inaccurate poses.

### Retrieval methods

#### DBoW2 (ORB, ORBvoc)

This is a faithful pure-numpy reimplementation of what ORB-SLAM3 uses (`vpr/dbow.py`):

- ORB-SLAM3's `ORBvoc.txt`, copied out of the ORB-SLAM3 container: k=10, L=6, 971,814 words.
- Greedy Hamming descent through the tree, as in `TemplatedVocabulary::transform`.
- TF-IDF weights, L1-normalised BoW vectors, and the L1 score
  `s = 1 - 0.5·|v/|v| - w/|w||₁`.
- An inverted-index database. Shared-word counts are recorded for diagnostics.

ORB features come from OpenCV ORB with ORB-SLAM3's parameters (1200 features, scale 1.2,
8 levels, FAST threshold 7). Keypoints are spread over the image with a round-robin over a 40 px
grid, which approximates ORB-SLAM3's octree distribution. Ranking is the plain L1 score.

ORB-SLAM3's additional candidate logic is **not** emulated:

- the covisibility-group score accumulation;
- the `0.8·maxCommonWords` filter.

That logic can only remove candidates. DBoW2's ranking is already poor here (see the shared-words
table), so this omission does not make the baseline worse than ORB-SLAM3.

#### AnyLoc (DINOv2-G VLAD)

A compact reimplementation of AnyLoc-VLAD-DINOv2 (`vpr/models.py`):

- **Backbone:** DINOv2 **ViT-G/14** from torch.hub (`dinov2_vitg14`). The full model, not ViT-L, fit
  on the L40S (7.4 GB peak).
- **Features:** the **value** facet of layer 31, hooked on `blocks[31].attn.qkv`.
  - Images are center-cropped to a multiple of 14: 630×350 gives 1125 patches × 1536 dimensions.
  - Patch features are L2-normalised.
- **VLAD:** hard assignment, intra-normalisation, then L2 normalisation, giving 32 × 1536 = 49,152
  dimensions.
- **Vocabulary:** k = 32 clusters, cosine k-means with k-means++ seeding.
  - It is built **per map session, from that session's database frames only**: 200k patch
    features sampled across the database frames.
  - Query frames are never used. The database and the queries are both encoded with the map's
    vocabulary.

#### MixVPR (R50, 4096)

Official MixVPR (`amaralibey/MixVPR`): ResNet50 cut after layer3, 4 mixer layers, output
1024 × 4 = 4096 dimensions.

- **Weights:** the README's pretrained GSV-Cities checkpoint (`resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt`),
  downloaded with gdown.
- **Input:** resized to 320×320, as in the paper. This distorts the 16:9 aspect ratio.
- The repo's modules are loaded directly with importlib, because its package `__init__` imports timm,
  which is not in the image.

#### CosPlace (R50, 2048)

Optional extra: `gmberton/cosplace` from torch.hub, ResNet50 with a 2048-dimensional descriptor, at
native resolution.

#### Oracle (GT nearest)

Database frames ranked by GT distance, with frames outside the angle threshold pushed back. This is
the upper bound for verification.

Global descriptors are ranked by cosine similarity with brute-force search.

### End-to-end relocalization

For each query, the top-K retrieved database frames (K = 1 or K = 5) are each matched against the
query:

- **sp_lg:** SuperPoint (1200 keypoints) + LightGlue.
- **orb:** OpenCV ORB, 1200 features, with Hamming ratio and mutual check.

Matched database keypoints are lifted to 3D in the map's world frame, using the database frame's GT
pose and its depth:

- **Sim:** both the **GT depth** and **stereo depth**.
- **Real:** stereo depth only.

Stereo depth is OpenCV **SGBM** (3-way, block 5) on the rectified pair, with a left-right
consistency check of ≤1 px. numDisparities is 32 in sim and 96 on real data.

Pose estimation reuses `localize()` from `sim/omnilrs/cross_lighting_reloc.py`: EPnP RANSAC (2000
iterations, 3 px), then LM refinement.

The candidate with the most inliers is chosen. A query with fewer than 20 inliers counts as
"no localization". The pose error is computed against GT (`pose_error`).

## How to reproduce

```bash
# 1) sessions (venv; real GT read from the bags, ~11 min)
source ~/scratch/slam-bench-work/venv/bin/activate
python scripts/vpr_prepare_sessions.py --out /storage/ice1/3/6/bkuang7/slam-bench-work/vpr/sessions
# 2) benchmark on an L40S inside the LAC image (sim ~2 h, real ~1 h incl. e2e)
sbatch scripts/vpr_benchmark.sbatch --suite sim  --out /storage/ice1/3/6/bkuang7/slam-bench-work/vpr/results/sim
sbatch scripts/vpr_benchmark.sbatch --suite real --out /storage/ice1/3/6/bkuang7/slam-bench-work/vpr/results/real
# 3) tables
python scripts/vpr_summary.py --sim .../vpr/results/sim --real .../vpr/results/real --out tables.md
```

### Weights and assets

All of these live under `/storage/ice1/3/6/bkuang7/slam-bench-work/`:

- `vpr_weights/ORBvoc.txt`, plus a parse cache `ORBvoc.txt.npz`.
- `vpr_weights/MixVPR/` (the repo) and `vpr_weights/mixvpr_resnet50_4096.ckpt`.
- `torch_home/hub/`: dinov2 and cosplace hub repos and checkpoints.

### Outputs

Each results directory holds:

- `recall.csv`
- `rank_<map>__<query>.npz` (rankings)
- `e2e.csv` (per query × method × verifier × depth × K)
- `timing.json`
- `meta.json`

Unit tests for the pure-python parts are in `tests/test_vpr.py`: positives, recall@k, oracle
ranking, subsampling, and DBoW2 transform / TF-IDF / L1 scoring against a toy vocabulary.

## Results

### 1. Retrieval: simulation (4×4 lighting matrix)

Rows are the map lighting and columns the query lighting. 360 queries per pair, positive = within
2 m and 30°.

**Sim R@1** (%, rows = map lighting, columns = query lighting)

| method | map \ query | nominal | low_sun | dim | glare | cross-lighting mean |
|---|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | nominal | 100 | 3 | 61 | 2 | 22 |
|  | low_sun | 2 | 100 | 4 | 2 | 3 |
|  | dim | 57 | 3 | 100 | 2 | 21 |
|  | glare | 3 | 2 | 2 | 100 | 3 |
| AnyLoc (DINOv2-G VLAD) | nominal | 100 | 85 | 100 | 9 | 65 |
|  | low_sun | 64 | 100 | 68 | 31 | 54 |
|  | dim | 100 | 86 | 100 | 14 | 66 |
|  | glare | 4 | 3 | 2 | 100 | 3 |
| MixVPR (R50, 4096) | nominal | 100 | 56 | 99 | 12 | 56 |
|  | low_sun | 41 | 100 | 35 | 15 | 30 |
|  | dim | 100 | 54 | 100 | 13 | 56 |
|  | glare | 2 | 9 | 0 | 100 | 4 |
| CosPlace (R50, 2048) | nominal | 100 | 2 | 97 | 4 | 34 |
|  | low_sun | 8 | 100 | 9 | 2 | 6 |
|  | dim | 95 | 7 | 100 | 4 | 35 |
|  | glare | 0 | 0 | 1 | 100 | 0 |
| oracle (GT nearest) | nominal | 100 | 100 | 100 | 100 | 100 |
|  | low_sun | 100 | 100 | 100 | 100 | 100 |
|  | dim | 100 | 100 | 100 | 100 | 100 |
|  | glare | 100 | 100 | 100 | 100 | 100 |

**Sim R@5** (%, rows = map lighting, columns = query lighting)

| method | map \ query | nominal | low_sun | dim | glare | cross-lighting mean |
|---|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | nominal | 100 | 11 | 86 | 7 | 35 |
|  | low_sun | 12 | 100 | 13 | 9 | 11 |
|  | dim | 82 | 12 | 100 | 12 | 35 |
|  | glare | 12 | 8 | 8 | 100 | 9 |
| AnyLoc (DINOv2-G VLAD) | nominal | 100 | 95 | 100 | 32 | 76 |
|  | low_sun | 91 | 100 | 90 | 46 | 75 |
|  | dim | 100 | 97 | 100 | 36 | 78 |
|  | glare | 7 | 4 | 4 | 100 | 5 |
| MixVPR (R50, 4096) | nominal | 100 | 78 | 100 | 30 | 69 |
|  | low_sun | 68 | 100 | 64 | 28 | 53 |
|  | dim | 100 | 79 | 100 | 28 | 69 |
|  | glare | 12 | 26 | 4 | 100 | 14 |
| CosPlace (R50, 2048) | nominal | 100 | 15 | 99 | 9 | 41 |
|  | low_sun | 16 | 100 | 18 | 5 | 13 |
|  | dim | 99 | 14 | 100 | 7 | 40 |
|  | glare | 2 | 2 | 5 | 100 | 3 |
| oracle (GT nearest) | nominal | 100 | 100 | 100 | 100 | 100 |
|  | low_sun | 100 | 100 | 100 | 100 | 100 |
|  | dim | 100 | 100 | 100 | 100 | 100 |
|  | glare | 100 | 100 | 100 | 100 | 100 |

**Why DBoW2 fails:** shared vocabulary words (median over queries), best positive / retrieved top-1:

| map \ query | nominal | low_sun | dim | glare |
|---|---|---|---|---|
| nominal | 62 / 62 | 10 / 15 | 12 / 13 | 11 / 17 |
| low_sun | 10 / 15 | 76 / 76 | 7 / 11 | 12 / 19 |
| dim | 12 / 13 | 7 / 11 | 24 / 24 | 6 / 11 |
| glare | 11 / 17 | 13 / 19 | 6 / 10 | 78 / 78 |

Within the same lighting, the true place shares 24–78 vocabulary words with the query. Across a
sun-angle change it shares only 6–13, and the retrieved top-1 frame (a wrong place) shares more
(10–19). The ORB descriptors of the same terrain point change too much under a new light direction
to quantize to the same words. ORB-SLAM3's candidate filter (`minCommonWords = 0.8·maxCommonWords`,
covisibility grouping) cannot recover the true place from this, because wrong frames already win on
the raw word overlap. `dim` (same sun direction, 10× darker) is the one change DBoW2 survives
(57–61% R@1), because it changes contrast and not shadows.

### 2. Retrieval: real (LunaPolaris)

**Real (LunaPolaris) Recall@1 / @5 / @10** (%)

| map → query | queries (with positive) | DBoW2 (ORB, ORBvoc) | AnyLoc (DINOv2-G VLAD) | MixVPR (R50, 4096) | CosPlace (R50, 2048) | oracle (GT nearest) |
|---|---|---|---|---|---|---|
| seq01 → seq11 | 871 (871) | 3 / 15 / 29 | 37 / 69 / 81 | 39 / 61 / 68 | 14 / 41 / 51 | 100 / 100 / 100 |
| seq11 → seq01 | 825 (825) | 3 / 16 / 28 | 20 / 41 / 53 | 32 / 51 / 57 | 13 / 28 / 43 | 100 / 100 / 100 |
| seq01 → seq02 | 840 (840) | 31 / 60 / 77 | 98 / 100 / 100 | 92 / 100 / 100 | 89 / 98 / 99 | 100 / 100 / 100 |
| seq02 → seq01 | 825 (825) | 32 / 62 / 76 | 95 / 100 / 100 | 90 / 100 / 100 | 79 / 94 / 97 | 100 / 100 / 100 |
| seq11 → seq11 | 871 (871) | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| seq01 → seq01 | 825 (825) | 95 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 99 / 100 / 100 | 100 / 100 / 100 |

- **01↔02** (same ceiling light) is easy for every learned method: 90–98% R@1.
  DBoW2 reaches only 31–32% R@1, so even with the same lighting a second session is hard for it.
- **01↔11** (different light source and direction, more small rocks): all methods drop.
  - DBoW2: 3% R@1, 15% R@5.
  - AnyLoc: 20–37% R@1, 41–69% R@5.
  - MixVPR: 32–39% R@1, 51–61% R@5.
  - 11→01 is harder than 01→11, presumably because the low spotlight in seq11 makes its database
    frames dark and high-contrast.
- **Same-session shifted sanity checks** (11→11, 01→01): 95–100% for every method, so the pipeline
  and GT are consistent.

### 3. End-to-end relocalization

Cells are success % / FP % / gross FP % of **all** queries. The remainder is "no localization".

**sim end-to-end relocalization**: success % / FP % / gross FP % of all queries (success: error < 0.25 m and < 2.0°; localized = best of top-K by PnP inliers, ≥ 20 inliers; FP = localized but outside the success threshold; gross FP = error > 2.0 m or > 10°, i.e. wrong place)

| retrieval | verifier | 3D | K | same lighting | cross lighting (12 pairs) | nominal→low_sun | low_sun→glare | nominal→glare | glare→nominal |
|---|---|---|---|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | sp_lg | gt | 1 | 100 / 0 / 0 | 12 / 5 / 5 | 4 / 5 / 5 | 2 / 1 / 1 | 1 / 1 / 0 | 2 / 0 / 0 |
| DBoW2 (ORB, ORBvoc) | sp_lg | gt | 5 | 100 / 0 / 0 | 22 / 9 / 8 | 16 / 18 / 17 | 10 / 2 / 1 | 2 / 7 / 5 | 5 / 2 / 2 |
| DBoW2 (ORB, ORBvoc) | sp_lg | stereo | 1 | 100 / 0 / 0 | 12 / 5 / 5 | 4 / 6 / 5 | 2 / 1 / 1 | 1 / 1 / 0 | 2 / 1 / 1 |
| DBoW2 (ORB, ORBvoc) | sp_lg | stereo | 5 | 100 / 0 / 0 | 22 / 8 / 8 | 15 / 18 / 16 | 9 / 3 / 1 | 3 / 5 / 4 | 4 / 2 / 2 |
| DBoW2 (ORB, ORBvoc) | orb | gt | 1 | 100 / 0 / 0 | 10 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| DBoW2 (ORB, ORBvoc) | orb | gt | 5 | 100 / 0 / 0 | 14 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| DBoW2 (ORB, ORBvoc) | orb | stereo | 1 | 100 / 0 / 0 | 10 / 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| DBoW2 (ORB, ORBvoc) | orb | stereo | 5 | 100 / 0 / 0 | 14 / 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| AnyLoc (DINOv2-G VLAD) | sp_lg | gt | 1 | 100 / 0 / 0 | 46 / 2 / 2 | 85 / 2 / 2 | 32 / 2 / 1 | 3 / 2 / 1 | 0 / 2 / 2 |
| AnyLoc (DINOv2-G VLAD) | sp_lg | gt | 5 | 100 / 0 / 0 | 55 / 4 / 3 | 95 / 2 / 2 | 43 / 4 / 3 | 18 / 7 / 3 | 1 / 6 / 4 |
| AnyLoc (DINOv2-G VLAD) | sp_lg | stereo | 1 | 100 / 0 / 0 | 45 / 2 / 2 | 85 / 2 / 1 | 30 / 2 / 1 | 3 / 2 / 1 | 0 / 2 / 2 |
| AnyLoc (DINOv2-G VLAD) | sp_lg | stereo | 5 | 100 / 0 / 0 | 55 / 3 / 2 | 95 / 2 / 2 | 44 / 2 / 2 | 16 / 7 / 2 | 1 / 5 / 4 |
| AnyLoc (DINOv2-G VLAD) | orb | gt | 1 | 100 / 0 / 0 | 17 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| AnyLoc (DINOv2-G VLAD) | orb | gt | 5 | 100 / 0 / 0 | 17 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| AnyLoc (DINOv2-G VLAD) | orb | stereo | 1 | 100 / 0 / 0 | 17 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| AnyLoc (DINOv2-G VLAD) | orb | stereo | 5 | 100 / 0 / 0 | 17 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| MixVPR (R50, 4096) | sp_lg | gt | 1 | 100 / 0 / 0 | 35 / 3 / 2 | 57 / 1 / 1 | 15 / 1 / 1 | 6 / 2 / 1 | 1 / 3 / 2 |
| MixVPR (R50, 4096) | sp_lg | gt | 5 | 100 / 0 / 0 | 49 / 5 / 5 | 79 / 4 / 4 | 27 / 3 / 3 | 19 / 7 / 4 | 7 / 7 / 6 |
| MixVPR (R50, 4096) | sp_lg | stereo | 1 | 100 / 0 / 0 | 35 / 2 / 2 | 56 / 2 / 1 | 15 / 1 / 1 | 6 / 1 / 0 | 1 / 1 / 1 |
| MixVPR (R50, 4096) | sp_lg | stereo | 5 | 100 / 0 / 0 | 49 / 5 / 4 | 78 / 5 / 4 | 27 / 2 / 2 | 19 / 4 / 2 | 7 / 6 / 5 |
| MixVPR (R50, 4096) | orb | gt | 1 | 100 / 0 / 0 | 17 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| MixVPR (R50, 4096) | orb | gt | 5 | 100 / 0 / 0 | 17 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| MixVPR (R50, 4096) | orb | stereo | 1 | 100 / 0 / 0 | 16 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| MixVPR (R50, 4096) | orb | stereo | 5 | 100 / 0 / 0 | 17 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| CosPlace (R50, 2048) | sp_lg | gt | 1 | 100 / 0 / 0 | 19 / 2 / 2 | 3 / 4 / 4 | 2 / 2 / 2 | 2 / 1 / 1 | 0 / 1 / 1 |
| CosPlace (R50, 2048) | sp_lg | gt | 5 | 100 / 0 / 0 | 24 / 6 / 5 | 18 / 14 / 14 | 5 / 6 / 6 | 6 / 2 / 2 | 1 / 4 / 4 |
| CosPlace (R50, 2048) | sp_lg | stereo | 1 | 100 / 0 / 0 | 19 / 2 / 2 | 3 / 4 / 4 | 2 / 2 / 2 | 2 / 1 / 1 | 0 / 1 / 1 |
| CosPlace (R50, 2048) | sp_lg | stereo | 5 | 100 / 0 / 0 | 23 / 5 / 5 | 17 / 13 / 12 | 5 / 7 / 6 | 5 / 3 / 1 | 1 / 4 / 4 |
| CosPlace (R50, 2048) | orb | gt | 1 | 100 / 0 / 0 | 16 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| CosPlace (R50, 2048) | orb | gt | 5 | 100 / 0 / 0 | 17 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| CosPlace (R50, 2048) | orb | stereo | 1 | 100 / 0 / 0 | 16 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| CosPlace (R50, 2048) | orb | stereo | 5 | 100 / 0 / 0 | 16 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| oracle (GT nearest) | sp_lg | gt | 1 | 100 / 0 / 0 | 83 / 3 / 0 | 100 / 0 / 0 | 96 / 1 / 0 | 56 / 11 / 1 | 56 / 8 / 0 |
| oracle (GT nearest) | sp_lg | gt | 5 | 100 / 0 / 0 | 87 / 4 / 1 | 100 / 0 / 0 | 98 / 2 / 1 | 66 / 10 / 2 | 64 / 9 / 1 |
| oracle (GT nearest) | sp_lg | stereo | 1 | 100 / 0 / 0 | 83 / 3 / 0 | 100 / 0 / 0 | 96 / 1 / 0 | 55 / 9 / 1 | 56 / 7 / 0 |
| oracle (GT nearest) | sp_lg | stereo | 5 | 100 / 0 / 0 | 87 / 4 / 0 | 100 / 0 / 0 | 98 / 1 / 0 | 65 / 10 / 1 | 60 / 11 / 0 |
| oracle (GT nearest) | orb | gt | 1 | 100 / 0 / 0 | 17 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| oracle (GT nearest) | orb | gt | 5 | 100 / 0 / 0 | 17 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| oracle (GT nearest) | orb | stereo | 1 | 100 / 0 / 0 | 17 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| oracle (GT nearest) | orb | stereo | 5 | 100 / 0 / 0 | 17 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

queries per pair: nominal→nominal 360, nominal→low_sun 360, nominal→dim 360, nominal→glare 360, low_sun→nominal 360, low_sun→low_sun 360, low_sun→dim 360, low_sun→glare 360, dim→nominal 360, dim→low_sun 360, dim→dim 360, dim→glare 360, glare→nominal 360, glare→low_sun 360, glare→dim 360, glare→glare 360

Full 4×4 for the main configuration (SP+LightGlue verification, top-5):

**Sim 4×4, SuperPoint+LightGlue, top-5, stereo depth**: success % / FP % / gross FP %

| retrieval | map \ query | nominal | low_sun | dim | glare |
|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | nominal | 100 / 0 / 0 | 15 / 18 / 16 | 87 / 13 / 12 | 3 / 5 / 4 |
|  | low_sun | 14 / 21 / 20 | 100 / 0 / 0 | 15 / 9 / 9 | 9 / 3 / 1 |
|  | dim | 83 / 12 / 12 | 14 / 11 / 10 | 100 / 0 / 0 | 4 / 1 / 1 |
|  | glare | 4 / 2 / 2 | 8 / 3 / 2 | 2 / 3 / 2 | 100 / 0 / 0 |
| AnyLoc (DINOv2-G VLAD) | nominal | 100 / 0 / 0 | 95 / 2 / 2 | 100 / 0 / 0 | 16 / 7 / 2 |
|  | low_sun | 91 / 4 / 4 | 100 / 0 / 0 | 91 / 3 / 2 | 44 / 2 / 2 |
|  | dim | 100 / 0 / 0 | 97 / 0 / 0 | 100 / 0 / 0 | 21 / 5 / 1 |
|  | glare | 1 / 5 / 4 | 4 / 6 / 6 | 0 / 4 / 4 | 100 / 0 / 0 |
| MixVPR (R50, 4096) | nominal | 100 / 0 / 0 | 78 / 5 / 4 | 100 / 0 / 0 | 19 / 4 / 2 |
|  | low_sun | 68 / 13 / 13 | 100 / 0 / 0 | 65 / 9 / 9 | 27 / 2 / 2 |
|  | dim | 100 / 0 / 0 | 81 / 2 / 2 | 100 / 0 / 0 | 14 / 4 / 2 |
|  | glare | 7 / 6 / 5 | 24 / 4 / 4 | 1 / 6 / 5 | 100 / 0 / 0 |
| CosPlace (R50, 2048) | nominal | 100 / 0 / 0 | 17 / 13 / 12 | 99 / 1 / 1 | 5 / 3 / 1 |
|  | low_sun | 17 / 9 / 9 | 100 / 0 / 0 | 18 / 5 / 5 | 5 / 7 / 6 |
|  | dim | 99 / 1 / 1 | 16 / 7 / 6 | 100 / 0 / 0 | 3 / 3 / 0 |
|  | glare | 1 / 4 / 4 | 1 / 8 / 8 | 2 / 5 / 4 | 100 / 0 / 0 |
| oracle (GT nearest) | nominal | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 65 / 10 / 1 |
|  | low_sun | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 98 / 1 / 0 |
|  | dim | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 60 / 13 / 1 |
|  | glare | 60 / 11 / 0 | 97 / 1 / 0 | 58 / 9 / 1 | 100 / 0 / 0 |

<details><summary>Same with GT depth (nearly identical)</summary>

**Sim 4×4, SuperPoint+LightGlue, top-5, gt depth**: success % / FP % / gross FP %

| retrieval | map \ query | nominal | low_sun | dim | glare |
|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | nominal | 100 / 0 / 0 | 16 / 18 / 17 | 88 / 12 / 12 | 2 / 7 / 5 |
|  | low_sun | 15 / 21 / 20 | 100 / 0 / 0 | 15 / 10 / 10 | 10 / 2 / 1 |
|  | dim | 83 / 12 / 12 | 15 / 11 / 11 | 100 / 0 / 0 | 4 / 3 / 2 |
|  | glare | 5 / 2 / 2 | 9 / 2 / 2 | 2 / 3 / 2 | 100 / 0 / 0 |
| AnyLoc (DINOv2-G VLAD) | nominal | 100 / 0 / 0 | 95 / 2 / 2 | 100 / 0 / 0 | 18 / 7 / 3 |
|  | low_sun | 91 / 4 / 4 | 100 / 0 / 0 | 91 / 2 / 2 | 43 / 4 / 3 |
|  | dim | 100 / 0 / 0 | 97 / 0 / 0 | 100 / 0 / 0 | 22 / 7 / 4 |
|  | glare | 1 / 6 / 4 | 4 / 8 / 8 | 0 / 4 / 4 | 100 / 0 / 0 |
| MixVPR (R50, 4096) | nominal | 100 / 0 / 0 | 79 / 4 / 4 | 100 / 0 / 0 | 19 / 7 / 4 |
|  | low_sun | 68 / 13 / 13 | 100 / 0 / 0 | 65 / 10 / 10 | 27 / 3 / 3 |
|  | dim | 100 / 0 / 0 | 81 / 2 / 2 | 100 / 0 / 0 | 15 / 4 / 3 |
|  | glare | 7 / 7 / 6 | 24 / 6 / 6 | 1 / 7 / 7 | 100 / 0 / 0 |
| CosPlace (R50, 2048) | nominal | 100 / 0 / 0 | 18 / 14 / 14 | 99 / 1 / 1 | 6 / 2 / 2 |
|  | low_sun | 17 / 10 / 10 | 100 / 0 / 0 | 19 / 5 / 5 | 5 / 6 / 6 |
|  | dim | 99 / 1 / 1 | 16 / 7 / 7 | 100 / 0 / 0 | 6 / 1 / 0 |
|  | glare | 1 / 4 / 4 | 1 / 10 / 10 | 2 / 5 / 5 | 100 / 0 / 0 |
| oracle (GT nearest) | nominal | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 66 / 10 / 2 |
|  | low_sun | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 98 / 2 / 1 |
|  | dim | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 62 / 11 / 2 |
|  | glare | 64 / 9 / 1 | 98 / 0 / 0 | 59 / 10 / 1 | 100 / 0 / 0 |

</details>

**real end-to-end relocalization**: success % / FP % / gross FP % of all queries (success: error < 0.1 m and < 2.0°; localized = best of top-K by PnP inliers, ≥ 20 inliers; FP = localized but outside the success threshold; gross FP = error > 0.5 m or > 10°, i.e. wrong place)

| retrieval | verifier | 3D | K | seq01→seq11 | seq11→seq01 | seq01→seq02 | seq02→seq01 | seq11→seq11 | seq01→seq01 |
|---|---|---|---|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | sp_lg | stereo | 1 | 1 / 3 / 0 | 1 / 3 / 0 | 34 / 27 / 10 | 35 / 28 / 8 | 98 / 2 / 0 | 91 / 6 / 0 |
| DBoW2 (ORB, ORBvoc) | sp_lg | stereo | 5 | 5 / 13 / 3 | 5 / 12 / 3 | 63 / 32 / 8 | 63 / 32 / 7 | 98 / 2 / 0 | 94 / 6 / 0 |
| DBoW2 (ORB, ORBvoc) | orb | stereo | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 6 / 2 / 0 | 6 / 2 / 0 | 97 / 2 / 0 | 88 / 4 / 0 |
| DBoW2 (ORB, ORBvoc) | orb | stereo | 5 | 0 / 0 / 0 | 0 / 0 / 0 | 11 / 3 / 0 | 12 / 3 / 0 | 98 / 2 / 0 | 94 / 5 / 0 |
| AnyLoc (DINOv2-G VLAD) | sp_lg | stereo | 1 | 18 / 17 / 2 | 14 / 12 / 2 | 93 / 7 / 0 | 93 / 7 / 0 | 98 / 2 / 0 | 96 / 4 / 0 |
| AnyLoc (DINOv2-G VLAD) | sp_lg | stereo | 5 | 32 / 20 / 1 | 25 / 20 / 6 | 94 / 6 / 0 | 94 / 6 / 0 | 98 / 2 / 0 | 95 / 5 / 0 |
| AnyLoc (DINOv2-G VLAD) | orb | stereo | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 13 / 2 / 0 | 16 / 4 / 0 | 98 / 2 / 0 | 95 / 4 / 0 |
| AnyLoc (DINOv2-G VLAD) | orb | stereo | 5 | 0 / 0 / 0 | 0 / 0 / 0 | 18 / 3 / 0 | 18 / 4 / 0 | 98 / 2 / 0 | 95 / 5 / 0 |
| MixVPR (R50, 4096) | sp_lg | stereo | 1 | 25 / 17 / 2 | 18 / 17 / 2 | 91 / 9 / 0 | 89 / 11 / 0 | 98 / 2 / 0 | 96 / 4 / 0 |
| MixVPR (R50, 4096) | sp_lg | stereo | 5 | 35 / 18 / 1 | 28 / 23 / 1 | 94 / 6 / 0 | 93 / 7 / 0 | 98 / 2 / 0 | 95 / 5 / 0 |
| MixVPR (R50, 4096) | orb | stereo | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 13 / 2 / 0 | 16 / 3 / 0 | 98 / 2 / 0 | 94 / 4 / 0 |
| MixVPR (R50, 4096) | orb | stereo | 5 | 0 / 0 / 0 | 0 / 0 / 0 | 18 / 3 / 0 | 18 / 5 / 0 | 98 / 2 / 0 | 95 / 5 / 0 |
| CosPlace (R50, 2048) | sp_lg | stereo | 1 | 8 / 12 / 2 | 9 / 13 / 4 | 88 / 11 / 0 | 79 / 19 / 3 | 98 / 2 / 0 | 95 / 5 / 0 |
| CosPlace (R50, 2048) | sp_lg | stereo | 5 | 20 / 21 / 3 | 16 / 20 / 7 | 92 / 8 / 0 | 89 / 11 / 1 | 98 / 2 / 0 | 95 / 5 / 0 |
| CosPlace (R50, 2048) | orb | stereo | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 13 / 3 / 0 | 13 / 3 / 0 | 97 / 2 / 0 | 93 / 4 / 0 |
| CosPlace (R50, 2048) | orb | stereo | 5 | 0 / 0 / 0 | 0 / 0 / 0 | 16 / 3 / 0 | 16 / 4 / 0 | 98 / 2 / 0 | 95 / 5 / 0 |
| oracle (GT nearest) | sp_lg | stereo | 1 | 38 / 18 / 0 | 39 / 15 / 1 | 91 / 9 / 0 | 92 / 8 / 0 | 98 / 2 / 0 | 95 / 5 / 0 |
| oracle (GT nearest) | sp_lg | stereo | 5 | 43 / 23 / 0 | 41 / 21 / 1 | 94 / 6 / 0 | 93 / 7 / 0 | 98 / 2 / 0 | 95 / 5 / 0 |
| oracle (GT nearest) | orb | stereo | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 15 / 3 / 0 | 15 / 3 / 0 | 97 / 3 / 0 | 95 / 5 / 0 |
| oracle (GT nearest) | orb | stereo | 5 | 0 / 0 / 0 | 0 / 0 / 0 | 18 / 3 / 0 | 18 / 5 / 0 | 98 / 2 / 0 | 95 / 5 / 0 |

queries per pair: seq01→seq11 871, seq11→seq01 825, seq01→seq02 840, seq02→seq01 825, seq11→seq11 871, seq01→seq01 825

### 4. False positives

sim: localized queries split by whether the chosen database frame is a GT positive (K=5, SP+LG; cross-session pairs pooled)

| retrieval | 3D | localized % | chosen frame positive % (of localized) | success % (of localized, positive) | gross FP % (of localized, not positive) | median err of localized m / ° |
|---|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | gt | 31 | 67 | 98 | 79 | 0.018 / 0.13 |
| DBoW2 (ORB, ORBvoc) | stereo | 30 | 68 | 98 | 79 | 0.024 / 0.18 |
| AnyLoc (DINOv2-G VLAD) | gt | 59 | 95 | 99 | 93 | 0.016 / 0.12 |
| AnyLoc (DINOv2-G VLAD) | stereo | 58 | 96 | 99 | 91 | 0.018 / 0.13 |
| MixVPR (R50, 4096) | gt | 54 | 90 | 99 | 91 | 0.017 / 0.12 |
| MixVPR (R50, 4096) | stereo | 53 | 92 | 99 | 92 | 0.017 / 0.13 |
| CosPlace (R50, 2048) | gt | 30 | 79 | 99 | 82 | 0.008 / 0.07 |
| CosPlace (R50, 2048) | stereo | 29 | 80 | 99 | 81 | 0.009 / 0.08 |
| oracle (GT nearest) | gt | 91 | 100 | 96 | – | 0.030 / 0.18 |
| oracle (GT nearest) | stereo | 90 | 100 | 96 | – | 0.031 / 0.19 |

real: localized queries split by whether the chosen database frame is a GT positive (K=5, SP+LG; cross-session pairs pooled)

| retrieval | 3D | localized % | chosen frame positive % (of localized) | success % (of localized, positive) | gross FP % (of localized, not positive) | median err of localized m / ° |
|---|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | stereo | 56 | 60 | 88 | 23 | 0.037 / 1.27 |
| AnyLoc (DINOv2-G VLAD) | stereo | 74 | 93 | 87 | 29 | 0.019 / 0.81 |
| MixVPR (R50, 4096) | stereo | 76 | 93 | 87 | 10 | 0.019 / 0.81 |
| CosPlace (R50, 2048) | stereo | 69 | 87 | 87 | 26 | 0.019 / 0.84 |
| oracle (GT nearest) | stereo | 82 | 100 | 83 | 0 | 0.021 / 0.83 |

sim: inlier gate on the best of top-5 (SP+LG, cross-session pairs pooled): success % / gross FP % of all queries

| retrieval | 3D | ≥ 20 inliers | ≥ 50 inliers | ≥ 100 inliers | ≥ 200 inliers |
|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | gt | 22 / 8.0 | 21 / 3.3 | 20 / 1.9 | 17 / 1.4 |
| DBoW2 (ORB, ORBvoc) | stereo | 22 / 7.6 | 21 / 3.0 | 19 / 1.8 | 16 / 1.3 |
| AnyLoc (DINOv2-G VLAD) | gt | 55 / 3.0 | 54 / 0.3 | 52 / 0.0 | 49 / 0.0 |
| AnyLoc (DINOv2-G VLAD) | stereo | 55 / 2.4 | 54 / 0.2 | 52 / 0.0 | 47 / 0.0 |
| MixVPR (R50, 4096) | gt | 49 / 4.9 | 48 / 0.8 | 46 / 0.0 | 42 / 0.0 |
| MixVPR (R50, 4096) | stereo | 49 / 4.1 | 48 / 0.7 | 45 / 0.0 | 41 / 0.0 |
| CosPlace (R50, 2048) | gt | 24 / 5.2 | 24 / 0.8 | 23 / 0.1 | 21 / 0.1 |
| CosPlace (R50, 2048) | stereo | 23 / 4.6 | 23 / 0.6 | 22 / 0.1 | 21 / 0.0 |
| oracle (GT nearest) | gt | 87 / 0.5 | 84 / 0.1 | 73 / 0.1 | 56 / 0.0 |
| oracle (GT nearest) | stereo | 87 / 0.3 | 83 / 0.0 | 71 / 0.0 | 54 / 0.0 |

real: inlier gate on the best of top-5 (SP+LG, cross-session pairs pooled): success % / gross FP % of all queries

| retrieval | 3D | ≥ 20 inliers | ≥ 50 inliers | ≥ 100 inliers | ≥ 200 inliers |
|---|---|---|---|---|---|
| DBoW2 (ORB, ORBvoc) | stereo | 34 / 5.2 | 33 / 1.6 | 32 / 0.1 | 26 / 0.0 |
| AnyLoc (DINOv2-G VLAD) | stereo | 61 / 1.7 | 57 / 0.2 | 52 / 0.1 | 44 / 0.0 |
| MixVPR (R50, 4096) | stereo | 62 / 0.6 | 59 / 0.0 | 54 / 0.0 | 44 / 0.0 |
| CosPlace (R50, 2048) | stereo | 54 / 2.4 | 52 / 0.4 | 50 / 0.1 | 43 / 0.0 |
| oracle (GT nearest) | stereo | 67 / 0.4 | 62 / 0.1 | 54 / 0.0 | 44 / 0.0 |

The real pooled numbers include the easy 01↔02 pairs.

**What the sim gross false positives are.** Almost all of them are verifications against a database
frame at a different place, often with 100–300 SuperPoint inliers. The pose errors cluster at
multiples of 5 m (5, 10, 15, 20, 25 m). The cause is that the sim terrain material (`Basalt`,
`texture_scale 0.2`) **repeats its texture every 5 m**, so two places 5·n m apart can have
pixel-identical ground texture. The texture repetition is a simulator artifact, but it is a fair
stress test: real regolith is also self-similar.

- DBoW2 retrieval proposes such aliased frames often, because it scores local texture words.
- Learned global descriptors also encode layout (rocks, craters, shadows) and propose them less:
  - ≥20 inliers: 2.4–4.9% gross false positives, against 7.6–8.0% for DBoW2.
  - ≥50 inliers: ≤0.8%, against 3.0–3.3%.
  - ≥100 inliers: 0.0%, against 1.8–1.9%.

**What the real false positives are.** They are almost all **inaccurate, not wrong**.

- Oracle localizations have a median error of 2 cm / 0.8°.
- The non-gross false positives are 0.1–0.5 m or 2–10° off. They come from the strict real
  threshold (0.1 m / 2°) combined with SGBM depth on a dark, low-texture floor.
- The real gross false-positive rate is 0.6–1.7% for MixVPR and AnyLoc at ≥20 inliers, and ≤0.2% at
  ≥50.

### 5. Timing and memory

**Timing and memory** (per query frame, batch 1; GPU named per suite)

| suite | GPU | method | extract median ms | retrieval ms/query (db size) | bytes per db frame | GPU peak MB | params M |
|---|---|---|---|---|---|---|---|
| sim | NVIDIA L40S | DBoW2 (ORB, ORBvoc) | 27.4 | 0.31 (360) | 23,495 | 0 | 0 |
| sim | NVIDIA L40S | AnyLoc (DINOv2-G VLAD) | 52.4 | 0.06 (360) | 196,608 | 7385 | 1136 |
| sim | NVIDIA L40S | MixVPR (R50, 4096) | 5.4 | 0.02 (360) | 16,384 | 193 | 11 |
| sim | NVIDIA L40S | CosPlace (R50, 2048) | 5.7 | 0.02 (360) | 8,192 | 276 | 28 |
| real | NVIDIA L40S | DBoW2 (ORB, ORBvoc) | 27.1 | 0.26 (157) | 23,491 | 0 | 0 |
| real | NVIDIA L40S | AnyLoc (DINOv2-G VLAD) | 53.9 | 0.03 (157) | 196,608 | 7107 | 1136 |
| real | NVIDIA L40S | MixVPR (R50, 4096) | 5.4 | 0.01 (157) | 16,384 | 193 | 11 |
| real | NVIDIA L40S | CosPlace (R50, 2048) | 5.7 | 0.01 (157) | 8,192 | 276 | 28 |

**Verification cost** (median ms per call)

| suite | SP extract | LightGlue match | ORB extract | ORB match | PnP (SP) | SGBM stereo |
|---|---|---|---|---|---|---|
| sim | 8.1 | 10.7 | 12.0 | 4.1 | 22.1 | 19.5 |
| real | 7.8 | 12.0 | 11.9 | 4.0 | 7.9 | 30.5 |

- Measured on an NVIDIA L40S at batch size 1, after warm-up. Extraction times are medians.
- DBoW2 runs on the CPU in this pure-numpy implementation. The C++ DBoW2 transform takes a few ms.
- Retrieval is a brute-force dot product over 157–360 database frames, which is negligible.
  A map of 10⁴ keyframes would still be < 1 ms on a GPU.
- "Bytes per db frame":
  - global descriptors: float32 (AnyLoc 49,152 dimensions; MixVPR 4096; CosPlace 2048);
  - DBoW2: inverted-index entries (word id + weight).
- **Verification per candidate:**
  - SP+LightGlue: ≈ 8 ms extraction (query only, once) + 11–12 ms matching + 8–22 ms PnP RANSAC.
  - ORB: 12 + 4 ms, but PnP took ~95 ms in the smoke run because of its low inlier ratio.
- **Per query, MixVPR + top-5 SP+LightGlue costs ≈ 5 + 8 + 5 × (11 + ~15) ≈ 150 ms on an L40S.**
  - Most of that is the 5 match + PnP calls, which can be batched or stopped early once one
    candidate passes.
- **Onboard cost:**
  - AnyLoc ViT-G (1.1 B parameters, 7 GB GPU memory, 52 ms on an L40S) is not realistic.
  - MixVPR (11 M parameters, 193 MB, 5 ms) is a ResNet-50 front-end and fits a Jetson-class
    device.
  - AnyLoc with ViT-S/B, or a PCA-reduced VLAD, would be the next thing to try if AnyLoc's extra
    accuracy is needed.
  - Database size for 10⁴ keyframes:
    - MixVPR: 16 KB per frame, 160 MB total;
    - AnyLoc: 197 KB per frame, ~2 GB total, so PCA to 512–1024 dimensions would be needed.

## Caveats

- **DBoW2 baseline.** The candidate logic is simplified: plain L1 ranking, without covisibility
  grouping or the common-words filter. Keypoints come from OpenCV ORB with a grid distribution,
  not ORB-SLAM3's octree extractor. The shared-words diagnostics show the failure is in the
  quantization itself, so neither simplification changes the conclusion.
- **AnyLoc's vocabulary is fit on the map session's own database frames.** This is legitimate,
  because the map is known before the query, but it differs from the paper's domain vocabularies.
  ViT-G is used, not ViT-L.
- **MixVPR is trained on GSV-Cities (urban street view)** and resizes 640×360 to 320×320, which
  distorts the aspect ratio. Nothing was fine-tuned.
- **Sim queries fly the identical trajectory as the map**, offset by ~0.27 m. The viewpoint change
  is small, so cross-lighting is the only real difficulty.
  - The oracle rates here (nominal→low_sun 100%, nominal→glare 65%) are higher than the
    feature-level study's (65% / 26%). That study averaged over query offsets up to 30 frames
    (~4 m).
- **The sim texture repeats every 5 m** (above). This inflates false positives for any
  texture-based method.
- **Real sequences drive nearly the same path.** The "map" is a set of database frames with GT
  poses and stereo depth, not a SLAM map. A real system adds its own map error.
  - The 0.1 m / 2° success threshold is strict relative to SGBM depth quality, so many correct-place
    localizations count as false positives (inaccurate). Read the gross column for wrong-place
    risk.
- The **ORB verifier** uses ratio + mutual matching, not ORB-SLAM3's BoW-guided `SearchByBoW`.
  It fails even with the oracle frame, which matches the feature-level study.
- K=5 verification takes the candidate with the most inliers. There is no pose-consistency check
  across candidates or over time.

## Interpretation and recommendation for map reuse

1. **System-level relocalization fails at retrieval first.** Across sun-angle changes the true map
   frame is in DBoW2's top-5 for only ~10% of queries (sim, excluding dim). Even when it is, ORB
   matching then fails. Both stages have to be replaced; fixing only one gains almost nothing:
   - ORB verification gets 0% even with oracle retrieval;
   - DBoW2 + SP+LightGlue gets 15% on nominal→low_sun.
2. **Learned retrieval + SP/LightGlue + PnP works across moderate sun-angle changes.**
   - Sim nominal/dim↔low_sun: 91–97% (AnyLoc) and 65–81% (MixVPR) at top-5, near the oracle's 100%.
   - Real 01↔11: 25–35% against an oracle of 41–43%. Under real lighting changes, verification
     (local features on a dark floor plus stereo depth) becomes the main limit.
3. **Shadow inversion (glare, sun azimuth flipped) is not solved by any off-the-shelf global
   descriptor.** If the mission allows very different sun azimuths, plan to:
   - keep multiple map sessions (re-map at a few sun azimuths), or
   - use a pose prior from the lander's navigation (IMU/altimeter/coarse terrain-relative
     navigation) to restrict candidates spatially. Verification still succeeds 58–98% of the time
     when the right frame is proposed.
4. **Recommended integration path** in ORB-SLAM3 / SP_SLAM3:
   - **Index:** replace `KeyFrameDatabase::DetectRelocalizationCandidates`, and
     `DetectNBestCandidates` for merging and loop closure, with a global-descriptor index.
     - Compute **MixVPR** for each new keyframe in LocalMapping: 5 ms, 16 KB. It is the best
       onboard trade-off.
     - Use AnyLoc (ViT-B/L + PCA) if accuracy is worth the compute.
     - Store the descriptor in the atlas.
   - **Retrieval:** take the top-5 by cosine similarity, optionally restricted by a pose prior.
   - **Verification:** match each candidate with **SuperPoint+LightGlue** against the candidate
     keyframe's map points. SP_SLAM3 already has SP descriptors on its map points.
     - Use PnP RANSAC, then the existing `PoseOptimization` and `SearchByProjection` refinement.
     - Accept with **≥50 inliers**. That matches ORB-SLAM3's own threshold, and in this benchmark
       it cuts gross false positives to ≤0.8% at a 1–4 point success cost.
     - For a lander, also require **temporal consistency**: two or three consecutive frames that
       relocalize to a consistent pose before trusting the map.
   - **Map content:** keep the ORB/SP map content for tracking. Only the place-recognition and
     verification front-end changes, which is a contained modification of `Tracking::Relocalization`
     and `LoopClosing::NewDetectCommonRegions`.

## All tables

`scripts/vpr_summary.py` regenerates these. The complete end-to-end tables (K=1 and K=5, both
verifiers, GT and stereo depth) are above. The remaining raw rows are in `e2e.csv`.
