# Lens-flare restoration for LEAD: is fine-tuning worth it, and how would we know?

Date 2026-10-28. Sources: van der Meer PhD thesis (`dave_thesis.pdf`, Ch. 4 §4.6, Ch. 5 §5.3, Ch. 6,
Ch. 7 §7.2), Dave's e-mail, this repository's results (`docs/results/*.md`), and a new headroom probe
(`scripts/flare_headroom_probe.py`, results below).

## Recommendation (short)

1. **Do not start by fine-tuning a restoration network.** On the one flare sequence that we have (seq11),
   none of our five estimators shows a measurable flare-attributable trajectory error (§4). There is
   currently nothing for a restorer to fix, and Dave's own SLAM results were mixed (§2).
2. **First, measure the problem where it actually occurs.** Download LunaPolaris seq04–10 and 12. Dave's
   RTAB-Map **lost odometry in seq09** at the spot turn toward the light, and seq11/12 had their largest local
   error at that same turn (thesis §4.6.1). Those events are the real target, and we have not tested them.
3. **If flare failures appear, add the cheap flare-aware mechanisms before restoration:** a flare mask,
   down-weighting of keypoints, inflation of the visual measurement covariance, and LiDAR–IMU carry-through.
   Test restoration (pretrained, then lunar fine-tuned) as a *comparison arm* against those.
4. **Fine-tune only if restoration beats masking** on correspondence-level and event-level metrics. When
   you do, improve on Dave's recipe in four ways: camera-matched flare, stereo/temporal consistency,
   a "do no harm" constraint, and a SLAM-relevant loss (§6, Stage 4).
5. **Success means event-level localization and correspondence quality on real flare sequences, with no
   regression on clean ones.** PSNR/SSIM and full-trajectory ATE are not success criteria (§5).

---

## 1. What Dave did

**Pipeline** (thesis §6.3, pp. 191–201):
1. Reproduced **FlareX** without changing the architecture. He trained it from scratch because no weights
   were public. He used the original recipe: 9,500 2D flare templates plus 3,000 Blender-rendered pairs on
   ~24k Flickr backgrounds, 325k iterations over 48 epochs, with DPT depth awareness on.
2. **Lunar fine-tuning** by continued training on 7,695 clean backgrounds:

   | source | images |
   |---|---:|
   | Apollo 11–17 (images with flare removed) | 2,483 |
   | Chang'e-6 / RCLLD | 1,212 |
   | BIG-LunaShot (Blender, 2 scenes) | 4,000 |

   The training input is `Composite(I_clean, F_FlareX)` and the target is `I_clean`. BIG-LunaShot's own
   compositor flares were dropped as unrealistic. Only its clean renders were used.
3. **Downstream test** (§6.4–6.5): RTAB-Map stereo on LunaPolaris seq04–12, with **loop closure off**.
   Five features were tested (SIFT, ORB, ORB-Octree, BRISK, SuperPoint), each under three variants
   (original / pretrained / Moon), 5 repeats each. The metric was ATE with full SE(3) alignment.

**The e-mail adds:** no architecture change. "Trained from scratch" refers to the general model and
"fine-tuned" to the lunar stage. Pretrained weights suffice unless Moon-specific robustness is needed.
The default network ignores stereo/temporal relations, and the lunar training data can be shared.

## 2. What his results show, and what they don't

| observation (thesis) | implication |
|---|---|
| Table 6.2: ATE change is between −5.4 % and +3.9 %. The sign depends on the feature and flips between the pretrained and Moon models for ORB, BRISK and SuperPoint. | No consistent effect. With 5 repeats, no confidence intervals and sequence variance that dominates the spread (Fig. 6.18), none of these deltas can be shown to be non-zero. |
| Baseline ATE is 0.75–2.5 m on a ~30 m path (Fig. 6.17). | The baseline is pure VO (loop closure off) and is ~10–30× worse than our tuned stacks on the same data. Our seq11 results: ORB-SLAM3 0.081 m, RTAB-Map 0.082 m. In our repo, RTAB-Map's default `Stereo/OpticalFlow=true` alone gave seq01 2.65 m against 0.072 m with it off (commit `9d7d8da`). *Hypothesis:* his deltas sit on a baseline dominated by non-flare error. Ask him for the config. |
| Fig. 6.23/6.24: stereo correspondences drop when flare appears, and **after restoration the count "remains nearly unchanged"**. | The restorer did not recover the one quantity that SLAM consumes. |
| §6.5.3: up to 60 % of keypoints lie near flare regions, and the damage is mostly in the *surroundings* (veiling glare), not the flare core. | The mechanism is contrast loss plus spurious structure. That is where a mask or weighting acts directly. |
| §6.6.2: frame-independent restoration flickers, so tracking becomes unstable. | This confirms the e-mail. It is a structural limitation, not a data limitation. |
| §4.6.1: odometry loss in seq09 at the spot turn toward the light, and the largest local error (up to 1.05 m) in seq11/12 at the same turn. | Flare damage is **event-localized**. Full-trajectory ATE dilutes it, so the evaluation must be stratified by event. |

His conclusion and future work (§6.6.2, §7.2) already point to multi-view/temporally consistent models
and to **"incorporating detected flare regions ... enabling adaptive weighting of keypoints"**.

## 3. First principles: what flare does to a stereo-SLAM measurement

A flare-corrupted pixel is roughly `I = clip(g · (S + F_veil + F_ghost) + n)`, where:
- `S` is the scene;
- `F_veil` is low-frequency veiling glare;
- `F_ghost` are reflections such as ghosts and orbs;
- `g` is the exposure gain;
- `clip` is sensor saturation.

Each term breaks SLAM in a different way, and each calls for a different remedy:

| term | effect on SLAM | can restoration fix it? | cheaper alternative |
|---|---|---|---|
| `F_veil` (additive, smooth) | Contrast near the source drops, corners fall under FAST/SuperPoint thresholds, and descriptors change. We measure the 5th-percentile grey on seq11 rising from 28 to 66 under flare. | **Partly.** Signal is still present above noise, so removing the veil can recover contrast. This is the one place restoration can *add* usable correspondences. | Local contrast normalization (CLAHE; in an earlier smoke-filled indoor test it helped before tracking failure but hurt map continuity). |
| `F_ghost` (structured, moves with the source, not the scene) | False keypoints that violate epipolar and static-world assumptions, and outliers. SuperPoint fires on them (Fig. 6.22). | Yes, but they are mostly already rejected by RANSAC. | **Mask or down-weight.** The effect is the same, with zero hallucination risk. |
| `clip` (saturation, blooming) | The information is destroyed. | **No.** Any texture painted there is hallucinated. If it differs between left/right or t/t+1, it creates *confident wrong* correspondences, which is worse than none. | Mask. |
| `g` (auto-exposure reacts to the source) | The rest of the frame darkens and shadow texture is lost. | Not addressed by flare removal. | Exposure control, or HDR. |

**Consequence.** Restoration cannot add information about the scene. It can only remove nuisance
structure (which masking also does) and re-map intensity (which contrast normalization partly does). Its
expected value over **mask + contrast normalization** is therefore limited to recovering contrast under the
veil. Its distinctive risk is hallucinated texture that is inconsistent across views. Hence the ordering:
mask/weight first, restoration only as the challenger, and an explicit test for hallucination.

## 4. Our evidence (new probe, 2026-10-28)

**Method** (`scripts/flare_headroom_probe.py`):
- **Flare-in-view score.** Saturated blobs (≥ 300 px) are kept if they have a *bright and smooth* 30 px
  halo. Saturation alone is useless: sunlit rocks and regolith in seq11 are 9–13 % saturated without any
  flare. The score was checked by eye on montages. Classes: none < 0.17 ≤ moderate < 0.40 ≤ strong. It
  labels **0 % of seq01/02** as flare (matching Dave: seq01–03 have no flare) and **43 % of seq11**
  (27 % moderate, 16 % strong; three long episodes of 52 s, 140 s and 189 s).
- **Correspondences.** Every 3rd frame: ORB row-constrained stereo matches, and temporal RANSAC inliers
  (t → t+3).
- **Local drift.** The existing runs' online trajectories (evaluator `local_drift`), 1-s window,
  |Δest − Δgt|.

**Correspondences, seq11:**

| flare | frames | 5th-pct grey | ORB stereo matches median / p5 | temporal inliers median / p5 |
|---|---|---|---|---|
| none | 2536 | 28 | 532 / 370 | 577 / 458 |
| moderate | 1205 | 66 | 340 / 246 | 502 / 415 |
| strong | 720 | 66 | 310 / 231 | 524 / 398 |

(For reference, seq01 has a median of 279 and seq02 309. Both have no flare.)

**1-s local drift, seq11, mm (RMS / p99):**

| estimator | runs | none | moderate | strong | strong share of worst 1 % (base 16 %) |
|---|---|---|---|---|---|
| ORB-SLAM3 (lean) | 3 | 3.4 / 9.6 | 2.6 / 6.1 | 3.7 / **13.6** | **44 %** |
| SP_SLAM3 | 1 | 3.8 / 10.6 | 2.7 / 6.2 | 3.0 / 7.2 | 2 % |
| RTAB-Map (lean) | 3 | 5.5 / 5.9 | 2.3 / 5.3 | 2.0 / 4.3 | 2 % |
| LAC-kfvo | 1 | 11.2 / 39.2 | 10.3 / 33.5 | 6.6 / 19.6 | 0 % |
| FAST-LIO2 (LiDAR control) | 1 | 12.9 / 47.7 | 10.9 / 32.6 | 9.9 / 24.2 | – |

**Findings:**
- **Flare removes about 40 % of stereo matches** and lifts the black level, consistent with Dave. But ≥ 230
  matches remain at the 5th percentile, and every estimator tracked 100 % of seq11 with zero loss events.
- **No estimator has higher typical local drift under flare.** The LiDAR control, which cannot see flare,
  shows the same "clean windows are harder" pattern. So the clean/flare difference is trajectory content
  (turns, dark shadow periods), and any flare effect is below that confound.
- **The only flare signal is ORB-SLAM3's tail.** Strong flare is 3× over-represented in its worst 1 % of
  windows (p99 13.6 vs 9.6 mm). This is a real but cm-scale effect.
- **Oracle headroom on seq11 ≈ 0.** Even a perfect restorer cannot measurably improve ATE or RPE here. ATE is
  already 6–8 cm, near the GT extrinsic floor (4.3° marker offset; `docs/results/fastlio_lunapolaris.md`).
- **The failures we have actually seen are not flare.** Map reuse seq01↔seq11 localizes 0 %
  (`map_reuse.md`), and sim cross-lighting fails with the **sun never in view** (`lunaryard_aerial_lighting.md`).
  Those failures are shadow/appearance change. Flare removal does not change shadows.

**Limits:**
- Only one flare sequence. seq11 includes the onboard light, which helps.
- The single-run estimators (SP_SLAM3, LAC, FAST-LIO2) have no repeat spread.
- The heuristic flare score is not a FlareX mask.
- The analysis is correlational, not causal. Stage 2 below fixes this with paired sim data.

## 5. Metrics for claiming success

Use a hierarchy of metrics. Lower tiers are diagnostic only. Claims are made at Tiers 2–3, on real flare
sequences, against a tuned baseline.

**Tier 0. Restoration fidelity** (synthetic held-out pairs, plus paired sim from Stage 2).
- PSNR/SSIM, G-PSNR/S-PSNR (Flare7K++), and LPIPS.
- Necessary but **never sufficient**. Report them so the work can be compared with the flare-removal
  literature.

**Tier 1. Correspondence quality** (real data; no clean image or GT pose needed).
- **Stereo:** row-constrained match count and inlier ratio inside and around flare regions.
- **Disparity vs LiDAR:** depth error of matched points against the LiDAR reference
  (`scripts/stereo_depth_bias.py`). This catches hallucinated correspondences.
- **Temporal:** track length, frame-to-frame RANSAC inlier ratio, and **flicker**. Flicker is the photometric
  difference between restored frames t and t+1 after warping with GT pose plus LiDAR or stereo depth,
  measured relative to the same quantity on raw frames.
- **Hallucination rate:** keypoints and matches inside saturated (`clip`) regions after restoration. These
  should not increase.

**Tier 2. Localization** (GT poses, event-stratified, ≥ 5 repeats, paired comparison).
- Tracking-loss events and recovery time (the seq09 case). `n_recovered_in_new_map` and map continuity, as in
  the CLAHE study.
- 1-s and 1-m RPE stratified by flare class (none/moderate/strong), with RMS, **p99 and max**.
- **Event windows:** error over the ±10 s around each spot turn toward the light.
- Full-trajectory ATE, online and final, with loop closure on *and* off. Report it, but treat it as
  secondary.
- **Non-regression:** on seq01/02 and the flare-free windows, RPE/ATE must not get worse (±5 %, inside the
  repeat CI).

**Tier 3. Mission relevance.**
- Map reuse and relocalization in flare conditions: localized %, false-relocalization rate
  (`map_reuse.md` method).
- Compute: ms/frame, GPU memory, and whether it fits the 68 ms frame period.
- For fusion: filter consistency (NEES) and whether visual measurements are down-weighted *when* they are
  bad (precision/recall of the flare mask against Tier-1 outliers).

**Pre-register the success criteria**, for example:
- on flare sequences, **0 tracking-loss events** where the baseline loses, and **≥ 30 % lower p99 1-s RPE**
  in strong-flare windows, with bootstrap 95 % CI excluding 0;
- **no regression** on clean data;
- **no increase** in LiDAR-disagreeing stereo matches;
- real time on the target compute.

## 6. Incremental plan (each stage has a go/no-go)

| stage | work | cost | go criterion for the next stage |
|---|---|---|---|
| **0. Establish the problem** | Download seq04–10 and 12. Run ORB-SLAM3, SP_SLAM3, RTAB-Map and FAST-LIO2 (×3–5). Run the headroom probe on each. Ask Dave for per-frame flare labels and his RTAB-Map config. | days, CPU | A flare-attributable failure exists: loss events, or a strong-flare p99 RPE ≥ 2× clean, on a tuned estimator. **If not, stop.** Report "our stack is flare-robust on LunaPolaris" (itself a result). |
| **1. Cheap mitigations** | (a) Flare mask (pretrained FlareX/Flare7K++ flare layer, or our halo score) used to drop or down-weight keypoints. (b) Veil-aware contrast normalization. (c) **Pretrained** restoration (Dave's advice). Same Tier 1–2 metrics. | 1–2 weeks | Restoration beats masking on Tier 1 (more LiDAR-consistent matches) *and* Tier 2. Otherwise go straight to Stage 5. |
| **2. Paired, causal test data** | OmniLRS Lunalab replay of a seq09/12 trajectory with the projector/sun **in view**, rendered with `--lens-flares on` and `off`. This gives pixel-aligned pairs with exact depth and poses, separates flare from scene content, and is Dave's missing piece. Optionally, a real static paired capture in the lab: the same viewpoint with and without a small occluder over the source. | 1 week | Isaac's flare looks like ZED2 flare (compare the statistics of the 5th-percentile grey and halo profile to real data). |
| **3. Lunar fine-tune** | Get Dave's data, splits, config and checkpoint, and reproduce his checkpoint. Ablate pretrained vs Moon-FT vs Moon-FT + **ZED2-measured flare templates** (point the ZED2 at the lab spotlight in the dark) with ZED2-like exposure and saturation. | weeks, GPU | Moon-FT beats pretrained on Tier 1–2 on held-out real sequences. Not on PSNR. |
| **4. SLAM-consistent restoration** (research contribution) | Stereo-consistent synthesis (the same physical flare, rendered for each camera). Losses: L/R photometric consistency at known disparity, temporal warp consistency, a keypoint-repeatability/descriptor loss against the clean image, an identity loss on flare-free inputs, and editing confined to the predicted flare layer. Alternative: skip pixels and fine-tune SuperPoint/LightGlue with flare augmentation. | months | Beats Stage 1 masking on Tier 2 at real-time cost. |
| **5. Reliability-aware fusion** (main line) | The flare mask and frame quality become per-feature weights and visual covariance inflation in the factor graph. LiDAR–IMU carries through degraded windows. | weeks | NEES consistency, and no loss events on flare sequences. |

Stages 1 and 5 are the default path, in line with LEAD's localization goals. Restoration (Stages 3–4)
continues only if the evidence earns it.

## 7. What to ask Dave for

- The lunar backgrounds, the corruption config and code, the train/val splits, and the Moon checkpoint.
  These let us separate the gain from lunar adaptation from the gain from restoration.
- The RTAB-Map parameters used in Ch. 5–6, especially `Stereo/OpticalFlow`, odometry strategy and feature
  settings. Without them his baseline cannot be compared with ours.
- The per-frame light/strong flare labels behind Fig. 6.23–6.26, and the FlareX flare masks for LunaPolaris.
  These would validate our halo score.
- The frame indices of the seq09 odometry loss.

## 8. Threats to validity

- **Confounding.** Flare co-occurs with spot turns. Use paired sim (Stage 2) and event windows rather than
  raw class averages.
- **Lab vs Moon.** LunaLab has a spotlight at a few metres, walls, and an indoor ZED2. Real solar flare,
  exposure and the absence of atmosphere differ. Claims should be scoped to the analogue.
- **Baseline strength.** Every comparison must use our tuned estimators. A weak baseline makes any
  preprocessing look good, or bad.
- **GT floor.** The 4.3° marker extrinsic and mocap noise limit how small a difference is resolvable.
  Report the repeat spread next to every delta.
