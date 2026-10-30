# Research summary and handoff

Last updated 2026-10-30 (commit `40ac1d8`). This is the entry point for anyone, human or agent, picking
up this work. It states the research question, the argument the results support so far, the evidence
for each claim (with links), what is open, and where everything lives. Details and caveats are in the
linked result pages; this page should stay short and current.

## Goal

A drone (or rover) that builds **accurate, reusable terrain maps for lunar landers**: a map built in one
session must be usable for localization in a later session, under a different sun angle, with a known
and low false-localization rate. This repository is the benchmark used to find out what breaks and
what fixes it, on real lunar-analogue data (LunaPolaris, LunaLab Luxembourg) and simulation (OmniLRS
in Isaac Sim, with controlled lighting).

## The argument (one paragraph)

Single-session stereo visual SLAM already works on lunar-analogue terrain, and learned features do not
improve it. The real problem is **map reuse under changing illumination**: reuse works under the same
lighting but fails completely (0 %) when the sun angle changes. The failure lies almost entirely in
**place recognition** (bag-of-words retrieval), not in feature matching or pose estimation. Swapping in
**learned global retrieval (AnyLoc/MixVPR) plus learned matching (SuperPoint + LightGlue)** recovers most
of it (sim 15 % → 78–95 % end to end). What remains (glare, the hardest real pair) motivates a
lighting-independent geometric channel. LiDAR–IMU alone, however, **fails silently on open terrain
without vertical structure**. The two modalities fail in complementary ways, which is the case for
camera–LiDAR–IMU fusion with **cross-modal verification**.

## Claims and evidence

| # | Claim | Key numbers | Evidence |
|---|---|---|---|
| 1 | Single-session stereo SLAM is accurate on LunaPolaris | ORB-SLAM3 ATE 3–8 cm, 25 ms/frame; RTAB-Map 6–8 cm; FAST-LIO2 5–7 cm | [milestone1](results/milestone1.md), [compute_comparison](results/compute_comparison.md), [fastlio_lunapolaris](results/fastlio_lunapolaris.md) |
| 2 | Learned features do not improve single-session tracking | SP_SLAM3 5 / 9 cm vs ORB-SLAM3 3 / 8 cm (seq01/11), 4–5× slower. LAC (SuperPoint+LightGlue+GTSAM) 0.16 m after the keyframe-VO fix; 1.2–1.8 m with ORB | [lac_lunapolaris](results/lac_lunapolaris.md) |
| 3 | The sim agrees, for an aerial survey under four lightings | ORB-SLAM3 best in all four (0.13–0.15 m); no run lost tracking | [lunaryard_aerial_lighting](results/lunaryard_aerial_lighting.md) |
| 4 | Map reuse works under the same lighting | 92–100 % of frames localized at the mapping run's accuracy (sim nominal→nominal/dim, seq01→seq01, seq11→seq11) | [map_reuse](results/map_reuse.md) |
| 5 | Map reuse fails when lighting changes | Sim low_sun/glare: 0 %, both systems and modes. Real seq01↔seq11: 0 % in both directions. seq01→seq02 (similar lighting) works | [map_reuse](results/map_reuse.md) |
| 6 | Feature matching survives the lighting change; ORB does not | Cross-lighting PnP success with GT 3D: ORB 16 %, SP+NN 29 %, **SP+LightGlue 58 %**, ALIKED+LG 43 %, DISK+LG 36 % | [lunaryard_aerial_lighting](results/lunaryard_aerial_lighting.md) (cross-lighting section) |
| 7 | Retrieval is where it breaks | Recall@1 sim nominal→low_sun: DBoW2 3 %, AnyLoc 85 %, MixVPR 56 %. Real 01→11: 3 % vs 37–39 % | [place_recognition](results/place_recognition.md) |
| 8 | Learned retrieval + LightGlue verification fixes most of it | End to end, sim nominal→low_sun: DBoW2 15 %, learned 78–95 %, oracle 100 %. Real 01→11: 5 % vs 32–35 % (oracle 43 %). False positives ≤ 0.8 % at a 50-inlier gate. MixVPR 5 ms / 193 MB GPU per query | [place_recognition](results/place_recognition.md) |
| 9 | LiDAR–IMU alone fails silently on open terrain | Ground-only FAST-LIO2 diverges on every sequence (hundreds of km) and still reports 100 % coverage. Horizontal translation and yaw are unconstrained | [lidar_ground_only](research/lidar_ground_only.md) |

Supporting work (method and diagnosis, not separate claims):
* **Benchmark harness:** dataset/algorithm adapters, offline replay, ATE/RPE with explicit frames and
  alignment, compute cost, LiDAR-referenced map quality, and **map-reuse scoring in a fixed map frame**
  (session B is never re-aligned). See [adapter_interface](adapter_interface.md), [evaluation](evaluation.md).
* **LAC drift diagnosis:** frame-to-frame VO drift fixed by keyframe-referenced VO (1.03 → 0.16 m); stride and
  sim tests rule out slow motion as the cause. See [lac_lunapolaris](results/lac_lunapolaris.md).
* **SmokeRoom CLAHE repeats:** run-to-run variance is real; compare medians of repeats. See
  [smokeroom_clahe_repeats](results/smokeroom_clahe_repeats.md).
* **Lens flare:** no measurable flare-attributable error on seq11 yet. See
  [lens_flare_restoration](research/lens_flare_restoration.md).
* **Videos:** trajectory replays and match visualizations (inliers/outliers) in
  `slam-bench-work/replays/` on scratch (not in git).

## What is open (ordered by how much it strengthens the argument)

1. **Close the loop inside SLAM.** Replace the ORB-SLAM3 / SP_SLAM3 `KeyFrameDatabase` candidate detection
   with a MixVPR index stored in the atlas. Verify the top 5 with SuperPoint+LightGlue+PnP, accept at ≥ 50
   inliers, and require temporal consistency. Rerun the [map_reuse](results/map_reuse.md) matrix. Success
   means low_sun and seq01→seq11 move from 0 % localized, with no rise in false relocalizations. So far,
   claim 8 is shown only offline.
2. **Camera rescue of degenerate LiDAR.** Ground-only LiDAR + stereo, loosely coupled first, with and
   without cross-modal verification, plus an oracle. Add a degeneracy-aware LIO baseline so the camera's
   benefit is not overstated. See "Next experiments" in [lidar_ground_only](research/lidar_ground_only.md).
3. **A fusion baseline adapter:** FAST-LIVO2, LVI-SAM or R3LIVE.
4. **Glare (sun azimuth flipped)** is unsolved by every retrieval method. Verification still works when the right
   frame is proposed (58–98 %), so try a pose prior or multi-session maps (one per lighting regime).
5. **More realistic sim:** a simulated LiDAR on the aerial renders, noise/blur, altitude and heading changes
   between sessions, and a non-repeating ground texture. The current texture repeats every 5 m, which causes
   most sim wrong-place matches.
6. **More real lighting variety:** LunaPolaris seq04–10 and 12 (seq09 has a documented odometry loss at
   the turn toward the light).

Map-quality evaluation needs a reference that is independent of LiDAR once LiDAR becomes an input: use
the sim GT mesh/depth, or mocap.

## Known data quirks (read before trusting a number)

* **LunaPolaris rotation error is about 4.2° for every method:** a constant marker→sensor offset in the GT
  extrinsics. Compare translation.
* **seq11** is a different day: low spotlight plus robot light, more rocks, about 7 % saturated pixels,
  `right_row_shift_px: -1.63`.
* **TUM timestamps around 1.8e9 s lose ns precision.** Match by nearest stamp (≤ 1 ms) or by row index.
* **Sim renders:** identical trajectory across lightings; `depth0` is uint16 mm; the principal point is
  the image centre. Sim queries fly the same path as the map, so sim reuse is optimistic about viewpoint change.
* **Multi-threaded ORB-SLAM3 is not deterministic.** Use repeats (median of 3) for accuracy comparisons.

## Where things live

| What | Where |
|---|---|
| Framework (adapters, evaluation, runner) | `slambench/` |
| Run configs (one YAML per experiment) | `configs/` (`lunapolaris_*`, `omnilrs_*`, `lunapolaris_mapreuse_*`) |
| Estimator builds and patches | `third_party/{orbslam3,spslam3,rtabmap,fastlio,lac}/` |
| Simulation (render, aerial poses, cross-lighting relocalization, videos) | `sim/omnilrs/` |
| Place recognition | `vpr/`, `scripts/vpr_*.py`, `scripts/vpr_benchmark.sbatch` |
| Run outputs | `/storage/ice1/3/6/bkuang7/slam-bench-work/runs/<config name>/<run id>/` (`summary.csv`, `eval/metrics.json`, `algorithm/`) |
| VPR / cross-lighting / replay outputs | `slam-bench-work/{vpr/results,xl_reloc,replays}/` |
| Apptainer images | `slam-bench-work/containers/` (never committed) |

Operational notes for PACE ICE are in [`AGENTS.md`](../AGENTS.md).
