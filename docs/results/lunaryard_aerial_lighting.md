# Aerial Lunaryard lighting sweep: ORB-SLAM3 vs SP_SLAM3 vs LAC (simulated)

This is a stereo SLAM test from a drone-like viewpoint over the OmniLRS Lunaryard 80 m procedural crater
terrain. The camera flies at 5 m AGL and 2 m/s, pitched 60° down, along a 120 s / 240 m lawnmower path with
smooth turns (1800 frames at 15 Hz). There are 4 lighting conditions with the same trajectory, terrain and
rocks. Setup, reproduction and caveats: [`docs/datasets/omnilrs.md`](../datasets/omnilrs.md#aerial-lunaryard-lighting-sweep).
Estimator settings are identical to the real LunaPolaris configs. Configs:
`configs/omnilrs_lunaryard_aerial_<lighting>_{orbslam3_stereo,spslam3_stereo,lac_stereo_kfvo}.yaml`.

| lighting | sun az / el, intensity | appearance | sample left image (`/storage/ice1/3/6/bkuang7/omnilrs/renders/`) |
|---|---|---|---|
| nominal | 180° / 45°, 1750 | even light, short shadows | `lunaryard_aerial_nominal/cam0/1800000008666666667.png` |
| low_sun | 180° / 7°, 1750 | grazing light, black crater interiors, long rock shadows | `lunaryard_aerial_low_sun/cam0/1800000008666666667.png` |
| dim | 180° / 45°, 175 | 10× underexposed (mean grey ≈ 13/255) | `lunaryard_aerial_dim/cam0/1800000008666666667.png` |
| glare | 270° / 10°, 1750 | sun ahead on the +x legs: backlit, shadows toward the camera, specular sheen at the frame top; sun disk never in view | `lunaryard_aerial_glare/cam0/1800000008666666667.png` |

## Results
Mean [min, max] over repeats: ORB-SLAM3 and SP_SLAM3 3 successful runs each, LAC 1 run.
* **ATE**: RMSE of the final trajectory after SE(3) alignment, or after Sim(3) alignment with its scale. A scale
  above 1 means the estimate is too short.
* **RPE 1 s**: translation RPE over 1 s.
* **Frame time**: mean per-frame tracking time.
* **ORB-SLAM3 nominal**: 3 of 4 runs succeeded; 1 run aborted (`Sophus::SO3::exp` NaN, a known ORB-SLAM3
  failure, also seen in `smokeroom_clahe_repeats.md`).

| lighting | estimator | runs ok | ATE SE3 m | ATE Sim3 m | Sim3 scale | RPE 1 s m | coverage | loss events | n_maps | frame time ms |
|---|---|---|---|---|---|---|---|---|---|---|
| nominal | orbslam3 | 3/4 | 0.126 [0.117, 0.137] | 0.113 [0.111, 0.117] | 0.997 [0.996, 0.998] | 0.011 [0.010, 0.012] | 1.00 [1.00, 1.00] | 0 [0, 0] | 1 [1, 1] | 25 [24, 25] |
| nominal | spslam3 | 3/3 | 0.181 [0.135, 0.243] | 0.174 [0.129, 0.230] | 1.000 [0.996, 1.002] | 0.018 [0.018, 0.019] | 1.00 [1.00, 1.00] | 0 [0, 0] | 1 [1, 1] | 156 [155, 157] |
| nominal | lac | 1/1 | 0.589 | 0.380 | 1.024 | 0.055 | 1.00 | 0 | 1 | 61 |
| low_sun | orbslam3 | 3/3 | 0.175 [0.164, 0.192] | 0.124 [0.119, 0.128] | 0.994 [0.993, 0.994] | 0.010 [0.009, 0.010] | 1.00 [1.00, 1.00] | 0 [0, 0] | 1 [1, 1] | 28 [27, 29] |
| low_sun | spslam3 | 3/3 | 0.249 [0.187, 0.284] | 0.171 [0.153, 0.180] | 1.009 [1.006, 1.011] | 0.029 [0.024, 0.032] | 1.00 [1.00, 1.00] | 0 [0, 0] | 1 [1, 1] | 160 [159, 161] |
| low_sun | lac | 1/1 | 0.600 | 0.482 | 1.019 | 0.059 | 1.00 | 0 | 1 | 65 |
| dim | orbslam3 | 3/3 | 0.169 [0.151, 0.179] | 0.139 [0.127, 0.145] | 0.995 [0.995, 0.996] | 0.011 [0.011, 0.012] | 1.00 [1.00, 1.00] | 0 [0, 0] | 1 [1, 1] | 16 [16, 16] |
| dim | spslam3 | 3/3 | 0.199 [0.138, 0.232] | 0.194 [0.138, 0.229] | 1.002 [1.000, 1.004] | 0.022 [0.015, 0.025] | 1.00 [1.00, 1.00] | 0 [0, 0] | 1 [1, 1] | 159 [159, 160] |
| dim | lac | 1/1 | 0.616 | 0.542 | 1.015 | 0.057 | 1.00 | 0 | 1 | 63 |
| glare | orbslam3 | 3/3 | 0.152 [0.127, 0.191] | 0.106 [0.086, 0.135] | 0.994 [0.993, 0.995] | 0.012 [0.012, 0.013] | 1.00 [1.00, 1.00] | 0 [0, 0] | 1 [1, 1] | 28 [27, 30] |
| glare | spslam3 | 3/3 | 0.387 [0.316, 0.469] | 0.196 [0.179, 0.210] | 1.017 [1.013, 1.023] | 0.040 [0.037, 0.046] | 1.00 [1.00, 1.00] | 0 [0, 0] | 1 [1, 1] | 162 [162, 163] |
| glare | lac | 1/1 | 0.691 | 0.589 | 1.019 | 0.059 | 1.00 | 0 | 1 | 68 |

All 28 successful runs tracked every frame (coverage 1.00) in one map, with no tracking-loss events.
5 s displacement ratio (estimate / GT, median; `segscale.py`):

| lighting | ORB-SLAM3 | SP_SLAM3 | LAC |
|---|---|---|---|
| nominal | 1.000 | 0.994 | 0.980 |
| low_sun | 1.001 | 0.986–0.991 | 0.978 |
| dim | 1.001 | 0.994–0.997 | 0.980 |
| glare | 1.002 | 0.977–0.984 | 0.979 |

RPE 1 s split by trajectory part (RMSE, m):

| lighting | estimator | +x legs (into sun for glare) | −x legs | turns |
|---|---|---|---|---|
| glare | ORB-SLAM3 | 0.010 | 0.011 | 0.016 |
| glare | SP_SLAM3 | 0.034 | 0.044 | 0.045 |
| glare | LAC | 0.058 | 0.052 | 0.068 |
| nominal | ORB-SLAM3 | 0.010 | 0.009 | 0.014 |
| nominal | SP_SLAM3 | 0.016 | 0.018 | 0.021 |

## Interpretation
* **ORB-SLAM3 is the most accurate in every condition**, as on the real slow-rover LunaPolaris data. Lighting
  barely affects it:
  * ATE is 0.13–0.18 m over 240 m (< 0.08% of path).
  * RPE 1 s stays at ~1 cm under every lighting.
  * Metric scale is within 0.6%.
* **SP_SLAM3 does not beat ORB under any lighting tested.**
  * Its closest result is `dim`: 0.199 [0.138, 0.232] vs 0.169 [0.151, 0.179]; the ranges overlap.
    * The 10× darker images cost ORB nothing measurable. FAST at threshold 20 finds ~1 corner per frame, but
      the fallback threshold of 7 still yields ~530.
    * Without sensor noise, a dark image is just a lower-contrast one. Real underexposure adds shot and read
      noise and could change this ranking.
  * Its worst result is `glare` (0.387 m; RPE 1 s 4 cm vs 1 cm for ORB). The Sim3 ATE (0.196) is much lower
    than the SE3 ATE, and the 5 s displacement ratio is ~0.98, so much of the error is a ~2% scale shrink.
  * Low sun also gives SP a ~1% scale shrink. Long, hard-edged shadows that move with the viewpoint appear to
    bias SuperPoint+LightGlue stereo depth. ORB stays at scale 1.000 ± 0.002.
  * Within `glare`, the into-sun (+x) legs are not worse than the −x legs for any estimator. The forward-scatter
    sheen is not the problem; the low-sun shadow structure is.
* **LAC (kfvo) is the least accurate everywhere.** ATE is 0.59–0.69 m and its scale is ~2% short (5 s ratio
  0.978–0.980) under all four lightings. This is the stereo depth/scale bias seen on real LunaPolaris
  (`lac_lunapolaris.md`), not a lighting effect.
* **No condition broke tracking.** With fixed exposure, sharp noise-free images, rich texture and 2 m/s at 5 m
  AGL, the aerial motion is still easy for all three. A harder test needs
  sensor noise / motion blur, faster or lower flight, or the sun in view (camera pitched < 30° down).
* **Compute:** ORB-SLAM3 runs at 16–28 ms per frame on 8 CPU cores, real-time with margin. SP_SLAM3 needs
  ~160 ms per frame on a V100, 2.4× slower than 15 Hz; it was throttled, not dropping frames. LAC needs ~60 ms.
