# SmokeRoom: ORB-SLAM3 mono with and without CLAHE (controlled repeats)

Date 2026-10-11. Commit `e36cacf` for every run; 5 repeats each of `subtmrs_smokeroom_orbslam3_mono`
(plain) and `subtmrs_smokeroom_orbslam3_mono_clahe` (CLAHE(3.0, 8x8) inside the estimator image), same
prepared input (`mono_b5ec1bc4056f`, 140-px bottom crop), settings identical except `clahe`,
`playback_rate: 1.0`, PACE `ice-cpu` nodes (all Xeon Gold 6226, 8 cores). Run directories:
`~/scratch/slam-bench-work/runs/clahe_repeats_e36cacf/`. Report, CSVs and plot regenerated with
```
python scripts/compare_variants.py -o <runs>/comparison \
    --variant 'plain=<runs>/subtmrs_smokeroom_orbslam3_mono/*' \
    --variant 'clahe=<runs>/subtmrs_smokeroom_orbslam3_mono_clahe/*'
```
All errors are aligned benchmark accuracy: Sim(3) fitted per coordinate session over the whole
session (see `docs/evaluation.md`). A new monocular map is a new session with its own frame and
scale; its error says nothing about whether it can be related to the earlier map in real time.
Values are median [min–max] over runs.

## Outage and map continuity (268–333 s)
| | plain (n=5) | CLAHE (n=5) |
|---|---|---|
| tracking lost at (s) | 274.2 [274.2–274.2] | 285.0 [279.6–285.0] |
| tracking OK again at (s) | 333.3 [332.8–333.3] | 315.4 [315.4–315.7] |
| outage (s) | 59.1 [58.6–59.1] | 30.6 [30.3–35.8] |
| recovery | new map, 5/5 | new map, 5/5 |
| new map merged back into the original map (~382.5–382.8 s) | 5/5 | 1/5 |
| tracking lost again after recovery (318.5–322.6 s, map 1 → 2) | 0/5 | 2/5 |
| crashed (SIGABRT at ~323 s, NaN in `Sophus::SO3::exp`, new map) | 0/5 | 1/5 |
| frame coverage | 0.584 [0.584–0.585] | 0.640 [0.424–0.643] |
| final trajectory (largest map) frame coverage | 0.584 [0.584–0.585] | 0.405 [0.000–0.640] |

Neither variant relocalises in the original map: about 3 s after the loss ORB-SLAM3 starts map 1 and
re-initialises monocularly.

## Online accuracy
| | plain | CLAHE |
|---|---|---|
| RMSE on timestamps every run tracks (37.8 % of frames) (m) | 0.192 [0.141–0.635] | 0.177 [0.140–0.193] |
| RMSE on each run's full coverage (m) | 0.233 [0.122–0.517] | 0.196 [0.165–1.116] |
| pairwise (25 pairs) Δ RMSE CLAHE − plain on shared frames (m) | | 0.003 [−0.445–0.563]; CLAHE lower in 11/25 |
| frames with a valid pose and error < 0.25 m (fraction of all frames) | 0.523 [0.449–0.541] | 0.543 [0.332–0.597] |
| frames with a valid pose and error < 0.5 m | 0.565 [0.485–0.583] | 0.635 [0.411–0.641] |

Frames only CLAHE tracks (5.6 % of the sequence, all inside 274–333 s): median error 0.22 m
[0.17–2.6 over run pairs]; share below 0.5 m 0.93 [0.25–1.00].

Per 5-s bin (online error / 1-s relative drift |Δest − Δgt|, median over runs):
* before the loss, 260–275 s: plain 0.31–0.47 m / 0.08–0.27 m per s (one run reaches 3.1 m in
  270–275 s); CLAHE 0.08–0.18 m / 0.03–0.04 m per s.
* CLAHE's extra seconds, 275–285 s: 0.29–0.34 m (max 0.42 m) and 4–5 cm/s drift. The delayed
  failure carries usable poses, not a slow divergence.
* after CLAHE re-initialises, 315–325 s: median 0.14–0.18 m but up to 5.5 m, drift 0.11–0.25 m per s
  (max 1.3). From 325 s most runs are at 0.13–0.19 m / ~2–6 cm/s; one run (bad initialisation)
  stays at 1.7–2.8 m.

## Finding
In five controlled repeats per configuration, CLAHE delayed the tracking failure by 5–11 s with
accurate online poses and halved the outage (59 s → 30–36 s). On the frames both configurations
track, online accuracy did not change. CLAHE's earlier recovery was a monocular re-initialisation
inside the smoke into a new, disconnected map. That map was fragile: 2/5 runs lost it again, 1/5
crashed, and only 1/5 merged back into the original map (plain: 5/5). As a result, CLAHE's final
trajectories covered less of the sequence. So CLAHE helps before the failure. After it, it trades
a shorter outage for a less reliable recovery.

Limits: one sequence; Sim(3) per session is non-causal; runs are not bit-reproducible (real-time
playback, multi-threaded), which the repeats are meant to capture.
