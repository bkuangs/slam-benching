# Ground-only LiDAR: does LiDAR-inertial odometry survive open terrain? (FAST-LIO2, LunaPolaris)

Date 2026-10-28. Question: on smooth, geometrically repetitive terrain, is LiDAR registration poorly
constrained enough that a camera is needed? FAST-LIO2's good LunaPolaris results
(`docs/results/fastlio_lunapolaris.md`, ATE 5–7 cm) could come from the LunaLab walls rather than the
terrain. A lunar flyer sees no walls.

## Method

* **Ground-only crop.** This is a LiDAR preparation option (`docs/datasets/lunapolaris.md`). Returns are
  dropped if they are higher than *h*, or farther than *r* horizontally, from the rover ground-plane frame
  `leo01/base_footprint`. The transform comes from /tf_static only; no ground truth is used. Two crops:
  * `ground`: *h* 0.5 m, *r* 3 m.
  * `ground_harsh`: *h* 0.3 m, *r* 2.5 m.

  A height-only crop is not enough, because the lower part of the walls stays in the scan (table below).
* **Point-count control (`decim`).** Full scans with the walls kept, but only every n-th point is used:
  * n = 5 gives ~9k returns per scan, the same as the ground crop.
  * n = 15 gives ~3k, about a third of the ground crop.
* **Settings.** Everything else is identical to the baseline configs: ZED2 IMU, FAST-LIO2 `7cc4175`,
  `point_filter_num` 3, 0.2 m voxels, real-time playback, same node type.
* **Baseline.** The existing single full-scan runs from 2026-10-19. These use the same image and settings;
  the cache could not be regenerated under the scratch quota.
* **Configs.** `configs/lunapolaris_seq{01,02,11}_fastlio_lidar_inertial_ground.yaml`,
  `seq{01,11}_*_ground_harsh.yaml`, `seq{01,11}_*_decim5.yaml`, `seq11_*_decim15.yaml`.

## 1. Registration conditioning per scan (`scripts/lidar_degeneracy_probe.py`, 16 scans per sequence)

Conditioning is λ_min/λ_max of the translational and rotational blocks of the point-to-plane ICP
information matrix, from a single scan with planar normals. Weak directions are given in the ground-plane
frame (x forward, z up).

| seq | crop | planar points | vertical surfaces | translation cond. median (min) | weakest translation | rotation cond. | weakest rotation |
|---|---|---|---|---|---|---|---|
| 01 | full | 15.2k | 74 % | 0.291 (0.162) | lateral (y) | 0.147 | roll (x) |
| 01 | h 0.5 only | 14.3k | 60 % | 0.266 (0.169) | lateral (y) | 0.203 | roll |
| 01 | **ground** | 6.0k | 1 % | **0.031** (0.017) | **horizontal (x, y)** | 0.059 | **yaw (z)** |
| 01 | ground_harsh | 3.9k | 0 % | 0.024 (0.004) | horizontal | 0.049 | yaw |
| 02 | full | 15.2k | 77 % | 0.350 (0.076) | vertical | 0.094 | roll |
| 02 | **ground** | 5.6k | 7 % | **0.052** (0.013) | horizontal | 0.080 | yaw |
| 11 | full | 15.3k | 74 % | 0.388 (0.074) | lateral / vertical | 0.106 | roll |
| 11 | h 0.5 only | 14.5k | 61 % | 0.367 (0.063) | lateral | 0.167 | roll |
| 11 | **ground** | 6.1k | 8 % | **0.051** (0.014) | horizontal | 0.067 | yaw |
| 11 | ground_harsh | 3.7k | 2 % | 0.038 (0.005) | horizontal | 0.042 | yaw |

Figure: `docs/results/figures/lidar_crop_seq01.png` (top view of a full scan with the arena walls, and the
same scan after the ground crop).

## 2. FAST-LIO2 with ground-only LiDAR (`scripts/fastlio_divergence_report.py`)

Column definitions:
* **Final ATE:** SE(3)-aligned over the whole run.
* **120 s ATE:** the first 120 s scored on its own, with its own alignment. This is accuracy before the
  divergence.
* **n_eff:** FAST-LIO's effective point-to-plane matches per scan, as the median over the first 60 s.
* **Runaway:** the first time the estimated speed exceeds 1 m/s. The rover moves at about 0.05 m/s.
* **Starved:** the first time n_eff stays 0 for at least 10 scans.

| seq | variant | final ATE | 120 s ATE | coverage / loss events (reported) | n_eff | runaway | starved |
|---|---|---|---|---|---|---|---|
| 01 | full | 0.066 m | 0.028 m | 0.992 / 0 | 762 | – | – |
| 01 | decim5 | 0.055 m | 0.024 m | 0.992 / 0 | 613 | – | – |
| 01 | **ground** (×2) | **5.7e5 m** | 0.314 m | 0.992 / **0** | 158 | 158 s | 231 s |
| 01 | ground_harsh | 8.1e5 m | 0.683 m | 0.992 / 0 | 94 | 108 s | 169 s |
| 02 | full | 0.051 m | 0.017 m | 0.995 / 0 | 794 | – | – |
| 02 | **ground** (×2) | **4.8e5 m** | 0.261 m | 0.995 / **0** | 165 | 200 s | 307 s |
| 11 | full | 0.061 m | 0.029 m | 0.999 / 0 | 752 | – | – |
| 11 | decim5 | 0.059 m | 0.026 m | 0.999 / 0 | 613 | – | – |
| 11 | decim15 | 0.039 m | 0.017 m | 0.999 / 0 | 434 | – | – |
| 11 | **ground** (×2) | **4.1e5 m** | 0.139 m | 0.999 / **0** | 170 | 169 s | 318 s |
| 11 | ground_harsh | 6.4e5 m | 0.300 m | 0.999 / 0 | 107 | 134 s | 203 s |

The repeats are identical (FAST-LIO offline is deterministic), so "×2" adds no spread information.

## Findings

1. **The walls carry FAST-LIO2 on LunaPolaris.**
   * Keeping only the ground within 3 m lowers translational conditioning 6–9× (0.29–0.39 → 0.03–0.05).
   * The weak directions become exactly the ones expected on near-planar terrain: horizontal translation
     and yaw.
   * The rocks and craters of seq02/11 condition it slightly better than seq01, but not enough.
2. **Ground-only LiDAR–IMU fails on every sequence and both crops.**
   * It tracks for 1.5–3 minutes, already 5–12× less accurately than full scans (0.14–0.31 m vs
     0.017–0.029 m over 120 s).
   * The velocity estimate then runs away (108–200 s), and the trajectory ends hundreds of kilometres away.
   * Mechanism (from `frame_log.csv`): the accelerometer bias absorbs the unconstrained horizontal motion.
     It reaches 0.22 m/s² by 60 s, against 0.07 in the full run. Once the pose is off, scan-to-map matching
     loses its correspondences and the filter integrates the IMU alone.
3. **The cause is geometry, not point count.** Full scans thinned to the same number of points as the crop,
   or to a third of it, track normally (ATE 0.039–0.059 m).
4. **The failure is silent.**
   * Every diverged run reports 99–100 % coverage and zero tracking-loss events.
   * FAST-LIO2 has no degeneracy or consistency check. Even n_eff = 0 appears only 60–110 s after the
     runaway.
   * Any fused system that trusts the LIO covariance would accept these poses.
5. **Cheap GT-free signals separate the cases from the first minute:**
   * n_eff of 94–170 against 434–794 for every healthy run (including the point-matched controls);
   * the registration conditioning in §1;
   * a platform-speed plausibility check, which fires at the start of the runaway.

   Here the degradation is constant over the whole run, so these signals flag the entire run. They do not
   yet measure *detection lead time*.

## What this means for the camera–LiDAR–IMU architecture

* **The premise holds on real data.** LiDAR–IMU alone does not survive gently undulating terrain without
  vertical structure. The camera's job is to constrain the **horizontal-translation and yaw subspace**. That
  is the concrete target for directional, reliability-aware fusion.
* **The fair LiDAR-only baseline is a degeneracy-aware LIO, not plain FAST-LIO2.** Examples are solution
  remapping (Zhang, Kaess & Singh, ICRA 2016) and X-ICP-style localizability constraints. Such a LIO would
  hold the degenerate directions to the IMU prediction instead of letting the bias absorb them. Without that
  baseline, any camera benefit is overstated.
* **The silent failure is itself an argument for cross-modal checks.** For example, stereo VO and LIO
  disagree within seconds once the LIO velocity runs away. (Note: ORB-SLAM3 stereo tracks all of these
  sequences at 3–8 cm.)

## Caveats

* The crop also shrinks range and field of view (3 m). A flyer at altitude sees a larger terrain footprint,
  possibly with more relief, but no more vertical structure.
* LunaLab relief is gentle and is not a validated lunar terrain model.
* IMU noise parameters are FAST-LIO defaults, and the ZED2 IMU is consumer-grade. Calibrated parameters or a
  better IMU might delay the divergence. They would not remove the unobservability.
* The LiDAR extrinsic is the nominal URDF. The full-scan baseline is a single run from 2026-10-19.
* There is only one estimator (FAST-LIO2). Other LIO back ends may degrade differently.

## Next experiments

1. **Camera rescue:** run the same ground-only LiDAR with stereo, with versus without cross-modal
   verification, plus an oracle. Start loosely coupled. Add a tightly coupled LVI system (e.g. FAST-LIVO2 or
   LVI-SAM) as an adapter.
2. **Time-varying degradation:** walls removed only in some windows. This measures detection latency and
   lead time, and recovery, which the constant crop cannot.
3. **Degeneracy-aware LIO baseline:** freeze the degenerate directions to the IMU prediction (see above).
4. **OmniLRS aerial runs with a simulated LiDAR:** flat mare versus cratered terrain, to check that the
   result transfers to flight geometry.

## Reproduce

```bash
python scripts/lidar_degeneracy_probe.py --seq 01 02 11 --scans 16 --plot-dir docs/results/figures
for c in seq01_fastlio_lidar_inertial_ground seq02_fastlio_lidar_inertial_ground seq11_fastlio_lidar_inertial_ground \
         seq01_fastlio_lidar_inertial_ground_harsh seq11_fastlio_lidar_inertial_ground_harsh \
         seq01_fastlio_lidar_inertial_decim5 seq11_fastlio_lidar_inertial_decim5 seq11_fastlio_lidar_inertial_decim15; do
  sbatch -p ice-cpu -c 8 --mem=32G -t 03:00:00 --constraint=gold6226 scripts/slurm_slambench.sbatch run configs/lunapolaris_$c.yaml
done
python scripts/fastlio_divergence_report.py ~/scratch/slam-bench-work/runs/lunapolaris_seq*_fastlio_lidar_inertial*/<run> --window-ate 120
```

Each cropped or decimated cache is 0.6–1.6 GB. A full-scan cache is about 8 GB; launching several at once
exceeded the 300 GB scratch quota.
