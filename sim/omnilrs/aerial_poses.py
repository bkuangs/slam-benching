"""Aerial (drone-like) lawnmower survey trajectory for the OmniLRS stereo renderer (render_stereo.py --env lunaryard).

Writes the same .npz as export_poses.py (timestamps_ns, T_world_cam0 = left OpenCV optical camera-to-world,
width/height/fx/fy/cx/cy/baseline_m/source) plus per-frame extras (agl_m, yaw_deg, pitch_deg, roll_deg, leg).

The path is a boustrophedon over the terrain centre: straight legs along world ±x joined by semicircular turns
flown at constant speed (constant yaw rate v/r, no instantaneous rotations). The camera looks along the direction
of travel, pitched down by --pitch-deg; gentle smooth altitude/attitude perturbations are added. Altitude is
--altitude-m above the DEM mean (constant apart from the perturbation); per-frame AGL is the height above the DEM
directly below the camera. The leg count/length are chosen to fill --duration-s, and the script checks that the
image footprint stays on the terrain. Run in the slam-benching venv:
    python sim/omnilrs/aerial_poses.py poses/lunaryard_aerial.npz --dem renders/<any lunaryard render>/dem.npy
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# ZED2 rectified stereo as used for LunaPolaris (see export_poses.py / docs/datasets/omnilrs.md).
ZED2 = dict(width=640, height=360, fx=263.27, baseline_m=0.11983)


def lawnmower_xy(s, centre, leg_len, spacing, n_legs):
    """Position and heading at arc lengths s along the survey (legs along +x/-x, stacked along +y)."""
    r = spacing / 2.0
    turn_len = np.pi * r
    period = leg_len + turn_len
    x0, y0 = centre[0] - leg_len / 2.0, centre[1] - (n_legs - 1) * spacing / 2.0
    xy, psi, leg = np.zeros((len(s), 2)), np.zeros(len(s)), np.zeros(len(s), int)
    for i, si in enumerate(s):
        k = min(int(si // period), n_legs - 1)
        u = si - k * period
        d = 1.0 if k % 2 == 0 else -1.0                     # +x on even legs
        ys = y0 + k * spacing
        xs = x0 if d > 0 else x0 + leg_len
        if u <= leg_len or k == n_legs - 1:
            xy[i] = (xs + d * u, ys)
            psi[i] = 0.0 if d > 0 else np.pi
            leg[i] = k
        else:                                               # semicircle towards +y around (leg end, ys + r)
            a = (u - leg_len) / r
            cx, cy = xs + d * leg_len, ys + r
            xy[i] = (cx + d * r * np.sin(a), cy - r * np.cos(a))
            psi[i] = (0.0 if d > 0 else np.pi) + d * a       # CCW on +x legs, CW on -x legs
            leg[i] = -1
    return xy, np.unwrap(psi), leg


def smooth_noise(t, amp, rng, periods=(13.0, 7.3, 4.1)):
    """Sum of sinusoids with random phases, scaled so max |value| == amp."""
    v = sum(np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi)) / (j + 1) for j, p in enumerate(periods))
    return amp * v / np.max(np.abs(v)) if amp > 0 else np.zeros_like(t)


def camera_rotation(psi, pitch, roll):
    """OpenCV optical camera-to-world rotations: z along heading psi, pitched down by `pitch`, rolled by `roll`."""
    f = np.stack([np.cos(pitch) * np.cos(psi), np.cos(pitch) * np.sin(psi), -np.sin(pitch)], 1)
    r = np.stack([np.sin(psi), -np.cos(psi), np.zeros_like(psi)], 1)
    r = Rotation.from_rotvec(f * roll[:, None]).apply(r)
    d = np.cross(f, r)
    return np.stack([r, d, f], axis=2)


class Dem:
    def __init__(self, path):
        path = Path(path)
        self.z = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        self.res = float(meta["resolution_m"])
        self.origin = np.asarray(meta.get("origin_xy_m", [0.0, 0.0]), float)
        self.size = np.array(self.z.shape[::-1]) * self.res   # (x extent, y extent)

    def height(self, xy):
        g = (np.atleast_2d(xy) - self.origin) / self.res
        ny, nx = self.z.shape
        gx, gy = np.clip(g[:, 0], 0, nx - 1.001), np.clip(g[:, 1], 0, ny - 1.001)
        x0, y0 = gx.astype(int), gy.astype(int)
        dx, dy = gx - x0, gy - y0
        z = self.z
        return (z[y0, x0] * (1 - dx) * (1 - dy) + z[y0, x0 + 1] * dx * (1 - dy)
                + z[y0 + 1, x0] * (1 - dx) * dy + z[y0 + 1, x0 + 1] * dx * dy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--dem", default=None, help="dem.npy (+ dem.json) from a render; without it the terrain is "
                    "assumed flat at z=0 with --size extent")
    ap.add_argument("--size", type=float, default=80.0, help="terrain extent (m) when no --dem is given")
    ap.add_argument("--altitude-m", type=float, default=5.0, help="above the DEM mean")
    ap.add_argument("--speed", type=float, default=2.0, help="m/s")
    ap.add_argument("--pitch-deg", type=float, default=60.0, help="camera pitch below horizontal")
    ap.add_argument("--rate-hz", type=float, default=15.0)
    ap.add_argument("--duration-s", type=float, default=120.0)
    ap.add_argument("--legs", type=int, default=5)
    ap.add_argument("--spacing", type=float, default=10.0, help="leg spacing = turn diameter (m)")
    ap.add_argument("--edge-margin", type=float, default=10.0, help="min camera distance from the terrain edge (m)")
    ap.add_argument("--alt-perturb-m", type=float, default=0.3)
    ap.add_argument("--att-perturb-deg", type=float, default=2.0, help="pitch/roll/yaw perturbation amplitude")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t0-ns", type=int, default=1_800_000_000_000_000_000)
    a = ap.parse_args()

    dem = Dem(a.dem) if a.dem else None
    origin = dem.origin if dem else np.zeros(2)
    size = dem.size if dem else np.array([a.size, a.size])
    centre = origin + size / 2.0
    z_ref = float(dem.z.mean()) if dem else 0.0

    n = int(round(a.duration_s * a.rate_hz))
    t = np.arange(n) / a.rate_hz
    r = a.spacing / 2.0
    leg_len = (a.speed * t[-1] - (a.legs - 1) * np.pi * r) / a.legs
    xy, psi, leg = lawnmower_xy(a.speed * t, centre, leg_len, a.spacing, a.legs)

    rng = np.random.default_rng(a.seed)
    att = np.radians(a.att_perturb_deg)
    z = z_ref + a.altitude_m + smooth_noise(t, a.alt_perturb_m, rng)
    pitch = np.radians(a.pitch_deg) + smooth_noise(t, att, rng, (9.0, 5.3, 3.1))
    roll = smooth_noise(t, att, rng, (8.0, 4.7, 2.9))
    yaw = psi + smooth_noise(t, att, rng, (11.0, 6.1, 3.7))

    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = camera_rotation(yaw, pitch, roll)
    T[:, :3, 3] = np.column_stack([xy, z])

    # Checks: distance from the edges, and the image-corner rays hit the ground inside the terrain.
    lo, hi = origin, origin + size
    edge = np.min(np.minimum(xy - lo, hi - xy))
    W, H, fx = ZED2["width"], ZED2["height"], ZED2["fx"]
    corners = np.array([[-W / 2, -H / 2, fx], [W / 2, -H / 2, fx], [-W / 2, H / 2, fx], [W / 2, H / 2, fx]]) / fx
    rays = np.einsum("nij,kj->nki", T[:, :3, :3], corners)
    agl = z - (dem.height(xy) if dem else 0.0)
    hit = xy[:, None, :] + rays[..., :2] * ((agl[:, None] / -rays[..., 2]))[..., None]
    foot_out = np.max(np.maximum(lo - hit, hit - hi))
    assert np.all(rays[..., 2] < 0), "an image corner looks above the horizon"
    assert edge >= a.edge_margin, f"camera comes {edge:.1f} m from the edge (< {a.edge_margin})"
    assert foot_out < 0, f"image footprint leaves the terrain by {foot_out:.1f} m"

    ts = a.t0_ns + np.round(t * 1e9).astype(np.int64)
    vel = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * a.rate_hz
    rot = Rotation.from_matrix(T[:-1, :3, :3]).inv() * Rotation.from_matrix(T[1:, :3, :3])
    wmax = np.degrees(np.max(rot.magnitude())) * a.rate_hz
    src = (f"aerial lawnmower: {a.legs} legs x {leg_len:.1f} m along +/-x, spacing {a.spacing} m, "
           f"{a.altitude_m} m above DEM mean, {a.speed} m/s, pitch {a.pitch_deg} deg, {a.rate_hz} Hz, "
           f"perturb +/-{a.alt_perturb_m} m / +/-{a.att_perturb_deg} deg, seed {a.seed}; "
           f"DEM {a.dem or 'flat (none given)'}")
    np.savez(a.out, timestamps_ns=ts, T_world_cam0=T, width=W, height=H, fx=fx, fy=fx, cx=W / 2.0, cy=H / 2.0,
             baseline_m=ZED2["baseline_m"], source=src, agl_m=agl, yaw_deg=np.degrees(yaw),
             pitch_deg=np.degrees(pitch), roll_deg=np.degrees(roll), leg=leg)
    print(src)
    print(f"{n} frames over {t[-1]:.1f} s, path {np.sum(vel) / a.rate_hz:.1f} m, speed {vel.min():.2f}-"
          f"{vel.max():.2f} m/s, max rotation rate {wmax:.1f} deg/s; x {xy[:, 0].min():.1f}-{xy[:, 0].max():.1f}, "
          f"y {xy[:, 1].min():.1f}-{xy[:, 1].max():.1f}; edge distance >= {edge:.1f} m, footprint margin "
          f"{-foot_out:.1f} m; AGL {agl.min():.2f}-{agl.max():.2f} m (mean {agl.mean():.2f}); frames on +x legs "
          f"{np.sum((leg >= 0) & (leg % 2 == 0))}, -x legs {np.sum((leg >= 0) & (leg % 2 == 1))}, turns {np.sum(leg < 0)} -> {a.out}")


if __name__ == "__main__":
    main()
