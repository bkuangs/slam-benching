"""Render a stereo sequence in the OmniLRS Lunalab along a given camera trajectory (headless Isaac Sim).

Builds the Lunalab (lab USD + a LiDAR-scanned lab DEM + OmniLRS rocks), places a rectified pinhole stereo pair
(ZED2-like, from export_poses.py), and for each pose renders left/right RGB and left z-depth. Output layout:
    out/cam0/<ns>.png, out/cam1/<ns>.png    rectified RGB (BGR PNG, like the prepared LunaPolaris input)
    out/depth0/<ns>.png                     left z-depth, uint16 millimetres (0 = no hit / beyond 65 m)
    out/frames.csv                          timestamp_ns,left_path,right_path
    out/gt_cam0.tum                         left optical-frame pose per frame (TUM, world = Lunalab frame)
    out/calib.json                          intrinsics / baseline actually rendered
    out/dem.npy, out/dem.json               terrain height map in world XY (row = y index, col = x index)
    out/rocks.json                          rock instance poses
Run with /isaac-sim/python.sh inside the Isaac Sim 5.0.0 container, from the OmniLRS repo root
(PYTHONPATH must include the OmniLRS root and its extra pip deps). See render.sbatch.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("poses", help=".npz from export_poses.py")
ap.add_argument("out")
ap.add_argument("--terrain", default="Terrain_lidar01")
ap.add_argument("--max-frames", type=int, default=None)
ap.add_argument("--rt-subframes", type=int, default=4)
ap.add_argument("--projector", default="on", choices=["on", "off"])
ap.add_argument("--projector-pos", type=float, nargs=3, default=None, help="x y z (m)")
ap.add_argument("--projector-intensity", type=float, default=None)
ap.add_argument("--room-lights", default="on", choices=["on", "off"])
ap.add_argument("--rocks", type=int, default=10, help="rocks per instancer request (0 disables rocks)")
ap.add_argument("--preview", type=int, default=0, help="only render this many frames spread over the sequence")
a = ap.parse_args()

t_start = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 3})

import cv2  # noqa: E402
import omni  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

sys.path.insert(0, os.getcwd())
from run import instantiateConfigs, omegaconfToDict  # noqa: E402  (OmniLRS run.py; registers resolvers)
from src.configurations.simulator_mode_enum import SimulatorMode  # noqa: E402
from src.environments.lunalab import LunalabController  # noqa: E402

P = np.load(a.poses)
t_ns = P["timestamps_ns"]
T_wc = P["T_world_cam0"]
W, H = int(P["width"]), int(P["height"])
fx, baseline = float(P["fx"]), float(P["baseline_m"])
if a.preview:
    idx = np.linspace(0, len(t_ns) - 1, a.preview).astype(int)
    t_ns, T_wc = t_ns[idx], T_wc[idx]
if a.max_frames:
    t_ns, T_wc = t_ns[: a.max_frames], T_wc[: a.max_frames]

with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "cfg"), version_base="1.3"):
    cfg = compose("config", overrides=["environment=lunalab", "mode=SDG"])
cfg = instantiateConfigs(omegaconfToDict(cfg))
env_cfg = dict(cfg["environment"])
env_cfg.pop("name", None)
env_cfg.pop("seed", None)
env_cfg.pop("robots_settings", None)

world = World(stage_units_in_meters=1.0)
world.reset()
LC = LunalabController(mode=SimulatorMode.SDG, **env_cfg)
LC.load()
LC.T.loadTerrainByName(a.terrain)
LC.load_DEM()
LC.RM.updateImageData(LC.dem, LC.mask)
if a.rocks > 0:
    LC.RM.randomizeInstancers(a.rocks)
else:
    LC.enable_rocks(False)
LC.turn_projector_on_off(a.projector == "on")
LC.turn_room_lights_on_off(a.room_lights == "on")
if a.projector_pos is not None:
    LC.set_projector_pose(tuple(a.projector_pos))
if a.projector_intensity is not None:
    LC.set_projector_intensity(a.projector_intensity)
for _ in range(20):
    world.step(render=True)
stage = omni.usd.get_context().get_stage()
print(f"RENDER scene ready in {time.time() - t_start:.1f}s terrain={a.terrain}", flush=True)

# Rectified pinhole pair, principal point at the image centre (USD aperture offsets left at 0).
H_AP = 20.955
focal = fx * H_AP / W
UsdGeom.Xform.Define(stage, "/StereoRig")
cams = []
for name in ("left", "right"):
    c = UsdGeom.Camera.Define(stage, f"/StereoRig/{name}")
    c.CreateFocalLengthAttr(focal)
    c.CreateHorizontalApertureAttr(H_AP)
    c.CreateVerticalApertureAttr(H_AP * H / W)
    c.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
    c.CreateFStopAttr(0.0)
    cams.append(c)
xf = [UsdGeom.Xformable(c.GetPrim()) for c in cams]
ops = [x.AddTransformOp() for x in xf]
CV_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV optical (x right, y down, z fwd) -> USD camera (-z fwd, y up)
T_lr = np.eye(4)
T_lr[0, 3] = baseline


def set_pose(T_world_left):
    for op, T in zip(ops, (T_world_left, T_world_left @ T_lr)):
        op.Set(Gf.Matrix4d((T @ CV_TO_USD).T.tolist()))  # Gf uses row vectors -> transpose


rps = [rep.create.render_product(c.GetPath().pathString, (W, H)) for c in cams]
rgb = [rep.AnnotatorRegistry.get_annotator("rgb") for _ in rps]
for r, rp in zip(rgb, rps):
    r.attach(rp)
depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
depth.attach(rps[0])
rep.orchestrator.set_capture_on_play(False)

for d in ("cam0", "cam1", "depth0"):
    os.makedirs(os.path.join(a.out, d), exist_ok=True)

dem = np.flip(LC.T.getDEM(), 0)  # row = y / res, col = x / res (see TerrainManager.update)
res = float(LC.T._grid_size)
np.save(os.path.join(a.out, "dem.npy"), dem.astype(np.float32))
json.dump({"resolution_m": res, "origin_xy_m": list(map(float, LC.T._mesh_pos[:2])), "terrain": a.terrain,
           "indexing": "dem[iy, ix] = height at world (x0 + ix*res, y0 + iy*res)"},
          open(os.path.join(a.out, "dem.json"), "w"), indent=1)
rocks = []
for prim in Usd.PrimRange(stage.GetPrimAtPath("/Lunalab/Rocks")):
    if prim.IsA(UsdGeom.PointInstancer):
        pi = UsdGeom.PointInstancer(prim)
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        pos = np.array(pi.GetPositionsAttr().Get() or [])
        ori = pi.GetOrientationsAttr().Get() or []
        scl = np.array(pi.GetScalesAttr().Get() or np.ones((len(pos), 3)))
        ids = np.array(pi.GetProtoIndicesAttr().Get() or [])
        protos = [str(t) for t in pi.GetPrototypesRel().GetTargets()]
        for k in range(len(pos)):
            q = ori[k]
            rocks.append({"instancer": str(prim.GetPath()), "prototype": protos[int(ids[k])] if len(ids) else None,
                          "position_world": (M[:3, :3] @ pos[k] + M[:3, 3]).tolist(),
                          "orientation_wxyz_local": [q.GetReal(), *q.GetImaginary()], "scale": scl[k].tolist()})
json.dump(rocks, open(os.path.join(a.out, "rocks.json"), "w"))
json.dump({"width": W, "height": H, "fx": fx, "fy": fx, "cx": W / 2.0, "cy": H / 2.0, "baseline_m": baseline,
           "distortion": [0, 0, 0, 0], "source": str(P["source"]), "renderer": "RayTracedLighting",
           "rt_subframes": a.rt_subframes, "lighting": {"projector": a.projector, "room_lights": a.room_lights,
           "projector_pos": a.projector_pos, "projector_intensity": a.projector_intensity}, "rocks": a.rocks},
          open(os.path.join(a.out, "calib.json"), "w"), indent=1)

# Warm up so the first frame is converged.
set_pose(T_wc[0])
for _ in range(10):
    rep.orchestrator.step(rt_subframes=a.rt_subframes, delta_time=0.0, pause_timeline=True)

from scipy.spatial.transform import Rotation  # noqa: E402

fcsv = open(os.path.join(a.out, "frames.csv"), "w")
fcsv.write("timestamp_ns,left_path,right_path\n")
ftum = open(os.path.join(a.out, "gt_cam0.tum"), "w")
t0 = time.time()
for i, (t, T) in enumerate(zip(t_ns, T_wc)):
    set_pose(T)
    rep.orchestrator.step(rt_subframes=a.rt_subframes, delta_time=0.0, pause_timeline=True)
    for k, r in enumerate(rgb):
        img = np.asarray(r.get_data())[..., :3]
        cv2.imwrite(os.path.join(a.out, f"cam{k}", f"{t}.png"), img[..., ::-1])
    z = np.asarray(depth.get_data())
    z = np.where(np.isfinite(z) & (z > 0) & (z < 65.0), z * 1000.0, 0).astype(np.uint16)
    cv2.imwrite(os.path.join(a.out, "depth0", f"{t}.png"), z)
    fcsv.write(f"{t},cam0/{t}.png,cam1/{t}.png\n")
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    ftum.write(f"{t * 1e-9:.9f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
    if i % 100 == 0:
        print(f"RENDER {i}/{len(t_ns)} {(time.time() - t0) / (i + 1):.3f}s/frame", flush=True)
fcsv.close()
ftum.close()
print(f"RENDER done {len(t_ns)} frames in {time.time() - t0:.1f}s; total {time.time() - t_start:.1f}s", flush=True)
rep.orchestrator.wait_until_complete()
app.close()
