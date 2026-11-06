"""Render a stereo sequence in an OmniLRS environment along a given camera trajectory (headless Isaac Sim).

--env lunalab (default): the Lunalab (lab USD + a LiDAR-scanned lab DEM + OmniLRS rocks).
--env lunaryard: the Lunaryard (procedural DEM with craters + rocks, fixed seed; 20/40/80 m) lit by a static
    distant-light sun (--sun-azimuth/--sun-elevation/--sun-intensity; stellar-engine time updates disabled).
--env largescale: the LargeScale environment (a real LOLA 5 m/px south-pole DEM, cfg default Site20, with
    procedurally generated 2.5 cm/px high-resolution blocks + craters + rocks streamed around the camera through
    geometry clipmaps), lit by the same static sun. Local/world frame: origin = the cfg's starting_position
    (--ls-start), z = absolute DEM elevation (km-scale). Before each frame the terrain is re-centred on the camera xy
    (LargeScaleTerrainManager.update_visual_mesh, which blocks until the new high-res blocks are generated). Extra
    outputs: out/terrain.csv (per frame: high-res terrain height under the camera, AGL, mesh update, timings).
    Depth is stored in units of --depth-unit-mm (default 10 mm here, so up to 655 m; 1 mm for the other envs).
    --terrain-shader bombed replaces the 2 m-periodic OmniPBR regolith with a world-space, texture-bombed MDL with
    large-scale albedo noise (regolith_mdl.py).
Places a rectified pinhole stereo pair (ZED2-like, from export_poses.py or aerial_poses.py), and for each pose
renders left/right RGB and left z-depth with a fixed exposure (auto-exposure off). Output layout:
    out/cam0/<ns>.png, out/cam1/<ns>.png    rectified RGB (BGR PNG, like the prepared LunaPolaris input)
    out/depth0/<ns>.png                     left z-depth, uint16 in calib.json depth0_unit_mm (default 1 mm;
                                            0 = no hit / beyond 65535 units)
    out/frames.csv                          timestamp_ns,left_path,right_path
    out/gt_cam0.tum                         left optical-frame pose per frame (TUM, world = Lunalab frame)
    out/calib.json                          intrinsics / baseline actually rendered
    out/dem.npy, out/dem.json               terrain height map in world XY (row = y index, col = x index)
    out/rocks.json                          rock instance poses
    out/preview_index.json                  (only with --preview) source indices of the rendered frames
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
ap.add_argument("--env", default="lunalab", choices=["lunalab", "lunaryard", "largescale"])
ap.add_argument("--terrain", default="Terrain_lidar01", help="lunalab only")
ap.add_argument("--max-frames", type=int, default=None)
ap.add_argument("--rt-subframes", type=int, default=4)
ap.add_argument("--projector", default="on", choices=["on", "off"])
ap.add_argument("--projector-pos", type=float, nargs=3, default=None, help="x y z (m)")
ap.add_argument("--projector-intensity", type=float, default=None)
ap.add_argument("--room-lights", default="on", choices=["on", "off"])
ap.add_argument("--rocks", type=int, default=10, help="rocks per instancer request (0 disables rocks)")
ap.add_argument("--yard-size", type=int, default=80, choices=[20, 40, 80], help="lunaryard only: cfg lunaryard_<N>m")
ap.add_argument("--seed", type=int, default=None, help="lunaryard only: terrain/rock seed (default: the cfg's)")
ap.add_argument("--sun-azimuth", type=float, default=180.0,
                help="lunaryard/largescale, deg; OmniLRS convention: direction to the sun = (-sin az, cos az) in world xy")
ap.add_argument("--sun-elevation", type=float, default=45.0, help="lunaryard/largescale, deg above the horizon")
ap.add_argument("--sun-intensity", type=float, default=None, help="lunaryard/largescale (cfg default 1750)")
ap.add_argument("--terrain-material", default=None, choices=["Basalt", "Sand", "LunarRegolith8k"],
                help="lunaryard: terrain look (default Basalt = GravelStones.mdl); largescale: the material to modify "
                     "with --texture-scale/--albedo-brightness (default and cfg: LunarRegolith8k)")
ap.add_argument("--ls-start", type=float, nargs=2, default=None,
                help="largescale only: starting_position (global m from the DEM centre) = local origin (cfg 2800 -2200)")
ap.add_argument("--ls-dem", default=None, help="largescale only: lr_dem_name (cfg default Site20_final_adj_5mpp_surf)")
ap.add_argument("--ls-update-m", type=float, default=None,
                help="largescale only: update_every_n_meters (cfg 2). The terrain mesh is re-centred in steps of this "
                     "and its UVs are mesh-local, so the texture stays fixed to the ground only if this is a multiple "
                     "of the texture period 1/texture_scale (LunarRegolith8k: 0.5 -> 2 m)")
ap.add_argument("--depth-unit-mm", type=float, default=None,
                help="depth0 PNG unit (default 1 mm; 10 mm for largescale)")
ap.add_argument("--mesh-settle", type=int, default=1,
                help="largescale only: extra app updates after a terrain mesh update, before rendering")
ap.add_argument("--texture-scale", type=float, default=None,
                help="lunaryard only: OmniPBR texture_scale of the terrain material; terrain UVs are in m and are "
                     "multiplied by it, so the texture repeats every 1/S m (cfg MDLs: Basalt 5, LunarRegolith8k 0.5, "
                     "Sand 1 -- far too fine to resolve from a few m altitude)")
ap.add_argument("--albedo-brightness", type=float, default=None,
                help="lunaryard only: OmniPBR albedo_brightness of the terrain material (Basalt 0.3)")
ap.add_argument("--albedo-add", type=float, default=None,
                help="lunaryard/largescale: OmniPBR albedo_add of the terrain material (LunarRegolith8k 1 = washed out "
                     "white without auto-exposure; GravelStones 0)")
ap.add_argument("--terrain-shader", default="omnipbr", choices=["omnipbr", "bombed"],
                help="largescale only: 'omnipbr' = the cfg OmniPBR material on the clipmap UVs (LunarRegolith8k tiles "
                     "every 2 m); 'bombed' = regolith_mdl.py: world-space texture bombing of the LunarRegolith8k maps "
                     "(random rotation/offset per cell, variance-preserving blend) + world-space fbm albedo variation")
ap.add_argument("--bomb-cell-m", type=float, default=1.0, help="bombed: lattice cell size (m)")
ap.add_argument("--bright-amp", type=float, default=0.2,
                help="bombed: albedo variation, +-amp ~ 2 sigma (std amp/2), clamped at +-1.5 amp; 0 disables")
ap.add_argument("--bright-wavelengths", type=float, nargs="+", default=[48.0, 23.0, 11.0, 5.3],
                help="bombed: fbm octave wavelengths (m)")
ap.add_argument("--shader-seed", type=float, default=0.0, help="bombed: hash seed of the bombing and the albedo noise")
ap.add_argument("--lens-flares", default="off", choices=["on", "off"], help="RTX lens-flare post effect")
ap.add_argument("--dem-only", action="store_true", help="build the scene, write dem/rocks/calib, render nothing")
ap.add_argument("--preview", type=int, default=0, help="only render this many frames spread over the sequence")
a = ap.parse_args()
if a.terrain_shader != "omnipbr" and a.env != "largescale":
    ap.error("--terrain-shader bombed is implemented for --env largescale only")
if a.terrain_material is None:
    a.terrain_material = "LunarRegolith8k" if a.env == "largescale" else "Basalt"
if a.depth_unit_mm is None:
    a.depth_unit_mm = 10.0 if a.env == "largescale" else 1.0

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
preview_idx = None
if a.preview:
    preview_idx = np.linspace(0, len(t_ns) - 1, a.preview).astype(int)
    t_ns, T_wc = t_ns[preview_idx], T_wc[preview_idx]
if a.max_frames:
    t_ns, T_wc = t_ns[: a.max_frames], T_wc[: a.max_frames]

env_name = {"lunalab": "lunalab", "lunaryard": f"lunaryard_{a.yard_size}m", "largescale": "largescale"}[a.env]
with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "cfg"), version_base="1.3"):
    overrides = [f"environment={env_name}", "mode=SDG"]
    if a.env == "lunaryard" and a.seed is not None:
        overrides.append(f"environment.seed={a.seed}")
    if a.env == "lunaryard":
        overrides.append(f"environment.terrain_manager.texture_path=/Lunaryard/Looks/{a.terrain_material}")
    if a.env == "largescale" and a.ls_start is not None:
        overrides.append(f"environment.large_scale_terrain.starting_position=[{a.ls_start[0]},{a.ls_start[1]}]")
    if a.env == "largescale" and a.ls_update_m is not None:
        overrides.append(f"environment.large_scale_terrain.update_every_n_meters={float(a.ls_update_m)}")
    if a.env == "largescale" and a.ls_dem is not None:
        overrides.append(f"+environment.large_scale_terrain.lr_dem_name={a.ls_dem}")
    cfg = compose("config", overrides=overrides)
cfg = instantiateConfigs(omegaconfToDict(cfg))
env_cfg = dict(cfg["environment"])
env_cfg.pop("name", None)
seed = env_cfg.pop("seed", None)
env_cfg.pop("robots_settings", None)

world = World(stage_units_in_meters=1.0)
world.reset()
lighting = {}
if a.env == "lunalab":
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
    lighting = {"projector": a.projector, "room_lights": a.room_lights, "projector_pos": a.projector_pos,
                "projector_intensity": a.projector_intensity}
    terrain_name, rocks_path = a.terrain, "/Lunalab/Rocks"
else:
    sun = env_cfg["sun_settings"]
    sun.azimuth, sun.elevation = float(a.sun_azimuth), float(a.sun_elevation)
    if a.sun_intensity is not None:
        sun.intensity = float(a.sun_intensity)
    env_cfg["stellar_engine_settings"] = None  # static sun: create_sun() places it from sun_settings, no updates
if a.env == "largescale":
    from src.environments.large_scale_lunar import LargeScaleController  # noqa: E402

    t_load = time.time()
    # The HR DEM generator's monitor thread polls is_simulation_alive and shuts the workers down when it turns False.
    LC = LargeScaleController(mode=SimulatorMode.SDG, is_simulation_alive=app.is_running, **env_cfg)
    LC.load()  # builds the clipmaps + HR blocks around the local origin (blocking)
    ls_cfg = env_cfg["large_scale_terrain"]
    ls_start = tuple(map(float, ls_cfg.starting_position))
    clip = LC.LSTM.nested_clipmap_manager
    print(f"RENDER LargeScale terrain {ls_cfg.lr_dem_name} start {ls_start} built in {time.time() - t_load:.1f}s; "
          f"HR DEM {LC.LSTM.map_manager.get_hr_dem_shape()} @ {LC.LSTM.map_manager.get_hr_dem_res()} m, fine "
          f"clipmap {clip.fine_clipmap_specs.numMeshLODLevels} levels, coarse {clip.coarse_clipmap_specs.numMeshLODLevels}",
          flush=True)
    terrain_name, rocks_path = f"LargeScale {ls_cfg.lr_dem_name} start={ls_start} seed={seed}", None
    ts = a.texture_scale if a.texture_scale is not None else {"Basalt": 5.0, "Sand": 1.0, "LunarRegolith8k": 0.5}[
        a.terrain_material]  # the MDLs' texture_scale
    k = ls_cfg.update_every_n_meters * ts
    # (the bombed shader projects in world space and does not use the mesh UVs)
    assert a.terrain_shader == "bombed" or (abs(k - round(k)) < 1e-6 and round(k) >= 1), (
        f"update_every_n_meters {ls_cfg.update_every_n_meters} is not a multiple of the texture period {1 / ts} m: "
        "the texture would slide over the terrain at every mesh update (use --ls-update-m)")
    from src.terrain_management.large_scale_terrain.pxr_utils import bind_material, load_material  # noqa: E402

    if a.terrain_shader == "bombed":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import regolith_mdl  # noqa: E402

        shader = {"cell_m": a.bomb_cell_m, "tex_period_m": 1.0 / ts, "bright_amp": a.bright_amp,
                  "bright_wavelengths_m": tuple(a.bright_wavelengths), "offset_xy": ls_start, "seed": a.shader_seed,
                  "albedo_brightness": a.albedo_brightness if a.albedo_brightness is not None else 0.6}
        import hashlib

        name = "RegolithBombed_" + hashlib.sha1(json.dumps(shader, sort_keys=True).encode()).hexdigest()[:10]
        mdl = os.path.join(os.getcwd(), "assets", "Textures", name + ".mdl")
        with open(mdl, "w") as f:
            f.write(regolith_mdl.mdl_text(name=name, **shader))
        mtl = load_material(name, mdl, "/LargeScaleLunar/Looks")
        for cm in (clip.fine_clipmap_manager, clip.coarse_clipmap_manager):
            bind_material(cm._stage, mtl, cm._mesh_path)
        print("terrain material", mtl, "from", mdl, shader, flush=True)
elif a.env == "lunaryard":
    from src.environments.lunaryard import LunaryardController  # noqa: E402

    LC = LunaryardController(mode=SimulatorMode.SDG, **env_cfg)
    LC.load()  # terrain_id -1: procedural DEM (craters) from the seeded generator, then 10 rocks per request
    if a.rocks != 10:
        if a.rocks > 0:
            LC.RM.randomizeInstancers(a.rocks)
        else:
            LC.enable_rocks(False)
if a.env != "lunalab" and a.terrain_shader == "omnipbr":
    if a.texture_scale is not None or a.albedo_brightness is not None or a.albedo_add is not None:
        # The OmniLRS terrain MDLs are `material X(*)` (parameters baked in, not overridable from USD), so write a
        # copy with the new values next to the original (relative texture paths keep working) and bind that.
        import re

        from src.terrain_management.large_scale_terrain.pxr_utils import load_material  # noqa: E402
        from WorldBuilders import pxr_utils  # noqa: E402

        src_mdl = {"Basalt": "GravelStones", "Sand": "Sand", "LunarRegolith8k": "LunarRegolith8k"}[a.terrain_material]
        src_mdl = os.path.join(os.getcwd(), "assets", "Textures", src_mdl + ".mdl")
        text = open(src_mdl).read()
        name = a.terrain_material
        if a.texture_scale is not None:
            text, n = re.subn(r"(\n\s*texture_scale: float2\()[^)]*\)", rf"\g<1>{a.texture_scale:.4f}f)", text)
            assert n == 1, "texture_scale not found in " + src_mdl
            name += f"_ts{a.texture_scale:g}"
        if a.albedo_brightness is not None:
            text, n = re.subn(r"(albedo_brightness: )[^,]*,", rf"\g<1>{a.albedo_brightness:.4f}f,", text)
            assert n == 1, "albedo_brightness not found in " + src_mdl
            name += f"_ab{a.albedo_brightness:g}"
        if a.albedo_add is not None:
            text, n = re.subn(r"(albedo_add: )[^,]*,", rf"\g<1>{a.albedo_add:.4f}f,", text)
            assert n == 1, "albedo_add not found in " + src_mdl
            name += f"_aa{a.albedo_add:g}"
        name = name.replace(".", "p").replace("-", "m")
        text, n = re.subn(r"export material \w+\(", f"export material {name}(", text)
        assert n == 1
        mdl = os.path.join(os.path.dirname(src_mdl), name + ".mdl")
        with open(mdl, "w") as f:
            f.write(text)
        if a.env == "lunaryard":
            mtl = load_material(name, mdl, "/Lunaryard/Looks")
            pxr_utils.applyMaterialFromPath(LC.stage, LC.T._mesh_path, mtl)
        else:
            mtl = load_material(name, mdl, "/LargeScaleLunar/Looks")
            for cm in (clip.fine_clipmap_manager, clip.coarse_clipmap_manager):
                bind_material(cm._stage, mtl, cm._mesh_path)
        print("terrain material", mtl, "from", mdl, flush=True)
if a.env != "lunalab":
    if a.env == "lunaryard":
        terrain_name, rocks_path = f"{env_name} procedural seed={seed}", "/Lunaryard/Rocks"

import carb  # noqa: E402

settings = carb.settings.get_settings()
if a.env != "lunalab":  # (Lunalab renders keep the Kit defaults they were made with)
    settings.set("/rtx/post/histogram/enabled", False)  # no auto-exposure: a dimmer sun gives darker images
    settings.set("/rtx/post/lensFlares/enabled", a.lens_flares == "on")
if a.lens_flares == "on":  # OmniLRS rendering cfg defaults (cfg/rendering/ray_tracing.yaml)
    for k, v in {"flareScale": 0.5, "blades": 9, "apertureRotation": 0.0, "sensorDiagonal": 28.0,
                 "sensorAspectRatio": 1.5, "fNumber": 2.8, "focalLength": 12.0}.items():
        settings.set(f"/rtx/post/lensFlares/{k}", v)
for _ in range(20):
    world.step(render=True)
stage = omni.usd.get_context().get_stage()
from pxr import UsdLux  # noqa: E402

lights = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdLux.LightAPI):
        ia = prim.GetAttribute("inputs:intensity")
        lights.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                       "intensity": ia.Get() if ia and ia.IsValid() else None,
                       "visible": UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible})
if a.env != "lunalab":
    sun_prim = stage.GetPrimAtPath(LC.get_sun_prim_path() + "/sun")
    M = np.array(UsdGeom.Xformable(sun_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    to_sun = M[:3, :3] @ np.array([0.0, 0.0, 1.0])  # a DistantLight emits along its local -Z
    to_sun /= np.linalg.norm(to_sun)
    lighting = {"type": "static distant light (stellar engine off)", "sun_azimuth_deg": a.sun_azimuth,
                "sun_elevation_deg": a.sun_elevation, "sun_intensity": float(sun.intensity),
                "sun_angle_deg": float(sun.angle), "to_sun_world": to_sun.round(6).tolist(),
                "to_sun_elevation_deg_measured": float(np.degrees(np.arcsin(to_sun[2])))}


def _tree(path):
    v = settings.get(path)
    return v if isinstance(v, (dict, bool, int, float, str)) or v is None else str(v)


lighting.update({"lens_flares": a.lens_flares, "lights": lights,
                 "exposure": {"auto_exposure_histogram": _tree("/rtx/post/histogram/enabled"),
                              "tonemap": _tree("/rtx/post/tonemap")}})
print(f"RENDER scene ready in {time.time() - t_start:.1f}s terrain={terrain_name} lighting="
      f"{json.dumps({k: v for k, v in lighting.items() if k != 'exposure'}, default=str)}", flush=True)

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
    c.CreateClippingRangeAttr(Gf.Vec2f(0.02, 2000.0 if a.env == "largescale" else 100.0))
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

if a.env == "largescale":
    # The 5 m/px source DEM (+/- 1 km around the origin) in the local frame; the rendered surface adds the 2.5 cm
    # procedural detail and craters on top (bicubic-interpolated), so check how well the two agree near the origin.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from largescale_dem import crop_local  # noqa: E402

    res = float(LC.LSTM.map_manager.get_lr_dem_res())
    dem, dem_origin = crop_local(LC.LSTM.map_manager.get_lr_dem(), res, ls_start, 1000.0)
    g = np.arange(-150.0, 151.0, 7.3)
    gx, gy = [v.ravel() for v in np.meshgrid(g, g)]
    hr = np.array([LC.get_terrain_height((x, y)) for x, y in zip(gx, gy)])
    fx_, fy_ = (gx - dem_origin[0]) / res, (gy - dem_origin[1]) / res
    ix, iy = fx_.astype(int), fy_.astype(int)
    dx, dy = fx_ - ix, fy_ - iy
    lr_h = (dem[iy, ix] * (1 - dx) * (1 - dy) + dem[iy, ix + 1] * dx * (1 - dy) + dem[iy + 1, ix] * (1 - dx) * dy
            + dem[iy + 1, ix + 1] * dx * dy)
    d = hr - lr_h
    dem_check = {"n": int(len(d)), "mean_m": float(d.mean()), "std_m": float(d.std()), "max_abs_m": float(np.abs(d).max())}
    print(f"RENDER HR terrain height - LR DEM (bilinear) within 150 m of the origin: {dem_check}", flush=True)
    dem_meta = {"origin_xy_m": list(map(float, dem_origin)), "source": "LargeScale LR DEM crop (see largescale_dem.py)",
                "starting_position": ls_start, "hr_minus_lr_check": dem_check}
else:
    dem = np.flip(LC.T.getDEM(), 0)  # row = y / res, col = x / res (see TerrainManager.update)
    res = float(LC.T._grid_size)
    dem_meta = {"origin_xy_m": list(map(float, LC.T._mesh_pos[:2]))}
np.save(os.path.join(a.out, "dem.npy"), dem.astype(np.float32))
json.dump({"resolution_m": res, **dem_meta, "terrain": terrain_name,
           "indexing": "dem[iy, ix] = height at world (x0 + ix*res, y0 + iy*res)"},
          open(os.path.join(a.out, "dem.json"), "w"), indent=1, default=str)
rocks = []
for prim in Usd.PrimRange(stage.GetPrimAtPath(rocks_path)) if rocks_path else []:
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
calib = {"width": W, "height": H, "fx": fx, "fy": fx, "cx": W / 2.0, "cy": H / 2.0, "baseline_m": baseline,
         "distortion": [0, 0, 0, 0], "source": str(P["source"]), "renderer": "RayTracedLighting",
         "rt_subframes": a.rt_subframes, "environment": env_name, "terrain": terrain_name, "lighting": lighting,
         "rocks": a.rocks, "n_rock_instances": len(rocks), "motion_blur": False, "sensor_noise": False,
         "depth0_unit_mm": a.depth_unit_mm}
if a.env == "largescale":
    calib["rocks"] = "cfg rock_gen_cfgs (streamed per block; not listed in rocks.json)"
    calib["largescale"] = {"lr_dem_name": ls_cfg.lr_dem_name, "starting_position": ls_start,
                           "hr_dem_resolution_m": ls_cfg.hr_dem_resolution, "block_size_m": ls_cfg.block_size,
                           "hr_dem_num_blocks": ls_cfg.hr_dem_num_blocks,
                           "update_every_n_meters": ls_cfg.update_every_n_meters, "mesh_settle": a.mesh_settle}
if a.env != "lunalab":
    calib.update({"terrain_material": a.terrain_material, "texture_scale": a.texture_scale,
                  "albedo_brightness": a.albedo_brightness, "albedo_add": a.albedo_add})
    if a.terrain_shader == "bombed":
        calib.update({"terrain_shader": "bombed (sim/omnilrs/regolith_mdl.py)", "terrain_shader_params": shader,
                      "albedo_add": None})
    # Where the sun projects in the left image (OpenCV optical frame) along the trajectory.
    d = np.einsum("nji,j->ni", T_wc[:, :3, :3], np.asarray(lighting["to_sun_world"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = fx * d[:, 0] / d[:, 2] + W / 2.0, fx * d[:, 1] / d[:, 2] + H / 2.0
    inside = (d[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    top_ray = np.array([0.0, -H / 2.0 / fx, 1.0]) / np.linalg.norm([0.0, H / 2.0 / fx, 1.0])
    ang = np.degrees(np.arccos(np.clip(d @ top_ray, -1, 1)))  # sun to top-centre-pixel ray
    calib["sun_in_frame"] = {"frames_with_sun_disk_in_left_image": int(inside.sum()), "frames": int(len(d)),
                             "min_angle_sun_to_top_centre_ray_deg": float(ang.min())}
json.dump(calib, open(os.path.join(a.out, "calib.json"), "w"), indent=1, default=str)
if preview_idx is not None:
    json.dump(preview_idx.tolist(), open(os.path.join(a.out, "preview_index.json"), "w"))


def shutdown():
    if a.env == "largescale":  # let the HR DEM monitor thread stop its worker pools (it does so when it exits)
        mon = LC.LSTM.map_manager.hr_dem_gen.monitor_thread
        mon.event.set()
        mon.thread.join(timeout=60)


def update_terrain(T):
    """LargeScale: re-centre the streamed terrain on the camera; returns (updated, seconds)."""
    if a.env != "largescale":
        return False, 0.0
    t1 = time.time()
    updated, _ = LC.LSTM.update_visual_mesh((float(T[0, 3]), float(T[1, 3])))
    if updated:
        for _ in range(a.mesh_settle):
            app.update()
    return updated, time.time() - t1


if a.dem_only:
    print(f"RENDER dem-only done; dem {dem.shape} mean {dem.mean():.3f} m, {len(rocks)} rocks", flush=True)
    shutdown()
    app.close()
    sys.exit(0)

# Warm up so the first frame is converged.
set_pose(T_wc[0])
update_terrain(T_wc[0])
for _ in range(10):
    rep.orchestrator.step(rt_subframes=a.rt_subframes, delta_time=0.0, pause_timeline=True)

from scipy.spatial.transform import Rotation  # noqa: E402

fcsv = open(os.path.join(a.out, "frames.csv"), "w")
fcsv.write("timestamp_ns,left_path,right_path\n")
ftum = open(os.path.join(a.out, "gt_cam0.tum"), "w")
if a.env == "largescale":
    fter = open(os.path.join(a.out, "terrain.csv"), "w")
    fter.write("timestamp_ns,terrain_z_m,agl_m,mesh_updated,mesh_update_s,render_s\n")
t0 = time.time()
t_mesh = 0.0
for i, (t, T) in enumerate(zip(t_ns, T_wc)):
    set_pose(T)
    updated, dt_mesh = update_terrain(T)
    t_mesh += dt_mesh
    t1 = time.time()
    rep.orchestrator.step(rt_subframes=a.rt_subframes, delta_time=0.0, pause_timeline=True)
    for k, r in enumerate(rgb):
        img = np.asarray(r.get_data())[..., :3]
        cv2.imwrite(os.path.join(a.out, f"cam{k}", f"{t}.png"), img[..., ::-1])
    z = np.asarray(depth.get_data())
    z = z * (1000.0 / a.depth_unit_mm)
    z = np.where(np.isfinite(z) & (z > 0) & (z < 65535.0), z, 0).astype(np.uint16)
    cv2.imwrite(os.path.join(a.out, "depth0", f"{t}.png"), z)
    fcsv.write(f"{t},cam0/{t}.png,cam1/{t}.png\n")
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    ftum.write(f"{t * 1e-9:.9f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
    if a.env == "largescale":
        hz = float(LC.get_terrain_height((float(T[0, 3]), float(T[1, 3]))))
        fter.write(f"{t},{hz:.4f},{T[2, 3] - hz:.4f},{int(updated)},{dt_mesh:.4f},{time.time() - t1:.4f}\n")
    if i % 100 == 0 or (a.env == "largescale" and i % 25 == 0):
        print(f"RENDER {i}/{len(t_ns)} {(time.time() - t0) / (i + 1):.3f}s/frame (terrain updates "
              f"{t_mesh / (i + 1):.3f}s/frame)", flush=True)
fcsv.close()
ftum.close()
if a.env == "largescale":
    fter.close()
print(f"RENDER done {len(t_ns)} frames in {time.time() - t0:.1f}s; total {time.time() - t_start:.1f}s", flush=True)
rep.orchestrator.wait_until_complete()
shutdown()
app.close()
