"""Low-resolution DEM of the OmniLRS LargeScale environment in the simulator's local world frame (pure numpy).

OmniLRS (MapManager.load) loads the GeoTIFF-derived `dem.npy` (row = image row, north up) as `lr = flip(dem.T, 1)`,
so `lr[i, j]` is the height at global (x, y) = (res * (i - n_x // 2), res * (j - n_y // 2)) metres from the DEM
centre (x = easting, y = northing; HighResDEMGen.querry_low_res_dem). The LargeScale scene places
`starting_position` (global) at the local / world origin, and the terrain z is the absolute DEM elevation.
`crop_local` returns a crop in the same layout as render_stereo.py's dem.npy/dem.json
(dem[iy, ix] = height at world (x0 + ix*res, y0 + iy*res)), which aerial_poses.py reads.
"""
import numpy as np


def lr_from_raw(raw):
    """OmniLRS MapManager.load: raw GeoTIFF array (row 0 = north) -> lr[i = x index, j = y index]."""
    return np.flip(np.asarray(raw).T, axis=1)


def crop_local(lr, res, start_xy, half_extent_m, centre_xy=(0.0, 0.0)):
    """Crop of the LR DEM around local `centre_xy`, as (dem[iy, ix], origin_xy) in the local world frame."""
    nx, ny = lr.shape
    n = int(np.ceil(half_extent_m / res))
    # local x = res * (i - nx // 2) - start_x  ->  i = (x + start_x) / res + nx // 2
    ic = int(round((centre_xy[0] + start_xy[0]) / res)) + nx // 2
    jc = int(round((centre_xy[1] + start_xy[1]) / res)) + ny // 2
    i0, i1 = max(ic - n, 0), min(ic + n + 1, nx)
    j0, j1 = max(jc - n, 0), min(jc + n + 1, ny)
    dem = np.ascontiguousarray(lr[i0:i1, j0:j1].T).astype(np.float32)
    origin = (res * (i0 - nx // 2) - start_xy[0], res * (j0 - ny // 2) - start_xy[1])
    return dem, origin
