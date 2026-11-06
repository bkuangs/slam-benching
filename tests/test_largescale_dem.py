"""Tests for the OmniLRS LargeScale DEM frame mapping (sim/omnilrs/largescale_dem.py) used by the aerial trajectories."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim" / "omnilrs"))
from aerial_poses import Dem, gaussian_smooth  # noqa: E402
from largescale_dem import crop_local, lr_from_raw  # noqa: E402


def test_lr_from_raw_orientation():
    # GeoTIFF row 0 is north: lr[i, j] must increase with x (columns) and with y (towards row 0).
    raw = np.add.outer(-10.0 * np.arange(4), np.arange(5))  # height = col - 10 * row  (x east, y north)
    lr = lr_from_raw(raw)
    assert lr.shape == (5, 4)
    assert np.all(np.diff(lr, axis=0) == 1) and np.all(np.diff(lr, axis=1) == 10)


def test_crop_local_matches_global_mapping():
    res, nx, ny = 5.0, 40, 30
    gx, gy = res * (np.arange(nx) - nx // 2), res * (np.arange(ny) - ny // 2)
    lr = np.add.outer(gx, 1000.0 * gy)                                # height = gx + 1000 gy (global metres)
    start = (15.0, -20.0)
    dem, origin = crop_local(lr, res, start, half_extent_m=30.0)
    d = Dem(z=dem, res=res, origin=origin)
    xy = np.array([[0.0, 0.0], [12.5, -7.5], [-20.0, 25.0]])          # local; global = local + start
    expect = (xy[:, 0] + start[0]) + 1000.0 * (xy[:, 1] + start[1])
    np.testing.assert_allclose(d.height(xy), expect, atol=1e-6)


def test_crop_local_clips_at_dem_edge():
    lr = np.zeros((20, 20))
    dem, origin = crop_local(lr, 5.0, (40.0, 0.0), half_extent_m=30.0)  # centre at index 18: only 1 column right
    assert dem.shape == (13, 8) and origin == (5.0 * (12 - 10) - 40.0, 5.0 * (4 - 10))


def test_gaussian_smooth_keeps_slope_and_ends():
    v = np.linspace(0.0, 10.0, 101)
    np.testing.assert_allclose(gaussian_smooth(v, 5.0)[20:-20], v[20:-20], atol=1e-9)  # linear trend preserved
    step = np.r_[np.zeros(50), np.ones(50)]
    s = gaussian_smooth(step, 3.0)
    assert len(s) == 100 and abs(s[0]) < 1e-12 and abs(s[-1] - 1) < 1e-12 and 0.4 < s[50] < 0.6
