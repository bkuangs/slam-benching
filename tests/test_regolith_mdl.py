import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim", "omnilrs"))
import regolith_mdl  # noqa: E402

WL = (48.0, 23.0, 11.0, 5.3)


def test_brightness_normalization_gives_requested_std():
    norm = 0.5 / regolith_mdl.fbm_std(WL)
    # a different area than the calibration sample
    x, y = np.meshgrid(np.arange(5000.0, 6000.0, 1.0), np.arange(-3000.0, -2000.0, 1.0))
    m = 1 + np.clip(0.2 * norm * regolith_mdl.fbm_np(x, y, WL), -0.3, 0.3)
    assert abs(m.mean() - 1) < 0.02
    assert 0.08 < m.std() < 0.12


def test_noise_is_smooth_and_not_periodic_at_the_texture_period():
    x = np.arange(0.0, 400.0, 0.05)
    f = regolith_mdl.fbm_np(x, np.zeros_like(x), WL)
    assert np.abs(np.diff(f)).max() < 0.1  # continuous
    shift = int(2.0 / 0.05)  # the 2 m period of the OmniPBR regolith
    assert np.corrcoef(f[:-shift], f[shift:])[0, 1] < 0.999
    assert not np.allclose(f[:4000], f[4000:8000])


def test_mdl_text_bakes_parameters():
    t = regolith_mdl.mdl_text(name="X1", cell_m=1.5, bright_amp=0.1, offset_xy=(2800, -2200), fbm_norm=0.9)
    assert "export material X1(" in t
    assert "float cell_m = 1.5f" in t and "float bright_amp = 0.1f" in t and "float bright_norm = 0.9f" in t
    assert "float2(2800.0f, -2200.0f)" in t
    assert t.count("{") == t.count("}") and t.count("(") == t.count(")")
    assert t.count("vnoise(p /") == len(WL)
