"""OmniLRS rendered-sequence dataset adapter."""
import json

import numpy as np

from slambench.datasets import DATASETS
from slambench.types import SensorMode


def _write_seq(root, n=5, baseline=0.12, **calib):
    d = root / "seq"
    for cam in ("cam0", "cam1"):
        (d / cam).mkdir(parents=True)
    t = 1_750_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 68_000_000
    with open(d / "frames.csv", "w") as f, open(d / "gt_cam0.tum", "w") as g:
        f.write("timestamp_ns,left_path,right_path\n")
        for i, ti in enumerate(t):
            for cam in ("cam0", "cam1"):
                (d / cam / f"{ti}.png").write_bytes(b"png")
            f.write(f"{ti},cam0/{ti}.png,cam1/{ti}.png\n")
            g.write(f"{ti * 1e-9:.9f} {0.1 * i:.6f} 2.0 0.4 0 0 0 1\n")
    (d / "calib.json").write_text(json.dumps({"width": 640, "height": 360, "fx": 263.27, "fy": 263.27, "cx": 320.0,
                                              "cy": 180.0, "baseline_m": baseline, "source": "test", **calib}))
    return t


def test_omnilrs_calibration_gt_and_prepare(tmp_path):
    t = _write_seq(tmp_path)
    ds = DATASETS["omnilrs"](tmp_path, "seq")
    assert ds.validate(SensorMode.STEREO) == []
    assert ds.validate(SensorMode.STEREO_INERTIAL)  # no IMU
    info = ds.sequence_info()
    assert info.streams["cam0"].count == 5 and info.has_ground_truth

    cal = ds.stereo_calibration()
    assert cal.left.cx == 320.0 and cal.baseline_m == 0.12 and cal.T_left_right[0, 3] == 0.12

    gt = ds.ground_truth()
    np.testing.assert_array_equal(gt.timestamps_ns, t)  # exact ns recovered from frames.csv
    assert gt.body_frame == "cam0" and np.allclose(ds.T_gt_body_sensor("cam0"), np.eye(4))
    assert np.isclose(gt.positions[4, 0], 0.4)

    prep = ds.prepare_cached(SensorMode.STEREO, tmp_path / "prepared")
    assert prep.n_frames == 5
    assert (prep.directory / "cam1" / f"{t[2]}.png").read_bytes() == b"png"
    assert prep.manifest["hardlinked_files"] + prep.manifest["copied_files"] == 10
    lines = (prep.directory / "frames.csv").read_text().splitlines()
    assert lines[1] == f"{t[0]},cam0/{t[0]}.png,cam1/{t[0]}.png"
    assert not (prep.directory / "gt_cam0.tum").exists() and not (prep.directory / "depth0").exists()


def test_omnilrs_environment_from_calib(tmp_path):
    _write_seq(tmp_path)
    ds = DATASETS["omnilrs"](tmp_path, "seq")
    assert ds.ground_truth().reference_frame == "lunalab_world"  # older renders carry no "environment"

    _write_seq(tmp_path / "yard", environment="lunaryard_80m", lighting={"sun_elevation_deg": 7.0})
    ds = DATASETS["omnilrs"](tmp_path / "yard", "seq")
    assert ds.ground_truth().reference_frame == "lunaryard_80m_world"
    assert "lunaryard_80m" in ds.sequence_info().notes[0]
    assert ds.provenance()["render"]["lighting"]["sun_elevation_deg"] == 7.0


def test_prepare_cached_concurrent_jobs(tmp_path):
    """Two jobs preparing the same sequence at once: the later one uses the earlier one's result."""
    _write_seq(tmp_path)
    cache = tmp_path / "prepared"
    first = DATASETS["omnilrs"](tmp_path, "seq")
    original = first.prepare_offline
    other = {}

    def racing_prepare(mode, out_dir):
        other["prep"] = DATASETS["omnilrs"](tmp_path, "seq").prepare_cached(mode, cache)  # finishes first
        return original(mode, out_dir)

    first.prepare_offline = racing_prepare
    prep = first.prepare_cached(SensorMode.STEREO, cache)
    assert prep.directory == other["prep"].directory and prep.n_frames == 5
    assert [p.name for p in prep.directory.parent.iterdir()] == [prep.directory.name]  # no staging dirs left
