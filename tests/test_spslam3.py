"""SP_SLAM3 adapter tests: settings validation, generated settings file, command and collection (no image)."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from slambench.algorithms import ALGORITHMS
from slambench.algorithms import spslam3 as sp
from slambench.algorithms.base import RunContext
from slambench.algorithms.orbslam3 import OrbSlam3Adapter
from slambench.algorithms.spslam3 import SpSlam3Adapter
from slambench.trajectory import read_tum
from slambench.types import CameraIntrinsics, RunStatus, SensorMode, StereoCalibration

from synthetic import FakeDataset, sample

REPO = Path(__file__).resolve().parents[1]


def _ctx(tmp_path, settings=None, adapter=SpSlam3Adapter):
    ds = FakeDataset(tmp_path, "s1")
    (tmp_path / "prep").mkdir(exist_ok=True)
    prep = ds.prepare_offline(SensorMode.STEREO, tmp_path / "prep")
    a = adapter({"container": str(tmp_path / "x.sif")})
    return a, ds, RunContext(tmp_path, tmp_path / "algo", ds, SensorMode.STEREO, prep,
                             a.resolve_settings(settings or {}))


def test_registered_stereo_only_and_validation(tmp_path):
    assert ALGORITHMS["spslam3"] is SpSlam3Adapter
    a, ds, ctx = _ctx(tmp_path)
    s = ctx.settings
    for mode in (SensorMode.MONO, SensorMode.STEREO_INERTIAL, SensorMode.LIDAR_INERTIAL):
        assert "does not support sensor mode" in a.validate(mode, ds, s)[0]
    assert a.validate(SensorMode.STEREO, ds, s) == [f"SP_SLAM3 container not found: {tmp_path / 'x.sif'} "
                                                    "(build with third_party/spslam3/build_container.sh)"]
    (tmp_path / "x.sif").write_bytes(b"")
    assert a.validate(SensorMode.STEREO, ds, s) == []

    class Fisheye(FakeDataset):
        def stereo_calibration(self):
            c = CameraIntrinsics("kannala_brandt", 640, 360, 260.0, 260.0, 320.0, 180.0, [0.1, 0, 0, 0], False)
            return StereoCalibration(c, c, np.eye(4), 0.12, "fake")
    assert "rectified pinhole stereo only" in a.validate(SensorMode.STEREO, Fisheye(tmp_path, "s1"), s)[0]
    assert a.estimate_frame(SensorMode.STEREO) == "cam0"


def test_defaults_and_rejected_settings(tmp_path):
    a = SpSlam3Adapter({"container": str(tmp_path / "x.sif")})
    s = a.resolve_settings({})
    assert (s["sp_n_features"], s["sp_n_levels"], s["sp_ini_th"], s["sp_min_th"]) == (1200, 1, 0.155, 0.055)
    assert s["lightglue"] is True and s["sp_fp16"] is False and s["lightglue_fp16"] is False
    assert s["loop_closure"] is True and s["stereo_th_depth"] == 40.0
    for bad in ({"orb_n_features": 1200}, {"loop_closure": False}, {"sp_n_features": 0},
                {"sp_n_features": 100.5}, {"sp_n_levels": 0}, {"sp_scale_factor": 1.0},
                {"sp_ini_th": 0.05, "sp_min_th": 0.1}, {"sp_min_th": 0}, {"sp_ini_th": 1.0},
                {"stereo_th_depth": 0}, {"clahe_clip_limit": 0}, {"clahe_tile_grid": 2.5},
                {"playback_rate": -1}):
        with pytest.raises(ValueError):
            a.resolve_settings(bad)


def test_settings_file_is_old_format_with_superpoint_keys(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    a.prepare(ctx)
    text = (tmp_path / "algo" / "spslam3_settings.yaml").read_text()
    assert text.startswith("%YAML:1.0\n")
    fs = cv2.FileStorage(str(tmp_path / "algo" / "spslam3_settings.yaml"), cv2.FILE_STORAGE_READ)
    assert fs.isOpened()
    assert fs.getNode("Camera.type").string() == "PinHole"
    # Tracking::ParseCamParamFile / ParseORBParamFile reject non-real values for these keys.
    for key, val in (("Camera.fx", 260.0), ("Camera.fy", 260.0), ("Camera.cx", 320.0), ("Camera.cy", 180.0),
                     ("Camera.k1", 0.0), ("Camera.k2", 0.0), ("Camera.p1", 0.0), ("Camera.p2", 0.0),
                     ("Camera.bf", 260.0 * 0.12), ("ThDepth", 40.0), ("ORBextractor.scaleFactor", 1.2),
                     ("ORBextractor.iniThFAST", 0.155), ("ORBextractor.minThFAST", 0.055),
                     ("Viewer.KeyFrameSize", 0.05), ("Viewer.ViewpointF", 500.0)):
        node = fs.getNode(key)
        assert node.isReal(), key
        assert node.real() == pytest.approx(val), key
    for key, val in (("ORBextractor.nFeatures", 1200), ("ORBextractor.nLevels", 1), ("Camera.width", 640),
                     ("Camera.height", 360), ("Camera.fps", 15), ("Camera.RGB", 0), ("SuperPoint.useFP16", 0),
                     ("LightGlue.useFP16", 0)):
        assert fs.getNode(key).isInt() and int(fs.getNode(key).real()) == val, key
    assert fs.getNode("LightGlue.model_path").string() == "/opt/SP_SLAM3/lightglue.pt"
    assert "ORBextractor.iniThFAST: 0.155\n" in text and "ORBextractor.minThFAST: 0.055\n" in text
    # no new-format keys, no loopClosing switch, no place-recognition model (DBoW3 is used)
    for key in ("File.version", "Stereo.b", "Stereo.ThDepth", "Camera1.fx", "loopClosing",
                "PlaceRecognition.model_path"):
        assert fs.getNode(key).empty(), key

    ctx.settings.update(a.resolve_settings({"lightglue": False, "sp_n_features": 2000, "sp_n_levels": 3,
                                            "camera_fps": 30}))
    a.prepare(ctx)
    text = (tmp_path / "algo" / "spslam3_settings.yaml").read_text()
    assert "LightGlue" not in text
    assert "ORBextractor.nFeatures: 2000\n" in text and "ORBextractor.nLevels: 3\n" in text
    assert "Camera.fps: 30\n" in text

    # very low SuperPoint thresholds (LunaPolaris configs) are written positionally and read back as reals
    ctx.settings.update(a.resolve_settings({"sp_ini_th": 1e-5, "sp_min_th": 1e-6}))
    a.prepare(ctx)
    text = (tmp_path / "algo" / "spslam3_settings.yaml").read_text()
    assert "ORBextractor.iniThFAST: 0.00001\n" in text and "ORBextractor.minThFAST: 0.000001\n" in text
    fs = cv2.FileStorage(str(tmp_path / "algo" / "spslam3_settings.yaml"), cv2.FILE_STORAGE_READ)
    for key, val in (("ORBextractor.iniThFAST", 1e-5), ("ORBextractor.minThFAST", 1e-6)):
        assert fs.getNode(key).isReal() and fs.getNode(key).real() == pytest.approx(val), key


def test_command_uses_gpu_driver_and_vocabulary(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    cmd = a.command(ctx)
    assert cmd[1:4] == ["exec", "--nv", "--cleanenv"]
    i = cmd.index(str(tmp_path / "x.sif"))
    assert cmd[i + 1:i + 4] == ["/opt/slambench/bin/spslam3_driver", "/opt/SP_SLAM3/Vocabulary/superpoint_voc.dbow3",
                                str((tmp_path / "algo" / "spslam3_settings.yaml").resolve())]
    assert cmd[i + 4] == "stereo"
    assert cmd[cmd.index("--playback-rate") + 1] == "1.0" and "--log-features" in cmd and "--clahe" not in cmd
    for b in a._binds(ctx):
        assert cmd[cmd.index(b) - 1] == "-B"
    ctx.settings.update(a.resolve_settings({"clahe": True, "clahe_clip_limit": 2.0, "log_features": False,
                                            "playback_rate": 0}))
    cmd = a.command(ctx)
    assert cmd[cmd.index("--clahe-clip") + 1] == "2.0" and "--log-features" not in cmd
    assert cmd[cmd.index("--playback-rate") + 1] == "0.0"


def test_driver_source_hash_covers_driver_preprocess_and_patches():
    h = sp.driver_source_sha1()
    assert h and len(h) == 40
    patches = sorted((REPO / "third_party" / "spslam3" / "patches").glob("*.patch"))
    assert len(patches) >= 3
    import hashlib
    ref = hashlib.sha1()
    for f in [REPO / "third_party/spslam3/slambench_spslam3_driver.cc",
              REPO / "third_party/orbslam3/slambench_preprocess.h", *patches]:
        ref.update(f.read_bytes())
    assert h == ref.hexdigest()
    definition = (REPO / "third_party" / "spslam3" / "spslam3.def").read_text()
    assert ("cat /opt/slambench/src/slambench_spslam3_driver.cc /opt/slambench/src/slambench_preprocess.h "
            "$(LC_ALL=C ls /opt/slambench/patches/*.patch)") in definition
    assert sp.SPSLAM3_COMMIT in definition


def test_version_info_rejects_other_commit_or_driver(tmp_path, monkeypatch):
    a = SpSlam3Adapter({"container": str(tmp_path / "x.sif"), "apptainer": "apptainer"})

    def fake_run(versions):
        class R:
            returncode = 0
            stdout = versions

        def run(cmd, **kw):
            if cmd[0] == "nvidia-smi":
                raise FileNotFoundError("nvidia-smi")
            return R()
        return run

    good = f"spslam3_commit={sp.SPSLAM3_COMMIT}\ndriver_source_sha1={sp.driver_source_sha1()}\n"
    monkeypatch.setattr(sp.subprocess, "run", fake_run(good))
    info = a.version_info()
    assert info["container_versions"]["spslam3_commit"] == sp.SPSLAM3_COMMIT
    monkeypatch.setattr(sp.subprocess, "run", fake_run("spslam3_commit=deadbeef\n"))
    with pytest.raises(RuntimeError, match="commit"):
        a.version_info()
    monkeypatch.setattr(sp.subprocess, "run", fake_run(f"spslam3_commit={sp.SPSLAM3_COMMIT}\n"
                                                        "driver_source_sha1=0000\n"))
    with pytest.raises(RuntimeError, match="driver source"):
        a.version_info()


def test_collect_labels_trajectories_as_spslam3(tmp_path):
    a, ds, ctx = _ctx(tmp_path)
    d = ctx.algo_dir
    d.mkdir(parents=True)
    (ctx.run_dir / "stdout.log").write_text("LightGlue model loaded\n")
    gt = sample(ds.gt, ds.frames)
    cols = ["frame_idx", "timestamp_ns", "state", "map_id", "track_time_s", "coord_session",
            "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    with open(d / "frame_log.csv", "w", newline="") as f, open(d / "trajectory_euroc.txt", "w") as g:
        w = csv.writer(f)
        w.writerow(cols)
        for i, t in enumerate(ds.frames):
            st = 1 if i < 3 else 2
            p, q = gt.positions[i], gt.quaternions_xyzw[i]
            w.writerow([i, t, st, 0, 0.02, 0, *(list(p) + list(q) if st == 2 else [""] * 7)])
            if st == 2:
                g.write(f"{float(t):.6f} " + " ".join(f"{v:.9f}" for v in [*p, *q]) + "\n")
    (d / "driver_status.json").write_text(json.dumps({"status": "completed", "message": "",
                                                      "cuda_available": True}))
    out = a.collect(ctx, 0, False)
    assert out.status == RunStatus.SUCCESS, out.message
    assert out.extra["n_frames_tracking_ok"] == len(ds.frames) - 3
    assert out.extra["driver_status"]["cuda_available"] is True
    assert {t.kind for t in out.trajectories} == {"online", "final"}
    for name in ("trajectory.json", "trajectory_online.json"):
        meta = json.loads((ctx.run_dir / name).read_text())
        assert meta["estimator"] == "SP_SLAM3", name
    assert len(read_tum(ctx.run_dir / "trajectory.tum").timestamps_ns) == len(ds.frames) - 3


def test_orbslam3_settings_unchanged_by_shared_code(tmp_path):
    """The ORB-SLAM3 adapter is reused by SP_SLAM3; its generated settings must not change."""
    a, ds, ctx = _ctx(tmp_path, adapter=OrbSlam3Adapter)
    a.prepare(ctx)
    text = (tmp_path / "algo" / "orbslam3_settings.yaml").read_text()
    assert text == (
        '%YAML:1.0\n# Generated by slam-benching from dataset calibration; do not edit.\nFile.version: "1.0"\n'
        'Camera.type: "Rectified"\nCamera1.fx: 260.0\nCamera1.fy: 260.0\nCamera1.cx: 320.0\nCamera1.cy: 180.0\n'
        "Camera.width: 640\nCamera.height: 360\nCamera.fps: 15\nCamera.RGB: 0\nStereo.b: 0.12\n"
        "Stereo.ThDepth: 40.0\nORBextractor.nFeatures: 1200\nORBextractor.scaleFactor: 1.2\n"
        "ORBextractor.nLevels: 8\nORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\nloopClosing: 1\n"
        "Viewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 0.9\n"
        "Viewer.PointSize: 2.0\nViewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3.0\nViewer.ViewpointX: 0.0\n"
        "Viewer.ViewpointY: -0.7\nViewer.ViewpointZ: -1.8\nViewer.ViewpointF: 500.0\n")
    assert a.estimator_label == "ORB-SLAM3"


@pytest.mark.parametrize("seq", ["01", "11"])
def test_configs_match_orbslam3_lean_baselines(seq):
    sp_cfg = yaml.safe_load((REPO / f"configs/lunapolaris_seq{seq}_spslam3_stereo.yaml").read_text())
    orb = yaml.safe_load((REPO / f"configs/lunapolaris_seq{seq}_orbslam3_stereo_lean.yaml").read_text())
    for k in ("dataset", "evaluation", "execution", "output_root", "work_dir"):
        assert sp_cfg[k] == orb[k], k
    assert sp_cfg["algorithm"]["name"] == "spslam3" and sp_cfg["algorithm"]["mode"] == "stereo"
    s, o = sp_cfg["algorithm"]["settings"], orb["algorithm"]["settings"]
    assert s["sp_n_features"] == o["orb_n_features"]
    for k in ("stereo_th_depth", "loop_closure", "clahe", "playback_rate", "log_features"):
        assert s[k] == o[k], k
    SpSlam3Adapter({}).resolve_settings(s)
