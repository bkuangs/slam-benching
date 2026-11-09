import numpy as np
import pytest

pytest.importorskip("rerun")

from slambench.viz.map_reuse_export import _quat_to_R, read_ply_vertices  # noqa: E402

HEADER = ("ply\nformat {fmt} 1.0\ncomment test\nelement vertex {n}\nproperty float x\nproperty float y\n"
          "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
          "property float nx\nelement camera 1\nproperty float view_px\nend_header\n")


def _points(n=50):
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(n, 3)).astype(np.uint8)
    return xyz, rgb


def test_binary_ply_vertices_and_trailing_elements(tmp_path):
    xyz, rgb = _points()
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("nx", "<f4")])
    v = np.zeros(len(xyz), dt)
    for k, c in enumerate("xyz"):
        v[c] = xyz[:, k]
    for k, c in enumerate(("red", "green", "blue")):
        v[c] = rgb[:, k]
    v["x"][3] = np.nan  # dropped
    p = tmp_path / "c.ply"
    with open(p, "wb") as f:
        f.write(HEADER.format(fmt="binary_little_endian", n=len(xyz)).encode())
        f.write(v.tobytes())
        f.write(np.float32(7).tobytes())  # camera element after the vertices
    out, col = read_ply_vertices(p)
    keep = np.arange(len(xyz)) != 3
    assert np.allclose(out, xyz[keep]) and (col == rgb[keep]).all()
    sub, subc = read_ply_vertices(p, max_points=10)
    assert len(sub) == 10 and len(subc) == 10


def test_ascii_ply_vertices(tmp_path):
    xyz, rgb = _points(5)
    p = tmp_path / "a.ply"
    lines = [" ".join([*(f"{x:.6f}" for x in a), *(str(int(c)) for c in b), "0"]) for a, b in zip(xyz, rgb)]
    p.write_text(HEADER.format(fmt="ascii", n=5) + "\n".join(lines) + "\n7\n")
    out, col = read_ply_vertices(p)
    assert np.allclose(out, xyz, atol=1e-5) and (col == rgb).all()


def test_quaternion_optical_axis():
    # cam0 optical frame rotated into an x-forward base: optical z must point along +x
    q = np.array([[-0.5, 0.5, -0.5, 0.5]])
    assert np.allclose(_quat_to_R(q)[0][:, 2], [1, 0, 0], atol=1e-9)


# ---- ORB-SLAM3 / SP_SLAM3 localization run (synthetic map run + query run)

T0 = 1_750_000_000_000_000_000
DT = 100_000_000  # 10 Hz


def _write_csv(path, header, rows):
    path.write_text(",".join(header) + "\n" + "".join(",".join(str(x) for x in r) + "\n" for r in rows))


def _sec(t_ns):
    return f"{t_ns // 10**9}.{t_ns % 10**9:09d}"


def _prepared(d, ts):
    import cv2
    (d / "cam0").mkdir(parents=True)
    for t in ts:
        cv2.imwrite(str(d / "cam0" / f"{t}.png"), np.full((24, 32), 100, np.uint8))
    _write_csv(d / "frames.csv", ["timestamp_ns", "left", "right"], [(t, f"cam0/{t}.png", "") for t in ts])


def _spslam3_runs(tmp_path):
    import json
    import yaml
    q = np.array([0, 0, 0, 1.0])
    # mapping run: keyframes 10/11/12 at x = 0/1/2 (stamps off by ~100 ns, as after a double), map->GT = +5 m in x
    m = tmp_path / "map_run"
    (m / "algorithm").mkdir(parents=True)
    (m / "eval").mkdir()
    map_ts = [T0 + k * DT for k in range(20, 23)]
    _prepared(tmp_path / "prep_map", map_ts)
    (m / "prepared_input.json").write_text(json.dumps({"directory": str(tmp_path / "prep_map")}))
    _write_csv(m / "algorithm" / "keyframes.csv",
               ["map_id", "keyframe_id", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "in_final_frame"],
               [(0, 10 + k, t - 96, float(k), 0, 0, *q, 1) for k, t in enumerate(map_ts)])
    _write_csv(m / "algorithm" / "map_points.csv", ["map_id", "point_id", "x", "y", "z", "n_observations",
                                                    "in_final_frame"],
               [(0, k, k, 1, 0.1 * k, 3, int(k < 4)) for k in range(5)])
    (m / "eval" / "metrics.json").write_text(json.dumps({"run_id": "map", "sequence": "s_map", "accuracy": {"final": {
        "sessions": [{"alignment": {"type": "se3", "R": np.eye(3).tolist(), "t": [5.0, 0, 0]}}],
        "ate": {"translation_m": {"rmse": 0.05}}, "ground_truth": {}}}}))
    # query run: lost for 4 frames, learned relocalization on frame 4 against keyframe 11, anchor on frame 8
    r = tmp_path / "query_run"
    (r / "algorithm").mkdir(parents=True)
    ts = [T0 + k * DT for k in range(10)]
    _prepared(tmp_path / "prep_q", ts)
    (r / "prepared_input.json").write_text(json.dumps({"directory": str(tmp_path / "prep_q")}))
    (r / "config.resolved.yaml").write_text(yaml.safe_dump({
        "name": "q", "dataset": {"name": "no_such_dataset", "sequence": "s_q"},
        "algorithm": {"name": "spslam3", "settings": {"learned_reloc_min_inliers": 50}}}))
    (r / "status.json").write_text(json.dumps({"status": "success", "run_id": "q"}))
    (r / "algorithm_output.json").write_text(json.dumps({"extra": {"map_reuse": {
        "map_run": str(m), "reuse_mode": "relocalize"}}}))
    (r / "algorithm" / "loaded_map.json").write_text(json.dumps({"T_final_map": np.eye(4).tolist()}))
    log = []
    for k, t in enumerate(ts):
        pose = [""] * 7 if k < 4 else [0.25 * k, 0, 0, *q]
        log.append((k, t, 4 if k < 4 else 2, 0, 0, 0.01, 0, 3, 4, *pose, 1, int(k == 4), int(k == 7)))
    _write_csv(r / "algorithm" / "frame_log.csv",
               ["frame_idx", "timestamp_ns", "state", "map_id", "map_changed", "track_time_s", "coord_session",
                "map_kfs", "map_points", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "in_loaded_map", "relocalized",
                "vo_only"], log)
    lr_cols = ["timestamp", "frame_id", "mode", "cand_kf_ids", "best_rank", "best_kf_id", "best_inliers",
               "accepted", "confirmed"]
    _write_csv(r / "algorithm" / "learned_reloc.csv", lr_cols,
               [(_sec(ts[3]), 503, "lost", "12;10", -1, -1, -1, 0, 0),
                (_sec(ts[4]), 504, "lost", "12;11;10", 1, 11, 63, 1, 1)])
    _write_csv(r / "algorithm" / "map_anchor.csv",
               ["timestamp", "frame_id", "kf_ids", "lg_matches", "mp_matches", "pose_inliers", "local_inliers",
                "correction_m", "correction_deg", "accepted", "ms"],
               [(_sec(ts[6]), 506, "10;11;12", 90, 40, 8, 20, 0.5, 2.0, 0, 50),
                (_sec(ts[8]), 508, "12;11;10", 150, 70, 30, 71, 0.17, 3.08, 1, 94)])
    return r


def test_spslam3_relocalization_replay(tmp_path):
    import rerun as rr

    from slambench.viz.map_reuse_export import export_map_reuse

    run = _spslam3_runs(tmp_path)
    rep = export_map_reuse(run, stride=1, log=lambda *_: None)
    assert (rep["map_nodes"], rep["map_cloud_points"], rep["n_localized_poses"]) == (3, 4, 6)
    assert (rep["n_accepted"], rep["n_anchors"], rep["n_images"], rep["ground_truth"]) == (1, 1, 10, False)

    rec = rr.dataframe.load_recording(rep["out"])

    def column(entity, comp, static=False):
        v = rec.view(index="frame", contents=entity)
        t = (v.select_static() if static else v.select()).read_all().to_pandas()
        c = next(c for c in t.columns if c.endswith(comp))
        return t["frame"].tolist() if "frame" in t else None, t[c].tolist()

    _, txt = column("/events/accepted", "TextLog:text")
    assert [x for x in txt if x is not None] == [
        ["FIRST localization (learned relocalization: MixVPR rank 2 of 3, 63 inliers) against map keyframe 11, "
         "error after n/a"]]
    _, anc = column("/events/anchor", "TextLog:text")
    assert "keyframes 12, 11, 10" in [x for x in anc if x is not None][0][0]
    # poses and the matched keyframe in the GT world with the mapping run's transform (+5 m in x)
    _, pts = column("/world/query/localized", "Points3D:positions", True)
    xs = sorted(p[0] for p in np.concatenate([x for x in pts if x is not None]))
    assert np.allclose(xs, [5 + 0.25 * k for k in range(4, 10)])
    _, line = column("/world/query/accepted_match", "LineStrips3D:strips", True)
    seg = np.asarray(next(x for x in line if x is not None)[0].tolist())
    assert np.allclose(seg, [[6.0, 0, 0], [6.0, 0, 0]])
    # map image: the verified keyframe right after the relocalization, the anchor's keyframe after the anchor
    fr, lab = column("/camera/map/label", "Points2D:labels")
    lab = {f: x[0] for f, x in zip(fr, lab) if x is not None}
    assert lab[4] == "map keyframe 11: matched by the last accepted localization"
    assert lab[8] == "map keyframe 12: matched by the last map anchor"
    _, inl = column("/plots/counts/reloc_best_inliers", "Scalars:scalars")
    assert sorted(float(x[0]) for x in inl if x is not None) == [0.0, 63.0]
