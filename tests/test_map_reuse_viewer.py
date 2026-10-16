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
