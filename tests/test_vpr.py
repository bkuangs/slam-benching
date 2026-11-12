"""Pure-python parts of the VPR benchmark (vpr/metrics.py, vpr/dbow.py, vpr/slam_matching.py, vpr/sessions.py);
no torch needed."""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vpr import metrics as M  # noqa: E402
from vpr import slam_matching as SM  # noqa: E402
from vpr.dbow import DBOW3_MAGIC, BowDatabase, Vocabulary, bow_from_words, hamming, l1_score  # noqa: E402
from vpr.sessions import load_session, write_session  # noqa: E402


def pose(x, yaw_deg=0.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = [x, 0, 0]
    return T


def test_rot_angle_and_positives():
    db = np.stack([pose(0), pose(1), pose(1, 40), pose(5)])
    q = np.stack([pose(0.9, 10), pose(10)])
    assert np.isclose(M.rot_angle_deg(db[2, :3, :3], db[1, :3, :3]), 40)
    pos = M.positives(q, db, 2.0, 30.0)
    assert pos.tolist() == [[True, True, False, False], [False, False, False, False]]


def test_recall_at_k_ignores_queries_without_positive():
    pos = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], bool)
    ranking = np.array([[1, 0, 2], [2, 1, 0], [0, 1, 2]])
    r = M.recall_at_k(ranking, pos, (1, 2, 3))
    assert r["n_with_positive"] == 2 and r["n_queries"] == 3
    assert r["R@1"] == 0.5 and r["R@2"] == 0.5 and r["R@3"] == 1.0


def test_recall_nan_without_positives():
    r = M.recall_at_k(np.zeros((2, 1), int), np.zeros((2, 3), bool), (1,))
    assert np.isnan(r["R@1"])


def test_rank_and_oracle():
    sim = np.array([[0.1, 0.9, 0.5], [0.3, 0.3, 0.2]])
    assert M.rank_by_similarity(sim, 2).tolist() == [[1, 2], [0, 1]]
    db = np.stack([pose(0), pose(0.5, 90), pose(1)])
    o = M.oracle_ranking(np.stack([pose(0.4)]), db, 30.0, 3)
    assert o.tolist() == [[0, 2, 1]]  # the closer frame 1 looks the wrong way


def test_subsample_by_motion_and_query_shift():
    T = np.stack([pose(0.05 * i) for i in range(20)] + [pose(0.95, 20)])
    valid = np.ones(len(T), bool)
    valid[4] = False
    kept = M.subsample_by_motion(T, valid, 0.2, 15.0)
    assert kept[:3] == [0, 5, 9] and kept[-1] == 20  # 4 has no GT; the last frame turned 20 deg
    assert M.subsample_by_motion(T, valid, 0.2, 15.0, min_gap=7)[:2] == [0, 7]
    assert M.queries_avoiding([0, 3, 5, 19], [0, 1, 5, 6], 1, 20) == [2, 3, 7, 19]
    assert M.queries_avoiding([19], [19], 1, 20) == []


# ---------------------------------------------------------------------------------------- DBoW2
def toy_vocab(tmp_path):
    """k=2, L=2 tree with 4 words; descriptors are 32 bytes, distinct by their first byte."""
    def d(b0):
        x = np.zeros(32, int)
        x[0] = b0
        return " ".join(map(str, x))
    lines = ["2 2  0 0",
             f"0 0 {d(0b00000000)} 0",   # node 1
             f"0 0 {d(0b11110000)} 0",   # node 2
             f"1 1 {d(0b00000001)} 1.0",  # node 3 -> word 0
             f"1 1 {d(0b00000011)} 2.0",  # node 4 -> word 1
             f"2 1 {d(0b11110000)} 0.5",  # node 5 -> word 2
             f"2 1 {d(0b11111111)} 0.0"]  # node 6 -> word 3 (zero idf: stop word)
    p = tmp_path / "voc.txt"
    p.write_text("\n".join(lines) + "\n")
    return Vocabulary.from_text(str(p))


def desc(b0):
    x = np.zeros((1, 32), np.uint8)
    x[0, 0] = b0
    return x


def test_hamming():
    assert hamming(np.array([0b1011], np.uint8), np.array([0b0001], np.uint8)) == 2
    a = np.zeros((2, 32), np.uint8)
    a[1, 5] = 255
    assert hamming(a, np.zeros(32, np.uint8)).tolist() == [0, 8]


def test_vocab_structure_and_descent(tmp_path):
    v = toy_vocab(tmp_path)
    assert v.n_words == 4 and v.children[0].tolist() == [1, 2] and v.children[1].tolist() == [3, 4]
    assert v.word_id[[3, 4, 5, 6]].tolist() == [0, 1, 2, 3]
    D = np.concatenate([desc(0b00000001), desc(0b00000111), desc(0b11100000), desc(0b11111110)])
    assert v.words(D).tolist() == [3, 4, 5, 6]
    # tie at the root (distance 2 to both children): first child wins (DBoW2 uses strict '<')
    assert v.words(desc(0b00110000)).tolist() == [3]


def test_transform_tfidf_l1(tmp_path):
    v = toy_vocab(tmp_path)
    D = np.concatenate([desc(1), desc(1), desc(3), desc(0b11110000), desc(0xFF)])
    ids, w = v.transform(D)
    # word0: 2 x idf 1.0, word1: idf 2.0, word2: 0.5, word3 dropped (zero weight); L1-normalised
    assert ids.tolist() == [0, 1, 2]
    assert np.allclose(w, np.array([2.0, 2.0, 0.5]) / 4.5)
    assert v.transform(np.zeros((0, 32), np.uint8))[0].size == 0


def test_l1_score_matches_definition_and_database():
    a = bow_from_words(np.array([1, 2, 2, 5]), np.array([1.0, 1.0, 1.0, 2.0]))
    b = bow_from_words(np.array([2, 5, 7]), np.array([1.0, 1.0, 2.0]))
    c = bow_from_words(np.array([9]), np.array([1.0]))
    dense = lambda x: np.bincount(x[0], weights=x[1], minlength=10)  # noqa: E731
    expect = 1 - 0.5 * np.abs(dense(a) - dense(b)).sum()
    assert np.isclose(l1_score(a, b), expect)
    assert np.isclose(l1_score(a, a), 1.0) and l1_score(a, c) == 0.0
    db = BowDatabase([a, b, c])
    s, shared = db.query(b)
    assert np.allclose(s, [l1_score(b, a), 1.0, 0.0])
    assert shared.tolist() == [2, 3, 0]
    assert db.nbytes_per_entry() > 0


def test_vocab_npz_cache(tmp_path):
    v = toy_vocab(tmp_path)
    p = str(tmp_path / "voc.txt")
    v1 = Vocabulary.load(p)
    v2 = Vocabulary.load(p)  # from cache
    assert (tmp_path / "voc.txt.npz").exists()
    assert np.array_equal(v1.children, v2.children) and v2.n_words == v.n_words


def test_unsupported_scoring_rejected(tmp_path):
    p = tmp_path / "v.txt"
    p.write_text("2 1  1 0\n0 1 " + " ".join(["0"] * 32) + " 1.0\n")
    with pytest.raises(ValueError):
        Vocabulary.from_text(str(p))


def test_feature_vector_levels(tmp_path):
    v = toy_vocab(tmp_path)  # L = 2
    D = np.concatenate([desc(1), desc(3), desc(0b11110000), desc(0xFF)])
    (ids, _), fv = v.transform(D, levelsup=1)   # level 1: nodes 1 / 2; the stop word is not in the vector
    assert ids.tolist() == [0, 1, 2] and fv.tolist() == [1, 1, 2, -1]
    assert v.transform(D, levelsup=0)[1].tolist() == [3, 4, 5, -1]   # leaves
    assert v.transform(D, levelsup=4)[1].tolist() == [0, 0, 0, -1]   # above the root -> root


def write_dbow3(path, nodes, words, k=2, L=2, compressed=False):
    """Uncompressed DBoW3 stream: nodes = [(nid, parent, weight, float descriptor)], words = [(wid, nid)]."""
    b = struct.pack("<Q?I", DBOW3_MAGIC, compressed, len(nodes) + 1) + struct.pack("<4i", k, L, 0, 0)
    for nid, par, w, d in nodes:
        d = np.asarray(d, np.float32)
        b += struct.pack("<IId3i", nid, par, w, len(d), 1, 5) + d.tobytes()
    b += struct.pack("<I", len(words)) + b"".join(struct.pack("<II", wi, ni) for wi, ni in words)
    path.write_bytes(b)


def test_dbow3_float_vocabulary(tmp_path):
    p = tmp_path / "v.dbow3"
    # stream order differs from id order; node 3 is a leaf at level 1 (shallower than L)
    nodes = [(2, 0, 0.0, [0, 1]), (1, 0, 0.0, [1, 0]), (4, 1, 1.0, [1, 0.2]), (5, 1, 2.0, [1, -0.2]),
             (3, 2, 0.5, [0, 1])]
    write_dbow3(p, nodes, [(0, 4), (1, 5), (2, 3)])
    v = Vocabulary.load(str(p))
    assert v.children[0, :2].tolist() == [2, 1] and v.word_id[[3, 4, 5]].tolist() == [2, 0, 1]
    D = np.array([[0.9, 0.3], [0.9, -0.3], [0.1, 0.9]], np.float32)
    assert v.words(D).tolist() == [4, 5, 3]  # squared L2 descent
    (ids, w), fv = v.transform(D, levelsup=0)
    assert ids.tolist() == [0, 1, 2] and np.allclose(w, np.array([1.0, 2.0, 0.5]) / 3.5)
    assert fv.tolist() == [4, 5, 3]          # leaf above the requested level: its own node
    assert np.array_equal(Vocabulary.load(str(p)).desc, v.desc)  # npz cache
    write_dbow3(tmp_path / "c.dbow3", nodes, [], compressed=True)
    with pytest.raises(ValueError, match="compressed"):
        Vocabulary.from_dbow3(str(tmp_path / "c.dbow3"))


def test_three_maxima_and_rotation_bin():
    assert SM.three_maxima([0, 10, 5, 3]) == (1, 2, 3)
    assert SM.three_maxima([0, 10, 0, 0]) == (1, -1, -1)
    assert SM.three_maxima([0, 100, 20, 5]) == (1, 2, -1)        # third bin < 10 % of the maximum
    assert SM.rotation_bin(10, 5) == 0 and SM.rotation_bin(50, 5) == 2 and SM.rotation_bin(5, 10) == 12


def orb_desc(rows):
    d = np.zeros((len(rows), 32), np.uint8)
    for r, ones in enumerate(rows):
        d[r, :ones // 8] = 255
        if ones % 8:
            d[r, ones // 8] = (1 << (ones % 8)) - 1
    return d


def test_search_by_bow_nodes_thresholds_ratio_and_uniqueness():
    spec = SM.ORB_SLAM3
    ref = orb_desc([0, 0, 0, 0])
    q = orb_desc([10, 40, 60, 0])           # Hamming distances 10 / 40 / 60 / 0 from every ref descriptor
    fv_r = np.array([1, 1, 2, 3])
    fv_q = np.array([1, 1, 2, -1])
    ang = np.zeros(4)
    m = SM.search_by_bow(spec, fv_r, fv_q, ref, q, np.ones(4, bool), ang, ang)
    # ref 0 takes q 0 (10 < 0.75 * 40); ref 1 then only has q 1 left (40 <= TH_LOW); ref 2: 60 > TH_LOW;
    # ref 3's node is not in the query vector (q 3 has none)
    assert m.tolist() == [[0, 0], [1, 1]]
    valid = np.array([False, True, True, True])     # ref 0 has no map point
    m = SM.search_by_bow(spec, fv_r, fv_q, ref, q, valid, ang, ang)
    assert m.tolist() == [[1, 0]]
    q2 = orb_desc([10, 12, 60, 0])                  # ratio test: 10 >= 0.75 * 12
    assert len(SM.search_by_bow(spec, fv_r, fv_q, ref, q2, np.ones(4, bool), ang, ang)) == 0


def test_search_by_bow_orientation_histogram():
    n = 14
    ref, q = orb_desc([0] * n), orb_desc([0] * n)
    fv = np.arange(n)
    a_r = np.zeros(n)
    # rotations 0 / 60 / 180 / 120 deg: 8 in bin 0, 3 in bin 2, 2 in bin 6, 1 in bin 4
    a_q = np.r_[np.zeros(8), np.full(3, 300.0), 180.0, 180.0, 240.0]
    m = SM.search_by_bow(SM.ORB_SLAM3, fv, fv, ref, q, np.ones(n, bool), a_r, a_q)
    assert sorted(m[:, 0].tolist()) == list(range(13))  # only the three largest bins survive
    m = SM.search_by_bow(SM.SP_SLAM3_UNIT, fv, fv, ref.astype(np.float32), q.astype(np.float32),
                         np.ones(n, bool))
    assert len(m) == n                                  # SP_SLAM3: no orientation check


def test_relocalize_projection_pass():
    import cv2
    rng = np.random.default_rng(0)
    K = np.array([[300.0, 0, 320], [0, 300.0, 180], [0, 0, 1]])
    n = 200
    uv = rng.uniform([20, 20], [620, 340], (n, 2))
    z = rng.uniform(2, 6, n)
    X = (np.linalg.solve(K, np.c_[uv, np.ones(n)].T) * z).T     # reference camera = world
    T_q = np.eye(4)
    T_q[:3, 3] = [0.1, 0, 0]
    kq = (K @ (X - T_q[:3, 3]).T).T
    kq = (kq[:, :2] / kq[:, 2:]).astype(np.float32)
    D = rng.normal(size=(n, 256)).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    def pnp(m):
        ok, rv, tv, inl = cv2.solvePnPRansac(X[m[:, 0]], kq[m[:, 1]].astype(np.float64), K, None,
                                             reprojectionError=2.0, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None, 0, np.zeros(0, int)
        R = cv2.Rodrigues(rv)[0]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.T, (-R.T @ tv).ravel()
        return T, len(inl), inl[:, 0]

    spec = SM.SP_SLAM3_UNIT
    ctx = {"K": K, "img_wh": (640, 360), "X_ref": X, "dist_ref": np.linalg.norm(X, axis=1),
           "octave_ref": np.zeros(n, int), "ref_valid": np.ones(n, bool), "kp_q": kq,
           "octave_q": np.zeros(n, int), "desc_ref": D, "desc_q": D}
    bow = np.c_[np.arange(10), np.arange(10)]
    assert SM.relocalize(spec, bow, pnp, ctx)[3] == "few_bow"
    bow = np.c_[np.arange(20), np.arange(20)]
    T, ninl, m, stage = SM.relocalize(spec, bow, pnp, ctx)
    assert stage in ("projection", "projection2") and ninl >= 50
    assert np.all(m[:, 0] == m[:, 1]) and np.allclose(T[:3, 3], T_q[:3, 3], atol=1e-3)
    assert SM.relocalize(spec, bow, pnp, None)[3] == "bow"     # no projection context: BoW matches only


def test_session_roundtrip(tmp_path):
    img = tmp_path / "imgs"
    img.mkdir()
    (img / "frames.csv").write_text("timestamp_ns,left_path,right_path\n1,cam0/1.png,cam1/1.png\n"
                                    "2,cam0/2.png,cam1/2.png\n")
    T = np.stack([pose(0), pose(1)])
    meta = {"name": "x", "kind": "sim", "image_root": str(img), "fx": 100.0, "fy": 100.0, "cx": 5.0, "cy": 6.0,
            "baseline_m": 0.1, "depth_dir": str(img / "depth0")}
    write_session(str(tmp_path / "s"), meta, T, np.array([True, False]), np.array([1, 2]))
    S = load_session(str(tmp_path / "s"))
    assert len(S) == 2 and S.valid.tolist() == [True, False] and S.K[0, 2] == 5.0
    assert S.left_path(1).endswith("cam0/2.png") and S.depth_path(0).endswith("depth0/1.png")
