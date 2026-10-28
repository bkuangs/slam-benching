"""Pure-python parts of the VPR benchmark (vpr/metrics.py, vpr/dbow.py, vpr/sessions.py); no torch needed."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vpr import metrics as M  # noqa: E402
from vpr.dbow import BowDatabase, Vocabulary, bow_from_words, hamming, l1_score  # noqa: E402
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
