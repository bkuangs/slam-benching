"""Compact pure-numpy reimplementation of DBoW2 as used by ORB-SLAM3 (FORB descriptors, ORBvoc.txt).

Faithful to DBoW2's TemplatedVocabulary: text-file loading (node order defines node and word ids),
greedy Hamming descent from the root choosing the first child with the smallest distance, TF-IDF weighting
(each feature adds its leaf's IDF weight to the word), L1 normalisation, and L1 scoring
    s(v, w) = -1/2 * sum_common(|v_i - w_i| - |v_i| - |w_i|)  ( = sum_common min(v_i, w_i) for v, w >= 0 ).
"""
import os
from typing import Dict, List, Tuple

import numpy as np

_POPCNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance between uint8 descriptor arrays (broadcasting over leading axes)."""
    return _POPCNT[np.bitwise_xor(a, b)].sum(axis=-1, dtype=np.int32)


class Vocabulary:
    """DBoW2 vocabulary tree. Arrays: desc (n_nodes, 32) uint8, weight (n_nodes,), children (n_nodes, k)
    int32 (-1 = none), word_id (n_nodes,) int32 (-1 for non-leaves). Node 0 is the root."""

    def __init__(self, k: int, L: int, scoring: int, weighting: int, parent: np.ndarray, is_leaf: np.ndarray,
                 desc: np.ndarray, weight: np.ndarray):
        if scoring != 0 or weighting != 0:
            raise ValueError("only L1 scoring + TF-IDF weighting (ORBvoc.txt) are implemented")
        self.k, self.L = int(k), int(L)
        n = len(parent) + 1
        self.desc = np.zeros((n, desc.shape[1]), np.uint8)
        self.desc[1:] = desc
        self.weight = np.zeros(n)
        self.weight[1:] = weight
        self.children = np.full((n, self.k), -1, np.int32)
        fill = np.zeros(n, np.int32)
        for nid, p in enumerate(parent, start=1):  # DBoW2 appends children in file order
            self.children[p, fill[p]] = nid
            fill[p] += 1
        self.word_id = np.full(n, -1, np.int32)
        leaves = np.flatnonzero(is_leaf) + 1
        self.word_id[leaves] = np.arange(len(leaves), dtype=np.int32)
        self.n_words = len(leaves)

    @classmethod
    def from_text(cls, path: str) -> "Vocabulary":
        with open(path) as f:
            k, L, scoring, weighting = (int(x) for x in f.readline().split())
        a = np.loadtxt(path, skiprows=1, dtype=np.float64, ndmin=2)
        return cls(k, L, scoring, weighting, a[:, 0].astype(np.int64), a[:, 1].astype(bool),
                   a[:, 2:-1].astype(np.uint8), a[:, -1])

    @classmethod
    def load(cls, path: str, cache: str = None) -> "Vocabulary":
        """Load ORBvoc.txt, caching the parsed arrays as .npz (parsing the 145 MB text takes ~1 min)."""
        cache = cache or path + ".npz"
        if os.path.exists(cache):
            z = np.load(cache)
            v = cls.__new__(cls)
            for key in ("desc", "weight", "children", "word_id"):
                setattr(v, key, z[key])
            v.k, v.L, v.n_words = (int(x) for x in z["meta"])
            return v
        v = cls.from_text(path)
        np.savez(cache, desc=v.desc, weight=v.weight, children=v.children, word_id=v.word_id,
                 meta=np.array([v.k, v.L, v.n_words]))
        return v

    def words(self, descriptors: np.ndarray) -> np.ndarray:
        """Leaf node id of each descriptor (greedy descent, first minimum wins as in DBoW2)."""
        d = np.asarray(descriptors, np.uint8).reshape(-1, self.desc.shape[1])
        node = np.zeros(len(d), np.int64)
        active = self.children[node, 0] >= 0
        while active.any():
            idx = np.flatnonzero(active)
            ch = self.children[node[idx]]                          # (m, k)
            dist = hamming(self.desc[np.maximum(ch, 0)], d[idx, None, :]).astype(np.int64)
            dist[ch < 0] = np.iinfo(np.int64).max
            node[idx] = ch[np.arange(len(idx)), np.argmin(dist, axis=1)]
            active = self.children[node, 0] >= 0
        return node

    def transform(self, descriptors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """BoW vector as (sorted word ids, L1-normalised TF-IDF weights)."""
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(0, np.int32), np.zeros(0)
        node = self.words(descriptors)
        w = self.weight[node]
        keep = w > 0
        return bow_from_words(self.word_id[node[keep]], w[keep])


def bow_from_words(word_ids: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate per-feature weights into words and L1-normalise (DBoW2 TF-IDF + L1_NORM)."""
    ids, inv = np.unique(np.asarray(word_ids), return_inverse=True)
    v = np.bincount(inv, weights=np.asarray(weights, float), minlength=len(ids))
    s = np.abs(v).sum()
    return ids.astype(np.int32), (v / s if s > 0 else v)


def l1_score(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]) -> float:
    """DBoW2 L1Scoring::score of two normalised BoW vectors."""
    (ia, va), (ib, vb) = a, b
    common, xa, xb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    if len(common) == 0:
        return 0.0
    x, y = va[xa], vb[xb]
    return float(-0.5 * np.sum(np.abs(x - y) - np.abs(x) - np.abs(y)))


class BowDatabase:
    """Inverted-index database: L1 scores and shared-word counts of a query against all entries."""

    def __init__(self, bows: List[Tuple[np.ndarray, np.ndarray]]):
        self.n = len(bows)
        w = np.concatenate([b[0] for b in bows]) if bows else np.zeros(0, np.int32)
        v = np.concatenate([b[1] for b in bows]) if bows else np.zeros(0)
        f = np.concatenate([np.full(len(b[0]), i, np.int64) for i, b in enumerate(bows)]) if bows else \
            np.zeros(0, np.int64)
        order = np.argsort(w, kind="stable")
        self.word, self.val, self.frame = w[order], v[order], f[order]

    def query(self, bow: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """(scores (n,), shared-word counts (n,)) for a query BoW vector."""
        ids, vals = bow
        lo = np.searchsorted(self.word, ids, "left")
        hi = np.searchsorted(self.word, ids, "right")
        cnt = hi - lo
        if cnt.sum() == 0:
            return np.zeros(self.n), np.zeros(self.n, np.int64)
        qi = np.repeat(np.arange(len(ids)), cnt)
        pos = np.repeat(lo - np.cumsum(np.r_[0, cnt[:-1]]), cnt) + np.arange(cnt.sum())
        dv, fr, qv = self.val[pos], self.frame[pos], vals[qi]
        contrib = -0.5 * (np.abs(qv - dv) - np.abs(qv) - np.abs(dv))
        return (np.bincount(fr, weights=contrib, minlength=self.n),
                np.bincount(fr, minlength=self.n).astype(np.int64))

    def nbytes_per_entry(self) -> float:
        """Inverted-index storage (word id int32 + weight float64 + frame id) per database frame."""
        return (self.word.nbytes + self.val.nbytes + self.frame.nbytes) / max(self.n, 1)
