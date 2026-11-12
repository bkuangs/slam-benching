"""Compact pure-numpy reimplementation of DBoW2 as used by ORB-SLAM3 (FORB descriptors, ORBvoc.txt), and of
DBoW3 as used by SP_SLAM3 (float SuperPoint descriptors, superpoint_voc.dbow3).

Faithful to DBoW2's TemplatedVocabulary / DBoW3's Vocabulary: text-file (DBoW2) or binary-stream (DBoW3)
loading, greedy descent from the root choosing the first child with the smallest distance (Hamming for binary,
squared L2 for float descriptors), TF-IDF weighting (each feature adds its leaf's IDF weight to the word),
L1 normalisation, L1 scoring
    s(v, w) = -1/2 * sum_common(|v_i - w_i| - |v_i| - |w_i|)  ( = sum_common min(v_i, w_i) for v, w >= 0 ),
and the FeatureVector (features grouped by their ancestor node `levelsup` levels above the leaves) that
ORB-SLAM3 (levelsup 4) and SP_SLAM3 (levelsup 0) use to restrict SearchByBoW candidates.
"""
import os
import struct
from typing import Dict, List, Tuple

import numpy as np

_POPCNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance between uint8 descriptor arrays (broadcasting over leading axes)."""
    return _POPCNT[np.bitwise_xor(a, b)].sum(axis=-1, dtype=np.int32)


DBOW3_MAGIC = 88877711233


class Vocabulary:
    """DBoW2/DBoW3 vocabulary tree. Arrays: desc (n_nodes, D) uint8 (binary) or float32, weight (n_nodes,),
    children (n_nodes, k) int32 (-1 = none, in insertion order), word_id (n_nodes,) int32 (-1 for non-leaves).
    Node 0 is the root (level 0); leaves are at most L levels below it."""

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
    def from_dbow3(cls, path: str) -> "Vocabulary":
        """Uncompressed DBoW3 binary vocabulary (Vocabulary::toStream). Compressed (quicklz) files must first be
        expanded with scripts/dbow3_decompress.sh."""
        with open(path, "rb") as f:
            b = f.read()
        sig, compressed, n = struct.unpack_from("<Q?I", b, 0)
        if sig != DBOW3_MAGIC:
            raise ValueError(f"{path}: not a DBoW3 binary vocabulary")
        if compressed:
            raise ValueError(f"{path}: quicklz-compressed, expand it with scripts/dbow3_decompress.sh first")
        k, L, scoring, weighting = struct.unpack_from("<4i", b, 13)
        if scoring != 0 or weighting != 0:
            raise ValueError("only L1 scoring + TF-IDF weighting are implemented")
        off = 29
        cols, _, typ = struct.unpack_from("<3i", b, off + 16)
        dt = {0: np.uint8, 5: np.float32}[typ]
        rec = np.dtype([("nid", "<u4"), ("parent", "<u4"), ("weight", "<f8"), ("cols", "<i4"), ("rows", "<i4"),
                        ("type", "<i4"), ("d", dt, (cols,))])
        r = np.frombuffer(b, rec, n - 1, off)
        if (r["cols"] != cols).any() or (r["rows"] != 1).any() or (r["type"] != typ).any():
            raise ValueError(f"{path}: inconsistent node descriptors")
        off += rec.itemsize * (n - 1)
        (n_words,) = struct.unpack_from("<I", b, off)
        wn = np.frombuffer(b, np.dtype([("wid", "<u4"), ("nid", "<u4")]), n_words, off + 4)
        v = cls.__new__(cls)
        v.k, v.L, v.n_words = k, L, int(n_words)
        v.desc = np.zeros((n, cols), dt)
        v.desc[r["nid"]] = r["d"]
        v.weight = np.zeros(n)
        v.weight[r["nid"]] = r["weight"]
        v.children = np.full((n, k), -1, np.int32)
        fill = np.zeros(n, np.int32)
        for nid, p in zip(r["nid"], r["parent"]):  # DBoW3 appends children in stream order
            v.children[p, fill[p]] = nid
            fill[p] += 1
        v.word_id = np.full(n, -1, np.int32)
        v.word_id[wn["nid"]] = wn["wid"]
        return v

    @classmethod
    def load(cls, path: str, cache: str = None) -> "Vocabulary":
        """Load ORBvoc.txt (DBoW2 text) or an uncompressed .dbow3, caching the parsed arrays as .npz (parsing
        the 145 MB ORBvoc.txt takes ~1 min)."""
        cache = cache or path + ".npz"
        if os.path.exists(cache):
            z = np.load(cache)
            v = cls.__new__(cls)
            for key in ("desc", "weight", "children", "word_id"):
                setattr(v, key, z[key])
            v.k, v.L, v.n_words = (int(x) for x in z["meta"])
            return v
        v = cls.from_dbow3(path) if path.endswith(".dbow3") else cls.from_text(path)
        np.savez(cache, desc=v.desc, weight=v.weight, children=v.children, word_id=v.word_id,
                 meta=np.array([v.k, v.L, v.n_words]))
        return v

    @property
    def binary(self) -> bool:
        return self.desc.dtype == np.uint8

    def _distance(self, nodes: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self.binary:
            return hamming(self.desc[nodes], d[:, None, :]).astype(np.float64)
        diff = self.desc[nodes] - d[:, None, :]
        return np.einsum("mkd,mkd->mk", diff, diff, dtype=np.float64)

    def words(self, descriptors: np.ndarray, levelsup: int = None):
        """Leaf node id of each descriptor (greedy descent, first minimum wins as in DBoW2/DBoW3). With
        `levelsup`, also the node at level L - levelsup on the descent path (the FeatureVector node; the root
        if that level is <= 0, the leaf if the leaf is shallower)."""
        d = np.asarray(descriptors, self.desc.dtype).reshape(-1, self.desc.shape[1])
        node = np.zeros(len(d), np.int64)
        nid_level = None if levelsup is None else self.L - levelsup
        fv = np.zeros(len(d), np.int64)
        level = 0
        active = self.children[node, 0] >= 0
        while active.any():
            level += 1
            idx = np.flatnonzero(active)
            ch = self.children[node[idx]]                          # (m, k)
            dist = self._distance(np.maximum(ch, 0), d[idx])
            dist[ch < 0] = np.inf
            node[idx] = ch[np.arange(len(idx)), np.argmin(dist, axis=1)]
            if nid_level is not None and level == nid_level:
                fv[idx] = node[idx]
            active = self.children[node, 0] >= 0
        if nid_level is None:
            return node
        if nid_level > 0:
            shallow = fv == 0
            fv[shallow] = node[shallow]
        return node, fv

    def transform(self, descriptors: np.ndarray, levelsup: int = None):
        """BoW vector as (sorted word ids, L1-normalised TF-IDF weights). With `levelsup`, returns
        (bow, fv) where fv[i] is feature i's FeatureVector node (-1 if its word has zero weight, as DBoW
        skips such features)."""
        if descriptors is None or len(descriptors) == 0:
            bow = np.zeros(0, np.int32), np.zeros(0)
            return bow if levelsup is None else (bow, np.zeros(0, np.int64))
        if levelsup is None:
            node = self.words(descriptors)
        else:
            node, fv = self.words(descriptors, levelsup)
        w = self.weight[node]
        keep = w > 0
        bow = bow_from_words(self.word_id[node[keep]], w[keep])
        if levelsup is None:
            return bow
        return bow, np.where(keep, fv, -1)


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
