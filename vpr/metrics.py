"""Retrieval ground truth, Recall@K and frame subsampling (pure numpy)."""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotation(s) Ra and Rb, broadcasting over leading axes."""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)  # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def pose_distances(T_q: np.ndarray, T_db: np.ndarray):
    """Pairwise (n_q, n_db) translation distance [m] and rotation angle [deg] between cam-to-world poses."""
    dt = np.linalg.norm(T_q[:, None, :3, 3] - T_db[None, :, :3, 3], axis=-1)
    dr = rot_angle_deg(T_q[:, None, :3, :3], T_db[None, :, :3, :3])
    return dt, dr


def positives(T_q: np.ndarray, T_db: np.ndarray, max_dist_m: float, max_angle_deg: float) -> np.ndarray:
    """Boolean (n_q, n_db): database frame within max_dist_m AND max_angle_deg of the query pose."""
    dt, dr = pose_distances(T_q, T_db)
    return (dt <= max_dist_m) & (dr <= max_angle_deg)


def rank_by_similarity(sim: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Database indices sorted by decreasing similarity per query (stable for ties), first k columns."""
    order = np.argsort(-sim, axis=1, kind="stable")
    return order if k is None else order[:, :k]


def recall_at_k(ranking: np.ndarray, pos: np.ndarray, ks: Iterable[int] = (1, 5, 10)) -> Dict[str, float]:
    """Fraction of queries (with >= 1 positive) that have a positive among their top-k retrieved frames.

    ranking: (n_q, >= max(ks)) database indices; pos: (n_q, n_db) boolean positives."""
    has = pos.any(axis=1)
    out = {"n_queries": int(len(pos)), "n_with_positive": int(has.sum())}
    if not has.any():
        return {**out, **{f"R@{k}": float("nan") for k in ks}}
    hits = np.take_along_axis(pos, ranking, axis=1)
    for k in ks:
        out[f"R@{k}"] = float(hits[has, :k].any(axis=1).mean())
    return out


def oracle_ranking(T_q: np.ndarray, T_db: np.ndarray, max_angle_deg: float, k: int) -> np.ndarray:
    """GT 'retrieval': database frames ordered by translation distance, frames outside the viewing-angle
    threshold pushed behind all frames inside it."""
    dt, dr = pose_distances(T_q, T_db)
    cost = dt + np.where(dr <= max_angle_deg, 0.0, 1e6)
    return np.argsort(cost, axis=1, kind="stable")[:, :k]


def subsample_by_motion(T: np.ndarray, valid: np.ndarray, min_dist_m: float, min_angle_deg: float,
                        min_gap: int = 1) -> List[int]:
    """Keyframe-style selection: keep a frame when it moved >= min_dist_m or turned >= min_angle_deg since
    the last kept frame (and is >= min_gap frames after it). Frames with invalid GT are skipped."""
    kept: List[int] = []
    for i in np.flatnonzero(valid):
        if not kept:
            kept.append(int(i))
            continue
        j = kept[-1]
        if i - j < min_gap:
            continue
        d = np.linalg.norm(T[i, :3, 3] - T[j, :3, 3])
        a = rot_angle_deg(T[i, :3, :3], T[j, :3, :3])
        if d >= min_dist_m or a >= min_angle_deg:
            kept.append(int(i))
    return kept


def queries_avoiding(candidates: Sequence[int], exclude: Iterable[int], shift: int, n: int) -> List[int]:
    """Shift each candidate query index forward (by `shift` steps at a time) until it is not an excluded
    (database) index and is < n. Used so same-session sanity queries are never the identical database image."""
    ex = set(int(e) for e in exclude)
    out = []
    for c in candidates:
        c = int(c)
        while c in ex:
            c += shift
        if c < n:
            out.append(c)
    return sorted(set(out))
