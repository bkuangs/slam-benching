"""ORB-SLAM3 / SP_SLAM3 relocalization matching, ported from Tracking::Relocalization and ORBmatcher / SPmatcher.

`search_by_bow` is ORBmatcher::SearchByBoW(KeyFrame*, Frame&): only features in the same FeatureVector node are
compared, each keyframe map point takes its nearest unmatched frame feature if the distance is <= TH_LOW and
< ratio x the second nearest, then (ORB only) a rotation-consistency histogram keeps the three dominant bins.
`search_by_projection` is SearchByProjection(Frame&, KeyFrame*, sAlreadyFound, th, dist): map points not yet
found are projected with the current pose estimate and matched to the nearest unmatched feature in a
th x scale(predicted level) window at levels predicted-1..predicted+1 (no ratio test).
`relocalize` chains them as Tracking::Relocalization does (>= 15 BoW matches, PnP, projection passes when
fewer than 50 inliers), with the caller's PnP in place of MLPnP + PoseOptimization.

Keyframe "map points" are the reference features with a valid depth (the caller's mask).
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from vpr.dbow import hamming

HISTO_LENGTH = 30


@dataclass(frozen=True)
class SlamMatcherSpec:
    binary: bool            # Hamming (ORB) or L2 (SuperPoint) descriptor distance
    th_low: float           # SearchByBoW acceptance / second projection pass threshold base
    th_high: float          # first projection pass threshold
    th_second: float        # second projection pass threshold (ORB 64 = TH_LOW + 0.28 (TH_HIGH - TH_LOW))
    init_dist: float        # bestDist initial value (256 / 2.0)
    check_orientation: bool
    scale_factor: float = 1.2
    n_levels: int = 8
    bow_ratio: float = 0.75  # ORBmatcher matcher(0.75, true)
    min_bow_matches: int = 15
    min_inliers: int = 50


# ORB-SLAM3: TH_LOW 50, TH_HIGH 100, second pass 64, orientation check, 8 levels x 1.2
ORB_SLAM3 = SlamMatcherSpec(True, 50, 100, 64, 256, True)
# SP_SLAM3 with unit descriptors (patch 0008): 0.52 / 1.19, second pass TH_LOW + 0.28 (TH_HIGH - TH_LOW), no
# orientation (SuperPoint has none), one level
SP_SLAM3_UNIT = SlamMatcherSpec(False, 0.52, 1.19, 0.52 + 0.28 * (1.19 - 0.52), 2.0, False, 1.2, 1)


def distances(spec: SlamMatcherSpec, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) descriptor distances."""
    if spec.binary:
        return hamming(a[:, None, :], b[None, :, :]).astype(np.float64)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float64)


def rotation_bin(angle_ref: float, angle_q: float) -> int:
    """ORB-SLAM3's bin: round(rot * 1/HISTO_LENGTH) with rot in degrees, i.e. 30-degree bins 0..12 (sic)."""
    rot = angle_ref - angle_q
    if rot < 0.0:
        rot += 360.0
    b = int(np.floor(rot / HISTO_LENGTH + 0.5))
    return 0 if b == HISTO_LENGTH else b


def three_maxima(counts) -> Tuple[int, int, int]:
    """ORBmatcher::ComputeThreeMaxima (strict >, earlier bins win ties; drop bins < 10 % of the maximum)."""
    m1 = m2 = m3 = 0
    i1 = i2 = i3 = -1
    for i, s in enumerate(counts):
        if s > m1:
            m3, m2, m1, i3, i2, i1 = m2, m1, s, i2, i1, i
        elif s > m2:
            m3, m2, i3, i2 = m2, s, i2, i
        elif s > m3:
            m3, i3 = s, i
    if m2 < 0.1 * m1:
        i2 = i3 = -1
    elif m3 < 0.1 * m1:
        i3 = -1
    return i1, i2, i3


def _orientation_filter(pairs, angle_ref, angle_q):
    """Keep pairs (ref, q) whose rotation bin is one of the three dominant bins."""
    if not pairs:
        return pairs
    bins = [rotation_bin(angle_ref[r], angle_q[q]) for r, q in pairs]
    keep = set(three_maxima(np.bincount(bins, minlength=HISTO_LENGTH))) - {-1}
    return [p for p, b in zip(pairs, bins) if b in keep]


def search_by_bow(spec: SlamMatcherSpec, fv_ref: np.ndarray, fv_q: np.ndarray, desc_ref: np.ndarray,
                  desc_q: np.ndarray, ref_valid: np.ndarray, angle_ref=None, angle_q=None) -> np.ndarray:
    """(n, 2) matches (ref index, query index). fv_*: FeatureVector node per feature (-1 = not in the vector)."""
    q_used = np.zeros(len(desc_q), bool)
    pairs = []
    nodes = np.intersect1d(fv_ref[fv_ref >= 0], fv_q[fv_q >= 0])   # std::map iteration: ascending node id
    for n in nodes:
        ir = np.flatnonzero((fv_ref == n) & ref_valid)                 # feature order within a node
        iq = np.flatnonzero(fv_q == n)
        if len(ir) == 0:
            continue
        D = distances(spec, desc_ref[ir], desc_q[iq])
        for a, r in enumerate(ir):
            d = np.where(q_used[iq], np.inf, D[a])
            if not np.isfinite(d).any():
                continue
            j = int(np.argmin(d))                                      # first minimum, strict <
            best1 = d[j]
            rest = np.delete(d, j)
            best2 = min(spec.init_dist, rest.min()) if len(rest) else spec.init_dist
            if best1 <= spec.th_low and best1 < spec.bow_ratio * best2:
                q_used[iq[j]] = True
                pairs.append((int(r), int(iq[j])))
    if spec.check_orientation:
        pairs = _orientation_filter(pairs, angle_ref, angle_q)
    return np.array(pairs, int).reshape(-1, 2)


def search_by_projection(spec: SlamMatcherSpec, T_wq: np.ndarray, K: np.ndarray, img_wh, X_ref: np.ndarray,
                         dist_ref: np.ndarray, octave_ref: np.ndarray, ref_candidates: np.ndarray,
                         kp_q: np.ndarray, octave_q: np.ndarray, desc_ref: np.ndarray, desc_q: np.ndarray,
                         q_used: np.ndarray, th: float, max_dist: float, angle_ref=None, angle_q=None) -> np.ndarray:
    """New (ref, query) matches for reference features `ref_candidates` (world points X_ref, distance from the
    reference camera dist_ref). q_used is updated in place."""
    R, t = T_wq[:3, :3], T_wq[:3, 3]
    Ow = t
    f = np.asarray(spec.scale_factor ** np.arange(spec.n_levels))
    log_sf = np.log(spec.scale_factor)
    w, h = img_wh
    pairs = []
    for r in ref_candidates:
        Xc = R.T @ (X_ref[r] - t)
        if Xc[2] <= 0:
            continue
        u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
        v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
        if u < 0 or u > w or v < 0 or v > h:
            continue
        # MapPoint::UpdateNormalAndDepth / Get{Max,Min}DistanceInvariance / PredictScale
        max_d = dist_ref[r] * f[octave_ref[r]]
        min_d = max_d / f[-1]
        d3 = float(np.linalg.norm(X_ref[r] - Ow))
        if d3 < 0.8 * min_d or d3 > 1.2 * max_d:
            continue
        lvl = int(np.clip(np.ceil(np.log(max_d / d3) / log_sf), 0, spec.n_levels - 1)) if spec.n_levels > 1 else 0
        rad = th * f[lvl]
        cand = np.flatnonzero((np.abs(kp_q[:, 0] - u) < rad) & (np.abs(kp_q[:, 1] - v) < rad)
                              & (octave_q >= lvl - 1) & (octave_q <= lvl + 1) & ~q_used)
        if len(cand) == 0:
            continue
        d = distances(spec, desc_ref[r:r + 1], desc_q[cand])[0]
        j = int(np.argmin(d))
        if d[j] <= max_dist:
            q_used[cand[j]] = True
            pairs.append((int(r), int(cand[j])))
    if spec.check_orientation:
        kept = _orientation_filter(pairs, angle_ref, angle_q)
        for p in set(pairs) - set(kept):
            q_used[p[1]] = False
        pairs = kept
    return np.array(pairs, int).reshape(-1, 2)


def relocalize(spec: SlamMatcherSpec, bow_matches: np.ndarray, pnp: Callable, project_ctx: Optional[dict]):
    """Tracking::Relocalization for one candidate keyframe.

    pnp(matches) -> (T_wq or None, n_inliers, inlier indices into matches). project_ctx holds the
    search_by_projection inputs except T_wq, ref_candidates, q_used, th and max_dist (None: skip the passes).
    Returns (T_wq or None, n_inliers, final matches, stage) with stage in
    few_bow | pnp_fail | bow | projection | projection2."""
    if len(bow_matches) < spec.min_bow_matches:
        return None, 0, bow_matches, "few_bow"
    T, n, inl = pnp(bow_matches)
    if T is None or n < 10:
        return T, n, bow_matches, "pnp_fail"
    if n >= spec.min_inliers or project_ctx is None:
        return T, n, bow_matches, "bow"
    found = bow_matches[inl]
    c = project_ctx
    n_ref = len(c["X_ref"])

    def project(T_cur, found_m, th, max_dist):
        q_used = np.zeros(len(c["kp_q"]), bool)
        q_used[found_m[:, 1]] = True
        seen = np.zeros(n_ref, bool)
        seen[found_m[:, 0]] = True
        cand = np.flatnonzero(c["ref_valid"] & ~seen)
        return search_by_projection(spec, T_cur, c["K"], c["img_wh"], c["X_ref"], c["dist_ref"], c["octave_ref"],
                                    cand, c["kp_q"], c["octave_q"], c["desc_ref"], c["desc_q"], q_used, th,
                                    max_dist, c.get("angle_ref"), c.get("angle_q"))

    extra = project(T, found, 10, spec.th_high)
    if n + len(extra) < spec.min_inliers:
        return T, n, found, "bow"
    m = np.vstack([found, extra])
    T2, n2, inl2 = pnp(m)
    if T2 is None:
        return T, n, found, "bow"
    stage = "projection"
    T, n, found = T2, n2, m[inl2]
    if 30 < n < spec.min_inliers:
        extra = project(T, found, 3, spec.th_second)
        if n + len(extra) >= spec.min_inliers:
            m = np.vstack([found, extra])
            T3, n3, inl3 = pnp(m)
            if T3 is not None:
                T, n, found, stage = T3, n3, m[inl3], "projection2"
    return T, n, found, stage
