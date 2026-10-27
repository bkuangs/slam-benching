#!/usr/bin/env python3
"""Export LightGlue (SuperPoint weights) to TorchScript for SP_SLAM3, usable on CPU and CUDA.

Wraps SP_SLAM3's scripts/export_lightglue.py: same LightGlueExport module and the interface
LightGlue.cc expects (forward(kpts0, kpts1, desc0, desc1) -> (matches [K,2], scores [K])).
Differences from running that script directly:

* The image is built without a GPU, so the model is traced on the CPU. A trace bakes the device
  of tensors created or moved inside the graph: LightGlue's filter_matches creates its index
  tensors with torch.arange(..., device=...) and normalize_keypoints calls size.to(kpts), so a CPU
  trace fails with a device mismatch on CUDA. Both are replaced by equivalents that derive
  everything from their input tensors (normalize_keypoints only for the path used here, without
  image_size), and the export fails if the traced graph still contains a device constant.
* The saved trace is checked against the unmodified eager model (LightGlue's own functions) on
  inputs of other sizes than the tracing inputs (it must not have specialised on the number of
  keypoints).

Usage: export_lightglue_slambench.py <SP_SLAM3 scripts dir> <output .pt>
"""

import sys

import torch

import lightglue.lightglue as lg_impl


def filter_matches(scores: torch.Tensor, th: float):
    """lightglue.lightglue.filter_matches without device-bound tensor creation."""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.ones_like(m0).cumsum(1) - 1
    indices1 = torch.ones_like(m1).cumsum(1) - 1
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = torch.zeros_like(max0_exp)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), torch.zeros_like(mscores0.gather(1, m1)))
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


_normalize_keypoints = lg_impl.normalize_keypoints


def normalize_keypoints(kpts: torch.Tensor, size=None) -> torch.Tensor:
    """lightglue.lightglue.normalize_keypoints without the (no-op) size.to(kpts) for size=None."""
    if size is not None:
        return _normalize_keypoints(kpts, size)
    size = 1 + kpts.max(-2).values - kpts.min(-2).values
    shift = size / 2
    scale = size.max(-1).values / 2
    return (kpts - shift[..., None, :]) / scale[..., None, None]


def make_pair(n: int, m: int, gen: torch.Generator):
    """Keypoints in [-1, 1] and unit descriptors where most of image1 re-observes image0."""
    kpts0 = torch.rand(1, n, 2, generator=gen) * 2 - 1
    desc0 = torch.nn.functional.normalize(torch.randn(1, n, 256, generator=gen), dim=-1)
    k = min(n, m) * 3 // 4
    perm = torch.randperm(n, generator=gen)[:k]
    kpts1 = torch.rand(1, m, 2, generator=gen) * 2 - 1
    desc1 = torch.nn.functional.normalize(torch.randn(1, m, 256, generator=gen), dim=-1)
    kpts1[0, :k] = (kpts0[0, perm] + 0.02 * torch.randn(k, 2, generator=gen)).clamp(-1, 1)
    desc1[0, :k] = torch.nn.functional.normalize(desc0[0, perm] + 0.05 * torch.randn(k, 256, generator=gen),
                                                 dim=-1)
    return kpts0, kpts1, desc0, desc1


def main() -> int:
    scripts_dir, output = sys.argv[1], sys.argv[2]
    originals = lg_impl.filter_matches, lg_impl.normalize_keypoints
    lg_impl.filter_matches = filter_matches
    lg_impl.normalize_keypoints = normalize_keypoints
    sys.path.insert(0, scripts_dir)
    import export_lightglue as upstream  # SP_SLAM3 scripts/export_lightglue.py

    torch.manual_seed(0)
    model = upstream.LightGlueExport().eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        traced = torch.jit.trace(model, make_pair(200, 160, gen), strict=False)
    bad = [line.strip() for line in str(traced.inlined_graph).splitlines()
           if "Device = prim::Constant" in line or 'value="cpu"' in line or 'value="cuda' in line]
    if bad:
        print("traced LightGlue graph contains device constants:", *bad[:20], sep="\n  ", file=sys.stderr)
        return 1
    traced.save(output)

    # Reference: the eager model with LightGlue's own filter_matches / normalize_keypoints.
    lg_impl.filter_matches, lg_impl.normalize_keypoints = originals
    loaded = torch.jit.load(output, map_location="cpu")
    for n, m in ((500, 700), (1200, 1150), (37, 900)):
        inputs = make_pair(n, m, gen)
        with torch.no_grad():
            ref_m, ref_s = model(*inputs)
            got_m, got_s = loaded(*inputs)
        if not (torch.equal(ref_m, got_m) and torch.allclose(ref_s, got_s, atol=1e-4)):
            print(f"traced LightGlue differs from eager model for N={n}, M={m}", file=sys.stderr)
            return 1
        print(f"check N={n} M={m}: {ref_m.shape[0]} matches, traced output identical")
    print(f"LightGlue TorchScript model written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
