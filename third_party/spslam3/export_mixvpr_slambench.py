#!/usr/bin/env python3
"""Export MixVPR (official GSV-Cities ResNet-50 / 4096-d checkpoint) to TorchScript for SP_SLAM3.

The exported module contains the complete preprocessing of the offline benchmark
(vpr.models.GlobalExtractor("mixvpr")), so C++ only hands over the raw image:

    forward(img: uint8 [1, H, W, 3], RGB) -> float32 [1, 4096], L2-normalised

inside: /255, ImageNet mean/std normalisation, bilinear resize to 320x320 with antialias
(aspect ratio not preserved, as MixVPR's demo and the offline benchmark), ResNet-50 cut after
layer3, MixVPR aggregator, L2 normalisation. The mean/std are registered buffers and the module
is traced on the CPU, so it moves to CUDA with Module::to(device) (the export fails if the traced
graph contains a device constant, as export_lightglue_slambench.py).

Checks (the export fails otherwise): the traced module, run on images of another size than the
tracing input, equals the eager wrapper, and the cosine similarity to the offline extractor
(vpr.models.GlobalExtractor, which the place-recognition benchmark used) is >= 0.999 on real
images, on the CPU and, if available, on CUDA.

Run in the LAC image (torch 2.3.1, loadable by SP_SLAM3's LibTorch 2.3.0), from the repo root:

    apptainer exec --nv --bind /storage/ice1/3/6/bkuang7 containers/lac_26c3a2b.sif \\
        python third_party/spslam3/export_mixvpr_slambench.py \\
            --weights /storage/ice1/3/6/bkuang7/slam-bench-work/vpr_weights \\
            --out /storage/ice1/3/6/bkuang7/slam-bench-work/vpr_weights/mixvpr_slambench.pt \\
            --images <png> [<png> ...]
"""

import argparse
import hashlib
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from vpr.models import GlobalExtractor, MixVPRModel  # noqa: E402

INPUT_SIZE = (320, 320)


class MixVPRSlambench(torch.nn.Module):
    """uint8 RGB [1,H,W,3] -> L2-normalised [1,4096] (preprocessing of vpr.models.GlobalExtractor)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.permute(0, 3, 1, 2).to(self.mean.dtype).div(255.0)
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=INPUT_SIZE, mode="bilinear", align_corners=False, antialias=True)
        return F.normalize(self.model(x).float(), dim=-1)


def to_input(bgr: np.ndarray) -> torch.Tensor:
    """What the SP_SLAM3 patch passes: the left image as RGB uint8, [1,H,W,3]."""
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))[None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="dir with MixVPR/ (repo) and mixvpr_resnet50_4096.ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="real images for the equivalence check")
    args = ap.parse_args()

    model = MixVPRModel(os.path.join(args.weights, "MixVPR"),
                        os.path.join(args.weights, "mixvpr_resnet50_4096.ckpt")).eval()
    wrapper = MixVPRSlambench(model).eval()
    gen = torch.Generator().manual_seed(0)
    example = torch.randint(0, 256, (1, 360, 640, 3), dtype=torch.uint8, generator=gen)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example,), strict=False)
    bad = [line.strip() for line in str(traced.inlined_graph).splitlines()
           if "Device = prim::Constant" in line or 'value="cpu"' in line or 'value="cuda' in line]
    if bad:
        print("traced MixVPR graph contains device constants:", *bad[:20], sep="\n  ", file=sys.stderr)
        return 1
    traced.save(args.out)

    loaded = torch.jit.load(args.out, map_location="cpu").eval()
    with torch.no_grad():
        other = torch.randint(0, 256, (1, 480, 752, 3), dtype=torch.uint8, generator=gen)
        d = (loaded(other) - wrapper(other)).abs().max().item()
    if d > 1e-5:
        print(f"traced MixVPR differs from the eager wrapper on a 480x752 input (max abs diff {d})", file=sys.stderr)
        return 1
    print(f"traced vs eager wrapper on 480x752 input: max abs diff {d:.2e}")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    worst = 1.0
    for dev in devices:
        ref = GlobalExtractor("mixvpr", torch.device(dev), args.weights)
        ts = torch.jit.load(args.out, map_location="cpu").eval().to(dev)
        for p in args.images:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"cannot read {p}", file=sys.stderr)
                return 1
            want = ref(bgr)
            with torch.no_grad():
                got = ts(to_input(bgr).to(dev))[0].cpu().numpy()
            cos = float(np.dot(want, got) / (np.linalg.norm(want) * np.linalg.norm(got)))
            worst = min(worst, cos)
            print(f"{dev}: {os.path.basename(os.path.dirname(os.path.dirname(p)))}/{os.path.basename(p)} "
                  f"{bgr.shape[1]}x{bgr.shape[0]} dim={got.shape[0]} |d|={np.linalg.norm(got):.4f} cosine={cos:.6f}")
    if worst < 0.999:
        print(f"cosine to the offline extractor {worst:.6f} < 0.999", file=sys.stderr)
        return 1
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"min cosine {worst:.6f} over {len(args.images)} images x {devices}")
    print(f"MixVPR TorchScript model written to {args.out} (sha256 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
