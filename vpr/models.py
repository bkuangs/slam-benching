"""Global-descriptor extractors (torch): AnyLoc (DINOv2 + VLAD), MixVPR, CosPlace; plus DBoW2 ORB features.

All extractors take a BGR uint8 image (OpenCV) and return a float32 numpy vector (or BoW tuple).
"""
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(bgr: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1)[None].float().div(255)
    return ((x - MEAN) / STD).to(device)


# --------------------------------------------------------------------------------------------- AnyLoc
class DinoV2Facet:
    """DINOv2 ViT patch descriptors of one layer/facet (AnyLoc: ViT-G/14, layer 31, 'value'), L2-normalised.

    Runs the transformer only up to the requested layer. Input is center-cropped to a multiple of 14."""

    def __init__(self, device, arch="dinov2_vitg14", layer=31, facet="value"):
        self.dev, self.layer, self.facet, self.arch = device, layer, facet, arch
        self.model = torch.hub.load("facebookresearch/dinov2", arch).eval().to(device)
        self._out = None
        blk = self.model.blocks[layer]
        if facet in ("query", "key", "value"):
            blk.attn.qkv.register_forward_hook(lambda m, i, o: setattr(self, "_out", o))
        else:
            blk.register_forward_hook(lambda m, i, o: setattr(self, "_out", o))

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> torch.Tensor:
        h, w = bgr.shape[:2]
        h14, w14 = h // 14 * 14, w // 14 * 14
        y0, x0 = (h - h14) // 2, (w - w14) // 2
        x = to_tensor(bgr[y0:y0 + h14, x0:x0 + w14], self.dev)
        m = self.model
        t = m.prepare_tokens_with_masks(x)
        for blk in m.blocks[:self.layer + 1]:
            t = blk(t)
        o = self._out
        if self.facet in ("query", "key", "value"):
            C = m.embed_dim
            o = o.reshape(o.shape[0], o.shape[1], 3, C)[:, :, ("query", "key", "value").index(self.facet)]
        o = o[0, 1 + getattr(m, "num_register_tokens", 0):]  # drop CLS (+ registers)
        return F.normalize(o.float(), dim=-1)  # (n_patches, C)


def kmeans_cosine(x: torch.Tensor, k: int, iters: int = 100, seed: int = 0) -> torch.Tensor:
    """Spherical k-means (cosine) with k-means++ seeding; returns unit-norm centroids (k, D)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = F.normalize(x.float(), dim=-1)
    n = len(x)
    c = [x[torch.randint(n, (1,), generator=g).item()]]
    d = torch.full((n,), float("inf"), device=x.device)
    for _ in range(1, k):
        d = torch.minimum(d, 1 - x @ c[-1])
        p = d.clamp(min=0).cpu().double()
        c.append(x[torch.multinomial(p / p.sum(), 1, generator=g).item()])
    c = torch.stack(c)
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        new = torch.zeros_like(c).index_add_(0, a, x)
        empty = torch.bincount(a, minlength=k) == 0
        new[empty] = c[empty]
        new = F.normalize(new, dim=-1)
        if torch.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def vlad(desc: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """AnyLoc hard-assignment VLAD: residual sums per cosine-nearest centroid, intra-normalised, L2."""
    a = (desc @ centroids.T).argmax(1)
    v = torch.zeros_like(centroids).index_add_(0, a, desc - centroids[a])
    v = F.normalize(v, dim=-1)
    return F.normalize(v.flatten(), dim=0)


# --------------------------------------------------------------------------------------------- MixVPR
def _load_file(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0] + "_mixvpr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class MixVPRModel(torch.nn.Module):
    """ResNet50 (cropped after layer3) + MixVPR aggregator (4 mixers, 1024 ch x 4 rows = 4096-d), 320x320."""

    def __init__(self, repo_dir: str, ckpt: str):
        super().__init__()
        # load the two module files directly (the repo's package __init__ files import timm etc.)
        MixVPR = _load_file(os.path.join(repo_dir, "models/aggregators/mixvpr.py")).MixVPR
        ResNet = _load_file(os.path.join(repo_dir, "models/backbones/resnet.py")).ResNet
        self.backbone = ResNet("resnet50", pretrained=False, layers_to_freeze=2, layers_to_crop=[4])
        self.aggregator = MixVPR(in_channels=1024, in_h=20, in_w=20, out_channels=1024, mix_depth=4,
                                 mlp_ratio=1, out_rows=4)
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"MixVPR checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    def forward(self, x):
        return self.aggregator(self.backbone(x))


class GlobalExtractor:
    """Uniform wrapper: name -> callable(bgr) -> np.float32 vector (L2-normalised)."""

    def __init__(self, name: str, device, weights_dir: str):
        self.name, self.dev = name, device
        if name == "mixvpr":
            self.model = MixVPRModel(os.path.join(weights_dir, "MixVPR"),
                                     os.path.join(weights_dir, "mixvpr_resnet50_4096.ckpt")).eval().to(device)
        elif name == "cosplace":
            self.model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50",
                                        fc_output_dim=2048).eval().to(device)
        else:
            raise ValueError(name)

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        x = to_tensor(bgr, self.dev)
        if self.name == "mixvpr":  # paper / demo.py: resize to 320x320 (aspect not preserved)
            x = F.interpolate(x, size=(320, 320), mode="bilinear", align_corners=False, antialias=True)
        d = F.normalize(self.model(x).float(), dim=-1)[0]
        return d.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------------------------- ORB for DBoW2
class OrbSlamLikeORB:
    """ORB keypoints spread over the image as ORB-SLAM3's octree distribution does (approximation: detect
    3x the budget with OpenCV ORB, FAST threshold 7, then take keypoints round-robin over 40x40 px cells by
    response), 1200 features, scale 1.2, 8 levels - the ORB-SLAM3/SP_SLAM3 config budget."""

    def __init__(self, n=1200, cell=40):
        self.n, self.cell = n, cell
        self.orb = cv2.ORB_create(nfeatures=3 * n, scaleFactor=1.2, nlevels=8, fastThreshold=7)

    def __call__(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kps = self.orb.detect(g, None)
        if len(kps) > self.n:
            pts = np.array([k.pt for k in kps])
            resp = np.array([k.response for k in kps])
            cell = (pts[:, 1] // self.cell).astype(int) * 1000 + (pts[:, 0] // self.cell).astype(int)
            order = np.lexsort((-resp, cell))
            rank = np.empty(len(kps), int)
            _, first = np.unique(cell[order], return_index=True)
            starts = np.repeat(first, np.diff(np.r_[first, len(order)]))
            rank[order] = np.arange(len(order)) - starts
            keep = np.lexsort((-resp, rank))[:self.n]
            kps = [kps[i] for i in keep]
        kps, desc = self.orb.compute(g, kps)
        return desc
