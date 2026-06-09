"""DINOv3 conditioning alignment (REPA-inspired) — see DINO_ALIGNMENT_DESIGN.md.

Gives the connector a DIRECT regression target: make `h_φ(connector(Qwen))` match
DINOv3(render) patch features, the exact distribution TRELLIS's frozen flow
cross-attn was pretrained on. Train-only auxiliary loss; head dropped at inference.

Per-task alignment (the part that matters):
  • image_to_3d:        align the image's vision-token block of the cond to DINOv3(render).
  • multi_image_to_3d:  EACH view's vision-token block → THAT view's DINOv3 features.
                        Qwen lays each image's vision tokens as a contiguous block;
                        block size = (t·h·w)/merge² from image_grid_thw, in flat order.
  • text_to_3d:         no image_pad tokens → no DINOv3 target → contributes nothing
                        (connector still gets pure flow-loss gradient on these steps).

DINOv3 model = the SAME one TRELLIS used (facebook/dinov3-vitl16-pretrain-lvd1689m
@512), reused via trellis2.modules.image_feature_extractor.DinoV3FeatureExtractor —
aligning to any other encoder would defeat the purpose.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _paths  # noqa: F401  — sets sys.path so trellis2 imports

# TRELLIS's exact DINOv3 wrapper + the model_name from its ss_flow config.
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor  # type: ignore

# facebook/dinov3-vitl16-pretrain-lvd1689m is gated; use the byte-identical
# non-gated mirror that's already cached locally (same DINOv3-L/16 weights TRELLIS used).
TRELLIS_DINOV3_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_IMAGE_SIZE = 512
DINO_DIM = 1024     # DINOv3 ViT-L feature dim
GRID = DINOV3_IMAGE_SIZE // 16   # 32×32 patch grid for the 512px DINOv3


class _ProjHead(nn.Module):
    """REPA's projection head h_φ — 3-layer MLP + SiLU. Train-only; aligns
    h_φ(cond) to DINOv3 so `cond` only needs to be DINOv3-MAPPABLE, not equal
    (this is the text-protection mechanism)."""

    def __init__(self, in_dim: int = 1024, hidden: int = 2048, out_dim: int = DINO_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DinoAligner(nn.Module):
    """Frozen DINOv3 extractor + projection head + the per-task alignment loss.

    mode:
      "spatial" (default) — reshape each image's Qwen vision tokens to their (h,w)
        grid, bilinear-resize to DINOv3's 32×32, patch-wise cosine. Targets local
        geometry (the jagged/holey roughness).
      "pooled" — mean-pool each image's tokens on both sides, cosine. Coarser,
        simplest, lowest-risk; aligns global semantic only.
    """

    def __init__(self, cond_dim: int = 1024, mode: str = "spatial",
                 model_name: str = TRELLIS_DINOV3_NAME):
        super().__init__()
        if mode not in ("spatial", "pooled"):
            raise ValueError(f"mode must be 'spatial' or 'pooled', got {mode!r}")
        self.mode = mode
        self.head = _ProjHead(in_dim=cond_dim, out_dim=DINO_DIM)
        # Frozen DINOv3 (not an nn.Module child → keep it off the state_dict /
        # DeepSpeed param partitioning; it's inference-only).
        self._extractor = DinoV3FeatureExtractor(model_name, image_size=DINOV3_IMAGE_SIZE)
        self._extractor.model.eval()
        for p in self._extractor.model.parameters():
            p.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # move the (non-child) DINOv3 too
        dev = None
        for a in args:
            if isinstance(a, (str, torch.device)):
                dev = a
        if dev is not None:
            self._extractor.to(dev)
        return self

    @torch.no_grad()
    def _dino_patches(self, dino_images: torch.Tensor) -> torch.Tensor:
        """(M,3,512,512) → (M, 32, 32, DINO_DIM). Drops CLS/register prefix tokens,
        keeps the trailing GRID² patch tokens, reshapes to the spatial grid.

        DINOv3 is frozen float32. The training forward runs under bf16 autocast,
        which would feed the DINOv3 conv a bf16 input against its float32 bias
        (→ dtype mismatch). Force float32 + autocast OFF so DINOv3 runs cleanly,
        then return float32 patches (caller casts to cond dtype for the cosine)."""
        # Ensure frozen DINOv3 is on the same device as the input (it's a non-child
        # plain object, so model.to()/.cuda() may not have moved it — move lazily).
        w = self._extractor.model.embeddings.patch_embeddings.weight
        if w.device != dino_images.device:
            self._extractor.model.to(dino_images.device)
            w = self._extractor.model.embeddings.patch_embeddings.weight
        with torch.autocast(device_type="cuda", enabled=False):
            feats = self._extractor(dino_images.to(device=w.device, dtype=w.dtype))  # (M,N,DINO_DIM)
        patches = feats[:, -GRID * GRID:, :].float()           # trailing GRID² patch tokens
        M = patches.shape[0]
        return patches.reshape(M, GRID, GRID, DINO_DIM)

    def _zero_touch(self, cond: torch.Tensor) -> torch.Tensor:
        """A 0-valued loss for text/no-image steps that routes through the SAME
        structure image steps use: connector-output → head.forward → sum → ×0.

        Why through head.FORWARD (not just sum(head.params)×0): under DeepSpeed
        ZeRO-2, the autograd-graph STRUCTURE must be consistent across steps. Image
        steps backprop head via `head(cond_block)`; if text steps instead touch the
        head PARAMS directly, the head.forward node is absent from the text-step
        graph → the image↔text transition changes the graph → ZeRO-2 deadlocks at
        the gradient reduction (reproduced: image-only OK, text-only OK, MIXED hangs).
        Running head on a 1-token slice of cond keeps connector→head.forward in the
        graph every step (zero gradient, ~free)."""
        return (self.head(cond[:, :1, :]).sum() * 0.0).to(cond.dtype)

    def forward(
        self,
        cond: torch.Tensor,                 # (B, T, cond_dim) — AFTER connector
        input_ids: torch.LongTensor,        # (B, T)
        image_grid_thw: Optional[torch.Tensor],   # (M, 3) flat across batch, or None
        dino_images: Optional[torch.Tensor],      # (M, 3, 512, 512) flat across batch, or None
        image_token_id: int,
        merge_size: int,
    ) -> torch.Tensor:
        """Mean cosine-alignment loss over ALL images in the batch (image &
        multi-image samples). 0 for text-only batches — but the zero STILL touches
        the head params (see _zero_touch) so they're never 'unused parameters':
        otherwise DDP/DeepSpeed hangs at the gradient reduction on text steps
        waiting for the head's gradient that never comes."""
        if image_grid_thw is None or dino_images is None or dino_images.numel() == 0:
            return self._zero_touch(cond)

        dino_grids = self._dino_patches(dino_images)   # (M,32,32,D) float32 (DINOv3 frozen fp32)
        ms2 = merge_size * merge_size
        B, T, _ = cond.shape
        g = 0   # global flat image index (matches image_grid_thw / dino_images order)
        cos_terms: List[torch.Tensor] = []

        for b in range(B):
            pos = (input_ids[b] == image_token_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                continue   # text-only sample → no alignment (text protection)
            p = 0
            while p < pos.numel():
                t, h, w = [int(x) for x in image_grid_thw[g]]
                n = (t * h * w) // ms2                # this image's vision-token count
                block = pos[p:p + n]                  # its positions in the sequence
                cond_block = cond[b, block]           # (n, cond_dim)
                hh, ww = h // merge_size, w // merge_size
                proj = self.head(cond_block).float()  # (n, DINO_DIM) — fp32 for cosine vs fp32 dino
                dino = dino_grids[g].float()          # (32,32,DINO_DIM)

                if self.mode == "pooled":
                    a = proj.mean(0)                  # (D,)
                    d = dino.reshape(-1, DINO_DIM).mean(0)
                    cos_terms.append(F.cosine_similarity(a, d, dim=0))
                else:  # spatial: resize cond grid → DINOv3 32×32, patch-wise cosine
                    cond_grid = proj.reshape(hh, ww, DINO_DIM).permute(2, 0, 1).unsqueeze(0)
                    cond_rs = F.interpolate(cond_grid, size=(GRID, GRID),
                                            mode="bilinear", align_corners=False)
                    cond_rs = cond_rs.squeeze(0).permute(1, 2, 0)     # (32,32,D)
                    cs = F.cosine_similarity(
                        cond_rs.reshape(-1, DINO_DIM),
                        dino.reshape(-1, DINO_DIM), dim=-1)           # (1024,)
                    cos_terms.append(cs.mean())
                p += n
                g += 1

        if not cos_terms:
            return self._zero_touch(cond)
        # maximize cosine → loss = 1 - mean(cos)
        return 1.0 - torch.stack(cos_terms).mean()
