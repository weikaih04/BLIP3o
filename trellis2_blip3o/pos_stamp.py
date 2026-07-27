"""DINO-native position stamp for the qwen cond segment — prod port of the ablation's
crossdpos arm (memory blip3o-rope-position-hole: qwen hiddens are position-blind; stamping
the DINOv3 position signature the pretrained flow natively parses restored held-out IoU
0.202→0.264 at 16k scale). Applied AFTER the connector, on the 1024 image-token span.

Two layouts (same prompt, cache builder build_vlm_cache_v22.py):
  full sequence (training path, boiler kept):   image tokens at [10, 1034)
  keep-compacted (eval build_cond, boiler dropped): [7, 1031)
The signature: mean dino_hidden over 256 assets, centered (make_dino_pos.py)."""
import numpy as np
import torch
import torch.nn as nn

DPOS_NPZ = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/dino_pos32.npz"
IMG_SPAN_FULL = slice(10, 10 + 1024)   # train path (full seq incl. chat boilerplate)
IMG_SPAN_KEPT = slice(7, 7 + 1024)     # eval path (boiler-compacted)


class DinoPosStamp(nn.Module):
    """cond_q (B, T, 1024) → cond_q with pos signature added at `span`. Learnable scalar
    scale (init 1). Guarded no-op when T is too short (non-single-view / odd batches)."""

    def __init__(self):
        super().__init__()
        dp = np.load(DPOS_NPZ)["pos"].astype(np.float32)           # (1024, 1024)
        self.register_buffer("dpos", torch.from_numpy(dp))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, cond_q: torch.Tensor, span: slice) -> torch.Tensor:
        if cond_q.shape[1] < span.stop:
            return cond_q
        cond_q = cond_q.clone()
        cond_q[:, span] = cond_q[:, span] + self.scale.to(cond_q.dtype) * \
            self.dpos[None].to(cond_q.dtype)
        return cond_q
