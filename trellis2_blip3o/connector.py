"""TRELLIS2Connector — projects VLM hidden states into TRELLIS.2 SS Flow cond space.

DISTRIBUTION MATCHING (critical):
  TRELLIS.2 SS Flow's cross-attn was pretrained on
      F.layer_norm(dinov3_features, [1024])
  (see trellis2/modules/image_feature_extractor.py:92 — DinoV3FeatureExtractor).
  That gives per-token mean=0, std=1 along the feature dim.

  Our connector ends with `nn.LayerNorm(1024, elementwise_affine=True)` initialized
  to the identity transform (weight=ones, bias=zeros, which torch's LayerNorm does
  by default). At step 0 the output distribution exactly matches what SS Flow's
  cross-attn was trained on. During training the affine can drift if needed.

STRUCTURE (mirrors BLIP3o-NEXT's diffusion_connector):
  Linear(vlm_dim -> cond_dim) -> GELU -> Linear(cond_dim -> cond_dim) -> LayerNorm

  BLIP3o-NEXT used `RMSNorm(2304, weight=sqrt(5.5))` because Sana DiT's cross-attn
  was trained on Gemma-2 hidden (RMS≈sqrt(5.5), un-centered). We use LayerNorm
  because DINOv3 is layer-normed (centered, std=1) — different downstream pretrain
  → different norm type, same methodology.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TRELLIS2Connector(nn.Module):
    """VLM hidden (B, T, vlm_dim) → TRELLIS.2 cond (B, T, cond_dim).

    Args:
        vlm_hidden_dim: VLM final-layer hidden size.
                        Qwen3.5-2B = 2048 (current supported backbone).
                        # DEPRECATED examples: BLIP3o-NEXT-Pretrain-3B = 2048; Qwen3-VL-8B = 3584.
        trellis_cond_dim: SS Flow cond_channels (1024 across TRELLIS.2-4B stages).
    """

    def __init__(self, vlm_hidden_dim: int = 2048, trellis_cond_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(vlm_hidden_dim, trellis_cond_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(trellis_cond_dim, trellis_cond_dim)
        # LayerNorm with default init (weight=1, bias=0) — exactly DINOv3's
        # unaffine F.layer_norm distribution at step 0, learnable thereafter.
        self.out_norm = nn.LayerNorm(trellis_cond_dim, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.fc1(hidden_states)
        x = self.act(x)
        x = self.fc2(x)
        return self.out_norm(x)
