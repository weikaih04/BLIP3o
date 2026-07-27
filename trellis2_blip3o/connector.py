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

    def forward(self, hidden_states: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        # key_mask accepted for a uniform call signature with the Transformer adapter;
        # the MLP is token-wise so it ignores it (no cross-token mixing → no padding leak).
        x = self.fc1(hidden_states)
        x = self.act(x)
        x = self.fc2(x)
        return self.out_norm(x)


class _AdapterBlock(nn.Module):
    """Pre-norm Transformer block with QK-Norm — matches i1's TextEncoderAdapterTransformer
    block (i1 torch_train/models/dit.py): `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`,
    bidirectional self-attn (ADAPTER, not LM → no causal mask), no in-adapter positional
    encoding (i1 applies RoPE only in the main DiT; position here is the separate pos_stamp axis).

    STANDARD init (i1 does NOT zero-init the residual branches): the two blocks are ACTIVE from
    step 0. An earlier zero-init version made the blocks no-ops at start (adapter degenerates to a
    linear read weaker than the MLP), which crippled early convergence in the 8000-step ablation —
    a false-negative risk. The output dist-match is still guaranteed by the adapter's out_norm."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(self.head_dim)   # QK-Norm (i1: use_qknorm on q,k)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )   # PyTorch default (kaiming) init — active from step 0, like i1.

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # (B, nH, T, hd)
        q = self.q_norm(q); k = self.k_norm(k)          # QK-Norm
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(o)
        x = x + self.mlp(self.norm2(x))
        return x


class TRELLIS2TransformerAdapter(nn.Module):
    """i1-style deep text adapter (i1 §3.1: 2-block Transformer, saturates >2 blocks).

    Same contract as TRELLIS2Connector — (B, T, vlm_dim) → (B, T, cond_dim), dist-matched
    to DINOv3's layer-norm at the output — but interposes n_blocks of self-attention + FFN
    so the language→generation interface has real capacity to reorganize VLM hidden into
    diffusion control signals (the i1 finding: the bottleneck is the interface, not model size).
    ~27M params at defaults (2 blocks, dim 1024) vs the MLP's 3.1M.

    key_mask (B, T) bool, True = real token: used as the self-attn key-padding mask so padded
    tokens don't leak into real ones. None → full attention (bs1 inference, no padding)."""

    def __init__(self, vlm_hidden_dim: int = 2048, trellis_cond_dim: int = 1024,
                 n_blocks: int = 2, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.in_proj = nn.Linear(vlm_hidden_dim, trellis_cond_dim)
        # i1 connector_in init: xavier_uniform weight + zero bias (i1DiT.init_weights).
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        self.blocks = nn.ModuleList([
            _AdapterBlock(trellis_cond_dim, n_heads, mlp_ratio) for _ in range(n_blocks)
        ])
        # dist-match output LayerNorm, identity init (same rationale as the MLP connector) —
        # guarantees mean0/std1 output for the pretrained cross-attn regardless of block content.
        # (i1 trains from scratch so needs no dist-match; our warm-started flow does.)
        self.out_norm = nn.LayerNorm(trellis_cond_dim, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_proj(hidden_states)
        attn_mask = None
        if key_mask is not None:
            # (B,1,1,T) bool broadcast over query dim — each query attends only to real keys.
            attn_mask = key_mask[:, None, None, :].to(torch.bool)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.out_norm(x)
