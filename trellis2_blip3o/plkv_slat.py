"""Per-layer KV (arm ③, MolmoAct2/π0 lineage) for the TRELLIS.2 SLAT flows.

Verified references (2026-07-03, code-read):
  • MolmoAct2 (arXiv 2605.02881): each expert block cross-attends the VLM's K/V at the
    SAME depth — the literal mechanism implemented here (cross-attn form).
  • π0 (openpi `gemma.py`/`pi0.py`): the joint-attention form of the same idea — expert
    reads frozen VLM per-layer KV; blockwise-causal so the VLM never sees the expert
    (read-only, cacheable). NOT Ψ0 (USC) — Ψ0 reads hidden_states[-1] and co-evolves (arm ②).

Design (ablation-clean: arm ① vs ③ differ ONLY in the condition's depth/source):
  • The pretrained block machinery is kept VERBATIM — self_attn, cross_attn weights
    (to_q/to_kv/to_out + cross qk-rms), mlp, modulation. Block ℓ's cross-attn context:
        arm ①:  connector(VLM FINAL hidden)          — one tensor, all 30 blocks
        arm ③:  P_ℓ(tap_norm(VLM layer g(ℓ) hidden)) — per-block adapter + depth-mapped tap
    P_ℓ: Linear(vlm_dim→cond_channels) fresh (~63M total); the pretrained to_kv then reads
    P_ℓ's output — the learned "how to read cond" machinery is reused (warm), P_ℓ only has
    to map VLM layer-ℓ features into the pretrained cond space.
  • Real-ckpt lesson from arm ② applies here too: fresh adapters emit garbage at init and
    the pretrained cross-attn would ingest it → per-head ZERO-INIT gate on the cross-attn
    branch (g·per-head-out before to_out). g=0 ⇒ bit-exact pretrained backbone; gates open
    by gradient. Gate granularity matches arm ② (per head per block).
  • Tap mapping g(ℓ) = round(ℓ·(n_taps−1)/(num_blocks−1)) — 30 blocks onto K cached tap
    layers (default 8). n_taps=1 degenerates to arm ① routing (useful control).

Build via `SLatFlowPLKV.from_cross_dit(pretrained_slat_flow, vlm_dim=2048, n_taps=8)`.
forward(x, t, taps, ...) with taps (B, n_taps, Lc, vlm_dim) [+ cond_mask (B, Lc)].
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _paths  # noqa: F401
from trellis2.models.structured_latent_flow import SLatFlowModel  # type: ignore
from trellis2.modules import sparse as sp  # type: ignore
from trellis2.modules.norm import LayerNorm32  # type: ignore
from trellis2.modules.sparse.attention.modules import SparseMultiHeadRMSNorm  # type: ignore
from trellis2.modules.sparse.transformer.blocks import SparseFeedForwardNet  # type: ignore
from trellis2.modules.sparse.attention.modules import SparseMultiHeadAttention  # type: ignore
from trellis2.modules.attention.modules import MultiHeadRMSNorm  # type: ignore
from trellis2.modules.sparse.fused_modulate import (  # type: ignore
    fused_norm_modulate, fused_gate_residual)
from trellis2.modules.utils import manual_cast  # type: ignore

from .mmdit_slat import _seqlens, _pad, _unpad
from trellis2.modules.sparse.attention.full_attn import (  # type: ignore
    sparse_scaled_dot_product_attention as sparse_sdpa)


class GatedSparseCrossAttention(nn.Module):
    """The pretrained sparse cross-attention (to_q/to_kv/to_out + cross qk-rms) with
    zero-init gates making g=0 an EXACT no-op:

        out = W_out(g ⊙ attn_heads) + g_bias · b_out
        (g: per-head, init 0;  g_bias: scalar, init 0 — needed because to_out(0)=bias≠0)

    Both gates 0 ⇒ out ≡ 0 exactly (any weight scale — the arm-② real-ckpt lesson);
    (g=1, g_bias=1) recovers the pretrained cross-attn function exactly.
    """

    def __init__(self, channels: int, ctx_channels: int, num_heads: int,
                 qkv_bias: bool = True, qk_rms_norm: bool = True):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qk_rms_norm = qk_rms_norm
        # pretrained cross-attn params (names match SparseMultiHeadAttention for migration)
        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
        self.to_kv = nn.Linear(ctx_channels, channels * 2, bias=qkv_bias)
        if qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)
        # zero-init gates: per-head on the value path + scalar on to_out's bias
        self.c_gate = nn.Parameter(torch.zeros(num_heads))
        self.c_gate_bias = nn.Parameter(torch.zeros(()))

    def forward(self, hx: sp.SparseTensor, ctx: torch.Tensor,
                cond_mask: Optional[torch.Tensor] = None) -> sp.SparseTensor:
        """hx: normed sparse x; ctx: (B, Lc, ctx_ch) dense; cond_mask (B, Lc) bool."""
        H, D = self.num_heads, self.head_dim
        seqlens = _seqlens(hx)
        B, Lc = ctx.shape[0], ctx.shape[1]
        dt = hx.feats.dtype

        q = hx.replace(self.to_q(hx.feats).reshape(-1, H, D))
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
        kv = self.to_kv(ctx).reshape(B, Lc, 2, H, D)
        k, v = kv.unbind(dim=2)
        if self.qk_rms_norm:
            k = self.k_rms_norm(k)

        fast = hx.feats.is_cuda and dt in (torch.float16, torch.bfloat16)
        if fast:
            # flash-varlen via the TRELLIS wrapper (padding-free; masked cond DROPPED —
            # exactly the pretrained cross-attn's kv path, FA3-dispatched)
            if cond_mask is None:
                o = sparse_sdpa(q, k, v).feats                  # (N,H,D)
            else:
                k_vl = sp.VarLenTensor.from_tensor_list(
                    [k[b][cond_mask[b]] for b in range(B)])
                v_vl = sp.VarLenTensor.from_tensor_list(
                    [v[b][cond_mask[b]] for b in range(B)])
                o = sparse_sdpa(q, k_vl, v_vl).feats
        else:
            qp, _ = _pad(q.feats, seqlens)                      # (B,Lx,H,D)
            if cond_mask is not None:
                m = torch.where(cond_mask[:, None, None, :],
                                torch.zeros((), dtype=dt, device=ctx.device),
                                torch.full((), torch.finfo(dt).min, dtype=dt, device=ctx.device))
            else:
                m = None
            o = F.scaled_dot_product_attention(
                qp.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
                attn_mask=m)                                    # (B,H,Lx,D)
            o = _unpad(o.permute(0, 2, 1, 3), seqlens)          # (N,H,D)
        # per-head gate on values, bias gated separately → g=0 ⇒ EXACT zero output
        o = o * self.c_gate.to(dt).view(1, H, 1)
        out = F.linear(o.reshape(-1, H * D), self.to_out.weight) \
            + self.c_gate_bias.to(dt) * self.to_out.bias
        return hx.replace(out)


class SLatPLKVBlock(nn.Module):
    """The pretrained ModulatedSparseTransformerCrossBlock, verbatim op order, with the
    cross step swapped for the gated version (context supplied per block by the model)."""

    def __init__(self, channels: int, ctx_channels: int, num_heads: int,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, use_rope: bool = True,
                 rope_freq: Tuple[float, float] = (1.0, 10000.0), qk_rms_norm: bool = True,
                 qk_rms_norm_cross: bool = True, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels, num_heads=num_heads, type="self", attn_mode="full",
            qkv_bias=qkv_bias, use_rope=use_rope, rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm)
        self.cross_attn = GatedSparseCrossAttention(
            channels, ctx_channels, num_heads, qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross)
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x, mod, context, cond_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        h = fused_norm_modulate(x, self.norm1, scale_msa, shift_msa)
        h = self.self_attn(h)
        x = fused_gate_residual(x, h, gate_msa)
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context, cond_mask=cond_mask)
        x = x + h
        h = fused_norm_modulate(x, self.norm3, scale_mlp, shift_mlp)
        h = self.mlp(h)
        x = fused_gate_residual(x, h, gate_mlp)
        return x

    def forward(self, x, mod, context, cond_mask=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, cond_mask, use_reentrant=False)
        return self._forward(x, mod, context, cond_mask)


class SLatFlowPLKV(SLatFlowModel):
    """SLAT flow whose block-ℓ cross-attn reads the frozen VLM's layer-g(ℓ) hidden through
    a per-block adapter — MolmoAct2's per-layer KV, ablation arm ③."""

    def __init__(self, *args, vlm_dim: int = 2048, n_taps: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.share_mod, "SLatPLKVBlock assumes share_mod=True"
        self.vlm_dim = vlm_dim
        self.n_taps = n_taps
        ch, nh = self.model_channels, self.num_heads
        n = len(self.blocks)
        ref = self.blocks[0]
        self.blocks = nn.ModuleList([
            SLatPLKVBlock(ch, self.cond_channels, nh, mlp_ratio=self.mlp_ratio,
                          qkv_bias=ref.self_attn.to_qkv.bias is not None,
                          use_rope=ref.self_attn.use_rope,
                          qk_rms_norm=ref.self_attn.qk_rms_norm,
                          qk_rms_norm_cross=ref.cross_attn.qk_rms_norm,
                          use_checkpoint=ref.use_checkpoint)
            for _ in range(n)
        ])
        # per-block adapters: VLM layer hidden → pretrained cond space (read by to_kv)
        self.tap_norm = nn.RMSNorm(vlm_dim, eps=1e-6)
        self.tap_adapters = nn.ModuleList([
            nn.Linear(vlm_dim, self.cond_channels) for _ in range(n)])
        # block → tap index (linear depth mapping)
        self.tap_index = [round(i * (n_taps - 1) / max(n - 1, 1)) for i in range(n)]
        self.convert_to(self.dtype)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, taps: torch.Tensor,
                concat_cond: Optional[sp.SparseTensor] = None,
                cond_mask: Optional[torch.Tensor] = None, **kwargs) -> sp.SparseTensor:
        """taps: (B, n_taps, Lc, vlm_dim) — the frozen VLM's tap-layer hiddens."""
        assert taps.dim() == 4 and taps.shape[1] == self.n_taps, \
            f"taps must be (B,{self.n_taps},Lc,{self.vlm_dim}), got {tuple(taps.shape)}"
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        mod = manual_cast(self.adaLN_modulation(t_emb), self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)

        # adapt each USED tap once (fp32 model-level, like the parent's cond heads), then cast
        taps_n = self.tap_norm(taps)
        ctxs = {}
        for i, blk in enumerate(self.blocks):
            ti = self.tap_index[i]
            ctxs[i] = manual_cast(self.tap_adapters[i](taps_n[:, ti]), self.dtype)

        for i, blk in enumerate(self.blocks):
            h = blk(h, mod, ctxs[i], cond_mask=cond_mask)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h

    # ---------- migration ----------
    @classmethod
    def from_cross_dit(cls, src: SLatFlowModel, vlm_dim: int = 2048,
                       n_taps: int = 8) -> "SLatFlowPLKV":
        ref = src.blocks[0]
        dtype_str = {torch.float32: "float32", torch.float16: "float16",
                     torch.bfloat16: "bfloat16"}[src.dtype]
        dst = cls(
            resolution=src.resolution, in_channels=src.in_channels,
            model_channels=src.model_channels, cond_channels=src.cond_channels,
            out_channels=src.out_channels, num_blocks=len(src.blocks),
            num_heads=src.num_heads, mlp_ratio=src.mlp_ratio, pe_mode=src.pe_mode,
            dtype=dtype_str, use_checkpoint=ref.use_checkpoint, share_mod=src.share_mod,
            initialization=src.initialization, qk_rms_norm=src.qk_rms_norm,
            qk_rms_norm_cross=src.qk_rms_norm_cross, vlm_dim=vlm_dim, n_taps=n_taps,
        )
        dst.input_layer.load_state_dict(src.input_layer.state_dict())
        dst.out_layer.load_state_dict(src.out_layer.state_dict())
        dst.t_embedder.load_state_dict(src.t_embedder.state_dict())
        dst.adaLN_modulation.load_state_dict(src.adaLN_modulation.state_dict())
        if src.pe_mode == "ape":
            dst.pos_embedder.load_state_dict(src.pos_embedder.state_dict())

        for db, sb in zip(dst.blocks, src.blocks):
            db.modulation.data.copy_(sb.modulation.data)
            db.norm2.load_state_dict(sb.norm2.state_dict())
            db.self_attn.load_state_dict(sb.self_attn.state_dict())
            db.mlp.load_state_dict(sb.mlp.state_dict())
            # cross-attn weights into the gated version (gates stay 0)
            db.cross_attn.to_q.load_state_dict(sb.cross_attn.to_q.state_dict())
            db.cross_attn.to_kv.load_state_dict(sb.cross_attn.to_kv.state_dict())
            db.cross_attn.to_out.load_state_dict(sb.cross_attn.to_out.state_dict())
            if sb.cross_attn.qk_rms_norm:
                db.cross_attn.q_rms_norm.load_state_dict(sb.cross_attn.q_rms_norm.state_dict())
                db.cross_attn.k_rms_norm.load_state_dict(sb.cross_attn.k_rms_norm.state_dict())
        return dst

    def c_gate_report(self) -> str:
        g = torch.stack([b.cross_attn.c_gate.detach().abs().max() for b in self.blocks])
        gb = torch.stack([b.cross_attn.c_gate_bias.detach().abs() for b in self.blocks])
        return (f"c_gate |max| per block: min {g.min():.3f} max {g.max():.3f} | "
                f"bias-gate max {gb.max():.3f} (init 0 = pretrained backbone)")
