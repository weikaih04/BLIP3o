"""MMDiT (double-stream joint-attention) variant of the TRELLIS.2 SS flow — ablation arm ②.

Deep-aligned to the verified reference implementations (2026-07-03):
  • Qwen-Image  diffusers `transformer_qwenimage.py::QwenImageTransformerBlock`
      — per-stream (shift,scale,gate)×2 modulation; norm→mod→JOINT-ATTN→gate→res;
        norm→mod→MLP→gate→res for BOTH streams; streams same width; both FFN ratio 4.
  • Ψ0 (USC)    `Psi0/src/psi/models/psi0.py::VLATransformerBlock/JointVLAAttnProcessor`
      — SD3 JointAttnProcessor: sample to_q/k/v + context add_{q,k,v}_proj → concat →
        ONE sdpa → split → per-stream out (to_out / to_add_out); context stream co-evolves
        (own FFN, same dim, ratio 4); `context_pre_only` on the last block.
  • π0 contrast (arm ③) is NOT this file: there the cond K/V come from the frozen VLM's
    per-layer states; here cond enters ONCE (bottom) and co-evolves — Qwen-Image/Ψ0 lineage.

Graft-onto-TRELLIS specifics (the retrofit part, absent from the from-scratch references):
  • x-stream params are the PRETRAINED TRELLIS block (to_qkv/q_rms/k_rms/to_out/mlp/
    modulation); op order preserved exactly (LayerNorm32, MultiHeadRMSNorm, RoPE, tanh-GELU).
  • The pretrained cross_attn step is REMOVED (its role subsumed by joint attention);
    its to_q/to_kv/to_out and norm2 are the only pretrained weights dropped.
  • x→c attention logits carry a learnable per-head bias `c_bias` init −10: at init the
    x-stream forward ≈ pretrained-with-cross-zeroed (leak e^-10 ≈ 5e-5, far below bf16
    noise); as c_bias trains toward 0 the math becomes EXACTLY the SD3/Ψ0 joint softmax.
    `fold_c_bias_report()` says when it is foldable for export.
  • c-stream modulation mirrors the x-stream's pretrained share_mod scheme (one shared
    SiLU+Linear at model level + per-block learnable offsets) instead of SD3's per-block
    adaLN Linears — the one intentional deviation (symmetry with our pretrained x scheme;
    saves ~425M params that would dwarf the ablation).

Build for training via `SparseStructureFlowMMDiT.from_cross_dit(pretrained_ss_flow)`.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _paths  # noqa: F401  (adds TRELLIS.2 to sys.path)
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel  # type: ignore
from trellis2.modules.attention.modules import MultiHeadRMSNorm  # type: ignore
from trellis2.modules.attention import RotaryPositionEmbedder  # type: ignore
from trellis2.modules.norm import LayerNorm32  # type: ignore
from trellis2.modules.transformer.blocks import FeedForwardNet  # type: ignore
from trellis2.modules.utils import manual_cast  # type: ignore


class SSJointAttention(nn.Module):
    """Joint attention over [x ; c] with per-stream projections (SD3 JointAttnProcessor
    naming: sample = to_qkv/to_out, context = add_{q,k,v}_proj / to_add_out) plus the
    retrofit `c_bias` on x→c logits.

    x-side math replicates TRELLIS MultiHeadAttention(self, qk_rms_norm, rope) exactly:
    to_qkv → reshape(B,L,3,H,D) → unbind → q/k MultiHeadRMSNorm → RoPE → sdpa → to_out.
    c tokens get NO RoPE (identity — FLUX-style; Qwen-Image assigns text separate ids).
    """

    def __init__(self, channels: int, num_heads: int, qkv_bias: bool = True,
                 qk_rms_norm: bool = True, context_pre_only: bool = False):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qk_rms_norm = qk_rms_norm
        self.context_pre_only = context_pre_only

        # ---- x stream (weights migrated from pretrained TRELLIS self_attn) ----
        self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

        # ---- c stream (fresh; SD3/diffusers naming). context_pre_only blocks only READ
        # the context (K/V) — like diffusers, they get no add_q_proj / to_add_out. ----
        self.add_k_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.add_v_proj = nn.Linear(channels, channels, bias=qkv_bias)
        if qk_rms_norm:  # mirror x-side qk-norm (SD3's norm_added_k)
            self.norm_added_k = MultiHeadRMSNorm(self.head_dim, num_heads)
        if not context_pre_only:
            self.add_q_proj = nn.Linear(channels, channels, bias=qkv_bias)
            if qk_rms_norm:
                self.norm_added_q = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.to_add_out = nn.Linear(channels, channels)

        # retrofit gate: additive logit bias for x-queries → c-keys, per head, init −10.
        # single-softmax form → converges to the exact SD3 joint softmax as it → 0.
        self.c_bias = nn.Parameter(torch.full((num_heads,), -10.0))

    def forward(self, hx: torch.Tensor, hc: torch.Tensor,
                phases: Optional[torch.Tensor] = None,
                cond_mask: Optional[torch.Tensor] = None):
        """hx: (B,Lx,C) modulated x; hc: (B,Lc,C) modulated c;
        phases: RoPE phases for x tokens; cond_mask: (B,Lc) bool, True = real token.
        Returns (out_x, out_c) — out_c is None when context_pre_only."""
        B, Lx, C = hx.shape
        Lc = hc.shape[1]
        H, D = self.num_heads, self.head_dim

        qkv = self.to_qkv(hx).reshape(B, Lx, 3, H, D)
        q_x, k_x, v_x = qkv.unbind(dim=2)                       # (B,Lx,H,D)
        if self.qk_rms_norm:
            q_x = self.q_rms_norm(q_x)
            k_x = self.k_rms_norm(k_x)
        if phases is not None:
            q_x = RotaryPositionEmbedder.apply_rotary_embedding(q_x, phases)
            k_x = RotaryPositionEmbedder.apply_rotary_embedding(k_x, phases)

        k_c = self.add_k_proj(hc).reshape(B, Lc, H, D)
        v_c = self.add_v_proj(hc).reshape(B, Lc, H, D)
        if self.qk_rms_norm:
            k_c = self.norm_added_k(k_c)
        if not self.context_pre_only:
            q_c = self.add_q_proj(hc).reshape(B, Lc, H, D)
            if self.qk_rms_norm:
                q_c = self.norm_added_q(q_c)
            q_c = q_c.permute(0, 2, 1, 3)
        # (no RoPE on c)

        # → (B,H,L,D) for F.scaled_dot_product_attention
        q_x, k_x, v_x, k_c, v_c = (t.permute(0, 2, 1, 3)
                                   for t in (q_x, k_x, v_x, k_c, v_c))
        k = torch.cat([k_x, k_c], dim=2)                        # (B,H,Lx+Lc,D)
        v = torch.cat([v_x, v_c], dim=2)

        # additive mask on the c-columns: c_bias (retrofit gate) + padding −inf
        neg_inf = torch.finfo(hx.dtype).min
        bias_c = self.c_bias.to(hx.dtype).view(1, H, 1, 1).expand(B, H, 1, Lc)
        if cond_mask is not None:
            pad = torch.where(cond_mask[:, None, None, :],
                              torch.zeros((), dtype=hx.dtype, device=hx.device),
                              torch.full((), neg_inf, dtype=hx.dtype, device=hx.device))
            bias_c = bias_c + pad
        zeros_x = torch.zeros(B, H, 1, Lx, dtype=hx.dtype, device=hx.device)

        # x-queries: ONE softmax over [x ; c] with c_bias added on the c part
        mask_xq = torch.cat([zeros_x, bias_c], dim=-1)          # (B,H,1,Lx+Lc)
        out_x = F.scaled_dot_product_attention(q_x, k, v, attn_mask=mask_xq)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, Lx, C)
        out_x = self.to_out(out_x)

        if self.context_pre_only:
            return out_x, None

        # c-queries: plain joint softmax over [x ; c] (padding mask only, no c_bias —
        # the c stream is fresh, nothing to preserve; exactly SD3)
        if cond_mask is not None:
            pad = torch.where(cond_mask[:, None, None, :],
                              torch.zeros((), dtype=hx.dtype, device=hx.device),
                              torch.full((), neg_inf, dtype=hx.dtype, device=hx.device))
            mask_cq = torch.cat([torch.zeros(B, 1, 1, Lx, dtype=hx.dtype, device=hx.device),
                                 pad], dim=-1).expand(B, 1, Lc, Lx + Lc)
        else:
            mask_cq = None
        out_c = F.scaled_dot_product_attention(q_c, k, v, attn_mask=mask_cq)
        out_c = out_c.permute(0, 2, 1, 3).reshape(B, Lc, C)
        out_c = self.to_add_out(out_c)
        return out_x, out_c


class SSJointBlock(nn.Module):
    """Double-stream MMDiT block grafted onto a pretrained ModulatedTransformerCrossBlock.

    x-stream op order = the TRELLIS block minus its cross_attn step:
        x += gate_msa · JOINTATTN_x(mod(norm1(x)))           ← was: self_attn
        (cross_attn step deleted)
        x += gate_mlp · mlp(mod(norm3(x)))
    c-stream mirrors Qwen-Image's txt stream with fresh params:
        c += c_gate_msa · JOINTATTN_c(mod_c(norm1_c(c)))
        c += c_gate_mlp · ffn_c(mod_c(norm2_c(c)))
    Both modulations are share_mod style: (per-block offset + shared model-level mod).chunk(6).
    """

    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 5.3334,
                 mlp_ratio_c: float = 4.0, qkv_bias: bool = True, qk_rms_norm: bool = True,
                 use_checkpoint: bool = False, context_pre_only: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.context_pre_only = context_pre_only

        # ---- x stream (migrated; FeedForwardNet = TRELLIS's own → keys + tanh-GELU match) ----
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

        # ---- c stream (fresh; FFN ratio 4, tanh-GELU — matches Qwen-Image/Ψ0 FeedForward) ----
        self.norm1_c = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        if not context_pre_only:
            self.norm2_c = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
            self.ffn_c = FeedForwardNet(channels, mlp_ratio=mlp_ratio_c)
        self.modulation_c = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

        self.attn = SSJointAttention(channels, num_heads, qkv_bias=qkv_bias,
                                     qk_rms_norm=qk_rms_norm,
                                     context_pre_only=context_pre_only)

    def _forward(self, x, c, mod, mod_c, phases, cond_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
            (self.modulation_c + mod_c).type(mod_c.dtype).chunk(6, dim=1)

        hx = self.norm1(x)
        hx = hx * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        hc = self.norm1_c(c)
        hc = hc * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)

        out_x, out_c = self.attn(hx, hc, phases=phases, cond_mask=cond_mask)
        x = x + out_x * gate_msa.unsqueeze(1)
        if not self.context_pre_only:
            c = c + out_c * c_gate_msa.unsqueeze(1)

        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + h * gate_mlp.unsqueeze(1)

        if not self.context_pre_only:
            hc2 = self.norm2_c(c)
            hc2 = hc2 * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1)
            c = c + self.ffn_c(hc2) * c_gate_mlp.unsqueeze(1)
        return x, c

    def forward(self, x, c, mod, mod_c, phases=None, cond_mask=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, c, mod, mod_c, phases, cond_mask, use_reentrant=False)
        return self._forward(x, c, mod, mod_c, phases, cond_mask)


class SparseStructureFlowMMDiT(SparseStructureFlowModel):
    """SS flow with the cross-attn blocks swapped for double-stream MMDiT blocks.

    Reuses the parent's patchify/pos-emb/t-embedder/rope/out layers verbatim. Adds the
    Qwen-Image-style context intake (txt_norm RMSNorm + txt_in Linear cond→model dim) and
    a c-side shared modulation head mirroring the pretrained share_mod scheme.
    Same forward signature as the parent → drop-in for the existing loss/trainer wiring.
    """

    def __init__(self, *args, mlp_ratio_c: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.share_mod, "SSJointBlock assumes share_mod=True (the SS config)"
        ch, nh = self.model_channels, self.num_heads
        # derive block config from the parent-built cross blocks, then replace them
        n = len(self.blocks)
        ref = self.blocks[0]
        mlp_ratio = ref.mlp.mlp[0].out_features / ch
        qk_rms = ref.self_attn.qk_rms_norm
        qkv_bias = ref.self_attn.to_qkv.bias is not None
        use_ckpt = ref.use_checkpoint
        self.blocks = nn.ModuleList([
            SSJointBlock(ch, nh, mlp_ratio=mlp_ratio, mlp_ratio_c=mlp_ratio_c,
                         qkv_bias=qkv_bias, qk_rms_norm=qk_rms, use_checkpoint=use_ckpt,
                         context_pre_only=(i == n - 1))
            for i in range(n)
        ])
        # context intake (Qwen-Image: txt_norm → txt_in)
        self.txt_norm = nn.RMSNorm(self.cond_channels, eps=1e-6)
        self.txt_in = nn.Linear(self.cond_channels, ch)
        # c-side shared modulation head (mirrors the parent's share_mod adaLN_modulation)
        self.adaLN_modulation_c = nn.Sequential(nn.SiLU(), nn.Linear(ch, 6 * ch, bias=True))
        # parent ran convert_to(dtype) on the OLD blocks before we replaced them
        self.convert_to(self.dtype)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
            f"Input shape mismatch, got {x.shape}"
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None]
        t_emb_raw = self.t_embedder(t)
        mod = self.adaLN_modulation(t_emb_raw)          # x-side shared mod (pretrained)
        mod_c = self.adaLN_modulation_c(t_emb_raw)      # c-side shared mod (fresh)

        mod = manual_cast(mod, self.dtype)
        mod_c = manual_cast(mod_c, self.dtype)
        h = manual_cast(h, self.dtype)

        # context intake runs in fp32 (model-level, like the parent's cond heads), THEN cast
        c = manual_cast(self.txt_in(self.txt_norm(cond)), self.dtype)
        for block in self.blocks:
            h, c = block(h, c, mod, mod_c, phases=self.rope_phases, cond_mask=cond_mask)

        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()
        return h

    # ---------- migration ----------
    @classmethod
    def from_cross_dit(cls, src: SparseStructureFlowModel,
                       mlp_ratio_c: float = 4.0) -> "SparseStructureFlowMMDiT":
        """Build an MMDiT model from a pretrained cross-attn SS flow, migrating every
        x-stream weight; c-stream fresh; cross_attn (to_q/to_kv/to_out + norm2) dropped."""
        ref = src.blocks[0]
        dtype_str = {torch.float32: "float32", torch.float16: "float16",
                     torch.bfloat16: "bfloat16"}[src.dtype]
        dst = cls(
            resolution=src.resolution, in_channels=src.in_channels,
            model_channels=src.model_channels, cond_channels=src.cond_channels,
            out_channels=src.out_channels, num_blocks=len(src.blocks),
            num_heads=src.num_heads,
            mlp_ratio=ref.mlp.mlp[0].out_features / src.model_channels,
            pe_mode=src.pe_mode, dtype=dtype_str, use_checkpoint=ref.use_checkpoint,
            share_mod=src.share_mod, qk_rms_norm=ref.self_attn.qk_rms_norm,
            qk_rms_norm_cross=False, mlp_ratio_c=mlp_ratio_c,
        )
        # model-level (shared) pieces
        dst.input_layer.load_state_dict(src.input_layer.state_dict())
        dst.out_layer.load_state_dict(src.out_layer.state_dict())
        dst.t_embedder.load_state_dict(src.t_embedder.state_dict())
        dst.adaLN_modulation.load_state_dict(src.adaLN_modulation.state_dict())
        if src.pe_mode == "ape":
            dst.pos_emb.data.copy_(src.pos_emb.data)

        # per-block x-stream migration
        for db, sb in zip(dst.blocks, src.blocks):
            db.modulation.data.copy_(sb.modulation.data)
            db.attn.to_qkv.load_state_dict(sb.self_attn.to_qkv.state_dict())
            db.attn.to_out.load_state_dict(sb.self_attn.to_out.state_dict())
            if sb.self_attn.qk_rms_norm:
                db.attn.q_rms_norm.load_state_dict(sb.self_attn.q_rms_norm.state_dict())
                db.attn.k_rms_norm.load_state_dict(sb.self_attn.k_rms_norm.state_dict())
            db.mlp.load_state_dict(sb.mlp.state_dict())
            # (sb.cross_attn + sb.norm2 intentionally dropped)
        return dst

    def fold_c_bias_report(self) -> str:
        """When |c_bias| ≈ 0 everywhere the model IS a vanilla SD3 joint block — the bias
        can be dropped at export. Report current magnitudes."""
        mags = torch.stack([b.attn.c_bias.detach().abs().max() for b in self.blocks])
        return (f"c_bias |max| per block: min {mags.min():.3f} max {mags.max():.3f} "
                f"(init 10.0; foldable when ≈0)")
