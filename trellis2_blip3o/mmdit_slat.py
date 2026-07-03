"""MMDiT (double-stream joint-attention) variant of the TRELLIS.2 SLAT flows — the
512-structure (shape) / texture SLAT DiT version of `mmdit_ss.py`. Ablation arm ② on the
sparse flows.

Same verified recipe as mmdit_ss.py (SD3/Qwen-Image/Ψ0 block; x-stream = pretrained
TRELLIS weights; cross_attn deleted), adapted to SPARSE x. RETROFIT GATE (revised after
the real-ckpt test): LLaMA-Adapter-style zero-init POST-SOFTMAX gate
    out_x = softmax(x→x)·V_x + g·softmax(x→c)·V_c,   g per-head, init 0
— exact pretrained behavior at init for ANY weight scale. (The earlier single-softmax
constant logit-bias −10 was NOT scale-robust: real qk-rms γ push x→c logits to O(10²⁻³);
measured block-0 rel-divergence 0.52. c-queries keep the SD3 joint softmax unchanged.)

  • x is a SparseTensor (varlen tokens per sample). The x-side pipeline reuses the
    PRETRAINED submodules verbatim: to_qkv → SparseMultiHeadRMSNorm → SparseRotary-
    PositionEmbedder (rope from coords) — identical math to SparseMultiHeadAttention's
    self path up to the attention op.
  • x→x uses the TRELLIS sparse attention wrapper VERBATIM (flash varlen — the exact
    pretrained op, full speed). x→c is a small dense cross attention (padded x queries ×
    ~1k c keys). c-queries do the SD3 joint softmax over [x_pad ; c].
  • c stream is dense and identical to mmdit_ss (add_{q,k,v}_proj / to_add_out / ffn_c
    ratio-4 tanh-GELU / share_mod-style modulation / context_pre_only last block).
  • x-side norm-modulate/gate-residual go through the verified lossless Triton helpers
    (`fused_norm_modulate` / `fused_gate_residual`) exactly like the pretrained block.

Build for training via `SLatFlowMMDiT.from_cross_dit(pretrained_slat_flow)`.
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _paths  # noqa: F401
from trellis2.models.structured_latent_flow import SLatFlowModel  # type: ignore
from trellis2.modules import sparse as sp  # type: ignore
from trellis2.modules.norm import LayerNorm32  # type: ignore
from trellis2.modules.transformer.blocks import FeedForwardNet  # type: ignore
from trellis2.modules.sparse.attention.modules import (  # type: ignore
    SparseMultiHeadRMSNorm, SparseRotaryPositionEmbedder)
from trellis2.modules.sparse.transformer.blocks import SparseFeedForwardNet  # type: ignore
from trellis2.modules.sparse.fused_modulate import (  # type: ignore
    fused_norm_modulate, fused_gate_residual)
from trellis2.modules.attention.modules import MultiHeadRMSNorm  # type: ignore
from trellis2.modules.utils import manual_cast  # type: ignore
from trellis2.modules.sparse.attention.full_attn import (  # type: ignore
    sparse_scaled_dot_product_attention as sparse_sdpa)


def _seqlens(x: sp.SparseTensor) -> torch.Tensor:
    """Per-sample token counts of a SparseTensor (int64, device of x)."""
    if hasattr(x, "seqlen") and x.seqlen is not None:
        return torch.as_tensor(x.seqlen, device=x.feats.device, dtype=torch.long)
    # fallback: derive from batch column of coords
    return torch.bincount(x.coords[:, 0].long())


def _pad(feats: torch.Tensor, seqlens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N,*rest) varlen-flat → (B,Lmax,*rest) padded + (B,Lmax) bool valid-mask."""
    parts = torch.split(feats, seqlens.tolist(), dim=0)
    padded = torch.nn.utils.rnn.pad_sequence(parts, batch_first=True)
    B, Lmax = padded.shape[:2]
    ar = torch.arange(Lmax, device=feats.device)
    valid = ar[None, :] < seqlens[:, None]
    return padded, valid


def _unpad(padded: torch.Tensor, seqlens: torch.Tensor) -> torch.Tensor:
    """(B,Lmax,*rest) → (N,*rest) flat, dropping padding."""
    return torch.cat([padded[i, :n] for i, n in enumerate(seqlens.tolist())], dim=0)


class SLatJointAttention(nn.Module):
    """Joint attention over [sparse x ; dense c] with the zero-init post-softmax gate.

    x-side params/names mirror SparseMultiHeadAttention(self) for key-for-key migration:
    to_qkv, q_rms_norm, k_rms_norm, rope, to_out. c-side uses SD3/diffusers naming.
    """

    def __init__(self, channels: int, num_heads: int, qkv_bias: bool = True,
                 qk_rms_norm: bool = True, use_rope: bool = True,
                 rope_freq: Tuple[float, float] = (1.0, 10000.0),
                 context_pre_only: bool = False):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qk_rms_norm = qk_rms_norm
        self.use_rope = use_rope
        self.context_pre_only = context_pre_only

        # ---- x stream (migrated from pretrained SparseMultiHeadAttention) ----
        self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        if qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)
        self.to_out = nn.Linear(channels, channels)

        # ---- c stream (fresh; SD3 naming; dense MultiHeadRMSNorm) ----
        self.add_k_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.add_v_proj = nn.Linear(channels, channels, bias=qkv_bias)
        if qk_rms_norm:
            self.norm_added_k = MultiHeadRMSNorm(self.head_dim, num_heads)
        if not context_pre_only:
            self.add_q_proj = nn.Linear(channels, channels, bias=qkv_bias)
            if qk_rms_norm:
                self.norm_added_q = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.to_add_out = nn.Linear(channels, channels)

        self.c_gate = nn.Parameter(torch.zeros(num_heads))

    def forward(self, hx: sp.SparseTensor, hc: torch.Tensor,
                cond_mask: Optional[torch.Tensor] = None):
        """hx: modulated sparse x; hc: (B,Lc,C) modulated dense c;
        cond_mask: (B,Lc) bool, True = real cond token.
        Returns (out_x: SparseTensor, out_c: (B,Lc,C) or None)."""
        H, D = self.num_heads, self.head_dim
        seqlens = _seqlens(hx)
        B = int(seqlens.shape[0])
        Lc = hc.shape[1]
        dt = hx.feats.dtype

        # ---- x qkv: EXACT pretrained pipeline (qkv → rms → rope) on the sparse tensor ----
        qkv = hx.replace(self.to_qkv(hx.feats).reshape(-1, 3, H, D))
        q, k, v = qkv.unbind(dim=-3)                          # SparseTensors (N,H,D)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
        if self.use_rope:
            q, k = self.rope(q, k)
        q_x, k_x, v_x = q.feats, k.feats, v.feats             # (N,H,D) flat

        # ---- c kv (+q) ----
        k_c = self.add_k_proj(hc).reshape(B, Lc, H, D)
        v_c = self.add_v_proj(hc).reshape(B, Lc, H, D)
        if self.qk_rms_norm:
            k_c = self.norm_added_k(k_c)

        # ---- x→x: EXACT pretrained attention op (TRELLIS sparse wrapper — flash varlen).
        # Fallback (CPU or fp32, where flash is unavailable): padded F.sdpa — numerically
        # the same softmax (verified vs fp32 manual reference). ----
        if hx.feats.is_cuda and dt in (torch.float16, torch.bfloat16):
            out_xx = sparse_sdpa(q, k, v)                      # SparseTensor (N,H,D)
        else:
            qp_, valid_ = _pad(q_x, seqlens)
            kp_, _ = _pad(k_x, seqlens)
            vp_, _ = _pad(v_x, seqlens)
            m_ = torch.where(valid_[:, None, None, :],
                             torch.zeros((), dtype=dt), torch.full((), torch.finfo(dt).min, dtype=dt))
            o_ = F.scaled_dot_product_attention(
                qp_.permute(0, 2, 1, 3), kp_.permute(0, 2, 1, 3), vp_.permute(0, 2, 1, 3),
                attn_mask=m_)
            out_xx = hx.replace(_unpad(o_.permute(0, 2, 1, 3), seqlens))

        neg_inf = torch.finfo(dt).min
        zero = torch.zeros((), dtype=dt, device=hc.device)
        ninf = torch.full((), neg_inf, dtype=dt, device=hc.device)
        if cond_mask is not None:
            pad_c_cols = torch.where(cond_mask[:, None, None, :], zero, ninf)  # (B,1,1,Lc)
        else:
            pad_c_cols = None

        fast = hx.feats.is_cuda and dt in (torch.float16, torch.bfloat16)

        # ---- x→c: flash-varlen via the TRELLIS wrapper (padding-free — masked cond tokens
        # are DROPPED into a VarLenTensor, exactly the pretrained cross-attn's kv path).
        # Fallback (CPU/fp32): padded sdpa with additive mask — same math, verified. ----
        if fast:
            if cond_mask is None:
                out_xc = sparse_sdpa(q, k_c, v_c).feats                     # (N,H,D)
            else:
                k_vl = sp.VarLenTensor.from_tensor_list(
                    [k_c[b][cond_mask[b]] for b in range(B)])
                v_vl = sp.VarLenTensor.from_tensor_list(
                    [v_c[b][cond_mask[b]] for b in range(B)])
                out_xc = sparse_sdpa(q, k_vl, v_vl).feats
        else:
            qx_p, _valid = _pad(q_x, seqlens)                  # (B,Lx,H,D)
            o_ = F.scaled_dot_product_attention(
                qx_p.permute(0, 2, 1, 3), k_c.permute(0, 2, 1, 3), v_c.permute(0, 2, 1, 3),
                attn_mask=pad_c_cols)                          # (B,H,Lx,D)
            out_xc = _unpad(o_.permute(0, 2, 1, 3), seqlens)   # (N,H,D)

        # ---- LLaMA-Adapter zero-init gate: out = softmax(xx)·Vx + g·softmax(xc)·Vc ----
        # g=0 ⇒ EXACTLY the pretrained self-attention, robust to ANY logit scale (a constant
        # logit bias is NOT: real ckpts have qk-rms γ that push x→c logits to O(10²-10³),
        # which no fixed bias can close — measured block-0 rel 0.52 before this fix).
        gate = self.c_gate.to(dt).view(1, H, 1)                # (1,H,1)
        out_x = out_xx.feats + gate * out_xc                   # (N,H,D)
        out_x = hx.replace(self.to_out(out_x.reshape(-1, H * D)))

        if self.context_pre_only:
            return out_x, None

        # ---- c-queries: plain joint softmax over [x ; c] (SD3 form, fresh stream) ----
        q_c = self.add_q_proj(hc).reshape(B, Lc, H, D)
        if self.qk_rms_norm:
            q_c = self.norm_added_q(q_c)

        if fast:
            # flash-varlen: per-sample joint KV = [x_i ; real c_i]; padded c queries dropped
            # and scattered back as zeros (they are masked as keys downstream anyway).
            offs = [0] + torch.cumsum(seqlens, 0).tolist()
            cm = cond_mask if cond_mask is not None else torch.ones(
                B, Lc, dtype=torch.bool, device=hc.device)
            k_joint = sp.VarLenTensor.from_tensor_list(
                [torch.cat([k_x[offs[b]:offs[b + 1]], k_c[b][cm[b]]]) for b in range(B)])
            v_joint = sp.VarLenTensor.from_tensor_list(
                [torch.cat([v_x[offs[b]:offs[b + 1]], v_c[b][cm[b]]]) for b in range(B)])
            q_c_vl = sp.VarLenTensor.from_tensor_list(
                [q_c[b][cm[b]] for b in range(B)])
            oc = sparse_sdpa(q_c_vl, k_joint, v_joint).feats                # (Nc,H,D)
            out_c = torch.zeros(B, Lc, H * D, dtype=dt, device=hc.device)
            off = 0
            for b in range(B):
                nb = int(cm[b].sum())
                out_c[b, cm[b]] = oc[off:off + nb].reshape(nb, H * D)
                off += nb
        else:
            kx_p, valid = _pad(k_x, seqlens)
            vx_p, _ = _pad(v_x, seqlens)
            K = torch.cat([kx_p, k_c], dim=1).permute(0, 2, 1, 3)           # (B,H,Lx+Lc,D)
            V = torch.cat([vx_p, v_c], dim=1).permute(0, 2, 1, 3)
            pad_x_cols = torch.where(valid[:, None, None, :], zero, ninf)   # (B,1,1,Lx)
            if pad_c_cols is None:
                pad_c_cols = torch.zeros(B, 1, 1, Lc, dtype=dt, device=hc.device)
            mask_cq = torch.cat([pad_x_cols, pad_c_cols], dim=-1)           # (B,1,1,Lx+Lc)
            o_ = F.scaled_dot_product_attention(
                q_c.permute(0, 2, 1, 3), K, V, attn_mask=mask_cq)
            out_c = o_.permute(0, 2, 1, 3).reshape(B, Lc, H * D)
        out_c = self.to_add_out(out_c)
        return out_x, out_c


class SLatJointBlock(nn.Module):
    """Double-stream MMDiT block grafted onto ModulatedSparseTransformerCrossBlock.

    x-stream = the pretrained sparse block minus its cross_attn step (fused Triton
    norm-modulate/gate-residual kept); c-stream = dense Qwen-Image txt stream."""

    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 4.0,
                 mlp_ratio_c: float = 4.0, qkv_bias: bool = True, qk_rms_norm: bool = True,
                 use_rope: bool = True, rope_freq: Tuple[float, float] = (1.0, 10000.0),
                 use_checkpoint: bool = False, context_pre_only: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.context_pre_only = context_pre_only

        # ---- x stream (migrated) ----
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

        # ---- c stream (fresh, dense) ----
        self.norm1_c = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        if not context_pre_only:
            self.norm2_c = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
            self.ffn_c = FeedForwardNet(channels, mlp_ratio=mlp_ratio_c)
        self.modulation_c = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

        self.attn = SLatJointAttention(channels, num_heads, qkv_bias=qkv_bias,
                                       qk_rms_norm=qk_rms_norm, use_rope=use_rope,
                                       rope_freq=rope_freq,
                                       context_pre_only=context_pre_only)

    def _forward(self, x: sp.SparseTensor, c: torch.Tensor, mod: torch.Tensor,
                 mod_c: torch.Tensor, cond_mask: Optional[torch.Tensor]):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
            (self.modulation_c + mod_c).type(mod_c.dtype).chunk(6, dim=1)

        # joint attention (x-side fused norm-modulate = pretrained path)
        hx = fused_norm_modulate(x, self.norm1, scale_msa, shift_msa)
        hc = self.norm1_c(c)
        hc = hc * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)
        out_x, out_c = self.attn(hx, hc, cond_mask=cond_mask)
        x = fused_gate_residual(x, out_x, gate_msa)
        if not self.context_pre_only:
            c = c + out_c * c_gate_msa.unsqueeze(1)

        # (pretrained cross_attn step deleted)

        # FFNs
        h = fused_norm_modulate(x, self.norm3, scale_mlp, shift_mlp)
        h = self.mlp(h)
        x = fused_gate_residual(x, h, gate_mlp)
        if not self.context_pre_only:
            hc2 = self.norm2_c(c)
            hc2 = hc2 * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1)
            c = c + self.ffn_c(hc2) * c_gate_mlp.unsqueeze(1)
        return x, c

    def forward(self, x, c, mod, mod_c, cond_mask=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, c, mod, mod_c, cond_mask, use_reentrant=False)
        return self._forward(x, c, mod, mod_c, cond_mask)


class SLatFlowMMDiT(SLatFlowModel):
    """SLAT flow (shape/tex) with cross-attn blocks swapped for double-stream MMDiT blocks.

    Parent pieces (input/out SparseLinear, t_embedder, share_mod adaLN, pos_embedder)
    reused verbatim. Adds Qwen-Image-style context intake + c-side shared modulation.
    forward accepts cond as dense (B,Lc,cond_ch) [+ cond_mask], a per-sample list, or a
    VarLenTensor (padded internally) — covering every call pattern in the blip3o loss.
    """

    def __init__(self, *args, mlp_ratio_c: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.share_mod, "SLatJointBlock assumes share_mod=True"
        ch, nh = self.model_channels, self.num_heads
        n = len(self.blocks)
        ref = self.blocks[0]
        use_rope = ref.self_attn.use_rope
        rope_freq = tuple(ref.self_attn.rope.rope_freq) if use_rope and hasattr(ref.self_attn, "rope") \
            and hasattr(ref.self_attn.rope, "rope_freq") else (1.0, 10000.0)
        qkv_bias = ref.self_attn.to_qkv.bias is not None
        self.blocks = nn.ModuleList([
            SLatJointBlock(ch, nh, mlp_ratio=self.mlp_ratio, mlp_ratio_c=mlp_ratio_c,
                           qkv_bias=qkv_bias, qk_rms_norm=ref.self_attn.qk_rms_norm,
                           use_rope=use_rope, rope_freq=rope_freq,
                           use_checkpoint=ref.use_checkpoint,
                           context_pre_only=(i == n - 1))
            for i in range(n)
        ])
        self.txt_norm = nn.RMSNorm(self.cond_channels, eps=1e-6)
        self.txt_in = nn.Linear(self.cond_channels, ch)
        self.adaLN_modulation_c = nn.Sequential(nn.SiLU(), nn.Linear(ch, 6 * ch, bias=True))
        # parent ran convert_to(dtype) on the OLD blocks before we replaced them — convert
        # the new joint blocks too (txt_*/adaLN_c stay fp32 like the parent's cond/t heads).
        self.convert_to(self.dtype)

    @staticmethod
    def _cond_to_dense(cond, device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """dense tensor | list[Tensor] | VarLenTensor → (B,Lc,C) + (B,Lc) bool mask."""
        if isinstance(cond, torch.Tensor) and cond.dim() == 3:
            return cond, None
        if isinstance(cond, (list, tuple)):
            lens = torch.tensor([c.shape[0] for c in cond], device=device)
            padded = torch.nn.utils.rnn.pad_sequence(list(cond), batch_first=True)
            ar = torch.arange(padded.shape[1], device=device)
            return padded, ar[None, :] < lens[:, None]
        # VarLenTensor
        lens = _seqlens(cond)
        padded, valid = _pad(cond.feats, lens)
        return padded, valid

    def forward(self, x: sp.SparseTensor, t: torch.Tensor,
                cond: Union[torch.Tensor, List[torch.Tensor], "sp.VarLenTensor"],
                concat_cond: Optional[sp.SparseTensor] = None,
                cond_mask: Optional[torch.Tensor] = None, **kwargs) -> sp.SparseTensor:
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        cond_dense, derived_mask = self._cond_to_dense(cond, x.feats.device)
        if cond_mask is None:
            cond_mask = derived_mask

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb_raw = self.t_embedder(t)
        mod = self.adaLN_modulation(t_emb_raw)          # x-side shared mod (pretrained)
        mod_c = self.adaLN_modulation_c(t_emb_raw)      # c-side shared mod (fresh)
        mod = manual_cast(mod, self.dtype)
        mod_c = manual_cast(mod_c, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)

        # context intake runs in fp32 (txt_norm/txt_in are model-level, like the parent's
        # cond heads), THEN cast to the block dtype
        c = manual_cast(self.txt_in(self.txt_norm(cond_dense)), self.dtype)
        for block in self.blocks:
            h, c = block(h, c, mod, mod_c, cond_mask=cond_mask)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h

    # ---------- migration ----------
    @classmethod
    def from_cross_dit(cls, src: SLatFlowModel, mlp_ratio_c: float = 4.0) -> "SLatFlowMMDiT":
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
            qk_rms_norm_cross=False, mlp_ratio_c=mlp_ratio_c,
        )
        dst.input_layer.load_state_dict(src.input_layer.state_dict())
        dst.out_layer.load_state_dict(src.out_layer.state_dict())
        dst.t_embedder.load_state_dict(src.t_embedder.state_dict())
        dst.adaLN_modulation.load_state_dict(src.adaLN_modulation.state_dict())
        if src.pe_mode == "ape":
            dst.pos_embedder.load_state_dict(src.pos_embedder.state_dict())

        for db, sb in zip(dst.blocks, src.blocks):
            db.modulation.data.copy_(sb.modulation.data)
            db.attn.to_qkv.load_state_dict(sb.self_attn.to_qkv.state_dict())
            db.attn.to_out.load_state_dict(sb.self_attn.to_out.state_dict())
            if sb.self_attn.qk_rms_norm:
                db.attn.q_rms_norm.load_state_dict(sb.self_attn.q_rms_norm.state_dict())
                db.attn.k_rms_norm.load_state_dict(sb.self_attn.k_rms_norm.state_dict())
            if sb.self_attn.use_rope:
                db.attn.rope.load_state_dict(sb.self_attn.rope.state_dict())
            db.mlp.load_state_dict(sb.mlp.state_dict())
            # (sb.cross_attn + sb.norm2 intentionally dropped)
        return dst

    def c_gate_report(self) -> str:
        mags = torch.stack([b.attn.c_gate.detach().abs().max() for b in self.blocks])
        return (f"c_gate |max| per block: min {mags.min():.3f} max {mags.max():.3f} "
                f"(init 0 = pretrained; grows as cond is used)")
