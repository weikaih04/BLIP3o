"""Dual-branch conditioning for the TRELLIS native-VLM variant (Know3D-style additive).

Motivation (see RESULTS.md "Open architecture decision", reference_know3d_paper):
our current path REPLACES TRELLIS's pretrained DINOv3 image cross-attn with a
from-scratch Qwen connector → the pretrained flow gets an input distribution it never
saw → unstable (high seed variance). Know3D's fix: keep the original cross-attn intact
as a geometry ANCHOR and ADD the new prior as a parallel, zero-init cross-attn so the
flow is never disrupted (ControlNet zero-conv trick).

Per TRELLIS DiT block we therefore compute (Know3D Eq.3, adapted):

    x_out  = block(h, t, H_dino, ...)                 # ORIGINAL cross-attn ← DINOv3 anchor
    x_out += gate · cross_attn_qwen(norm(x_out), H_qwen)  # NEW branch; `gate` = per-block scalar,
                                                          # zero-init ⇒ ΔF = 0 at start (no disruption),
                                                          # grows as the branch learns. One scalar per
                                                          # DiT block (depth) — readable strength knob.

- **H_dino** = raw DINOv3 features (1024-d) — exactly what the pretrained TRELLIS cross-attn
  was trained on (`get_cond` feeds `image_cond_model(image)` straight in, no projection). For
  image tasks it anchors visible geometry to ground truth; for text it is a null cond (the
  Qwen branch then carries everything); for multi-image it is the per-view DINOv3 features
  concatenated along tokens + a learned per-view embedding so the flow can tell the views
  apart (else it may fuse K views into K separate objects).
- **H_qwen** = connector(Qwen hidden) — the unified cond present on EVERY task (sole cond for text).

ZeRO-2 note: the Qwen branch (norm + cross_attn) must stay in the autograd graph on every
step or the image↔text transition changes graph structure → gradient-reduction deadlock
(same failure mode dino_align._zero_touch guards). We keep H_qwen non-None on all tasks
(text included), so the branch always runs — no zero-touch needed here.

This module touches NO third-party TRELLIS code; it wraps blocks exactly like
depth_fusion._CondInjectBlock. SS flow (dense) is implemented; SLAT (sparse) is TODO.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _paths  # noqa: F401  — sets sys.path so trellis2 imports
from trellis2.modules.attention import MultiHeadAttention  # type: ignore

COND_ARG_IDX = 2          # block.forward(h, t_emb, cond, ...) — cond (context) is the 3rd arg
DINO_DIM = 1024           # DINOv3 ViT-L feature dim == TRELLIS cond_channels (no projection needed)


class DualCondRouter(nn.Module):
    """Per-forward conditioning store + the learned per-view embedding for DINOv3 concat.

    Holds (set fresh each forward, cleared after):
      - `_dino_cond`: (B, N_dino, 1024) DINOv3 anchor, shared across all blocks. For text it is
        a single zero token (null cond). For multi-image it is the per-view concat (see
        `build_dino_anchor`).
      - `_qwen_cond`: (B, T, 1024) connector output, fed to every block's new Qwen cross-attn.
      - `_qwen_mask`: (B,1,1,T) sdpa key mask (True = attend) or None.

    `view_embed` is zero-init so single-view conditioning is byte-identical to "no embedding"
    at the start of training.
    """

    def __init__(self, max_views: int = 8, dino_dim: int = DINO_DIM, anchor_pool: int = 1,
                 anchor_token_budget: int = 0):
        super().__init__()
        self.view_embed = nn.Parameter(torch.zeros(max_views, dino_dim))
        # anchor_pool>1: fixed avg-pool of each view's DINOv3 patch grid (2 ⇒ 32×32→16×16).
        # anchor_token_budget>0: ADAPTIVE — cap the TOTAL anchor patch tokens across all V views
        # to this budget by pooling each view by the smallest power-of-2 factor that fits. So the
        # worst-case cross-attn cost is BOUNDED regardless of view count (4 views pool harder than
        # 2), keeping all views' info (coarser per-view) instead of dropping views. Budget wins
        # over fixed anchor_pool when set. Prefix (CLS/register) tokens kept un-pooled.
        self.anchor_pool = int(anchor_pool)
        self.anchor_token_budget = int(anchor_token_budget)
        self._dino_cond: Optional[torch.Tensor] = None
        self._dino_mask: Optional[torch.Tensor] = None
        # text-task flag (no image → null anchor): the Qwen branch is then the SOLE conditioning,
        # so it must NOT be gated to ≈0 — blocks use gate=1 for text. Image tasks (incl. CFG
        # anchor-drop) keep the learned gate so it can grow as "how much Qwen to mix on top of DINOv3".
        self._text_mode: bool = False
        # BOTH-CFG'd inference path: when True, the cond arg fed to each block is
        # concat([H_dino, H_qwen], dim=1) (coming through the CFG cond_dict, so the sampler drops
        # BOTH in the neg pass → both get CFG-sharpened). The wrapper splits at `_n_dino`:
        # H_dino → original cross-attn, H_qwen → the Qwen branch. False = old set_dino routing.
        self._packed_cond: bool = False
        self._n_dino: int = 0

    @staticmethod
    def _grid_of(N: int) -> int:
        import math
        g = int(math.isqrt(N))
        while g * g > N:
            g -= 1
        return g

    def _pool_view(self, feat: torch.Tensor, p: int) -> torch.Tensor:
        """feat (B, N, C): split prefix (N - grid²) + patch grid, avg-pool the grid by factor p."""
        if p <= 1:
            return feat
        B, N, C = feat.shape
        g = self._grid_of(N)
        if g == 0 or g % p != 0:
            return feat                       # can't pool cleanly → leave as-is
        npatch = g * g
        prefix, grid = feat[:, : N - npatch, :], feat[:, N - npatch:, :]
        grid = grid.transpose(1, 2).reshape(B, C, g, g)
        grid = F.avg_pool2d(grid.float(), kernel_size=p, stride=p).to(feat.dtype)
        grid = grid.reshape(B, C, -1).transpose(1, 2)   # (B, (g/p)², C)
        return torch.cat([prefix, grid], dim=1)

    def _pool_factor(self, V: int, N: int) -> int:
        """Smallest power-of-2 pool factor so V views' patch tokens fit anchor_token_budget."""
        if self.anchor_token_budget <= 0:
            return max(1, self.anchor_pool)
        g = self._grid_of(N)
        per_view = g * g
        p = 1
        while V * (per_view // (p * p)) > self.anchor_token_budget and (g % (p * 2) == 0):
            p *= 2
        return p

    def build_dino_anchor(self, per_view: List[torch.Tensor]) -> torch.Tensor:
        """per_view: list of V tensors (B, N_v, 1024) → (B, Σ pooled_N_v, 1024); view v's learned
        embedding added to its block. Pool factor is adaptive (token budget) or fixed."""
        V = len(per_view)
        p = self._pool_factor(V, per_view[0].shape[1]) if per_view else 1
        parts = []
        for v, feat in enumerate(per_view):
            parts.append(self._pool_view(feat, p) + self.view_embed[v].to(feat.dtype))
        return torch.cat(parts, dim=1)

    def set_dino(self, dino_cond: Optional[torch.Tensor], dino_mask: Optional[torch.Tensor] = None):
        self._dino_cond = dino_cond
        self._dino_mask = dino_mask

    def clear(self):
        self._dino_cond = self._dino_mask = None
        self._text_mode = False
        self._packed_cond = False
        self._n_dino = 0


class DualCondInjectBlock(nn.Module):
    """Wraps ONE dense TRELLIS cross-attn block (SS flow). The wrapped block's original
    cross-attn is fed the DINOv3 anchor (arg idx 2); a new zero-init Qwen cross-attn is
    added on the block output. Signature-transparent (*args/**kwargs)."""

    def __init__(self, block: nn.Module, router: DualCondRouter, cond_arg_idx: int = COND_ARG_IDX,
                 inject_qwen: bool = True):
        super().__init__()
        self.block = block
        object.__setattr__(self, "router", router)   # plain ref, not a submodule
        self.cond_arg_idx = cond_arg_idx
        # inject_qwen=False ⇒ this block ONLY substitutes the DINOv3 anchor into the original
        # cross-attn (the anchor must be on every block) and SKIPS the Qwen branch — used to
        # halve the Qwen-branch cost on SLAT (every-other block) so dual fits at 512.
        self.inject_qwen = inject_qwen
        if not inject_qwen:
            self._last_gate = 0.0
            return
        # Match the block's own cross-attn dims so the new branch is architecturally identical.
        orig = block.cross_attn
        channels = orig.channels
        num_heads = orig.num_heads
        ctx_channels = orig.ctx_channels
        self.norm_q = nn.LayerNorm(channels, elementwise_affine=True, eps=1e-6)
        self.cross_attn_qwen = MultiHeadAttention(
            channels, num_heads=num_heads, ctx_channels=ctx_channels,
            type="cross", attn_mode="full", qk_rms_norm=getattr(orig, "qk_rms_norm", False),
        )
        # Per-block scalar gate (zero-init). The cross-attn is NORMALLY initialized (real output
        # from step 0); ΔF_qwen = gate · cross_attn(...), gate a single learnable scalar per block.
        # At init gate=0 ⇒ ΔF_qwen=0 ⇒ behaviour == untouched pretrained flow (no disruption).
        # NOTE: a *linear* gate after the cross-attn's own to_out would be redundant (two linears,
        # no nonlinearity between ⇒ collapses to one, adds no expressiveness). A scalar is the
        # right zero-init "strength" knob: tiny, non-redundant, and directly readable as how much
        # this depth uses the Qwen branch (anchor-lean diagnostic). One scalar per block (depth).
        self.gate = nn.Parameter(torch.zeros(()))
        self._last_gate: float = 0.0   # diagnostic (gate value), set each forward
        # Gradient-checkpoint the Qwen branch (norm + cross-attn): SS flow is torch.compiled and
        # NOT covered by the elastic SLAT GC, so without this its 30 dual cross-attns keep their
        # forward activations live for backward through the WHOLE cascade (SS→shape→tex) → ~4GB
        # held during the SLAT forward where OOM happens. Recompute in backward instead.
        import os as _os
        self.checkpoint_dual = _os.environ.get("DUAL_CKPT", "1") == "1"

    def _qwen_delta(self, x_out, q, q_mask):
        # Pass NO attn_mask → uses the flash backend (O(N+M)). Passing a mask forces TRELLIS's
        # sdpa-math path which MATERIALIZES the full voxel×token score matrix (huge for big
        # voxel counts). At BS=1 the cond has no padding so the mask is all-true (a no-op) →
        # dropping it is exact AND flash-efficient. (BS>1 padding handling: TODO if we go BS>1.)
        return self.cross_attn_qwen(self.norm_q(x_out), q, attn_mask=None)

    def forward(self, *args, **kwargs):
        # The flow calls: block(h, t_emb, cond, rope_phases, cond_mask=cond_mask).
        # `cond` (arg idx 2) is the cascade's connector output = H_qwen; `cond_mask` (kwarg)
        # is its padding mask. We RE-PURPOSE: H_qwen → the new Qwen branch; the original
        # cross-attn instead gets the DINOv3 anchor (no padding ⇒ mask None).
        args = list(args)
        if len(args) <= self.cond_arg_idx:
            return self.block(*args, **kwargs)
        if getattr(self.router, "_packed_cond", False):
            # BOTH-CFG'd: cond arg = concat([H_dino, H_qwen], dim=1) via the CFG cond_dict → both
            # are dropped in the neg pass → both sharpened. Split: H_dino → original cross-attn,
            # H_qwen → the Qwen branch.
            packed = args[self.cond_arg_idx]
            _n = self.router._n_dino
            dino = packed[:, :_n].to(args[0].dtype)
            q = packed[:, _n:]
            q_mask = None
            args[self.cond_arg_idx] = dino
            kwargs = dict(kwargs); kwargs["cond_mask"] = None
        elif self.router._dino_cond is not None:
            q = args[self.cond_arg_idx]                       # H_qwen
            q_mask = kwargs.get("cond_mask", None)
            dino = self.router._dino_cond.to(args[0].dtype)   # H_dino (match block input dtype)
            args[self.cond_arg_idx] = dino
            kwargs = dict(kwargs); kwargs["cond_mask"] = self.router._dino_mask
        else:
            return self.block(*args, **kwargs)   # dual not armed → passthrough
        x_out = self.block(*args, **kwargs)               # original cross-attn ← DINOv3 anchor
        if self.inject_qwen and q is not None:            # parallel Qwen cross-attn × scalar gate
            if self.checkpoint_dual and self.training:
                delta = torch.utils.checkpoint.checkpoint(
                    self._qwen_delta, x_out, q, q_mask, use_reentrant=False)
            else:
                delta = self._qwen_delta(x_out, q, q_mask)
            # text task: anchor is null so the Qwen branch is the ONLY cond → bypass the gate
            # (use 1.0) so text→3D is conditioned from step 0 and cross_attn_qwen gets full
            # gradient. Image tasks (incl. CFG anchor-drop) keep the learned scalar gate.
            g = 1.0 if getattr(self.router, "_text_mode", False) else self.gate
            x_out = x_out + g * delta
            self._last_gate = float(self.gate.detach())   # always log the LEARNED param (not effective g)
        return x_out


class DualCondSparseInjectBlock(nn.Module):
    """SLAT (sparse) analogue of DualCondInjectBlock. The wrapped sparse block's original
    cross-attn gets the DINOv3 anchor; a new sparse Qwen cross-attn × scalar gate is added on
    the block output. SLAT blocks call `block(x, mod, context)` (context at idx 2, NO cond_mask —
    the sparse cross-attn takes no key mask), x is an sp.SparseTensor (ops go through `.feats`)."""

    def __init__(self, block: nn.Module, router: DualCondRouter, cond_arg_idx: int = COND_ARG_IDX,
                 inject_qwen: bool = True):
        super().__init__()
        self.block = block
        object.__setattr__(self, "router", router)
        self.cond_arg_idx = cond_arg_idx
        # inject_qwen=False ⇒ anchor-only block (DINOv3 into the original cross-attn, no Qwen branch).
        # Used to inject Qwen on only every-other SLAT block → halve the (heavy) SLAT dual cost.
        self.inject_qwen = inject_qwen
        if not inject_qwen:
            self._last_gate = 0.0
            return
        from trellis2.modules.sparse import SparseMultiHeadAttention  # type: ignore
        from trellis2.modules.norm import LayerNorm32  # type: ignore
        orig = block.cross_attn
        self.norm_q = LayerNorm32(orig.channels, elementwise_affine=True, eps=1e-6)
        self.cross_attn_qwen = SparseMultiHeadAttention(
            orig.channels, num_heads=orig.num_heads, ctx_channels=orig.ctx_channels,
            type="cross", attn_mode="full", qk_rms_norm=getattr(orig, "qk_rms_norm", False),
        )
        self.gate = nn.Parameter(torch.zeros(()))   # per-block scalar strength, zero-init
        self._last_gate: float = 0.0

    def forward(self, *args, **kwargs):
        args = list(args)
        if len(args) <= self.cond_arg_idx:
            return self.block(*args, **kwargs)
        if getattr(self.router, "_packed_cond", False):
            # BOTH-CFG'd: cond arg = concat([H_dino, H_qwen]) → split (see dense block).
            packed = args[self.cond_arg_idx]
            _n = self.router._n_dino
            q = packed[:, _n:]                                # H_qwen part
            args[self.cond_arg_idx] = packed[:, :_n]          # H_dino → original cross-attn
        elif self.router._dino_cond is not None:
            q = args[self.cond_arg_idx]                            # H_qwen (dense (B,T,1024))
            args[self.cond_arg_idx] = self.router._dino_cond.to(q.dtype)   # H_dino
        else:
            return self.block(*args, **kwargs)
        x_out = self.block(*args, **kwargs)                   # sp.SparseTensor
        if self.inject_qwen and q is not None:                # sparse Qwen cross-attn × scalar gate
            h = x_out.replace(self.norm_q(x_out.feats))
            delta = self.cross_attn_qwen(h, q)                # SparseTensor (same coords as x_out)
            x_out = x_out + delta.replace(self.gate * delta.feats)
            self._last_gate = float(self.gate.detach())
        return x_out


def install_dual_routing(flow: nn.Module, router: DualCondRouter, qwen_stride: int = 1,
                         qwen_last_frac: float = 0.0) -> nn.Module:
    """Wrap every block of `flow.blocks` with the dual-cond wrapper (dense for SS, sparse for
    SLAT). The DINOv3 anchor goes on EVERY block (reuses the existing cross_attn, no new module,
    no extra mem). The Qwen branch (NEW cross_attn, the heavy part — doubles activation where
    injected) goes only where selected:
      - `qwen_last_frac` > 0 (preferred): inject ONLY on the last `frac` fraction of blocks
        (contiguous late blocks). frac=0.2 → last 20% of blocks. This aligns the Qwen branch
        with the trainable tail (flow_tune last15/20) — early frozen blocks have gate≈0 anyway,
        so their Qwen cross-attn is pure wasted activation. Cuts dual's +9.5GB by ~80% → fits
        16-GPU/2-node despite the ~5.5GB NCCL IB overhead.
      - else `i % qwen_stride == 0`: every-Nth block (legacy stride mode).
    Call once on a fresh flow. Returns the flow."""
    blocks = flow.blocks
    n = len(blocks)
    if qwen_last_frac and qwen_last_frac > 0:
        start = int(round(n * (1.0 - float(qwen_last_frac))))
        start = max(0, min(n - 1, start))   # always inject ≥1 block
    else:
        start = None
    for i in range(n):
        b = blocks[i]
        if not hasattr(b, "cross_attn"):
            raise NotImplementedError(
                f"block {i} ({type(b).__name__}) has no `.cross_attn` — dual routing needs a "
                "cross-attn block (ModulatedTransformerCrossBlock / ModulatedSparseTransformerCrossBlock)."
            )
        if start is not None:
            inject = (i >= start)
        else:
            inject = (i % max(1, qwen_stride) == 0)
        cls = DualCondSparseInjectBlock if "Sparse" in type(b).__name__ else DualCondInjectBlock
        blocks[i] = cls(b, router, inject_qwen=inject)
    return flow
