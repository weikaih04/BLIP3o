"""Unified Geo-Tex DiT — the two SLAT specialists stitched into one dual-stream
MMDiT with per-modality timesteps (t_s, t_x).

Design doc: docs/UNIFIED_GEOTEX_DIT_DESIGN.md · plan: gentle-sparking-gem v2.
Reference repos audited line-by-line under
/fsx/home/weikai.huang/3dgen/reference_repos/ — provenance annotated inline.

GEO↔TEX COUPLING — two implementations, ONE flag (user decision 2026-08-11:
"try MF first, measure"; geo queries attend only geo keys in both — one-way spec):
    coupling="union" (DEFAULT): MF-faithful bare single-softmax over [k_x;k_s]
        (modality-forcing dit.py:137-148; per-stream projections). No gate. MF
        trains this ungated at 9B — but its depth stream is a CLONE (identical
        qk-rms γ scales). Ours are two INDEPENDENTLY trained specialists, and
        mmdit_slat.py:190-196 MEASURED O(1e2-1e3) cross-logit mismatch for such
        pairs — so G0 MEASURES our actual cross/self logit ratio on the real
        ckpts before any training commits to this coupling.
    coupling="gated" (READY FALLBACK): in-house mmdit_slat pattern —
        out = softmax(q_x·k_x)·v_x + g·softmax(q_x·k_s)·v_s, per-head zero-init
        g (LLaMA-Adapter lineage). g=0 ⇒ exact identity, robust to any logit
        scale; both terms on the sparse flash kernels. Switched in if the G0
        [SCALE] measurement shows the pathological ratio.

The TWO first-class variants differ only in COND injection:
    cond_mode="cross_attn"  (variant 1, "mmdit + cross", Stage-1):
        [shape;tex] joint attention; cond via per-stream read-once cross-attn
        (production path; June cond-ablation winner).
    cond_mode="stream"      (variant 2, "mmdit joint", Stage-2 A/B arm):
        [shape;tex;cond] all in the joint attention (cond promoted to a stream;
        zero-init input proj; cond never reads voxels -> cacheable; cross-attn
        kept during transition). NOT implemented yet — placeholder raises.

Shared machinery:
  * geo stream: frozen, no_grad, per-block (k_s, v_s) borrowed (post rms-norm +
    RoPE — phases derive from shared coords, so cross terms carry correct
    relative positions for free; rope.py:36-59).
  * tex stream: fully trained; concat_cond fed x_{t_s} (the t_s-noised shape
    latent, used DIRECTLY — the two norm-stat config keys are measured
    bit-identical; guarded by tests).
  * cross-t awareness (MF v2, dit.py:277-288, 388-413): t_s enters tex's t_emb
    once at vec level via cross_alpha (zeros(1), FSDP rule) x a FRESH-init
    embedder (MF uses a fresh MLPEmbedder, not a clone).
  * G0: gated arm is bit-exact by construction (g=0, same kernel); union arm
    has NO init identity (MF-style perturbed start) — assembly is certified via
    a TEST-ONLY -inf mask on the geo segment, and the [SCALE] measurement
    decides union-vs-gated empirically.

Timestep convention (verified across ALL reference codebases and ours):
  t=0 clean, t=1 noise, x_t=(1-t)x0+t*eps, v=eps-x0, t_in = t*1000 at the flow
  boundary (loss.py:181). Ported verbatim, zero re-derivation.
"""
from typing import Dict
from contextlib import contextmanager
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import trellis2_blip3o._paths  # noqa: F401  (repo path injection, existing convention)
from trellis2.modules import sparse as sp
from trellis2.modules.attention import RotaryPositionEmbedder
from trellis2.modules.attention.full_attn import scaled_dot_product_attention as _dense_sdpa
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
# the block's OWN fused kernels (modulated.py:149-160 calls these; G0 bisect
# proved eager-equivalent math lands 1 ULP off the Triton path at real scale)
from trellis2.modules.sparse.transformer.modulated import (
    fused_norm_modulate, fused_gate_residual)
from trellis2.modules.utils import manual_cast


# ─────────────────────────────────────────────────────────────────────────────
# helpers (faithful replications of module internals; G0 certifies fidelity)
# ─────────────────────────────────────────────────────────────────────────────

def _block_mod_params(block, mod: torch.Tensor):
    """share_mod unpack — sparse/transformer/modulated.py:143-147."""
    return (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)


def _attn_qkv(attn, h_sp):
    """SparseMultiHeadAttention self-path up to roped (q, k, v) —
    attention/modules.py:110-125."""
    qkv = attn._linear(attn.to_qkv, h_sp)
    qkv = attn._fused_pre(qkv, num_fused=3)          # feats (T, 3, H, C)
    q, k, v = qkv.unbind(dim=-3)
    if attn.qk_rms_norm:
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    if attn.use_rope:
        q, k = attn.rope(q, k)
    return q, k, v


def _attn_out(attn, h_sp):
    """Attention tail: heads -> channels -> to_out."""
    h_sp = attn._reshape_chs(h_sp, (-1,))
    return attn._linear(attn.to_out, h_sp)


def _rotate_pad_pair(k_sp):
    """STREAM TAG (MF stream-id RoPE axis, warm-start-compatible adaptation):
    rotate the IDENTITY-PAD rotation pair (last 2 head dims — TRELLIS ropes
    63/64 pairs, the 64th passes unrotated) by π/2 on FOREIGN keys at the
    borrow site. Within-stream attention is untouched (bit-exact G0 holds);
    cross-stream logits gain an antisymmetric "which stream" phase. π/2 is
    float-EXACT: (x, y) → (−y, x), a swap + negate, zero rounding."""
    f = k_sp.feats
    x, y = f[..., -2:-1], f[..., -1:]
    return k_sp.replace(torch.cat([f[..., :-2], -y, x], dim=-1))


# ── v10: dense (SS) lane helpers ─────────────────────────────────────────────
# The sparse helpers above cannot be reused: the dense block broadcasts its
# modulation with .unsqueeze(1) over a (B, L, C) tensor, while the sparse call
# sites broadcast per-sample over a flat voxel list. Same formula, different
# shape discipline — reusing one for the other is a silent, plausible-looking bug.
def _dense_mod_params(block, mod: torch.Tensor):
    """modulated.py:148-149, share_mod branch. Six (B, C) tensors."""
    assert block.share_mod, "SS blocks are share_mod=True; the adaLN branch is untested here"
    return (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)


def _dense_attn_qkv(attn, h: torch.Tensor, phases: torch.Tensor):
    """modules.py:73-87 for type='self'. Returns (q, k, v, k_pre_rope).

    k_pre_rope is the EXPORT POINT for the cross-tower read: post-qk_rms_norm,
    PRE-rope. It has to be pre-rope because a borrowed key is re-roped into the
    32^3 frame at the consumer; handing over the already-roped k would rotate it
    twice and quietly destroy every relative position it encodes. The rms-then-
    rope order here is identical to the sparse path, so the two towers export
    the same object.
    """
    B, L, _ = h.shape
    qkv = attn.to_qkv(h).reshape(B, L, 3, attn.num_heads, -1)
    q, k, v = qkv.unbind(dim=2)
    if attn.qk_rms_norm:
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    k_pre = k
    if attn.use_rope:
        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
    return q, k, v, k_pre


def _dense_attn_out(attn, h: torch.Tensor):
    """modules.py:109-110."""
    B, L = h.shape[:2]
    return attn.to_out(h.reshape(B, L, -1))


def _fusion_plan(layout, corner_on, device):
    """Gather indices for the FUSED joint-attention path. Source stacking is
    rows [0,T) = geo, [T,2T) = tex (a plain cat of the two lanes' k/v).

      fused_idx / fused_layout : per sample [geo_b ; tex_b]  (kv for the tex lane,
                                 and for JOINT samples in the geo lane)
      geo_kv_idx / geo_layout  : per sample [geo_b] if t_s=0 (corner ⇒ one-way)
                                 else [geo_b ; tex_b]

    Varlen attention takes per-sample kv lengths, so the corner mask is encoded
    in the LAYOUT — no attention mask, no batch grouping, no python loop over
    samples at attention time. Computed ONCE per forward, reused by all 30
    blocks (the plan depends only on the layout and t_s, both step-constant)."""
    T = layout[-1].stop
    geo_rows = [torch.arange(s.start, s.stop, device=device) for s in layout]
    tex_rows = [r + T for r in geo_rows]
    fused_idx = torch.cat([torch.cat([g, t]) for g, t in zip(geo_rows, tex_rows)])
    # layout is contiguous & cumulative ⇒ doubling both ends is exact
    fused_layout = [slice(2 * s.start, 2 * s.stop) for s in layout]
    if bool(corner_on.all()):                      # every sample joint
        return fused_idx, fused_layout, fused_idx, fused_layout
    if bool((~corner_on).all()):                   # every sample at the corner
        return fused_idx, fused_layout, torch.cat(geo_rows), list(layout)
    parts, lens = [], []
    for b, s in enumerate(layout):
        if bool(corner_on[b]):
            parts.append(torch.cat([geo_rows[b], tex_rows[b]])); lens.append(2 * (s.stop - s.start))
        else:
            parts.append(geo_rows[b]); lens.append(s.stop - s.start)
    return (fused_idx, fused_layout,
            torch.cat(parts), sp.VarLenTensor.layout_from_seqlen(lens))


def _attn_qkv_norope(attn, h):
    """qkv WITHOUT rope — the cond stream has no coords (unordered token set).
    Precedent: FLUX ropes txt at position 0 (≈ identity rotation), TRELLIS
    never ropes cond; both reduce to un-roped keys in the joint softmax."""
    qkv = attn._linear(attn.to_qkv, h)
    qkv = attn._fused_pre(qkv, num_fused=3)
    q, k, v = qkv.unbind(dim=-3)
    if attn.qk_rms_norm:
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    return q, k, v


# ─────────────────────────────────────────────────────────────────────────────
# the unified model
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedGeoTexFlow(nn.Module):
    """geo_flow (frozen) + tex_flow (trained); coupling = union (default) | gated.

    forward() replicates SLatFlowModel.forward (structured_latent_flow.py:169-198)
    for both streams, block-interleaved; G0 certifies replication fidelity.
    """

    def __init__(self, geo_flow: nn.Module, tex_flow: nn.Module,
                 cond_mode: str = "cross_attn",
                 cond_stream_blocks: int = 10,
                 coupling: str = "union",
                 bidirectional: bool = False,
                 ss_flow: nn.Module = None,
                 all_trainable: bool = False,
                 cond_seg_embed: bool = False,
                 cond_patch_pos: str = "off",      # off | zero | dino_sig
                 cond_patch_lattice: int = 32):
        """coupling: "union" (DEFAULT — MF-faithful bare single-softmax union,
        dit.py:137-148; user decision 2026-08-11: try MF first, measure) or
        "gated" (in-house mmdit_slat.py:190-196 two-softmax + per-head zero-init
        gate — the ready fallback if G0's cross/self logit-scale measurement
        shows the O(1e2-1e3) mismatch mmdit_slat measured for ITS weight pair).
        G0 reports that ratio on our real ckpts so the choice is empirical."""
        super().__init__()
        if cond_mode == "stream":
            pass  # variant B ("mmdit joint") — constructed at the end of __init__
        assert cond_mode in ("cross_attn", "stream"), cond_mode
        assert coupling in ("union", "gated"), coupling
        # bidirectional × cond-stream = the full three-stream MMDiT (approved
        # 2026-08-14). All three streams read each other: geo<->tex as in S2b
        # (corner-masked), and cond both feeds AND reads the voxel streams —
        # the SD3/FLUX property that makes a cond STREAM more than a deeper
        # connector. Costs the cond K/V cache (cond stops being per-sample
        # constant); that trade was made deliberately.
        self.coupling = coupling
        # ── architecture parity asserts (silent-misload guard) ──
        for f, name in ((geo_flow, "geo"), (tex_flow, "tex")):
            assert getattr(f, "pe_mode", None) == "rope", f"{name}: pe_mode must be rope"
            assert f.share_mod, f"{name}: share_mod expected True (t50b ckpts)"
        assert len(geo_flow.blocks) == len(tex_flow.blocks), "block count mismatch"
        assert geo_flow.blocks[0].self_attn.channels == \
            tex_flow.blocks[0].self_attn.channels, "width mismatch"
        # RoPE parity: cross-read correctness relies on IDENTICAL phases from the
        # shared coords — same freq band + head_dim on both streams, or q_x·k_s
        # relative positions silently break (rope.py phases depend only on
        # coords and this config).
        ga = geo_flow.blocks[0].self_attn
        ta = tex_flow.blocks[0].self_attn
        assert ga.use_rope and ta.use_rope, "both streams must use rope"
        assert ga.rope.rope_freq == ta.rope.rope_freq, "rope_freq mismatch"
        assert ga.rope.head_dim == ta.rope.head_dim, "rope head_dim mismatch"
        if ss_flow is not None:
            # The SS tower is DENSE (16^3 = 4096 tokens, full attention), so its
            # phases live in a precomputed buffer rather than in a rope module,
            # and its rope_freq is NOT stored anywhere: sparse_structure_flow.py
            # builds the table with RotaryPositionEmbedder(head_dim, 3) and no
            # rope_freq argument, so it is always the (1.0, 10000.0) default and
            # the slat towers must match that LITERAL, not a sibling attribute.
            assert getattr(ss_flow, "pe_mode", None) == "rope", "ss: pe_mode must be rope"
            assert getattr(ss_flow, "rope_phases", None) is not None, "ss: no rope_phases"
            assert ss_flow.rope_phases.is_complex(), (
                "ss.rope_phases lost its imaginary part — something cast the whole "
                "module to bf16 instead of going through to_bf16_keep_complex; the "
                "model still runs and the geometry is garbage")
            assert tuple(ga.rope.rope_freq) == (1.0, 10000.0), (
                f"slat rope_freq {tuple(ga.rope.rope_freq)} != the (1.0, 10000.0) the "
                "SS phase table is hard-wired to (sparse_structure_flow.py)")
            _ss_hd = ss_flow.model_channels // ss_flow.num_heads
            assert _ss_hd == ga.rope.head_dim, f"ss head_dim {_ss_hd} != slat {ga.rope.head_dim}"
            assert ss_flow.num_heads == ga.num_heads == ta.num_heads, "head count mismatch"
            assert ss_flow.blocks[0].self_attn.qk_rms_norm == ga.qk_rms_norm, "qk_rms mismatch"
            assert len(ss_flow.blocks) == len(geo_flow.blocks), "ss block count mismatch"

        self.cond_mode = cond_mode
        self.geo_flow = geo_flow
        self.tex_flow = tex_flow

        # ── v10: the SS tower as a THIRD stream ──
        # Registered here (prefix unified_geotex.ss_flow.) and NOT also on
        # TrellisNativeVLM: a second registration would duplicate 1.3B in the
        # state dict and double-count it in the EMA shadow.
        self.ss_flow = ss_flow
        # all_trainable is v10's "nothing is frozen" switch. It cannot be
        # expressed with --flow_tune: that flag's "full" mode returns without
        # unfreezing anything, and its tag list never matches a unified_geotex.*
        # parameter, so it is a silent no-op on this path.
        self.all_trainable = bool(all_trainable)
        self._geo_unfrozen = self.all_trainable

        # geo: frozen unless v10 says otherwise (one-way spec makes this exact)
        if not self.all_trainable:
            for p in self.geo_flow.parameters():
                p.requires_grad_(False)
            self.geo_flow.eval()

        # ── bidirectional (user topology decision 2026-08-11): geo ALSO reads
        # tex in the joint region, CORNER-MASKED off at t_s=0 per sample (the
        # tex|mesh operating point keeps geo time-invariant → K/V cache + exact
        # geometry lock survive; DF noise-level-conditioned masking precedent).
        # geo←tex reads sit behind zero-init per-head b_gates (gated) or bare
        # union with hard corner exclusion (union) — either way the t_s=0
        # config is EXACTLY the specialist. In S1 geo WEIGHTS stay frozen but
        # its lane runs in-graph so b_gates + the geo-mediated tex feedback
        # receive gradients from the tex loss.
        self.bidirectional = bool(bidirectional)
        # fused MMDiT attention path (union+bidir only): one native varlen call
        # per lane instead of the per-sample sdpa loop. OFF until G0-fused
        # certifies it; set model.fused_attn = True to enable.
        self.fused_attn = False
        n_heads = tex_flow.blocks[0].self_attn.num_heads
        if self.bidirectional and coupling == "gated":
            self.b_gates = nn.Parameter(torch.zeros(len(tex_flow.blocks), n_heads))
        else:
            self.b_gates = None
        # geo-side cross-t mixer (symmetric MF v2; user 2026-08-12: geo must
        # know t_x too). Zero-init scalar ⇒ exact-zero contribution at init;
        # the NEW params train via the tex loss through the in-graph geo lane
        # (same mechanism as b_gates). The frozen adaLN never changes — it just
        # processes a (learnably) shifted t_emb input once the gate opens.
        if self.bidirectional:
            self.t_mixer_s = type(geo_flow.t_embedder)(geo_flow.model_channels)
            self.cross_alpha_s = nn.Parameter(torch.zeros(1))
            # mirror the module it clones (geo's t_embedder), not the first param
            self.t_mixer_s.to(next(geo_flow.t_embedder.parameters()).dtype)
        else:
            self.t_mixer_s = None
            self.cross_alpha_s = None
        if coupling == "gated":
            # per-head zero-init cross gate — mmdit_slat.py:190-196 verbatim
            # (in-house measured; LLaMA-Adapter lineage). One (H,) per block.
            self.c_gates = nn.Parameter(torch.zeros(len(tex_flow.blocks), n_heads))
        else:
            self.c_gates = None
            # union coupling has no init-identity — G0 certifies assembly via a
            # TEST-ONLY -inf mask on the geo segment (never used in training).
            self.test_identity_mask = False

        # cross-t mixer (MF v2 verbatim: a FRESH-initialized embedder for the other
        # modality's timestep + a zero scalar gate; dit.py:277-288 uses a fresh
        # MLPEmbedder, NOT a clone). zeros(1) not zeros(()) — MF's FSDP note.
        self.t_mixer = type(tex_flow.t_embedder)(tex_flow.model_channels)
        self.cross_alpha = nn.Parameter(torch.zeros(1))
        # dtype parity with the bf16 body (audit: MF keeps gate in body dtype;
        # mixed fp32 params break DeepSpeed flat buckets)
        # dtype must MIRROR the module we are cloning, not "whatever parameter
        # happens to be first" — under TRELLIS's official mixed precision the
        # t_embedder is fp32 while the blocks are bf16, and `next(parameters())`
        # would pick up either depending on registration order (2026-08-12 audit).
        t_dtype = next(tex_flow.t_embedder.parameters()).dtype
        self.t_mixer.to(t_dtype)
        body_dtype = tex_flow.blocks[0].mlp.mlp[0].weight.dtype   # the torso dtype
        # (c_gates/cross_alpha stay fp32-safe: scalars broadcast, cast at use)

        # NOTE (measured 2026-08-11): the img2shape "normalization" and the
        # imgshape2tex "shape_slat_normalization" stat sets are BIT-IDENTICAL —
        # geo-stream states are directly valid as tex concat_cond, no adapter.
        # tests/test_unified_conventions.py guards the equality so a future
        # TRELLIS.2 config change fails loudly instead of silently mis-scaling.

        # ── variant B: cond-as-stream ("mmdit joint"; design doc §3.5, plan §2) ──
        # [shape;tex;cond] all in joint attention. Approved spec: zero-init input
        # proj 1024→1536; cond stream = TEX-block clones (cross_attn/norm2
        # stripped → +1.04B ≈ spec's +1.06B); cond reads NOTHING back (row
        # masked) → per-sample constant → per-block K/V cacheable at inference;
        # the read-once cross-attn stays DURING the transition (warm start ≈
        # variant A). Implementation decisions (documented for the A/B review):
        #   * cond stream runs at CONSTANT t_c≡0 modulation — forced by the
        #     approved cacheability property (live-t would break it);
        #   * cond keys enter the joint softmax UN-roped — FLUX ropes txt at
        #     pos-0 (≈identity) and TRELLIS never ropes cond:双先例;
        #   * cond stream input = the TEX connector's cond (the trainable
        #     pathway); geo's kept cross-attn still reads its own cond_s;
        #   * coupling="gated" adds zero-init per-head gates for BOTH voxel
        #     streams' cond reads (same mmdit_slat pattern as c_gates) → the
        #     whole stream variant is EXACTLY identity at init. union = bare
        #     concat into the shared softmax (MF/FLUX-faithful), init-perturbed
        #     like coupling union, absorbed by training.
        self.cond_proj = None
        self.cond_blocks = None
        self.cond_gates_tex = None
        self.cond_gates_geo = None
        # cross-attn ANNEAL (2026-08-15): the read-once cross-attn is the
        # conditioning pathway the pretrained weights were built around, so it
        # cannot simply be deleted at warm start. Instead every cross-attn
        # output is scaled by `xattn_scale`, driven 1 -> 0 by the trainer. At
        # step 0 the model is bit-exact with S2b; once the scale reaches 0 the
        # ONLY conditioning pathway left is the cond stream, i.e. the model is a
        # true three-stream MMDiT and the (now dead) cross-attn weights can be
        # pruned. Keeping cross-attn AND a cond stream would compare a bigger
        # hybrid against the baseline instead of comparing the two pathways.
        self.xattn_scale = 1.0
        self.cond_reads_gate = None
        self.cond_first_idx = len(tex_flow.blocks)
        if cond_mode == "stream":
            import copy
            D = tex_flow.model_channels
            C = tex_flow.cond_channels
            # LAST-N only (cost control, approved 2026-08-14): cross-attn still
            # delivers cond to all 30 blocks, so early blocks are not starved.
            # The stream's marginal value is "cond adapts to the object being
            # formed", which needs the object to exist -> late blocks.
            n_cs = len(tex_flow.blocks) if cond_stream_blocks <= 0 else \
                min(cond_stream_blocks, len(tex_flow.blocks))
            self.cond_first_idx = len(tex_flow.blocks) - n_cs
            self.cond_proj = nn.Linear(C, D)
            # NOT zero-init (changed 2026-08-14). Identity at warm start is the
            # zero GATES' job; a zero projection would ALSO make the cond stream
            # input-independent, so the gates' first gradients would carry no
            # conditioning signal and they would learn a constant direction.
            self.cond_blocks = nn.ModuleList()
            for blk in tex_flow.blocks[self.cond_first_idx:]:
                cb = copy.deepcopy(blk)
                delattr(cb, "cross_attn")   # cond has no second cond source
                delattr(cb, "norm2")
                cb.use_checkpoint = False
                self.cond_blocks.append(cb)
            self.cond_proj.to(body_dtype)
            self.cond_blocks.to(body_dtype)
            # These gates exist for EVERY coupling, not only "gated": a bare
            # union CANNOT be identity for a NEW segment — zero-valued cond keys
            # still enter the softmax denominator and dilute every voxel
            # attention. Per-head zero-init, mmdit_slat.py:190-196.
            self.cond_gates_tex = nn.Parameter(torch.zeros(n_cs, n_heads))
            self.cond_gates_geo = nn.Parameter(torch.zeros(n_cs, n_heads))
            # cond's OWN reads of the two voxel streams (the true-MMDiT
            # direction): [:, 0] = geo, [:, 1] = tex. Zero-init, so at warm
            # start the cond lane is exactly the old read-only lane.
            self.cond_reads_gate = nn.Parameter(torch.zeros(n_cs, 2, n_heads))

        # ── v10 cross-tower reads (both directions), all zero-init ──
        # These are SEPARATE gated attentions whose output is ADDED before
        # to_out, not extra keys in the shared union softmax. That is not a
        # stylistic choice: a bare union has NO init identity for a new segment
        # (even all-zero keys enter the softmax denominator and dilute every
        # existing attention), so concatenating SS keys would move the model on
        # step 0 and forfeit the only thing that makes the three-tower assembly
        # certifiable against the two-tower one. Being a separate softmax is
        # also why this composes with the fused union path unchanged.
        if ss_flow is not None:
            n_b = len(tex_flow.blocks)
            self.ss_gates_geo = nn.Parameter(torch.zeros(n_b, n_heads))
            self.ss_gates_tex = nn.Parameter(torch.zeros(n_b, n_heads))
            # SS <- slat. Present so v10 trains it, DORMANT at inference until a
            # later phase turns it on; [:, 0] = geo, [:, 1] = tex.
            self.ss_reads_gate = nn.Parameter(torch.zeros(n_b, 2, n_heads))
            self.ss_reads_enabled = True     # plain attr: inference flips it, not a Parameter

            # Re-rope table: the SS grid mapped into the slat frame, c -> 2c+0.5.
            # SS cell c covers slat voxels 2c and 2c+1, so 2c+0.5 is their exact
            # midpoint — the position a slat query should see the SS key at. The
            # table depends only on the fixed 16^3 grid, so it is built once and
            # reused by all 30 blocks and both consumer lanes.
            # persistent=False keeps a complex64 tensor out of every state_dict
            # and strict-load path (and out of reach of a future whole-model
            # .to(bf16), which would silently zero its imaginary part).
            from trellis2.modules.attention import RotaryPositionEmbedder
            _res = ss_flow.resolution
            _c = torch.stack(torch.meshgrid(
                *[torch.arange(_res, dtype=torch.float32)] * 3, indexing="ij"),
                dim=-1).reshape(-1, 3)          # SAME order as the SS token flatten
            _rp = RotaryPositionEmbedder(ga.rope.head_dim, 3, rope_freq=(1.0, 10000.0))
            self.register_buffer("ss_phases_slat", _rp(2.0 * _c + 0.5), persistent=False)
        else:
            self.ss_gates_geo = None
            self.ss_gates_tex = None
            self.ss_reads_gate = None
            self.ss_reads_enabled = False

        # ── cond segment code: (stream, segment, C), segment 0 = image, 1 = text ──
        # One per stream because each tower reads the cond through its own
        # connector and may want a different emphasis. 3 x 2 x 1024 = 6144 params,
        # zero-init, so enabling it does not move step 0.
        if cond_seg_embed:
            _C = tex_flow.cond_channels
            self.cond_seg_embed = nn.Parameter(torch.zeros(3, 2, _C))
        else:
            self.cond_seg_embed = None

        # ── per-patch position code for the QWEN image span ──
        # The DINO half already carries per-patch position implicitly: it is a
        # ViT's own output and the pretrained flow was trained to read exactly
        # that. The qwen visual tokens are the ones the flow cannot place — its
        # cross-attn ropes nothing, and whatever M-RoPE put into them is not the
        # signature the flow parses.
        #
        # LEARNABLE, INITIALISED FROM THAT SIGNATURE — not zero, and not fixed.
        #   zero-init makes the model learn three things (that position matters,
        #     a code for it, and readers for that code) when its readers already
        #     know one;
        #   fixed direction (what DinoPosStamp does, with only a learnable
        #     scalar) cannot correct the assumption underneath it: that qwen
        #     token i and dino patch i are the same place, when the two come
        #     from different pipelines (DINO 512px/patch16 vs a 1024px qwen
        #     canvas) and only happen to land on 32x32 grids.
        # Starting at the signature buys the head start; being learnable lets
        # training walk away from the borrowed correspondence if it is wrong.
        self.cond_patch_pos = None
        if cond_patch_pos != "off":
            _C, _P = tex_flow.cond_channels, int(cond_patch_lattice)
            if cond_patch_pos == "dino_sig":
                # The signature is a 32x32 table. If the table resolution is set
                # to anything else, interpolate it there — the same operation a
                # ViT does to reuse a position embedding at a new input size, and
                # the reason the resolution can be chosen freely without giving
                # up the warm start.
                import numpy as _np
                from .pos_stamp import DPOS_NPZ
                _p = torch.from_numpy(_np.load(DPOS_NPZ)["pos"].astype("float32"))
                _S = int(round(_p.shape[0] ** 0.5))
                assert _S * _S == _p.shape[0] and _p.shape[1] == _C, \
                    f"dino signature {tuple(_p.shape)} is not (SxS, {_C})"
                if _S != _P:
                    _p = F.interpolate(_p.view(_S, _S, _C).permute(2, 0, 1)[None],
                                       size=(_P, _P), mode="bilinear",
                                       align_corners=False)[0].permute(1, 2, 0).reshape(_P * _P, _C)
            elif cond_patch_pos == "zero":
                _p = torch.zeros(_P * _P, _C)
            else:
                raise ValueError(f"cond_patch_pos={cond_patch_pos!r}")
            # one per stream, same reasoning as the segment code
            self.cond_patch_pos = nn.Parameter(_p[None].repeat(3, 1, 1).clone())

    # ── coupling A (DEFAULT): MF-faithful bare union softmax ────────────────
    def _union_attn(self, q_x, k_x, v_x, k_s, v_s, k_c=None, v_c=None,
                    seg_on=None):
        """ONE shared softmax over K=[k_x;k_s(;k_c)] (MF dit.py:137-148
        semantics, per-stream projections; cond segment only in stream mode).
        seg_on: optional (B,) bool — per-sample HARD inclusion of the foreign
        segments (bidir corner mask: False at t_s=0 ⇒ this sample's softmax is
        pure self, exactly the specialist). Per-sample torch sdpa (sparse
        kernel has no mask arg; the G0 instrument needs one)."""
        outs = []
        for b in range(q_x.shape[0]):
            sl = q_x.layout[b]
            qb = q_x.feats[sl.start:sl.stop]
            segs = [(k_x.feats[sl.start:sl.stop], v_x.feats[sl.start:sl.stop], False)]
            fon = seg_on is None or bool(seg_on[b])
            if k_s is not None and fon:
                ss = k_s.layout[b]
                segs.append((k_s.feats[ss.start:ss.stop],
                             v_s.feats[ss.start:ss.stop], True))
            if k_c is not None and fon:
                cs = k_c.layout[b]
                segs.append((k_c.feats[cs.start:cs.stop],
                             v_c.feats[cs.start:cs.stop], True))
            K = torch.cat([s[0] for s in segs], dim=0)
            V = torch.cat([s[1] for s in segs], dim=0)
            bias = None
            if getattr(self, "test_identity_mask", False):     # G0 only
                H = qb.shape[1]
                parts = []
                for kf, _vf, foreign in segs:   # foreign segments get -inf
                    parts.append(
                        torch.full((H, kf.shape[0]), float("-inf"), device=qb.device)
                        if foreign else torch.zeros(H, kf.shape[0], device=qb.device))
                bias = torch.cat(parts, dim=-1)[None, :, None, :].to(qb.dtype)
            out = F.scaled_dot_product_attention(
                qb.transpose(0, 1)[None], K.transpose(0, 1)[None],
                V.transpose(0, 1)[None], attn_mask=bias,
            )[0].transpose(0, 1)
            outs.append(out)
        return q_x.replace(torch.cat(outs, dim=0))

    # ── coupling B (fallback): in-house gated two-softmax ───────────────────
    def _gated_cross_attn(self, idx, q_x, k_x, v_x, k_s, v_s, k_c=None, v_c=None,
                          cond_gates=None, gates=None, tok_on=None):
        """out = softmax(q_x·k_x)·v_x + g_idx · softmax(q_x·k_s)·v_s
        (+ gc_idx · softmax(q_x·k_c)·v_c in stream mode)
        (mmdit_slat.py:190-196 verbatim: per-head zero-init gate; robust to the
        MEASURED O(1e2-1e3) cross-logit scale of independently-trained ckpts).
        gates: the per-head gate table for the k_s term (default self.c_gates;
        the geo lane passes b_gates). tok_on: optional (T,1,1) 0/1 — per-token
        corner mask multiplying the foreign terms (bidir: exact 0 at t_s=0).
        All terms run the existing sparse flash kernels — no python-loop sdpa,
        no masks (audit item: kernel-reuse per mmdit_slat.py:213-217)."""
        qkv_x = q_x.replace(torch.stack([q_x.feats, k_x.feats, v_x.feats], dim=1))
        out_self = sparse_scaled_dot_product_attention(qkv_x)      # (T, H, D)
        out = out_self.feats
        if k_s is not None:
            out_cross = sparse_scaled_dot_product_attention(q_x, k_s, v_s)
            g_tab = self.c_gates if gates is None else gates
            gate = g_tab[idx].to(out.dtype).view(1, -1, 1)
            term = gate * out_cross.feats
            if tok_on is not None:
                term = term * tok_on.to(out.dtype)
            out = out + term
        if k_c is not None:
            out_cond = sparse_scaled_dot_product_attention(q_x, k_c, v_c)
            gc = cond_gates[idx].to(out.dtype).view(1, -1, 1)
            term = gc * out_cond.feats
            if tok_on is not None:
                term = term * tok_on.to(out.dtype)
            out = out + term
        return out_self.replace(out)

    # ── one stitched block pair ─────────────────────────────────────────────
    def _run_block_pair(self, idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x):
        gblk = self.geo_flow.blocks[idx]
        tblk = self.tex_flow.blocks[idx]

        # geo block — op-for-op the block's own _forward (modulated.py:142-161),
        # INCLUDING its fused Triton kernels: the eager-equivalent math is 1 ULP
        # off at real scale (G0 bisect, block 0) and breaks bit-exactness.
        with torch.no_grad():
            sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = _block_mod_params(gblk, mod_s)
            hn = fused_norm_modulate(h_s, gblk.norm1, sc_msa, sh_msa)
            q_s, k_s, v_s = _attn_qkv(gblk.self_attn, hn)
            qkv_s = q_s.replace(torch.stack([q_s.feats, k_s.feats, v_s.feats], dim=1))
            a = sparse_scaled_dot_product_attention(qkv_s)
            a = _attn_out(gblk.self_attn, a)
            h_s = fused_gate_residual(h_s, a, g_msa)
            hc = h_s.replace(gblk.norm2(h_s.feats))
            h_s = h_s + gblk.cross_attn(hc, cond_s) * self.xattn_scale
            hm = fused_norm_modulate(h_s, gblk.norm3, sc_mlp, sh_mlp)
            h_s = fused_gate_residual(h_s, gblk.mlp(hm), g_mlp)
            k_s = k_s.replace(k_s.feats.detach())
            v_s = v_s.replace(v_s.feats.detach())

        # tex block — same replication; self-attn step = gated cross-read.
        # use_checkpoint honored (audit: base block wraps _forward in
        # torch.utils.checkpoint when set; 30x1.3B without it = OOM)
        if getattr(tblk, "use_checkpoint", False) and torch.is_grad_enabled():
            import torch.utils.checkpoint as _ckpt
            return h_s, _ckpt.checkpoint(
                lambda hx: self._tex_block_inner(idx, hx, mod_x, cond_x, k_s, v_s),
                h_x, use_reentrant=False)
        return h_s, self._tex_block_inner(idx, h_x, mod_x, cond_x, k_s, v_s)

    # ── FUSED bidirectional block (MMDiT-native: one varlen call per lane) ──
    def _fused_joint_attn(self, q_s, k_s, v_s, q_x, k_x, v_x, plan):
        """Two native varlen flash calls, no mask, no per-sample python loop.
        MF dit.py:138-148 does exactly this (cat q/k/v of all streams → ONE
        attention → split by segment); the sparse kernel's per-sample kv
        lengths let us fold the corner exclusion into the layout instead."""
        fused_idx, fused_layout, geo_idx, geo_layout = plan
        k_all = torch.cat([k_s.feats, k_x.feats], dim=0)
        v_all = torch.cat([v_s.feats, v_x.feats], dim=0)
        k_f = sp.VarLenTensor(k_all[fused_idx], fused_layout)
        v_f = sp.VarLenTensor(v_all[fused_idx], fused_layout)
        a_x = sparse_scaled_dot_product_attention(q_x, k_f, v_f)
        if geo_idx is fused_idx:
            a_s = sparse_scaled_dot_product_attention(q_s, k_f, v_f)
        else:
            a_s = sparse_scaled_dot_product_attention(
                q_s, sp.VarLenTensor(k_all[geo_idx], geo_layout),
                sp.VarLenTensor(v_all[geo_idx], geo_layout))
        return a_s, a_x

    def _add_ss_read(self, a, q_untagged, gate_tab, idx, ss_kv):
        """slat <- SS: a SEPARATE gated softmax, added before to_out.

        q must be the UNTAGGED query. The +-pi/2 rotation of the 64th (identity
        pad) rope pair is a stream tag for the SHARED union softmax, which cannot
        otherwise tell which stream is asking. This read has its own softmax, so
        it needs no tag — and using the tagged q against untagged SS keys would
        apply a spurious half-turn to every cross-tower logit, invisible at
        init because the gate is zero.

        k/v stay DENSE (B, 4096, H, D): the sparse kernel has a native
        (VarLen q, dense k, dense v) overload that folds them to kv_seqlen
        [4096]*B itself, so there is no VarLen-ification and no per-sample loop.
        """
        if ss_kv is None or gate_tab is None:
            return a
        k_ss, v_ss = ss_kv
        r = sparse_scaled_dot_product_attention(q_untagged, k_ss, v_ss)
        g = gate_tab[idx].to(a.feats.dtype).reshape(1, -1, 1)
        return a.replace(a.feats + g * r.feats)

    def _run_block_pair_fused(self, idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x,
                              plan, ss_kv=None):
        """Standard-MMDiT block on the fused path. Stream tag is ABSOLUTE here
        (tex's q AND k rotated by π/2, geo's not) — a single shared softmax
        cannot know which stream is querying, so the tag must live on the
        tokens, exactly like MF's stream-id RoPE axis (model.py:70 gives the
        depth stream time_id=1.0 while img keeps 0.0, and apply_rope hits both
        q and k). Within-stream logits are unchanged (both sides rotate);
        cross-stream logits get ±π/2, antisymmetric by direction."""
        gblk = self.geo_flow.blocks[idx]
        tblk = self.tex_flow.blocks[idx]

        def inner(hs_in, hx_in):
            gsh, gsc, gg, gsh2, gsc2, gg2 = _block_mod_params(gblk, mod_s)
            tsh, tsc, tg, tsh2, tsc2, tg2 = _block_mod_params(tblk, mod_x)
            hn_s = fused_norm_modulate(hs_in, gblk.norm1, gsc, gsh)
            q_s, k_s, v_s = _attn_qkv(gblk.self_attn, hn_s)
            hn_x = fused_norm_modulate(hx_in, tblk.norm1, tsc, tsh)
            q_x, k_x, v_x = _attn_qkv(tblk.self_attn, hn_x)
            q_x_raw = q_x                                   # pre-tag, for the SS read
            q_x, k_x = _rotate_pad_pair(q_x), _rotate_pad_pair(k_x)   # stream tag
            a_s, a_x = self._fused_joint_attn(q_s, k_s, v_s, q_x, k_x, v_x, plan)
            a_s = self._add_ss_read(a_s, q_s, self.ss_gates_geo, idx, ss_kv)
            a_x = self._add_ss_read(a_x, q_x_raw, self.ss_gates_tex, idx, ss_kv)

            a_s = _attn_out(gblk.self_attn, a_s)
            hs = fused_gate_residual(hs_in, a_s, gg)
            hcs = hs.replace(gblk.norm2(hs.feats))
            hs = hs + gblk.cross_attn(hcs, cond_s) * self.xattn_scale
            hms = fused_norm_modulate(hs, gblk.norm3, gsc2, gsh2)
            hs = fused_gate_residual(hs, gblk.mlp(hms), gg2)

            a_x = _attn_out(tblk.self_attn, a_x)
            hx = fused_gate_residual(hx_in, a_x, tg)
            hcx = hx.replace(tblk.norm2(hx.feats))
            hx = hx + tblk.cross_attn(hcx, cond_x) * self.xattn_scale
            hmx = fused_norm_modulate(hx, tblk.norm3, tsc2, tsh2)
            hx = fused_gate_residual(hx, tblk.mlp(hmx), tg2)
            return hs, hx

        if getattr(tblk, "use_checkpoint", False) and torch.is_grad_enabled():
            import torch.utils.checkpoint as _ckpt
            return _ckpt.checkpoint(inner, h_s, h_x, use_reentrant=False)
        return inner(h_s, h_x)

    def _run_block_pair_bidir(self, idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x,
                              corner_on, tok_on, ss_kv=None):
        """BIDIRECTIONAL stitched block (user topology 2026-08-11): standard
        MMDiT phase order — both streams' QKV from PRE-update hiddens, geo
        attends [self; tex] (corner-masked), tex attends [self; geo], then both
        advance (residual + read-once cross-attn + MLP, verbatim fused ops).
        Geo WEIGHTS are frozen in S1 but the lane runs IN-GRAPH: b_gates and
        the geo-mediated tex feedback train through the tex loss. The whole
        pair is gradient-checkpointed (two 1.3B lanes live otherwise).
        corner_on: (B,) bool, t_s>0; tok_on: (T,1,1) float per-token version."""
        gblk = self.geo_flow.blocks[idx]
        tblk = self.tex_flow.blocks[idx]

        def inner(hs_in, hx_in):
            gsh, gsc, gg, gsh2, gsc2, gg2 = _block_mod_params(gblk, mod_s)
            tsh, tsc, tg, tsh2, tsc2, tg2 = _block_mod_params(tblk, mod_x)
            hn_s = fused_norm_modulate(hs_in, gblk.norm1, gsc, gsh)
            q_s, k_s, v_s = _attn_qkv(gblk.self_attn, hn_s)
            hn_x = fused_norm_modulate(hx_in, tblk.norm1, tsc, tsh)
            q_x, k_x, v_x = _attn_qkv(tblk.self_attn, hn_x)
            # geo lane: self + corner-masked tex read. Corner (reads-OFF) rows
            # take the FLASH pure-self result — bit-exact vs the specialist AND
            # kernel-identical to the tex|mesh cached path (the sdpa union is
            # 1-ULP-class off flash at real scale; G0-bidir caught it).
            # ABSOLUTE stream tag: tex's q/k rotated, geo's untouched (same
            # convention as the fused path and the cached inference path).
            q_x_raw = q_x                                   # pre-tag, for the SS read
            q_x, k_x = _rotate_pad_pair(q_x), _rotate_pad_pair(k_x)
            k_x_tag = k_x
            if self.coupling == "union":
                qkv_s = q_s.replace(torch.stack(
                    [q_s.feats, k_s.feats, v_s.feats], dim=1))
                a_flash = sparse_scaled_dot_product_attention(qkv_s)
                if bool(corner_on.any()):
                    a_u = self._union_attn(q_s, k_s, v_s, k_x_tag, v_x,
                                           seg_on=corner_on)
                    sel = tok_on.to(a_flash.feats.dtype)
                    a_s = a_flash.replace(a_u.feats * sel
                                          + a_flash.feats * (1 - sel))
                else:
                    a_s = a_flash
            else:
                a_s = self._gated_cross_attn(idx, q_s, k_s, v_s, k_x_tag, v_x,
                                             gates=self.b_gates, tok_on=tok_on)
            a_s = self._add_ss_read(a_s, q_s, self.ss_gates_geo, idx, ss_kv)
            a_s = _attn_out(gblk.self_attn, a_s)
            hs = fused_gate_residual(hs_in, a_s, gg)
            hcs = hs.replace(gblk.norm2(hs.feats))
            hs = hs + gblk.cross_attn(hcs, cond_s) * self.xattn_scale
            hms = fused_norm_modulate(hs, gblk.norm3, gsc2, gsh2)
            hs = fused_gate_residual(hs, gblk.mlp(hms), gg2)
            # tex lane: self + geo read (always on; foreign keys stream-tagged)
            k_s_tag = _rotate_pad_pair(k_s)
            if self.coupling == "union":
                a_x = self._union_attn(q_x, k_x, v_x, k_s_tag, v_s)
            else:
                a_x = self._gated_cross_attn(idx, q_x, k_x, v_x, k_s_tag, v_s)
            a_x = self._add_ss_read(a_x, q_x_raw, self.ss_gates_tex, idx, ss_kv)
            a_x = _attn_out(tblk.self_attn, a_x)
            hx = fused_gate_residual(hx_in, a_x, tg)
            hcx = hx.replace(tblk.norm2(hx.feats))
            hx = hx + tblk.cross_attn(hcx, cond_x) * self.xattn_scale
            hmx = fused_norm_modulate(hx, tblk.norm3, tsc2, tsh2)
            hx = fused_gate_residual(hx, tblk.mlp(hmx), tg2)
            return hs, hx

        if getattr(tblk, "use_checkpoint", False) and torch.is_grad_enabled():
            import torch.utils.checkpoint as _ckpt
            return _ckpt.checkpoint(inner, h_s, h_x, use_reentrant=False)
        return inner(h_s, h_x)

    def _tex_block_inner(self, idx, h_x, mod_x, cond_x, k_s, v_s, k_c=None, v_c=None,
                         ss_kv=None):
        tblk = self.tex_flow.blocks[idx]
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = _block_mod_params(tblk, mod_x)
        hn = fused_norm_modulate(h_x, tblk.norm1, sc_msa, sh_msa)
        q_x, k_x, v_x = _attn_qkv(tblk.self_attn, hn)
        # ABSOLUTE stream tag (the convention training runs under, via the fused
        # path): rotate THIS stream's q and k; foreign keys stay unrotated.
        # Within-stream logits unchanged, cross-stream get ∓π/2. Keeping the
        # cached tex|mesh inference path on the same convention is mandatory —
        # a relative tag here would silently mismatch the trained model.
        q_x_raw = q_x                                       # pre-tag, for the SS read
        q_x, k_x = _rotate_pad_pair(q_x), _rotate_pad_pair(k_x)
        if self.coupling == "union":
            a = self._union_attn(q_x, k_x, v_x, k_s, v_s, k_c, v_c)
        else:
            a = self._gated_cross_attn(idx, q_x, k_x, v_x, k_s, v_s, k_c, v_c,
                                       self.cond_gates_tex)
        a = self._add_ss_read(a, q_x_raw, self.ss_gates_tex, idx, ss_kv)
        a = _attn_out(tblk.self_attn, a)
        h_x = fused_gate_residual(h_x, a, g_msa)
        hc = h_x.replace(tblk.norm2(h_x.feats))
        h_x = h_x + tblk.cross_attn(hc, cond_x) * self.xattn_scale
        hm = fused_norm_modulate(h_x, tblk.norm3, sc_mlp, sh_mlp)
        h_x = fused_gate_residual(h_x, tblk.mlp(hm), g_mlp)
        return h_x

    # ── variant-B lanes (cond_mode="stream") ────────────────────────────────
    def _cond_block_lane(self, idx, h_c, mod_c, k_s=None, v_s=None,
                         k_x=None, v_x=None, tok_on_c=None):
        """One cond-stream block. Self-attn (NO rope — cond has no coords) plus,
        when voxel k/v are supplied, GATED reads of the geo and tex streams —
        this is what makes cond a real MMDiT stream rather than a deeper
        connector (SD3/FLUX: the text stream is updated by image context).
        `cond_reads_gate[i]` is per-head zero-init, so at warm start this lane
        is bit-identical to the read-only version. Eager modulate: the fused
        kernels are a bit-exactness tool for the CLONED voxel lanes; the cond
        stream is new capacity with no reference to match.
        h_c: VarLenTensor (B, L_i, D). Returns (h_c', k_c, v_c)."""
        i = idx - self.cond_first_idx
        cb = self.cond_blocks[i]
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = _block_mod_params(cb, mod_c)
        hn = h_c.replace(cb.norm1(h_c.feats))
        hn = hn * (1 + sc_msa) + sh_msa
        q_c, k_c, v_c = _attn_qkv_norope(cb.self_attn, hn)
        qkv_c = q_c.replace(torch.stack([q_c.feats, k_c.feats, v_c.feats], dim=1))
        a = sparse_scaled_dot_product_attention(qkv_c)
        out = a.feats
        for j, (kf, vf) in enumerate(((k_s, v_s), (k_x, v_x))):
            if kf is None:
                continue
            g = self.cond_reads_gate[i, j].to(out.dtype).view(1, -1, 1)
            term = g * sparse_scaled_dot_product_attention(q_c, kf, vf).feats
            if j == 1 and tok_on_c is not None:
                # CORNER MASK on cond<-tex, and this one pays for the whole K/V
                # cache. At t_s=0 (tex|mesh, the flagship mode) geometry is
                # locked, so if cond may read tex then cond changes every
                # denoising step, and because geo reads cond, GEO changes too —
                # both caches die and the flagship path costs ~3x. Masking this
                # single edge at the corner keeps cond and geo step-constant, so
                # only tex runs per step exactly as today, while t_s>0 (joint /
                # mesh-only) still gets the full three-way coupling.
                term = term * tok_on_c.to(out.dtype)
            out = out + term
        a = _attn_out(cb.self_attn, a.replace(out))
        h_c = h_c + a * g_msa
        hm = h_c.replace(cb.norm3(h_c.feats))
        hm = hm * (1 + sc_mlp) + sh_mlp
        h_c = h_c + cb.mlp(hm) * g_mlp
        return h_c, k_c, v_c

    def _run_block_triple(self, idx, h_s, h_x, h_c, mod_s, mod_x, mod_c,
                          cond_s, cond_x, geo_frozen, corner_on, tok_on,
                          tok_on_c=None):
        """THREE-STREAM MMDiT block (bidirectional, 2026-08-14).

        Blocks before `cond_first_idx` have no cond stream and delegate to the
        exact S2b two-stream block — so those layers stay bit-identical to the
        warm start and only the last N layers carry new machinery.

        For a cond block, MMDiT phase order: every stream's q/k/v comes from the
        PRE-update hidden, then each stream attends and advances.
          geo  <- self + corner-masked tex + gated cond
          tex  <- self + geo             + gated cond
          cond <- self + gated geo + gated tex        (the true-MMDiT direction)
        Every NEW read is behind a per-head zero-init gate, so at warm start the
        whole block collapses to the S2b block exactly — a bare union cannot do
        this for a new segment (its zero keys still enter the denominator).

        Stream tag: geo unrotated, tex = one pad-pair rotation, cond = two
        (180°, still a swap+negate so float-exact). Any tag is safe at init
        because the cond gates are zero; it becomes meaningful as they open."""
        if idx < self.cond_first_idx:
            h_s, h_x = self._run_block_pair_bidir(
                idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x, corner_on, tok_on)
            return h_s, h_x, h_c

        i = idx - self.cond_first_idx
        gblk = self.geo_flow.blocks[idx]
        tblk = self.tex_flow.blocks[idx]

        def inner(hs_in, hx_in, hc_in):
            gsh, gsc, gg, gsh2, gsc2, gg2 = _block_mod_params(gblk, mod_s)
            tsh, tsc, tg, tsh2, tsc2, tg2 = _block_mod_params(tblk, mod_x)
            hn_s = fused_norm_modulate(hs_in, gblk.norm1, gsc, gsh)
            q_s, k_s, v_s = _attn_qkv(gblk.self_attn, hn_s)
            hn_x = fused_norm_modulate(hx_in, tblk.norm1, tsc, tsh)
            q_x, k_x, v_x = _attn_qkv(tblk.self_attn, hn_x)
            q_x, k_x = _rotate_pad_pair(q_x), _rotate_pad_pair(k_x)

            # cond lane first: it reads the PRE-update voxel streams (gated) and
            # hands its k/v to both voxel lanes in the same block.
            hc_out, k_c, v_c = self._cond_block_lane(
                idx, hc_in, mod_c, k_s, v_s, k_x, v_x, tok_on_c)
            k_c = _rotate_pad_pair(_rotate_pad_pair(k_c))          # cond tag

            def add_cond(a, q, gate_tab):
                g = gate_tab[i].to(a.feats.dtype).view(1, -1, 1)
                return a.replace(a.feats + g * sparse_scaled_dot_product_attention(
                    q, k_c, v_c).feats)

            # ── geo lane ──
            if self.coupling == "union":
                qkv_s = q_s.replace(torch.stack(
                    [q_s.feats, k_s.feats, v_s.feats], dim=1))
                a_flash = sparse_scaled_dot_product_attention(qkv_s)
                if bool(corner_on.any()):
                    a_u = self._union_attn(q_s, k_s, v_s, k_x, v_x,
                                           seg_on=corner_on)
                    sel = tok_on.to(a_flash.feats.dtype)
                    a_s = a_flash.replace(a_u.feats * sel
                                          + a_flash.feats * (1 - sel))
                else:
                    a_s = a_flash
            else:
                a_s = self._gated_cross_attn(idx, q_s, k_s, v_s, k_x, v_x,
                                             gates=self.b_gates, tok_on=tok_on)
            a_s = add_cond(a_s, q_s, self.cond_gates_geo)
            a_s = _attn_out(gblk.self_attn, a_s)
            hs = fused_gate_residual(hs_in, a_s, gg)
            hcs = hs.replace(gblk.norm2(hs.feats))
            hs = hs + gblk.cross_attn(hcs, cond_s) * self.xattn_scale
            hms = fused_norm_modulate(hs, gblk.norm3, gsc2, gsh2)
            hs = fused_gate_residual(hs, gblk.mlp(hms), gg2)

            # ── tex lane ──
            k_s_tag = _rotate_pad_pair(k_s)
            if self.coupling == "union":
                a_x = self._union_attn(q_x, k_x, v_x, k_s_tag, v_s)
            else:
                a_x = self._gated_cross_attn(idx, q_x, k_x, v_x, k_s_tag, v_s)
            a_x = add_cond(a_x, q_x, self.cond_gates_tex)
            a_x = _attn_out(tblk.self_attn, a_x)
            hx = fused_gate_residual(hx_in, a_x, tg)
            hcx = hx.replace(tblk.norm2(hx.feats))
            hx = hx + tblk.cross_attn(hcx, cond_x) * self.xattn_scale
            hmx = fused_norm_modulate(hx, tblk.norm3, tsc2, tsh2)
            hx = fused_gate_residual(hx, tblk.mlp(hmx), tg2)
            return hs, hx, hc_out

        # ALWAYS checkpoint a three-stream block, regardless of what the elastic
        # controller decided. TRELLIS's policy checkpoints the FIRST n blocks,
        # but the cond stream sits on the LAST N — i.e. exactly the blocks the
        # controller leaves un-checkpointed, and now the heaviest ones (three
        # lanes of activations instead of two). Honouring the controller here
        # OOM'd every rank at bs4 on the first step.
        if torch.is_grad_enabled():
            import torch.utils.checkpoint as _ckpt
            return _ckpt.checkpoint(inner, h_s, h_x, h_c, use_reentrant=False)
        return inner(h_s, h_x, h_c)

    # ── elastic activation checkpointing (TRELLIS's own controller) ─────────
    # Audit finding 2026-08-12: the first version replaced TRELLIS's ADAPTIVE
    # policy with a static fraction. The block-selection formula was already
    # identical to theirs (checkpoint the first n blocks — sparse_elastic_mixin
    # .py:17-23); what was missing is the LinearMemoryController that picks n
    # per step from a fitted memory model. Here we provide only the two hooks
    # (input size + the mem_ratio context) and let their controller decide, the
    # same contract ElasticSLatFlowModel uses. Static geotex_gc stays available
    # as a kill-switch.
    def register_memory_controller(self, controller):
        self._memory_controller = controller
        return self

    def _get_input_size(self, x_s, *args, **kwargs):
        return x_s.feats.shape[0]

    @contextmanager
    def _with_mem_ratio(self, mem_ratio: float = 1.0):
        """Formula verbatim from sparse_elastic_mixin.py:17-23, applied to the
        TEX block list (the fused/bidir pair is checkpointed through that flag)."""
        blocks = self.tex_flow.blocks
        n = len(blocks)
        if mem_ratio >= 1.0:
            for b in blocks:
                b.use_checkpoint = False
            yield 1.0
        else:
            n_ck = min(math.ceil((1 - mem_ratio) * n) + 1, n)
            exact = 1 - (n_ck - 1) / n
            for i, b in enumerate(blocks):
                b.use_checkpoint = (i < n_ck)
            yield exact
        for b in blocks:                       # restore (their convention)
            b.use_checkpoint = False

    # ── v10: the SS lane, block by block ────────────────────────────────────
    def ss_prologue(self, x_ss, t_ss, cond_ss):
        """sparse_structure_flow.py:234-241 — everything before the block loop.

        Returns (h, mod_ss, cond_ss) in the tower's own dtype. Split out from the
        loop so the caller can interleave SS blocks with the slat pair rather
        than running the tower to completion first: pre-running it would mean
        holding all 30 blocks' (k, v) live across the whole slat pass, which is
        755 MB per sample at 4096 tokens.
        """
        ss = self.ss_flow
        assert list(x_ss.shape) == [x_ss.shape[0], ss.in_channels] + [ss.resolution] * 3, \
            f"SS input {tuple(x_ss.shape)} != (B, {ss.in_channels}, {ss.resolution}^3)"
        h = x_ss.view(*x_ss.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = ss.input_layer(h)
        t_emb = ss.t_embedder(t_ss)
        if ss.share_mod:
            t_emb = ss.adaLN_modulation(t_emb)
        return (manual_cast(h, ss.dtype), manual_cast(t_emb, ss.dtype),
                manual_cast(cond_ss, ss.dtype))

    def _run_ss_block(self, idx, h, mod_ss, cond_ss, ss_cond_mask=None,
                      want_kv: bool = False, ss_read=None):
        """modulated.py:148-165 replicated op-for-op for one dense SS block.

        want_kv returns (k_pre_rope, v) for the slat lanes to borrow.
        ss_read is the SS<-slat term (v10 keeps it dormant); it is ADDED to the
        attention output before to_out, the same place the slat lanes add theirs.
        """
        ss = self.ss_flow
        blk = ss.blocks[idx]
        sh1, sc1, g1, sh2, sc2, g2 = _dense_mod_params(blk, mod_ss)

        hn = blk.norm1(h)
        hn = hn * (1 + sc1.unsqueeze(1)) + sh1.unsqueeze(1)
        q, k, v, k_pre = _dense_attn_qkv(blk.self_attn, hn, ss.rope_phases)
        a = _dense_sdpa(q, k, v)
        if ss_read is not None:
            a = a + ss_read
        a = _dense_attn_out(blk.self_attn, a)
        h = h + a * g1.unsqueeze(1)

        # * xattn_scale for the same reason the two slat lanes do: it is the
        # knob that anneals the read-once cross-attn away when a cond STREAM
        # takes over. Inert at 1.0 (the cross_attn cond_mode v10 runs), but
        # leaving it off would mean an anneal silently starved two towers of
        # conditioning while the third kept its own — an asymmetry nothing in
        # the logs would show.
        h = h + blk.cross_attn(blk.norm2(h), cond_ss,
                               attn_mask=ss_cond_mask) * self.xattn_scale

        hm = blk.norm3(h)
        hm = hm * (1 + sc2.unsqueeze(1)) + sh2.unsqueeze(1)
        h = h + blk.mlp(hm) * g2.unsqueeze(1)
        return (h, k_pre, v) if want_kv else (h, None, None)

    def ss_epilogue(self, h, out_dtype):
        """sparse_structure_flow.py:243-247 — layer_norm, out_layer, reshape back
        to (B, C, res, res, res)."""
        ss = self.ss_flow
        h = manual_cast(h, out_dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = ss.out_layer(h)
        return h.permute(0, 2, 1).view(
            h.shape[0], h.shape[2], *[ss.resolution] * 3).contiguous()

    def ss_forward(self, x_ss, t_ss, cond_ss, ss_cond_mask=None):
        """The SS tower run standalone THROUGH the replicated lane. Exists so the
        replication can be certified against ss_flow(...) directly; the joint
        forward uses the same three pieces interleaved with the slat blocks."""
        h, mod_ss, c = self.ss_prologue(x_ss, t_ss, cond_ss)
        for i in range(len(self.ss_flow.blocks)):
            h, _, _ = self._run_ss_block(i, h, mod_ss, c, ss_cond_mask)
        return self.ss_epilogue(h, x_ss.dtype)

    def _ss_reads_slat(self, idx, h_ss, h_s, h_x, ss_read_on):
        """SS <- slat, the dormant direction. Returns None unless explicitly on.

        THE ROW MASK IS CORRECTNESS, NOT AN OPTIMISATION. At training time the
        slat lanes live on GT-derived coords, which ARE the occupancy the SS
        tower is being asked to predict. A row where SS may read them is a row
        where SS can copy the answer, and the symptom is a BETTER loss curve, so
        nothing downstream will complain. ss_read_on is the per-sample gate: it
        is on only for lag rows (t_ss > 0 with a slat context that is itself
        noised), off for the t_ss=0 clean rows where SS already holds the answer
        and off for solo rows where the slat lanes are pure noise.

        ss_reads_enabled is the separate, global inference switch — v10 samples
        with it False so the SS tower's trajectory is exactly the specialist's.
        """
        if (self.ss_reads_gate is None or not self.ss_reads_enabled
                or ss_read_on is None or not bool(ss_read_on.any())):
            return None
        ss, gblk = self.ss_flow, self.geo_flow.blocks[idx]
        tblk = self.tex_flow.blocks[idx]
        # q from the SS hidden through the SS block's own q projection; k/v from
        # the slat lanes' PRE-update hiddens, so the direction is symmetric with
        # the slat<-SS read that happens in the same block.
        blk = ss.blocks[idx]
        B, L, _ = h_ss.shape
        q = blk.self_attn.to_qkv(h_ss).reshape(B, L, 3, blk.self_attn.num_heads, -1)[:, :, 0]
        if blk.self_attn.qk_rms_norm:
            q = blk.self_attn.q_rms_norm(q)
        out = 0.0
        for j, (hh, hb) in enumerate(((h_s, gblk), (h_x, tblk))):
            k, v = _attn_qkv(hb.self_attn, hh)[1:]
            r = sparse_scaled_dot_product_attention(q, k, v)
            g = self.ss_reads_gate[idx, j].to(r.dtype).reshape(1, 1, -1, 1)
            out = out + g * r
        m = ss_read_on.to(out.dtype).reshape(-1, 1, 1, 1)
        return out * m

    def ss_kv_for_slat(self, k_pre, v):
        """Re-rope a borrowed SS key into the 32^3 slat frame.

        The key is rotated by the SS grid mapped through c -> 2c+0.5, so a slat
        query at voxel p and an SS key at cell c see the relative position
        (p - (2c+0.5)) that they would if both lived in the 32^3 grid. v is never
        roped anywhere in TRELLIS, so it passes through untouched.
        """
        ph = self.ss_phases_slat.to(k_pre.device)
        return RotaryPositionEmbedder.apply_rotary_embedding(k_pre, ph), v

    def unfreeze_geo(self):
        """Stage-2 switch: geo joins training (three-pack per design doc —
        tri-modal data + self-distill + real G3 must ride along)."""
        for p in self.geo_flow.parameters():
            p.requires_grad_(True)
        self._geo_unfrozen = True
        return self

    def train(self, mode: bool = True):
        """Trainer.train() must NOT flip the frozen geo stream out of eval
        (audit: durability of the freeze guarantee). unfreeze_geo() lifts it."""
        super().train(mode)
        if not getattr(self, "_geo_unfrozen", False):
            self.geo_flow.eval()
        return self

    def trainable_towers(self):
        """(names, param counts) of every tower that will receive gradient — the
        one place to read what `all_trainable` actually did."""
        out = {}
        for nm in ("ss_flow", "geo_flow", "tex_flow"):
            m = getattr(self, nm, None)
            if m is not None:
                out[nm] = sum(p.numel() for p in m.parameters() if p.requires_grad)
        out["gates"] = sum(p.numel() for n, p in self.named_parameters()
                           if p.requires_grad and ("gates" in n or "mixer" in n
                                                   or "alpha" in n))
        return out

    # ── full forward ────────────────────────────────────────────────────────
    def forward(self, x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=None,
                x_ss=None, t_ss=None, cond_ss=None, ss_cond_mask=None,
                ss_read_on=None):
        """Elastic-checkpointing wrapper (TRELLIS contract, elastic_utils.py:
        get_mem_ratio → with_mem_ratio → update_run_states) around _forward_impl.
        Inactive unless a controller is registered, so the static geotex_gc
        path is untouched."""
        ctrl = getattr(self, "_memory_controller", None)
        if ctrl is None or not torch.is_grad_enabled() or not self.training:
            return self._forward_impl(x_s, x_x, t_s, t_x, cond_s, cond_x,
                                      tex_concat_cond, x_ss, t_ss, cond_ss,
                                      ss_cond_mask, ss_read_on)
        n = self._get_input_size(x_s)
        with self._with_mem_ratio(ctrl.get_mem_ratio(n)) as exact:
            out = self._forward_impl(x_s, x_x, t_s, t_x, cond_s, cond_x,
                                     tex_concat_cond, x_ss, t_ss, cond_ss,
                                     ss_cond_mask, ss_read_on)
        ctrl.update_run_states(n, exact)
        return out

    def _forward_impl(self, x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=None,
                      x_ss=None, t_ss=None, cond_ss=None, ss_cond_mask=None,
                      ss_read_on=None):
        """x_s/x_x: noisy SparseTensors (32ch each, shared coords). t_s/t_x: (B,)
        in flow units (already *1000). cond_s/cond_x: per-stream cond tensors
        (SLatFlowModel contract). tex_concat_cond: SparseTensor (32ch, TEX norm
        space ≡ geo space, measured identical) — joint-mode callers pass the geo
        stream's current state directly.
        Returns (v_s_pred, v_x_pred)."""
        geo, tex = self.geo_flow, self.tex_flow

        # coords parity (audit: sparse_cat and the cross-read both pair tokens
        # POSITIONALLY — a coord-order mismatch would silently misalign; the
        # EditV1 90-degree lesson class)
        if tex_concat_cond is not None:
            assert torch.equal(x_s.coords, x_x.coords), "geo/tex coords mismatch"
            assert torch.equal(x_x.coords, tex_concat_cond.coords), "concat_cond coords mismatch"
            x_x_in = sp.sparse_cat([x_x, tex_concat_cond], dim=-1)
        else:
            assert torch.equal(x_s.coords, x_x.coords), "geo/tex coords mismatch"
            x_x_in = x_x
        if isinstance(cond_s, list):
            cond_s = sp.VarLenTensor.from_tensor_list(cond_s)
        if isinstance(cond_x, list):
            cond_x = sp.VarLenTensor.from_tensor_list(cond_x)

        import contextlib
        geo_frozen = not next(geo.parameters()).requires_grad
        gctx = (torch.no_grad if geo_frozen else contextlib.nullcontext)

        with gctx():
            h_s = manual_cast(geo.input_layer(x_s), geo.dtype)
            cond_s = manual_cast(cond_s, geo.dtype)
        if self.bidirectional:
            # geo knows t_x (symmetric mixer) — IN-GRAPH: cross_alpha_s/mixer_s
            # are new trainable params fed by the tex loss
            t_emb_s = geo.t_embedder(t_s) + self.cross_alpha_s * self.t_mixer_s(t_x)
            mod_s = manual_cast(geo.adaLN_modulation(t_emb_s), geo.dtype)
        else:
            with gctx():
                mod_s = manual_cast(geo.adaLN_modulation(geo.t_embedder(t_s)), geo.dtype)

        h_x = manual_cast(tex.input_layer(x_x_in), tex.dtype)
        # cross-t mixing BEFORE adaLN (vec level, once — MF dit.py:388-413)
        t_emb_x = tex.t_embedder(t_x) + self.cross_alpha * self.t_mixer(t_s)
        mod_x = manual_cast(tex.adaLN_modulation(t_emb_x), tex.dtype)
        cond_x = manual_cast(cond_x, tex.dtype)

        if self.cond_mode == "stream":
            # cond stream input = the tex connector's cond; dense (B,L,C) test
            # inputs get VarLen-ified so the lane API is uniform
            cx_vl = cond_x
            if torch.is_tensor(cx_vl):
                cx_vl = sp.VarLenTensor.from_tensor_list(list(cx_vl.unbind(0)))
            h_c = cx_vl.replace(self.cond_proj(cx_vl.feats))
            # constant t_c≡0 modulation (the approved per-sample-constant / K/V
            # cache property; live-t would break it)
            mod_c = manual_cast(
                tex.adaLN_modulation(tex.t_embedder(torch.zeros_like(t_x))), tex.dtype)
            corner_on = (t_s != 0)
            tok_on = torch.cat([
                corner_on[b].to(x_s.feats.dtype).expand(sl.stop - sl.start)
                for b, sl in enumerate(x_s.layout)]).view(-1, 1, 1)
            tok_on_c = torch.cat([
                corner_on[b].to(h_c.feats.dtype).expand(sl.stop - sl.start)
                for b, sl in enumerate(h_c.layout)]).view(-1, 1, 1)
            for idx in range(len(geo.blocks)):
                h_s, h_x, h_c = self._run_block_triple(
                    idx, h_s, h_x, h_c, mod_s, mod_x, mod_c, cond_s, cond_x,
                    geo_frozen, corner_on, tok_on, tok_on_c)
        elif self.bidirectional:
            # corner mask from t (flow units: 0 iff raw t_s==0) — per-sample
            # bool for hard union exclusion + per-token 0/1 for the gated term
            corner_on = (t_s != 0)
            # ── v10: the SS lane runs INTERLEAVED with the slat pair ──
            # Block i of SS produces the (k, v) block i of the slat lanes read,
            # so only one block's pair is ever live. Running the tower to
            # completion first would hold all 30 pairs across the whole slat
            # pass: 30 x 2 x 4096 x 1536 x 2B = 755 MB per sample.
            ss_on = x_ss is not None and self.ss_flow is not None
            if ss_on:
                h_ss, mod_ss, c_ss = self.ss_prologue(x_ss, t_ss, cond_ss)
            if self.fused_attn and self.coupling == "union":
                # plan computed ONCE per forward (layout + t_s are step-constant)
                plan = _fusion_plan(x_s.layout, corner_on, x_s.feats.device)
                for idx in range(len(geo.blocks)):
                    ss_kv = None
                    if ss_on:
                        h_ss, k_pre, v_ss = self._run_ss_block(
                            idx, h_ss, mod_ss, c_ss, ss_cond_mask, want_kv=True,
                            ss_read=self._ss_reads_slat(idx, h_ss, h_s, h_x, ss_read_on))
                        ss_kv = self.ss_kv_for_slat(k_pre, v_ss)
                    h_s, h_x = self._run_block_pair_fused(
                        idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x, plan, ss_kv=ss_kv)
            else:
                tok_on = torch.cat([
                    corner_on[b].to(x_s.feats.dtype).expand(sl.stop - sl.start)
                    for b, sl in enumerate(x_s.layout)]).view(-1, 1, 1)
                for idx in range(len(geo.blocks)):
                    ss_kv = None
                    if ss_on:
                        h_ss, k_pre, v_ss = self._run_ss_block(
                            idx, h_ss, mod_ss, c_ss, ss_cond_mask, want_kv=True,
                            ss_read=self._ss_reads_slat(idx, h_ss, h_s, h_x, ss_read_on))
                        ss_kv = self.ss_kv_for_slat(k_pre, v_ss)
                    h_s, h_x = self._run_block_pair_bidir(
                        idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x, corner_on, tok_on,
                        ss_kv=ss_kv)
        else:
            for idx in range(len(geo.blocks)):
                h_s, h_x = self._run_block_pair(idx, h_s, h_x, mod_s, mod_x,
                                                cond_s, cond_x)

        with gctx():
            h_s = manual_cast(h_s, x_s.dtype)
            h_s = h_s.replace(F.layer_norm(h_s.feats, h_s.feats.shape[-1:]))
            v_s = geo.out_layer(h_s)
        h_x = manual_cast(h_x, x_x.dtype)
        h_x = h_x.replace(F.layer_norm(h_x.feats, h_x.feats.shape[-1:]))
        v_x = tex.out_layer(h_x)
        if x_ss is not None and self.ss_flow is not None:
            return v_s, v_x, self.ss_epilogue(h_ss, x_ss.dtype)
        return v_s, v_x

    # ── inference helpers (3-mode sampler; geotex_sampler.py) ───────────────
    @torch.no_grad()
    def precompute_geo_kv(self, x_s, t_s, cond_s, want_v: bool = False):
        """One full geo pass; returns (kv, v_s|None) with kv[i] = the block-i
        (k_s, v_s) post rms+RoPE — exactly what _run_block_pair borrows.
        tex|mesh mode calls this ONCE (t_s≡0 + clean shape state are constant
        across all tex steps → geo cost amortized to a single pass); joint mode
        calls it per step with want_v=True (v_s integrates the geo stream, kv
        feeds the same step's tex read — no double geo compute).
        Ops are verbatim _run_block_pair's geo lane (G0-certified bit-exact)."""
        geo = self.geo_flow
        if isinstance(cond_s, list):
            cond_s = sp.VarLenTensor.from_tensor_list(cond_s)
        h_s = manual_cast(geo.input_layer(x_s), geo.dtype)
        mod_s = manual_cast(geo.adaLN_modulation(geo.t_embedder(t_s)), geo.dtype)
        cond_s = manual_cast(cond_s, geo.dtype)
        kv = []
        for gblk in geo.blocks:
            sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = _block_mod_params(gblk, mod_s)
            hn = fused_norm_modulate(h_s, gblk.norm1, sc_msa, sh_msa)
            q_s, k_s, v_s = _attn_qkv(gblk.self_attn, hn)
            kv.append((k_s, v_s))
            qkv_s = q_s.replace(torch.stack([q_s.feats, k_s.feats, v_s.feats], dim=1))
            a = sparse_scaled_dot_product_attention(qkv_s)
            a = _attn_out(gblk.self_attn, a)
            h_s = fused_gate_residual(h_s, a, g_msa)
            hc = h_s.replace(gblk.norm2(h_s.feats))
            h_s = h_s + gblk.cross_attn(hc, cond_s) * self.xattn_scale
            hm = fused_norm_modulate(h_s, gblk.norm3, sc_mlp, sh_mlp)
            h_s = fused_gate_residual(h_s, gblk.mlp(hm), g_mlp)
        v = None
        if want_v:
            h_s = manual_cast(h_s, x_s.dtype)
            h_s = h_s.replace(F.layer_norm(h_s.feats, h_s.feats.shape[-1:]))
            v = geo.out_layer(h_s)
        return kv, v

    @torch.no_grad()
    def tex_forward_cached(self, x_x, t_x, t_s, cond_x, tex_concat_cond, kv):
        """Tex lane only, consuming precomputed geo (k,v). Mirrors forward()'s
        tex lane exactly — including the cross-t mixing, which still needs t_s
        even when the geo pass is cached."""
        tex = self.tex_flow
        assert torch.equal(x_x.coords, tex_concat_cond.coords), "concat_cond coords mismatch"
        x_x_in = sp.sparse_cat([x_x, tex_concat_cond], dim=-1)
        if isinstance(cond_x, list):
            cond_x = sp.VarLenTensor.from_tensor_list(cond_x)
        h_x = manual_cast(tex.input_layer(x_x_in), tex.dtype)
        t_emb_x = tex.t_embedder(t_x) + self.cross_alpha * self.t_mixer(t_s)
        mod_x = manual_cast(tex.adaLN_modulation(t_emb_x), tex.dtype)
        cond_x = manual_cast(cond_x, tex.dtype)
        for idx in range(len(tex.blocks)):
            k_s, v_s = kv[idx]
            h_x = self._tex_block_inner(idx, h_x, mod_x, cond_x, k_s, v_s)
        h_x = manual_cast(h_x, x_x.dtype)
        h_x = h_x.replace(F.layer_norm(h_x.feats, h_x.feats.shape[-1:]))
        return tex.out_layer(h_x)

    # ── trainable set (Stage-1) ─────────────────────────────────────────────
    def stage1_trainable_parameters(self, connector_tex=None):
        """Full tex stream + gates + t mixer (+ the tex connector — audit: the
        loss runs it with grad, so the optimizer must own it). Geo excluded."""
        params = list(self.tex_flow.parameters())
        params += list(self.t_mixer.parameters())
        params.append(self.cross_alpha)
        if self.c_gates is not None:
            params.append(self.c_gates)
        if self.b_gates is not None:
            # geo←tex gates train THROUGH the tex loss (geo lane runs in-graph
            # with frozen weights — the only gradient source they have in S1)
            params.append(self.b_gates)
        if self.t_mixer_s is not None:
            params += list(self.t_mixer_s.parameters())
            params.append(self.cross_alpha_s)
        if connector_tex is not None:
            params += list(connector_tex.parameters())
        return [p for p in params if p.requires_grad]

    def stage2_trainable_parameters(self, connector_tex=None, connector_geo=None):
        """Variant-B / Stage-2 full set: stage-1 set + cond stream (proj, blocks,
        gates) + — after unfreeze_geo() — the geo stream (+ geo connector).
        Design doc: geo unfreezing must ride with the three-pack (tri-modal data
        + self-distill + real G3)."""
        params = self.stage1_trainable_parameters(connector_tex)
        if self.cond_proj is not None:
            params += list(self.cond_proj.parameters())
            params += list(self.cond_blocks.parameters())
            for g in (self.cond_gates_tex, self.cond_gates_geo,
                      self.cond_reads_gate):
                if g is not None:
                    params.append(g)
        params += [p for p in self.geo_flow.parameters() if p.requires_grad]
        if connector_geo is not None:
            params += [p for p in connector_geo.parameters() if p.requires_grad]
        return [p for p in params if p.requires_grad]


# ─────────────────────────────────────────────────────────────────────────────
# assembly from two production run checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _load_prefixed_state(ckpt_dir: str, prefix: str, weights_file: str = "model.safetensors",
                         use_ema: bool = True) -> Dict[str, torch.Tensor]:
    """Prefix-filtered state dict WITH the EMA overlay (benchmarks/checkpoint.py
    convention: state.update(ema.safetensors) — every published eval/demo number
    was measured on EMA weights; review finding 2026-08-11)."""
    from safetensors.torch import load_file
    full = load_file(os.path.join(ckpt_dir, weights_file))
    ema_path = os.path.join(ckpt_dir, "ema.safetensors")
    if use_ema and os.path.isfile(ema_path):
        full.update(load_file(ema_path))
    out = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"prefix {prefix!r} matched nothing in {ckpt_dir}/{weights_file}")
    return out


def assemble_unified(shape_run_ckpt: str, tex_run_ckpt: str,
                     cond_mode: str = "cross_attn", coupling: str = "union",
                     bidirectional: bool = False,
                     weights_file: str = "model.safetensors") -> UnifiedGeoTexFlow:
    """Build both specialist flows via the production builders and load their
    trained weights by prefix. STRICT loading (silent-misload guard)."""
    from blip3o.model.multimodal_decoder.builder import (
        build_shape_slat_512, build_tex_slat_512)

    class _Cfg:                                    # builders read only these attrs
        trellis_shape_slat_ckpt = None
        trellis_tex_slat_ckpt = None

    geo = build_shape_slat_512(_Cfg())
    tex = build_tex_slat_512(_Cfg())
    geo.load_state_dict(_load_prefixed_state(
        shape_run_ckpt, "shape_slat_512.", weights_file), strict=True)
    tex.load_state_dict(_load_prefixed_state(
        tex_run_ckpt, "tex_slat_512.", weights_file), strict=True)
    return UnifiedGeoTexFlow(geo, tex, cond_mode=cond_mode, coupling=coupling,
                             bidirectional=bidirectional)


def assemble_unified_tri(shape_run_ckpt: str, tex_run_ckpt: str, ss_run_ckpt: str,
                         cond_mode: str = "cross_attn", coupling: str = "union",
                         bidirectional: bool = True, all_trainable: bool = True,
                         cond_seg_embed: bool = False,
                         cond_patch_pos: str = "off", cond_patch_lattice: int = 32,
                         weights_file: str = "model.safetensors") -> UnifiedGeoTexFlow:
    """v10: the THREE-tower assembly, warm from the s3_t50 specialists.

    Warm rather than official on purpose. This project's own adaptation
    curriculum was connector-only -> --flow_tune last20 -> full, and the s3_*
    towers are the output of the first two steps: they already read our v2.2
    conditioning. Starting from official weights would additionally require a
    fresh connector, i.e. breaking two priors at once on the tower that leads the
    cascade. (The s3_ss EMA holding ~132 tensors against a 641-tensor tower is
    what `last20` looks like from the outside — blocks 24-29 plus the connector.)

    STRICT loading everywhere: a prefix that matches nothing raises, and a
    partial match would be the silent-misload this repo has paid for before.
    """
    from blip3o.model.multimodal_decoder.builder import (
        build_shape_slat_512, build_ss_flow, build_tex_slat_512)

    class _Cfg:                                    # builders read only these attrs
        trellis_shape_slat_ckpt = None
        trellis_tex_slat_ckpt = None
        trellis_ss_flow_ckpt = None

    geo = build_shape_slat_512(_Cfg())
    tex = build_tex_slat_512(_Cfg())
    ss = build_ss_flow(_Cfg())
    geo.load_state_dict(_load_prefixed_state(
        shape_run_ckpt, "shape_slat_512.", weights_file), strict=True)
    tex.load_state_dict(_load_prefixed_state(
        tex_run_ckpt, "tex_slat_512.", weights_file), strict=True)
    # `ss_flow.` and NOT `ss_flow._orig_mod.`: whether the run was torch.compiled
    # is a property of that run, not of the weights, so strip the wrapper prefix
    # if it is there and let strict=True catch anything else.
    ss_sd = _load_prefixed_state(ss_run_ckpt, "ss_flow.", weights_file)
    ss_sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in ss_sd.items()}
    ss.load_state_dict(ss_sd, strict=True)
    return UnifiedGeoTexFlow(geo, tex, cond_mode=cond_mode, coupling=coupling,
                             bidirectional=bidirectional, ss_flow=ss,
                             all_trainable=all_trainable,
                             cond_seg_embed=cond_seg_embed,
                             cond_patch_pos=cond_patch_pos,
                             cond_patch_lattice=cond_patch_lattice)


def load_tri_connectors(shape_run_ckpt: str, tex_run_ckpt: str, ss_run_ckpt: str,
                        weights_file: str = "model.safetensors"):
    """(geo, tex, ss) connectors — each stream keeps the one it was trained with.

    Every s3 run stores its own under `diffusion_connector.` (that name is the
    stage-agnostic convention, not a claim about which stream it serves), so all
    three come from the same prefix in three different run dirs.
    """
    from trellis2_blip3o.connector import TRELLIS2Connector
    conns = []
    for ck in (shape_run_ckpt, tex_run_ckpt, ss_run_ckpt):
        cfg = json.load(open(os.path.join(ck, "config.json")))
        assert cfg.get("cond_adapter") == "mlp", \
            f"{ck}: cond_adapter={cfg.get('cond_adapter')!r} — arch-flag mismatch"
        assert not cfg.get("cond_pos_stamp", False), f"{ck}: unexpected pos_stamp"
        sd = _load_prefixed_state(ck, "diffusion_connector.", weights_file)
        cond_dim, vlm_dim = sd["fc1.weight"].shape
        conn = TRELLIS2Connector(vlm_hidden_dim=vlm_dim, trellis_cond_dim=cond_dim)
        conn.load_state_dict(sd, strict=True)
        conns.append(conn)
    return conns[0], conns[1], conns[2]


def assemble_unified_from_run(run_ckpt: str, cond_mode: str = "cross_attn",
                              coupling: str = "union", bidirectional: bool = False,
                              weights_file: str = "model.safetensors",
                              cond_stream_blocks: int = 10) -> UnifiedGeoTexFlow:
    """CONTINUE TRAINING from a geotex run checkpoint (S1 -> S2b).

    A geotex run saves the whole unified model under `unified_geotex.*`, so
    assemble_unified()'s specialist prefixes (`shape_slat_512.` /
    `tex_slat_512.`) match nothing there. This builds the same architecture via
    the TRAINING builders (uniform bf16 — see builder._uniform_bf16) and loads
    the run's state strictly, so S2b starts from everything S1 learned: the
    adapted tex stream AND the opened cross-t gates, not from the raw
    specialists."""
    from blip3o.model.multimodal_decoder.builder import (
        build_shape_slat_512, build_tex_slat_512)

    class _Cfg:
        trellis_shape_slat_ckpt = None
        trellis_tex_slat_ckpt = None

    uni = UnifiedGeoTexFlow(build_shape_slat_512(_Cfg()), build_tex_slat_512(_Cfg()),
                            cond_mode=cond_mode, coupling=coupling,
                            bidirectional=bidirectional,
                            cond_stream_blocks=cond_stream_blocks)
    sd = _load_prefixed_state(run_ckpt, "unified_geotex.", weights_file)
    if cond_mode == "stream":
        # A two-stream ckpt has none of the cond-stream tensors. strict=False is
        # correct here but must not become a silent sink for typos: assert that
        # EVERY missing key is a cond-stream key and that nothing is unexpected.
        missing, unexpected = uni.load_state_dict(sd, strict=False)
        bad = [k for k in missing
               if not k.startswith(("cond_proj.", "cond_blocks.", "cond_gates_",
                                    "cond_reads_gate"))]
        assert not bad, f"[geotex] non-cond keys missing from ckpt: {bad[:8]}"
        assert not unexpected, f"[geotex] unexpected keys in ckpt: {list(unexpected)[:8]}"
        print(f"[geotex] cond-stream: {len(missing)} new tensors initialised "
              f"(blocks >= {uni.cond_first_idx}); all other weights loaded")
    else:
        uni.load_state_dict(sd, strict=True)
    return uni


def _from_pretrained_pair():
    """The two base flows in TRELLIS's INFERENCE dtype layout: fp32 boundary
    layers + bf16 blocks + self.dtype=bf16 (t2models.from_pretrained — measured
    2026-08-11: input_layer fp32, block mlp bf16). The training builders instead
    cast EVERYTHING bf16 (measured 2026-08-12: that hard cast is worth 2.1x
    training throughput because the fused Triton modulate kernels need uniform
    dtype — see builder._uniform_bf16), and an all-bf16 input_layer rejects the
    fp32 latents every sampler in this repo uses.

    So train and inference deliberately run different dtype LAYOUTS (same
    weights): training all-bf16 for the fused kernels, sampling mixed for fp32
    latents. This is NOT an oversight and NOT specific to the unified model —
    eval_fusion_v22 / eval_tex_v22 load the production specialists exactly the
    same way, which is what keeps the G2 specialist-vs-unified comparison
    apples-to-apples. Inference is the higher-precision side of the pair."""
    from trellis2 import models as t2models
    from trellis2_blip3o.tr2_modules import DEFAULT_SHAPE_SLAT, DEFAULT_TEX_SLAT
    return (t2models.from_pretrained(DEFAULT_SHAPE_SLAT),
            t2models.from_pretrained(DEFAULT_TEX_SLAT))


def assemble_unified_inference(shape_run_ckpt: str, tex_run_ckpt: str,
                               coupling: str = "union",
                               bidirectional: bool = False) -> UnifiedGeoTexFlow:
    """Warm-start assembly for SAMPLING (fp32 latents): from_pretrained layout,
    run weights overlaid with EMA (load_state_dict casts into each param's
    existing dtype, mirroring the production eval loaders)."""
    geo, tex = _from_pretrained_pair()
    geo.load_state_dict(_load_prefixed_state(shape_run_ckpt, "shape_slat_512."), strict=True)
    tex.load_state_dict(_load_prefixed_state(tex_run_ckpt, "tex_slat_512."), strict=True)
    return UnifiedGeoTexFlow(geo, tex, coupling=coupling, bidirectional=bidirectional)


def load_unified_inference(unified_run_ckpt: str,
                           coupling: str = "union",
                           bidirectional: bool = False,
                           use_ema: bool = True) -> UnifiedGeoTexFlow:
    """Load a TRAINED geotex run checkpoint (train_stages="geotex" output:
    unified_geotex.* prefixes + EMA overlay on the trainables) for G1/G2 eval,
    in the inference dtype layout. bidirectional must match the run's flag
    (strict load catches a mismatch via the b_gates key).

    use_ema was not selectable before, and the overlay is not always the better
    weights: EMA written by a run SHORTER than its decay's time constant still
    carries the random init (0.9999 over 10k steps leaves 36.8% of it). Runs from
    after the decay warmup landed are safe; pass False to check an older one
    against its raw weights."""
    geo, tex = _from_pretrained_pair()
    uni = UnifiedGeoTexFlow(geo, tex, coupling=coupling, bidirectional=bidirectional)
    uni.load_state_dict(_load_prefixed_state(unified_run_ckpt, "unified_geotex.",
                                             use_ema=use_ema),
                        strict=True)
    return uni


def load_run_connectors(unified_run_ckpt: str):
    """(geo_connector, tex_connector) from a trained geotex run ckpt — prefixes
    geo_connector. / diffusion_connector. (EMA overlays the trained tex one)."""
    from trellis2_blip3o.connector import TRELLIS2Connector
    conns = []
    for prefix in ("geo_connector.", "diffusion_connector."):
        sd = _load_prefixed_state(unified_run_ckpt, prefix)
        cond_dim, vlm_dim = sd["fc1.weight"].shape
        conn = TRELLIS2Connector(vlm_hidden_dim=vlm_dim, trellis_cond_dim=cond_dim)
        conn.load_state_dict(sd, strict=True)
        conns.append(conn)
    return conns[0], conns[1]


def load_connectors(shape_run_ckpt: str, tex_run_ckpt: str,
                    weights_file: str = "model.safetensors"):
    """Two per-stream TRELLIS2Connectors; dims inferred from checkpoint tensor
    shapes (fc1.weight = (cond_dim, vlm_dim), connector.py:39-41); strict load."""
    from trellis2_blip3o.connector import TRELLIS2Connector

    conns = []
    for ck in (shape_run_ckpt, tex_run_ckpt):
        cfg = json.load(open(os.path.join(ck, "config.json")))
        assert cfg.get("cond_adapter") == "mlp", \
            f"{ck}: cond_adapter={cfg.get('cond_adapter')!r} — arch-flag mismatch"
        assert not cfg.get("cond_pos_stamp", False), f"{ck}: unexpected pos_stamp"
        sd = _load_prefixed_state(ck, "diffusion_connector.", weights_file)
        cond_dim, vlm_dim = sd["fc1.weight"].shape
        conn = TRELLIS2Connector(vlm_hidden_dim=vlm_dim, trellis_cond_dim=cond_dim)
        conn.load_state_dict(sd, strict=True)
        conns.append(conn)
    return conns[0], conns[1]
