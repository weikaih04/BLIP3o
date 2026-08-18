"""Three-stream sparse MMDiT for SLAT — FROM SCRATCH (no TRELLIS.2 weights).

PROVENANCE RULE: every module here is TRELLIS.2's own or one of our
G0-certified stitching helpers. Nothing is a module invented for this file.
Where a design CHOICE had to be made, the reference that made the same choice
is named inline.

  building block   ModulatedSparseTransformerBlock  TRELLIS.2 sparse/transformer/modulated.py:11
  timestep embed   TimestepEmbedder                 TRELLIS.2 models/sparse_structure_flow.py:12
  shared adaLN     nn.SiLU + Linear(C, 6C)          TRELLIS.2 models/structured_latent_flow.py:55-58
  weight init      initialize_weights               TRELLIS.2 models/structured_latent_flow.py:101-126
  final norm       F.layer_norm before out_layer    TRELLIS.2 models/structured_latent_flow.py:197
  mod unpack       _block_mod_params                ours, unified_geotex.py:75 (modulated.py:60)
  cond key mask    not packed, rather than masked    ours; the pipeline's own mask
                   ^ flow_heads.py:212-225 marks padding AND the dropped-DINO
                     curriculum (dino_drop_prob=0.3) in a key mask. A varlen model
                     should simply not pack those rows.
  qkv / out proj   _attn_qkv / _attn_out            ours, unified_geotex.py:80+ (G0-certified)
  norm + modulate  fused_norm_modulate              TRELLIS.2 sparse/fused_modulate.py:91
  gated residual   fused_gate_residual              TRELLIS.2 sparse/fused_modulate.py:147
  attention        sparse_scaled_dot_product_attn   TRELLIS.2 sparse/attention/full_attn.py
  rope             SparseRotaryPositionEmbedder     TRELLIS.2, config UNTOUCHED
                     (3 axes, 21 pairs each; freqs tensor bit-identical)
  segment tag      tag_rows                          ours, generalises
                     unified_geotex._rotate_pad_pair (G0-certified, trained
                     through S1/S2b) from one whole-tensor 90-degree turn to a
                     per-row choice of 0/90/180/270
  per-forward plan _seg_plan                        ours, mirrors unified_geotex._fusion_plan

ARCHITECTURE = FLUX / Hunyuan3D-2.1 / Modality Forcing:
    N double(here triple)-stream blocks -> merge -> 2N single-stream blocks.
  * Hunyuan3D-2.1 (hy3dshape/.../hunyuan3ddit.py): 16 double + 32 single; cond is
    a full stream and IS rewritten (`latent, cond = block(...)`); merge is
    `torch.cat((cond, latent), 1)`; cond dropped at the end. Zero cross-attn.
  * Modality Forcing (Duisterhof/modality-forcing, flux_rgbd/dit.py):
    8 triple -> `joint = torch.cat([txt, img, depth], 1)` (dit.py:444) -> 24
    single -> plus FOUR depth-only SingleStreamBlocks and a depth final layer
    (dit.py:227,329-337,344, run at :457-460). So depth is NOT weightless after
    the merge, as an earlier version of this note claimed: it shares the 24 with
    its own per-token modulation (`single_stream_modulation_depth`, dit.py:323)
    AND keeps a private 4-block decoder. We deliberately do NOT copy that
    decoder — our two generated streams already keep separate out layers, and a
    private tail would reintroduce the specialist asymmetry the merge is for.
  Ratio 1:2 by the owner's decision (2026-08-15); MF's shared stack is 1:3, and
  1:3+4 counting its decoder.

WHAT THEY DID NOT HAVE TO SOLVE:
  1. Latents are SPARSE VOXELS, variable count per asset, with real coords — not
     Hunyuan's fixed-length unordered vecset (which is why they run dense sdpa
     with `pe = None`). Everything here is SparseTensor + varlen flash, the same
     path SLAT blocks already use. "Sparse" in TRELLIS means the TOKENS are the
     active voxels; attention among them is `attn_mode="full"`, so joint
     attention = ONE full varlen call over the concatenated sequence.
  2. PER-MODALITY TIMESTEP. FLUX/Hunyuan broadcast one `vec`; our mode matrix
     (mesh-only / tex|mesh / joint) exists only because the generated streams
     carry INDEPENDENT timesteps. The shared blocks therefore share WEIGHTS but
     not MODULATION — exactly MF's `_stack_per_token_mod` (dit.py:180-201).
  3. GEO AND TEX SHARE COORDS EXACTLY. MF's depth is spatially aligned with RGB
     and it fixes that with a RoPE axis holding a modality id. Ours is the same
     idea for one pair instead of sixteen: TRELLIS ropes 63 of head_dim 128's 64
     pairs and leaves ONE unrotated, so a quarter-turn of that spare pair tags
     the segment while the spatial ladder keeps all 21 pairs per axis.
     An intermediate version DID give the segment its own 4th rope axis, the way
     MF does. Measured, that cost each spatial axis 5 of its 21 pairs (lowest
     frequency 1.55e-4 -> 1.78e-4) and bought a segment signal in which an id
     difference of 1 turned only 3 of that axis's 16 pairs by >0.3 rad — a
     certain loss for an ambiguous gain, on the axes that ARE the geometry.

SCALING: growing the model is `dim` / `num_heads` / `depth_double` /
`depth_single` and nothing else — the module construction, the block stacks, the
packing plan and the elastic-GC block count are all derived. `gen_channels` (an
ordered dict of generated stream -> LATENT channels) and `SEGMENTS` drive the
tower construction and the segment ids.

Adding a THIRD generated modality is NOT one line, and this docstring used to
claim it was. Two things fix the stream set at (geo, tex): `forward`'s signature
is `(x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond)` because it must stay
byte-identical to UnifiedGeoTexFlow's for flow_heads/the sampler to call it
unchanged, and `SEGMENTS` has room for exactly four ids (one rope pad pair = four
quarter-turns), of which cond already spends two. A third modality needs a wider
forward and a second tag pair.
"""
import math
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt  # torch.utils.checkpoint is not auto-imported
import torch.nn.functional as F

from . import _paths  # noqa: F401  — puts TRELLIS.2 on sys.path (unified_geotex.py:61)
from trellis2.modules import sparse as sp
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from trellis2.modules.sparse.transformer.modulated import ModulatedSparseTransformerBlock
from trellis2.models.sparse_structure_flow import TimestepEmbedder
from trellis2.modules.utils import convert_module_to, manual_cast

from .unified_geotex import (_attn_qkv, _attn_out, _block_mod_params,
                             fused_norm_modulate, fused_gate_residual)

N_SPATIAL_AXES = 3                      # SLAT voxels are (x, y, z)
N_ROPE_AXES = N_SPATIAL_AXES            # TRELLIS's rope, config untouched

# Segment ids are quarter-turns of the rope identity-pad pair (see `tag_rows`),
# NOT a rope axis. FOUR entries, not three: the cond sequence is [DINO ; Qwen]
# (eval_fusion_v22.py:69 builds it that way) and nothing else tells the model
# which part is which. Adding a modality = one more entry. Four is also the
# ceiling of this mechanism — one pair holds exactly four distinguishable turns;
# a fifth segment needs a second spare pair or a rope axis.
SEGMENTS: Dict[str, int] = {"geo": 0, "tex": 1, "cond_dino": 2, "cond_qwen": 3}
# tag_rows resolves turns modulo 4 through its where-chain, so a 5th id would
# silently alias onto id 3 — two streams with IDENTICAL q/k tags, indistinguishable
# in the attention logits, no error anywhere. Enforce the ceiling the mechanism has.
assert set(SEGMENTS.values()) <= {0, 1, 2, 3}, \
    "one rope pad pair holds exactly four turns; a 5th segment needs a second pair"


def tag_rows(t, turns: torch.Tensor):
    """SEGMENT TAG, applied PER ROW: rotate the rope identity-pad pair by
    `turns` quarter-turns. Generalises this repo's `_rotate_pad_pair`
    (unified_geotex.py:100, G0-certified bit-exact, trained through S1/S2b) from
    one whole-tensor 90-degree turn to a per-token choice of 0/90/180/270, which
    is what lets ONE cond stream carry two different segment ids (DINO, Qwen).
    Every case is a swap and/or a sign flip, so all four are float-EXACT.

    Chosen over giving the segment its own rope axis (Modality Forcing's time_id
    approach, which we did build and measured): a 4th axis costs each SPATIAL
    axis 5 of its 21 frequency pairs, and spatial resolution IS the geometry.
    TRELLIS's 3-axis ladder already leaves exactly one of the 64 pairs unrotated,
    so the tag is free. What the pair contributes to a cross-segment logit is
    cos(90 deg * delta_id): adjacent ids (geo-tex, dino-qwen) drop it to 0,
    ids two apart (geo-dino, tex-qwen) NEGATE it. Both are decisive marks; only
    delta_id = 0 leaves the pair untouched, which is the point."""
    f = t.feats
    x, y = f[..., -2], f[..., -1]
    k = turns.view(-1, *([1] * (x.dim() - 1)))
    xr = torch.where(k == 0, x, torch.where(k == 1, -y, torch.where(k == 2, -x, y)))
    yr = torch.where(k == 0, y, torch.where(k == 1, x, torch.where(k == 2, -y, -x)))
    return t.replace(torch.cat([f[..., :-2], xr[..., None], yr[..., None]], -1))


# ── per-forward segment plan ───────────────────────────────────────────────
# The streams' layouts do NOT change from block to block, so the packing plan is
# computed ONCE per forward and reused by every block — the same reason
# unified_geotex._fusion_plan is hoisted out of its block loop. Recomputing it
# per block (and once per q/k/v) meant 3 x depth python loops over the batch.
def _seg_plan(parts: List[sp.VarLenTensor]):
    B = len(parts[0].layout)
    # A stream with a different batch size would otherwise die as a bare
    # IndexError inside _seg_cat, pointing at a slice list instead of the cause.
    assert all(len(p.layout) == B for p in parts), \
        f"streams disagree on batch size: {[len(p.layout) for p in parts]}"
    spans = [[] for _ in parts]
    rows_per_sample, off = [], 0
    for b in range(B):
        tot = 0
        for i, p in enumerate(parts):
            sl = p.layout[b]
            n = sl.stop - sl.start
            spans[i].append(slice(off, off + n))
            off += n
            tot += n
        rows_per_sample.append(tot)
    return {"spans": spans, "seqlen": rows_per_sample, "total": off}


def _seg_cat(parts: List[sp.VarLenTensor], plan) -> sp.VarLenTensor:
    """Per-sample concatenation into ONE varlen sequence, so a single flash call
    IS the joint attention. Dense analogues: Hunyuan3D `q = torch.cat((txt_q,
    img_q), dim=2)` (hunyuan3ddit.py:205) and MF `q = torch.cat([q_txt, q_img,
    q_depth], dim=2)` (dit.py:139-142) — both concatenate Q/K/V, which is what
    this does. (MF's dit.py:444 concatenates HIDDEN STATES; that is the merge
    into the shared stack, a different operation, and here it is implicit —
    the streams simply start sharing weights.) Varlen because our segment
    lengths differ per asset AND per stream.

    torch.cat, NOT new_empty + slice assignment. The slice-assign version is one
    `CopySlices` autograd node PER SPAN, and each one clones the whole packed
    gradient in backward: measured 24 chained nodes and 2.6x the fwd+bwd cost of
    this version (86.4 s vs 33.1 s over 5 iterations at B=8, 3 streams, 4000
    rows/stream, H=6, D=128). cat is a single node."""
    # torch.cat TYPE-PROMOTES rather than raising (measured: bf16 + fp32 parts
    # concatenate silently to fp32), and it takes device/dtype from the promotion
    # rule, not from parts[0] — so the check has to be explicit. A stream
    # arriving in the wrong dtype means one of them skipped the manual_cast into
    # the torso dtype, which would be invisible in the loss.
    f0 = parts[0].feats
    assert all(p.feats.dtype == f0.dtype and p.feats.device == f0.device
               for p in parts), (
        "streams must share dtype/device before packing: "
        f"{[(str(p.feats.dtype), str(p.feats.device)) for p in parts]}")
    order = [p.feats[p.layout[b]] for b in range(len(parts[0].layout)) for p in parts]
    return sp.VarLenTensor(torch.cat(order, 0),
                           sp.VarLenTensor.layout_from_seqlen(plan["seqlen"]))


def _seg_split(packed: torch.Tensor, plan, parts: List[sp.VarLenTensor]):
    """Inverse of _seg_cat. Also torch.cat rather than slice-assignment, for the
    same backward-cost reason; `layout` always tiles [0, N) in ascending b, so
    concatenating the per-sample pieces in b order restores the original rows."""
    return [p.replace(torch.cat([packed[plan["spans"][i][b]]
                                 for b in range(len(p.layout))], 0))
            for i, p in enumerate(parts)]


def _make_block(dim: int, num_heads: int, mlp_ratio: float):
    """One TRELLIS.2 block, freshly initialised, with the segment-carrying rope."""
    head_dim = dim // num_heads
    assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
    # rope.py splits head_dim//2 pairs evenly across its axes and pads the
    # remainder with identity — a silent capacity loss if it does not divide.
    # rope.py:47-52 pads EVERY leftover pair with identity and tag_rows touches
    # only the last one, so any leftover >= 1 works: head_dim 128 leaves 1
    # (3x21+1), head_dim 64 leaves 2 (3x10+2). Zero leftover is the failure —
    # the tag would then overwrite a real spatial frequency.
    assert (head_dim // 2) % N_ROPE_AXES != 0, (
        f"head_dim {head_dim} leaves no spare rope pair ({head_dim // 2} pairs "
        f"divide evenly across {N_ROPE_AXES} axes); the segment tag needs one")
    return ModulatedSparseTransformerBlock(
        dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_mode="full",
        use_rope=True, qk_rms_norm=True, share_mod=True)


def _stream_pre(blk, h, mod, turns):
    """norm1 -> modulate -> qkv. Op-for-op the geo/tex lane of
    unified_geotex._run_block_pair_bidir, which G0 certified bit-exact against
    the block's own forward."""
    sh, sc, g, sh2, sc2, g2 = _block_mod_params(blk, mod)
    hn = fused_norm_modulate(h, blk.norm1, sc, sh)
    q, k, v = _attn_qkv(blk.attn, hn)
    return (tag_rows(q, turns), tag_rows(k, turns), v), (g, sh2, sc2, g2)


def _stream_post(blk, h, a, carry):
    """out proj -> gated residual -> norm2 -> mlp -> gated residual. Same source.
    norm2 IS the MLP norm here: the non-cross block is norm1->attn, norm2->mlp
    (modulated.py:58-69); only the cross variant spends norm2 on cross-attn."""
    g, sh2, sc2, g2 = carry
    h = fused_gate_residual(h, _attn_out(blk.attn, a), g)
    hm = fused_norm_modulate(h, blk.norm2, sc2, sh2)
    return fused_gate_residual(h, blk.mlp(hm), g2)


def _joint_attention(qkvs, plan):
    """Concatenate the streams' q/k/v per sample and run ONE varlen flash call —
    TRELLIS's own kernel, the one every SLAT block uses."""
    qs = [z[0] for z in qkvs]
    # Stack q/k/v PER STREAM first, then pack ONCE. Packing the three separately
    # and stacking after allocates four full (total, ...) buffers instead of one
    # — ~1.5 GB of transient per block at B=8 / 40k rows / H=6 / D=128 / bf16,
    # paid twice under checkpointing.
    packed = _seg_cat([q.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
                       for q, k, v in qkvs], plan)
    a = sparse_scaled_dot_product_attention(packed)
    return _seg_split(a.feats, plan, qs)


class _StreamTower(nn.Module):
    """One stream's own weights in the double-stream region, laid out the way
    TRELLIS's SLatFlowModel is (input_layer / t_embedder / adaLN_modulation /
    blocks / out_layer) so that PARAMETER NAMES match what the rest of the
    codebase keys off: train_native.py:643-651's freeze audit matches prefixes
    like `unified_geotex.tex_flow.`, and trellis_native_vlm.py:349,357 reaches
    for `.geo_flow` and `.tex_flow.blocks` directly. Structuring the model to
    satisfy that contract is safer than bolting compatibility attributes on.

    `in_channels` / `out_channels` mean what SLatFlowModel means by them — the
    LATENT widths (structured_latent_flow.py:39-40). They are NOT the input
    projection's width, which for tex also carries the concatenated shape latent
    (config slat_flow_imgshape2tex_dit_1_3B_512_bf16.json: in 64 = tex 32 +
    shape 32, out 32). geotex_sampler.py:61,84,140 draws its initial noise from
    `geo_flow.in_channels`, so getting this wrong is a silent width error."""

    def __init__(self, latent_ch, in_ch, out_ch, dim, num_heads, mlp_ratio, depth):
        super().__init__()
        self.in_channels = latent_ch
        self.out_channels = out_ch
        self.input_layer = sp.SparseLinear(in_ch, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.blocks = nn.ModuleList(
            [_make_block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.out_layer = sp.SparseLinear(dim, out_ch) if out_ch else None
        # Not a submodule (a single-element list keeps nn.Module from
        # registering it, so it stays out of the state_dict) — a back-reference
        # to the owning MMDiT3D, for code that has a tower and needs the model.
        self._owner: list = []

    def forward(self, x, t, cond, *a, **kw):
        """A tower is WEIGHTS, not a model: in a joint MMDiT there is no
        standalone per-stream velocity function, because the stream's value
        depends on the other streams through every block's joint attention.

        This raises rather than quietly standing in for one. The natural
        stand-in — the MARGINAL, this stream with the others pinned at their t=1
        noise corner — needs noise the CALLER controls: CFG evaluates the model
        twice per step, and drawing fresh noise inside each call makes
        `cfg*v_pos + (1-cfg)*v_neg` amplify a noise difference instead of
        isolating the conditioning difference (measured: two identical calls
        differed by 0.11 max-abs). geotex_sampler.sample_mesh_only_marginal
        draws it once per step from its seeded generator and passes the SAME
        tensor to both branches, which is why it is the entry point to use."""
        raise NotImplementedError(
            "a joint MMDiT has no standalone per-stream forward; use "
            "GeoTexSampler.sample_mesh_only_marginal (it owns the other "
            "stream's noise, which CFG requires to be shared between the "
            "conditional and unconditional calls)")


def _run_joint(blks, plan, hs, mods, turns, use_ckpt: bool):
    """One joint block: every stream's q/k/v, ONE varlen flash, then each stream
    advances with ITS OWN (or the shared) weights. `blks` is three blocks in the
    double region and the same block three times in the shared region — that is
    the ONLY difference between the two regions."""
    def inner(*args):
        n = len(blks)
        h, m, t = list(args[:n]), list(args[n:2 * n]), list(args[2 * n:])
        pre = [_stream_pre(b, hh, mm, tt) for b, hh, mm, tt in zip(blks, h, m, t)]
        att = _joint_attention([q[0] for q in pre], plan)
        return [_stream_post(b, hh, a, q[1]) for b, hh, a, q in zip(blks, h, att, pre)]

    if use_ckpt and torch.is_grad_enabled():
        return ckpt.checkpoint(inner, *hs, *mods, *turns, use_reentrant=False)
    return inner(*hs, *mods, *turns)


class MMDiT3D(nn.Module):
    """Three-stream sparse MMDiT, structured to satisfy the SAME contract as
    UnifiedGeoTexFlow so the whole training and eval path is reused unchanged."""

    def __init__(self, dim: int = 768, num_heads: int = 6,
                 depth_double: int = 8, depth_single: int = 16,
                 gen_channels: Optional[Dict[str, int]] = None,
                 cond_ch: int = 1024, mlp_ratio: float = 5.375,
                 concat_cond: bool = False, pooled_cond: bool = True,
                 use_checkpoint: bool = True,
                 initialization: str = "scaled", dtype: str = "float32"):
        # mlp_ratio 5.375, NOT the released 5.3334. The released number is not a
        # ratio anyone chose — it is 16/3, picked so that the official width
        # 1536 * 16/3 lands exactly on 8192. At our dim=1024 the same ratio
        # gives hidden 5461 = 43*127, which no tensor-core tile divides:
        # measured 2.619 s/step against 1.600 for mlp 4.0 (hidden 4096).
        # Rounding up to 5.375 -> hidden 5504 (64*86) costs 5M MORE parameters
        # and runs at 1.670 s/step, i.e. 57% faster than the "official" ratio.
        # What to copy from the release is an ALIGNED hidden width, not 16/3.
        # initialization="scaled" is the released value; we shipped 4.0 +
        # vanilla xavier by mistake, audited 2026-08-17.
        super().__init__()
        # LATENT widths, both 32. The tex specialist's config says in 64 / out 32
        # because its 64 is tex(32) + the concatenated shape latent(32) — that
        # concat is `extra` below, not part of the latent.
        gen_channels = gen_channels or {"geo": 32, "tex": 32}
        assert all(n in SEGMENTS for n in gen_channels), \
            f"every generated stream needs a segment id: {set(gen_channels) - set(SEGMENTS)}"
        self.gen_names = list(gen_channels)
        self.gen_channels = dict(gen_channels)
        self.stream_names = self.gen_names + ["cond"]
        self.concat_cond = concat_cond
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self._memory_controller = None
        # accepted and ignored: trellis_native_vlm sets it on the unified model,
        # but this architecture has a single joint-attention path, no variant.
        self.fused_attn = True

        # concat_cond: TRELLIS.2 feeds the tex flow the shape latent per VOXEL.
        # That is the CASCADE's mechanism — there the shape is an external input
        # and attention to it does not exist, so the concat is the only path.
        #
        # DEFAULT FLIPPED TO FALSE 2026-08-17. In a joint model the geo stream
        # sits in the same attention, so the concat is a SECOND copy of
        # information the tex stream can already reach. Measured on
        # checkpoint-60000 at t_s=0 (clean GT geometry) / t_x=1: zeroing the
        # concat costs the tex prediction 2.0%, while removing the geo stream's
        # tokens costs 7.3% — attention is doing the work, the concat is not.
        #
        # It is also the ONLY structural asymmetry between the two generated
        # streams (tex input 64 = 32+32, geo input 32 with no second path), and
        # the geo stream is the one that failed to learn to read its
        # conditioning. Keeping the streams symmetric removes that confound.
        # Set concat_cond=True to restore the cascade behaviour as an A/B arm.
        extra = {n: 0 for n in self.gen_names}
        if concat_cond and {"geo", "tex"} <= set(self.gen_names):
            extra["tex"] = self.gen_channels["geo"]
        for n, c in self.gen_channels.items():
            setattr(self, f"{n}_flow", _StreamTower(c, c + extra[n], c, dim, num_heads,
                                                    mlp_ratio, depth_double))
        self.cond_flow = _StreamTower(cond_ch, cond_ch, None, dim, num_heads,
                                      mlp_ratio, depth_double)
        # cond has no timestep of its own; all three references feed the
        # condition stream the DIFFUSED stream's timestep through its own
        # projection (MF dit.py:410, Hunyuan hunyuan3ddit.py:190-191, FLUX.2
        # _flux2/model.py:140-147). We have two diffused streams, so cond takes
        # the SUM of their embeddings — informed in every mode, with no
        # arbitrary choice between them. Two SEPARATE embedders keep the sum
        # asymmetric, so (t_s, t_x) and (t_x, t_s) do not collide.
        self.cond_flow.t_embedder = None
        self.cond_t = nn.ModuleDict({n: TimestepEmbedder(dim) for n in self.gen_names})
        # MMDiT's SECOND conditioning path, which this model was missing.
        #
        # Condition tokens reach a voxel only as extra attention keys, and that
        # whole attention output is scaled by gate_msa = modulation + adaLN(t).
        # adaLN is zero-initialised and modulation is randn(6C)/sqrt(C), so at
        # dim=1024 the gate opens at |g| ~ 0.026: the image arrives at ~1/39 of
        # full strength and has to push the gate open before it means anything.
        # Every shipped comparable model gives conditioning a route that the
        # gate cannot close — TRELLIS.2 and the released Hunyuan3D-2.1 through
        # an ungated cross-attention residual, SD3 and FLUX by adding a POOLED
        # condition vector to the timestep embedding, which is what produces
        # shift/scale/gate in the first place. Only the second one is available
        # to a single-softmax MMDiT, so it is what we copy, from
        # diffusers/models/embeddings.py:1601-1609 (SD3) and :1632-1633 (FLUX):
        #
        #     text_embedder = PixArtAlphaTextProjection(pooled_dim, dim, "silu")
        #                   = Linear -> SiLU -> Linear      (:2213-2222)
        #     conditioning  = timesteps_emb + text_embedder(pooled)
        #
        # Their pooled vector is CLIP's pooled output; ours is the mean over the
        # condition tokens that survived the drop curriculum, which is defined
        # in every mode (DINO-dropped, Qwen-dropped, and the CFG-uncond branch)
        # where a fixed slot such as DINO's CLS would not be.
        self.pooled_cond = pooled_cond
        self.cond_pool = nn.Sequential(
            nn.Linear(cond_ch, dim), nn.SiLU(), nn.Linear(dim, dim),
        ) if pooled_cond else None
        self.shared_blocks = nn.ModuleList(
            [_make_block(dim, num_heads, mlp_ratio) for _ in range(depth_single)])
        for tw in self._towers():
            tw._owner.append(self)
        # Read by the entry points to skip warm-start-only steps: freezing
        # geo_flow (trellis_native_vlm.py:349 — for a pretrained specialist that
        # is the S1 recipe; here it would pin a RANDOM geo forever, and
        # unfreeze_geo() cannot undo it because nothing ever pretrained it) and
        # the freeze audit's allow-list (train_native.py:643).
        self.from_scratch = True
        self.initialization = initialization
        # Body dtype, SLatFlowModel's protocol verbatim (structured_latent_flow
        # .py:51,94-99,183-196): input/out layers stay in the latent's dtype, the
        # TORSO is converted, and manual_cast bridges the two. Training never
        # needs it (autocast covers everything), but the sampler runs outside
        # autocast on fp32 latents — without this, flash attention gets fp32 and
        # raises "FlashAttention only support fp16 and bf16 data type".
        # SLatFlowModel takes dtype from its config and self-converts at the end
        # of __init__ (structured_latent_flow.py:51,85). The two official configs
        # for the SAME class differ here: the TRAINING config omits dtype
        # (-> float32 torso, amp bf16 does the rest) while the RELEASED
        # checkpoint config says "bfloat16". Mirror that contract so an inference
        # entry point cannot silently end up with an fp32 torso just because it
        # forgot to call convert_to.
        self.dtype = torch.float32
        (self.initialize_weights_scaled if initialization == "scaled"
         else self.initialize_weights)()
        if dtype != "float32":
            self.convert_to(getattr(torch, dtype))

    def convert_to(self, dtype: torch.dtype) -> None:
        """structured_latent_flow.py:94-99 — torso only. Here the torso is every
        joint block: the towers' own blocks AND the shared stack."""
        self.dtype = dtype
        for tw in self._towers():
            tw.blocks.apply(partial(convert_module_to, dtype=dtype))
        self.shared_blocks.apply(partial(convert_module_to, dtype=dtype))

    # ── the three towers, in the order they are packed ─────────────────────
    def _towers(self):
        return [getattr(self, f"{n}_flow") for n in self.gen_names] + [self.cond_flow]

    def initialize_weights_scaled(self) -> None:
        """structured_latent_flow.py:128-167 — the `"scaled"` branch, which is
        what the released 512 shape/tex configs actually set
        (slat_flow_img2shape_dit_1_3B_512_bf16.json: "initialization": "scaled").
        We shipped the `vanilla` branch by mistake; audited 2026-08-17.

        Three things vanilla does NOT do, all of which matter at depth 30:
          * bodies at std sqrt(2/(5C)) rather than xavier;
          * DEPTH-SCALED residual exits — attn.to_out and mlp[2] at
            std 1/sqrt(5 * num_blocks * C), so the residual stream does not grow
            with depth;
          * input_layer at std 1/sqrt(in_channels) so the first representation
            has unit variance.
        num_blocks here is the depth a token actually traverses (double +
        shared), matching what the official count means for its stack."""
        n_blocks = len(self.shared_blocks) + len(
            getattr(self, f"{self.gen_names[0]}_flow").blocks)
        C = self.dim

        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / (5.0 * C)))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)

        def _scaled(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(5 * n_blocks * C))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for blk in list(self.shared_blocks) + [b for tw in self._towers() for b in tw.blocks]:
            blk.attn.to_out.apply(_scaled)     # no cross_attn in this architecture
            blk.mlp.mlp[2].apply(_scaled)

        for tw in self._towers():
            nn.init.normal_(tw.input_layer.weight,
                            std=1.0 / math.sqrt(tw.input_layer.in_features))
            nn.init.zeros_(tw.input_layer.bias)
            if tw.t_embedder is not None:
                nn.init.normal_(tw.t_embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(tw.t_embedder.mlp[2].weight, std=0.02)
            nn.init.constant_(tw.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(tw.adaLN_modulation[-1].bias, 0)
            if tw.out_layer is not None:
                nn.init.constant_(tw.out_layer.weight, 0)
                nn.init.constant_(tw.out_layer.bias, 0)
        for te in self.cond_t.values():
            nn.init.normal_(te.mlp[0].weight, std=0.02)
            nn.init.normal_(te.mlp[2].weight, std=0.02)
        if self.cond_pool is not None:
            # It is summed with t_embedder's output, so it gets t_embedder's
            # scale (structured_latent_flow.py:153-154) rather than the body's.
            for lin in (self.cond_pool[0], self.cond_pool[2]):
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)

    def initialize_weights(self) -> None:
        """structured_latent_flow.py:101-126: xavier bodies (:106), timestep
        embedders at std 0.02 (:112-113), adaLN zeroed (:117-118), out layers
        zeroed (:125-126) so the model starts predicting zero velocity.

        Zeroing adaLN does NOT make the blocks identity, despite what an earlier
        version of this note said: with share_mod=True each block adds its own
        `modulation = randn(6C)/sqrt(C)` before the chunk (modulated.py:56,60),
        so scales and gates start at that random vector. TRELLIS's own SLAT flow
        has exactly this property; we keep it rather than zero `blk.modulation`,
        which would be a deviation. What the zero-init out_layer buys is the
        thing that actually matters — the initial velocity is exactly 0."""
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)
        for tw in self._towers():
            if tw.t_embedder is not None:
                nn.init.normal_(tw.t_embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(tw.t_embedder.mlp[2].weight, std=0.02)
            nn.init.constant_(tw.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(tw.adaLN_modulation[-1].bias, 0)
            if tw.out_layer is not None:
                nn.init.constant_(tw.out_layer.weight, 0)
                nn.init.constant_(tw.out_layer.bias, 0)
        for te in self.cond_t.values():          # cond's embedders too
            nn.init.normal_(te.mlp[0].weight, std=0.02)
            nn.init.normal_(te.mlp[2].weight, std=0.02)
        if self.cond_pool is not None:            # summed with them, same scale
            for lin in (self.cond_pool[0], self.cond_pool[2]):
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)

    # ── contract: elastic activation checkpointing ─────────────────────────
    def register_memory_controller(self, controller):
        self._memory_controller = controller
        return self

    def _get_input_size(self, x_s, *args, **kwargs):
        return x_s.feats.shape[0]

    @contextmanager
    def _with_mem_ratio(self, mem_ratio: float = 1.0):
        """Block-selection formula verbatim from sparse_elastic_mixin.py:17-23,
        over the FULL depth (double + shared) since every block here is joint."""
        n = len(self.shared_blocks) + len(getattr(self, f"{self.gen_names[0]}_flow").blocks)
        # try/finally, not a bare trailing reset: one OOM inside the forward
        # would otherwise leave _ckpt_upto pinned at that step's count for every
        # later forward, silently changing the memory/recompute trade.
        try:
            if mem_ratio >= 1.0:
                self._ckpt_upto = 0
                yield 1.0
            else:
                n_ck = min(math.ceil((1 - mem_ratio) * n) + 1, n)
                self._ckpt_upto = n_ck
                yield 1 - (n_ck - 1) / n
        finally:
            # 0, not None. TRELLIS restores to use_checkpoint=False after the
            # context (sparse_elastic_mixin.py:23-24) and so does this repo's
            # own unified_geotex.py:787-788; falling back to self.use_checkpoint
            # (default True) would make the next forward outside the context
            # checkpoint EVERY block, which is not what either reference does.
            self._ckpt_upto = 0

    def _ck(self, i):
        upto = getattr(self, "_ckpt_upto", None)
        return self.use_checkpoint if upto is None else (i < upto)

    # ── contract: stage switches ───────────────────────────────────────────
    def unfreeze_geo(self):
        """Nothing is frozen from scratch, but this must genuinely re-enable
        rather than no-op: trellis_native_vlm.py:349 freezes `geo_flow` on the
        way in (right for a pretrained specialist), and if that line is ever
        reached for this model a silent no-op here would leave geo pinned at
        random init with a zero-init out_layer — v_s == 0 forever, and the S2b
        geo/distill terms constant with no gradient. The entry point also skips
        line 349 via `from_scratch`; this is the second line of defence."""
        for n in self.gen_names:
            getattr(self, f"{n}_flow").requires_grad_(True)
        return self

    # ── contract: forward, byte-for-byte the same signature as
    #    UnifiedGeoTexFlow.forward, so flow_heads calls it unchanged ─────────
    def forward(self, x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=None):
        ctrl = self._memory_controller
        if ctrl is None or not torch.is_grad_enabled() or not self.training:
            return self._forward_impl(x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond)
        n = self._get_input_size(x_s)
        with self._with_mem_ratio(ctrl.get_mem_ratio(n)) as exact:
            out = self._forward_impl(x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond)
        ctrl.update_run_states(n, exact)
        return out

    @staticmethod
    def _cond_list(cond_x):
        """cond as a per-sample list of (T_b, cond_ch), whatever it arrived as."""
        if torch.is_tensor(cond_x):
            return list(cond_x.unbind(0))
        if isinstance(cond_x, sp.VarLenTensor):
            return [cond_x.feats[sl] for sl in cond_x.layout]
        return cond_x

    def _pooled_cond(self, cond_list):
        """SD3's `text_embedder(pooled_projection)` — embeddings.py:1607.

        Mean over each sample's surviving condition rows. flow_heads has already
        removed padding and curriculum-dropped tokens (_masked_list), so this is
        a plain mean, and it stays consistent between training and sampling
        because both pool the SAME tensor the attention keys come from — the CFG
        uncond branch included, whose pooled vector is simply the pooled
        [zeros(DINO); connector(0)] it is built from."""
        pooled = torch.stack([c.mean(0) for c in cond_list])       # (B, cond_ch)
        return self.cond_pool(pooled.to(self.cond_pool[0].weight.dtype))

    def _pack_cond(self, cond_x):
        """cond must be a SparseTensor, not a VarLenTensor: TRELLIS's fused
        modulate/gate kernels index a spatial cache only SparseTensor carries,
        and the point of this design is that cond takes the SAME path as the
        voxel streams. Spatial coords are all zero -> phase 0 -> identity
        rotation, i.e. exactly the "no positional encoding for cond" that
        TRELLIS's cross-attn uses (attention/modules.py:127-138 ropes nothing in
        the cross branch). The commonly-cited FLUX.1 precedent — zeroed txt_ids —
        is NOT verifiable on this box: no FLUX.1 source is checked out here, only
        FLUX.2 via modality-forcing/flux_rgbd/_flux2, which never builds ids at
        all. Treat it as hearsay, not as a citation; segment identity
        rides on the pad-pair turn instead.

        flow_heads has ALREADY dropped padded and curriculum-dropped tokens
        (`_masked_list`, flow_heads.py:575) and hands us a per-sample list, so
        there is no mask to apply here."""
        cond_x = self._cond_list(cond_x)
        rows, coords = [], []
        dt = self.cond_flow.input_layer.weight.dtype
        for b, c in enumerate(cond_x):
            # A sample with no surviving cond token would contribute no rows, so
            # SparseTensor's batch size would come out < B and every later
            # per-sample index would be off by one — an IndexError deep in
            # _seg_plan at best, a shifted batch at worst. Fail here instead.
            assert c.shape[0] > 0, (
                f"sample {b} has zero cond tokens; the joint sequence needs at "
                "least one row per sample per stream")
            # The sampler runs OUTSIDE autocast on fp32 latents
            # (geotex_sampler.py:110,114) while cond may arrive bf16 — the old
            # model normalised this with manual_cast (unified_geotex.py:868).
            rows.append(c.to(dt))
            coords.append(torch.cat([
                torch.full((c.shape[0], 1), b, dtype=torch.int32, device=c.device),
                torch.zeros((c.shape[0], N_SPATIAL_AXES), dtype=torch.int32,
                            device=c.device)], -1))
        return sp.SparseTensor(torch.cat(rows, 0), torch.cat(coords, 0))

    def _prepare(self, x_s, x_x, t_s, t_x, cond_x, tex_concat_cond):
        """Inputs -> (hiddens, modulations, segment turns) for the three streams.
        `cond_s` is unused: the old model gave each voxel stream its own
        connector, this one has a single cond stream."""
        x = {"geo": x_s, "tex": x_x}
        t = {"geo": t_s, "tex": t_x}
        hs, mods, turns = [], [], []
        cond_list = self._cond_list(cond_x)
        # SD3 feeds ONE `conditioning` vector to every stream's norm1 (its
        # norm1 and norm1_context take the same temb, attention.py:203-211), so
        # the pooled term is computed once and added to all three here.
        vec = self._pooled_cond(cond_list) if self.cond_pool is not None else None
        for n in self.gen_names:
            tw, xn = getattr(self, f"{n}_flow"), x[n]
            if self.concat_cond and n == "tex" and "geo" in self.gen_names:
                assert tex_concat_cond is not None, "concat_cond=True needs tex_concat_cond"
                # POSITIONAL concat: row i of the tex latent must be the same
                # voxel as row i of the shape latent. A mismatch is silent —
                # texture would simply be learned against the wrong geometry.
                assert torch.equal(tex_concat_cond.coords, xn.coords), \
                    "tex_concat_cond must sit on exactly the tex stream's voxels"
                xn = xn.replace(torch.cat([xn.feats, tex_concat_cond.feats], -1))
            # manual_cast into the torso dtype right after the input layer and
            # the adaLN, exactly where SLatFlowModel does it (:183, :187).
            h = manual_cast(tw.input_layer(xn), self.dtype)
            hs.append(h)
            emb = tw.t_embedder(t[n])
            if vec is not None:
                emb = emb + vec.to(emb.dtype)      # embeddings.py:1609
            mods.append(manual_cast(tw.adaLN_modulation(emb), self.dtype))
            turns.append(torch.full((h.feats.shape[0],), SEGMENTS[n],
                                    dtype=torch.int32, device=h.feats.device))
        hc = self._pack_cond(cond_list)
        # SparseLinear takes the SparseTensor, not .feats
        hs.append(manual_cast(self.cond_flow.input_layer(hc), self.dtype))
        cemb = sum(self.cond_t[n](t[n]) for n in self.gen_names)
        if vec is not None:
            cemb = cemb + vec.to(cemb.dtype)
        mods.append(manual_cast(self.cond_flow.adaLN_modulation(cemb), self.dtype))
        turns.append(self._cond_turns(hs[-1]))
        return hs, mods, turns

    def _cond_turns(self, hc):
        """Cond's two sources are concatenated as [DINO ; Qwen] upstream, but the
        boundary moves per sample once the drop curriculum has removed tokens.
        Without it every cond token takes a single id — the SAFE default: one
        undivided condition stream, which is strictly less information, never
        wrong information. (It is NOT "what MF does": MF leaves text on the same
        modality id as RGB, 0.0, and separates it by a per-token arange on
        another axis instead — flux_rgbd/model.py:35-45. We have no free axis for
        an arange, which is exactly why the split is a discrete id here.)"""
        lens = getattr(self, "_dino_lengths", None)
        t = torch.full((hc.feats.shape[0],), SEGMENTS["cond_qwen"],
                       dtype=torch.int32, device=hc.feats.device)
        if lens is not None:
            # A stale count from a previous batch would tag the wrong rows
            # silently, so the batch size must match exactly.
            assert len(lens) == len(hc.layout), (
                f"dino lengths are for {len(lens)} samples, this batch has "
                f"{len(hc.layout)} — set_cond_dino_lengths must be called per batch")
            for b, sl in enumerate(hc.layout):
                # The count must be the SURVIVING one, after the drop curriculum.
                # A nominal count would run past this sample's span and tag the
                # NEXT sample's leading rows as DINO — silently, since the rows
                # are contiguous across samples in the packed buffer.
                n = int(lens[b])
                assert 0 <= n <= sl.stop - sl.start, (
                    f"sample {b}: {n} dino tokens claimed but the sample has only "
                    f"{sl.stop - sl.start} cond rows (pass the post-drop count)")
                t[sl.start:sl.start + n] = SEGMENTS["cond_dino"]
        return t

    def set_cond_dino_lengths(self, lengths):
        """Per-sample count of surviving DINO tokens (they lead the cond
        sequence), valid for the NEXT forward only.

        NOTHING CALLS THIS YET, so `SEGMENTS["cond_dino"]` is currently unused
        and the cond stream carries one id — which is what Modality Forcing's
        text stream effectively does (its modality axis stays 0, the same value
        RGB uses; MF-model:35-45). Wiring it needs flow_heads to pass down the
        per-sample surviving-DINO count that `_masked_list` already computes."""
        self._dino_lengths = lengths

    def _run_stack(self, hs, mods, turns, plan):
        towers = self._towers()
        depth_d = len(towers[0].blocks)
        for i in range(depth_d):
            hs = _run_joint([tw.blocks[i] for tw in towers], plan, hs, mods, turns,
                            self._ck(i))
        for j, blk in enumerate(self.shared_blocks):
            hs = _run_joint([blk] * len(towers), plan, hs, mods, turns,
                            self._ck(depth_d + j))
        return hs

    def _forward_impl(self, x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=None):
        hs, mods, turns = self._prepare(x_s, x_x, t_s, t_x, cond_x, tex_concat_cond)
        plan = _seg_plan(hs)          # layouts are block-invariant: build once
        hs = self._run_stack(hs, mods, turns, plan)
        out = []
        latent_dtype = {"geo": x_s.feats.dtype, "tex": x_x.feats.dtype}
        for i, n in enumerate(self.gen_names):
            tw = getattr(self, f"{n}_flow")
            # back out of the torso dtype, then the final LayerNorm before the
            # output projection — structured_latent_flow.py:196-198, same order.
            h = manual_cast(hs[i], latent_dtype[n])
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            out.append(tw.out_layer(h))
        return tuple(out)

    # ── contract: the sampler's two entry points (geotex_sampler.py:112,120) ─
    # There is NO in-DiT K/V cache in this architecture, by decision: the only
    # cache this project keeps is the VLM/DINO one on disk, which lives before
    # the DiT and has no timestep dependence. So `precompute_geo_kv` returns the
    # geo INPUTS rather than per-block keys, and `tex_forward_cached` runs the
    # full joint forward and returns the tex velocity. The sampler needs no
    # change; it simply pays a geo pass per step, which is the trade already
    # accepted when the in-DiT cache was dropped.
    def precompute_geo_kv(self, x_s, t_s, cond_s, want_v: bool = False):
        # unified_geotex.py:954-957 returns the real geo velocity for want_v;
        # there is no free geo velocity here (no per-block cache to read it out
        # of), and returning None would have a migrating caller integrate None.
        assert not want_v, (
            "want_v has no cheap answer in a joint MMDiT — call the model and "
            "take v_s, or use GeoTexSampler.sample_mesh_only_marginal")
        return {"x_s": x_s, "t_s": t_s, "cond_s": cond_s}, None

    def tex_forward_cached(self, x_x, t_x, t_s, cond_x, tex_concat_cond, kv):
        _, v_x = self._forward_impl(kv["x_s"], x_x, kv["t_s"], t_x,
                                    kv["cond_s"], cond_x, tex_concat_cond)
        return v_x
