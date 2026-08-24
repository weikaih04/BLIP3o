"""TrellisNativeVLM — native-VLM encoder → TRELLIS.2 cascade (continuous, no-discrete).

See QWEN35_VLM_DESIGN.md. Pipeline:

    text (+ optional image / multi-image / video)
        → native VLM (Qwen3.5-2B; vision baked in)               [encode_cond]
        → hidden_states[-1]  (2048-d, full sequence)
        → TRELLIS2Connector (2048 → 1024, UNCHANGED; DINOv3 dist-match)
        → cross-attn into TRELLIS SS / Shape-SLAT / Tex-SLAT flows  (weights reused)
        → SC-VAE decode  (inference only)
    Loss = 3-stage flow MSE (NO CE, NO discrete <I*> codebook).

Design choices (vs blip3oQwenForCausalLM):
  * Composition (HAS-A VLM), NOT inheritance from blip3oMeta (tangled w/ TA-Tok/codebook).
  * VLM is a frozen-or-trained *encoder* — no AR codebook generation, no CE.
  * Option α (keep TRELLIS cross-attn) — NOT a Qwen-Image-Edit MMDiT clone.

# DEPRECATED (2026-05-28): The code is **backbone-agnostic by design** (any HF
# `AutoModelForImageTextToText` with hidden_size=2048 plugs in), but the only
# TESTED / SUPPORTED backbone going forward is Qwen3.5-2B. Qwen3-VL-2B-Instruct
# and Qwen2.5-VL-3B-Instruct were earlier A/B candidates and may still load,
# but they are not a validated training path.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import trellis2_blip3o._paths  # noqa: F401  — sets sys.path so trellis2 imports
from trellis2_blip3o.connector import TRELLIS2Connector
from trellis2_blip3o import flow_heads
from blip3o.model.multimodal_decoder.builder import (
    build_ss_flow,
    build_shape_slat_512,
    build_tex_slat_512,
    build_shape_slat_1024,
    build_tex_slat_1024,
    build_trellis_decoders,
)
from blip3o.utils import rank0_print

# Cross-attention dim of the TRELLIS flow DiTs (cond_channels=1024 across 4B stages).
TRELLIS_COND_DIM = 1024


class TrellisNativeVLMConfig(PretrainedConfig):
    model_type = "trellis_native_vlm"

    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen3.5-2B",
        vlm_hidden_size: int = 2048,          # Qwen3.5-2B = 2048
        freeze_vlm: bool = True,              # v1 smoke-test: frozen (matches Qwen-Image-Edit)
        build_slat: bool = True,              # False ⇒ SS-only (skip shape/tex flows + decoders)
        slat_resolution: int = 512,           # 512 (paired w/ shape/tex_512 latents) or 1024 (HR cascade)
        detach_cond: bool = False,            # frozen ⇒ detach moot; kept for parity
        cond_max_length: int = 8192,
        mask_drop_prob: float = 0.1,
        anchor_drop_prob: float = 0.0,        # v2: prob to drop ANCHOR-only (keep Qwen) on image tasks → forces Qwen learning (text→3D)
        cfg_joint_drop_prob: float = 0.0,     # v2: prob to JOINT-drop anchor+Qwen (clean uncond) → TRELLIS-aligned CFG (image seed-stability)
        cond_fusion: str = "none",            # "none" (hidden[-1]) | "penultimate" ([-2]) | "depthwise"
        fusion_layers: int = 0,               # depthwise: # of VLM layer outputs to fuse (0 = all)
        flow_weight: float = 1.0,
        flow_stage_weights: str = "ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0",
        # DINOv3 conditioning alignment (REPA-inspired; see DINO_ALIGNMENT_DESIGN.md).
        # OFF by default → does not affect existing runs. Image tasks only.
        dino_align: bool = False,
        dino_align_weight: float = 0.5,         # λ; REPA default, robust 0.25–1.0
        dino_align_mode: str = "spatial",       # "spatial" (patch-wise) | "pooled"
        dino_image_token_id: int = 248056,      # Qwen3.5 <|image_pad|>
        dino_merge_size: int = 2,               # Qwen3.5 vision merge_size
        # Dual-branch conditioning (Know3D-style additive; see DUAL_COND_DESIGN.md). When ON,
        # the original TRELLIS DINOv3 cross-attn is kept as a geometry anchor (fed real DINOv3
        # features) and a parallel zero-init-gated Qwen cross-attn is added. Replaces the
        # dino_align crutch. OFF by default → existing runs unaffected.
        dual_cond: bool = False,
        dual_cond_max_views: int = 8,           # per-view embedding table size (multi-image)
        dual_slat_qwen_stride: int = 2,         # Qwen branch on every-Nth SLAT block (2=half, fits 512)
        dual_qwen_last_frac: float = 0.0,       # >0: inject Qwen ONLY on last frac of blocks (0.2=last20%); overrides stride, fits 16-GPU
        dual_anchor_pool: int = 1,              # fixed avg-pool DINOv3 anchor grid by N (2=32×32→16×16)
        dual_anchor_token_budget: int = 0,      # >0: adaptive — cap TOTAL anchor tokens (bounds worst-case)
        dual_ss_checkpoint: bool = False,       # gradient-checkpoint the WHOLE SS block (not just dual) — SS isn't covered by elastic SLAT GC; needs compile OFF
        # V3 condition-swap distillation (docs/V3_DISTILL_DESIGN.md). Teacher = the SAME
        # frozen flow fed single-view DINOv3 cond on the SAME (x_t, t); student =
        # connector(Qwen). Adds kd_v (velocity MSE) + kd_f (block-feature relative MSE).
        # Stage-1 (I1-only, connector-only) use; OFF by default → existing runs unaffected.
        distill_dino: bool = False,
        distill_v_weight: float = 1.0,
        distill_f_weight: float = 0.5,
        distill_f_blocks: str = "auto5",        # "autoK" = K evenly-spaced inner blocks | "3,9,15"
        # Stage-1.5 CFG-AWARE KD (fixes the ×s amplification 'sand'; see loss.py docstring).
        # distill_cfg_hi>0 enables: kd_v matches the GUIDED velocity v_u+s(v_c−v_u),
        # s ~ U[lo, hi] per step. null_grad=False = lite (student null pass no_grad).
        distill_cfg_lo: float = 3.0,
        distill_cfg_hi: float = 0.0,
        distill_cfg_null_grad: bool = False,
        # Vision-token contract: views are upscaled so each yields this many Qwen vision
        # tokens (1024 = 32×32 grid = DINOv3-teacher parity). Stored IN the checkpoint so
        # inference reads it from config and cannot silently mismatch training (the
        # collator implements the resize; this field is the source of truth). 0 = native.
        target_tokens_per_view: int = 0,
        # VLM-hidden cache mode (vlm_cache.py): False skips loading the 2B VLM entirely —
        # conds must arrive via forward(cond_hidden=...) from the precomputed cache.
        build_vlm: bool = True,
        # ── fusion: cond = [raw DINOv3 tokens; connector(Qwen hidden)] into the ONE
        # pretrained cross-attn (no new modules). DINO arrives via forward(dino_hidden=...)
        # (cached d-keys). dino_drop_prob = DINO-dropout curriculum (segment masked off
        # per-sample, training only) — anti rich-get-richer, makes Qwen self-sufficient.
        fuse_dino: bool = False,
        dino_drop_prob: float = 0.1,
        qwen_drop_prob: float = 0.0,   # mirror of dino_drop (fusion only; T unaffected)
        # IM multi-view identity for the DINO segment (Qwen segment self-identifies via
        # template/M-RoPE). ZERO-INIT learned per-ordinal embedding added to the DINO
        # tokens — starts as a no-op (I1-trained fusion ckpts unaffected), learns at S3.
        dino_view_embed_max: int = 8,
        view_embed_mode: str = "learned",   # "learned" (zero-init) | "sincos" (HY3D-mv fixed)
        # ── REPA-style SS auxiliary alignment (trellis2_blip3o/repa.py; official
        # sihyun-yu/REPA recipe: 3-layer MLP projector on the output of the SS block at
        # ~1/3 depth, negative-cosine to per-asset VGGT voxel targets, all t uniformly,
        # total = flow + repa_coeff·proj). repa_root = target cache root ("" = OFF —
        # existing runs unaffected). Requires COMPILE_SS=0 / --compile_ss_flow False
        # (forward hooks don't survive torch.compile). CFG-dropped samples get aux 0.
        repa_root: str = "",
        repa_coeff: float = 0.5,
        repa_depth: int = 0,                    # 1-indexed tap depth; 0 = auto num_blocks//3
        repa_zdim: int = 2049,                  # target dim (density 1 + RAW 2048-d VGGT feature)
        # Stage-split training: build/train only one component per job (see __init__).
        train_stages: str = "all",              # all | ss | shape | tex | geotex
        # ── unified geo-tex DiT (train_stages="geotex"; docs/UNIFIED_GEOTEX_DIT_DESIGN.md) ──
        # Warm-start comes from TWO run ckpts (shape + tex specialists), each loaded with its
        # EMA overlay + its own connector — NOT via --init_from_checkpoint (single-ckpt path).
        geotex_shape_init: str = "",            # e.g. runs/s3_shape_t50b/checkpoint-8000
        geotex_tex_init: str = "",              # e.g. runs/s3_tex_t50b/checkpoint-8000
        geotex_coupling: str = "union",         # union (MF-bare; G0 [SCALE]=0.168 cleared) | gated
        geotex_cond_mode: str = "cross_attn",   # cross_attn (variant A) | stream (variant B, three-stream MMDiT)
        geotex_cond_stream_blocks: int = 10,    # cond stream on the LAST N blocks; <=0 = all 30
        geotex_xattn_anneal_start: int = 0,     # cross-attn scale 1->0 over [start, end] steps
        geotex_xattn_anneal_end: int = 0,
        geotex_bidir: bool = False,             # corner-masked bidirectional geo<->tex (user topology 2026-08-11)
        geotex_fused: bool = True,              # fused MMDiT attention (1 varlen call/lane): MFU 18.7%->30.0%, G0-fused certified
        geotex_gc: float = 1.0,                 # fraction of block pairs gradient-checkpointed (1=all, 0=none)
        # ── S2b: unfreeze geo (design doc "three-pack"; only with G3 as a red line) ──
        geotex_unfreeze_geo: bool = False,      # geo becomes trainable (joint's value needs this)
        geotex_geo_loss_w: float = 1.0,         # geo's own velocity loss (else geo just serves tex)
        geotex_p_corner: float = 0.4,           # t_s=0 corner mass (user 2026-08-11: flagship-mode priority; MF-exact 0.2 = A1 arm)
        geotex_p_corner2: float = 0.2,          # t_x=1 corner (mesh-only marginal; MF's second corner, bidir design)
        geotex_mismatch_w: float = 0.0,         # mismatched-image hinge weight (experiment)
        geotex_mismatch_margin: float = 0.15,
        geotex_concat_cond: bool = False,       # per-voxel shape concat into tex (cascade legacy)
        logitnorm_mean: float = 1.0,
        logitnorm_std: float = 1.0,
        flow_sigma_min: float = 1e-5,
        # TRELLIS ckpt overrides (None → multimodal_decoder.builder defaults)
        trellis_ss_flow_ckpt: Optional[str] = None,
        trellis_shape_slat_ckpt: Optional[str] = None,
        trellis_tex_slat_ckpt: Optional[str] = None,
        trellis_sc_vae_ckpt: Optional[str] = None,
        **kwargs,
    ):
        self.vlm_model = vlm_model
        self.vlm_hidden_size = vlm_hidden_size
        self.freeze_vlm = freeze_vlm
        self.build_slat = build_slat
        self.slat_resolution = slat_resolution
        self.detach_cond = detach_cond
        self.cond_max_length = cond_max_length
        self.mask_drop_prob = mask_drop_prob
        self.anchor_drop_prob = anchor_drop_prob
        self.cfg_joint_drop_prob = cfg_joint_drop_prob
        self.cond_fusion = cond_fusion
        self.fusion_layers = fusion_layers
        self.flow_weight = flow_weight
        self.flow_stage_weights = flow_stage_weights
        self.dino_align = dino_align
        self.dino_align_weight = dino_align_weight
        self.dino_align_mode = dino_align_mode
        self.dino_image_token_id = dino_image_token_id
        self.dino_merge_size = dino_merge_size
        self.dual_cond = dual_cond
        self.dual_cond_max_views = dual_cond_max_views
        self.dual_slat_qwen_stride = dual_slat_qwen_stride
        self.dual_qwen_last_frac = dual_qwen_last_frac
        self.dual_anchor_pool = dual_anchor_pool
        self.dual_anchor_token_budget = dual_anchor_token_budget
        self.dual_ss_checkpoint = dual_ss_checkpoint
        self.distill_dino = distill_dino
        self.distill_v_weight = distill_v_weight
        self.distill_f_weight = distill_f_weight
        self.distill_f_blocks = distill_f_blocks
        self.distill_cfg_lo = distill_cfg_lo
        self.distill_cfg_hi = distill_cfg_hi
        self.distill_cfg_null_grad = distill_cfg_null_grad
        self.target_tokens_per_view = target_tokens_per_view
        self.build_vlm = build_vlm
        self.repa_root = repa_root
        self.repa_coeff = repa_coeff
        self.repa_depth = repa_depth
        self.repa_zdim = repa_zdim
        # (REPA λ-warmup env is read lazily in the MODEL's _repa_lambda, not here — this is the
        # config class; the earlier copy here was a misplaced-edit bug that crashed on empty env.)
        self.train_stages = train_stages
        self.geotex_shape_init = geotex_shape_init
        self.geotex_tex_init = geotex_tex_init
        self.geotex_coupling = geotex_coupling
        self.geotex_cond_mode = geotex_cond_mode
        self.geotex_cond_stream_blocks = geotex_cond_stream_blocks
        self.geotex_xattn_anneal_start = geotex_xattn_anneal_start
        self.geotex_xattn_anneal_end = geotex_xattn_anneal_end
        self.geotex_bidir = geotex_bidir
        self.geotex_fused = geotex_fused
        self.geotex_gc = geotex_gc
        self.geotex_unfreeze_geo = geotex_unfreeze_geo
        self.geotex_geo_loss_w = geotex_geo_loss_w
        self.geotex_p_corner = geotex_p_corner
        self.geotex_p_corner2 = geotex_p_corner2
        self.geotex_mismatch_w = geotex_mismatch_w
        self.geotex_mismatch_margin = geotex_mismatch_margin
        self.geotex_concat_cond = geotex_concat_cond
        self.fuse_dino = fuse_dino
        self.dino_drop_prob = dino_drop_prob
        self.qwen_drop_prob = qwen_drop_prob
        self.dino_view_embed_max = dino_view_embed_max
        self.view_embed_mode = view_embed_mode
        self.logitnorm_mean = logitnorm_mean
        self.logitnorm_std = logitnorm_std
        self.flow_sigma_min = flow_sigma_min
        self.trellis_ss_flow_ckpt = trellis_ss_flow_ckpt
        self.trellis_shape_slat_ckpt = trellis_shape_slat_ckpt
        self.trellis_tex_slat_ckpt = trellis_tex_slat_ckpt
        self.trellis_sc_vae_ckpt = trellis_sc_vae_ckpt
        super().__init__(**kwargs)


class TrellisNativeVLMForConditionalGeneration(PreTrainedModel):
    config_class = TrellisNativeVLMConfig
    # The VLM submodule has its own _supports_* flags; we don't gate on them here.
    supports_gradient_checkpointing = True

    def __init__(self, config: TrellisNativeVLMConfig):
        super().__init__(config)

        # --- native VLM encoder (vision baked in) ---
        # TODO(save/load): loading pretrained weights in __init__ is a skeleton
        #   shortcut. For clean save_pretrained/from_pretrained of the WHOLE
        #   composite, switch to building the VLM from a nested vlm_config and
        #   loading the backbone separately (or override save/load to exclude the
        #   frozen VLM). Fine for v1 training (we only save connector+flows anyway).
        from transformers import AutoModelForImageTextToText  # local import: new-ish auto class
        # `dtype=` (not the deprecated `torch_dtype=`) — required by transformers 5.2+
        # (the env `blip3o_trellis_qwen35` we run Qwen3.5 in). DEPRECATED backbones
        # Qwen3-VL / Qwen2.5-VL still load via 4.57's older `torch_dtype` path if you
        # accept the warning.
        # VLM attn: keep sdpa (safe default). Earlier attempt at flash_attention_2 here
        # coincided with NaN-from-step-4 in the first elastic cascade run; reverting to
        # sdpa first before re-attempting (numerical instability of flash_attn_2 on
        # Qwen3.5-2B's 6/24 full-attn layers under bf16 + variable cond_len is suspected).
        if getattr(config, "build_vlm", True):
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                config.vlm_model, dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            if config.freeze_vlm:
                self.vlm.requires_grad_(False)
                self.vlm.eval()
        else:
            # VLM-hidden CACHE mode (vlm_cache.py): all conds arrive precomputed via the
            # forward's cond_hidden fast path → skip loading the 2B VLM entirely
            # (~5-6 GB GPU + load time saved → bigger batch). encode_cond is unusable.
            self.vlm = None
            rank0_print("[vlm_cache] build_vlm=False — VLM NOT loaded; conds must come "
                        "from the cache (forward cond_hidden=...)")

        # Derive the LLM hidden dim from the loaded VLM (Qwen3.5-2B = 2048) rather
        # than trusting config.vlm_hidden_size. Falls back to the config value.
        if self.vlm is not None:
            try:
                text_cfg = self.vlm.config.get_text_config()
            except Exception:
                text_cfg = getattr(self.vlm.config, "text_config", self.vlm.config)
            vlm_hidden = getattr(text_cfg, "hidden_size", None) or config.vlm_hidden_size
            config.vlm_hidden_size = int(vlm_hidden)  # keep config in sync for save/reload

        # --- TRELLIS cascade (these builders LOAD the pretrained TRELLIS ckpts) ---
        # Stage-split (--train_stages all|ss|shape|tex): build ONLY the flow(s) this job
        # trains. Legal because the 3 stages are GT-decoupled in training (shape conditions
        # GT coords, tex teacher-forces GT shape SLAT) — TRELLIS官方 trains them as separate
        # runs too. A split job gets its own connector copy (per-stage connectors are a
        # feature: more capacity, assembled at inference). flow_heads skips None flows.
        _stages = str(getattr(config, "train_stages", "all"))
        assert _stages in ("all", "ss", "shape", "tex", "geotex"), f"bad train_stages={_stages!r}"
        self.ss_flow = build_ss_flow(config) if _stages in ("all", "ss") else None
        self.unified_geotex = None
        # From-scratch three-stream MMDiT (no TRELLIS.2 weights). Everything
        # downstream of construction — elastic GC, the freeze audit, the loss,
        # the sampler — is the SAME path; only the pieces that only make sense
        # for a warm start are skipped, each guarded on this flag.
        _gt_scratch = (_stages == "geotex"
                       and bool(getattr(config, "geotex_from_scratch", False)))
        if _gt_scratch:
            from trellis2_blip3o.mmdit3d import MMDiT3D
            self.unified_geotex = MMDiT3D(
                dim=int(getattr(config, "geotex_dim", 768)),
                num_heads=int(getattr(config, "geotex_heads", 6)),
                depth_double=int(getattr(config, "geotex_depth_double", 8)),
                depth_single=int(getattr(config, "geotex_depth_single", 16)),
                mlp_ratio=float(getattr(config, "geotex_mlp_ratio", 5.375)),
                initialization=str(getattr(config, "geotex_init", "scaled")),
                # False since 2026-08-17 — the per-voxel shape concat is the
                # cascade's mechanism and measures at 2% here, while it is the
                # only structural asymmetry between the two generated streams.
                concat_cond=bool(getattr(config, "geotex_concat_cond", False)),
                # SD3/FLUX's pooled-condition -> adaLN path. Default ON: without
                # it the image reaches a voxel only through gate_msa, which
                # opens at |g| ~ 0.026 (see MMDiT3D.__init__).
                pooled_cond=bool(getattr(config, "geotex_pooled_cond", True)))
            # ONE cond stream ⇒ one connector. It is the standard-named
            # diffusion_connector, built fresh a few dozen lines below by the
            # ordinary (non-geotex) path, so EMA/save/load conventions hold.
            self.geo_connector = None
            rank0_print(
                f"[geotex] FROM SCRATCH: three-stream MMDiT3D "
                f"dim={self.unified_geotex.dim} "
                f"{len(self.unified_geotex.geo_flow.blocks)} triple + "
                f"{len(self.unified_geotex.shared_blocks)} shared, "
                f"{sum(p.numel() for p in self.unified_geotex.parameters())/1e6:.0f}M params")
        elif _stages == "geotex":
            # Unified geo-tex DiT (Stage-1): both SLAT specialists assembled into ONE
            # dual-stream model, warm-started from TWO run ckpts with EMA overlay
            # (unified_geotex.assemble_unified; G0-certified bit-exact). Geo stream is
            # FROZEN; tex stream + t-mixer (+ gates) train. The geo run's connector is
            # loaded frozen alongside; the tex run's connector becomes THE
            # diffusion_connector (standard name → EMA/save/inference conventions hold).
            from trellis2_blip3o.unified_geotex import (
                assemble_unified, load_connectors, assemble_unified_from_run,
                load_run_connectors)
            assert config.geotex_shape_init and config.geotex_tex_init, \
                "[geotex] --geotex_shape_init and --geotex_tex_init are required"
            # A geotex RUN checkpoint (S1 -> S2b) stores `unified_geotex.*`; the
            # specialist pair stores `shape_slat_512.` / `tex_slat_512.`. Detect
            # rather than make the caller remember which flag to use.
            _resume = None
            if config.geotex_shape_init == config.geotex_tex_init:
                from safetensors import safe_open
                with safe_open(os.path.join(config.geotex_shape_init,
                                            "model.safetensors"), framework="pt") as _f:
                    if any(k.startswith("unified_geotex.") for k in _f.keys()):
                        _resume = config.geotex_shape_init
            _kw = dict(cond_mode=getattr(config, "geotex_cond_mode", "cross_attn"),
                       coupling=config.geotex_coupling,
                       bidirectional=bool(getattr(config, "geotex_bidir", False)))
            if _kw["cond_mode"] == "stream":
                _kw["cond_stream_blocks"] = int(
                    getattr(config, "geotex_cond_stream_blocks", 10))
            if _resume:
                self.unified_geotex = assemble_unified_from_run(_resume, **_kw)
                rank0_print(f"[geotex] RESUMED unified model from run ckpt {_resume}")
            else:
                self.unified_geotex = assemble_unified(
                    config.geotex_shape_init, config.geotex_tex_init, **_kw)
            self.unified_geotex.fused_attn = bool(getattr(config, "geotex_fused", True))
            # connectors follow the same resume/fresh split (a run ckpt stores
            # them as geo_connector.* / diffusion_connector.*)
            if _resume:
                self.geo_connector, self.diffusion_connector = load_run_connectors(_resume)
            else:
                self.geo_connector, self.diffusion_connector = load_connectors(
                    config.geotex_shape_init, config.geotex_tex_init)
            self.geo_connector.requires_grad_(False)
            self.geo_connector.eval()
            # geo_flow is a PRETRAINED specialist on this branch and S1 freezes
            # it on purpose. Doing the same to a from-scratch MMDiT3D would pin
            # a RANDOM geo tower with a zero-init out_layer for the whole run
            # (v_s == 0 forever, S2b's geo/distill terms constant with no
            # gradient) — and the freeze audit could not catch it, since it only
            # inspects params that still require grad. Hence the branch.
            self.unified_geotex.geo_flow.requires_grad_(False)

        if _stages == "geotex":
            # Activation checkpointing. In the bidir path the WHOLE block pair is
            # wrapped, so GC costs a full extra forward (~29% of total work) —
            # worth trading for memory only when memory is actually tight.
            # geotex_gc: 1.0 = every block (default), 0.0 = none, 0<f<1 = the
            # first f fraction of blocks (early blocks hold activations longest,
            # so checkpointing those buys the most memory per unit of recompute).
            _gc = float(getattr(config, "geotex_gc", 1.0))
            if getattr(self.unified_geotex, "from_scratch", False):
                # MMDiT3D drives the blocks itself (it reaches into blk.attn /
                # norm1 / norm2 / mlp and never calls block.forward), so the
                # per-block use_checkpoint flags below are dead there — setting
                # them would have made --geotex_gc 0 silently keep checkpointing
                # all 24 joint blocks. Its own switch is the model-level one.
                self.unified_geotex.use_checkpoint = _gc > 0
                _nb = (len(self.unified_geotex.shared_blocks)
                       + len(self.unified_geotex.geo_flow.blocks))
                rank0_print(f"[geotex] gradient checkpointing "
                            f"{'ON' if _gc > 0 else 'OFF'} for all {_nb} joint blocks "
                            f"(geotex_gc={_gc}; elastic GC overrides per step)")
            else:
                _nb = len(self.unified_geotex.tex_flow.blocks)
                _n_ckpt = int(round(_gc * _nb))
                for _i, _b in enumerate(self.unified_geotex.tex_flow.blocks):
                    _b.use_checkpoint = (_i < _n_ckpt)
                rank0_print(f"[geotex] gradient checkpointing on {_n_ckpt}/{_nb} block pairs "
                            f"(geotex_gc={_gc})")
            # ── S2b: unfreeze geo ──
            if getattr(config, "geotex_unfreeze_geo", False):
                self.unified_geotex.unfreeze_geo()
                rank0_print(f"[geotex] S2b: geo UNFROZEN "
                            f"(geo_loss_w={config.geotex_geo_loss_w})")
            rank0_print(f"[geotex] unified assembled: shape={config.geotex_shape_init} "
                        f"tex={config.geotex_tex_init} coupling={config.geotex_coupling} "
                        f"(geo frozen, tex GC on)")
        if config.build_slat and _stages in ("all", "shape", "tex"):
            # Attribute names keep the "_512" suffix for cross-file compatibility; the
            # underlying flow is the 1024 variant when slat_resolution=1024 (manifest
            # must point target_shape_slat_512/tex_slat_512 to the 1024 latent paths).
            _res = int(getattr(config, "slat_resolution", 512))
            _b_shape = _stages in ("all", "shape")
            _b_tex = _stages in ("all", "tex")
            if _res == 1024:
                self.shape_slat_512 = build_shape_slat_1024(config) if _b_shape else None
                self.tex_slat_512 = build_tex_slat_1024(config) if _b_tex else None
            else:
                self.shape_slat_512 = build_shape_slat_512(config) if _b_shape else None
                self.tex_slat_512 = build_tex_slat_512(config) if _b_tex else None
            rank0_print(f"[slat] resolution={_res} stages={_stages} → shape="
                        f"{_b_shape} tex={_b_tex}")
            self.trellis_decoders = build_trellis_decoders(config)  # frozen, inference-only
        else:
            # SS-only training: skip the SLAT flows + decoders (lighter, faster).
            # forward()/compute_cascade_flow_loss already guard these as None.
            self.shape_slat_512 = None
            self.tex_slat_512 = None
            self.trellis_decoders = None

        # --- connector: MLP (default) or i1-style 2-block Transformer adapter (cond_adapter="xf2").
        # Both map (B,T,vlm_dim)→(B,T,1024) with a dist-matched output LayerNorm; the adapter adds
        # self-attn capacity to the language→generation interface (i1 §3.1). Named diffusion_connector
        # either way → always-trainable (train_native flow-freeze) + saved/loaded with the ckpt.
        if _stages == "geotex" and not _gt_scratch:
            pass  # both connectors already loaded from the two warm-start ckpts above
        elif getattr(config, "cond_adapter", "mlp") == "xf2":
            from trellis2_blip3o.connector import TRELLIS2TransformerAdapter
            self.diffusion_connector = TRELLIS2TransformerAdapter(
                vlm_hidden_dim=config.vlm_hidden_size,
                trellis_cond_dim=TRELLIS_COND_DIM,
            )
        else:
            self.diffusion_connector = TRELLIS2Connector(
                vlm_hidden_dim=config.vlm_hidden_size,
                trellis_cond_dim=TRELLIS_COND_DIM,
            )
        # DINO position signature stamped on the qwen cond segment (ablation crossdpos win;
        # memory blip3o-rope-position-hole). Connector submodule → saved/loaded with the ckpt.
        if getattr(config, "cond_pos_stamp", False):
            from trellis2_blip3o.pos_stamp import DinoPosStamp
            self.diffusion_connector.pos_stamp = DinoPosStamp()

        # --- fusion: per-view-ordinal embedding for the DINO segment (multi-image identity) ---
        # "learned"(default): zero-init trainable — no-op at init so I1 fusion ckpts load clean,
        #   but empirically stays a weak slot tag (L2~2) and never enables view routing.
        # "sincos": FIXED full-strength 1D sin-cos over view ordinal (Hunyuan3D-2-mv recipe;
        #   same idea as our dpos position-stamp fix) — gives the model a strong, orthogonal
        #   "which view" signal from step 0 so cross-attn can route per-view instead of averaging.
        if not getattr(config, "fuse_dino", False):
            self.dino_view_embed = None
        else:
            _vmax = int(getattr(config, "dino_view_embed_max", 8))
            if getattr(config, "view_embed_mode", "learned") == "sincos":
                import numpy as _np, os as _os
                D = TRELLIS_COND_DIM
                omega = _np.arange(D // 2, dtype=_np.float64) / (D / 2.0)
                omega = 1.0 / (10000 ** omega)
                pos = _np.arange(_vmax, dtype=_np.float64)
                out = _np.einsum("m,d->md", pos, omega)            # (Vmax, D/2)
                emb = _np.concatenate([_np.sin(out), _np.cos(out)], axis=1)  # (Vmax, D)
                # SCALE: raw sincos L2≈22.6 ≈ 0.7× the DINO token L2 (~32) — that drowns the
                # content and the flow flees to the qwen segment (DINO attn share collapsed
                # 0.72→0.22, killing the routing gain). Scale to ~15% of DINO (L2≈4.8) so the
                # view signal is clear but doesn't out-weigh the features. env VIEW_EMBED_SCALE.
                _scale = float(_os.environ.get("VIEW_EMBED_SCALE", "1.0") or "1.0")
                self.register_buffer("dino_view_embed",
                                     torch.from_numpy(emb).float() * _scale, persistent=True)
            else:
                self.dino_view_embed = torch.nn.Parameter(torch.zeros(_vmax, TRELLIS_COND_DIM))

        # --- DINOv3 conditioning aligner (REPA-inspired; image tasks only) ---
        # Frozen DINOv3 + train-only projection head. OFF unless config.dino_align.
        self.dino_aligner = None
        if getattr(config, "dino_align", False):
            from trellis2_blip3o.dino_align import DinoAligner
            self.dino_aligner = DinoAligner(
                cond_dim=TRELLIS_COND_DIM, mode=getattr(config, "dino_align_mode", "spatial"),
            )

        # --- dual-branch conditioning (Know3D-style; see DUAL_COND_DESIGN.md) ---
        # Keep the original TRELLIS DINOv3 cross-attn as a geometry ANCHOR (fed real DINOv3
        # features) + add a parallel zero-init-gated Qwen cross-attn per block. SS flow (dense)
        # only for now; SLAT (sparse) is TODO. Mutually exclusive with depthwise routing.
        self.dual_router = None
        if getattr(config, "dual_cond", False):
            from trellis2_blip3o.dual_cond import DualCondRouter, install_dual_routing
            from trellis2_blip3o.dino_align import TRELLIS_DINOV3_NAME, DINOV3_IMAGE_SIZE
            from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor  # type: ignore
            self.dual_router = DualCondRouter(
                max_views=int(config.dual_cond_max_views),
                anchor_pool=int(getattr(config, "dual_anchor_pool", 1)),
                anchor_token_budget=int(getattr(config, "dual_anchor_token_budget", 0)),
            )
            # SS: Qwen on every block (cheap, dense). SLAT: Qwen on every-other block
            # (qwen_stride=2) — the SLAT Qwen cross-attn is the heavy part (doubles SLAT activation
            # ~+19GB at 512); halving it keeps dual under the ~28GB headroom. anchor (DINOv3) is on
            # ALL blocks regardless. stride configurable via dual_slat_qwen_stride.
            _slat_stride = int(getattr(config, "dual_slat_qwen_stride", 2))
            # PREFERRED: inject Qwen only on the last `dual_qwen_last_frac` fraction of EACH flow's
            # blocks (e.g. 0.2 → last 20%). Aligns the heavy Qwen branch with the trainable tail
            # (flow_tune last15/20) and cuts dual's +9.5GB by ~80% → fits 16-GPU/2-node. Applies
            # uniformly to SS + both SLAT flows; overrides stride when >0.
            _qlf = float(getattr(config, "dual_qwen_last_frac", 0.0))
            install_dual_routing(self.ss_flow, self.dual_router, qwen_stride=1, qwen_last_frac=_qlf)
            if config.build_slat:   # SLAT flows (sparse) share the same router (same H_dino/H_qwen)
                install_dual_routing(self.shape_slat_512, self.dual_router, qwen_stride=_slat_stride, qwen_last_frac=_qlf)
                install_dual_routing(self.tex_slat_512, self.dual_router, qwen_stride=_slat_stride, qwen_last_frac=_qlf)
            if getattr(config, "dual_ss_checkpoint", False):
                # gradient-checkpoint the WHOLE SS block (the elastic SLAT GC doesn't cover SS,
                # which is otherwise compiled & fully live). Each block is now a DualCondInjectBlock
                # wrapper; set its inner block's use_checkpoint. (Needs compile OFF to take effect.)
                for b in self.ss_flow.blocks:
                    inner = getattr(b, "block", b)
                    if hasattr(inner, "use_checkpoint"):
                        inner.use_checkpoint = True
                rank0_print("[dual_cond] SS block gradient-checkpointing ON (whole block, not just dual branch)")
            # Frozen DINOv3 (raw features = the anchor cond). Non-child plain object so it stays
            # off the state_dict / DeepSpeed param partitioning (inference-only), like dino_align.
            _dino = DinoV3FeatureExtractor(TRELLIS_DINOV3_NAME, image_size=DINOV3_IMAGE_SIZE)
            _dino.model.eval()
            for p in _dino.model.parameters():
                p.requires_grad_(False)
            # bf16: halves the frozen DINOv3 weights (~0.6GB) AND its forward activations. Safe —
            # the earlier fp32-forcing (dino_align) was to avoid a bf16-input vs fp32-weight conv
            # MISMATCH under autocast; making BOTH bf16 (weights here + input cast in _build_dino_anchor)
            # + autocast OFF is consistent. Frozen inference-only → no training-instability concern.
            _dino.model.to(torch.bfloat16)
            object.__setattr__(self, "_dino_extractor", _dino)
            _nb = len(self.ss_flow.blocks) + (
                len(self.shape_slat_512.blocks) + len(self.tex_slat_512.blocks)
                if config.build_slat else 0)
            rank0_print(f"[dual_cond] anchor ON: DINOv3 cross-attn (frozen) + scalar-gated Qwen "
                        f"branch on {'SS+SLAT' if config.build_slat else 'SS'} ({_nb} blocks total); "
                        f"per-view embed ≤{config.dual_cond_max_views}")

        # --- V3 distillation teacher input: frozen DINOv3 (shared with dual when both on) ---
        if getattr(config, "distill_dino", False) and getattr(self, "_dino_extractor", None) is None:
            from trellis2_blip3o.dino_align import TRELLIS_DINOV3_NAME, DINOV3_IMAGE_SIZE
            from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor  # type: ignore
            _dino = DinoV3FeatureExtractor(TRELLIS_DINOV3_NAME, image_size=DINOV3_IMAGE_SIZE)
            _dino.model.eval()
            for p in _dino.model.parameters():
                p.requires_grad_(False)
            _dino.model.to(torch.bfloat16)   # frozen inference-only; same rationale as dual above
            object.__setattr__(self, "_dino_extractor", _dino)   # non-child → off state_dict/ZeRO
            rank0_print(f"[distill] V3 condition-swap KD ON: teacher=frozen flow+DINOv3, "
                        f"λ_v={config.distill_v_weight} λ_f={config.distill_f_weight} "
                        f"blocks={config.distill_f_blocks}")

        # --- per-stage flow loss fns (shared helper) ---
        self._loss_fn_ss, self._loss_fn_slat = flow_heads.build_flow_loss_fns(config)

        # --- REPA-style SS aux alignment (repa.py; config.repa_root gates it) ---
        # Projector = official REPA build_mlp (hidden = SS block width → 2048 → 2048 →
        # z_dim); a forward hook on ss_flow.blocks[depth-1] stashes that block's OUTPUT
        # (B, 4096, C) into self._repa_stash during the SS forward; flow_heads pops it,
        # projects, and cosine-aligns to the batch's repa_target. The projector is a
        # child module → trainable (not matched by the flow-freeze tags) + saved/loaded
        # with the checkpoint.
        self.repa_projector = None
        self._repa_stash = None
        if getattr(config, "repa_root", ""):
            import os as _os_r
            if self.ss_flow is None:
                raise ValueError("[repa] repa_root set but no SS flow built — REPA is an "
                                 "SS-flow aux; use --train_stages all|ss.")
            # hooks do NOT survive torch.compile of the blocks (Dynamo inlines the block
            # forward). IM SS runs use COMPILE_SS=0 anyway; enforce it loudly here, and
            # train_native.py enforces the matching --compile_ss_flow False.
            if _os_r.environ.get("COMPILE_SS", "0") == "1":
                raise RuntimeError("[repa] requires COMPILE_SS=0 — the SS block forward "
                                   "hook is skipped inside torch.compile'd blocks.")
            for _flag, _why in (("dual_cond", "blocks are wrapped by dual routing"),
                                ("distill_dino", "KD teacher passes re-fire the hook")):
                if getattr(config, _flag, False):
                    raise ValueError(f"[repa] not supported with {_flag} ({_why}).")
            if getattr(config, "cond_fusion", "none") == "depthwise":
                raise ValueError("[repa] not supported with cond_fusion=depthwise "
                                 "(blocks are wrapped by depth routing).")
            from trellis2_blip3o.repa import build_repa_projector
            _nb = int(self.ss_flow.num_blocks)
            _depth = int(getattr(config, "repa_depth", 0)) or (_nb // 3)
            if not (1 <= _depth <= _nb):
                raise ValueError(f"[repa] repa_depth={_depth} out of range 1..{_nb}")
            _hid = int(self.ss_flow.model_channels)
            _zd = int(getattr(config, "repa_zdim", 2049))
            self.repa_projector = build_repa_projector(_hid, _zd)
            self._repa_depth = _depth
            self._repa_stash = {}

            def _repa_hook(_mod, _inp, out, _stash=self._repa_stash, _model=self):
                if _model.training:               # training-only (sampling won't stash)
                    _stash["h"] = out

            self._repa_hook_handle = self.ss_flow.blocks[_depth - 1].register_forward_hook(
                _repa_hook)
            rank0_print(f"[repa] SS aux alignment ON: tap=blocks[{_depth - 1}] (depth "
                        f"{_depth}/{_nb}), width {_hid} → z_dim {_zd}, coeff="
                        f"{getattr(config, 'repa_coeff', 0.5)}, root={config.repa_root}")

        # --- depth-wise Semantic Routing (optional): per-block fusion over VLM layers ---
        # One router PER flow (each is a separate DiT); the connector is SHARED. Each flow
        # block is wrapped so block d uses router._cond_per_block[d] (cond is args[2]).
        self.cond_fusion = getattr(config, "cond_fusion", "none")
        self._fusion_layers = getattr(config, "fusion_layers", 0)  # 0 = all VLM layer outputs
        if self.cond_fusion == "depthwise":
            from trellis2_blip3o.depth_fusion import DepthFusionRouter, install_depth_routing
            try:
                tcfg = self.vlm.config.get_text_config()
            except Exception:
                tcfg = getattr(self.vlm.config, "text_config", self.vlm.config)
            n_vlm = int(getattr(tcfg, "num_hidden_layers", 24))
            L = n_vlm if self._fusion_layers in (0, None) else min(self._fusion_layers, n_vlm)
            self._fuse_L = L
            self.ss_router = DepthFusionRouter(len(self.ss_flow.blocks), L, self.diffusion_connector)
            install_depth_routing(self.ss_flow, self.ss_router)
            if config.build_slat:
                self.shape_router = DepthFusionRouter(len(self.shape_slat_512.blocks), L, self.diffusion_connector)
                self.tex_router = DepthFusionRouter(len(self.tex_slat_512.blocks), L, self.diffusion_connector)
                install_depth_routing(self.shape_slat_512, self.shape_router)
                install_depth_routing(self.tex_slat_512, self.tex_router)
            rank0_print(f"[depthwise] routing on: fuse last {L} VLM layers → per-block "
                        f"(SS {len(self.ss_flow.blocks)} blocks"
                        + (f", Shape/Tex too" if config.build_slat else "") + ")")

        # optional: torch.compile the DENSE SS-flow blocks (regional, compiled once & reused).
        # Fuses the LayerNorm fp32 round-trips + modulate/gate elementwise (~130 ms/step of
        # cast+elementwise) → ~1.34x lossless on SS (loss Δ ~4e-4). SLAT flows can't compile
        # (sparse dynamic shapes). Enable with env COMPILE_SS=1.
        import os as _os
        if _os.environ.get("COMPILE_SS", "0") == "1" and self.ss_flow is not None and hasattr(self.ss_flow, "blocks"):
            import torch._dynamo as _dyn; _dyn.config.cache_size_limit = 64
            for _i in range(len(self.ss_flow.blocks)):
                self.ss_flow.blocks[_i] = torch.compile(self.ss_flow.blocks[_i], dynamic=True)
            rank0_print(f"[compile-ss] torch.compile on {len(self.ss_flow.blocks)} SS blocks")

    # HF's `--gradient_checkpointing True` only enables GC on the VLM. The TRELLIS flow
    # blocks have their OWN `use_checkpoint` (default False) — without this propagation,
    # the flow activations dominate memory and OOM at higher latent resolution (1024).
    # Overriding enable/disable so the flag actually flows through.
    def _set_flow_checkpointing(self, enable: bool):
        for name in ("ss_flow", "shape_slat_512", "tex_slat_512"):
            flow = getattr(self, name, None)
            if flow is None:
                continue
            for b in getattr(flow, "blocks", []):
                # depthwise wraps each block in _CondInjectBlock(.block=orig); unwrap.
                target = getattr(b, "block", b)
                if hasattr(target, "use_checkpoint"):
                    target.use_checkpoint = bool(enable)

    def gradient_checkpointing_enable(self, **kwargs):
        super().gradient_checkpointing_enable(**kwargs)
        self._set_flow_checkpointing(True)

    def gradient_checkpointing_disable(self):
        super().gradient_checkpointing_disable()
        self._set_flow_checkpointing(False)

    # ------------------------------------------------------------------
    # Core irreducible piece: run the VLM → full last-layer hidden + mask.
    # The SAME method a future offline caching script would call (caching is a
    # deferred, purely-additive optimization — see QWEN35_VLM_DESIGN.md §9).
    # ------------------------------------------------------------------
    def encode_cond(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ):
        """[text (+ image/multi-image/video)] → (cond_hidden (B,T,H), cond_key_mask (B,T)).
        If return_layers: returns (list of selected VLM layer hiddens, mask) for depth-wise fusion.

        Inputs are exactly what the native `AutoProcessor` emits (Phase-3 collator).
        Uses `hidden_states[-1]` (last layer) — matches Qwen-Image-Edit. Drops nothing
        in v1 (keep full hidden; template-prefix `drop_idx` is deferred polish).
        """
        # Only build the kwargs the VLM actually expects (avoid passing None video
        # tensors that some processors/models dislike).
        vlm_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if pixel_values is not None:
            vlm_inputs["pixel_values"] = pixel_values
            vlm_inputs["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            vlm_inputs["pixel_values_videos"] = pixel_values_videos
            vlm_inputs["video_grid_thw"] = video_grid_thw

        ctx = torch.no_grad() if self.config.freeze_vlm else contextlib.nullcontext()
        with ctx:
            out = self.vlm(**vlm_inputs)

        hs = out.hidden_states  # tuple: (embedding, layer_1, ..., layer_N)
        ref = hs[-1]
        if attention_mask is not None:
            cond_key_mask = attention_mask.bool().to(ref.device)
        else:
            cond_key_mask = torch.ones(ref.shape[:2], dtype=torch.bool, device=ref.device)

        if return_layers:
            # depth-wise fusion: the last L layer OUTPUTS (exclude the embedding hs[0]).
            layers = list(hs[1:])
            L = getattr(self, "_fuse_L", len(layers))
            if 0 < L < len(layers):
                layers = layers[-L:]
            return layers, cond_key_mask

        # single-layer cond: "penultimate" → hs[-2], else last layer hs[-1] (Qwen-Image default).
        idx = -2 if getattr(self.config, "cond_fusion", "none") == "penultimate" else -1
        return hs[idx], cond_key_mask

    def cond_and_null(self, cond_hidden: torch.Tensor):
        """(cond, neg_cond) for CFG sampling. cond = connector(hidden);
        neg_cond = connector(0) = the training-time unconditional (see
        flow_heads.null_cond_like — NOT zeros in cond space). Any generate()/
        sampling path MUST use this for the CFG negative branch."""
        cond = self.diffusion_connector(cond_hidden)
        neg_cond = flow_heads.null_cond_like(self.diffusion_connector, cond_hidden)
        return cond, neg_cond

    # Tasks routed to the LM-loss path (vlm.lm_head + CE on `labels`). Add a task
    # name here when registering a new chat-style task — that's the only model-
    # side wiring needed.
    _LM_TASKS = {"vqa", "grounding", "text_sft"}

    def _build_dino_anchor(self, dino_images, B: int, dtype, device) -> torch.Tensor:
        """Dual-branch DINOv3 anchor cond (B, N_anchor, 1024).

        - text (no dino_images): a single zero token → the original cross-attn runs its
          unconditional path; the Qwen branch carries everything.
        - image / multi-image: run the frozen DINOv3 on the V conditioning views (dino_images
          is flat (B*V,3,512,512) in cond-view order), reshape to per-view (B,N,1024), add the
          learned per-view embedding, concat along tokens. Raw DINOv3 features (full token set,
          incl. CLS/register) — exactly what the pretrained TRELLIS cross-attn was trained on.
        """
        DINO_DIM = 1024
        if dino_images is None or dino_images.numel() == 0:
            return torch.zeros(B, 1, DINO_DIM, dtype=dtype, device=device)
        ext = self._dino_extractor
        w = ext.model.embeddings.patch_embeddings.weight
        if w.device != dino_images.device:           # lazy move (non-child object)
            ext.model.to(dino_images.device)
            w = ext.model.embeddings.patch_embeddings.weight
        # frozen DINOv3 in its own dtype, autocast OFF (conv dtype safety — see dino_align).
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            feats = ext(dino_images.to(device=w.device, dtype=w.dtype))   # (B*V, N, 1024) fp32
        M, N, _ = feats.shape
        V = max(1, M // B)
        feats = feats.reshape(B, V, N, -1)
        per_view = [feats[:, v] for v in range(V)]   # V × (B, N, 1024)
        # build_dino_anchor adds the TRAINABLE per-view embedding → keep OUTSIDE no_grad so it
        # receives gradient (DINOv3 features are frozen constants; view_embed is the variable).
        anchor = self.dual_router.build_dino_anchor(per_view)            # (B, V*N, 1024)
        return anchor.to(dtype=dtype, device=device)

    def _repa_lambda(self) -> float:
        """Scheduled REPA λ. Constant self.repa_coeff unless REPA_COEFF_HI is set, then:
        λ = hi for the first W steps, cosine-decay hi→repa_coeff over the next W, then base.
        Advances an internal step counter each TRAINING call (rank-local; only sets the aux
        weight, so cross-rank drift of a few steps is harmless)."""
        import os as _os, math
        base = float(getattr(self.config, "repa_coeff", 0.5))
        # lazy-init (attrs may be absent: they were set on the config-loaded model path but not
        # every construction route runs the model __init__ block that defines them).
        if not hasattr(self, "_repa_coeff_hi"):
            _hi = _os.environ.get("REPA_COEFF_HI", "").strip()
            self._repa_coeff_hi = float(_hi) if _hi else None
            _w = _os.environ.get("REPA_WARMUP_STEPS", "").strip()
            self._repa_warmup = int(_w) if _w else 2000
            self._repa_step = 0
        if not self.training or self._repa_coeff_hi is None:
            return base
        t, W, hi = self._repa_step, max(1, self._repa_warmup), self._repa_coeff_hi
        self._repa_step += 1
        if t < W:
            return hi
        if t < 2 * W:
            return base + (hi - base) * 0.5 * (1 + math.cos(math.pi * (t - W) / W))
        return base

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        # 3D flow targets (from the existing prep; collator attaches these)
        target_ss_latent: Optional[torch.FloatTensor] = None,
        target_shape_slat_512: Optional[Any] = None,
        target_tex_slat_512: Optional[Any] = None,
        tex_concat_cond: Optional[Any] = None,
        # (B,) bool — per-sample texture-supervision validity (option C). None means
        # "all valid" (legacy all-pbr batches). Consumed only by the geotex loss.
        tex_valid: Optional[torch.Tensor] = None,
        # DINOv3 alignment renders (collator emits when images present; aligner-gated)
        dino_images: Optional[torch.Tensor] = None,
        # cached-cond fast path (deferred feature; v1 leaves these None → run VLM)
        cond_hidden: Optional[torch.Tensor] = None,
        cond_key_mask: Optional[torch.Tensor] = None,
        # fusion (config.fuse_dino): cached frozen-DINOv3 tokens, concatenated before
        # the Qwen segment inside compute_cascade_flow_loss (single cross-attn).
        dino_hidden: Optional[torch.Tensor] = None,
        dino_keep_mask: Optional[torch.Tensor] = None,
        dino_view_ids: Optional[torch.Tensor] = None,   # (B, N_d) view ordinals (IM)
        qwen_view_ids: Optional[torch.Tensor] = None,   # (B, T_q) view ordinals for the QWEN segment (-1 = text)
        # cond_keep_mask (collator): True = real CONTENT token (caption/image_pad); False = chat-
        # template boilerplate (im_start/im_end/vision_*/role/think). ANDed into cond_key_mask so
        # the flow cross-attn ignores boilerplate → cleaner cond (esp. text→3D).
        cond_keep_mask: Optional[torch.Tensor] = None,
        # REPA SS aux targets (collator emits when REPA_ROOT is set; config.repa_root gates use):
        # repa_target (B, 4096, z_dim) VGGT voxel features, repa_weight (B,) 1/0 (0 = missing).
        repa_target: Optional[torch.Tensor] = None,
        repa_weight: Optional[torch.Tensor] = None,
        # LM-task path (vqa / grounding / text_sft): CE loss on `labels`
        labels: Optional[torch.LongTensor] = None,
        # Task router stamp from MultiTaskCollator. None / "*_to_3d" → flow path.
        _task: Optional[str] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # ── LM-loss path ────────────────────────────────────────────────
        # When the collator stamps a chat-task name, route to the VLM's own
        # LM head + CE. This is what makes joint training with VQA / grounding /
        # pure-text SFT possible without a separate trainer.
        if _task in self._LM_TASKS:
            if self.config.freeze_vlm and not getattr(self, "_lm_freeze_warned", False):
                import warnings
                warnings.warn(
                    f"[TrellisNativeVLM] _task={_task!r} requires gradient through the VLM "
                    "but config.freeze_vlm=True. The LM CE will compute but NOT update the "
                    "backbone. Set --freeze_vlm False for joint LM-loss training."
                )
                self._lm_freeze_warned = True
            if labels is None:
                raise ValueError(
                    f"TrellisNativeVLM.forward got _task={_task!r} but no `labels`. "
                    "The chat collator must emit labels with -100 outside the answer span."
                )
            vlm_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            if pixel_values is not None:
                vlm_kwargs["pixel_values"] = pixel_values
                vlm_kwargs["image_grid_thw"] = image_grid_thw
            if pixel_values_videos is not None:
                vlm_kwargs["pixel_values_videos"] = pixel_values_videos
                vlm_kwargs["video_grid_thw"] = video_grid_thw
            out = self.vlm(**vlm_kwargs)
            loss = out.loss
            rank0_print(f"[loss] task={_task} lm_ce={float(loss.detach()):.4f}")
            return CausalLMOutputWithPast(loss=loss, logits=None)

        # ── Flow / 3D path (default; covers task ∈ {text_to_3d, image_to_3d,
        # multi_image_to_3d} OR _task=None / legacy) ─────────────────────
        if target_ss_latent is None:
            raise ValueError(
                "TrellisNativeVLM.forward requires target_ss_latent for the 3D-flow path. "
                f"Got _task={_task!r} and no target_ss_latent — did the collator stamp the "
                "right task name? (LM tasks: " + ", ".join(sorted(self._LM_TASKS)) + ")"
            )

        # 1. cond: either run the VLM (v1) or use precomputed cached cond (deferred).
        depthwise = (getattr(self, "cond_fusion", "none") == "depthwise") and (cond_hidden is None)
        _mask_drop = self.config.mask_drop_prob
        if depthwise:
            # Depth-wise Semantic Routing: get all VLM layer hiddens, set each flow's per-block
            # cond (the wrappers inject it; the cond passed below is a dummy, overridden). CFG
            # dropout is off on this path in v1 (handled at the layer-stack level later).
            layer_hiddens, cond_key_mask = self.encode_cond(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw,
                return_layers=True,
            )
            self.ss_router.set_cond(layer_hiddens)
            if self.config.build_slat:
                self.shape_router.set_cond(layer_hiddens)
                self.tex_router.set_cond(layer_hiddens)
            cond_hidden = layer_hiddens[-1]   # dummy for the shared helper (per-block overrides it)
            _mask_drop = 0.0
        elif cond_hidden is None:
            cond_hidden, cond_key_mask = self.encode_cond(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        else:
            # VLM-hidden CACHE fast path (vlm_cache.py): cond arrives precomputed.
            # The cached keep_mask already encodes attention ∧ ¬boilerplate AND pads
            # False → it IS the key mask. Cache stores fp16 → cast to the connector's
            # compute dtype (bf16 under DeepSpeed) to avoid a mixed-dtype matmul.
            _pdtype = next(self.diffusion_connector.parameters()).dtype
            cond_hidden = cond_hidden.to(_pdtype)
            if cond_key_mask is None and cond_keep_mask is not None:
                cond_key_mask = cond_keep_mask.bool()
                cond_keep_mask = None   # consumed (don't AND it twice below)
            if dino_hidden is not None:
                dino_hidden = dino_hidden.to(_pdtype)

        # fusion contract: config.fuse_dino runs MUST receive the DINO segment on
        # image tasks (silently training without it = the non-fusion model).
        if getattr(self.config, "fuse_dino", False) and dino_hidden is None \
                and _task not in self._LM_TASKS and _task != "text_to_3d":
            raise ValueError("[fusion] config.fuse_dino=True but batch has no dino_hidden — "
                             "dataset fuse_dino off or DINO cache entries missing")

        # repa contract: config.repa_root runs MUST receive repa keys from the collator.
        # (repa_target=None WITH repa_weight present = a whole batch of missing targets —
        # that's the graceful aux-weight-0 path; BOTH None = the dataset isn't wired.)
        if self.repa_projector is not None and self.training \
                and repa_target is None and repa_weight is None:
            raise ValueError("[repa] config.repa_root set but batch has no repa_target/"
                             "repa_weight — REPA_ROOT env unset in the data path or the "
                             "collator isn't stacking the targets")

        # drop chat-template boilerplate from the flow cond (keep caption + image patches only)
        if cond_keep_mask is not None and cond_key_mask is not None:
            cond_key_mask = cond_key_mask & cond_keep_mask.to(cond_key_mask.device, torch.bool)

        # ── unified geo-tex path (train_stages="geotex") ─────────────────────
        # Bypasses the cascade entirely: (t_s,t_x) pair sampling + noised-geo-GT
        # concat_cond + tex-velocity loss live in compute_unified_geotex_loss.
        # Geo cond runs the FROZEN geo-run connector (deterministic, no drops);
        # tex cond runs the trained connector with the production dropout stack.
        if self.unified_geotex is not None:
            if target_shape_slat_512 is None or target_tex_slat_512 is None:
                _missing = [n for n, v in (("target_shape_slat_512", target_shape_slat_512),
                                           ("target_tex_slat_512", target_tex_slat_512))
                            if v is None]
                raise ValueError(
                    f"[geotex] batch missing {_missing} — the unified loss needs both GT "
                    "SLATs. If tex is the missing one on a partial-pbr pool, set "
                    "tex_placeholder: true in the mixture yaml (per-sample masking, "
                    "option C) rather than filtering the pool. NOTE: this raise fires "
                    "per-rank while the other ranks proceed into their next collective — "
                    "on a multi-rank run the visible symptom is an NCCL timeout HANG, "
                    "and this message sits in ONE rank's log, not the launcher's.")
            _fuse = getattr(self.config, "fuse_dino", False)
            loss, glogs = flow_heads.compute_unified_geotex_loss(
                unified_model=self.unified_geotex,
                connector_geo=self.geo_connector,
                connector_tex=self.diffusion_connector,
                loss_fn_slat=self._loss_fn_slat,
                cond_hidden=cond_hidden,
                cond_key_mask=cond_key_mask,
                target_shape_slat_512=target_shape_slat_512,
                target_tex_slat_512=target_tex_slat_512,
                tex_valid=tex_valid,
                dino_hidden=dino_hidden if _fuse else None,
                dino_key_mask=dino_keep_mask.bool()
                    if (dino_keep_mask is not None and _fuse) else None,
                dino_view_ids=dino_view_ids if _fuse else None,
                qwen_view_ids=qwen_view_ids if _fuse else None,
                dino_view_embed=self.dino_view_embed if _fuse else None,
                mask_drop_prob=_mask_drop,
                dino_drop_prob=float(getattr(self.config, "dino_drop_prob", 0.0))
                    if self.training else 0.0,
                qwen_drop_prob=float(getattr(self.config, "qwen_drop_prob", 0.0)),
                cond_max_length=self.config.cond_max_length,
                p_corner=float(self.config.geotex_p_corner),
                p_corner2=float(getattr(self.config, "geotex_p_corner2", 0.2)),
                mismatch_w=float(getattr(self.config, "geotex_mismatch_w", 0.0)),
                mismatch_margin=float(getattr(self.config, "geotex_mismatch_margin", 0.15)),
                # "unfreeze" gates the geo loss because in a WARM START geo is
                # frozen by default. From scratch nothing is frozen and the flag
                # is meaningless — but leaving the gate as-is would silently
                # train geo with NO loss of its own, shaped only by what leaks
                # back through the tex loss. That is precisely the failure the
                # geo term exists to prevent (see flow_heads: "geo would be
                # optimized to SERVE tex"), and it would look like a healthy run.
                geo_loss_w=(float(getattr(self.config, "geotex_geo_loss_w", 0.0))
                            if (getattr(self.config, "geotex_unfreeze_geo", False)
                                or getattr(self.config, "geotex_from_scratch", False))
                            else 0.0),
                joint_cond_drop=bool(getattr(self.config, "geotex_joint_cond_drop", False)),
            )
            loss = loss * float(self.config.flow_weight)
            self._last_diag = {f"per_stage/{k}": float(v) for k, v in glogs.items()}
            return CausalLMOutputWithPast(loss=loss, logits=None)

        # 1b. dual-branch: build the DINOv3 anchor (image/multi-image) or a null token (text)
        # and stash it on the router so each SS block's ORIGINAL cross-attn gets DINOv3 while
        # the cascade's connector(cond) rides through as the Qwen branch (see dual_cond.py).
        if self.dual_router is not None:
            anchor = self._build_dino_anchor(
                dino_images, B=cond_hidden.shape[0],
                dtype=cond_hidden.dtype, device=cond_hidden.device,
            )
            # v2 COORDINATED dual-cond CFG dropout (TRELLIS uses p_uncond=0.1; we have two conds
            # so we sample ONE mode per step, homogeneous batch). Modes (image/multi-image task):
            #   • JOINT drop  (prob cfg_joint_drop_prob): anchor→null AND qwen→null (mask_drop=1)
            #       = clean unconditional → TRELLIS-aligned CFG (fixes image seed-variance).
            #   • ANCHOR-only (prob anchor_drop_prob): anchor→null, qwen kept → forces the Qwen
            #       branch to learn (fixes gate≈0 → text→3D).
            #   • else: both on (normal conditioning).
            # Text task (anchor already null): JOINT prob → drop qwen (its CFG uncond), else on.
            # mask_drop=1.0 → mask_drop() zeros cond_hidden → connector(0) == the null_cond_like
            # the inference CFG negative uses (consistent). Training-only.
            if self.training:
                _pj = float(getattr(self.config, "cfg_joint_drop_prob", 0.0))
                _pa = float(getattr(self.config, "anchor_drop_prob", 0.0))
                _is_img = dino_images is not None and dino_images.numel() > 0
                _u = float(torch.rand(()))
                if _is_img:
                    if _u < _pj:                      # joint uncond
                        anchor = torch.zeros(anchor.shape[0], 1, anchor.shape[-1],
                                             dtype=anchor.dtype, device=anchor.device)
                        _mask_drop = 1.0
                    elif _u < _pj + _pa:              # anchor-only → force Qwen
                        anchor = torch.zeros(anchor.shape[0], 1, anchor.shape[-1],
                                             dtype=anchor.dtype, device=anchor.device)
                        _mask_drop = 0.0
                    else:                             # both on
                        _mask_drop = 0.0
                else:                                 # text: anchor already null
                    _mask_drop = 1.0 if _u < _pj else 0.0
            # text task (no image → null anchor): the Qwen branch is the SOLE cond, so blocks
            # bypass the gate (gate=1). Keyed on dino_images, NOT on whether the anchor was
            # CFG-dropped — an image task with a dropped anchor still uses the learned gate.
            self.dual_router._text_mode = not (dino_images is not None and dino_images.numel() > 0)
            self.dual_router.set_dino(anchor)

        # 1c. V3 distillation teacher cond (docs/V3_DISTILL_DESIGN.md): frozen DINOv3 features
        # of the SAME conditioning view → the SAME frozen flow inside the loss fn becomes the
        # teacher. Stage-1 data is I1-only by design, so KD requires exactly 1 view/sample;
        # multi-view batches skip KD (warn once). text batches (no dino_images) → no teacher.
        teacher_cond = None
        if (self.training and getattr(self.config, "distill_dino", False)
                and dino_images is not None and dino_images.numel() > 0):
            if dino_images.shape[0] == cond_hidden.shape[0]:      # 1 view per sample (I1)
                _ext = self._dino_extractor
                _w = _ext.model.embeddings.patch_embeddings.weight
                if _w.device != dino_images.device:               # lazy move (non-child object)
                    _ext.model.to(dino_images.device)
                    _w = _ext.model.embeddings.patch_embeddings.weight
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                    teacher_cond = _ext(dino_images.to(device=_w.device, dtype=_w.dtype))
                teacher_cond = teacher_cond.to(cond_hidden.device)   # (B, 1029, 1024)
            elif not getattr(self, "_warned_im_kd", False):
                rank0_print("[distill] multi-view batch under distill_dino → KD skipped "
                            "(Stage-1 is I1-only; IM belongs to Stage-3 where KD is off)")
                self._warned_im_kd = True

        # OPTION-C GUARD: the cascade has no per-sample mask API — "the tex target
        # exists" IS its on-switch (flow_heads:321, blip3o_qwen:217). A tex_placeholder
        # dataset feeding this path would silently train the texture stage to predict
        # noise-minus-zeros on every pbr-less sample, with healthy-looking curves. Fail
        # loudly instead: placeholders are geotex-only by contract (threed.py docstring).
        if tex_valid is not None and not bool(tex_valid.all()):
            raise ValueError(
                "[cascade] batch carries tex_valid with invalid samples, i.e. placeholder "
                "texture targets — the cascade cannot mask per sample and would train on "
                "the zeros. tex_placeholder: true is only valid for train_stages=geotex.")
        # 2. cond → connector → 3-stage TRELLIS cascade flow loss (shared helper).
        loss, logs = flow_heads.compute_cascade_flow_loss(
            connector=self.diffusion_connector,
            ss_flow=self.ss_flow,
            shape_slat=self.shape_slat_512,
            tex_slat=self.tex_slat_512,
            loss_fn_ss=self._loss_fn_ss,
            loss_fn_slat=self._loss_fn_slat,
            cond_hidden=cond_hidden,
            cond_key_mask=cond_key_mask,
            target_ss_latent=target_ss_latent,
            target_shape_slat_512=target_shape_slat_512,
            target_tex_slat_512=target_tex_slat_512,
            tex_concat_cond=tex_concat_cond,
            cond_max_length=self.config.cond_max_length,
            detach_cond=self.config.detach_cond,
            mask_drop_prob=_mask_drop,
            flow_stage_weights=self.config.flow_stage_weights,
            flow_weight=self.config.flow_weight,
            teacher_cond=teacher_cond,
            kd_v_weight=float(getattr(self.config, "distill_v_weight", 1.0)),
            kd_f_weight=float(getattr(self.config, "distill_f_weight", 0.5)),
            kd_f_blocks=str(getattr(self.config, "distill_f_blocks", "auto5")),
            kd_cfg_lo=float(getattr(self.config, "distill_cfg_lo", 3.0)),
            kd_cfg_hi=float(getattr(self.config, "distill_cfg_hi", 0.0)),
            kd_cfg_null_grad=bool(getattr(self.config, "distill_cfg_null_grad", False)),
            # fusion is config-gated BOTH ways: config on + no dino_hidden raises above;
            # dino_hidden present + config off is ignored here (not silently fused).
            dino_hidden=dino_hidden if getattr(self.config, "fuse_dino", False) else None,
            dino_key_mask=dino_keep_mask.bool()
                if (dino_keep_mask is not None and getattr(self.config, "fuse_dino", False))
                else None,
            qwen_drop_prob=float(getattr(self.config, "qwen_drop_prob", 0.0)),
            dino_drop_prob=float(getattr(self.config, "dino_drop_prob", 0.0))
                if self.training else 0.0,
            dino_view_ids=dino_view_ids if getattr(self.config, "fuse_dino", False) else None,
            qwen_view_ids=qwen_view_ids if getattr(self.config, "fuse_dino", False) else None,
            dino_view_embed=self.dino_view_embed
                if getattr(self.config, "fuse_dino", False) else None,
            # REPA SS aux (repa.py): projector + hook-stash owned by this model; the
            # helper pops the stash after the SS forward and adds repa_coeff·proj_loss.
            # TRAINING-ONLY (the hook is training-gated): eval forwards pass None →
            # pure flow loss, no stale-stash errors.
            repa_projector=self.repa_projector if self.training else None,
            repa_stash=self._repa_stash if self.training else None,
            repa_target=repa_target if (self.repa_projector is not None and self.training)
                else None,
            repa_sample_weight=repa_weight
                if (self.repa_projector is not None and self.training) else None,
            repa_coeff=self._repa_lambda(),
        )

        if depthwise:   # release stashed per-block conds
            self.ss_router.clear()
            if self.config.build_slat:
                self.shape_router.clear(); self.tex_router.clear()
        if self.dual_router is not None:   # release stashed DINOv3 anchor
            self.dual_router.clear()

        # 2b. DINOv3 conditioning alignment (REPA-inspired aux loss; image tasks only).
        # Distill DINOv3(render) into the connector via a train-only projection head,
        # so the frozen flow cross-attn gets in-distribution cond. text→3D has no
        # image_pad tokens → the aligner returns 0 (pure flow gradient there). See
        # DINO_ALIGNMENT_DESIGN.md.
        # IMPORTANT: call the aligner on EVERY step when enabled — do NOT gate on
        # `dino_images is not None`. The aligner internally returns _zero_touch (a
        # 0-loss that still routes connector→head.forward) when there are no images.
        # If we gated on dino_images, text steps would skip the dino path entirely →
        # the head + dino-connector path leave the autograd graph → the graph STRUCTURE
        # changes between image and text steps → DeepSpeed ZeRO-2 deadlocks at the
        # gradient reduction on the image↔text transition (reproduced locally: image-only
        # OK, text-only OK, MIXED hangs at step 2). Always-call keeps the graph consistent.
        align_val = 0.0
        if getattr(self, "dino_aligner", None) is not None and input_ids is not None:
            cond_clean = self.diffusion_connector(cond_hidden)   # no mask_drop for alignment
            L_align = self.dino_aligner(
                cond=cond_clean, input_ids=input_ids,
                image_grid_thw=image_grid_thw, dino_images=dino_images,  # may be None → _zero_touch
                image_token_id=self.config.dino_image_token_id,
                merge_size=self.config.dino_merge_size,
            )
            loss = loss + self.config.dino_align_weight * L_align
            align_val = float(L_align.detach())

        # Stash THIS rank's per-component flow loss (ss / shape / tex) for this step. The
        # NativeTrainer accumulates these over the logging window and all-reduces across GPUs
        # at log() time, so wandb's per_stage/* matches train/loss exactly (same window-mean,
        # same cross-GPU mean) instead of being a single rank-0 sample. Nothing else logged
        # per-step (dropped voxel counts / cond-shape stats — no trend, just noise).
        self._last_diag = {f"per_stage/loss_{k}": float(logs["stages"][k])
                           for k in logs["stages"]}
        # V3 distill: kd curves = the direct "functional distance to the DINOv3-conditioned
        # flow" per stage. Absent on KD-skipped steps (CFG-dropped / text / multi-view).
        for _kd_key in ("kd_v", "kd_f"):
            if _kd_key in logs:
                for _stg, _val in logs[_kd_key].items():
                    self._last_diag[f"distill/{_kd_key}_{_stg}"] = float(_val)
        if getattr(self, "dino_aligner", None) is not None:
            self._last_diag["dino/align_loss"] = align_val
        if self.dual_router is not None:
            # mean |gate| across all dual-wrapped blocks (SS + SLAT) — anchor-lean diagnostic. Each
            # block's gate is a scalar strength (0 at init; grows as the Qwen branch learns). Compare
            # image vs text steps: low on image = leaning on the DINOv3 anchor; high on text = Qwen carries it.
            flows = [self.ss_flow]
            if self.config.build_slat:
                flows += [self.shape_slat_512, self.tex_slat_512]
            gates = [abs(getattr(b, "_last_gate", 0.0)) for f in flows for b in f.blocks]
            self._last_diag["dual/gate_abs"] = float(sum(gates) / max(1, len(gates)))

        # No logits (no LM head / no CE) — loss-only output for HF Trainer.
        return CausalLMOutputWithPast(loss=loss, logits=None)


# TODO(phaseN): register with Auto classes once save/load story is finalized.
# from transformers import AutoConfig, AutoModel
# AutoConfig.register("trellis_native_vlm", TrellisNativeVLMConfig)
# AutoModel.register(TrellisNativeVLMConfig, TrellisNativeVLMForConditionalGeneration)
