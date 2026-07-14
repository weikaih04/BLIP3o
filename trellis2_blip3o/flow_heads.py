"""Shared TRELLIS.2 cascade flow-loss head.

Factored out so BOTH `blip3o/model/language_model/blip3o_qwen.py` (discrete/TA-Tok
variant) and `trellis_native_vlm.py` (native-VLM continuous variant) use ONE
implementation of the 3-stage cascade flow loss (SS + Shape SLAT + Tex SLAT).
This avoids the two models drifting when the TRELLIS schedule / weighting changes.

SINGLE SOURCE OF TRUTH: both `blip3o_qwen.py` (discrete/TA-Tok variant) and
`trellis_native_vlm.py` (native-VLM variant) now call `compute_cascade_flow_loss()`.
blip3o_qwen's previous inline loop was migrated here and verified NUMERICALLY
IDENTICAL (tests/test_flow_heads_equiv.py: diff=0), so the migration changed no
training behavior.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from trellis2_blip3o.loss import TRELLIS2FlowMatchingLoss


def mask_drop(latents: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """Classifier-free-guidance dropout on cond (per-sample). Matches BLIP3o."""
    if drop_prob <= 0:
        return latents
    mask = torch.bernoulli(
        torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob
    )
    while len(mask.shape) < len(latents.shape):
        mask = mask.unsqueeze(-1)
    return latents * (1 - mask)


def null_cond_like(connector, cond_hidden: torch.Tensor) -> torch.Tensor:
    """The training-time UNCONDITIONAL cond — use this as the CFG negative cond.

    Training drops cond via `mask_drop`, which zeros the connector's INPUT
    (`cond_hidden`), then applies the connector. Because TRELLIS2Connector ends
    with an identity-init LayerNorm, `connector(0) != 0`. So the model's learned
    "unconditional" is `connector(0)`, NOT zeros in cond space. Feeding
    `zeros_like(cond)` (zeros AFTER the connector) gives the sampler an
    unconditional the model never saw → wrong guidance direction. Always build the
    CFG negative cond with this helper (matches `mask_drop` exactly).
    """
    return connector(torch.zeros_like(cond_hidden))


def parse_flow_stage_weights(spec: str) -> Dict[str, float]:
    """Parse 'ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0' → {'ss':1.0, ...}."""
    out: Dict[str, float] = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"flow_stage_weights entry must look like 'name=value', got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def build_flow_loss_fns(config) -> Tuple[TRELLIS2FlowMatchingLoss, TRELLIS2FlowMatchingLoss]:
    """(ss_loss_fn, slat_loss_fn) with TRELLIS.2's per-stage t-schedules.

    SS Flow → logitNormal(mean,std); SLAT (shape+tex) → uniform. Reads the same
    config knobs blip3o_qwen.py used (logitnorm_mean/std, flow_sigma_min).
    """
    ss = TRELLIS2FlowMatchingLoss(
        t_schedule="logitNormal",
        t_mean=getattr(config, "logitnorm_mean", 1.0),
        t_std=getattr(config, "logitnorm_std", 1.0),
        sigma_min=getattr(config, "flow_sigma_min", 1e-5),
    )
    slat = TRELLIS2FlowMatchingLoss(
        t_schedule="uniform",
        sigma_min=getattr(config, "flow_sigma_min", 1e-5),
    )
    return ss, slat


def compute_cascade_flow_loss(
    *,
    connector,
    ss_flow,
    shape_slat,
    tex_slat,
    loss_fn_ss: TRELLIS2FlowMatchingLoss,
    loss_fn_slat: TRELLIS2FlowMatchingLoss,
    cond_hidden: torch.Tensor,            # (B, T, vlm_dim)
    cond_key_mask: torch.Tensor,          # (B, T) bool, True = real token
    target_ss_latent: torch.Tensor,       # dense (B, C, D, H, W)
    target_shape_slat_512: Optional[Any] = None,   # sp.SparseTensor or None
    target_tex_slat_512: Optional[Any] = None,     # sp.SparseTensor or None
    tex_concat_cond: Optional[Any] = None,         # sp.SparseTensor (GT shape SLAT)
    cond_max_length: int = 8192,
    detach_cond: bool = False,
    mask_drop_prob: float = 0.1,
    flow_stage_weights: str = "ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0",
    flow_weight: float = 1.0,
    teacher_cond: Optional[torch.Tensor] = None,   # V3 distill: DINOv3 cond (B, N, 1024)
    kd_v_weight: float = 1.0,
    kd_f_weight: float = 0.5,
    kd_f_blocks: str = "auto5",
    kd_cfg_lo: float = 3.0,
    kd_cfg_hi: float = 0.0,                        # >0 → CFG-AWARE KD (Stage-1.5); see loss.py
    kd_cfg_null_grad: bool = False,
    # ── fusion: raw DINOv3 tokens concatenated BEFORE the Qwen segment (single
    # cross-attn, no new modules — docs/FUSION_DESIGN). DINO bypasses the connector:
    # its 1024-d tokens are the native distribution the pretrained cross-attn was
    # trained on. Segment order [DINO; Qwen] mirrors BOTH-CFG packed inference.
    dino_hidden: Optional[torch.Tensor] = None,    # (B, N_d, 1024) frozen DINOv3 tokens
    dino_key_mask: Optional[torch.Tensor] = None,  # (B, N_d) bool
    dino_drop_prob: float = 0.0,                   # DINO-dropout curriculum (anti rich-get-richer):
                                                   # per-sample, mask the WHOLE DINO segment OFF
                                                   # (segment absent — the text→3D/no-DINO regime)
    dino_view_ids: Optional[torch.Tensor] = None,  # (B, N_d) view ordinals (IM identity)
    dino_view_embed: Optional[torch.Tensor] = None,  # (V_max, 1024) ZERO-INIT learned param
) -> Tuple[torch.Tensor, Dict]:
    """VLM hidden → connector → TRELLIS cascade flow loss (SS + Shape + Tex).

    cond enters each flow DiT via cross-attention (cond as K/V), with `cond_key_mask`
    masking padded tokens — same path the pretrained TRELLIS cross-attn expects.
    Returns (flow_weight * weighted_mean(stage losses), logs).

    V3 distillation (docs/V3_DISTILL_DESIGN.md): when `teacher_cond` is given, each
    stage adds output- and feature-level KD vs the SAME frozen flow run on the SAME
    (x_t, t) with the DINOv3 teacher cond. CFG-dropped batches skip KD (the uncond
    direction is identical for both conds — no signal): we gate on "no sample in the
    batch was dropped", exact at the BS=1 reality of 512 training.
    """
    # 1. cap cond length (sized for ~8-frame short video). LOUD, never silent: truncation
    # drops the RIGHTMOST tokens — in multi-view batches that means whole later views
    # vanish from the conditioning. The collator (vlm_collate) budgets BELOW this cap, so
    # firing here means the two knobs have drifted — warn with the real numbers.
    if cond_hidden.size(1) > cond_max_length:
        import warnings
        warnings.warn(
            f"[flow_heads] cond length {cond_hidden.size(1)} > cond_max_length "
            f"{cond_max_length}: TRUNCATING (drops later views/content). Check the "
            f"collator token_budget vs this cap."
        )
        cond_hidden = cond_hidden[:, :cond_max_length, :]
        cond_key_mask = cond_key_mask[:, :cond_max_length]

    # 2. detach blocks flow grad into the VLM (frozen-encoder / Setup-A style).
    if detach_cond:
        cond_hidden = cond_hidden.detach()

    # 3. project to TRELLIS cond space (+ CFG dropout).
    if dino_hidden is not None:
        # ── fusion path. CFG dropout must hit BOTH segments with the SAME per-sample
        # decision (uncond = [zeros(DINO); connector(0)], matching BOTH-CFG inference),
        # so use an explicit drop mask instead of mask_drop's internal one.
        assert teacher_cond is None, "fusion + KD not supported (KD path archived)"
        B = cond_hidden.shape[0]
        drop = (torch.rand(B, device=cond_hidden.device) < mask_drop_prob)
        keep = (~drop).to(cond_hidden.dtype).view(B, 1, 1)
        cond_q = connector(cond_hidden * keep)                       # (B, T_q, 1024)
        if getattr(connector, "pos_stamp", None) is not None:        # DINO pos signature
            from .pos_stamp import IMG_SPAN_FULL
            cond_q = connector.pos_stamp(cond_q, IMG_SPAN_FULL)
        dino_seg = dino_hidden.to(cond_q.dtype)                      # (B, N_d, 1024)
        if dino_view_embed is not None and dino_view_ids is not None:
            # multi-image identity: zero-init per-ordinal embedding (grad flows to the
            # param; the frozen DINO tokens stay constants). Applied BEFORE the CFG
            # zeroing so the uncond pass sees plain zeros, same as inference neg.
            dino_seg = dino_seg + dino_view_embed[dino_view_ids].to(cond_q.dtype)
        dino_seg = dino_seg * keep
        dmask = dino_key_mask if dino_key_mask is not None else torch.ones(
            dino_hidden.shape[:2], dtype=torch.bool, device=dino_hidden.device)
        # DINO-dropout: per-sample segment ABSENT (keys masked off). Independent of the
        # CFG drop — a CFG-dropped sample keeps its (zeroed) DINO keys visible, exactly
        # like the packed inference neg pass.
        if dino_drop_prob > 0:
            ddrop = (torch.rand(B, device=dino_hidden.device) < dino_drop_prob)
            dmask = dmask & ~ddrop[:, None]
        cond = torch.cat([dino_seg, cond_q], dim=1)
        cond_key_mask = torch.cat([dmask, cond_key_mask], dim=1)
        kd_active = False
    elif teacher_cond is None:
        cond = connector(mask_drop(cond_hidden, mask_drop_prob))
        if getattr(connector, "pos_stamp", None) is not None:
            from .pos_stamp import IMG_SPAN_FULL
            cond = connector.pos_stamp(cond, IMG_SPAN_FULL)
        kd_active = False
    else:
        # Distill path: replicate mask_drop but KEEP the per-sample drop mask so we
        # can gate KD (dropped samples have no teacher signal — see docstring).
        # Existing (non-distill) runs keep the original mask_drop call above, so their
        # RNG stream / behavior is untouched.
        B = cond_hidden.shape[0]
        drop = (torch.rand(B, device=cond_hidden.device) < mask_drop_prob)
        keep = (~drop).to(cond_hidden.dtype).view(B, *([1] * (cond_hidden.dim() - 1)))
        cond = connector(cond_hidden * keep)
        kd_active = not bool(drop.any())
    sdpa_mask = cond_key_mask[:, None, None, :]   # (B,1,1,T) True = attend

    kd_kwargs = (
        dict(teacher_cond=teacher_cond, kd_v_weight=kd_v_weight,
             kd_f_weight=kd_f_weight, kd_f_blocks=kd_f_blocks)
        if (teacher_cond is not None and kd_active) else {}
    )
    if kd_kwargs and kd_cfg_hi > 0:
        # CFG-aware KD: the student's uncond = its train-time mask_drop convention,
        # connector(0-hidden) — NOT zeros in cond space (see null_cond_like docstring).
        kd_kwargs.update(
            kd_cfg_lo=kd_cfg_lo, kd_cfg_hi=kd_cfg_hi, kd_cfg_null_grad=kd_cfg_null_grad,
            student_null_cond=null_cond_like(connector, cond_hidden),
        )

    flow_losses: Dict[str, torch.Tensor] = {}
    stage_logs: Dict[str, Dict] = {}

    # Stage 1: SS Flow (dense, logitNormal t-schedule). ss_flow=None → stage-split job
    # (--train_stages shape|tex) that doesn't build/train SS.
    if ss_flow is not None and target_ss_latent is not None:
        ss_target = target_ss_latent.to(cond.dtype)
        L_ss, log_ss = loss_fn_ss(ss_flow, ss_target, cond, cond_mask=sdpa_mask, **kd_kwargs)
        flow_losses["ss"] = L_ss
        stage_logs["ss"] = log_ss

    # Stage 2: Shape SLAT (sparse, uniform t-schedule).
    if target_shape_slat_512 is not None and shape_slat is not None:
        shape_target = target_shape_slat_512.replace(target_shape_slat_512.feats.to(cond.dtype))
        L_shape, log_shape = loss_fn_slat(
            shape_slat, shape_target, cond, cond_mask=sdpa_mask, **kd_kwargs)
        flow_losses["shape_slat_512"] = L_shape
        stage_logs["shape_slat_512"] = log_shape

    # Stage 3: Tex SLAT (sparse, uniform, GT shape SLAT teacher-forced as concat_cond).
    if target_tex_slat_512 is not None and tex_concat_cond is not None and tex_slat is not None:
        tex_target = target_tex_slat_512.replace(target_tex_slat_512.feats.to(cond.dtype))
        tex_cc = tex_concat_cond.replace(tex_concat_cond.feats.to(cond.dtype))
        L_tex, log_tex = loss_fn_slat(
            tex_slat, tex_target, cond, cond_mask=sdpa_mask, concat_cond=tex_cc, **kd_kwargs
        )
        flow_losses["tex_slat_512"] = L_tex
        stage_logs["tex_slat_512"] = log_tex

    if not flow_losses:
        raise ValueError("[flow_heads] no stage produced a loss — check train_stages vs "
                         "provided targets (ss_flow/shape_slat/tex_slat all None or missing targets)")

    # 4. weighted-mean combination: L = flow_weight · Σ ŵ_i · L_i  (ŵ normalized).
    stage_weights_all = parse_flow_stage_weights(flow_stage_weights)
    active_w = {k: stage_weights_all.get(k, 1.0) for k in flow_losses}
    total_w = sum(active_w.values()) or 1.0
    normalized_w = {k: w / total_w for k, w in active_w.items()}
    flow_combined = sum(normalized_w[k] * v for k, v in flow_losses.items())

    logs = {
        "flow_combined": flow_combined.detach().float().item(),
        "stages": {k: stage_logs[k]["flow_mse"] for k in stage_logs},
    }
    # V3 distill logs: train/distill/{v,f}_{stage} — kd_v/kd_f are the direct
    # "functional distance to the DINOv3-conditioned flow" metrics.
    kd_v_logs = {k: lg["kd_v"] for k, lg in stage_logs.items() if "kd_v" in lg}
    kd_f_logs = {k: lg["kd_f"] for k, lg in stage_logs.items() if "kd_f" in lg}
    if kd_v_logs:
        logs["kd_v"] = kd_v_logs
    if kd_f_logs:
        logs["kd_f"] = kd_f_logs
    return flow_weight * flow_combined, logs
