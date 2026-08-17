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
    qwen_drop_prob: float = 0.0,                   # mirror of dino_drop: qwen segment masked
    dino_drop_prob: float = 0.0,                   # DINO-dropout curriculum (anti rich-get-richer):
                                                   # per-sample, mask the WHOLE DINO segment OFF
                                                   # (segment absent — the text→3D/no-DINO regime)
    dino_view_ids: Optional[torch.Tensor] = None,  # (B, N_d) view ordinals (IM identity)
    qwen_view_ids: Optional[torch.Tensor] = None,  # (B, T_q) qwen-segment view ordinals; -1 = no code
    dino_view_embed: Optional[torch.Tensor] = None,  # (V_max, 1024) ZERO-INIT learned param
    # ── REPA-style SS auxiliary alignment (repa.py; official sihyun-yu/REPA recipe) ──
    # The MODEL owns the projector + a forward hook on ss_flow.blocks[depth-1] that
    # writes the block output into repa_stash during the SS forward; we read it here
    # AFTER loss_fn_ss ran the flow, project, and cosine-align to repa_target.
    repa_projector: Optional[Any] = None,          # 3-layer MLP (trainable, ckpt-saved)
    repa_stash: Optional[Dict] = None,             # {"h": (B, 4096, C)} written by the hook
    repa_target: Optional[torch.Tensor] = None,    # (B, 4096, z_dim); None = all missing
    repa_sample_weight: Optional[torch.Tensor] = None,  # (B,) 1/0 — 0 = target missing
    repa_coeff: float = 0.5,
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

    # REPA active = the model built a projector + hook (config.repa_root). We track the
    # per-sample CFG drop mask on every path so dropped samples get aux weight 0 (our
    # deliberate deviation from official REPA — with the cond zeroed, the model cannot
    # satisfy alignment at high t from pure noise).
    repa_active = repa_projector is not None and repa_stash is not None
    if repa_active and teacher_cond is not None:
        raise ValueError("[repa] REPA + V3 distill (teacher_cond) not supported — the KD "
                         "teacher passes would also fire the SS block hook.")
    drop_mask = None   # (B,) bool — per-sample CFG drop (explicit-drop paths only)

    # 3. project to TRELLIS cond space (+ CFG dropout).
    if dino_hidden is not None:
        # ── fusion path. CFG dropout must hit BOTH segments with the SAME per-sample
        # decision (uncond = [zeros(DINO); connector(0)], matching BOTH-CFG inference),
        # so use an explicit drop mask instead of mask_drop's internal one.
        assert teacher_cond is None, "fusion + KD not supported (KD path archived)"
        B = cond_hidden.shape[0]
        drop = (torch.rand(B, device=cond_hidden.device) < mask_drop_prob)
        drop_mask = drop
        keep = (~drop).to(cond_hidden.dtype).view(B, 1, 1)
        cond_q = connector(cond_hidden * keep, key_mask=cond_key_mask)   # (B, T_q, 1024)
        # QWEN-segment view identity — the mirror of the DINO one, and for the same reason:
        # the flow's cross-attention has no positional encoding on its keys, so it reads the
        # cond as an unordered set. In the IM case the qwen tokens are ONE joint forward over
        # all 4 images, so without this the flow cannot tell which image a qwen token came
        # from (the <|vision_*|> delimiters are masked out, and would be unusable anyway).
        # SAME buffer as the DINO segment on purpose: one code ⇒ the flow learns a single
        # view detector that serves both segments. Applied AFTER the connector (its output
        # LayerNorm puts qwen at the same ~32 magnitude as DINO, so the one calibrated
        # scale transfers) and BEFORE the CFG zeroing, so the uncond pass stays plain.
        if dino_view_embed is not None and qwen_view_ids is not None:
            qm = qwen_view_ids.clamp_min(0)                      # -1 (text/pad) → row 0 …
            add = dino_view_embed[qm].to(cond_q.dtype)
            add = add * (qwen_view_ids >= 0).unsqueeze(-1).to(cond_q.dtype)   # … then zeroed
            cond_q = cond_q + add
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
        ddrop = torch.zeros(B, dtype=torch.bool, device=dino_hidden.device)
        if dino_drop_prob > 0:
            ddrop = (torch.rand(B, device=dino_hidden.device) < dino_drop_prob)
            dmask = dmask & ~ddrop[:, None]
        # QWEN-dropout: the mirror of DINO-dropout. Without it the flow can escape to the
        # qwen segment whenever the DINO side gets harder to read (measured: a strong view
        # embed collapsed DINO's attention share 0.72→0.22 and the multi-view gain with it).
        # MUTUALLY EXCLUSIVE with the DINO drop — dropping both would leave NO conditioning,
        # which is the CFG drop's job, not a modality-robustness signal. The text task never
        # reaches this branch (dino_hidden is None → qwen-only cond is never dropped there).
        if qwen_drop_prob > 0:
            qdrop = (torch.rand(B, device=cond_q.device) < qwen_drop_prob) & ~ddrop
            cond_key_mask = cond_key_mask & ~qdrop[:, None]
        cond = torch.cat([dino_seg, cond_q], dim=1)
        cond_key_mask = torch.cat([dmask, cond_key_mask], dim=1)
        kd_active = False
    elif teacher_cond is None:
        if repa_active:
            # REPA needs the per-sample drop mask → replicate mask_drop with an EXPLICIT
            # mask (same convention as the fusion/distill paths). Non-REPA runs keep the
            # original mask_drop call below, so their RNG stream / behavior is untouched.
            B = cond_hidden.shape[0]
            drop_mask = (torch.rand(B, device=cond_hidden.device) < mask_drop_prob)
            keep = (~drop_mask).to(cond_hidden.dtype).view(B, *([1] * (cond_hidden.dim() - 1)))
            cond = connector(cond_hidden * keep, key_mask=cond_key_mask)
        else:
            cond = connector(mask_drop(cond_hidden, mask_drop_prob), key_mask=cond_key_mask)
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
        drop_mask = drop
        keep = (~drop).to(cond_hidden.dtype).view(B, *([1] * (cond_hidden.dim() - 1)))
        cond = connector(cond_hidden * keep, key_mask=cond_key_mask)
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

    # ── REPA aux (repa.py): the model's forward hook stashed ss_flow.blocks[depth-1]'s
    # output (B, 4096, C) during the SS forward above — token order is the C-order raster
    # flatten of the 16^3 grid (sparse_structure_flow.py forward: h = x.view(B, C, -1)
    # .permute(0, 2, 1)), matching the target raster. Project → negative cosine vs the
    # per-sample target, CFG-dropped / target-missing samples weighted 0. Applied at ALL
    # t uniformly (official REPA). Pop the stash either way (frees the activation ref).
    repa_loss = None
    if repa_active:
        h = repa_stash.pop("h", None)
        if "ss" not in flow_losses:
            raise ValueError("[repa] configured but the SS stage did not run — REPA is an "
                             "SS-flow aux (train_stages must include ss and the batch must "
                             "carry target_ss_latent).")
        if h is None:
            raise RuntimeError("[repa] SS block hook stashed nothing — hook not installed "
                               "on this ss_flow (torch.compile wrapping?) or model not in "
                               "training mode.")
        from .repa import repa_cosine_loss
        _pdt = next(repa_projector.parameters()).dtype
        z_tilde = repa_projector(h.to(_pdt))                       # (B, 4096, z_dim)
        w = repa_sample_weight
        if drop_mask is not None:
            w = (torch.ones(z_tilde.shape[0], device=z_tilde.device) if w is None
                 else w.to(z_tilde.device).float())
            w = w * (~drop_mask).float()
        if repa_target is None:
            # whole batch missing targets (collator emitted weight=0s) → exact-0 aux that
            # still routes grad through the projector (graph consistent for DDP/ZeRO).
            repa_loss = z_tilde.float().sum() * 0.0
        else:
            repa_loss = repa_cosine_loss(z_tilde, repa_target, w)
        stage_logs["repa"] = {"flow_mse": float(repa_loss.detach())}

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
    total = flow_weight * flow_combined
    if repa_loss is not None:
        # official REPA: total = flow_loss + repa_coeff * proj_loss (outside flow_weight
        # and the stage-weight normalization — repa is an aux, not a cascade stage).
        total = total + repa_coeff * repa_loss
    return total, logs


# ═════════════════════════════════════════════════════════════════════════════
# Unified Geo-Tex DiT training path (docs/UNIFIED_GEOTEX_DIT_DESIGN.md, P3)
# ═════════════════════════════════════════════════════════════════════════════

def build_unified_cond(
    connector,
    cond_hidden: torch.Tensor,
    cond_key_mask: torch.Tensor,
    *,
    mask_drop_prob: float = 0.0,
    dino_hidden: Optional[torch.Tensor] = None,
    dino_key_mask: Optional[torch.Tensor] = None,
    dino_drop_prob: float = 0.0,
    qwen_drop_prob: float = 0.0,
    dino_view_ids: Optional[torch.Tensor] = None,
    qwen_view_ids: Optional[torch.Tensor] = None,
    dino_view_embed: Optional[torch.Tensor] = None,
    cond_max_length: int = 8192,
    detach_cond: bool = False,   # cascade-parity default (audit: was silently True)
):
    """Cond assembly for the UNIFIED path — a deliberate PARALLEL implementation
    of compute_cascade_flow_loss's fusion/plain branches (lines ~145-254), NOT a
    refactor: that function backs live runs and its comments repeatedly pin RNG
    stream order; extracting it would risk silently changing existing behavior.
    Drift guard: tests/test_unified_conventions.py asserts this function matches
    the original on the deterministic path (all drop probs 0).

    Unsupported here by design: teacher_cond/KD, REPA (assert below).
    Returns (cond, key_mask, sdpa_mask, drop_mask).
    """
    if cond_hidden.size(1) > cond_max_length:
        import warnings
        warnings.warn(f"[unified_cond] cond length {cond_hidden.size(1)} > "
                      f"{cond_max_length}: TRUNCATING (drops later views).")
        cond_hidden = cond_hidden[:, :cond_max_length, :]
        cond_key_mask = cond_key_mask[:, :cond_max_length]
    if detach_cond:
        cond_hidden = cond_hidden.detach()

    B = cond_hidden.shape[0]
    drop_mask = None
    if dino_hidden is not None:
        # fusion branch — mirrors compute_cascade_flow_loss exactly
        drop = (torch.rand(B, device=cond_hidden.device) < mask_drop_prob)
        drop_mask = drop
        keep = (~drop).to(cond_hidden.dtype).view(B, 1, 1)
        cond_q = connector(cond_hidden * keep, key_mask=cond_key_mask)
        if dino_view_embed is not None and qwen_view_ids is not None:
            qm = qwen_view_ids.clamp_min(0)
            add = dino_view_embed[qm].to(cond_q.dtype)
            add = add * (qwen_view_ids >= 0).unsqueeze(-1).to(cond_q.dtype)
            cond_q = cond_q + add
        if getattr(connector, "pos_stamp", None) is not None:
            from .pos_stamp import IMG_SPAN_FULL
            cond_q = connector.pos_stamp(cond_q, IMG_SPAN_FULL)
        dino_seg = dino_hidden.to(cond_q.dtype)
        if dino_view_embed is not None and dino_view_ids is not None:
            dino_seg = dino_seg + dino_view_embed[dino_view_ids].to(cond_q.dtype)
        dino_seg = dino_seg * keep
        dmask = dino_key_mask if dino_key_mask is not None else torch.ones(
            dino_hidden.shape[:2], dtype=torch.bool, device=dino_hidden.device)
        ddrop = torch.zeros(B, dtype=torch.bool, device=dino_hidden.device)
        if dino_drop_prob > 0:
            ddrop = (torch.rand(B, device=dino_hidden.device) < dino_drop_prob)
            dmask = dmask & ~ddrop[:, None]
        if qwen_drop_prob > 0:
            qdrop = (torch.rand(B, device=cond_q.device) < qwen_drop_prob) & ~ddrop
            cond_key_mask = cond_key_mask & ~qdrop[:, None]
        cond = torch.cat([dino_seg, cond_q], dim=1)
        cond_key_mask = torch.cat([dmask, cond_key_mask], dim=1)
    else:
        # plain branch (text task / no DINO)
        cond = connector(mask_drop(cond_hidden, mask_drop_prob), key_mask=cond_key_mask)
        if getattr(connector, "pos_stamp", None) is not None:
            from .pos_stamp import IMG_SPAN_FULL
            cond = connector.pos_stamp(cond, IMG_SPAN_FULL)
    sdpa_mask = cond_key_mask[:, None, None, :]
    return cond, cond_key_mask, sdpa_mask, drop_mask


def sample_timestep_pairs(
    B: int,
    device,
    p_corner: float = 0.4,      # t_s=0 corner (user 2026-08-11: tex|mesh is the
                                # flagship product mode — double MF's 0.2, which
                                # is the A1 ablation arm).
    p_corner2: float = 0.2,     # t_x=1 EXACT corner (bidir design 2026-08-11):
                                # MF's second corner resurrected as the mesh-only
                                # MARGINAL config (tex fed pure noise; diffuse at
                                # t=1 returns ε exactly — no data leakage). Under
                                # the bidirectional topology geo must learn/see
                                # "tex = no information"; also trains tex at max
                                # noise. 0 restores the single-corner scheme.
    p_band: float = 0.0,        # OFF by default. Verified reference practice
                                # (paper study jointdit_vs_modality_forcing):
                                # MF = 60% joint with BOTH t drawn independently
                                # (logit-normal mu=1.1) + 20% I2D + 20% D2I;
                                # JointDiT = 50% diagonal + 50% independent, no
                                # corners; UniDiffuser = 100% independent uniform.
                                # None of the three restricts the joint region to
                                # a lag triangle, so the default follows them.
                                # KNOWN DEVIATION from our own design doc S5,
                                # which specified 40% t_s~U[0,t_x] (lag) — that
                                # arm was displaced when corner2 was added for
                                # the bidirectional topology. Cost of the square:
                                # p_band=0 puts ~20% of the batch at t_s > t_x,
                                # which NO inference mode visits (all three stay
                                # in t_s <= t_x); only the t_s=1 edge earns its
                                # keep, as UniDiffuser's free modality-CFG uncond
                                # branch. A6 = turn this band on and A/B it; it
                                # generalizes JointDiT's diagonal (alpha=1 IS the
                                # diagonal) up to alpha=32 strong-lead.
    p_lag: float = 0.0,         # A6-primary: t_s ~ U[0, t_x] — UNIFORM over the
                                # upper triangle, i.e. over the union of ALL
                                # alpha-warp trajectories (design doc S5's
                                # original lag regime). Preferred over p_band:
                                # the band's log-uniform alpha gives density
                                # ~1/t_s (t_s ∝ 1/alpha for large alpha), piling
                                # mass next to the t_s=0 corner that already
                                # holds 40% — double-investment. Uniform covers
                                # every alpha the inference sweep may pick.
    p_marg_s: float = 0.0,      # t_s=1 EXACT (geo = pure noise), t_x ~ U[0,1]:
                                # the tex marginal, and the uncond branch of
                                # UniDiffuser's free modality-CFG for geometry
                                # adherence. The ONE tex-ahead configuration
                                # with a defined use — keep this edge, drop the
                                # rest of the lower triangle (user 2026-08-13:
                                # do not spend capacity on tex leading geo).
    alpha_lo: float = 1.0,
    alpha_hi: float = 32.0,
):
    """(t_s, t_x) sampler — MF-adapted recipe, JointDiT-style scenario switch.

    Semantics = Modality Forcing DUAL-corner (paper; repo is inference-only):
    p_corner at t_s=0 EXACT (tex|mesh config; geo↔tex reads corner-masked off
    there in the bidir model), p_corner2 at t_x=1 EXACT (mesh-only marginal —
    tex carries pure noise = "no information"), remainder independent draws over
    the full square. Per-branch density = uniform, matching OUR flows'
    pretraining t-schedule (build_flow_loss_fns: slat uniform) — the same
    follow-the-base-model logic MF applied by inheriting FLUX's shift.
    Code shape = JointDiT train.py:515-555 (scenario randint + per-sample select).
    Convention: t=0 clean, t=1 noise (all reference codebases + ours).
    """
    t_x = torch.rand(B, device=device)
    t_s = torch.rand(B, device=device)                       # joint regime default
    u = torch.rand(B, device=device)
    if p_band > 0:                                           # A6 arm only
        import math
        log_a = torch.empty(B, device=device).uniform_(
            math.log(alpha_lo), math.log(alpha_hi))          # audit: alpha_lo<1 was silently clamped
        alpha = torch.exp(log_a)
        t_s_band = t_x / (alpha - (alpha - 1.0) * t_x)       # exact f_alpha inverse
        band = (u >= p_corner + p_corner2) & (u < p_corner + p_corner2 + p_band)
        t_s = torch.where(band, t_s_band, t_s)
    # segments are laid out on u in a fixed cumulative order; anything left over
    # keeps the independent-square draw. Defaults (p_band=p_lag=p_marg_s=0)
    # reproduce the shipped S1/S2b recipe exactly.
    _c = p_corner + p_corner2 + p_band
    if p_lag > 0:                                            # A6 arm
        lag = (u >= _c) & (u < _c + p_lag)
        t_s = torch.where(lag, t_x * torch.rand(B, device=device), t_s)
    _c = _c + p_lag
    if p_marg_s > 0:                                         # tex-marginal edge
        marg = (u >= _c) & (u < _c + p_marg_s)
        t_s = torch.where(marg, torch.ones_like(t_s), t_s)
    corner2 = (u >= p_corner) & (u < p_corner + p_corner2)
    t_x = torch.where(corner2, torch.ones_like(t_x), t_x)
    t_s = torch.where(u < p_corner, torch.zeros_like(t_s), t_s)
    return t_s, t_x


def compute_unified_geotex_loss(
    *,
    unified_model,                    # UnifiedGeoTexFlow
    connector_geo,                    # frozen geo-run connector (deterministic prep)
    connector_tex,                    # trained tex-run connector
    loss_fn_slat: TRELLIS2FlowMatchingLoss,   # uniform t-schedule (build_flow_loss_fns)
    cond_hidden: torch.Tensor,
    cond_key_mask: torch.Tensor,
    target_shape_slat_512,            # sp.SparseTensor, SHAPE-norm space (geo stream space)
    target_tex_slat_512,              # sp.SparseTensor, tex_pbr-norm space
    # cond extras (same contract as the cascade path)
    dino_hidden=None, dino_key_mask=None, dino_view_ids=None, qwen_view_ids=None,
    dino_view_embed=None,
    mask_drop_prob: float = 0.1,
    dino_drop_prob: float = 0.0,
    qwen_drop_prob: float = 0.0,
    cond_max_length: int = 8192,
    # unified knobs
    p_corner: float = 0.4, p_corner2: float = 0.2, p_band: float = 0.0,
    p_lag: float = 0.0, p_marg_s: float = 0.0,
    alpha_lo: float = 1.0, alpha_hi: float = 32.0,
    # ── S2b: geo unfrozen (the "three-pack" of the design doc) ──
    geo_loss_w: float = 0.0,        # >0 turns on geo's OWN velocity loss
    geo_teacher=None,               # frozen S1-geo (a plain SLatFlowModel) for self-distill
    distill_w: float = 0.0,         # >0 turns on MF-style anti-drift distillation
    distill_lo: float = 0.1,        # weight when tex is CLEAN  (t_x->0): student may deviate
    distill_hi: float = 1.0,        # weight when tex is NOISE  (t_x->1): must match teacher
) -> Tuple[torch.Tensor, Dict]:
    """Stage-1 unified loss: sample (t_s,t_x), noise geo GT at t_s, feed its
    RENORMALIZED state as tex concat_cond (exactly what joint inference feeds —
    NOT the dataset's independently-noised tex-space copy), run the unified
    forward, tex-velocity MSE only. Geo has no loss (frozen)."""
    import torch.nn.functional as F

    B = cond_hidden.shape[0]
    dev = cond_hidden.device

    # cond ×2: geo deterministic (frozen stream needs no regularization noise),
    # tex with the production dropout stack.
    # NOTE (review finding 2026-08-11): the key masks MUST be consumed — the
    # dino/qwen drops are pure mask edits and pads are non-zero post-connector.
    # Convention = loss.py:236-242: slice per sample by mask into a LIST; the
    # flow auto-wraps it into a VarLenTensor (pads vanish; varlen cross-attn).
    def _masked_list(cond, key_mask):
        m = key_mask.bool()
        return [cond[b, m[b]] for b in range(cond.shape[0])]

    # A three-stream MMDiT has ONE cond stream, so it ignores cond_s entirely
    # (mmdit3d._prepare). Building it anyway costs a connector forward plus a
    # per-sample python mask loop every step, for a tensor nothing reads. The
    # distill teacher below is warm-start-only and unreachable in that mode.
    if getattr(unified_model, "from_scratch", False):
        cond_s = None
    else:
        with torch.no_grad():
            cond_s, key_s, _, _ = build_unified_cond(
                connector_geo, cond_hidden, cond_key_mask,
                mask_drop_prob=0.0, dino_hidden=dino_hidden, dino_key_mask=dino_key_mask,
                dino_drop_prob=0.0, qwen_drop_prob=0.0,
                dino_view_ids=dino_view_ids, qwen_view_ids=qwen_view_ids,
                dino_view_embed=dino_view_embed, cond_max_length=cond_max_length)
            cond_s = _masked_list(cond_s, key_s)
    cond_x, key_x, _, _ = build_unified_cond(
        connector_tex, cond_hidden, cond_key_mask,
        mask_drop_prob=mask_drop_prob, dino_hidden=dino_hidden,
        dino_key_mask=dino_key_mask, dino_drop_prob=dino_drop_prob,
        qwen_drop_prob=qwen_drop_prob, dino_view_ids=dino_view_ids,
        qwen_view_ids=qwen_view_ids, dino_view_embed=dino_view_embed,
        cond_max_length=cond_max_length)
    cond_x = _masked_list(cond_x, key_x)

    # timestep pair + noising (diffuse/get_v = upstream formulas, never re-derived)
    t_s, t_x = sample_timestep_pairs(B, dev, p_corner=p_corner, p_corner2=p_corner2,
                                     p_band=p_band, p_lag=p_lag, p_marg_s=p_marg_s,
                                     alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    # dtype discipline (audit; loss.py:172-177 verbatim rule): cast t to the
    # TARGET dtype — never float() — or diffuse upcasts x_t to fp32; and cast
    # targets to a common dtype like the cascade path does (:314/:322).
    x0_s = target_shape_slat_512
    x0_x = target_tex_slat_512
    common_dt = x0_x.feats.dtype
    x0_s = x0_s.replace(x0_s.feats.to(common_dt))
    t_s = t_s.to(common_dt)
    t_x = t_x.to(common_dt)
    noise_s = x0_s.replace(torch.randn_like(x0_s.feats))
    noise_x = x0_x.replace(torch.randn_like(x0_x.feats))
    x_ts = loss_fn_slat.diffuse(x0_s, t_s, noise=noise_s)
    x_tx = loss_fn_slat.diffuse(x0_x, t_x, noise=noise_x)
    v_target = loss_fn_slat.get_v(x0_x, noise_x, t_x)

    # concat_cond = the geo noisy state directly — the two norm-stat sets are
    # bit-identical (measured; guarded by tests), so no space conversion exists.
    # The joint regime covers NEAR-max t_s (exact t_s=1 has measure zero — audit;
    # if shape-adherence CFG is ever productized, pin an exact t_s=1 branch the
    # same way the t_s=0 corner is pinned).
    cc = x_ts

    # autocast wrapper: TimestepEmbedder forces fp32 t_freq into bf16 Linears —
    # loss.py:189-193 documents the exact crash; same convention here.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=x_tx.feats.is_cuda):
        v_s_pred, v_x_pred = unified_model(
            x_ts, x_tx, (t_s * 1000.0).to(x_ts.feats.dtype),
            (t_x * 1000.0).to(x_tx.feats.dtype), cond_s, cond_x,
            tex_concat_cond=cc)

    loss = F.mse_loss(v_x_pred.feats.float(), v_target.feats.float())
    logs = {"tex_flow_loss": loss.detach().item(),
            "t_s_mean": t_s.mean().item(), "t_x_mean": t_x.mean().item(),
            "corner_frac": (t_s == 0).float().mean().item(),
            "corner2_frac": (t_x == 1).float().mean().item()}

    # ── S2b term 1: geo's OWN velocity loss ─────────────────────────────────
    # Without it, geo's only gradient is whatever leaks back through the tex
    # loss — i.e. geo would be optimized to SERVE tex, and its own marginal
    # p(geo|cond) would drift. MASKED at the t_s=0 corner: there x_ts IS x0, so
    # the target v = eps - x0 is unpredictable from the input (E[v|x0] = -x0)
    # and 40% of the batch would inject pure variance. MF's corner sampling
    # likewise only supervises the modality being generated.
    if geo_loss_w > 0:
        v_s_target = loss_fn_slat.get_v(x0_s, noise_s, t_s)
        keep = (t_s != 0)
        if bool(keep.any()):
            rows = torch.cat([
                torch.ones(sl.stop - sl.start, dtype=torch.bool, device=dev) if keep[b]
                else torch.zeros(sl.stop - sl.start, dtype=torch.bool, device=dev)
                for b, sl in enumerate(x0_s.layout)])
            geo_loss = F.mse_loss(v_s_pred.feats[rows].float(),
                                  v_s_target.feats[rows].float())
            loss = loss + geo_loss_w * geo_loss
            logs["geo_flow_loss"] = geo_loss.detach().item()

    # ── S2b term 2: MF-style self-distillation (anti-drift) ─────────────────
    # Teacher = the FROZEN S1 geo, run ONE-WAY (it never learned to read tex, so
    # its function is the pretrained specialist's). Weight rises with t_x: when
    # the tex stream is pure noise it carries no information, so the student has
    # no excuse to deviate from the teacher; when tex is clean, deviation is the
    # whole point and the pull is relaxed.
    if distill_w > 0 and geo_teacher is not None:
        # There is no S1 geo to distil from in a from-scratch run, and cond_s is
        # None there — fail loudly instead of feeding the teacher a None cond.
        assert cond_s is not None, \
            "self-distillation needs the warm-start geo connector; not available from scratch"
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=x_tx.feats.is_cuda):
            v_s_teacher = geo_teacher(x_ts, (t_s * 1000.0).to(x_ts.feats.dtype), cond_s)
        w_per_sample = distill_lo + (distill_hi - distill_lo) * t_x.float()
        w_rows = torch.cat([w_per_sample[b].expand(sl.stop - sl.start)
                            for b, sl in enumerate(x0_s.layout)]).unsqueeze(-1)
        d = (v_s_pred.feats.float() - v_s_teacher.feats.float()) ** 2
        distill = (d * w_rows).mean()
        loss = loss + distill_w * distill
        logs["geo_distill"] = distill.detach().item()

    return loss, logs
