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

import contextlib

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

def _add_patch_pos(cond_q, cond_patch_pos, qwen_img_rc):
    """Per-patch position code, BILINEARLY sampled from a PxP table at the
    token's normalised position in its own view.

    Interpolating rather than snapping to a cell is what makes this survive a
    change in tokens-per-view — which has already happened once (64 -> 256) and
    will happen again. Nearest-cell breaks in both directions: a view FINER than
    the table collides two patches onto one cell, silently losing resolution; a
    view coarser than it picks one corner of the region it covers instead of the
    region. Bilinear is the same trick a ViT uses to reuse a position embedding
    at a new input resolution.

    Two properties worth knowing, both consequences of using cell CENTRES:
      * a view at exactly PxP samples the integer grid, so the interpolation is
        an identity and the table is used verbatim — a warm start from the DINO
        signature is bit-exact at i1;
      * a view at P/2 x P/2 samples half-way, so each token gets the MEAN of the
        2x2 cells it covers, which is the correct downsample rather than a corner.

    Called from BOTH branches of build_unified_cond: fuse_dino=False takes the
    plain branch and can still carry image tokens.
    """
    if cond_patch_pos is None or qwen_img_rc is None:
        return cond_q
    rc = qwen_img_rc.to(cond_q.device).float()
    m = rc[..., 0] >= 0
    if not bool(m.any()):
        return cond_q
    n = cond_patch_pos.shape[0]
    P = int(round(n ** 0.5))
    assert P * P == n, f"position table {n} is not a square lattice"
    tab = cond_patch_pos.to(cond_q.dtype).view(P, P, -1)
    # normalised centre -> continuous table coordinate (align_corners=False)
    y = (rc[..., 0] * P - 0.5).clamp(0, P - 1)
    x = (rc[..., 1] * P - 0.5).clamp(0, P - 1)
    y0, x0 = y.floor().long(), x.floor().long()
    y1, x1 = (y0 + 1).clamp(max=P - 1), (x0 + 1).clamp(max=P - 1)
    wy, wx = (y - y0.float()).unsqueeze(-1), (x - x0.float()).unsqueeze(-1)
    wy, wx = wy.to(cond_q.dtype), wx.to(cond_q.dtype)
    add = (tab[y0, x0] * (1 - wy) * (1 - wx) + tab[y1, x0] * wy * (1 - wx)
           + tab[y0, x1] * (1 - wy) * wx + tab[y1, x1] * wy * wx)
    return cond_q + add * m.unsqueeze(-1).to(cond_q.dtype)


def build_unified_cond(
    connector,
    cond_hidden: torch.Tensor,
    cond_key_mask: torch.Tensor,
    cond_seg_embed=None,          # (2, C): [image-segment code, text-segment code]
    cond_patch_pos=None,          # (P*P, C): per-patch code on a canonical PxP lattice
    qwen_img_rc=None,             # (B, T, 2) normalised (row, col) in its own view, -1 = not an image
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
    ext_drops=None,     # (drop, ddrop, qdrop) bool (B,) — REPLAY these instead of drawing.
                        # KD teacher conds use it to see the student's realized CFG drops;
                        # None = draw internally (bit-identical to the pre-KD code).
    drops_out=None,     # dict filled with the realized drop/ddrop/qdrop masks (recording
                        # only — zero effect on RNG or outputs).
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
        if ext_drops is not None:
            drop = ext_drops[0].to(cond_hidden.device).bool()
        else:
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
        cond_q = _add_patch_pos(cond_q, cond_patch_pos, qwen_img_rc)
        if cond_seg_embed is not None:
            # SEGMENT CODE: "you are an image token" vs "you are a text token".
            # cond is cat([dino ; qwen]) fed to a cross-attn that ropes NOTHING,
            # so the tower receives one unordered bag of ~2053 keys with no marker
            # saying which half is which. The from-scratch tower gave them
            # distinct segment ids for exactly this reason (mmdit3d SEGMENTS
            # cond_dino=2, cond_qwen=3); the warm path lost it because cond moved
            # out of the joint softmax and into cross-attn.
            # Zero-init, so step 0 is bit-exact with the warm start. Added HERE,
            # before the `* keep` below, so a CFG-dropped row stays all-zero: the
            # uncond must carry no information, structural included.
            # MAGNITUDE IS NOT FREE — the view embed had to be scaled to ~15% of
            # the DINO token norm because at 0.7x it drowned the content and the
            # flow fled to the qwen segment (DINO attention share 0.72 -> 0.22).
            # Zero-init lets training find the scale instead of us guessing it.
            dino_seg = dino_seg + cond_seg_embed[0].to(cond_q.dtype)
            cond_q = cond_q + cond_seg_embed[1].to(cond_q.dtype)
        dino_seg = dino_seg * keep
        dmask = dino_key_mask if dino_key_mask is not None else torch.ones(
            dino_hidden.shape[:2], dtype=torch.bool, device=dino_hidden.device)
        ddrop = torch.zeros(B, dtype=torch.bool, device=dino_hidden.device)
        qdrop = torch.zeros(B, dtype=torch.bool, device=cond_q.device)
        if ext_drops is not None:
            ddrop = ext_drops[1].to(dino_hidden.device).bool()
            dmask = dmask & ~ddrop[:, None]
            qdrop = ext_drops[2].to(cond_q.device).bool() & ~ddrop
            cond_key_mask = cond_key_mask & ~qdrop[:, None]
        else:
            if dino_drop_prob > 0:
                ddrop = (torch.rand(B, device=dino_hidden.device) < dino_drop_prob)
                dmask = dmask & ~ddrop[:, None]
            if qwen_drop_prob > 0:
                qdrop = (torch.rand(B, device=cond_q.device) < qwen_drop_prob) & ~ddrop
                cond_key_mask = cond_key_mask & ~qdrop[:, None]
        if drops_out is not None:
            drops_out.update(drop=drop, ddrop=ddrop, qdrop=qdrop)
        cond = torch.cat([dino_seg, cond_q], dim=1)
        cond_key_mask = torch.cat([dmask, cond_key_mask], dim=1)
    else:
        # plain branch (text task / no DINO)
        if ext_drops is not None:
            _keep = (~ext_drops[0].to(cond_hidden.device).bool()).to(cond_hidden.dtype)
            cond = connector(cond_hidden * _keep.view(B, 1, 1), key_mask=cond_key_mask)
        elif drops_out is not None and mask_drop_prob > 0:
            # explicit draw so the mask can be recorded; torch.bernoulli, identical
            # RNG consumption and semantics to mask_drop (flow_heads.py:23-33).
            _dm = torch.bernoulli(torch.zeros(
                B, device=cond_hidden.device, dtype=cond_hidden.dtype) + mask_drop_prob)
            drop_mask = _dm.bool()
            cond = connector(cond_hidden * (1.0 - _dm).view(B, 1, 1),
                             key_mask=cond_key_mask)
        else:
            cond = connector(mask_drop(cond_hidden, mask_drop_prob), key_mask=cond_key_mask)
        cond = _add_patch_pos(cond, cond_patch_pos, qwen_img_rc)
        if cond_seg_embed is not None:
            cond = cond + cond_seg_embed[1].to(cond.dtype)   # text-segment code
        if drops_out is not None:
            _z = torch.zeros(B, dtype=torch.bool, device=cond_hidden.device)
            drops_out.update(drop=drop_mask if drop_mask is not None else _z,
                             ddrop=_z, qdrop=_z)
        if getattr(connector, "pos_stamp", None) is not None:
            from .pos_stamp import IMG_SPAN_FULL
            cond = connector.pos_stamp(cond, IMG_SPAN_FULL)
    sdpa_mask = cond_key_mask[:, None, None, :]
    return cond, cond_key_mask, sdpa_mask, drop_mask


def sample_timestep_pairs(B: int, device, p_corner: float = 0.2,
                          p_corner2: float = 0.2):
    """Draw (t_s, t_x) — the geometry and texture noise levels. t=0 clean, t=1 noise.

    TWO KNOBS, because the space has exactly two degenerate edges and they are
    exactly the two inference modes we ship:

        t_s = 0   geometry given, texture moving   ->  texture | given mesh
        t_x = 1   texture is pure noise            ->  mesh-only, and the whole
                                                       geometry leg of joint
        interior  both moving                      ->  joint's intermediate states

    `t_s <= t_x` (geometry never noisier than texture) is a STRUCTURAL INVARIANT,
    not a parameter: it is the default draw, and both edges satisfy it. Every
    inference mode keeps t_s <= t_x, so the lower triangle is a region that is
    never visited.

    HISTORY, because this cost a 60K run. The joint regime used to be the
    independent SQUARE, with an opt-in `p_lag` slice carving a triangle out of
    it, plus `p_band` (a log-uniform alpha band) and `p_marg_s` (t_s=1, which
    VIOLATES the invariant) — five probabilities laid out cumulatively on one
    uniform draw with an implicit sixth "remainder" regime. The triangle then
    held only if the probabilities happened to sum to 1. The 2026-08-16 run
    missed it: 33% of every geometry-supervised sample landed at t_s > t_x, with
    nothing in the log to say so. All four extra knobs are deleted; the
    invariant is now structural.

    THE DRAW IS NOT UNIFORM ON THE TRIANGLE, and that is deliberate. Sorting two
    uniforms (t_s=min, t_x=max) gives the triangle its uniform measure, but it
    also makes t_s the MINIMUM — E[t_s] 0.333 and only 1.0% of samples above 0.9.
    The geometry stream would then spend 99% of training in the low-noise regime,
    where its own x_s already carries the answer and the image is unnecessary.
    Meanwhile t_x is the MAXIMUM: 19% above 0.9, where the image is the only
    source. Measured on the 2026-08-18 probe, that is exactly what the two
    streams learned — texture reached cond-sensitivity 1.76 in 2000 steps while
    geometry sat at 1.01 after 5700. Same attention, same cond, same rope; the
    only difference was the noise each one was fed. (It also rules RoPE out: the
    one-sided rotation applies identically to both streams, so it cannot explain
    an asymmetry this large.)

    Drawing t_s ~ U[0,1] FIRST and then t_x ~ U[t_s, 1] keeps t_s <= t_x while
    restoring t_s's uniform marginal: 10% of samples above 0.9 instead of 1%, a
    10x increase in the regime the rollout starts from. t_x only gets noisier
    (33% above 0.9), so the texture stream loses nothing.

    Defaults 0.2 / 0.2 measured over 600k draws: geometry supervised on 80% of
    samples (the t_s=0 edge gives it no loss), 25% at t_x exactly 1 and 14% in
    the band (0.9, 1.0). That band, not the edge, is where the alpha=32 rollout
    spends 10 of its 12 steps, and raising p_corner2 shrinks it.
    """
    assert p_corner + p_corner2 <= 1.0 + 1e-6, (
        f"p_corner {p_corner} + p_corner2 {p_corner2} > 1")
    t_s = torch.rand(B, device=device)                  # uniform marginal
    t_x = t_s + (1.0 - t_s) * torch.rand(B, device=device)   # >= t_s by construction
    u = torch.rand(B, device=device)
    edge_s = u < p_corner                                    # t_s = 0 edge
    edge_x = (u >= p_corner) & (u < p_corner + p_corner2)    # t_x = 1 edge
    # pin the two edges; anything not claimed keeps the triangle interior
    t_x = torch.where(edge_x, torch.ones_like(t_x), t_x)
    # The t_s=0 edge needs t_x REDRAWN, not inherited. t_x above is the max of two
    # uniforms, so keeping it here leaves the edge with density -ln(1-t_x): 3.5x
    # over-sampled at t_x=0.95 and 0.24x at t_x=0.21. That edge IS the shipped
    # texture|given-mesh rollout (and, with refine_tex, half of joint's tex
    # forwards), whose last step covers 21% of the trajectory and was getting
    # 2.5% of the samples. Once t_s is 0 the t_x >= t_s constraint is vacuous, so
    # the correct marginal is the uniform official trains on
    # (configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json: "uniform").
    t_x = torch.where(edge_s, torch.rand(B, device=device), t_x)
    t_s = torch.where(edge_s, torch.zeros_like(t_s), t_s)
    return t_s, t_x


# ── v10: the triple (t_ss, t_s, t_x) ────────────────────────────────────────
# Row CLASSES, not corners. A corner pins one coordinate of one draw; a class
# decides which of three OPERATING REGIMES the row rehearses, and the three
# regimes need different pins, different masks and different reads:
#
#   solo   SS alone       t_ss ~ logitNormal(1,1) (the official SS schedule),
#                         slat pinned to pure noise and its losses masked.
#                         Trains SS's own marginal, so SS-only sampling stays
#                         exactly the specialist's.
#   clean  SS finished    t_ss = 0 EXACTLY (the raw latent, not diffused),
#                         (t_s, t_x) from sample_timestep_pairs UNCHANGED.
#                         This is leader-first inference, i.e. the flagship,
#                         and keeping the inner rule bit-identical is what
#                         makes the v9 arm a valid control.
#   lag    both moving    (t_ss, t_s) walk the node-pairing curve family; the
#                         only class where SS may read the slat lanes.
#
# p_solo = p_lag = 0 collapses the whole thing back to v9 exactly — the retreat
# path, and the reason the clean class calls the v9 function rather than
# reimplementing it.
CLS_CLEAN, CLS_SOLO, CLS_LAG = 0, 1, 2


def _shift_grid(rescale_t: float, u: torch.Tensor) -> torch.Tensor:
    """flow_euler's node transform t' = r·u / (1 + (r−1)·u), as a CONTINUOUS map.

    geotex_sampler._t_seq is this same function sampled at u = 1 − j/steps. Using
    the continuous form lets the lag class draw anywhere on the curve instead of
    only at the 12 released nodes, which matters because training sees far more
    rows than a trajectory has steps.
    """
    return rescale_t * u / (1.0 + (rescale_t - 1.0) * u)


def sample_timestep_triples(B: int, device,
                            p_corner: float = 0.1, p_corner2: float = 0.2,
                            p_solo: float = 0.20, p_lag: float = 0.20,
                            k0_lo: int = 3, k0_hi: int = 11,
                            ss_logit_mean: float = 1.0, ss_logit_std: float = 1.0):
    """Draw (t_ss, t_s, t_x) + the row class. Invariant t_ss <= t_s <= t_x.

    THE INVARIANT IS STRUCTURAL. Every class constructs its triple by a formula
    that cannot violate it; nothing is rejected or re-drawn. This is the direct
    lesson of the 2026-08-16 run, where five cumulative probabilities had to sum
    to 1 for `t_s <= t_x` to hold, they did not, and 33% of geometry-supervised
    rows trained at t_s > t_x with nothing in the log to say so.

    RNG DISCIPLINE: every primitive is drawn UNCONDITIONALLY for every row and
    the classes are composed with torch.where. Class-dependent draw counts would
    (a) make a fixed seed non-reproducible across knob changes, and (b) correlate
    every downstream noise draw with the class — and, across ranks whose batches
    drew different class mixes, desynchronise the RNG streams entirely.

    LAG PAIRING. Both released grids are the same shift map on a uniform node
    coordinate: node j of grid r is _shift_grid(r, 1 − j/steps). Pairing "slat
    step j with SS node j+k0" is therefore the curve family

        t_s  = _shift_grid(3.0, u)                       (SHAPE_PARAMS rescale_t)
        t_ss = _shift_grid(5.0, max(u − k0/12, 0))       (SS_PARAMS rescale_t)

    Swept at 10001 points: max(t_ss − t_s) = +0.127 at k0=0, +0.0328 at k0=1, and
    exactly 0 for every k0 >= 2. k0_lo=3 therefore keeps a full node of margin
    and needs no clamping. Sampling the FAMILY (k0 ~ U{k0_lo..k0_hi}) rather than
    filling the (t_ss, t_s) triangle uniformly means one training run covers
    every spawn point B-lite through B-full might want, so k0 stays an inference
    knob instead of a retraining decision.

    u_lag IS DRAWN ABOVE k0/steps, NOT ON [0,1]. The discrete pairing is "slat
    step j with SS node j+k0", which only exists for j <= steps-k0, i.e.
    u = 1 - j/steps >= k0/steps. Drawing u on [0,1] and clamping instead puts
    P(u <= k0/steps) = E[k0]/steps of the class at t_ss EXACTLY 0 — at
    k0 ~ U{3..12} that is 62.5% of the lag class, which then (a) is not lag at
    all, and (b) carries a (t_s, t_x) law that is NOT the v9 one, quietly
    breaking the clean class's standing as a bit-identical v9 control. k0_hi is
    steps-1 for the same reason: k0 = steps leaves no slat step to pair.

    t_x for the lag class is RE-DERIVED above t_s_lag. Reusing the v9 draw would
    silently break t_s <= t_x, because that draw was taken above the v9 t_s.
    """
    assert 0.0 <= p_solo and 0.0 <= p_lag and p_solo + p_lag <= 1.0 + 1e-6, (
        f"p_solo {p_solo} + p_lag {p_lag} > 1")
    assert k0_lo >= 2, (
        f"k0_lo={k0_lo}: t_ss <= t_s fails for k0 < 2 (max violation +0.033 at "
        "k0=1, +0.127 at k0=0) — see the sweep in this docstring")
    assert k0_hi >= k0_lo

    # ── every primitive, every row, fixed order — no branching above this line ──
    u_cls = torch.rand(B, device=device)
    g_solo = torch.randn(B, device=device)                     # -> logitNormal
    u_lag = torch.rand(B, device=device)
    k0 = torch.randint(k0_lo, k0_hi + 1, (B,), device=device).float()
    w_lag = torch.rand(B, device=device)
    t_s9, t_x9 = sample_timestep_pairs(B, device, p_corner=p_corner,
                                       p_corner2=p_corner2)

    # logitNormal(mean, std) — the official SS t-schedule (flow_matching.py's
    # "logitNormal" branch), reproduced on THIS generator on purpose: calling
    # loss_fn_ss.sample_t would draw on the CPU stream and split the run's RNG.
    t_solo = torch.sigmoid(ss_logit_mean + ss_logit_std * g_solo)

    steps = float(len(_t_seq_len_probe()) - 1)                 # 12
    assert k0_hi <= steps - 1, (
        f"k0_hi={k0_hi} > steps-1={steps - 1:.0f}: k0=steps leaves no slat step "
        "to pair with, so the lag class would be empty by construction")
    # u restricted to (k0/steps, 1] — the range where the pairing exists at all.
    lo = k0 / steps
    u_eff = lo + (1.0 - lo) * u_lag
    t_s_lag = _shift_grid(3.0, u_eff)
    t_ss_lag = _shift_grid(5.0, u_eff - lo)                    # > 0 a.s., no clamp
    t_x_lag = t_s_lag + (1.0 - t_s_lag) * w_lag

    solo = u_cls < p_solo
    lag = (u_cls >= p_solo) & (u_cls < p_solo + p_lag)
    cls = torch.where(solo, torch.full_like(u_cls, CLS_SOLO, dtype=torch.long),
                      torch.where(lag, torch.full_like(u_cls, CLS_LAG, dtype=torch.long),
                                  torch.full_like(u_cls, CLS_CLEAN, dtype=torch.long)))

    zero, one = torch.zeros_like(t_s9), torch.ones_like(t_s9)
    t_ss = torch.where(solo, t_solo, torch.where(lag, t_ss_lag, zero))
    t_s = torch.where(solo, one, torch.where(lag, t_s_lag, t_s9))
    t_x = torch.where(solo, one, torch.where(lag, t_x_lag, t_x9))

    # Structural, so this can only fire if someone edits the construction above.
    assert bool((t_ss <= t_s + 1e-6).all()) and bool((t_s <= t_x + 1e-6).all()), \
        "triple order invariant violated — the composition above is wrong"
    return t_ss, t_s, t_x, cls


def _t_seq_len_probe():
    """The released node count (12 steps -> 13 nodes), read from the sampler so a
    change there cannot silently desync the lag pairing from the trajectory it is
    supposed to rehearse."""
    from .geotex_sampler import SS_PARAMS, _t_seq
    return _t_seq(SS_PARAMS["steps"], SS_PARAMS["rescale_t"])


def _row_balanced_mse(pred: torch.Tensor, target: torch.Tensor,
                      keep: torch.Tensor) -> torch.Tensor:
    """Dense twin of _voxel_balanced_mse, for the SS tower's (B, C, D, H, W).

    CONTAINS A COLLECTIVE — call it UNCONDITIONALLY. A rank whose whole
    micro-batch drew the clean class has zero kept rows and must STILL enter the
    all_reduce; guarding it behind `if keep.any():` makes the collective
    rank-conditional, which is an NCCL hang, not an error.

    Row SELECTION (pred[keep]) rather than multiply-by-zero: the normaliser is
    the all-reduced count of kept rows, so an empty local selection contributes
    0 to the numerator AND 0 to the denominator instead of diluting the loss
    scale, and the empty sum keeps a live grad_fn so DeepSpeed still sees the SS
    parameters as used. Same reasoning as the sparse helper below, one dimension
    down.
    """
    p_sel, t_sel = pred[keep], target[keep]
    # 0-dim, NOT torch.tensor([...]): a shape-[1] denominator broadcasts the
    # scalar numerator up to shape [1], and a 1-D loss silently propagates all
    # the way into the per-task log reducer, which stacks scalars.
    n_local = torch.tensor(float(p_sel.shape[0]), device=pred.device)
    sq = ((p_sel.float() - t_sel.float()) ** 2).flatten(1).mean(1).sum()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        n_glob = n_local.clone()
        torch.distributed.all_reduce(n_glob)
        # x world: DDP averages gradients across ranks, so scaling by the global
        # count alone would divide the loss twice.
        return sq * world / n_glob.clamp_min(1.0)
    return sq / n_local.clamp_min(1.0)


def _voxel_balanced_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE whose PER-VOXEL weight does not depend on how many voxels this rank drew.

    `F.mse_loss` over a sparse batch is a mean over every voxel the rank happens
    to hold, and each rank contributes equally to the averaged gradient, so a
    voxel's weight is 1/(R * N_r). Voxel counts are wildly uneven — measured over
    2883 assets: median 2039, p95 5597, max 13856 under the 8192 cap — and a
    micro-batch of 4 lands anywhere from 4713 to 14363 voxels, so the same voxel
    carries ~3x more gradient in a light batch than a heavy one. That is not bias
    (both directions are equally likely); it is variance injected straight into
    the effective learning rate.

    TRELLIS.2 removes it on the data side, twice: load_balanced_group_indices
    (datasets/structured_latent.py:166) equalises voxels across the gradient
    accumulation slices, and BalancedResumableSampler (utils/data_utils.py:213)
    equalises them across ranks. Neither is reachable from an IterableDataset
    mixture, so this does the same thing on the loss side instead — the fix
    HuggingFace shipped for the identical problem in token-level LM losses:
    normalise by the GLOBAL count rather than the local mean.

    Scaling by world_size makes DDP's gradient averaging cancel exactly, leaving
    every voxel at weight 1/N_global and the loss numerically equal to the mean
    over all voxels on all ranks. Falls back to a plain mean when distributed is
    not initialised, which is every eval path."""
    per_voxel = (pred - target).pow(2).mean(-1)          # (N,) — mean over channels
    n_local = per_voxel.numel()
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return per_voxel.mean() if n_local else per_voxel.sum()
    n = torch.tensor([float(n_local)], device=per_voxel.device)
    torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
    world = torch.distributed.get_world_size()
    return per_voxel.sum() * world / n.clamp_min(1.0).squeeze()


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
    tex_valid=None,                   # (B,) bool | None — per-sample tex supervision
                                      # validity (option C). None = all valid. Invalid
                                      # samples carry a zeros placeholder target: their
                                      # t_x is forced to the trained t_x=1 noise corner
                                      # (so the placeholder never enters the model
                                      # INPUT) and their voxels are row-masked out of
                                      # the tex loss (get_v is (1-s)*noise - x_0 with
                                      # NO t in it, so t_x=1 does NOT neutralize the
                                      # TARGET — the mask is the correctness, not an
                                      # optimization).
    # cond extras (same contract as the cascade path)
    dino_hidden=None, dino_key_mask=None, dino_view_ids=None, qwen_view_ids=None,
    qwen_img_rc=None,
    dino_view_embed=None,
    mask_drop_prob: float = 0.1,
    dino_drop_prob: float = 0.0,
    qwen_drop_prob: float = 0.0,
    cond_max_length: int = 8192,
    # unified knobs
    p_corner: float = 0.2, p_corner2: float = 0.2,
    # ── v10: the third tower ──
    ss_flow_present: bool = False,   # CONFIG-derived, rank-uniform: gates the
                                     # collectives and the log-key set
    target_ss_latent=None,           # (B, 8, 16, 16, 16) RAW — the SS config has
                                     # no normalization key, by upstream design
    connector_ss=None,
    loss_fn_ss=None,                 # logitNormal(1,1), sigma_min 1e-5
    ss_loss_w: float = 1.0,
    p_solo: float = 0.20, p_lag: float = 0.20,
    k0_lo: int = 3, k0_hi: int = 11,
    probe_every: int = 200,   # cond-sensitivity probe cadence; 0 = off
    # ── S2b: geo unfrozen (the "three-pack" of the design doc) ──
    geo_loss_w: float = 0.0,        # >0 turns on geo's OWN velocity loss
    mismatch_w: float = 0.0,        # >0 turns on the mismatched-image hinge (see below)
    mismatch_margin: float = 0.15,  # how much worse the WRONG image must be
    joint_cond_drop: bool = False,  # WARM path: replay cond_x's realized CFG/modality
                                    # drops onto cond_s (same per-sample masks, both
                                    # towers drop together — matches inference's joint
                                    # uncond, where BOTH streams' cond is nulled).
                                    # False = legacy S1/S2b convention (cond_s never
                                    # dropped). No effect from scratch (cond_s=None).
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

    _sdrops = {}   # realized CFG drops, recorded for cond_s joint-drop replay
                   # (recording has no RNG effect)
    _seg = getattr(unified_model, "cond_seg_embed", None)
    _pp = getattr(unified_model, "cond_patch_pos", None)
    cond_x, key_x, _, _ = build_unified_cond(
        connector_tex, cond_hidden, cond_key_mask,
        cond_seg_embed=None if _seg is None else _seg[1],
        cond_patch_pos=None if _pp is None else _pp[1], qwen_img_rc=qwen_img_rc,
        mask_drop_prob=mask_drop_prob, dino_hidden=dino_hidden,
        dino_key_mask=dino_key_mask, dino_drop_prob=dino_drop_prob,
        qwen_drop_prob=qwen_drop_prob, dino_view_ids=dino_view_ids,
        qwen_view_ids=qwen_view_ids, dino_view_embed=dino_view_embed,
        cond_max_length=cond_max_length, drops_out=_sdrops)
    cond_x = _masked_list(cond_x, key_x)
    # A three-stream MMDiT has ONE cond stream, so it ignores cond_s entirely
    # (mmdit3d._prepare). Building it anyway costs a connector forward plus a
    # per-sample python mask loop every step, for a tensor nothing reads.
    # (Moved AFTER the cond_x build so joint_cond_drop can replay its realized
    # masks; on the scratch path nothing between the two sites consumes RNG, so
    # the draw order is unchanged — guarded by the golden test.)
    if getattr(unified_model, "from_scratch", False):
        cond_s = None
    else:
        _cs_ext = None
        if joint_cond_drop and _sdrops.get("drop") is not None:
            # both towers drop TOGETHER on the same samples — the training-time
            # mirror of CFG inference, whose uncond branch nulls both conds.
            # connector_geo(0)+zeros-dino is the geo tower's own learned uncond
            # (s3 trained with mask_drop 0.1), so replayed drops are in-distribution.
            _cs_ext = (_sdrops["drop"], _sdrops["ddrop"], _sdrops["qdrop"])
        # no_grad ONLY while the geo connector is frozen. v10 trains it, and a
        # no_grad here would starve it forever — silently, since the geo tower
        # still learns through its own blocks and nothing would look broken.
        _cs_ctx = (contextlib.nullcontext()
                   if any(pm.requires_grad for pm in connector_geo.parameters())
                   else torch.no_grad())
        with _cs_ctx:
            cond_s, key_s, _, _ = build_unified_cond(
                connector_geo, cond_hidden, cond_key_mask,
                cond_seg_embed=None if _seg is None else _seg[0],
                cond_patch_pos=None if _pp is None else _pp[0], qwen_img_rc=qwen_img_rc,
                mask_drop_prob=0.0, dino_hidden=dino_hidden, dino_key_mask=dino_key_mask,
                dino_drop_prob=0.0, qwen_drop_prob=0.0,
                dino_view_ids=dino_view_ids, qwen_view_ids=qwen_view_ids,
                dino_view_embed=dino_view_embed, cond_max_length=cond_max_length,
                ext_drops=_cs_ext)
            cond_s = _masked_list(cond_s, key_s)

    # ── v10: the SS stream's cond, from the SAME realized per-sample masks ──
    # Unified dropout is not a nicety: at inference CFG nulls all three streams
    # together, so a row that trains with cond dropped on tex but present on SS
    # rehearses a combination that never occurs. Replaying via ext_drops also
    # consumes no RNG, so adding the third connector cannot shift the stream.
    # The SS lane takes the (B,1,1,T) sdpa mask, NOT the masked list the sparse
    # lanes use — its cross-attn is dense.
    cond_ss = sdpa_ss = None
    if ss_flow_present:
        assert connector_ss is not None, "ss_flow_present but connector_ss is None"
        _ss_ext = _cs_ext if (joint_cond_drop and _sdrops.get("drop") is not None) else (
            (_sdrops["drop"], _sdrops["ddrop"], _sdrops["qdrop"])
            if _sdrops.get("drop") is not None else None)
        cond_ss, key_ss, sdpa_ss, _ = build_unified_cond(
            connector_ss, cond_hidden, cond_key_mask,
            cond_seg_embed=None if _seg is None else _seg[2],
            cond_patch_pos=None if _pp is None else _pp[2], qwen_img_rc=qwen_img_rc,
            mask_drop_prob=mask_drop_prob, dino_hidden=dino_hidden,
            dino_key_mask=dino_key_mask, dino_drop_prob=dino_drop_prob,
            qwen_drop_prob=qwen_drop_prob, dino_view_ids=dino_view_ids,
            qwen_view_ids=qwen_view_ids, dino_view_embed=dino_view_embed,
            cond_max_length=cond_max_length, ext_drops=_ss_ext)

    # timestep pair/triple + noising (diffuse/get_v = upstream, never re-derived)
    if ss_flow_present:
        t_ss, t_s, t_x, cls = sample_timestep_triples(
            B, dev, p_corner=p_corner, p_corner2=p_corner2,
            p_solo=p_solo, p_lag=p_lag, k0_lo=k0_lo, k0_hi=k0_hi)
        # Row classes -> masks. Each is used in exactly one place and none of
        # them gates a collective; see the loss blocks below.
        m_solo = cls == CLS_SOLO
        m_lag = cls == CLS_LAG
        m_clean = cls == CLS_CLEAN
        # SS is supervised wherever it is actually diffusing. At t_ss=0 the input
        # IS x_0, so the v-target's noise term never enters the input and the
        # residual is unlearnable — the same reason geo is unsupervised at its
        # own t_s=0 corner, and the reason the official logitNormal schedule
        # never draws 0.
        m_ss_sup = ~m_clean
        # slat is supervised everywhere EXCEPT the solo rows, where its input is
        # pure noise standing in for "not present".
        m_slat_sup = ~m_solo
    else:
        t_s, t_x = sample_timestep_pairs(B, dev, p_corner=p_corner, p_corner2=p_corner2)
        t_ss = cls = None
        m_solo = m_lag = m_ss_sup = None
        m_clean = m_slat_sup = None
    # t_x_sampled survives for the t_x_mean / corner2_frac logs, which must reflect the
    # SCHEDULER rather than data availability (they exist to verify p_corner2 and would
    # otherwise become a mixture with the tex-less fraction).
    t_x_sampled = t_x.clone()
    if tex_valid is not None:
        tex_valid = tex_valid.to(dev).bool()
        t_x = torch.where(tex_valid, t_x, torch.ones_like(t_x))
    # ONE-TIME REPORT of what was actually sampled, not what was configured.
    # The 2026-08-16 run trained on the independent square (33% of every
    # geometry-supervised sample at t_s > t_x) and nothing said so,
    # and nothing anywhere printed the difference. The number that matters is
    # not the config value but the MEASURED share of the batch that lands in
    # the region inference traverses: geometry supervised (t_s != 0) AND
    # texture no cleaner than geometry (t_s <= t_x).
    if not getattr(compute_unified_geotex_loss, "_reported", False):
        compute_unified_geotex_loss._reported = True
        sup = (t_s != 0)
        tri = sup & (t_s <= t_x_sampled)   # sampled, not forced — the report describes the scheduler
        print(f"[timestep] corner {p_corner} corner2 {p_corner2} | "
              f"measured on this batch: "
              f"geo-supervised {sup.float().mean():.0%}, of which "
              f"inference-aligned (t_s<=t_x) "
              f"{(tri.sum() / sup.sum().clamp_min(1)).item():.0%}", flush=True)
        if ss_flow_present:
            # The pair report above describes t_s/t_x only and is SILENT about
            # the three row classes — which is exactly the shape of the failure
            # it was written to prevent. Report the triple too, measured.
            _n = float(B)
            print(f"[timestep-3] configured solo {p_solo} lag {p_lag} "
                  f"k0 {k0_lo}..{k0_hi} | measured on this batch (B={B}): "
                  f"solo {m_solo.float().mean():.2f} lag {m_lag.float().mean():.2f} "
                  f"clean {m_clean.float().mean():.2f} | "
                  f"t_ss mean {t_ss.mean():.3f}, ordered "
                  f"{((t_ss <= t_s + 1e-6) & (t_s <= t_x + 1e-6)).float().mean():.0%} | "
                  f"NOTE small B makes these noisy — trust the per_stage curves",
                  flush=True)
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
    ss_kw, x0_ss, v_ss_target = {}, None, None
    if ss_flow_present:
        assert target_ss_latent is not None and loss_fn_ss is not None, \
            "ss_flow_present but no target_ss_latent / loss_fn_ss — the geotex " \
            "forward would silently train two towers out of three"
        x0_ss = target_ss_latent.to(dev).to(common_dt)
        noise_ss = torch.randn_like(x0_ss)
        t_ss = t_ss.to(common_dt)
        x_tss = loss_fn_ss.diffuse(x0_ss, t_ss, noise=noise_ss)
        v_ss_target = loss_fn_ss.get_v(x0_ss, noise_ss, t_ss)
        ss_kw = dict(x_ss=x_tss, t_ss=(t_ss * 1000.0).to(x_tss.dtype),
                     cond_ss=cond_ss, ss_cond_mask=sdpa_ss,
                     # SS may read the slat lanes ONLY on lag rows: elsewhere the
                     # slat tokens sit on GT-derived coords, i.e. on the answer.
                     ss_read_on=m_lag)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=x_tx.feats.is_cuda):
        _out = unified_model(
            x_ts, x_tx, (t_s * 1000.0).to(x_ts.feats.dtype),
            (t_x * 1000.0).to(x_tx.feats.dtype), cond_s, cond_x,
            tex_concat_cond=cc, **ss_kw)
    if ss_flow_present:
        v_s_pred, v_x_pred, v_ss_pred = _out
    else:
        v_s_pred, v_x_pred = _out
        v_ss_pred = None

    # Row-select (feats[rows_x]), never multiply-by-zero: _voxel_balanced_mse
    # normalizes by the all-reduced count of rows PASSED IN, so selection keeps the
    # per-voxel weight at 1/N_global_valid (constant loss scale regardless of the
    # tex-less fraction) and a rank with zero valid voxels contributes n_local=0 —
    # correctly excluded from the global denominator instead of diluting it, which is
    # also what cancels the ZeRO cross-rank gradient dilution. No `.any()` guard: the
    # helper contains an all_reduce and a guarded call would make the collective
    # rank-conditional (NCCL hang). Empty selection is legal — sum() is 0.0 with a
    # live grad_fn, so DeepSpeed still sees the tex parameters as used.
    if tex_valid is not None:
        _tv = tex_valid.tolist()          # one host sync, not B (tex_valid[b] syncs per item)
        rows_x = torch.cat([
            torch.ones(sl.stop - sl.start, dtype=torch.bool, device=dev) if _tv[b]
            else torch.zeros(sl.stop - sl.start, dtype=torch.bool, device=dev)
            for b, sl in enumerate(x0_x.layout)])
        loss = _voxel_balanced_mse(v_x_pred.feats[rows_x].float(),
                                   v_target.feats[rows_x].float())
        _tvf = tex_valid.float().mean().item()
    else:
        rows_x = None
        loss = _voxel_balanced_mse(v_x_pred.feats.float(), v_target.feats.float())
        _tvf = 1.0
    if ss_flow_present:
        # Mask the SLAT losses off on solo rows by folding the row mask into the
        # voxel selection, rather than by skipping the call: the collective must
        # stay unconditional. rows_x already carries tex-availability; AND them.
        _slat_rows = torch.cat([
            m_slat_sup[b].expand(sl.stop - sl.start)
            for b, sl in enumerate(x0_x.layout)])
        if rows_x is None:
            rows_x = _slat_rows
        else:
            rows_x = rows_x & _slat_rows
        loss = _voxel_balanced_mse(v_x_pred.feats[rows_x].float(),
                                   v_target.feats[rows_x].float())
    # ── SS flow loss. Unconditional call, masked by ROW SELECTION. ──
    ss_loss = None
    # Snapshot the TEX term before anything else is folded into `loss`. The log
    # key below is named tex_flow_loss and read as the tex curve; adding the SS
    # term first made it report tex+ss, which on a batch whose slat rows are all
    # masked reads as a nonzero tex loss with no tex supervision behind it.
    tex_only = loss.detach()
    if ss_flow_present:
        ss_loss = _row_balanced_mse(v_ss_pred.float(), v_ss_target.float(), m_ss_sup)
        loss = loss + ss_loss_w * ss_loss

    # Every key below is emitted UNCONDITIONALLY on every step: the per_stage reducer
    # all_reduces a positionally-sorted value vector built from each rank's own key
    # set, so a sometimes-missing key hands every curve some other metric's number.
    # t_x_mean/corner2_frac use t_x_sampled — the scheduler's draw — so the schedule
    # stays verifiable; tex_valid_frac carries the data-availability signal separately.
    logs = {"tex_flow_loss": tex_only.item(),
            "t_s_mean": t_s.mean().item(), "t_x_mean": t_x_sampled.mean().item(),
            "corner_frac": (t_s == 0).float().mean().item(),
            "corner2_frac": (t_x_sampled == 1).float().mean().item(),
            # What the MODEL actually trains on, forced corners included. On pool1800k
            # (24.9% tex-less) this reads ~0.44 against a configured 0.25 — the tex
            # stream sits at its t_x=1 corner nearly twice as often as P_CORNER2 says,
            # and geometry correspondingly trains in the "texture is pure noise" regime
            # twice as often. corner2_frac (sampled) verifies the SCHEDULER; this key
            # verifies the DISTRIBUTION. geo metrics are not comparable across pools
            # with different pbr coverage — this is the number that says why.
            "corner2_frac_effective": (t_x == 1).float().mean().item(),
            "tex_valid_frac": _tvf}
    # v10 keys. Emitted whenever the third tower exists — the set is decided by
    # CONFIG (ss_flow_present), never by what this batch happened to draw, so it
    # is identical on every rank. The *_frac keys report the MEASURED class mix
    # rather than the configured one: the whole point of the 2026-08-16 lesson is
    # that a schedule which silently differs from its config costs a whole run.
    if ss_flow_present:
        logs.update({
            "ss_flow_loss": ss_loss.detach().item(),
            "t_ss_mean": t_ss.mean().item(),
            "ss_solo_frac": m_solo.float().mean().item(),
            "lag_frac": m_lag.float().mean().item(),
            "clean_frac": m_clean.float().mean().item(),
            "ss_sup_frac": m_ss_sup.float().mean().item(),
            "slat_sup_frac": m_slat_sup.float().mean().item(),
            # gate norms: the model's own vote on whether the cross-tower reads
            # are worth anything. Zero by construction at init; still ~0 late
            # means the third tower is decorative.
            "ss_gate_geo": float(unified_model.ss_gates_geo.detach().abs().mean()),
            "ss_gate_tex": float(unified_model.ss_gates_tex.detach().abs().mean()),
            "ss_reads_gate": float(unified_model.ss_reads_gate.detach().abs().mean()),
        })
    if getattr(unified_model, "cond_seg_embed", None) is not None:
        # Norm relative to the DINO token norm, because that ratio is the thing
        # that went wrong before: a cond-side code at 0.7x the token norm drove
        # the DINO attention share from 0.72 to 0.22. Zero at init; watch that it
        # settles well under 1.
        _dn = float(dino_hidden.float().norm(dim=-1).mean()) if dino_hidden is not None else 1.0
        _se = unified_model.cond_seg_embed.detach().float()
        for _i, _nm in enumerate(("geo", "tex", "ss")):
            logs[f"seg_img_{_nm}"] = float(_se[_i, 0].norm()) / max(_dn, 1e-6)
            logs[f"seg_txt_{_nm}"] = float(_se[_i, 1].norm()) / max(_dn, 1e-6)
    if _pp is not None:
        _dn2 = float(dino_hidden.float().norm(dim=-1).mean()) if dino_hidden is not None else 1.0
        _pd = _pp.detach().float()
        for _i, _nm in enumerate(("geo", "tex", "ss")):
            logs[f"patchpos_{_nm}"] = float(_pd[_i].norm(dim=-1).mean()) / max(_dn2, 1e-6)
        # COVERAGE, because the first version skipped multi-image rows in silence.
        # This is the number that would have shown it: the fraction of rows whose
        # qwen segment actually received a patch code.
        logs["patchpos_cov"] = (0.0 if qwen_img_rc is None
                                else float((qwen_img_rc[..., 0] >= 0).any(1).float().mean()))

    # ── S2b term 1: geo's OWN velocity loss ─────────────────────────────────
    # Without it, geo's only gradient is whatever leaks back through the tex
    # loss — i.e. geo would be optimized to SERVE tex, and its own marginal
    # p(geo|cond) would drift.
    #
    # MASKED at the t_s=0 corner. The reason stated here until 2026-08-17 —
    # "the target is unpredictable from the input" — was wrong and is corrected:
    # at t=0, x_t IS x_0, so the optimal prediction E[v|x_0] = -x_0 is the input
    # negated and is PERFECTLY predictable. That is exactly why the term is
    # worthless here: the model would spend p_corner (0.4) of its geo gradient
    # learning a trivial negation while absorbing the irreducible zero-mean eps
    # as pure variance. And NO inference mode reads geo velocity at t_s=0 — the
    # test is GLOBAL, not per-mode: mesh_only/joint evaluate the model only at
    # the first `steps` nodes, so the last geo evaluation is t_s = 0.2143
    # (12 steps, rescale_t=3) and the final node 0 is the destination, never an
    # input; tex_given_mesh does pin t_s=0 but mmdit3d.tex_forward_cached returns
    # `_, v_x` and discards the geo velocity.
    #
    # Contrast the OTHER sampled corner, t_x=1 (p_corner2), which is NOT masked.
    # mesh_only ignores the tex velocity there (geotex_sampler:195 `v_sp, _`),
    # but joint's FIRST step sits at exactly t_x = 1.0000 and integrates it
    # (:292 `v_sp, v_xp`) — one consumer is enough. It is also where the texture
    # learns E[x_0|cond] from pure noise, i.e. the generation task itself.
    #
    # The rule is therefore "mask a stream's loss at ITS OWN t=0, if nothing
    # reads that velocity", applied to both streams alike. t_x=0 would qualify
    # too, but the tex schedule is continuous and hits it with probability 0
    # (measured over 4M draws), so that branch never runs. Only the SAMPLING is
    # asymmetric — tex|mesh is a product mode, "texture given, generate geometry"
    # is not.
    #
    # What the corner IS for is the geo stream's internal features, which the tex
    # stream attends to; those are trained by the TEX loss flowing back through
    # the joint attention, not by this term.
    #
    # NO REFERENCE BACKS THIS — it is our call, and it is cheap to A/B later.
    #   * TRELLIS.2 gives no evidence either way: its schedule is
    #     `t = torch.rand(B)` on [0,1) (flow_matching.py:135), which hits exactly
    #     0 with probability ~2^-24, so it never trains that atom and has nothing
    #     to mask. "Official masks nothing" is NOT an argument for training here.
    #   * Modality Forcing (arXiv 2606.13676) samples the same three-way split we
    #     do (p_i2d = p_d2i = 0.2 against our 0.4/0.2) but its paper never states
    #     whether the clean modality's loss is excluded at those corners, and the
    #     public repo ships inference only — no loss, no training loop. The one
    #     related thing it does disclose points the same way: the self-distillation
    #     weight is (lambda_hi*t_depth + lambda_lo*(1-t_depth)), i.e. it DAMPS the
    #     RGB term as depth goes clean. That is a soft version of this mask, not
    #     evidence for a hard one.
    # The load-bearing argument is the first one above: nothing downstream reads
    # this output.
    if geo_loss_w > 0:
        v_s_target = loss_fn_slat.get_v(x0_s, noise_s, t_s)
        keep = (t_s != 0)
        # NO `if keep.any():` GUARD HERE, deliberately. _voxel_balanced_mse contains
        # a torch.distributed.all_reduce; guarding the call makes that collective
        # RANK-CONDITIONAL, and a rank whose whole micro-batch happened to draw
        # t_s=0 would skip it while every other rank blocks — an NCCL desync/hang,
        # not an error. At p_corner=0.1 and B=8 that is p=1e-8 per rank-step, i.e.
        # ~0.02 expected occurrences over a 60k-step 32-rank run: rare enough to
        # have never fired, common enough to be a real way to lose four nodes.
        # An all-False mask is a legal input: feats[rows] is empty, n_local=0, the
        # rank contributes 0 to the global voxel count (so it is correctly excluded
        # from the denominator rather than diluting it), and the empty sum still
        # carries a grad_fn so DeepSpeed sees the parameters as used.
        rows = torch.cat([
            torch.ones(sl.stop - sl.start, dtype=torch.bool, device=dev) if keep[b]
            else torch.zeros(sl.stop - sl.start, dtype=torch.bool, device=dev)
            for b, sl in enumerate(x0_s.layout)])
        geo_loss = _voxel_balanced_mse(v_s_pred.feats[rows].float(),
                                       v_s_target.feats[rows].float())
        loss = loss + geo_loss_w * geo_loss
        # Emitted UNCONDITIONALLY for the same class of reason: train_native.py's
        # per_stage reducer builds `keys = sorted(self._stage_sum)` per rank and
        # all_reduces the value vector positionally. Two ranks with the same NUMBER
        # of keys but different key SETS reduce cleanly and hand every curve some
        # other metric's number.
        logs["geo_flow_loss"] = geo_loss.detach().item()

            # ── mismatched-image hinge ──────────────────────────────────────
            # Measured 2026-08-20 across v8 checkpoints 2k..10k: handing geometry
            # a DIFFERENT asset's image costs it 2.6 / 4.1 / 4.3 / 4.7 / 4.3% —
            # it climbed until step 4000 and has been flat for the 6000 since,
            # while the released model pays 13.3%. So "conditioning will become
            # load-bearing with more steps" is not supported; the model reaches a
            # plateau where the image is worth little because GT coords already
            # determine most of the shape and nothing in the objective rewards
            # using the picture beyond that.
            #
            # This adds the reward directly: run the SAME noised inputs at the
            # SAME timesteps with the conditioning rolled by one sample, and
            # require the wrong image to cost at least `mismatch_margin`. A
            # hinge, not a plain gap-maximiser, so it stops pushing once the
            # margin is met and cannot trade real accuracy for separation.
            #
            # Costs one extra forward+backward, i.e. roughly 1.8x the step. Off
            # by default; this is an experiment, not a shipped default.
        # Guarded ONLY by mismatch_w and B — both identical on every rank, so the
        # all_reduce inside _voxel_balanced_mse below stays collective. Nesting
        # this under `keep.any()` (as it was) would have made it rank-conditional
        # the moment the hinge was enabled.
        if mismatch_w > 0 and B > 1:
            roll = lambda c: None if c is None else c[1:] + c[:1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=x_tx.feats.is_cuda):
                v_s_bad, _ = unified_model(
                    x_ts, x_tx, (t_s * 1000.0).to(x_ts.feats.dtype),
                    (t_x * 1000.0).to(x_tx.feats.dtype),
                    roll(cond_s), roll(cond_x), tex_concat_cond=cc)
            bad = _voxel_balanced_mse(v_s_bad.feats[rows].float(),
                                      v_s_target.feats[rows].float())
            hinge = torch.relu(geo_loss * (1.0 + mismatch_margin) - bad)
            loss = loss + mismatch_w * hinge
            logs["geo_mismatch_ratio"] = (bad / geo_loss.clamp_min(1e-8)).detach().item()
            logs["geo_mismatch_hinge"] = hinge.detach().item()

    if mismatch_w > 0 and geo_loss_w <= 0 and not getattr(
            compute_unified_geotex_loss, "_warned_dead_hinge", False):
        compute_unified_geotex_loss._warned_dead_hinge = True
        print("[geotex] WARNING: mismatch_w > 0 but geo_loss_w == 0 — the hinge block "
              "is nested inside the geo loss and is a SILENT NO-OP in this config "
              "(no gradient, no geo_mismatch_* keys). On warm starts geo_loss_w is "
              "forced to 0 unless geotex_unfreeze_geo; the experiment you think is "
              "running is not.", flush=True)

    # ── CONDITIONING SENSITIVITY PROBE ──────────────────────────────────────
    # THE diagnostic this project lacked. After 60,000 steps we still could not
    # say whether the geometry stream was undertrained or structurally unable to
    # learn, because nothing tracked whether it reads the image AT ALL — and
    # SAVE_KEEP had deleted every early checkpoint by the time we looked.
    #
    # Measure: run the model on PURE NOISE (t=1, the generation regime) once
    # with this batch's conditioning and once with the conditioning ROLLED by
    # one sample, so every sample gets a different asset's image. Report
    #     MSE(x0 | mismatched image) / MSE(x0 | correct image)
    # 1.0 = the stream ignores its conditioning entirely. Measured 2026-08-17 on
    # checkpoint-60000: geometry 1.045, texture 1.35, released TRELLIS.2 1.40.
    # Flat at 1.0 from step 2000 => structural, stop and fix. Climbing => it is
    # a question of training length.
    #
    # Costs two extra forwards every `probe_every` steps, no sampling, no grad.
    if probe_every > 0 and B > 1:
        _n = getattr(compute_unified_geotex_loss, "_probe_n", 0)
        compute_unified_geotex_loss._probe_n = _n + 1
        if _n % probe_every == 0:
            with torch.no_grad():
                one = torch.ones(B, device=dev, dtype=common_dt)
                ns = x0_s.replace(torch.randn_like(x0_s.feats))
                nx = x0_x.replace(torch.randn_like(x0_x.feats))
                # rolling the per-sample cond lists is what makes this a
                # MISMATCH rather than an ablation: the model still gets a real
                # image, just the wrong one, so a drop cannot be explained by
                # the conditioning going out of distribution.
                # cond_s is None on the from-scratch path (one cond stream,
                # so cond_x carries it); roll whichever lists exist.
                roll = lambda c: None if c is None else c[1:] + c[:1]
                bad_s, bad_x = roll(cond_s), roll(cond_x)
                out = {}
                for tag, (cs, cx) in (("ok", (cond_s, cond_x)),
                                      ("bad", (bad_s, bad_x))):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=x_tx.feats.is_cuda):
                        ps, px = unified_model(ns, nx, (one * 1000.0), (one * 1000.0),
                                               cs, cx, tex_concat_cond=ns)
                    # x0 = x_t - t*v with t = 1
                    # Tex arm MASKED to valid rows. Placeholder samples reconstruct
                    # the same meaningless zeros under "ok" and "bad" alike, adding a
                    # large near-identical term to numerator and denominator — which
                    # drags cond_sens_tex toward 1.0, the exact "the stream ignores
                    # its conditioning" signature this probe exists to detect.
                    _e_x = ((nx.feats.float() - px.feats.float()) - x0_x.feats.float()).pow(2)
                    if rows_x is not None:
                        _e_x = _e_x[rows_x]
                    out[tag] = (
                        float(((ns.feats.float() - ps.feats.float()) - x0_s.feats.float()).pow(2).mean()),
                        float(_e_x.mean()) if _e_x.numel() else None)
                _prev = getattr(compute_unified_geotex_loss, "_probe_last", (1.0, 1.0))
                compute_unified_geotex_loss._probe_last = (
                    out["bad"][0] / max(out["ok"][0], 1e-8),
                    # All-invalid batch on a probe step (p≈0.25^8 per batch at 75%
                    # coverage): no valid tex voxels to measure. Carry the previous
                    # tex reading — a NaN here would be summed into the 500-step
                    # per_stage window and blank the whole cond_sens_tex curve.
                    out["bad"][1] / max(out["ok"][1], 1e-8)
                    if out["ok"][1] is not None and out["bad"][1] is not None
                    else _prev[1])
        # CARRY THE LAST VALUE ON EVERY STEP. HF Trainer averages logged
        # scalars over `logging_steps` and treats a missing key as absent from
        # only some steps — a probe that fires once per 200 steps came out as
        # 1.0/5 = 0.2 in a 5-step window, i.e. a number that cannot even be a
        # ratio. Emitting the cached value every step makes the window mean
        # equal the value.
        _last = getattr(compute_unified_geotex_loss, "_probe_last", None)
        if _last is not None:
            logs["cond_sens_geo"], logs["cond_sens_tex"] = _last

    return loss, logs
