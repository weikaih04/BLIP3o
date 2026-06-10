"""Unified VLM collator — the SINGLE source of truth for building Qwen-VL inputs
(+ 3D flow targets) from dataset instances.

History: this logic existed as TWO drifted copies — `dataset_native.NativeVLMCollator`
(Phase-5 single-task era; budgets 400/1000; still used by the INFERENCE harness) and
`data/tasks/threed.py collate_fn` (multi-task port; budgets 4096/8192; used by TRAINING).
The 8× budget gap was a LIVE train/infer mismatch: e.g. eval_clear8 renders are 1024² →
training saw 1024 vision tok/view while the harness capped the same image to 400 (single)
/ ~333 (multi). Both entry points now delegate here with ONE set of budgets (the
training-side values, so training is bit-identical and inference snaps to training).

Budget mechanics (Qwen3.5: patch16 · merge2 → 1024 px per vision token):
    per_tok = min(max_tokens_single, token_budget // n_views)
    each view is DOWNSCALED (keep AR) so its pixels ≤ per_tok · px_per_tok.
`target_tokens_per_view > 0` additionally UPSCALES smaller views toward
min(target, per_tok) tokens — v3 distillation wants 1024/view (a 32×32 grid that
1:1 matches the DINOv3-teacher grid @512px). Default 0 = off (no upscaling).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

from . import _paths  # noqa: F401 — ensures trellis2 on sys.path
from trellis2.datasets.structured_latent import SLat  # type: ignore
from trellis2.datasets.structured_latent_svpbr import SLatPbr  # type: ignore

# ── Single source of truth for the cond token budget ──
# (was: 400/1000 in NativeVLMCollator vs 4096/8192 in threed.py — unified to the
# training-side values). 8192 matches flow_heads.cond_max_length (the flow's
# silent-truncation ceiling): the collator caps BELOW it so truncation can never
# silently drop views/content.
MAX_TOKENS_SINGLE: int = 4096
TOKEN_BUDGET: int = 8192
DINO_IMAGE_SIZE: int = 512  # DINOv3 input (TRELLIS standard) for the dino_images emission

# Process-wide default for `target_tokens_per_view` (callers pass None to inherit it).
# The mixture router invokes task collate_fns with a FIXED (batch, processor) signature,
# so the training launcher sets this once (train_native --target_tokens_per_view) instead
# of threading a kwarg through every task. 0 = no upscaling (today's behavior).
_DEFAULT_TARGET_TOKENS_PER_VIEW: int = 0


def set_default_target_tokens_per_view(n: int) -> None:
    global _DEFAULT_TARGET_TOKENS_PER_VIEW
    _DEFAULT_TARGET_TOKENS_PER_VIEW = int(n or 0)


def px_per_tok(processor) -> int:
    """Pixels per vision token = (patch_size·merge_size)², read from the ACTUAL
    processor (Qwen3.5: 16·2 → 1024). Falls back to 1024 if the processor lacks
    the attributes (text-only processors included — chat.py passes those too)."""
    ip = getattr(processor, "image_processor", None)
    ps = getattr(ip, "patch_size", None) if ip else None
    ms = getattr(ip, "merge_size", None) if ip else None
    if isinstance(ps, int) and isinstance(ms, int):
        return (ps * ms) ** 2
    return 1024


def cap_image(img: Image.Image, max_px: int) -> Image.Image:
    """Downscale-only resize (keep aspect ratio) so w·h ≤ max_px."""
    w, h = img.size
    if w * h <= max_px:
        return img
    s = (max_px / float(w * h)) ** 0.5
    return img.resize((max(32, round(w * s)), max(32, round(h * s))), Image.LANCZOS)


def upscale_image(img: Image.Image, min_px: int) -> Image.Image:
    """Upscale-only resize (keep aspect ratio) so w·h ≥ min_px. Used by
    target_tokens_per_view: upsampling adds no pixel information but densifies the
    vision-token grid (e.g. 512²=256 tok → 1024²=1024 tok = DINOv3@512's 32×32 grid)."""
    w, h = img.size
    if w * h >= min_px:
        return img
    s = (min_px / float(w * h)) ** 0.5
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def boiler_ids(tok, include_system: bool = False) -> set:
    """Chat-template STRUCTURAL token ids to mask out of the flow cond
    (<|im_start|>, <|im_end|>, <|vision_start|>, <|vision_end|>, <think>, </think>,
    role-name tokens). Cached on the tokenizer object (collate is a plain function).
    `include_system` adds the "system" role token (only when a system prompt is used —
    masking it unconditionally would also mask the word "system" inside captions)."""
    base = getattr(tok, "_boiler_ids_cache", None)
    if base is None:
        names = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                 "<think>", "</think>"]
        ids = set()
        for nm in names:
            tid = tok.convert_tokens_to_ids(nm)
            if isinstance(tid, int) and tid is not None and tid >= 0:
                ids.add(tid)
        for nm in ("user", "assistant"):   # role names directly follow <|im_start|>
            for tid in tok(nm, add_special_tokens=False).input_ids:
                ids.add(tid)
        base = ids
        tok._boiler_ids_cache = ids
    if not include_system:
        return base
    sys_ids = getattr(tok, "_boiler_ids_cache_sys", None)
    if sys_ids is None:
        sys_ids = set(tok("system", add_special_tokens=False).input_ids)
        tok._boiler_ids_cache_sys = sys_ids
    return base | sys_ids


def collate_vlm_3d(
    batch: Sequence[Dict],
    processor: Any,
    *,
    max_tokens_single: int = MAX_TOKENS_SINGLE,
    token_budget: int = TOKEN_BUDGET,
    target_tokens_per_view: Optional[int] = None,   # None → module default (see setter above)
    system_prompt: Optional[str] = None,
    emit_dino: bool = True,
    warn_over_budget: bool = True,
) -> Dict[str, Any]:
    """Chat-template → VLM inputs + stacked 3D targets. THE collate body — both
    `threed.py collate_fn` (training) and `NativeVLMCollator` (inference harness)
    delegate here, so the two paths cannot drift again."""
    if target_tokens_per_view is None:
        target_tokens_per_view = _DEFAULT_TARGET_TOKENS_PER_VIEW
    texts: List[str] = []
    flat_images: List[Image.Image] = []
    dino_pil: List[Image.Image] = []
    ppt = px_per_tok(processor)

    for inst in batch:
        imgs = inst["images"]
        n = max(len(imgs), 1)
        per_tok = min(max_tokens_single, token_budget // n)  # split budget over views
        max_px = per_tok * ppt                               # backbone-correct px/token
        if target_tokens_per_view > 0:                       # v3: densify the token grid
            min_px = min(target_tokens_per_view, per_tok) * ppt
            imgs_in = [upscale_image(im, min_px) for im in imgs]
        else:
            imgs_in = list(imgs)
        capped = [cap_image(im, max_px) for im in imgs_in]
        content: List[Dict[str, Any]] = [{"type": "image"} for _ in capped]
        content.append({"type": "text", "text": inst["caption"]})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        # add_generation_prompt=False: the VLM hidden is CONDITIONING, not generation —
        # no assistant-turn start / Qwen3.5 empty <think></think> boilerplate (7 tokens of
        # pure dilution, worst for text→3D where Qwen is the only cond).
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False))
        flat_images.extend(capped)
        # DINOv3 images (consumed by dual-cond anchor / dino_align / v3 distill teacher):
        # the SAME conditioning views, ORIGINALS (pre-cap/pre-upscale), in the SAME flat
        # order as Qwen's images / image_grid_thw. Cheap to always emit.
        dino_pil.extend(imgs)

    proc_kwargs = dict(text=texts, padding=True, return_tensors="pt")
    if flat_images:
        proc_kwargs["images"] = flat_images
    enc = processor(**proc_kwargs)

    # Safety net: if vision tokens still exceed the budget, the flow's cond_max_length
    # truncation would silently drop content/views — warn loudly with the real count.
    g = enc.get("image_grid_thw")
    if warn_over_budget and g is not None:
        ms = processor.image_processor.merge_size
        vtok = int((g[:, 0] * g[:, 1] * g[:, 2]).sum().item() // (ms * ms))
        if vtok > token_budget:
            warnings.warn(
                f"[vlm_collate] vision tokens={vtok} > token_budget={token_budget}: "
                f"flow cond_max_length will TRUNCATE (drops later views/content)."
            )

    # cond_keep_mask: which tokens the flow cross-attn attends to as conditioning
    # = attention_mask (real tokens) MINUS chat-template structural boilerplate.
    # We KEEP <|image_pad|> (expands to the image patches = real content) + caption text.
    # Masked at the cross-attn level only → does NOT shift positions / image_pad expansion.
    tok = processor.tokenizer
    _boiler = torch.tensor(sorted(boiler_ids(tok, include_system=bool(system_prompt))),
                           dtype=enc["input_ids"].dtype)
    cond_keep = enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], _boiler)

    out: Dict[str, Any] = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "cond_keep_mask": cond_keep,
    }
    if "_task" in batch[0]:
        out["_task"] = batch[0]["_task"]
    if "target_ss_latent" in batch[0]:
        out["target_ss_latent"] = torch.stack(
            [inst["target_ss_latent"] for inst in batch], dim=0)
    if "pixel_values" in enc:
        out["pixel_values"] = enc["pixel_values"]
        out["image_grid_thw"] = enc.get("image_grid_thw")
        if emit_dino and dino_pil:
            # (M,3,512,512) in [0,1], flat order matching image_grid_thw.
            # (DinoV3FeatureExtractor applies its own ImageNet Normalize.)
            import numpy as _np
            out["dino_images"] = torch.stack([
                torch.from_numpy(
                    _np.asarray(im.convert("RGB").resize((DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
                                                         Image.BILINEAR), dtype="float32")
                ).permute(2, 0, 1) / 255.0
                for im in dino_pil
            ], dim=0)
    if "pixel_values_videos" in enc:
        out["pixel_values_videos"] = enc["pixel_values_videos"]
        out["video_grid_thw"] = enc.get("video_grid_thw")

    # Sparse SLAT targets (reuse TRELLIS collate_fn — identical to upstream).
    if "target_shape_slat_512_item" in batch[0]:
        shape_pack = SLat.collate_fn(
            [inst["target_shape_slat_512_item"] for inst in batch])
        out["target_shape_slat_512"] = shape_pack["x_0"]
    if "target_tex_slat_512_item" in batch[0]:
        tex_pack = SLatPbr.collate_fn(
            [inst["target_tex_slat_512_item"] for inst in batch])
        out["target_tex_slat_512"] = tex_pack["x_0"]
        out["tex_concat_cond"] = tex_pack["concat_cond"]
    return out
