"""Native-VLM dataset + collator (Phase 3 of QWEN35_VLM_DESIGN.md).

Reuses the EXACT 3D-target loaders + TRELLIS norm stats from `TR2BLIP3oDataset`
(no drift), and swaps ONLY the input side: instead of TA-Tok + preprocess_qwen,
it yields the raw caption + PIL image(s) and lets the native `AutoProcessor`
(apply_chat_template + processor) build the VLM inputs in the collator.

One code path covers text→3D / single-img→3D / multi-view→3D (vary the content
items). `ss_only=True` loads just the SS target (v1 SS-only training).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

from . import _paths  # noqa: F401
from .dataset import TR2BLIP3oDataset
from trellis2.datasets.structured_latent import SLat            # type: ignore
from trellis2.datasets.structured_latent_svpbr import SLatPbr   # type: ignore


class TR2NativeVLMDataset(TR2BLIP3oDataset):
    """Same manifest + 3D-target loaders as TR2BLIP3oDataset, but emits raw
    caption + PIL images (no tokenization here — the collator's processor does it)."""

    def __init__(self, data_path: str, data_args, ss_only: bool = False):
        # tokenizer unused on the native path (processor lives in the collator).
        super().__init__(tokenizer=None, data_path=data_path, data_args=data_args)
        self.ss_only = ss_only
        self.num_cond_views = getattr(data_args, "num_cond_views", 1)

    def __getitem__(self, i):
        rec = self.records[i]
        rtype = rec.get("type", "I_2_3D")
        caption = rec.get("txt", "") or ""

        # --- input images (raw PIL; processor resizes) ---
        images: List[Image.Image] = []
        if rtype == "I_2_3D" and "image" in rec:
            views = rec.get("multi_view_renders") or [rec["image"]]
            for p in views[: self.num_cond_views]:
                images.append(Image.open(p).convert("RGB"))

        # --- 3D targets (reuse parent loaders verbatim) ---
        data: Dict[str, Any] = {
            "caption": caption,
            "images": images,
            "type": rtype,
            "id": rec.get("id", f"idx_{i}"),
            "target_ss_latent": self.process_target_ss_latent(rec["target_ss_latent"]),
        }
        if not self.ss_only:
            if "target_shape_slat_512" in rec:
                data["target_shape_slat_512_item"] = self.process_target_shape_slat(
                    rec["target_shape_slat_512"]
                )
            if "target_tex_slat_512" in rec and "target_shape_slat_512" in rec:
                data["target_tex_slat_512_item"] = self.process_target_tex_slat(
                    tex_path=rec["target_tex_slat_512"],
                    shape_path=rec["target_shape_slat_512"],
                )
        return data


@dataclass
class NativeVLMCollator:
    """Builds VLM inputs via the native AutoProcessor + stacks 3D targets.

    `processor` = AutoProcessor.from_pretrained(vlm_model). Emits exactly the
    kwargs TrellisNativeVLM.forward expects.
    """
    processor: Any
    system_prompt: Optional[str] = None  # keep minimal / None (drop_idx deferred)
    # Token-budget cap. Qwen vision tokens = pixels / (patch_size·merge_size)². This ratio is
    # BACKBONE-SPECIFIC: Qwen2.5-VL patch14 → 784 px/tok; Qwen3-VL/Qwen3.5 patch16 → 1024 px/tok.
    # We DERIVE it from the actual processor (_px_per_tok) so the cap is correct for any backbone
    # — hardcoding 1024 would under-cap Qwen2.5-VL by ~31% (more tokens than intended).
    # Single image → ≤ max_tokens_single (≈ DINOv3@1024 density). Multi-view → split token_budget
    # across views so the TOTAL cond stays ≤ token_budget (= cond_max_length; beyond it the flow
    # silently truncates → drops views). Per-call max_pixels is ignored by the processor, so we
    # resize here. Only downscales (never upscales); keeps aspect ratio.
    max_tokens_single: int = 4096
    token_budget: int = 8192
    _px_per_tok_fallback: int = 1024  # used only if the processor lacks patch_size/merge_size

    def _px_per_tok(self) -> int:
        """Pixels per vision token = (patch_size·merge_size)², read from the ACTUAL processor
        → backbone-correct (Qwen2.5-VL 14·2 → 784; Qwen3-VL/3.5 16·2 → 1024)."""
        ip = self.processor.image_processor
        ps, ms = getattr(ip, "patch_size", None), getattr(ip, "merge_size", None)
        if isinstance(ps, int) and isinstance(ms, int):
            return (ps * ms) ** 2
        return self._px_per_tok_fallback

    @staticmethod
    def _cap_image(img: "Image.Image", max_px: int) -> "Image.Image":
        w, h = img.size
        if w * h <= max_px:
            return img
        s = (max_px / float(w * h)) ** 0.5
        return img.resize((max(32, round(w * s)), max(32, round(h * s))), Image.LANCZOS)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        texts: List[str] = []
        flat_images: List[Image.Image] = []
        for inst in instances:
            imgs = inst["images"]
            n = max(len(imgs), 1)
            per_tok = min(self.max_tokens_single, self.token_budget // n)  # split budget over views
            max_px = per_tok * self._px_per_tok()  # backbone-correct px/token
            imgs = [self._cap_image(im, max_px) for im in imgs]
            content: List[Dict[str, Any]] = [{"type": "image"} for _ in imgs]
            content.append({"type": "text", "text": inst["caption"]})
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": content})
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            flat_images.extend(imgs)

        proc_kwargs = dict(text=texts, padding=True, return_tensors="pt")
        if flat_images:
            proc_kwargs["images"] = flat_images
        enc = self.processor(**proc_kwargs)

        # Safety net: if vision tokens still exceed the budget, the flow's cond_max_length
        # truncation would silently drop content/views — warn loudly with the real count.
        g = enc.get("image_grid_thw")
        if g is not None:
            ms = self.processor.image_processor.merge_size
            vtok = int((g[:, 0] * g[:, 1] * g[:, 2]).sum().item() // (ms * ms))
            if vtok > self.token_budget:
                import warnings
                warnings.warn(
                    f"[NativeVLMCollator] vision tokens={vtok} > token_budget={self.token_budget}: "
                    f"cond_max_length will TRUNCATE (drops later views/content)."
                )

        batch: Dict[str, Any] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target_ss_latent": torch.stack(
                [inst["target_ss_latent"] for inst in instances], dim=0
            ),
        }
        if "pixel_values" in enc:
            batch["pixel_values"] = enc["pixel_values"]
            batch["image_grid_thw"] = enc.get("image_grid_thw")
        if "pixel_values_videos" in enc:
            batch["pixel_values_videos"] = enc["pixel_values_videos"]
            batch["video_grid_thw"] = enc.get("video_grid_thw")

        # Sparse SLAT targets (reuse TRELLIS collate_fn — identical to upstream).
        if "target_shape_slat_512_item" in instances[0]:
            shape_pack = SLat.collate_fn(
                [inst["target_shape_slat_512_item"] for inst in instances]
            )
            batch["target_shape_slat_512"] = shape_pack["x_0"]
        if "target_tex_slat_512_item" in instances[0]:
            tex_pack = SLatPbr.collate_fn(
                [inst["target_tex_slat_512_item"] for inst in instances]
            )
            batch["target_tex_slat_512"] = tex_pack["x_0"]
            batch["tex_concat_cond"] = tex_pack["concat_cond"]
        return batch
