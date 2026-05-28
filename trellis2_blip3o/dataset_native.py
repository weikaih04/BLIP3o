"""Native-VLM dataset + collator (Phase 3 of QWEN35_VLM_DESIGN.md).

Reuses the EXACT 3D-target loaders + TRELLIS norm stats from `TR2BLIP3oDataset`
(no drift), and swaps ONLY the input side: instead of TA-Tok + preprocess_qwen,
it yields the raw caption + PIL image(s) and lets the native `AutoProcessor`
(apply_chat_template + processor) build the VLM inputs in the collator.

One code path covers text→3D / single-img→3D / multi-view→3D. Two manifest
schemas, auto-dispatched per record:

  UNIFIED (new, built by build_index.py):
    { sha256, subset,
      ss_latent_64, shape_latent_{512,1024}, pbr_latent_{512,1024},
      renders_dir, n_views, captions: list[str], aesthetic_score }
    Per __getitem__: sample task ∈ {T, I1, IM} from `task_mix`; for I1/IM
    sample random views from `renders_dir`; pick one caption from `captions`
    (T-task only). All three tasks come from ONE record + ONE manifest.

  LEGACY (existing overfit/abo_100 manifests):
    { id, type∈{T_2_3D,I_2_3D}, image, multi_view_renders, txt,
      target_ss_latent, target_shape_slat_512, target_tex_slat_512 }
    Behavior preserved bit-for-bit; auto-detected by absence of `renders_dir`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from . import _paths  # noqa: F401
from .dataset import TR2BLIP3oDataset
from trellis2.datasets.structured_latent import SLat            # type: ignore
from trellis2.datasets.structured_latent_svpbr import SLatPbr   # type: ignore


def _parse_task_mix(spec) -> Dict[str, float]:
    """'T:0.2,I1:0.4,IM:0.4' (or already a dict) → normalized {task: prob}.
    Tasks: T (text-only), I1 (single image), IM (multi-image, 2..max_views)."""
    if isinstance(spec, dict):
        mix = {str(k): float(v) for k, v in spec.items()}
    elif isinstance(spec, str):
        mix = {}
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":")
            mix[k.strip()] = float(v)
    else:
        raise TypeError(f"task_mix must be dict or 'k:v,k:v' string, got {type(spec)}")
    allowed = {"T", "I1", "IM"}
    bad = set(mix) - allowed
    if bad:
        raise ValueError(f"task_mix has unknown keys {bad}; allowed: {allowed}")
    s = sum(mix.values())
    if s <= 0:
        raise ValueError(f"task_mix probabilities must sum > 0, got {mix}")
    return {k: v / s for k, v in mix.items()}


class TR2NativeVLMDataset(TR2BLIP3oDataset):
    """Manifest-driven dataset; auto-detects UNIFIED vs LEGACY schema per record.

    UNIFIED-only data_args (all optional, read via getattr):
      task_mix         : 'T:0.2,I1:0.4,IM:0.4' (str) or dict — task distribution.
      max_views        : int (default 4) — upper bound for IM views;
                         n ~ U[2, max_views].
      slat_resolution  : 512 | 1024 — which shape/pbr latent to load.
      crop_to_object   : bool — tight alpha-bbox crop on RGBA renders before
                         RGB convert (matches TRELLIS.2 ImageConditionedMixin).
      min_aesthetic    : float | None — drop records whose aesthetic_score is
                         below this. Records with score=None (TexVerse) are kept.

    LEGACY data_args (existing, unchanged):
      num_cond_views   : int — takes `multi_view_renders[:N]` deterministically.
    """

    def __init__(self, data_path: str, data_args, ss_only: bool = False):
        # tokenizer unused on the native path (processor lives in the collator).
        super().__init__(tokenizer=None, data_path=data_path, data_args=data_args)
        self.ss_only = ss_only

        # --- legacy knobs (unchanged) ---
        self.num_cond_views = getattr(data_args, "num_cond_views", 1)

        # --- unified-schema knobs ---
        mix_spec = getattr(data_args, "task_mix", None) or "T:0.2,I1:0.4,IM:0.4"
        self.task_mix = _parse_task_mix(mix_spec)
        self.max_views = int(getattr(data_args, "max_views", 4))
        self.slat_resolution = int(getattr(data_args, "slat_resolution", 512))
        self.crop_to_object = bool(getattr(data_args, "crop_to_object", False))

        # Optional aesthetic filter at init (unified records only carry a score;
        # TexVerse rows have None and are always kept).
        min_aes = getattr(data_args, "min_aesthetic", None)
        if min_aes is not None:
            before = len(self.records)
            kept = []
            for r in self.records:
                aes = r.get("aesthetic_score")
                if aes is None or float(aes) >= float(min_aes):
                    kept.append(r)
            self.records = kept
            print(f"[TR2NativeVLMDataset] aesthetic filter ≥{min_aes}: "
                  f"{before} → {len(self.records)}")

    # ------------------------------------------------------------------
    # Schema dispatch
    # ------------------------------------------------------------------
    def __getitem__(self, i):
        rec = self.records[i]
        if "renders_dir" in rec:
            return self._getitem_unified(rec, i)
        return self._getitem_legacy(rec, i)

    # ------------------------------------------------------------------
    # Unified schema — per-call task / view / caption sampling
    # ------------------------------------------------------------------
    def _getitem_unified(self, rec: Dict[str, Any], i: int) -> Dict[str, Any]:
        rng = np.random.default_rng()  # fresh entropy per call

        caps: List[str] = [c for c in (rec.get("captions") or []) if c]
        task = self._sample_task(rng, allow_text=bool(caps))

        # Caption is only fed to the VLM on the T-task; for I1/IM we pass empty
        # string so the model truly does image-only conditioning.
        caption = str(rng.choice(caps)) if (task == "T" and caps) else ""

        images: List[Image.Image] = []
        if task in ("I1", "IM"):
            n_avail = int(rec.get("n_views", 16))
            if task == "I1":
                n = 1
            else:
                hi = max(3, self.max_views + 1)        # rng.integers high is exclusive
                n = int(rng.integers(2, hi))
            n = min(n, n_avail)
            view_ids = sorted(int(v) for v in rng.choice(n_avail, size=n, replace=False))
            images = self._load_views(rec["renders_dir"], view_ids)

        res = self.slat_resolution
        shape_path = rec.get(f"shape_latent_{res}")
        pbr_path   = rec.get(f"pbr_latent_{res}")
        ss_path    = rec["ss_latent_64"]

        data: Dict[str, Any] = {
            "caption": caption,
            "images": images,
            "type": task,
            "id": rec.get("sha256", f"idx_{i}"),
            "target_ss_latent": self.process_target_ss_latent(ss_path),
        }
        if not self.ss_only:
            if shape_path:
                data["target_shape_slat_512_item"] = self.process_target_shape_slat(shape_path)
            if pbr_path and shape_path:
                data["target_tex_slat_512_item"] = self.process_target_tex_slat(
                    tex_path=pbr_path, shape_path=shape_path,
                )
        return data

    def _sample_task(self, rng, allow_text: bool) -> str:
        mix = self.task_mix
        if not allow_text and mix.get("T", 0) > 0:
            mix = {k: v for k, v in mix.items() if k != "T"}
            s = sum(mix.values()) or 1.0
            mix = {k: v / s for k, v in mix.items()}
        keys = list(mix.keys())
        probs = [mix[k] for k in keys]
        return keys[int(rng.choice(len(keys), p=probs))]

    def _load_views(self, renders_dir: str, view_ids: Sequence[int]) -> List[Image.Image]:
        """Open <renders_dir>/{vid:03d}.{webp,png,jpg}; optional alpha-bbox crop."""
        out: List[Image.Image] = []
        for v in view_ids:
            path = None
            for ext in ("webp", "png", "jpg"):
                cand = os.path.join(renders_dir, f"{v:03d}.{ext}")
                if os.path.isfile(cand):
                    path = cand
                    break
            if path is None:
                continue   # silently skip missing views
            img = Image.open(path)
            if self.crop_to_object and img.mode == "RGBA":
                img = self._alpha_crop(img)
            out.append(img.convert("RGB"))
        return out

    @staticmethod
    def _alpha_crop(img: Image.Image) -> Image.Image:
        """Square crop centered on alpha bbox — matches TRELLIS.2 ImageConditionedMixin."""
        a = np.array(img.getchannel("A"))
        ys, xs = np.where(a > 0)
        if ys.size == 0:
            return img
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        h = max(x1 - x0, y1 - y0) / 2.0
        box = (int(cx - h), int(cy - h), int(cx + h), int(cy + h))
        return img.crop(box)

    # ------------------------------------------------------------------
    # Legacy schema — unchanged behavior
    # ------------------------------------------------------------------
    def _getitem_legacy(self, rec: Dict[str, Any], i: int) -> Dict[str, Any]:
        rtype = rec.get("type", "I_2_3D")
        caption = rec.get("txt", "") or ""

        images: List[Image.Image] = []
        if rtype == "I_2_3D" and "image" in rec:
            views = rec.get("multi_view_renders") or [rec["image"]]
            for p in views[: self.num_cond_views]:
                images.append(Image.open(p).convert("RGB"))

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
    # Token-budget cap. Qwen3.5 vision tokens = pixels / (patch_size·merge_size)²
    # = pixels / (16·2)² = pixels / 1024.
    # Single image → ≤ max_tokens_single. Multi-view → split token_budget across views so
    # TOTAL cond ≤ token_budget (= cond_max_length; beyond it the flow silently truncates →
    # drops views). max_pixels is ignored by the processor, so we resize here (downscale
    # only; keep aspect ratio). Defaults: single-view ≤400 tok (≈ 640²), multi-view total
    # ≤1000 (auto split per-view = token_budget // N). Override in __init__ if needed.
    # DEPRECATED note: The _px_per_tok() helper STILL derives px/tok from the processor
    # (so Qwen2.5-VL patch14 → 784 px/tok would also be correct if you load that backbone
    # for inspection). But Qwen2.5-VL / Qwen3-VL are no longer a tested training path.
    max_tokens_single: int = 400
    token_budget: int = 1000
    _px_per_tok_fallback: int = 1024  # used only if the processor lacks patch_size/merge_size

    def _px_per_tok(self) -> int:
        """Pixels per vision token = (patch_size·merge_size)², read from the ACTUAL processor.
        Qwen3.5: 16·2 → 1024."""
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
