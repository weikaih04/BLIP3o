"""3D-generation tasks: text→3D / image→3D / multi-image→3D.

All three share:
  - manifest format (unified ready_v1 schema)
  - target loaders (ss/shape/tex latents)
  - collator (chat-template → VLM inputs + stacked 3D targets)

Differences are PURELY in view sampling (T=0 views, I1=1 view, IM=n∈[2,max_views]).
We model that as one base class + three subclasses, each registered under its
own task name. From the mixture config's view, they look like three different
tasks; in code they share ~95% of the logic.

Model contract: the resulting batch carries `_task` + the same fields the
existing `TrellisNativeVLM._forward_3d` expects (input_ids, pixel_values,
target_ss_latent, target_shape_slat_512, target_tex_slat_512, ...).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ... import _paths  # noqa: F401 — ensures trellis2 + blip3o on sys.path
from ..registry import register_task

from ...tr2_modules import (
    load_norm_stats,
    SS_FLOW_CONFIG_PATH,
    SHAPE_SLAT_CONFIG_PATH,
    TEX_SLAT_CONFIG_PATH,
)
from trellis2.modules.sparse import SparseTensor          # type: ignore
from trellis2.datasets.structured_latent import SLat      # type: ignore
from trellis2.datasets.structured_latent_svpbr import SLatPbr  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# Shared base
# ────────────────────────────────────────────────────────────────────────────
class _ThreeDTaskBase(Dataset):
    """Unified-schema manifest reader + 3D-target loaders + view sampler.

    Subclasses pick the view-sampling MODE: 'T' (no images), 'I1' (single
    random view), 'IM' (n random views, n ~ U[2, max_views]).

    Init args (forwarded from yaml `args:`):
      manifest        : path to ready_v1-style JSONL.
      slat_resolution : 512 | 1024 — which shape/pbr latent to load.
      crop_to_object  : if True and render is RGBA, alpha-bbox crop before RGB.
      max_views       : upper bound for IM (uniform [2, max_views], inclusive).
      min_aesthetic   : drop records below this aesthetic_score (None → keep).
      require_caption : (T-task only) drop records without captions.
      ss_only         : skip loading shape/tex latents (v1 SS-only training).
    """

    mode: str = ""        # set by subclass: "T" | "I1" | "IM"
    task_name: str = ""   # set by @register_task

    def __init__(
        self,
        manifest: str,
        slat_resolution: int = 512,
        crop_to_object: bool = False,
        max_views: int = 4,
        min_aesthetic: Optional[float] = None,
        require_caption: Optional[bool] = None,   # default: True for T, False else
        ss_only: bool = False,
    ):
        super().__init__()
        self.manifest_path = manifest
        self.slat_resolution = int(slat_resolution)
        self.crop_to_object = bool(crop_to_object)
        self.max_views = int(max_views)
        self.ss_only = bool(ss_only)
        if require_caption is None:
            require_caption = (self.mode == "T")
        self.require_caption = bool(require_caption)

        # ── load manifest ──
        records: List[Dict[str, Any]] = []
        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        # ── filters ──
        before = len(records)
        if min_aesthetic is not None:
            min_aes = float(min_aesthetic)
            records = [
                r for r in records
                if r.get("aesthetic_score") is None
                or float(r["aesthetic_score"]) >= min_aes
            ]
        if self.require_caption:
            records = [r for r in records if r.get("captions")]
        self.records = records
        print(
            f"[{self.task_name or self.__class__.__name__}] {manifest}: "
            f"{before} → {len(self.records)} after filters "
            f"(min_aesthetic={min_aesthetic}, require_caption={self.require_caption})"
        )

        # ── TRELLIS norm stats (read once, identical to upstream training) ──
        self.ss_norm        = load_norm_stats(SS_FLOW_CONFIG_PATH, "normalization")
        self.shape_norm     = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
        self.tex_pbr_norm   = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
        self.tex_shape_norm = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")

    def __len__(self):
        return len(self.records)

    # --- per-batch param hook (called by MixtureIterableDataset) ---
    # When the mixture is about to emit `batch_size` items of this task, it calls
    # `set_batch_params(rng)` once. For mode="IM" we lock n_views for the whole
    # batch → ZERO intra-batch padding waste on vision tokens (a 2-view item next
    # to a 4-view item would otherwise force the short one to pad to the long one).
    # For "T" / "I1" there's no batch-level decision to make; the method is a no-op.
    _batch_n_views: Optional[int] = None

    def set_batch_params(self, rng) -> None:
        if self.mode == "IM":
            hi = max(3, self.max_views + 1)
            self._batch_n_views = int(rng.integers(2, hi))

    def clear_batch_params(self) -> None:
        self._batch_n_views = None

    # --- view sampling (mode-dependent) ---
    def _sample_n_views(self, rng) -> int:
        if self.mode == "T":
            return 0
        if self.mode == "I1":
            return 1
        if self.mode == "IM":
            # Use batch-locked value when mixture set one; otherwise fall back to
            # per-item random (e.g. when this dataset is iterated standalone).
            if self._batch_n_views is not None:
                return int(self._batch_n_views)
            hi = max(3, self.max_views + 1)
            return int(rng.integers(2, hi))
        raise ValueError(f"unknown mode {self.mode!r}")

    # --- 3D target loaders (mirror upstream get_instance exactly) ---
    def _load_ss(self, path: str) -> torch.Tensor:
        a = np.load(path)
        npz_key = "z" if "z" in a.files else "latent"
        z = torch.tensor(a[npz_key]).float()
        if self.ss_norm is not None:
            z = (z - self.ss_norm["mean"]) / self.ss_norm["std"]
        return z

    def _load_shape(self, path: str) -> Dict[str, torch.Tensor]:
        a = np.load(path)
        coords = torch.tensor(a["coords"]).int()
        feats = torch.tensor(a["feats"]).float()
        if self.shape_norm is not None:
            feats = (feats - self.shape_norm["mean"]) / self.shape_norm["std"]
        return {"coords": coords, "feats": feats}

    def _load_tex(self, tex_path: str, shape_path: str) -> Dict[str, SparseTensor]:
        data = np.load(tex_path)
        coords = torch.tensor(data["coords"]).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        feats = torch.tensor(data["feats"]).float()
        if self.tex_pbr_norm is not None:
            feats = (feats - self.tex_pbr_norm["mean"]) / self.tex_pbr_norm["std"]
        pbr_z = SparseTensor(feats, coords)

        data = np.load(shape_path)
        coords = torch.tensor(data["coords"]).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        feats = torch.tensor(data["feats"]).float()
        if self.tex_shape_norm is not None:
            feats = (feats - self.tex_shape_norm["mean"]) / self.tex_shape_norm["std"]
        shape_z = SparseTensor(feats, coords)

        assert torch.equal(shape_z.coords, pbr_z.coords), (
            f"Shape and PBR sparse coords differ ({shape_z.coords.shape} vs "
            f"{pbr_z.coords.shape}). Upstream encoder guarantees they share layout."
        )
        return {"x_0": pbr_z, "concat_cond": shape_z}

    # --- image loading + optional alpha crop ---
    def _load_views(self, renders_dir: str, view_ids: Sequence[int]) -> List[Image.Image]:
        out = []
        for v in view_ids:
            path = None
            for ext in ("webp", "png", "jpg"):
                cand = os.path.join(renders_dir, f"{v:03d}.{ext}")
                if os.path.isfile(cand):
                    path = cand
                    break
            if path is None:
                continue
            img = Image.open(path)
            if self.crop_to_object and img.mode == "RGBA":
                img = self._alpha_crop(img)
            out.append(img.convert("RGB"))
        return out

    @staticmethod
    def _alpha_crop(img: Image.Image) -> Image.Image:
        a = np.array(img.getchannel("A"))
        ys, xs = np.where(a > 0)
        if ys.size == 0:
            return img
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        h = max(x1 - x0, y1 - y0) / 2.0
        return img.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h)))

    # ------------------------------------------------------------------
    # __getitem__: emit raw caption + PIL images + 3D target tensors
    # ------------------------------------------------------------------
    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.records[i]
        rng = np.random.default_rng()

        caps: List[str] = [c for c in (rec.get("captions") or []) if c]
        caption = str(rng.choice(caps)) if (self.mode == "T" and caps) else ""

        images: List[Image.Image] = []
        if self.mode in ("I1", "IM"):
            n_avail = int(rec.get("n_views", 16))
            n = min(self._sample_n_views(rng), n_avail)
            ids = sorted(int(v) for v in rng.choice(n_avail, size=n, replace=False))
            images = self._load_views(rec["renders_dir"], ids)

        res = self.slat_resolution
        shape_path = rec.get(f"shape_latent_{res}")
        pbr_path   = rec.get(f"pbr_latent_{res}")
        ss_path    = rec["ss_latent_64"]

        data: Dict[str, Any] = {
            "_task": self.task_name,
            "caption": caption,
            "images": images,
            "id": rec.get("sha256", f"idx_{i}"),
            "target_ss_latent": self._load_ss(ss_path),
        }
        if not self.ss_only:
            if shape_path:
                data["target_shape_slat_512_item"] = self._load_shape(shape_path)
            if pbr_path and shape_path:
                data["target_tex_slat_512_item"] = self._load_tex(pbr_path, shape_path)
        return data

    # ------------------------------------------------------------------
    # Collator: chat-template → VLM inputs + stack 3D targets
    # (port of the original NativeVLMCollator.__call__, made task-aware)
    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch, processor, *,
                   max_tokens_single: int = 4096,
                   token_budget: int = 8192) -> Dict[str, Any]:
        texts, flat_images = [], []
        px_per_tok = _px_per_tok(processor)

        for inst in batch:
            imgs = inst["images"]
            n = max(len(imgs), 1)
            per_tok = min(max_tokens_single, token_budget // n)
            max_px = per_tok * px_per_tok
            imgs = [_cap_image(im, max_px) for im in imgs]
            content: List[Dict[str, Any]] = [{"type": "image"} for _ in imgs]
            content.append({"type": "text", "text": inst["caption"]})
            messages = [{"role": "user", "content": content}]
            texts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            flat_images.extend(imgs)

        proc_kwargs = dict(text=texts, padding=True, return_tensors="pt")
        if flat_images:
            proc_kwargs["images"] = flat_images
        enc = processor(**proc_kwargs)

        out: Dict[str, Any] = {
            "_task": batch[0]["_task"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target_ss_latent": torch.stack(
                [inst["target_ss_latent"] for inst in batch], dim=0),
        }
        if "pixel_values" in enc:
            out["pixel_values"] = enc["pixel_values"]
            out["image_grid_thw"] = enc.get("image_grid_thw")
        if "pixel_values_videos" in enc:
            out["pixel_values_videos"] = enc["pixel_values_videos"]
            out["video_grid_thw"] = enc.get("video_grid_thw")

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


# ────────────────────────────────────────────────────────────────────────────
# Helpers (processor introspection + image capping)
# ────────────────────────────────────────────────────────────────────────────
def _px_per_tok(processor) -> int:
    ip = processor.image_processor
    ps, ms = getattr(ip, "patch_size", None), getattr(ip, "merge_size", None)
    if isinstance(ps, int) and isinstance(ms, int):
        return (ps * ms) ** 2
    return 1024  # Qwen3.5 default (patch_size=16, merge_size=2 → 1024)


def _cap_image(img: Image.Image, max_px: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_px:
        return img
    s = (max_px / float(w * h)) ** 0.5
    return img.resize((max(32, round(w * s)), max(32, round(h * s))), Image.LANCZOS)


# ────────────────────────────────────────────────────────────────────────────
# Three registered task classes (differ only in `mode`)
# ────────────────────────────────────────────────────────────────────────────
@register_task("text_to_3d")
class TextTo3DDataset(_ThreeDTaskBase):
    mode = "T"


@register_task("image_to_3d")
class ImageTo3DDataset(_ThreeDTaskBase):
    mode = "I1"


@register_task("multi_image_to_3d")
class MultiImageTo3DDataset(_ThreeDTaskBase):
    mode = "IM"
