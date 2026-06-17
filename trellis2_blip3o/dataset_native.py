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
from .vlm_collate import MAX_TOKENS_SINGLE, TOKEN_BUDGET, collate_vlm_3d
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
        # View sampling: TRELLIS official randomly samples a cond view per step (viewpoint
        # augmentation → view-robust model). When True, _getitem_legacy draws num_cond_views
        # random views from the asset's 16 renders instead of always views[:N] (000.png).
        # Training sets this True; inference leaves it False (deterministic view 000).
        self.random_cond_view = bool(getattr(data_args, "random_cond_view", False))

        # SLAT voxel-count cap — matches TRELLIS official slat_flow_*_512 config
        # (`max_tokens: 8192`). Oversized assets blow up sparse activation memory
        # (activation ∝ voxel count); the elastic GC controller reacts AFTER the
        # fact and one giant outlier can still OOM or skew its linear fit. We cap
        # at the SOURCE like TRELLIS does. 0/None disables. Only meaningful when
        # SLAT is actually loaded (ss_only=False); SS latent is a fixed 16³ dense
        # tensor so it's never capped. Enforced in __getitem__ by RESAMPLING a
        # different index (drop-style, not crop — preserves voxel layout).
        self.max_slat_tokens = int(getattr(data_args, "max_slat_tokens", 8192) or 0)
        self._cap_resample_tries = 0  # diagnostic counter

        # --- unified-schema knobs ---
        mix_spec = getattr(data_args, "task_mix", None) or "T:0.2,I1:0.4,IM:0.4"
        self.task_mix = _parse_task_mix(mix_spec)
        self.max_views = int(getattr(data_args, "max_views", 4))
        self.slat_resolution = int(getattr(data_args, "slat_resolution", 512))
        self.crop_to_object = bool(getattr(data_args, "crop_to_object", False))

        # LOUD config echo (rank0): every knob above is read via getattr(..., default) — a
        # typo'd field name in the yaml/launcher would SILENTLY fall back to the default.
        # Printing the resolved values is the only way such a typo is ever noticed.
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[TR2NativeVLMDataset] resolved config: num_cond_views={self.num_cond_views} "
                  f"random_cond_view={self.random_cond_view} max_slat_tokens={self.max_slat_tokens} "
                  f"task_mix={self.task_mix} max_views={self.max_views} "
                  f"slat_resolution={self.slat_resolution} crop_to_object={self.crop_to_object} "
                  f"ss_only={self.ss_only}", flush=True)

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
        # Cap SLAT voxel count by resampling oversized assets to a different index
        # (TRELLIS official drops them at init via a precomputed token column; we
        # don't have that column, so we drop lazily on load). Bounded retries so a
        # pathological manifest can't loop forever — falls through to return the
        # last (oversized) sample rather than hang.
        n = len(self.records)
        for attempt in range(8):
            rec = self.records[i]
            data = (self._getitem_unified(rec, i) if "renders_dir" in rec
                    else self._getitem_legacy(rec, i))
            if self.ss_only or self.max_slat_tokens <= 0:
                return data
            ntok = self._slat_token_count(data)
            if ntok <= self.max_slat_tokens:
                return data
            # Oversized → resample a different index deterministically-ish.
            self._cap_resample_tries += 1
            i = (i + 1 + attempt) % n
        return data  # gave up after retries; let it through (elastic GC will cope)

    @staticmethod
    def _slat_token_count(data: Dict[str, Any]) -> int:
        """Voxel/token count of the loaded SLAT target (shape SLAT coords). 0 if absent."""
        shp = data.get("target_shape_slat_512_item")
        if isinstance(shp, dict) and "coords" in shp:
            return int(shp["coords"].shape[0])
        return 0

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
            # IM_VIEWS env (eval diagnostics): force an EXACT view list so resolution /
            # config arms compare on identical views. e.g. IM_VIEWS="0,4,8".
            _force = os.environ.get("IM_VIEWS")
            if _force and task == "IM":
                view_ids = sorted(int(v) for v in _force.split(",") if v.strip())
                view_ids = [v for v in view_ids if v < n_avail]
            else:
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
        """Square crop centered on alpha bbox — matches TRELLIS.2 ImageConditionedMixin.
        (Edge-case guards backported from data/tasks/threed.py so the two paths agree.)"""
        a = np.array(img.getchannel("A"))
        ys, xs = np.where(a > 0)
        if ys.size == 0:
            return img
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Half-size: never let a thin/single-pixel object collapse the box to 0.
        h = max(x1 - x0, y1 - y0, 1) / 2.0
        box = (int(cx - h), int(cy - h), int(cx + h), int(cy + h))
        cropped = img.crop(box)
        # crop() can still yield a tiny/empty image — bail to the original if so.
        return cropped if min(cropped.size) > 0 else img

    # ------------------------------------------------------------------
    # Legacy schema — unchanged behavior
    # ------------------------------------------------------------------
    def _getitem_legacy(self, rec: Dict[str, Any], i: int) -> Dict[str, Any]:
        rtype = rec.get("type", "I_2_3D")
        caption = rec.get("txt", "") or ""

        images: List[Image.Image] = []
        if rtype == "I_2_3D" and "image" in rec:
            views = rec.get("multi_view_renders") or [rec["image"]]
            if self.random_cond_view and len(views) > self.num_cond_views:
                import random
                chosen = random.sample(views, self.num_cond_views)   # viewpoint augmentation
            else:
                chosen = views[: self.num_cond_views]                # deterministic (view 000)
            for p in chosen:
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
    """Thin wrapper over the UNIFIED collator (trellis2_blip3o/vlm_collate.py).

    `processor` = AutoProcessor.from_pretrained(vlm_model). Emits exactly the
    kwargs TrellisNativeVLM.forward expects.

    HISTORY: this used to carry its own (drifted) copy of the collate body with
    budgets 400/1000 while training (data/tasks/threed.py) ran 4096/8192 — a LIVE
    train/infer mismatch (e.g. 1024² eval renders: training saw 1024 vision
    tok/view, this collator capped the same image to 400). Both now delegate to
    vlm_collate.collate_vlm_3d with ONE set of budgets (the training-side values).
    """
    processor: Any
    system_prompt: Optional[str] = None  # keep minimal / None (drop_idx deferred)
    max_tokens_single: int = MAX_TOKENS_SINGLE   # unified w/ training (was 400)
    token_budget: int = TOKEN_BUDGET             # unified w/ training (was 1000)
    target_tokens_per_view: Optional[int] = None  # v3 upscale knob; None → vlm_collate module default

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        return collate_vlm_3d(
            list(instances), self.processor,
            max_tokens_single=self.max_tokens_single,
            token_budget=self.token_budget,
            target_tokens_per_view=self.target_tokens_per_view,
            system_prompt=self.system_prompt,
        )
