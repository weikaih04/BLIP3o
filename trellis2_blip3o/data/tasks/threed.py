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
from ...vlm_collate import MAX_TOKENS_SINGLE, TOKEN_BUDGET, collate_vlm_3d

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
        max_slat_tokens: int = 8192,   # cap shape/pbr voxel count (TRELLIS max_tokens=8192);
                                        # oversized assets blow sparse activation memory → OOM.
                                        # 0 disables. Enforced in _load_one by resampling.
        # VLM-hidden cache (vlm_cache.py): when set, I1/T items load precomputed
        # cond_hidden npz instead of images (no PIL/processor/VLM at train time).
        # IM is NOT per-view cacheable (joint attention) → forbidden here.
        cached_hidden_root: Optional[str] = None,
        # fusion: also emit cached DINOv3 tokens (d-keys, same root) for the same view.
        fuse_dino: bool = False,
        # packed shards (scripts/pack_webdataset.py): when set, cond/ss/shape/pbr are
        # read by os.pread from big .tar shards (typically on local NVMe) instead of
        # per-sample small files on Lustre — kills the 8-rank small-file open contention.
        packed_root: Optional[str] = None,
    ):
        super().__init__()
        self.cached_hidden_root = cached_hidden_root or None
        self.fuse_dino = bool(fuse_dino)
        self.packed_root = packed_root or None
        self._packed = None   # lazy PackedShardReader, built per worker on first read
        self._im_combo_sizes: list = []
        if cached_hidden_root and self.mode == "IM":
            # IM is cacheable ONLY via pinned combos (m-keys: joint VLM hidden per
            # FIXED sha-seeded view set — free random combos are NOT cacheable since
            # views attend jointly inside one VLM sequence).
            from ...vlm_cache import check_meta
            meta = check_meta(cached_hidden_root)
            if not meta.get("im_combo_sizes"):
                raise ValueError("[vlm_cache] IM needs pinned-combo m-entries — run "
                                 "build_vlm_cache.py --mode combos first (free random "
                                 "multi-view combos are not cacheable)")
            self._im_combo_sizes = [int(s) for s in meta["im_combo_sizes"]]
        if self.fuse_dino and not self.cached_hidden_root:
            raise ValueError("[fusion] fuse_dino=True requires cached_hidden_root "
                             "(DINO tokens are served from the cache, d-keys)")
        self._cache_views = 0
        if self.cached_hidden_root:
            from ...vlm_cache import check_meta
            meta = check_meta(self.cached_hidden_root)
            # cache covers views [0, max_views) per asset — sample ONLY within coverage
            self._cache_views = int(meta.get("max_views", 4))
            if self.fuse_dino:
                if "dino_model" not in meta:
                    raise ValueError(f"[fusion] cache {self.cached_hidden_root} has no DINO "
                                     "entries (_meta.json lacks dino_model) — run "
                                     "scripts/build_dino_cache.py first")
                # DINO coverage may be narrower than VLM coverage — sample within BOTH
                self._cache_views = min(self._cache_views,
                                        int(meta.get("dino_max_views", 4)))
        self.manifest_path = manifest
        self.slat_resolution = int(slat_resolution)
        self.crop_to_object = bool(crop_to_object)
        self.max_views = int(max_views)
        self.ss_only = bool(ss_only)
        self.max_slat_tokens = int(max_slat_tokens or 0)
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
        # node-local restriction: with packed shards staged on this node's NVMe, keep ONLY
        # records whose sha is present locally (so multi-node runs train on disjoint slices,
        # and we drop the handful of assets that never got a cond-cache entry). NO Lustre
        # fallback to other nodes' data.
        n_pre_pack = len(records)
        if self.packed_root:
            local = self._reader().shas()
            records = [r for r in records if r.get("sha256") in local]
        self.records = records
        print(
            f"[{self.task_name or self.__class__.__name__}] {manifest}: "
            f"{before} → {len(self.records)} after filters "
            f"(min_aesthetic={min_aesthetic}, require_caption={self.require_caption}"
            + (f", packed-local {n_pre_pack}→{len(records)}" if self.packed_root else "") + ")"
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
    # ── packed-shard read backend (os.pread from big NVMe .tar; else the file path) ──
    def _reader(self):
        if self._packed is None:
            from ..packed_reader import PackedShardReader
            self._packed = PackedShardReader(self.packed_root)
        return self._packed

    def _src(self, sha: Optional[str], ext: str, path: str):
        """np.load source: packed BytesIO (packed_root set & sha/ext present) else `path`."""
        if self.packed_root and sha and self._reader().has(sha, ext):
            import io
            return io.BytesIO(self._reader().read(sha, ext))
        return path

    def _load_cond(self, sha: str, key: str):
        """cond entry (Qwen hidden + inline DINO). Packed v00 when available, else file.
        Replicates vlm_cache.load_entry's field mapping exactly."""
        from ...vlm_cache import load_entry, view_key
        if self.packed_root and sha and key == view_key(0) and self._reader().has(sha, "cond"):
            import io
            a = np.load(io.BytesIO(self._reader().read(sha, "cond")))
            out = {"cond_hidden": torch.from_numpy(a["hidden"]),
                   "cond_keep_mask": torch.from_numpy(a["keep_mask"])}
            if "dino_hidden" in a.files:
                out["dino_hidden"] = torch.from_numpy(a["dino_hidden"])
                out["dino_keep_mask"] = torch.from_numpy(a["dino_keep_mask"])
            for k in ("views", "dino_view_ids"):
                if k in a.files:
                    out[k] = torch.from_numpy(a[k])
            return out
        return load_entry(self.cached_hidden_root, sha, key)

    def _load_ss(self, path: str, sha: Optional[str] = None) -> torch.Tensor:
        a = np.load(self._src(sha, "ss", path))
        npz_key = "z" if "z" in a.files else "latent"
        z = torch.tensor(a[npz_key]).float()
        if self.ss_norm is not None:
            z = (z - self.ss_norm["mean"]) / self.ss_norm["std"]
        return z

    def _load_shape(self, path: str, sha: Optional[str] = None) -> Dict[str, torch.Tensor]:
        a = np.load(self._src(sha, "shape", path))
        coords = torch.tensor(a["coords"]).int()
        feats = torch.tensor(a["feats"]).float()
        if self.shape_norm is not None:
            feats = (feats - self.shape_norm["mean"]) / self.shape_norm["std"]
        return {"coords": coords, "feats": feats}

    def _load_tex(self, tex_path: str, shape_path: str, sha: Optional[str] = None) -> Dict[str, SparseTensor]:
        data = np.load(self._src(sha, "pbr", tex_path))
        coords = torch.tensor(data["coords"]).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        feats = torch.tensor(data["feats"]).float()
        if self.tex_pbr_norm is not None:
            feats = (feats - self.tex_pbr_norm["mean"]) / self.tex_pbr_norm["std"]
        pbr_z = SparseTensor(feats, coords)

        data = np.load(self._src(sha, "shape", shape_path))
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
    # Qwen's smart_resize divides by zero on a degenerate (0- or 1-px) image
    # dimension, and crashes in the COLLATOR (processor), AFTER __getitem__'s
    # resample guard — so a tiny/empty alpha crop would kill the whole run.
    # We enforce a minimum image size here so the processor never sees a
    # degenerate image. (Real culprit: assets whose render has a 1-px / empty
    # alpha region → alpha-crop → 0×0 or 1×1.)
    MIN_IMG_PX = 64

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
            img = img.convert("RGB")
            # Safety: upscale any degenerate / tiny image to a processor-safe size
            # (smart_resize needs ≥ one patch; a 1×1 crop → div-by-zero).
            w, h = img.size
            if w < self.MIN_IMG_PX or h < self.MIN_IMG_PX:
                s = self.MIN_IMG_PX / max(1, min(w, h))
                img = img.resize((max(self.MIN_IMG_PX, int(round(w * s))),
                                  max(self.MIN_IMG_PX, int(round(h * s)))),
                                 Image.LANCZOS)
            out.append(img)
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
        # Half-size: never let a thin/single-pixel object collapse the box to 0.
        h = max(x1 - x0, y1 - y0, 1) / 2.0
        box = (int(cx - h), int(cy - h), int(cx + h), int(cy + h))
        cropped = img.crop(box)
        # crop() can still yield a tiny image; the _load_views min-size guard
        # backstops it, but bail to the original if somehow empty.
        return cropped if min(cropped.size) > 0 else img

    # ------------------------------------------------------------------
    # __getitem__: emit raw caption + PIL images + 3D target tensors.
    #
    # Resilient-load (matches TRELLIS official StandardDatasetBase.__getitem__):
    # if an asset fails to load (missing/corrupt latent — e.g. a Phase-B encode
    # gap that slipped past `--filter_trainable`, or a transient weka IO fault),
    # resample a DIFFERENT index instead of crashing the whole run. Bounded tries
    # so a pathological manifest can't loop forever.
    # ------------------------------------------------------------------
    def __getitem__(self, i: int) -> Dict[str, Any]:
        n = len(self.records)
        for attempt in range(8):
            try:
                return self._load_one(i)
            # Resample on ANY per-sample load failure. Truncated/partial npz (rclone
            # pulls leave some files incomplete; filter_trainable only stat'd existence,
            # not integrity) raise EOFError / zipfile.BadZipFile / zlib.error — none of
            # which are OSError — so a single corrupt file would otherwise kill the whole
            # DDP run (observed: rank-5 EOFError in _load_shape, job 16197). A genuine
            # systematic bug still surfaces via the unguarded final attempt below.
            except Exception as e:
                bad = self.records[i].get("sha256", i)
                print(f"[{self.task_name}] load failed for {bad}: {e!r} — resampling")
                i = int(np.random.default_rng().integers(0, n)) if n > 1 else i
        # Last attempt unguarded → surface the real error if every resample failed.
        return self._load_one(i)

    def _load_one(self, i: int) -> Dict[str, Any]:
        rec = self.records[i]
        rng = np.random.default_rng()

        caps: List[str] = [c for c in (rec.get("captions") or []) if c]
        caption = str(rng.choice(caps)) if (self.mode == "T" and caps) else ""

        # ── VLM-hidden cache fast path: load precomputed cond instead of images ──
        # (the collate routes batches with `cond_hidden` straight past the processor;
        # the model's forward(cond_hidden=...) skips the VLM entirely.)
        if self.cached_hidden_root:
            from ...vlm_cache import load_entry, view_key, caption_key, dino_key, combo_key
            sha = rec.get("sha256", f"idx_{i}")
            if self.mode == "I1":
                n_avail = min(int(rec.get("n_views", 16)), self._cache_views)
                view = int(rng.integers(0, max(1, n_avail)))
                key = view_key(view)
            elif self.mode == "IM":
                # honor the batch-locked view count (homogeneous batches): combo id is
                # size-indexed (m00=2 views, m01=3, m02=4 by producer convention).
                view = None
                n = self._sample_n_views(rng)
                n = max(min(n, max(self._im_combo_sizes)), min(self._im_combo_sizes))
                key = combo_key(self._im_combo_sizes.index(n))
            else:  # T — caption index keyed (cache built over the captions list order)
                view = None
                key = caption_key(int(rng.integers(0, max(1, len(caps)))))
            entry = self._load_cond(sha, key)  # packed v00 if available; FileNotFoundError → resample
            data: Dict[str, Any] = {
                "_task": self.task_name,
                "id": sha,
                "cond_hidden": entry["cond_hidden"],
                "cond_keep_mask": entry["cond_keep_mask"],
                "target_ss_latent": self._load_ss(rec["ss_latent_64"], rec.get("sha256")),
            }
            if self.fuse_dino and self.mode in ("I1", "IM"):
                if "dino_hidden" in entry:
                    # merged v-entry (schema 2) or IM combo m-entry: DINO arrays inline
                    data["dino_hidden"] = entry["dino_hidden"]
                    data["dino_keep_mask"] = entry["dino_keep_mask"]
                    data["dino_view_ids"] = entry.get(
                        "dino_view_ids",
                        torch.zeros(entry["dino_hidden"].shape[0], dtype=torch.long))
                else:
                    # legacy: DINO tokens in a separate d-key file (2nd open)
                    dentry = load_entry(self.cached_hidden_root, sha, dino_key(view))
                    data["dino_hidden"] = dentry["cond_hidden"]        # (N_d, 1024) fp16
                    data["dino_keep_mask"] = dentry["cond_keep_mask"]  # (N_d,) bool
                    data["dino_view_ids"] = torch.zeros(
                        dentry["cond_hidden"].shape[0], dtype=torch.long)
            self._attach_slat_targets(data, rec)
            return data

        images: List[Image.Image] = []
        if self.mode in ("I1", "IM"):
            n_avail = int(rec.get("n_views", 16))
            n = min(self._sample_n_views(rng), n_avail)
            ids = sorted(int(v) for v in rng.choice(n_avail, size=n, replace=False))
            images = self._load_views(rec["renders_dir"], ids)

        data: Dict[str, Any] = {
            "_task": self.task_name,
            "caption": caption,
            "images": images,
            "id": rec.get("sha256", f"idx_{i}"),
            "target_ss_latent": self._load_ss(rec["ss_latent_64"], rec.get("sha256")),
        }
        self._attach_slat_targets(data, rec)
        return data

    def _attach_slat_targets(self, data: Dict[str, Any], rec: Dict[str, Any]) -> None:
        """Shared by the live path and the vlm_cache fast path."""
        if self.ss_only:
            return
        sha = rec.get("sha256")
        res = self.slat_resolution
        shape_path = rec.get(f"shape_latent_{res}")
        pbr_path = rec.get(f"pbr_latent_{res}")
        if shape_path:
            shape_item = self._load_shape(shape_path, sha)
            # Voxel cap: oversized sparse SLAT spikes activation memory → OOM
            # (TRELLIS caps at max_tokens=8192). Raise ValueError so __getitem__'s
            # resample loop picks a different (smaller) asset rather than OOM-ing
            # the whole 8-GPU run on one pathological sample.
            if self.max_slat_tokens > 0:
                nvox = int(shape_item["coords"].shape[0])
                if nvox > self.max_slat_tokens:
                    raise ValueError(
                        f"shape SLAT voxels {nvox} > max_slat_tokens "
                        f"{self.max_slat_tokens} (asset {rec.get('sha256','?')[:12]})"
                    )
            data["target_shape_slat_512_item"] = shape_item
        if pbr_path and shape_path:
            data["target_tex_slat_512_item"] = self._load_tex(pbr_path, shape_path, sha)

    # ------------------------------------------------------------------
    # Collator: chat-template → VLM inputs + stack 3D targets
    # (port of the original NativeVLMCollator.__call__, made task-aware)
    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch, processor, *,
                   max_tokens_single: int = MAX_TOKENS_SINGLE,
                   token_budget: int = TOKEN_BUDGET,
                   target_tokens_per_view: Optional[int] = None) -> Dict[str, Any]:
        """Delegates to the UNIFIED collator (trellis2_blip3o/vlm_collate.py) — single
        source of truth shared with the inference harness's NativeVLMCollator, so the
        training and inference input pipelines cannot drift again."""
        return collate_vlm_3d(
            batch, processor,
            max_tokens_single=max_tokens_single,
            token_budget=token_budget,
            target_tokens_per_view=target_tokens_per_view,
        )


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
