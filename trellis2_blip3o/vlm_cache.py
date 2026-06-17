"""VLM hidden-state cache — precompute frozen-VLM outputs once, train without the VLM.

The VLM (Qwen3.5-2B) is FROZEN in all v3 training; for the I1 task its hidden states
depend only on (view image, chat template, target_tokens_per_view). So we precompute
hidden per (asset, view) and at train time skip: PIL loading, the 1024² resize, the
processor, and the entire VLM forward — AND skip building the VLM at all (~5-6 GB GPU
freed → bigger batch). See docs/V3_DISTILL_DESIGN.md (Stage-2 infra track).

Cache key/layout (GENERIC string keys — one mechanism for all cacheable tasks):
    {root}/{sha[:2]}/{sha}/{key}.npz
      I1 single-view : key = f"v{view:03d}"
      T  text caption: key = f"t{caption_idx:03d}"
      IM pinned combo: key = f"m{combo_id:02d}"   (combos sha-seeded & FIXED at produce
                       time — free random combos are NOT cacheable: views attend jointly)
      LM tasks (vqa/grounding/sft): NOT cacheable by construction (they train the VLM
                       itself, freeze_vlm=False → hidden is not a constant)
        hidden:    (T, H) fp16   — encode_cond output, UNPADDED true length
        keep_mask: (T,)   bool   — attention ∧ ¬boilerplate (the flow-cond mask)
    {root}/_meta.json — {vlm, target_tokens_per_view, crop_to_object, schema: 1}
ROOT NAMING CONVENTION: encode the contract in the dir name, e.g.
    vlm_hidden_cache/qwen35-2b_tok1024_crop/
A cache is only valid for runs whose (vlm, tok target, crop) match _meta.json — the
consumer verifies and refuses on mismatch (no silent train/infer drift).

LIMIT: I1 (single-view) and T (text, key=caption-hash) only. IM multi-view hiddens
are NOT cacheable per-view (views attend jointly inside one VLM sequence —
concat(single-view hiddens) ≠ multi-view hidden). Consumers must assert mode.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch


SCHEMA_VERSION = 1


def view_key(view_id: int) -> str:
    return f"v{view_id:03d}"


def dino_key(view_id: int) -> str:
    """Frozen-DINOv3 token cache for the SAME view (fusion cond's low-level segment).
    Lives in the SAME root as the VLM entries (same sha dir, same crop pipeline);
    contract fields dino_model/dino_image_size are added to _meta.json by the producer."""
    return f"d{view_id:03d}"


def caption_key(caption_idx: int) -> str:
    return f"t{caption_idx:03d}"


def combo_key(combo_id: int) -> str:
    return f"m{combo_id:02d}"


def entry_path(root: str, sha: str, key) -> str:
    if isinstance(key, int):          # backward-compat: int = view id
        key = view_key(key)
    return os.path.join(root, sha[:2], sha, f"{key}.npz")


def save_entry(root: str, sha: str, key,
               hidden: torch.Tensor, keep_mask: torch.Tensor, **extras) -> str:
    """hidden (T,H) any float dtype → fp16; keep_mask (T,) bool. Atomic write.
    extras: additional numpy-able arrays stored verbatim (e.g. IM combo entries
    carry views / dino_hidden / dino_keep_mask / dino_view_ids)."""
    p = entry_path(root, sha, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    np.savez(tmp, hidden=hidden.detach().to(torch.float16).cpu().numpy(),
             keep_mask=keep_mask.detach().bool().cpu().numpy(),
             **{k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
                for k, v in extras.items()})
    os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, p)
    return p


def load_entry(root: str, sha: str, key) -> Dict[str, torch.Tensor]:
    """Raises FileNotFoundError on miss (datasets resample; producers must be complete).
    MERGED format (schema 2, scripts/merge_vd_cache.py): a v-key npz may also carry
    dino_hidden/dino_keep_mask — returned under those keys when present, so fusion
    runs do ONE file open per sample instead of two (the 8-rank small-file random-read
    contention measured 2026-06-11 was the fusion +51% step-time culprit)."""
    a = np.load(entry_path(root, sha, key))
    out = {"cond_hidden": torch.from_numpy(a["hidden"]),         # (T,H) fp16
           "cond_keep_mask": torch.from_numpy(a["keep_mask"])}   # (T,) bool
    if "dino_hidden" in a.files:
        out["dino_hidden"] = torch.from_numpy(a["dino_hidden"])
        out["dino_keep_mask"] = torch.from_numpy(a["dino_keep_mask"])
    for k in ("views", "dino_view_ids"):                         # IM combo extras
        if k in a.files:
            out[k] = torch.from_numpy(a[k])
    return out


def write_meta(root: str, *, vlm: str, target_tokens_per_view: int,
               crop_to_object: bool, max_views: int = 4) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "_meta.json"), "w") as f:
        json.dump({"schema": SCHEMA_VERSION, "vlm": vlm,
                   "target_tokens_per_view": int(target_tokens_per_view),
                   "crop_to_object": bool(crop_to_object),
                   "max_views": int(max_views)}, f, indent=2)


def check_meta(root: str, *, vlm: Optional[str] = None,
               target_tokens_per_view: Optional[int] = None,
               crop_to_object: Optional[bool] = None) -> Dict:
    """Consumer-side contract check — refuse silently-mismatched caches."""
    mp = os.path.join(root, "_meta.json")
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"[vlm_cache] no _meta.json under {root} — not a cache root?")
    meta = json.load(open(mp))
    if meta.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"[vlm_cache] schema {meta.get('schema')} != {SCHEMA_VERSION}")
    for k, want in (("vlm", vlm), ("target_tokens_per_view", target_tokens_per_view),
                    ("crop_to_object", crop_to_object)):
        if want is not None and meta.get(k) != want:
            raise ValueError(f"[vlm_cache] contract mismatch: cache {k}={meta.get(k)!r} "
                             f"but run wants {want!r} ({root})")
    return meta


def collate_cached(batch, pad_value: float = 0.0) -> Dict[str, torch.Tensor]:
    """Pad per-sample (T_i, H) hiddens to (B, T_max, H) + bool mask (B, T_max).
    keep_mask doubles as the attention/key mask (padded positions False)."""
    hs = [b["cond_hidden"] for b in batch]
    ms = [b["cond_keep_mask"] for b in batch]
    B, T, H = len(hs), max(h.shape[0] for h in hs), hs[0].shape[1]
    hidden = hs[0].new_full((B, T, H), pad_value)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, (h, m) in enumerate(zip(hs, ms)):
        hidden[i, :h.shape[0]] = h
        mask[i, :m.shape[0]] = m
    return {"cond_hidden": hidden, "cond_keep_mask": mask}
