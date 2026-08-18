"""Weighted-caption T task adapter for the v22_tcap4 cache (text-only S3 pilot).

The stock TextTo3DDataset (threed.py, mode="T") samples t-keys UNIFORMLY over the
record's captions list. The tcap4 cache contract (its _meta.json, built 2026-07-13 by
scripts/build_tcapT_2n.sbatch) instead specifies training-time sampling weights over
the FIXED caption order [long, medium, short, long+texture]:
    t000=.35  t001=.20  t002=.10  t003=.35,  t003 -> t000 fallback for the ~20 train
assets that lack a texture caption (their captions list has 3 non-empty entries, so
t003 was never built for them).

This subclass keeps the base class's manifest/filters/loaders and overrides ONLY the
cached-cond `_load_one` (small-adapter pattern, cf. data/tap_ss_task.py): weighted
t-key choice per _meta.json, then the same _load_cond/_load_ss/_attach_slat_targets.

Items still stamp `_task: "text_to_3d"` (NOT this class's registry name) so that
  (a) MultiTaskCollator routes to the shared _ThreeDTaskBase.collate_fn, and
  (b) TrellisNativeVLM.forward's fuse_dino exemption for text batches keeps applying.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from ..registry import register_task
from ...vlm_cache import entry_path as _entry_path
from .threed import TextTo3DDataset


@register_task("text_to_3d_weighted")
class WeightedTextTo3DDataset(TextTo3DDataset):
    mode = "T"

    def __init__(self, *args, load_tex: bool = True, **kwargs):
        # load_tex=False (shape-stage jobs): skip the pbr latent entirely — the shape
        # flow never consumes target_tex_slat_512, and the pbr npz was ~1/3 of the
        # per-sample SLAT read cost on the unpacked mixture path.
        self.load_tex = bool(load_tex)
        super().__init__(*args, **kwargs)
        if not self.cached_hidden_root:
            raise ValueError("[text_to_3d_weighted] requires cached_hidden_root "
                             "(the t-key cache the weights belong to)")
        meta = json.load(open(os.path.join(self.cached_hidden_root, "_meta.json")))
        w = meta.get("t_sampling_weights") or {}
        if not w:
            # The v22_3dvlm_tok1024_mv1 _meta.json carries no t_sampling_weights,
            # so this class silently ran UNIFORM 0.25 each — the one thing it
            # exists to not do. Uniform triples the short caption (.10 -> .25)
            # and cuts the two long ones from .70 to .50 combined. Fall back to
            # the contract in this file's docstring instead of to uniform; a
            # cache that does specify weights still wins.
            w = {"t000": 0.35, "t001": 0.20, "t002": 0.10, "t003": 0.35}
        self._t_keys = sorted(w)
        p = np.asarray([float(w.get(k, 1.0)) for k in self._t_keys], dtype=np.float64)
        self._t_probs = p / p.sum()
        self._t_fallback = dict(meta.get("t_sampling_fallback") or {})
        print(f"[text_to_3d_weighted] t-key sampling "
              f"{dict(zip(self._t_keys, self._t_probs.round(3)))} "
              f"fallback={self._t_fallback or '{}'}")

    def _load_one(self, i: int) -> Dict[str, Any]:
        rec = self.records[i]
        rng = np.random.default_rng()
        # Cheap voxel PRE-check before any other IO. The capT manifest is NOT
        # voxel-filtered (~10% of assets exceed max_slat_tokens=8192); the base-class
        # order loads cond+ss+shape(+pbr) and only THEN rejects in _attach_slat_targets
        # — measured 2026-07-13 as a big share of the shape_textonly dataloader stall
        # (23k full-cost rejects in the first ~200 steps). Reading just the coords
        # member is a fraction of the full load; the accepted-path re-read hits the
        # warm page cache.
        if not self.ss_only and self.max_slat_tokens > 0:
            shape_path = rec.get(f"shape_latent_{self.slat_resolution}")
            if shape_path:
                nvox = int(np.load(self._src(rec.get("sha256"), "shape", shape_path))
                           ["coords"].shape[0])
                if nvox > self.max_slat_tokens:
                    raise ValueError(
                        f"shape SLAT voxels {nvox} > max_slat_tokens "
                        f"{self.max_slat_tokens} (asset {rec.get('sha256','?')[:12]}, precheck)")
        # builder index space: enumerate over the NON-EMPTY captions (build_vlm_cache_v22
        # --mode captions filters empties the same way; only the LAST variant can be absent)
        caps = [c for c in (rec.get("captions") or []) if c]
        sha = rec["sha256"]
        key = self._t_keys[int(rng.choice(len(self._t_keys), p=self._t_probs))]
        if int(key[1:]) >= len(caps):              # variant absent for this asset
            key = self._t_fallback.get(key, "t000")
        elif not os.path.exists(_entry_path(self.cached_hidden_root, sha, key)):
            # The manifest saying an asset has 4 captions does not mean the cache
            # has 4 t-keys: the t-cache was built 2026-07-13 against v4's capT
            # list, while capT400k_train.jsonl was re-joined from the caption
            # store on 2026-08-17. Measured, 149 assets have no t-key at all and
            # 7 have t000-t002 only; the count-based test above passes them
            # straight into a FileNotFoundError and a resample.
            key = self._t_fallback.get(key, "t000")
        entry = self._load_cond(sha, key)          # FileNotFoundError -> __getitem__ resamples
        data: Dict[str, Any] = {
            "_task": "text_to_3d",                 # ride the stock T-batch contract (see docstring)
            "id": sha,
            "cond_hidden": entry["cond_hidden"],
            "cond_keep_mask": entry["cond_keep_mask"],
            "target_ss_latent": self._load_ss(rec["ss_latent_64"], sha),
        }
        self._attach_slat_targets(data, rec)
        return data

    def _attach_slat_targets(self, data: Dict[str, Any], rec: Dict[str, Any]) -> None:
        if self.ss_only or self.load_tex:
            return super()._attach_slat_targets(data, rec)
        # shape-only variant: identical to the base method minus the pbr/tex load.
        sha = rec.get("sha256")
        shape_path = rec.get(f"shape_latent_{self.slat_resolution}")
        if shape_path:
            shape_item = self._load_shape(shape_path, sha)
            if self.max_slat_tokens > 0:
                nvox = int(shape_item["coords"].shape[0])
                if nvox > self.max_slat_tokens:
                    raise ValueError(
                        f"shape SLAT voxels {nvox} > max_slat_tokens "
                        f"{self.max_slat_tokens} (asset {(sha or '?')[:12]})")
            data["target_shape_slat_512_item"] = shape_item
