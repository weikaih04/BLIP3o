"""MDS / MosaicML-Streaming task for image_to_3d (S1/S2 I1 fusion).

Wraps a `StreamingDataset` over the MDS shards built by scripts/build_mds.py and yields
samples in the SAME dict shape as ThreeDTask._load_one (cached I1 fusion path), so the
existing collator + model forward are unchanged. StreamingDataset gives node/rank-aware
sharding + deterministic resume for free (the reason we use MDS).

Decode/normalize mirror ThreeDTask exactly (kept here to stay decoupled from the map-style
path); a round-trip test asserts byte/▒value identity against the packed reader.
"""
from __future__ import annotations
import io
from typing import Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .. import _paths  # noqa: F401
from ..tr2_modules import (
    load_norm_stats, SS_FLOW_CONFIG_PATH, SHAPE_SLAT_CONFIG_PATH, TEX_SLAT_CONFIG_PATH)
from trellis2.modules.sparse import SparseTensor  # type: ignore


class StreamingImageTo3D(IterableDataset):
    task_name = "image_to_3d"

    def __init__(self, mds_root: str, *, fuse_dino: bool = True, ss_only: bool = False,
                 max_slat_tokens: int = 8192, shuffle: bool = True, batch_size: int = 1,
                 shuffle_seed: int = 9176, cache_limit: Optional[str] = None):
        super().__init__()
        from streaming import StreamingDataset
        self.fuse_dino = bool(fuse_dino)
        self.ss_only = bool(ss_only)
        self.max_slat_tokens = int(max_slat_tokens or 0)
        # node/rank-aware sharding + shuffle + resume are handled internally by StreamingDataset.
        self.ds = StreamingDataset(local=mds_root, shuffle=shuffle, batch_size=batch_size,
                                   shuffle_seed=shuffle_seed, cache_limit=cache_limit)
        self.ss_norm        = load_norm_stats(SS_FLOW_CONFIG_PATH, "normalization")
        self.shape_norm     = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
        self.tex_pbr_norm   = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
        self.tex_shape_norm = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")

    # ── decoders (byte-for-byte mirror of ThreeDTask) ──
    def _ss(self, b):
        a = np.load(io.BytesIO(b)); k = "z" if "z" in a.files else "latent"
        z = torch.tensor(a[k]).float()
        if self.ss_norm is not None:
            z = (z - self.ss_norm["mean"]) / self.ss_norm["std"]
        return z

    def _shape(self, b):
        a = np.load(io.BytesIO(b))
        coords = torch.tensor(a["coords"]).int()
        feats = torch.tensor(a["feats"]).float()
        if self.shape_norm is not None:
            feats = (feats - self.shape_norm["mean"]) / self.shape_norm["std"]
        return {"coords": coords, "feats": feats}

    def _tex(self, pbr_b, shape_b):
        d = np.load(io.BytesIO(pbr_b))
        c = torch.tensor(d["coords"]).int(); c = torch.cat([torch.zeros_like(c[:, :1]), c], 1)
        f = torch.tensor(d["feats"]).float()
        if self.tex_pbr_norm is not None:
            f = (f - self.tex_pbr_norm["mean"]) / self.tex_pbr_norm["std"]
        pbr_z = SparseTensor(f, c)
        d = np.load(io.BytesIO(shape_b))
        c = torch.tensor(d["coords"]).int(); c = torch.cat([torch.zeros_like(c[:, :1]), c], 1)
        f = torch.tensor(d["feats"]).float()
        if self.tex_shape_norm is not None:
            f = (f - self.tex_shape_norm["mean"]) / self.tex_shape_norm["std"]
        shape_z = SparseTensor(f, c)
        return {"x_0": pbr_z, "concat_cond": shape_z}

    def _cond(self, b):
        a = np.load(io.BytesIO(b))
        out = {"cond_hidden": torch.from_numpy(a["hidden"]),
               "cond_keep_mask": torch.from_numpy(a["keep_mask"])}
        if "dino_hidden" in a.files:
            out["dino_hidden"] = torch.from_numpy(a["dino_hidden"])
            out["dino_keep_mask"] = torch.from_numpy(a["dino_keep_mask"])
        return out

    def _build(self, s):
        sha = s["sha"]
        entry = self._cond(s["cond"])
        data = {"_task": self.task_name, "id": sha,
                "cond_hidden": entry["cond_hidden"], "cond_keep_mask": entry["cond_keep_mask"],
                "target_ss_latent": self._ss(s["ss"])}
        if self.fuse_dino and "dino_hidden" in entry:
            data["dino_hidden"] = entry["dino_hidden"]
            data["dino_keep_mask"] = entry["dino_keep_mask"]
            data["dino_view_ids"] = torch.zeros(entry["dino_hidden"].shape[0], dtype=torch.long)
        if not self.ss_only:
            shape_item = self._shape(s["shape"])
            if self.max_slat_tokens > 0 and int(shape_item["coords"].shape[0]) > self.max_slat_tokens:
                raise ValueError("oversized")          # → caller skips (stream resample)
            data["target_shape_slat_512_item"] = shape_item
            if s.get("pbr"):
                data["target_tex_slat_512_item"] = self._tex(s["pbr"], s["shape"])
        return data

    def __iter__(self):
        for s in self.ds:
            try:
                yield self._build(s)
            except ValueError:
                continue                                # oversized SLAT → skip, take next
