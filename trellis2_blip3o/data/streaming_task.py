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
import os
from typing import Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .. import _paths  # noqa: F401
from .rank_aware import install_iterable_shard_passthrough

# StreamingDataset already does node/rank-aware sharding; without this, accelerate
# shards it a SECOND time and each rank sees only 1/N of its own 1/N. See rank_aware.py.
install_iterable_shard_passthrough()
from ..repa import load_repa_target
from ..tr2_modules import (
    load_norm_stats, SS_FLOW_CONFIG_PATH, SHAPE_SLAT_CONFIG_PATH, TEX_SLAT_CONFIG_PATH)
from trellis2.modules.sparse import SparseTensor  # type: ignore


class StreamingImageTo3D(IterableDataset):
    task_name = "image_to_3d"
    # StreamingDataset partitions samples across ranks itself → tell accelerate not to
    # re-shard (that would drop (N-1)/N of every read AND shrink the epoch to 1/N**2).
    _rank_sharded = True

    def __init__(self, mds_root: Optional[str] = None, *, shard_dirs: Optional[list] = None,
                 fuse_dino: bool = True, ss_only: bool = False,
                 max_slat_tokens: int = 8192, shuffle: bool = True, batch_size: int = 1,
                 shuffle_seed: int = 9176, cache_limit: Optional[str] = None):
        super().__init__()
        from streaming import StreamingDataset, Stream
        self.fuse_dino = bool(fuse_dino)
        self.ss_only = bool(ss_only)
        self.max_slat_tokens = int(max_slat_tokens or 0)
        if shard_dirs:
            # NODE-SHARDED mode: this node holds only its owned shards locally (the MDS is too big
            # for /fsx to persist + replicating the full set to every node wastes build time/disk).
            # StreamingDataset's default partition spans the GLOBAL world (all 32 ranks); with only
            # 1/N present locally that silently drops (N-1)/N of the data. Fix: make StreamingDataset
            # shard NODE-LOCALLY — its 1/N split across this node's own ranks only. streaming reads
            # rank/world from ENV via streaming.base.distributed (NOT torch.distributed), so we
            # monkeypatch those three getters to node-local values. torch.distributed (DDP, already
            # initialized with the global world) is untouched; across nodes the disjoint shards tile
            # the full dataset. (8 local ranks × N nodes = global world → full coverage, no overlap.)
            import streaming.base.distributed as _sd
            _lws = _sd.get_local_world_size()      # ranks on THIS node (e.g. 8)
            _lr = _sd.get_local_rank()             # this rank within the node (0..lws-1)
            _sd.get_rank = lambda: _lr             # node-local rank
            _sd.get_world_size = lambda: _lws      # node-local world == one node
            # get_local_world_size stays == _lws  → World.detect ⇒ num_nodes=1, ranks_per_node=_lws
            streams = [Stream(local=d) for d in shard_dirs]
            self.ds = StreamingDataset(streams=streams, shuffle=shuffle, batch_size=batch_size,
                                       shuffle_seed=shuffle_seed, cache_limit=cache_limit)
        else:
            # FULL mode: every node has the whole MDS; StreamingDataset does global node/rank sharding.
            self.ds = StreamingDataset(local=mds_root, shuffle=shuffle, batch_size=batch_size,
                                       shuffle_seed=shuffle_seed, cache_limit=cache_limit)
        self.ss_norm        = load_norm_stats(SS_FLOW_CONFIG_PATH, "normalization")
        self.shape_norm     = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
        self.tex_pbr_norm   = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
        self.tex_shape_norm = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
        # REPA SS aux targets (repa.py): env REPA_ROOT (train_native exports it from
        # --repa_root before the dataset is built; workers inherit it). Unset → this
        # dataset's output is byte-identical to before (no repa keys emitted).
        # REPA_MIN_QUALITY (from --repa_min_quality): targets whose builder quality
        # score is below it load as None → aux weight 0 (gate: garbage VGGT clouds).
        self.repa_root = os.environ.get("REPA_ROOT", "") or None
        self.repa_min_quality = float(os.environ.get("REPA_MIN_QUALITY", "0.5"))

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
        if "dino_view_ids" in a.files:                      # IM combo m-entry: per-token view ordinal 0..n-1
            out["dino_view_ids"] = torch.from_numpy(a["dino_view_ids"])
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
            if "dino_view_ids" in entry:                    # IM (multi-view): real per-token 0..n-1 view ids
                data["dino_view_ids"] = entry["dino_view_ids"].long()
                # IM_VIEW_SUBSET=1 → TRELLIS-style view-count randomization (fusion pressure):
                # keep a random k∈{1..n} of the cached view blocks. Measured: always-4 training
                # gives zero pressure to fuse (the IM SS probe ignores distinct view content,
                # 4d≈4c≈1v) and regresses 1-view. Subsetting makes "this view alone must work"
                # and "extra views are complementary" both appear in training.
                # (The joint-4v qwen segment can't be subset — DINO carries the spatial signal.)
                if os.environ.get("IM_VIEW_SUBSET") == "1":
                    vids = data["dino_view_ids"]
                    uniq = torch.unique(vids)
                    if len(uniq) > 1:
                        k = int(torch.randint(1, len(uniq) + 1, ()).item())
                        keep_v = uniq[torch.randperm(len(uniq))[:k]]
                        m = torch.isin(vids, keep_v)
                        data["dino_hidden"] = data["dino_hidden"][m]
                        data["dino_keep_mask"] = data["dino_keep_mask"][m]
                        data["dino_view_ids"] = vids[m]
            else:                                           # single-image: no view axis → all zeros (unchanged)
                data["dino_view_ids"] = torch.zeros(entry["dino_hidden"].shape[0], dtype=torch.long)
        if self.repa_root:
            # (4096, z_dim) fp16 or None (missing/corrupt/low-quality file → the collator
            # turns None into a zero-target + aux weight 0; the sample still trains).
            data["repa_target"] = load_repa_target(self.repa_root, sha,
                                                   min_quality=self.repa_min_quality)
        if not self.ss_only:
            shape_item = self._shape(s["shape"])
            if self.max_slat_tokens > 0 and int(shape_item["coords"].shape[0]) > self.max_slat_tokens:
                raise ValueError("oversized")          # → caller skips (stream resample)
            data["target_shape_slat_512_item"] = shape_item
            if s.get("pbr"):
                data["target_tex_slat_512_item"] = self._tex(s["pbr"], s["shape"])
        return data

    def __iter__(self):
        import zipfile
        n_skip = 0
        for s in self.ds:
            # Build (decode) inside the try; yield OUTSIDE it. If we yield inside the try, an
            # exception raised by the CONSUMER (training loop) propagates back through the yield
            # and would be silently swallowed as a "skip" — hiding real downstream errors.
            try:
                item = self._build(s)
            except (ValueError, EOFError, OSError, KeyError, zipfile.BadZipFile) as e:
                # oversized SLAT (ValueError) OR a truncated/corrupt npz in the source data
                # ("No data left in file" EOFError, BadZipFile, ...) → skip, take next. Each rank
                # skips independently; DDP stays in step (1 fwd/bwd per step regardless of which sample).
                n_skip += 1
                if n_skip <= 5 or n_skip % 200 == 0:
                    print(f"[StreamingImageTo3D] skip #{n_skip} ({type(e).__name__}: {str(e)[:80]})", flush=True)
                continue
            yield item
