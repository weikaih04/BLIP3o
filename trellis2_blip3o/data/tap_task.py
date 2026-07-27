"""Node-local VP1 tap-cache dataset for the shape-512 conditioning ablation.

Each node holds ITS shard of /opt/dlami/nvme/vp1taps_v22 (see vlm3d_stage1/tap_logs/
SHARD_MAP.json). Every rank on a node iterates the node-LOCAL file list sharded by
LOCAL_RANK — across 4 nodes × 8 ranks the 45k assets tile exactly once per epoch
(DDP grads average across all 32 ranks; same pattern as the node-sharded MDS loader).

Yields per sample:
  taps        (n_taps, L, 2048) fp32   — VP1 v2.2 hiddens (arm ①/② use taps[-1], ③ all)
  cond_mask   (L,) bool                — True everywhere here (no padding at source);
                                         collator pads batch-wise and extends the mask
  target coords/feats                  — shape SLAT 512 (normalized, same as ThreeDTask)
Skips oversized SLAT (> max_slat_tokens) and assets missing a shape latent.
"""
import os, json, random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .. import _paths  # noqa: F401
from ..tr2_modules import load_norm_stats, SHAPE_SLAT_CONFIG_PATH

TAP_DIR = "/opt/dlami/nvme/vp1taps_v22"
MANIFESTS = [
    "/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v3/ready_v3_clean.jsonl",
]


def _sha_to_shape_index(cache_path: str) -> dict:
    """sha → shape_latent_512 path, cached to a local json. LOCAL_RANK 0 builds it once
    per node; other ranks wait (unique tmp name — a shared tmp raced across 8 ranks)."""
    import time
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        for _ in range(240):                     # wait up to 20 min for rank 0
            if os.path.exists(cache_path):
                return json.load(open(cache_path))
            time.sleep(5)
        raise RuntimeError(f"shape index never appeared: {cache_path}")
    idx = {}
    for mani in MANIFESTS:
        with open(mani) as f:
            for line in f:
                r = json.loads(line)
                p = r.get("shape_latent_512")
                if p:
                    idx[r["sha256"]] = p
    tmp = f"{cache_path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(idx, f)
    os.replace(tmp, cache_path)
    return idx


class TapShapeDataset(IterableDataset):
    """Iterates (taps, shape-SLAT target) pairs from the node-local tap shard."""

    def __init__(self, tap_dir: str = TAP_DIR, max_slat_tokens: int = 4096,
                 val: bool = False, val_frac: float = 0.02, seed: int = 0,
                 shuffle: bool = True):
        super().__init__()
        self.tap_dir = tap_dir
        self.max_slat_tokens = max_slat_tokens
        self.shuffle = shuffle
        self.shape_norm = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")

        files = []
        for d2 in sorted(os.listdir(tap_dir)):
            p2 = os.path.join(tap_dir, d2)
            if len(d2) == 2 and os.path.isdir(p2):
                files += [os.path.join(p2, f) for f in sorted(os.listdir(p2))
                          if f.endswith(".npz")]
        rng = random.Random(seed)
        rng.shuffle(files)                       # deterministic split/order per node
        n_val = max(1, int(len(files) * val_frac))
        self.files = files[:n_val] if val else files[n_val:]
        self.idx = _sha_to_shape_index(os.path.join(tap_dir, "_shape_index.json"))
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __iter__(self):
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        files = self.files[rank::world]
        wi = torch.utils.data.get_worker_info()
        if wi is not None:                       # DataLoader workers: split further
            files = files[wi.id::wi.num_workers]
        if self.shuffle:
            random.Random(1000 + self.epoch).shuffle(files)
        for fp in files:
            sha = os.path.basename(fp)[:-4]
            sp = self.idx.get(sha)
            if sp is None:
                continue
            try:
                a = np.load(fp)
                taps = torch.from_numpy(a["taps"]).view(torch.bfloat16).float()
                s = np.load(sp)
                coords = torch.from_numpy(s["coords"]).int()
                feats = torch.from_numpy(s["feats"]).float()
            except Exception:
                continue
            if coords.shape[0] > self.max_slat_tokens:
                continue
            if self.shape_norm is not None:
                feats = (feats - self.shape_norm["mean"]) / self.shape_norm["std"]
            yield {"taps": taps, "coords": coords, "feats": feats, "sha": sha}


def collate_taps(batch, device=None):
    """Pad taps to the batch max length (+mask); build the sparse SLAT target."""
    from trellis2.modules import sparse as sp  # type: ignore
    K = batch[0]["taps"].shape[0]
    H = batch[0]["taps"].shape[-1]
    Ls = [b["taps"].shape[1] for b in batch]
    Lm = max(Ls)
    B = len(batch)
    taps = torch.zeros(B, K, Lm, H)
    mask = torch.zeros(B, Lm, dtype=torch.bool)
    coords, feats = [], []
    for i, b in enumerate(batch):
        taps[i, :, :Ls[i]] = b["taps"]
        mask[i, :Ls[i]] = True
        c = b["coords"]
        coords.append(torch.cat([torch.full((c.shape[0], 1), i, dtype=torch.int32), c], 1))
        feats.append(b["feats"])
    x0 = sp.SparseTensor(torch.cat(feats), torch.cat(coords).int())
    return {"taps": taps, "cond_mask": mask, "x0": x0,
            "shas": [b["sha"] for b in batch]}
