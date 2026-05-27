"""Offline pre-compute SigLIP-2 + VQ codebook IDs for multi-view renders.

For Setup B/C only. Uses BLIP3o-NEXT's `TextAlignedTokenizer` (SigLIP-2 encoder
+ VQ bottleneck). Output is a per-object `.npz` file containing the codebook
indices for each of the N rendered views.

Usage (from project root):

    python -m trellis2_blip3o.codebook_prep \\
        --renders-root data/objxl_sketchfab/renders_cond \\
        --out-root     data/objxl_sketchfab/siglip_codebook_ids \\
        --views 0 1 2 3 \\
        --ta-tok-ckpt  /path/to/ta_tok.ckpt

Each input view image is encoded to (N_tokens,) codebook IDs. We stack across
views → (N_views, N_tokens) and save as `.npz` with key `codebook`.
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from . import _paths  # noqa: F401


def load_ta_tok(ckpt_path: str, device: str = "cuda"):
    """Load the BLIP3o TextAlignedTokenizer from checkpoint.

    Workaround: BLIP3o's `from_checkpoint` calls torch.load with default `weights_only`,
    which fails on PyTorch >= 2.6 because the ckpt contains easydict objects. We patch
    by registering safe globals and using a manual load path.
    """
    import easydict
    try:
        torch.serialization.add_safe_globals([easydict.EasyDict])
    except Exception:
        pass

    from tok.ta_tok import TextAlignedTokenizer  # type: ignore
    # Manual load instead of from_checkpoint (which has a bug — uses undefined ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_kwargs = ckpt["model"]["args"]
    tok = TextAlignedTokenizer(**ckpt_kwargs)
    sd = ckpt["model"]["sd"]
    sd = {k: v for k, v in sd.items() if not k.startswith('teacher')}
    tok.load_state_dict(sd, strict=True)
    tok = tok.to(device).eval()
    tok.bottleneck.regularizer.set_eval_deterministic(deterministic=True)
    return tok


@torch.no_grad()
def encode_image(tok, image: Image.Image) -> torch.Tensor:
    """Encode a single PIL image → codebook indices (N_tokens,)."""
    # tok.encode expects (B, C, H, W) in [0, 1]
    transform = transforms.Compose([
        transforms.Resize((tok.input_size, tok.input_size)),
        transforms.ToTensor(),  # → (C, H, W) in [0, 1]
    ])
    x = transform(image.convert("RGB")).unsqueeze(0).to(tok.device)
    out = tok.encode(x)
    # Bottleneck returns 'bottleneck_rep' = q_indices  (B, N_tokens)
    indices = out["bottleneck_rep"][0].cpu().numpy().astype(np.int32)
    return indices


def process_object(
    tok,
    renders_dir: str,
    view_indices: List[int],
    out_path: str,
):
    """For one object, encode N views and save .npz."""
    codebook_per_view = []
    for v in view_indices:
        # Standard TRELLIS render_cond outputs '000.png' .. '007.png'
        img_path = os.path.join(renders_dir, f"{v:03d}.png")
        if not os.path.exists(img_path):
            print(f"[skip] missing view {v}: {img_path}")
            return False
        img = Image.open(img_path).convert("RGB")
        ids = encode_image(tok, img)
        codebook_per_view.append(ids)
    codebook = np.stack(codebook_per_view, axis=0)  # (N_views, N_tokens)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, codebook=codebook)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders-root", required=True, help="Directory containing <sha256>/<view>.png")
    ap.add_argument("--out-root", required=True, help="Where to write <sha256>.npz")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--ta-tok-ckpt", required=True, help="TextAlignedTokenizer checkpoint path")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N objects (for testing)")
    args = ap.parse_args()

    tok = load_ta_tok(args.ta_tok_ckpt)
    print(f"loaded TextAlignedTokenizer (input_size={tok.input_size}, bottleneck_token_num={tok.bottleneck_token_num})")

    shas = sorted(
        d for d in os.listdir(args.renders_root)
        if os.path.isdir(os.path.join(args.renders_root, d))
    )
    if args.limit:
        shas = shas[: args.limit]

    succ, fail = 0, 0
    for sha in tqdm(shas, desc="codebook_prep"):
        out = os.path.join(args.out_root, f"{sha}.npz")
        if os.path.exists(out):
            succ += 1
            continue
        ok = process_object(
            tok,
            renders_dir=os.path.join(args.renders_root, sha),
            view_indices=args.views,
            out_path=out,
        )
        if ok:
            succ += 1
        else:
            fail += 1

    print(f"done: {succ} succeeded, {fail} failed → {args.out_root}")


if __name__ == "__main__":
    main()
