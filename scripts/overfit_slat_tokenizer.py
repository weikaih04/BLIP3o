"""T2 gate: overfit the 3D SLAT tokenizer on a few assets → does 512 tokens preserve the latent?

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/overfit_slat_tokenizer.py \
      --root /fsx/sfr/weikaih/3dgen/data/trellis2/Toys4k --modality shape --n 8 --steps 800
"""
import argparse, glob, os, time
import numpy as np
import torch
import trellis2_blip3o._paths  # noqa: F401
from trellis2_blip3o.slat_tokenizer import SlatTokenizer, densify, tokenizer_losses

SUBDIR = {"shape": ("shape_latents", "shape_enc_next_dc_f16c32_fp16_512"),
          "tex":   ("pbr_latents",   "tex_enc_next_dc_f16c32_fp16_512")}


def load_batch(root, modality, n, res=32):
    sub, name = SUBDIR[modality]
    files = sorted(glob.glob(os.path.join(root, sub, name, "*.npz")))[:n]
    assert files, f"no .npz under {os.path.join(root, sub, name)}"
    raw = [(np.load(f)["coords"].astype(np.int64), np.load(f)["feats"].astype(np.float32)) for f in files]
    # per-channel norm stats over ALL voxels in the set
    allf = np.concatenate([f for _, f in raw], 0)
    mean = torch.tensor(allf.mean(0)); std = torch.tensor(allf.std(0)).clamp(min=1e-4)
    grids = []
    for coords, feats in raw:
        c = torch.from_numpy(coords); f = (torch.from_numpy(feats) - mean) / std
        grids.append(densify(c, f, res))
    nv = [len(c) for c, _ in raw]
    return torch.stack(grids), mean, std, nv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/fsx/sfr/weikaih/3dgen/data/trellis2/Toys4k")
    ap.add_argument("--modality", choices=["shape", "tex"], default="shape")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--quant", choices=["fsq", "vq"], default="fsq")
    ap.add_argument("--z_ch", type=int, default=8)
    ap.add_argument("--n_e", type=int, default=8192)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    dev = "cuda"

    dense, mean, std, nv = load_batch(args.root, args.modality, args.n)
    dense = dense.to(dev)
    print(f"loaded {args.n} {args.modality} assets | dense {tuple(dense.shape)} | "
          f"active voxels/asset: min={min(nv)} max={max(nv)} mean={sum(nv)//len(nv)}")
    tok = SlatTokenizer(feat_ch=32, quant=args.quant, z_ch=args.z_ch, n_e=args.n_e, base=args.base).to(dev)
    n_param = sum(p.numel() for p in tok.parameters()) / 1e6
    grid_tokens = (32 // 4) ** 3
    print(f"tokenizer {n_param:.1f}M params | quant={args.quant} | bottleneck 8³={grid_tokens} tokens "
          f"| codebook {tok.codebook_size}")
    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        recon, vq_loss, idx = tok(dense)
        L = tokenizer_losses(recon, dense)
        loss = L["feat_l1"] + L["occ_bce"] + vq_loss
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)   # stability: clip
        opt.step(); sched.step()
        if step % 100 == 0 or step == 1:
            usage = idx.unique().numel()
            print(f"step {step:4d} | loss {loss.item():.4f} | feat_L1 {L['feat_l1'].item():.4f} "
                  f"| occ_bce {L['occ_bce'].item():.4f} | occ_IoU {L['occ_iou'].item():.3f} "
                  f"| gnorm {gn.item():.2f} | codebook_used {usage}/{tok.codebook_size}")
    print(f"done in {time.time()-t0:.0f}s")
    # verdict
    print(f"\nVERDICT: feat_L1={L['feat_l1'].item():.4f} (normalized units; <~0.3 = good) | "
          f"occ_IoU={L['occ_iou'].item():.3f} (>~0.9 = structure preserved)")
    print("OVERFIT_DONE")


if __name__ == "__main__":
    main()
