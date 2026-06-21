"""T2-decode: overfit the SLAT tokenizer, then DECODE both GT and reconstructed latents
through the frozen TRELLIS shape decoder and render side-by-side. Visual proof that a good
latent reconstruction (feat_L1) → a good mesh.

  CUDA_VISIBLE_DEVICES=0 python scripts/eval_decode_slat_tokenizer.py --n 4 --steps 1200 --eval 3
"""
import argparse, glob, os
import numpy as np
import torch
import trellis2_blip3o._paths  # noqa: F401
from trellis2_blip3o.slat_tokenizer import SlatTokenizer, densify, tokenizer_losses
from trellis2_blip3o.tr2_modules import build_sc_vae_shape_decoder_frozen
from trellis2.modules.sparse import SparseTensor
from trellis2.utils import render_utils
from PIL import Image

ROOT = "/fsx/sfr/weikaih/3dgen/data/trellis2/Toys4k/shape_latents/shape_enc_next_dc_f16c32_fp16_512"


def to_sparse(coords3, feats):  # coords3:(N,3) int, feats:(N,32) -> SparseTensor (batch 0)
    b = torch.zeros(len(coords3), 1, dtype=torch.int32, device=feats.device)
    coords4 = torch.cat([b, coords3.to(torch.int32)], 1)
    return SparseTensor(feats=feats, coords=coords4)


def decode_mesh(dec, coords3, feats_raw):
    dec.set_resolution(512)
    with torch.no_grad():
        ret = dec(to_sparse(coords3.cuda(), feats_raw.cuda()), return_subs=True)
    return ret[0][0]


def render_normal(mesh, nviews=2, res=384):
    out = render_utils.render_snapshot(mesh, resolution=res, nviews=nviews)
    key = "normal" if "normal" in out else ("color" if "color" in out else list(out)[0])
    return out[key]  # list of HxWx3 uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--eval", type=int, default=3)
    ap.add_argument("--out", default="/tmp/slat_tok_decode.png")
    args = ap.parse_args()
    dev = "cuda"

    files = sorted(glob.glob(os.path.join(ROOT, "*.npz")))[:args.n]
    raw = [(np.load(f)["coords"].astype(np.int64), np.load(f)["feats"].astype(np.float32)) for f in files]
    allf = np.concatenate([f for _, f in raw], 0)
    mean = torch.tensor(allf.mean(0)); std = torch.tensor(allf.std(0)).clamp(min=1e-4)
    dense = torch.stack([densify(torch.from_numpy(c), (torch.from_numpy(f) - mean) / std) for c, f in raw]).to(dev)

    tok = SlatTokenizer(feat_ch=32, quant="fsq").to(dev)
    opt = torch.optim.AdamW(tok.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    print(f"overfitting tokenizer on {args.n} assets for {args.steps} steps...")
    for step in range(1, args.steps + 1):
        recon, qloss, _ = tok(dense)
        L = tokenizer_losses(recon, dense)
        loss = L["feat_l1"] + L["occ_bce"] + qloss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0); opt.step(); sched.step()
        if step % 300 == 0:
            print(f"  step {step} | feat_L1 {L['feat_l1'].item():.4f} | occ_IoU {L['occ_iou'].item():.3f}")

    mean_d, std_d = mean.to(dev), std.to(dev)
    dec = build_sc_vae_shape_decoder_frozen().to(dev).eval()
    rows = []
    tok.eval()
    for i in range(min(args.eval, args.n)):
        coords_gt = torch.from_numpy(raw[i][0])                          # (N,3)
        feats_gt = torch.from_numpy(raw[i][1])                           # (N,32) raw
        with torch.no_grad():
            rec, _, _ = tok(dense[i:i+1])
        feat_pred, occ_logit = rec[0, :32], rec[0, 32]                   # (32,R,R,R),(R,R,R)
        active = (occ_logit > 0).nonzero(as_tuple=False)                 # (M,3)
        feats_rec_n = feat_pred[:, active[:, 0], active[:, 1], active[:, 2]].t()  # (M,32) normalized
        feats_rec = feats_rec_n * std_d + mean_d                        # denormalize -> raw
        print(f"asset {i}: GT {len(coords_gt)} vox | recon {len(active)} vox")
        m_gt = decode_mesh(dec, coords_gt, feats_gt)
        m_rc = decode_mesh(dec, active.cpu(), feats_rec.cpu())
        g = render_normal(m_gt); r = render_normal(m_rc)
        row = np.concatenate([g[0], r[0], g[1], r[1]], axis=1)          # GTv1 | RECv1 | GTv2 | RECv2
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(args.out)
    print(f"SAVED {args.out}  (columns: GT | RECON | GT | RECON ; rows = assets)")
    print("DECODE_EVAL_DONE")


if __name__ == "__main__":
    main()
