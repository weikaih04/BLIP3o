"""T2-decode for TEXTURE: overfit a tex SLAT tokenizer, then decode GT vs reconstructed
texture (on the SAME GT shape) through the frozen TRELLIS decoders + PBR render, side-by-side.
Isolates texture-reconstruction quality (shape held fixed).

Builds the decoders DIRECTLY (no full pipeline → avoids the gated DINOv3 image conditioner).

  CUDA_VISIBLE_DEVICES=0 python scripts/eval_decode_tex_tokenizer.py --n 4 --steps 1200 --eval 3
"""
import argparse, glob, os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # cv2's EXR codec is off by default
import numpy as np
import torch
import trellis2_blip3o._paths  # noqa: F401
from trellis2_blip3o.slat_tokenizer import SlatTokenizer, densify, tokenizer_losses
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen)
from trellis2.representations import MeshWithVoxel
from trellis2.modules.sparse import SparseTensor
from trellis2.utils import render_utils
from PIL import Image

DATA = "/fsx/sfr/weikaih/3dgen/data/trellis2/Toys4k"
SH = f"{DATA}/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
TX = f"{DATA}/pbr_latents/tex_enc_next_dc_f16c32_fp16_512"
HDR = "/fsx/sfr/weikaih/3dgen/model/third_party_3d_gen/TRELLIS.2/assets/hdri/forest.exr"
PBR_LAYOUT = {'base_color': slice(0, 3), 'metallic': slice(3, 4),
              'roughness': slice(4, 5), 'alpha': slice(5, 6)}


def sparse(coords3, feats):
    coords3 = coords3.to(feats.device)
    b = torch.zeros(len(coords3), 1, dtype=torch.int32, device=feats.device)
    return SparseTensor(feats=feats, coords=torch.cat([b, coords3.to(torch.int32)], 1))


def load_envmap():
    import cv2
    from trellis2.renderers import EnvMap
    hdr = cv2.cvtColor(cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(hdr).cuda())


def decode_textured(shape_dec, tex_dec, shape_slat, tex_slat, res=512):
    """Replicate pipeline.decode_latent without the pipeline (decoders only)."""
    shape_dec.set_resolution(res)
    with torch.no_grad():
        meshes, subs = shape_dec(shape_slat, return_subs=True)
        tex_voxels = tex_dec(tex_slat, guide_subs=subs) * 0.5 + 0.5
    m, v = meshes[0], (tex_voxels[0] if not torch.is_tensor(tex_voxels) else tex_voxels)
    m.fill_holes()
    return MeshWithVoxel(m.vertices, m.faces, origin=[-0.5, -0.5, -0.5], voxel_size=1 / res,
                         coords=v.coords[:, 1:], attrs=v.feats,
                         voxel_shape=torch.Size([*v.shape, *v.spatial_shape]), layout=PBR_LAYOUT)


def render_shaded(mesh, envmap, nviews=2, res=384):
    out = render_utils.render_snapshot(mesh, resolution=res, nviews=nviews, envmap=envmap)
    key = "shaded" if "shaded" in out else ("color" if "color" in out else list(out)[0])
    return out[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--eval", type=int, default=3)
    ap.add_argument("--out", default="/tmp/slat_tex_decode.png")
    args = ap.parse_args()
    dev = "cuda"

    shp = {os.path.basename(f): f for f in glob.glob(f"{SH}/*.npz")}
    txp = {os.path.basename(f): f for f in glob.glob(f"{TX}/*.npz")}
    paired = [s for s in sorted(shp) if s in txp][:args.n]
    assert paired, "no paired shape+tex assets found"
    tex_raw = [(np.load(txp[s])["coords"].astype(np.int64), np.load(txp[s])["feats"].astype(np.float32)) for s in paired]

    allf = np.concatenate([f for _, f in tex_raw], 0)
    mean = torch.tensor(allf.mean(0)); std = torch.tensor(allf.std(0)).clamp(min=1e-4)
    dense = torch.stack([densify(torch.from_numpy(c), (torch.from_numpy(f) - mean) / std) for c, f in tex_raw]).to(dev)
    tok = SlatTokenizer(feat_ch=32, quant="fsq").to(dev)
    opt = torch.optim.AdamW(tok.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    print(f"overfitting TEX tokenizer on {args.n} assets, {args.steps} steps...")
    for step in range(1, args.steps + 1):
        recon, qloss, _ = tok(dense)
        L = tokenizer_losses(recon, dense)
        (L["feat_l1"] + L["occ_bce"] + qloss).backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0); opt.step(); opt.zero_grad(); sched.step()
        if step % 300 == 0:
            print(f"  step {step} | feat_L1 {L['feat_l1'].item():.4f} | occ_IoU {L['occ_iou'].item():.3f}")
    tok.eval()

    print("building frozen shape+tex decoders + envmap...")
    shape_dec = build_sc_vae_shape_decoder_frozen().to(dev).eval()
    tex_dec = build_sc_vae_tex_decoder_frozen().to(dev).eval()
    envmap = load_envmap()
    mean_d, std_d = mean.to(dev), std.to(dev)

    rows = []
    for i in range(min(args.eval, args.n)):
        s = paired[i]
        sh = np.load(shp[s])
        shape_slat = sparse(torch.from_numpy(sh["coords"].astype(np.int32)),
                            torch.from_numpy(sh["feats"].astype(np.float32)).cuda())
        tex_gt = sparse(torch.from_numpy(tex_raw[i][0].astype(np.int32)),
                        torch.from_numpy(tex_raw[i][1]).cuda())
        with torch.no_grad():
            rec, _, _ = tok(dense[i:i+1])
        feat_pred, occ_logit = rec[0, :32], rec[0, 32]
        active = (occ_logit > 0).nonzero(as_tuple=False)
        feats_rec = (feat_pred[:, active[:, 0], active[:, 1], active[:, 2]].t() * std_d + mean_d)
        tex_rec = sparse(active.to(torch.int32), feats_rec)
        print(f"asset {i} ({s[:12]}): tex GT {len(tex_raw[i][0])} vox | recon {len(active)} vox")
        m_gt = decode_textured(shape_dec, tex_dec, shape_slat, tex_gt)
        m_rc = decode_textured(shape_dec, tex_dec, shape_slat, tex_rec)
        g, r = render_shaded(m_gt, envmap), render_shaded(m_rc, envmap)
        rows.append(np.concatenate([g[0], r[0], g[1], r[1]], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"SAVED {args.out}  (cols: GT-tex | RECON-tex | GT-tex | RECON-tex ; rows = assets)")
    print("TEX_DECODE_EVAL_DONE")


if __name__ == "__main__":
    main()
