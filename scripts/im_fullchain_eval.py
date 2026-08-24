"""FULL IM (4-view) → 3D cascade eval with the newly-trained IM checkpoints.

Chain (zero GT structure, EXACTLY as scripts/fused_fullchain_eval.py):
  cond → SS flow (64^3 occ) → IoU@64 vs GT → maxpool 32^3 coords → shape flow
  (on GENERATED coords) → shape-conditioned tex flow (on GENERATED shape) →
  textured envmap-shaded render + GLB export.

The ONLY change vs fused_fullchain_eval: conditioning comes from the IM 4-view m00
combo (build_cond_im, per-view dino + dve, joint qwen), applied at ALL THREE stages,
each through its OWN ckpt's connector+dve. Two runs per asset, same seed:
  (A) 4v : native m00 (all 4 distinct dino view segments)  [build_cond_im "4distinct"]
  (B) 1v : only dino view-0 block + dve[0], SAME joint qwen  [diag_im_multiview "1view"]

Sharded across GPUs: each rank computes assets[rank::world], writes per-asset artifacts
(render PNGs, GLBs, iou json) to _parts/; then IM_ASSEMBLE=1 (rank0) builds the grid.

Outputs:
  runs/cache_logs/im_fullchain_grid.png  [input 4 views | 1-view render | 4-view render | GT]
  runs/cache_logs/im_fullchain/<sha8>_{4v,1v}.glb
  runs/cache_logs/im_fullchain_iou.json  per-asset 4v/1v SS-IoU@64
"""
import os, sys, json, glob
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("FUSED_MODULATE", "1")
os.environ.setdefault("EVAL_COND_ROOT", "/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_heldout")
# eval_im_vs_i1_ss reads SS_CKPT_IM at import-time (module global); we only borrow its
# load_view_imgs helper, so a dummy satisfies the import without affecting anything.
os.environ.setdefault("SS_CKPT_IM", os.environ.get("IM_SS_CKPT",
    "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/s3_ss_im_mds/checkpoint-10000"))
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont

from trellis2_blip3o import _paths  # noqa
from trellis2 import models as t2models  # type: ignore
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # type: ignore
from trellis2.renderers import EnvMap  # type: ignore
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen,
                                         load_norm_stats, SHAPE_SLAT_CONFIG_PATH,
                                         TEX_SLAT_CONFIG_PATH, SS_FLOW_CONFIG_PATH)
import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import (load_flow_and_connector, sample_shape,
                                     good_view_b, input_image, cam_from_transforms)
from scripts.eval_tex_v22 import load_tex_flow, sample_tex, render_textured, HDR
from scripts.export_glb_fullchain import load_ss_flow, SSDEC
from scripts.export_glb_v22 import build_mw, export_glb
from scripts.eval_im_vs_i1_ss import load_view_imgs

ROOT = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
SS_CKPT = os.environ.get("IM_SS_CKPT", f"{ROOT}/runs/s3_ss_im_mds/checkpoint-10000")
SHAPE_CKPT = os.environ.get("IM_SHAPE_CKPT", f"{ROOT}/runs/s3_shape_im_mds/checkpoint-10000")
TEX_CKPT = os.environ.get("IM_TEX_CKPT", f"{ROOT}/runs/s3_tex_im_mds/checkpoint-10000")
COND_IM = os.environ.get("COND_IM", "/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_im4l")
MANI = os.environ.get("MANI", "/fsx/home/weikai.huang/3dgen/im_probe/heldout14.jsonl")

OUT_DIR = f"{ROOT}/runs/cache_logs/im_fullchain"
PARTS = f"{OUT_DIR}/_parts"
GRID_OUT = f"{ROOT}/runs/cache_logs/im_fullchain_grid.png"
IOU_JSON = f"{ROOT}/runs/cache_logs/im_fullchain_iou.json"
FONT = ("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/site-packages/"
        "matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
# Official SS sampler (= fused_fullchain_eval / diag_im_multiview / eval_im_vs_i1_ss)
SS_OFF = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
              guidance_interval=[0.6, 1.0], rescale_t=5.0)
SEED = 0
N_LIMIT = int(os.environ.get("IM_N", "14"))


def build_cond_im(conn, dve, m00_path, mode):
    """Build a fusion cond from the IM 4-view m00 through THIS stage's connector+dve.
    mode='4v' → native 4 distinct dino view blocks (build_cond_im "4distinct").
    mode='1v' → only dino view-0 block + dve[0], SAME joint qwen (diag "1view").
    Returns (cond, uncond, views)."""
    a = np.load(m00_path)
    qwen = torch.from_numpy(a["hidden"]).float().cuda()          # (Tq, 2048) joint 4-view
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()     # (Td, 1024)
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()    # (Td,) per-token view ordinal
    if mode == "1v":
        sel0 = (vids == 0)
        dino, dmask = dino[sel0], dmask[sel0]
        vids = torch.zeros(dino.shape[0], dtype=torch.long, device=dino.device)  # dve[0]
    elif mode != "4v":
        raise ValueError(mode)
    # One builder for the whole repo: trellis2_blip3o.eval_cond wraps
    # build_unified_cond, i.e. the function the training loop itself calls, so a
    # conditioning mismatch can no longer live in the gap between the two.
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    cond, uncond = cond_uncond_from_tensors(conn, qwen, qmask, dino, dmask, dve,
                                            dino_view_ids=vids)
    return cond, uncond, [int(v) for v in a["views"]]


@torch.no_grad()
def sample_ss_official(flow, sampler, cond, uncond, seed=SEED):
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution,
                        flow.resolution,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                        device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return sampler.sample(flow, noise, cond=cond, neg_cond=uncond,
                              verbose=False, **SS_OFF).samples


def compute():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD", "1"))
    os.makedirs(PARTS, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = [json.loads(l) for l in open(MANI)][:N_LIMIT]
    mine = recs[rank::world]
    print(f"[rank{rank}/{world}] {len(mine)} assets: {[r['sha256'][:8] for r in mine]}", flush=True)

    # --- models (all IM ckpts; each stage has its OWN connector+dve) ---
    ssflow, ssconn, ssdve = load_ss_flow(SS_CKPT)
    ss_sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    ssdec = t2models.from_pretrained(SSDEC).cuda().eval()
    flow, conn, dve = load_flow_and_connector(SHAPE_CKPT)
    tflow, tconn, tdve = load_tex_flow(TEX_CKPT)
    sdec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tdec = build_sc_vae_tex_decoder_frozen().cuda().eval()
    hdr_img = cv2.cvtColor(cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(hdr_img).cuda())

    ss_norm = load_norm_stats(SS_FLOW_CONFIG_PATH, "normalization")
    sn = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    tsn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    sm_, ssd_ = sn["mean"].cuda(), sn["std"].cuda()
    tm_, tsd_ = tn["mean"].cuda(), tn["std"].cuda()
    xm_, xsd_ = tsn["mean"].cuda(), tsn["std"].cuda()
    ssm = ss_norm["mean"].cuda().view(1, -1, 1, 1, 1) if ss_norm else 0
    sss = ss_norm["std"].cuda().view(1, -1, 1, 1, 1) if ss_norm else 1

    cell = 512

    def one_stage(mode, m00, coords_out):
        """Run the full SS→shape→tex→render cascade for one cond mode. Returns
        (render_PIL, iou, mw_or_None)."""
        # SS (FUSION, official sampler) on IM cond
        c_ss, u_ss, views = build_cond_im(ssconn, ssdve, m00, mode)
        z = sample_ss_official(ssflow, ss_sampler, c_ss, u_ss)
        z = z * sss + ssm
        occ = (ssdec(z) > 0)[0, 0]
        iou = coords_out["iou_fn"](occ)
        occ32 = F.max_pool3d(occ.float()[None, None], 2, 2) > 0.5
        cc = torch.argwhere(occ32[0, 0]).int()
        coords = torch.cat([torch.zeros(cc.shape[0], 1, dtype=torch.int32,
                                        device="cuda"), cc], 1).cpu()
        # shape on GENERATED coords
        c_sh, u_sh, _ = build_cond_im(conn, dve, m00, mode)
        slat = sample_shape(flow, c_sh, u_sh, coords)
        gen_shape_raw = slat.feats.float() * ssd_ + sm_
        # tex on GENERATED shape
        c_tx, u_tx, _ = build_cond_im(tconn, tdve, m00, mode)
        tex_n = sample_tex(tflow, c_tx, u_tx, coords, (gen_shape_raw - xm_) / xsd_)
        gen_tex_raw = tex_n.cuda() * tsd_ + tm_
        extr, intr = coords_out["extr"], coords_out["intr"]
        img = render_textured(sdec, tdec, coords, gen_shape_raw, gen_tex_raw, extr, intr, envmap)
        mw = build_mw(sdec, tdec, coords, gen_shape_raw, gen_tex_raw)
        return img, iou, int(occ.sum()), len(cc), views, mw

    for r in mine:
        sha = r["sha256"]; s8 = sha[:8]
        m00 = os.path.join(COND_IM, sha[:2], sha, "m00.npz")
        if not os.path.exists(m00):
            print(f"[skip] {s8}: no m00", flush=True); continue
        EV._VIEW_FILE = good_view_b(sha)
        extr, intr = cam_from_transforms(r["renders_dir"])
        # GT occ@64 for IoU
        try:
            gz = torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None]
            gt_occ = (ssdec(gz) > 0)[0, 0]
        except Exception:
            gt_occ = None

        def iou_fn(occ):
            if gt_occ is None:
                return -1.0
            return (occ & gt_occ).sum().item() / max(1, (occ | gt_occ).sum().item())

        ctx = dict(extr=extr, intr=intr, iou_fn=iou_fn)
        rec = {"sha8": s8, "sha": sha, "iou_4v": None, "iou_1v": None,
               "views": None, "fail": {}}
        for mode in ["4v", "1v"]:
            try:
                img, iou, vox, ncoord, views, mw = one_stage(mode, m00, ctx)
                img.resize((cell, cell), Image.LANCZOS).save(f"{PARTS}/{s8}_{mode}.png")
                rec[f"iou_{mode}"] = iou
                rec["views"] = views
                print(f"[rank{rank}] {s8} {mode}: SS-IoU={iou:.3f} vox64={vox} "
                      f"coords32={ncoord}", flush=True)
                try:
                    export_glb(mw, f"{OUT_DIR}/{s8}_{mode}.glb")
                except Exception as e:
                    rec["fail"][f"{mode}_glb"] = f"{type(e).__name__}: {e}"
                    print(f"[rank{rank}] {s8} {mode} GLB FAIL: {e}", flush=True)
            except Exception as e:
                rec["fail"][mode] = f"{type(e).__name__}: {e}"
                print(f"[rank{rank}] {s8} {mode} FAIL: {e}", flush=True)
        # input 4-view tile
        try:
            imgs = load_view_imgs(r["renders_dir"], rec["views"] or [0, 1, 2, 3])
            tile = Image.new("RGB", (cell, cell), (255, 255, 255))
            for k, im in enumerate(imgs[:4]):
                tile.paste(im.convert("RGB").resize((cell // 2, cell // 2), Image.LANCZOS),
                           ((k % 2) * (cell // 2), (k // 2) * (cell // 2)))
            tile.save(f"{PARTS}/{s8}_input.png")
        except Exception as e:
            rec["fail"]["input"] = f"{type(e).__name__}: {e}"
        # GT textured render
        try:
            gt_s = np.load(r["shape_latent_512"]); gt_t = np.load(r["pbr_latent_512"])
            cx = torch.from_numpy(gt_s["coords"]).int()
            gcoords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
            gimg = render_textured(sdec, tdec, gcoords,
                                   torch.from_numpy(gt_s["feats"]).float().cuda(),
                                   torch.from_numpy(gt_t["feats"]).float().cuda(),
                                   extr, intr, envmap)
            gimg.resize((cell, cell), Image.LANCZOS).save(f"{PARTS}/{s8}_gt.png")
        except Exception as e:
            rec["fail"]["gt"] = f"{type(e).__name__}: {e}"
            print(f"[rank{rank}] {s8} GT render FAIL: {e}", flush=True)
        json.dump(rec, open(f"{PARTS}/{s8}.json", "w"))
        print(f"[rank{rank}] {s8} DONE  4v={rec['iou_4v']} 1v={rec['iou_1v']}", flush=True)
    print(f"[rank{rank}] SHARD_DONE", flush=True)


def assemble():
    recs = [json.loads(l) for l in open(MANI)][:N_LIMIT]
    order = [r["sha256"][:8] for r in recs]
    parts = {}
    for jf in glob.glob(f"{PARTS}/*.json"):
        d = json.load(open(jf))
        parts[d["sha8"]] = d
    order = [s for s in order if s in parts]
    cell, hdr = 512, 64
    font = ImageFont.truetype(FONT, 34)
    rfont = ImageFont.truetype(FONT, 22)
    cols = ["input (4 views)", "1-view 3D (shaded)", "4-view 3D (shaded)", "GT render"]
    grid = Image.new("RGB", (cell * 4, hdr + cell * len(order)), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    for c, label in enumerate(cols):
        d.text((c * cell + 12, 14), label, fill=(10, 10, 10), font=font)

    def paste(col, y, path):
        if os.path.exists(path):
            grid.paste(Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS),
                       (col * cell, y))

    summ = {"ss_ckpt": SS_CKPT, "shape_ckpt": SHAPE_CKPT, "tex_ckpt": TEX_CKPT,
            "per_asset": []}
    for ri, s8 in enumerate(order):
        p = parts[s8]
        y = hdr + ri * cell
        paste(0, y, f"{PARTS}/{s8}_input.png")
        paste(1, y, f"{PARTS}/{s8}_1v.png")
        paste(2, y, f"{PARTS}/{s8}_4v.png")
        paste(3, y, f"{PARTS}/{s8}_gt.png")
        i4 = p.get("iou_4v"); i1 = p.get("iou_1v")
        for col, val, tag in [(1, i1, "1v"), (2, i4, "4v")]:
            lab = f"{tag} IoU {val:.3f}" if val is not None else f"{tag} FAIL"
            d.rectangle([col * cell + 4, y + 4, col * cell + 4 + 12 + int(rfont.getlength(lab)),
                         y + 34], fill=(255, 255, 255))
            d.text((col * cell + 10, y + 8), lab, fill=(10, 90, 10), font=rfont)
        d.rectangle([4, y + 4, 4 + 12 + int(rfont.getlength(s8)), y + 34], fill=(255, 255, 255))
        d.text((10, y + 8), s8, fill=(10, 10, 10), font=rfont)
        summ["per_asset"].append({"sha8": s8, "iou_4v": i4, "iou_1v": i1,
                                  "views": p.get("views"), "fail": p.get("fail", {})})
    grid.save(GRID_OUT)

    v4 = np.array([a["iou_4v"] for a in summ["per_asset"] if a["iou_4v"] is not None and a["iou_4v"] >= 0])
    v1 = np.array([a["iou_1v"] for a in summ["per_asset"] if a["iou_1v"] is not None and a["iou_1v"] >= 0])
    both = [(a["iou_4v"], a["iou_1v"]) for a in summ["per_asset"]
            if a["iou_4v"] is not None and a["iou_1v"] is not None and a["iou_4v"] >= 0 and a["iou_1v"] >= 0]
    summ["mean_iou_4v"] = float(v4.mean()) if len(v4) else None
    summ["mean_iou_1v"] = float(v1.mean()) if len(v1) else None
    summ["n"] = len(both)
    if both:
        b = np.array(both)
        summ["mean_delta_4v_minus_1v"] = float((b[:, 0] - b[:, 1]).mean())
        summ["n_4v_gt_1v"] = int((b[:, 0] > b[:, 1]).sum())
    json.dump(summ, open(IOU_JSON, "w"), indent=2)
    print(f"[grid] saved {GRID_OUT}", flush=True)
    print(f"[json] saved {IOU_JSON}", flush=True)
    print("\n=== IM FULLCHAIN SS-IoU@64 SUMMARY ===", flush=True)
    for a in summ["per_asset"]:
        print(f"  {a['sha8']}  4v={a['iou_4v']}  1v={a['iou_1v']}  "
              f"views={a['views']}  fail={a['fail']}", flush=True)
    print(f"  mean 4v={summ['mean_iou_4v']}  mean 1v={summ['mean_iou_1v']}  n={summ['n']}", flush=True)
    if both:
        print(f"  delta(4v-1v)={summ['mean_delta_4v_minus_1v']:+.4f}  "
              f"4v>1v on {summ['n_4v_gt_1v']}/{summ['n']}", flush=True)
    print("IM_FULLCHAIN_DONE", flush=True)


if __name__ == "__main__":
    if os.environ.get("IM_ASSEMBLE") == "1":
        assemble()
    else:
        compute()
