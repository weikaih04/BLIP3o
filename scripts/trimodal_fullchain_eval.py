"""TRI-MODAL full-cascade comparison for the S3 unified model (I1 / IM / T from ONE model).

For each held-out asset, run the SAME three checkpoints (runs/s3_{ss,shape,tex}_qvmlp/
checkpoint-3000 — all from the one tri-modal run) three times, changing ONLY the
conditioning modality:

  I1 : single-image fusion cond   [DINO(v000) + dve[0] ; connector(qwen v000)]
  IM : 4-view fusion cond         [DINO(m00) + dve[view_id] ; connector(qwen m00) + dve[qwen_view_ids]]
  T  : text caption, QWEN-ONLY    [connector(qwen t000)]          <- no DINO segment at all

Chain per modality (zero GT structure, same as fused_fullchain_eval / im_fullchain_eval):
  cond -> SS flow (64^3 occ, OFFICIAL sampler) -> IoU@64 vs GT -> maxpool 32^3 coords
  -> shape flow (on GENERATED coords) -> shape-conditioned tex flow (on GENERATED shape)
  -> envmap-shaded textured render from the input good-view camera + GLB export.

Same seed for all three modalities, so differences are conditioning and not noise.

TRAINING PARITY NOTES (get these wrong and the eval is meaningless):
  * connector arch comes from the ckpt config (benchmarks.checkpoint.load_connector reads
    cond_adapter; these are mlp) — all three loaders already route through it.
  * dino_view_embed is the FIXED sincos table baked into the ckpt (VIEW_EMBED_SCALE=0.2 at
    train time => row L2 4.53), so nothing needs to be re-derived here.
  * IM applies the view code to BOTH segments (flow_heads.py does): DINO via dino_view_ids,
    QWEN via the cache's fixed layout (292 tokens, four 64-token image blocks at
    10/76/142/208, stride 66; everything else gets no code). Copied from eval_im_vs_i1_ss.
  * I1 gets dve[0] on DINO only and NO qwen view code (threed.py only sets qwen_view_ids
    for mode IM).
  * T never reaches the fusion branch in flow_heads (dino_hidden is None) => plain
    connector(qwen), CFG uncond = connector(0), no view embedding anywhere.

Sharded across GPUs: rank r does assets[r::world], writes per-asset artifacts to _parts/;
then TRI_ASSEMBLE=1 builds the grid + summary.json.

Outputs (runs/cache_logs/trimodal/):
  grid.png        rows = assets, cols = [input 4 views | I1 | IM | T | GT]
  {sha8}_{i1,im,t}.glb
  summary.json    per-asset per-modality SS-IoU@64 + means
"""
import os, sys, json, glob, textwrap, traceback

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("FUSED_MODULATE", "1")
# flex_gemm autosaves its triton autotune cache to ~/.flex_gemm/autotune_cache.json via a
# tmp+rename — with N ranks on one node the renames race and one rank dies mid-decode
# ("No such file or directory: .../autotune_cache.json.tmp"). Give every rank its own file.
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", os.path.expanduser(
    f"~/.flex_gemm/autotune_cache_r{os.environ.get('RANK', '0')}.json"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EVAL_COND_ROOT",
                      "/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_heldout")
# eval_im_vs_i1_ss reads SS_CKPT_IM at import time (module global); we only borrow its
# load_view_imgs helper, so a dummy satisfies the import.
os.environ.setdefault("SS_CKPT_IM", os.environ.get(
    "TRI_SS_CKPT", f"{ROOT}/runs/s3_ss_qvmlp/checkpoint-3000"))

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
                                     good_view_b, cam_from_transforms)
from scripts.eval_tex_v22 import load_tex_flow, sample_tex, render_textured, HDR
from scripts.export_glb_fullchain import load_ss_flow, SSDEC
from scripts.export_glb_v22 import build_mw, export_glb
from scripts.eval_im_vs_i1_ss import load_view_imgs

SS_CKPT = os.environ.get("TRI_SS_CKPT", f"{ROOT}/runs/s3_ss_qvmlp/checkpoint-3000")
SHAPE_CKPT = os.environ.get("TRI_SHAPE_CKPT", f"{ROOT}/runs/s3_shape_qvmlp/checkpoint-3000")
TEX_CKPT = os.environ.get("TRI_TEX_CKPT", f"{ROOT}/runs/s3_tex_qvmlp/checkpoint-3000")
COND_I1 = os.environ["EVAL_COND_ROOT"]                      # v000 (+ t000 built beside it)
COND_IM = os.environ.get("COND_IM", "/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_im4l")
MANI = os.environ.get("MANI", "/fsx/home/weikai.huang/3dgen/im_probe/heldout14_newpaths.jsonl")
CAP_MANIS = os.environ.get("CAP_MANIS", ":".join(
    os.path.join("/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered", f)
    for f in ("heldout_eval_capT.jsonl", "heldout_eval2_capT.jsonl",
              "heldout_sketchfab_capT.jsonl"))).split(":")
CAP_IDX = int(os.environ.get("CAP_IDX", "0"))               # t000 = caption_long

OUT_DIR = os.environ.get("TRI_OUT", f"{ROOT}/runs/cache_logs/trimodal")
PARTS = f"{OUT_DIR}/_parts"
GRID_OUT = f"{OUT_DIR}/grid.png"
SUMM_OUT = f"{OUT_DIR}/summary.json"
FONT = ("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
# Official SS sampler (= fused_fullchain_eval / im_fullchain_eval / eval_im_vs_i1_ss)
SS_OFF = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
              guidance_interval=[0.6, 1.0], rescale_t=5.0)
SEED = 0
MODES = ("i1", "im", "t")
N_LIMIT = int(os.environ.get("TRI_N", "14"))
# QWEN image-block layout of the IM (m00) cache — VERIFIED constant across the whole cache
# (292 tokens; four 64-token blocks at 10/76/142/208, stride 66 = 64 + 2 delimiters).
QV_START, QV_STRIDE, QV_NTOK, QV_NVIEW = 10, 66, 64, 4


# ─────────────────────────── conditioning builders ───────────────────────────
@torch.no_grad()
def build_cond_i1(conn, dve, entry_dir):
    """Single-image fusion cond (eval_fusion_v22.build_cond): DINO + dve[0] ; connector(qwen).
    No qwen view code — threed.py sets qwen_view_ids for mode IM only."""
    a = np.load(os.path.join(entry_dir, "v000.npz"))
    qwen = torch.from_numpy(a["hidden"]).float().cuda()
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    if "dino_hidden" in a.files:
        dino = torch.from_numpy(a["dino_hidden"]).float().cuda()
        dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    else:                                     # legacy split d-entry
        b = np.load(os.path.join(entry_dir, "d000.npz"))
        dino = torch.from_numpy(b["hidden"]).float().cuda()
        dmask = torch.from_numpy(b["keep_mask"]).cuda()
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    cond, uncond = cond_uncond_from_tensors(conn, qwen, qmask, dino, dmask, dve)
    return cond, uncond, {"n_tok": int(cond.shape[1])}


@torch.no_grad()
def build_cond_im(conn, dve, m00_path):
    """4-view fusion cond. View identity on BOTH segments — DINO via dino_view_ids and
    QWEN via the cache's fixed image-block layout (mirror of flow_heads.py). Verbatim
    logic from scripts/eval_im_vs_i1_ss.build_cond_im."""
    a = np.load(m00_path)
    qwen = torch.from_numpy(a["hidden"]).float().cuda()          # (292, 2048) joint 4-image
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()     # (4*405, 1024)
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()
    # The CACHE does not store qwen_view_ids (the live encoder does), so the
    # fixed image-block layout is still derived here — but only to hand the
    # shared builder the same array a live record would carry.
    if not (dve is not None and qwen.shape[0] >= QV_START + QV_STRIDE * (QV_NVIEW - 1) + QV_NTOK):
        raise RuntimeError(f"unexpected qwen layout T_q={qwen.shape[0]} in {m00_path}")
    qv = torch.full((qwen.shape[0],), -1, dtype=torch.long, device=qwen.device)
    for v in range(QV_NVIEW):
        qv[QV_START + QV_STRIDE * v: QV_START + QV_STRIDE * v + QV_NTOK] = v
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    cond, uncond = cond_uncond_from_tensors(conn, qwen, qmask, dino, dmask, dve,
                                            dino_view_ids=vids, qwen_view_ids=qv)
    return cond, uncond, {"views": [int(v) for v in a["views"]],
                          "n_tok": int(cond.shape[1])}


@torch.no_grad()
def build_cond_t(conn, dve, entry_dir, cap_idx=CAP_IDX):
    """Text cond — QWEN ONLY, no DINO segment (that is how the T task was trained:
    dino_hidden is None => flow_heads takes the non-fusion branch, no view embedding)."""
    a = np.load(os.path.join(entry_dir, f"t{cap_idx:03d}.npz"))
    qwen = torch.from_numpy(a["hidden"]).float().cuda()
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    cond, uncond = cond_uncond_from_tensors(conn, qwen, qmask)   # dino=None -> plain branch
    return cond, uncond, {"n_tok": int(cond.shape[1])}


BUILDERS = {"i1": build_cond_i1, "im": build_cond_im, "t": build_cond_t}


def cond_arg(mode, sha):
    if mode == "im":
        return os.path.join(COND_IM, sha[:2], sha, "m00.npz")
    return os.path.join(COND_I1, sha[:2], sha)


@torch.no_grad()
def sample_ss_official(flow, sampler, cond, uncond, seed=SEED):
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution,
                        flow.resolution,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                        device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return sampler.sample(flow, noise, cond=cond, neg_cond=uncond,
                              verbose=False, **SS_OFF).samples


def load_captions():
    caps = {}
    for p in CAP_MANIS:
        if not os.path.isfile(p):
            continue
        for line in open(p):
            r = json.loads(line)
            caps[r["sha256"]] = [c for c in (r.get("captions") or []) if c]
    return caps


# ─────────────────────────────── compute ───────────────────────────────
def compute():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD", "1"))
    os.makedirs(PARTS, exist_ok=True)
    recs = [json.loads(l) for l in open(MANI)][:N_LIMIT]
    mine = recs[rank::world]
    caps = load_captions()
    print(f"[rank{rank}/{world}] {len(mine)} assets: {[r['sha256'][:8] for r in mine]}",
          flush=True)

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

    def one_modality(mode, sha, ctx):
        """Full SS -> shape -> tex -> render cascade for ONE conditioning modality.
        Each stage builds its cond through ITS OWN ckpt's connector + dve."""
        build = BUILDERS[mode]
        src = cond_arg(mode, sha)
        c_ss, u_ss, info = build(ssconn, ssdve, src)
        z = sample_ss_official(ssflow, ss_sampler, c_ss, u_ss)
        z = z * sss + ssm
        occ = (ssdec(z) > 0)[0, 0]
        iou = ctx["iou_fn"](occ)
        occ32 = F.max_pool3d(occ.float()[None, None], 2, 2) > 0.5
        cc = torch.argwhere(occ32[0, 0]).int()
        if cc.shape[0] == 0:
            raise RuntimeError("empty SS occupancy -> no coords for the shape stage")
        coords = torch.cat([torch.zeros(cc.shape[0], 1, dtype=torch.int32,
                                        device="cuda"), cc], 1).cpu()
        c_sh, u_sh, _ = build(conn, dve, src)
        slat = sample_shape(flow, c_sh, u_sh, coords, seed=SEED)
        gen_shape_raw = slat.feats.float() * ssd_ + sm_
        c_tx, u_tx, _ = build(tconn, tdve, src)
        tex_n = sample_tex(tflow, c_tx, u_tx, coords, (gen_shape_raw - xm_) / xsd_, seed=SEED)
        gen_tex_raw = tex_n.cuda() * tsd_ + tm_
        img = render_textured(sdec, tdec, coords, gen_shape_raw, gen_tex_raw,
                              ctx["extr"], ctx["intr"], envmap)
        mw = build_mw(sdec, tdec, coords, gen_shape_raw, gen_tex_raw)
        info.update(iou=iou, vox64=int(occ.sum()), coords32=int(cc.shape[0]))
        return img, info, mw

    for r in mine:
        sha = r["sha256"]; s8 = sha[:8]
        EV._VIEW_FILE = good_view_b(sha)                   # cond view B == render camera
        extr, intr = cam_from_transforms(r["renders_dir"])
        try:
            gz = torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None]
            with torch.no_grad():
                gt_occ = (ssdec(gz) > 0)[0, 0]
        except Exception as e:
            print(f"[rank{rank}] {s8} GT occ FAIL: {e}", flush=True)
            gt_occ = None

        def iou_fn(occ, _g=None):
            g = gt_occ
            if g is None:
                return None
            return (occ & g).sum().item() / max(1, (occ | g).sum().item())

        ctx = dict(extr=extr, intr=intr, iou_fn=iou_fn)
        cl = caps.get(sha, [])
        rec = {"sha8": s8, "sha": sha, "caption": cl[CAP_IDX] if CAP_IDX < len(cl) else "",
               "view_i1": int(EV._VIEW_FILE[:3]), "modes": {}, "fail": {}}
        for mode in MODES:
            try:
                img, info, mw = one_modality(mode, sha, ctx)
                img.resize((cell, cell), Image.LANCZOS).save(f"{PARTS}/{s8}_{mode}.png")
                rec["modes"][mode] = info
                print(f"[rank{rank}] {s8} {mode}: IoU={info['iou']} vox64={info['vox64']} "
                      f"coords32={info['coords32']} cond_tok={info['n_tok']}", flush=True)
                try:
                    export_glb(mw, f"{OUT_DIR}/{s8}_{mode}.glb")
                except Exception as e:
                    rec["fail"][f"{mode}_glb"] = f"{type(e).__name__}: {e}"
                    print(f"[rank{rank}] {s8} {mode} GLB FAIL: {e}", flush=True)
            except Exception as e:
                rec["fail"][mode] = f"{type(e).__name__}: {e}"
                print(f"[rank{rank}] {s8} {mode} FAIL: {e}\n{traceback.format_exc()}",
                      flush=True)
        # input view tile (the IM 4 views; the I1 / render view is among them)
        try:
            views = (rec["modes"].get("im") or {}).get("views")
            if not views:
                views = [int(v) for v in np.load(cond_arg("im", sha))["views"]]
            rec["views_im"] = views
            imgs = load_view_imgs(r["renders_dir"], views)
            tile = Image.new("RGB", (cell, cell), (255, 255, 255))
            dt = ImageDraw.Draw(tile)
            vf = ImageFont.truetype(FONT, 20)
            for k, im in enumerate(imgs[:4]):
                x, y = (k % 2) * (cell // 2), (k // 2) * (cell // 2)
                tile.paste(im.convert("RGB").resize((cell // 2, cell // 2), Image.LANCZOS),
                           (x, y))
                is_i1 = views[k] == rec["view_i1"]
                if is_i1:
                    dt.rectangle([x + 1, y + 1, x + cell // 2 - 2, y + cell // 2 - 2],
                                 outline=(10, 150, 10), width=4)
                dt.text((x + 8, y + 6), f"v{views[k]:03d}" + (" = I1" if is_i1 else ""),
                        fill=(10, 120, 10) if is_i1 else (60, 60, 60), font=vf)
            tile.save(f"{PARTS}/{s8}_input.png")
        except Exception as e:
            rec["fail"]["input"] = f"{type(e).__name__}: {e}"
        # GT textured render (same camera)
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
        print(f"[rank{rank}] {s8} DONE "
              f"{ {m: (rec['modes'].get(m) or {}).get('iou') for m in MODES} }", flush=True)
    print(f"[rank{rank}] SHARD_DONE", flush=True)


# ─────────────────────────────── assemble ───────────────────────────────
def assemble():
    recs = [json.loads(l) for l in open(MANI)][:N_LIMIT]
    order = [r["sha256"][:8] for r in recs]
    parts = {}
    for jf in glob.glob(f"{PARTS}/*.json"):
        d = json.load(open(jf))
        parts[d["sha8"]] = d
    order = [s for s in order if s in parts]
    cell, hdr, capbar = 512, 70, 78
    row_h = cell + capbar
    font = ImageFont.truetype(FONT, 26)
    rfont = ImageFont.truetype(FONT, 22)
    cfont = ImageFont.truetype(FONT, 17)
    cols = ["input views (green = I1/cam)", "I1  single image",
            "IM  4 views", "T  text caption", "GT"]
    W, H = cell * 5, hdr + row_h * len(order)
    grid = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    for c, label in enumerate(cols):
        d.text((c * cell + 10, 20), label, fill=(10, 10, 10), font=font)

    def paste(col, y, path):
        if os.path.exists(path):
            grid.paste(Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS),
                       (col * cell, y))

    def tag(col, y, text, color):
        w = int(rfont.getlength(text))
        d.rectangle([col * cell + 4, y + 4, col * cell + 18 + w, y + 34], fill=(255, 255, 255))
        d.text((col * cell + 10, y + 8), text, fill=color, font=rfont)

    summ = {"ss_ckpt": SS_CKPT, "shape_ckpt": SHAPE_CKPT, "tex_ckpt": TEX_CKPT,
            "cond_i1": COND_I1, "cond_im": COND_IM, "cap_idx": CAP_IDX,
            "ss_sampler": SS_OFF, "seed": SEED, "per_asset": []}
    for ri, s8 in enumerate(order):
        p = parts[s8]
        y = hdr + ri * row_h
        paste(0, y, f"{PARTS}/{s8}_input.png")
        for ci, mode in enumerate(MODES):
            paste(1 + ci, y, f"{PARTS}/{s8}_{mode}.png")
        paste(4, y, f"{PARTS}/{s8}_gt.png")
        tag(0, y, s8, (10, 10, 10))
        for ci, mode in enumerate(MODES):
            info = p["modes"].get(mode) or {}
            iou = info.get("iou")
            txt = f"{mode.upper()} IoU {iou:.3f}" if isinstance(iou, float) else f"{mode.upper()} FAIL"
            tag(1 + ci, y, txt, (10, 90, 10) if isinstance(iou, float) else (150, 10, 10))
        tag(4, y, "GT", (10, 10, 10))
        # caption strip under the T column (wrapped, truncated to 3 lines)
        cap = (p.get("caption") or "").strip()
        cy = y + cell + 4
        d.rectangle([0, y + cell, W, y + row_h - 1], fill=(238, 238, 240))
        if cap:
            lines = textwrap.wrap(cap, width=62)[:3]
            if len(textwrap.wrap(cap, width=62)) > 3:
                lines[-1] = lines[-1][:58] + " ..."
            for li, ln in enumerate(lines):
                d.text((3 * cell + 8, cy + li * 22), ln, fill=(30, 30, 30), font=cfont)
        d.text((10, cy + 2), f"{s8}   IM views {p.get('views_im')}   I1/cam view "
                             f"v{p.get('view_i1'):03d}", fill=(60, 60, 60), font=cfont)
        d.line([(0, y + row_h - 1), (W, y + row_h - 1)], fill=(200, 200, 205), width=1)
        summ["per_asset"].append({
            "sha8": s8, "sha": p["sha"], "caption": cap,
            "views_im": p.get("views_im"), "view_i1": p.get("view_i1"),
            "iou": {m: (p["modes"].get(m) or {}).get("iou") for m in MODES},
            "vox64": {m: (p["modes"].get(m) or {}).get("vox64") for m in MODES},
            "cond_tokens": {m: (p["modes"].get(m) or {}).get("n_tok") for m in MODES},
            "fail": p.get("fail", {})})
    os.makedirs(OUT_DIR, exist_ok=True)
    grid.save(GRID_OUT)

    for m in MODES:
        v = [a["iou"][m] for a in summ["per_asset"] if isinstance(a["iou"][m], float)]
        summ[f"mean_iou_{m}"] = float(np.mean(v)) if v else None
        summ[f"n_{m}"] = len(v)
    both = [(a["iou"]["i1"], a["iou"]["im"]) for a in summ["per_asset"]
            if isinstance(a["iou"]["i1"], float) and isinstance(a["iou"]["im"], float)]
    if both:
        b = np.array(both)
        summ["mean_delta_im_minus_i1"] = float((b[:, 1] - b[:, 0]).mean())
        summ["n_im_gt_i1"] = int((b[:, 1] > b[:, 0]).sum())
    json.dump(summ, open(SUMM_OUT, "w"), indent=2)
    print(f"[grid] {GRID_OUT}\n[json] {SUMM_OUT}", flush=True)
    print("\n=== TRI-MODAL SS-IoU@64 ===", flush=True)
    for a in summ["per_asset"]:
        io = a["iou"]
        f = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
        print(f"  {a['sha8']}  I1={f(io['i1'])}  IM={f(io['im'])}  T={f(io['t'])}"
              f"  fail={a['fail'] or ''}", flush=True)
    print(f"  MEAN  I1={summ['mean_iou_i1']}  IM={summ['mean_iou_im']}  T={summ['mean_iou_t']}",
          flush=True)
    if both:
        print(f"  delta(IM-I1)={summ['mean_delta_im_minus_i1']:+.4f}  "
              f"IM>I1 on {summ['n_im_gt_i1']}/{len(both)}", flush=True)
    print("TRIMODAL_DONE", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.environ.get("TRI_ASSEMBLE") == "1":
        assemble()
    else:
        compute()
