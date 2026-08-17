#!/usr/bin/env python
"""OVERFIT ACCEPTANCE for the from-scratch three-stream MMDiT.

The question: trained on 10 assets until it should have memorized them, can the
model reproduce them? If not, there is still a functional bug — no amount of
static auditing substitutes for this.

Everything except the model loader is imported from the existing eval stack
(scripts/eval_fusion_v22, scripts/eval_tex_v22, scripts/eval_geotex_g1g2), so
cond building, view selection, normalization and rendering are the SAME code the
production evals use — a bespoke pipeline here could flatter the model.

Reported per asset, in normalized latent space (std 1 by construction, so
MSE ~= 1.0 means "no better than predicting the mean" and is the number to beat):
  * tex | GT mesh   — the easiest mode: geometry is given
  * joint           — geometry and texture together, alpha-warped schedule
  * mesh only       — geometry alone (tex lane pinned at its noise corner)

Usage (compute node):
  EVAL_MANI=$PWD/manifests/overfit10.jsonl EVAL_ANY=1 \
  python scripts/eval_overfit10.py --ckpt runs/geotex_s1_overfit10/checkpoint-3000
"""
import argparse
import os
import sys

import numpy as np
import torch

# repo root is THREE levels up now (scripts/eval/x.py, scripts/data/x.py);
# it was two when these lived directly under scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from trellis2_blip3o import _paths  # noqa: F401
from trellis2.modules import sparse as sp

from scripts.eval_fusion_v22 import (build_cond, good_view_b, cam_from_transforms,
                                     input_image)
import scripts.eval_fusion_v22 as EV
from scripts.eval_geotex_g1g2 import pick_assets
from scripts.eval_tex_v22 import render_textured, HDR
from trellis2_blip3o.tr2_modules import (load_norm_stats, TEX_SLAT_CONFIG_PATH,
                                         build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen)
from trellis2_blip3o.mmdit3d import MMDiT3D
from trellis2_blip3o.connector import TRELLIS2Connector
from trellis2_blip3o.geotex_sampler import GeoTexSampler


def build_cond_text(conn, entry_dir, cap_idx=0):
    """TEXT conditioning. Deliberately NOT build_cond(): that one reads v000.npz
    and concatenates a DINO segment, and the text cache has neither — t00N.npz
    holds only (hidden, keep_mask), ~48 Qwen tokens vs ~1054 for an image.
    Training matches: threed.py:413 fuses DINO only for modes I1/IM, so the text
    path feeds the connector output alone. CFG uncond is connector(zeros), the
    same convention build_cond and flow_heads use."""
    a = np.load(os.path.join(entry_dir, f"t{cap_idx:03d}.npz"))
    qwen = torch.from_numpy(a["hidden"]).float().cuda()
    mask = torch.from_numpy(a["keep_mask"]).cuda()
    with torch.no_grad():
        cq = conn(qwen[None])
        c0 = conn(torch.zeros_like(qwen)[None])
        if getattr(conn, "pos_stamp", None) is not None:
            from trellis2_blip3o.pos_stamp import IMG_SPAN_FULL
            cq, c0 = conn.pos_stamp(cq, IMG_SPAN_FULL), conn.pos_stamp(c0, IMG_SPAN_FULL)
    m = mask.bool()
    return [cq[0][m]], [c0[0][m]]


IM_ROOT = "/fsx/home/weikai.huang/3dgen/vlm_hidden_cache/v22_im4r"


def build_cond_mv(conn, dve, sha, combo=0):
    """MULTI-VIEW conditioning. Separate from build_cond for two reasons the
    single-image path does not have to handle:
      * the cache lives in its own root, keyed m00/m01/... (vlm_cache.combo_key),
        not v000 — the IM entry is ONE joint VLM forward over all 4 images;
      * the view embedding is PER TOKEN. build_cond adds dve[0] to every DINO
        token because a single-image entry is all view 0; here the entry carries
        `dino_view_ids` (measured: 1620 DINO tokens spanning 4 views) and each
        token must get ITS OWN view's embedding. Using dve[0] throughout would
        tell the model all four views are the same view."""
    a = np.load(os.path.join(IM_ROOT, sha[:2], sha, f"m{combo:02d}.npz"))
    qwen = torch.from_numpy(a["hidden"]).float().cuda()
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()
    with torch.no_grad():
        cq = conn(qwen[None])
        c0 = conn(torch.zeros_like(qwen)[None])
        if getattr(conn, "pos_stamp", None) is not None:
            from trellis2_blip3o.pos_stamp import IMG_SPAN_FULL
            cq, c0 = conn.pos_stamp(cq, IMG_SPAN_FULL), conn.pos_stamp(c0, IMG_SPAN_FULL)
        dseg = dino[None]
        if dve is not None:
            dseg = dseg + dve[vids.clamp(max=dve.shape[0] - 1)][None].float()
        cond = torch.cat([dseg, cq], 1)
        uncond = torch.cat([torch.zeros_like(dseg), c0], 1)
    m = torch.cat([dmask, qmask]).bool()
    return [cond[0][m]], [uncond[0][m]]


def load_scratch(ckpt, ema=False):
    """From-scratch runs store `unified_geotex.*` (an MMDiT3D) and ONE connector
    (`diffusion_connector.*`) — there is no geo_connector, so
    unified_geotex.load_run_connectors does not apply. Both loads are strict:
    a silently partial load is exactly how an overfit test gets faked."""
    from safetensors.torch import load_file
    sd = load_file(os.path.join(ckpt, "model.safetensors"))
    if ema:
        # OVERLAY, never replace: the EMA shadow (train_native.py:87) tracks
        # PARAMETERS only, so `dino_view_embed` — a fixed buffer built from an
        # env var, not trained — is absent from it. Loading ema.safetensors
        # alone would drop the view embedding and silently evaluate the
        # multi-view path with no view identity at all.
        esd = load_file(os.path.join(ckpt, "ema.safetensors"))
        missing = set(sd) - set(esd)
        assert missing <= {"dino_view_embed"}, \
            f"EMA shadow is missing trained weights, not just buffers: {sorted(missing)[:5]}"
        sd.update(esd)
        print(f"[overfit] EMA weights overlaid ({len(esd)} tensors; "
              f"kept from base: {sorted(missing) or 'none'})")
    pick = lambda p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}

    uni_sd = pick("unified_geotex.")
    assert uni_sd, f"{ckpt} has no unified_geotex.* — not a geotex run"
    # size the model from the checkpoint rather than from defaults, so a run at
    # a different width loads instead of raising a shape error
    dim = uni_sd["geo_flow.adaLN_modulation.1.weight"].shape[1]
    nd = 1 + max(int(k.split(".")[2]) for k in uni_sd if k.startswith("geo_flow.blocks."))
    ns = 1 + max(int(k.split(".")[1]) for k in uni_sd if k.startswith("shared_blocks."))
    # mlp_ratio MUST come from the checkpoint, never from the constructor
    # default: that default moved 4.0 -> 5.3334 on 2026-08-17 to match the
    # release, and every checkpoint written before then is 4.0. Reading it back
    # keeps old runs loadable instead of dying on a shape mismatch.
    mlp_hidden = uni_sd["shared_blocks.0.mlp.mlp.0.weight"].shape[0]
    m = MMDiT3D(dim=dim, num_heads=dim // 128, depth_double=nd, depth_single=ns,
                mlp_ratio=mlp_hidden / dim)
    m.load_state_dict(uni_sd, strict=True)
    m = m.cuda().eval()
    m.convert_to(torch.bfloat16)   # torso only; the sampler feeds fp32 latents

    conn_sd = pick("diffusion_connector.")
    assert conn_sd, f"{ckpt} has no diffusion_connector.*"
    vlm_dim = conn_sd[[k for k in conn_sd if k.endswith("weight") and
                       conn_sd[k].dim() == 2][0]].shape[1]
    conn = TRELLIS2Connector(vlm_hidden_dim=vlm_dim, trellis_cond_dim=1024)
    conn.load_state_dict(conn_sd, strict=True)
    conn = conn.cuda().eval().float()

    dve = sd.get("dino_view_embed")
    dve = dve.cuda().float() if dve is not None else None
    print(f"[overfit] {ckpt}: dim={dim} {nd} triple + {ns} shared · "
          f"vlm_dim={vlm_dim} · dino_view_embed={'yes' if dve is not None else 'no'}")
    return m, conn, dve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=10)
    # None = the released TRELLIS.2-4B per-stream params (shape 12 steps /
    # strength 7.5 / rescale 0.5 / interval [0.6,1.0]; tex 12 / 1.0 / 0.0 /
    # [0.6,0.9]). Passing either overrides BOTH streams and deviates.
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", default="", help="write a comparison grid here")
    ap.add_argument("--official", nargs="*", default=[],
                    help="dirs of <sha>.png produced by run_trellis2_official.py; "
                         "each becomes a rightmost reference column")
    ap.add_argument("--ema", action="store_true",
                    help="evaluate the EMA shadow (ema.safetensors) instead of "
                         "the raw weights; decay 0.9999 = a ~10k-step average")
    ap.add_argument("--text", action="store_true",
                    help="condition on a CAPTION (t00N.npz) instead of an image")
    ap.add_argument("--mv", action="store_true",
                    help="condition on 4 VIEWS (v22_im4r/m00.npz) instead of one")
    ap.add_argument("--cap", type=int, default=0, help="which caption, 0-3")
    a = ap.parse_args()

    grid = dec = None
    if a.render:
        # Same decoders / envmap / camera the production evals use — a bespoke
        # render here could flatter the model.
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        from trellis2.renderers import EnvMap
        dec = (build_sc_vae_shape_decoder_frozen().cuda().eval(),
               build_sc_vae_tex_decoder_frozen().cuda().eval(),
               EnvMap(torch.tensor(cv2.cvtColor(
                   cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)).cuda()))
        CELL, HDRH = 512, 96   # 96, not 56: the official columns wrap to 2 lines
        # Column names say exactly what is SAMPLED and what is supplied. An
        # earlier version labelled the last column "mesh only" while feeding it
        # the GT texture, which read as the model producing material in a mode
        # that has no material output at all.
        first = ("caption (text cond)" if a.text
                 else "4 views (multi-image)" if a.mv else "input image")
        cols = [first, "GT (both)", "tex sampled | GT mesh",
                "joint (a=32): geo + tex", "CASCADE: tex on generated mesh",
                "mesh only: geo, mean material"]
        # Reference columns from the RELEASED microsoft/TRELLIS.2-4B, rendered by
        # scripts/run_trellis2_official.py through this same camera/envmap/
        # renderer. Label them with the ONE advantage we have that no metric
        # shows: our coords come from the GT shape latent (line 233), theirs are
        # generated by their own 64^3 ss_flow stage.
        for od in a.official:
            tag = os.path.basename(od.rstrip("/"))
            cols.append(f"OFFICIAL TRELLIS.2-4B {tag}\n(generates its own voxels)")
        grid = Image.new("RGB", (CELL * len(cols), HDRH + CELL * a.n), (250, 250, 250))
        font = ImageFont.truetype(
            "/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
            "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 32)
        d = ImageDraw.Draw(grid)
        for c, lab in enumerate(cols):
            d.text((c * CELL + 12, 10), lab, fill=(10, 10, 10), font=font)

    m, conn, dve = load_scratch(a.ckpt, ema=a.ema)
    smp = GeoTexSampler(m, steps=a.steps, cfg=a.cfg)
    sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    sm, ssd = sn["mean"].cuda(), sn["std"].cuda()
    tm, tsd = tn["mean"].cuda(), tn["std"].cuda()

    recs = pick_assets(a.n)
    ri = -1
    assert recs, "no assets — set EVAL_MANI to the overfit manifest and EVAL_ANY=1"
    print(f"[overfit] {len(recs)} assets · alpha {a.alpha} · "
          f"shape {smp.shape_p} · tex {smp.tex_p}\n")
    print(f"{'asset':14} {'vox':>6} {'tex|GTmesh':>11} {'joint tex':>10} "
          f"{'joint geo':>10} {'mesh only':>10} {'casc tex':>9} {'casc geo':>9}")
    acc = {k: [] for k in ("tm", "jt", "jg", "mo", "ct", "cg")}
    # latent MSE ALONE is the wrong score for a generative model: the MMSE
    # estimate — i.e. the conditional MEAN — minimises it by construction, so a
    # sampler that collapses to the mean wins on MSE while looking washed out.
    # Measured on checkpoint-48000 before the refine pass: joint had the LOWEST
    # MSE and the LOWEST std (66% of GT). Track std/GT alongside; a healthy
    # sample sits near 1.0.
    sd = {k: [] for k in ("gt_s", "gt_t", "tm", "jt", "jg", "mo", "ct", "cg")}
    for r, ed in recs:
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        gt_s, gt_t = np.load(r["shape_latent_512"]), np.load(r["pbr_latent_512"])
        cx = torch.from_numpy(gt_s["coords"]).int()
        coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
        s_gt = (torch.from_numpy(gt_s["feats"]).float().cuda() - sm) / ssd
        t_gt = (torch.from_numpy(gt_t["feats"]).float().cuda() - tm) / tsd
        c, u = (build_cond_text(conn, ed, a.cap) if a.text
                else build_cond_mv(conn, dve, sha) if a.mv
                else build_cond(conn, dve, ed))
        f = lambda z: z.feats if hasattr(z, "feats") else z
        mse = lambda z, g: float(((f(z) - g) ** 2).mean())

        with torch.no_grad():
            x_tm = smp.sample_tex_given_mesh(coords, s_gt, c, c, u, seed=a.seed)
            js, jt = smp.sample_joint(coords, c, u, c, u, alpha=a.alpha, seed=a.seed)
            x_mo = smp.sample_mesh_only(coords, c, u, seed=a.seed)
            # CASCADE (alpha=inf): geometry runs to completion FIRST, then texture
            # is generated conditioned on THAT geometry — the deployed pipeline,
            # and the only mode here whose texture stands on generated geometry
            # rather than on the GT mesh. Scored against GT on both.
            cs, ct = smp.sample_joint(coords, c, u, c, u,
                                      alpha=float("inf"), seed=a.seed)
        v_tm, v_jt, v_jg, v_mo = (mse(x_tm, t_gt), mse(jt, t_gt),
                                  mse(js, s_gt), mse(x_mo, s_gt))
        v_cg, v_ct = mse(cs, s_gt), mse(ct, t_gt)
        for k, v in (("tm", v_tm), ("jt", v_jt), ("jg", v_jg), ("mo", v_mo),
                     ("ct", v_ct), ("cg", v_cg)):
            acc[k].append(v)
        for k, z in (("gt_s", s_gt), ("gt_t", t_gt), ("tm", x_tm), ("jt", jt),
                     ("jg", js), ("mo", x_mo), ("ct", ct), ("cg", cs)):
            sd[k].append(float(f(z).std()))
        print(f"{sha[:12]:14} {cx.shape[0]:6d} {v_tm:11.4f} {v_jt:10.4f} "
              f"{v_jg:10.4f} {v_mo:10.4f} {v_ct:9.4f} {v_cg:9.4f}", flush=True)

        if grid is not None:
            ri += 1
            shape_dec, tex_dec, envmap = dec
            rdir = r.get("renders_dir") or r.get("renders_cond_dir")
            extr, intr = cam_from_transforms(rdir)
            s_raw = torch.from_numpy(gt_s["feats"]).float().cuda()
            t_raw = torch.from_numpy(gt_t["feats"]).float().cuda()
            R = lambda sf, tf: render_textured(shape_dec, tex_dec, coords, sf, tf,
                                               extr, intr, envmap)
            # mesh-only has NO texture output (its stream emits 32-ch shape
            # velocity and nothing else), so it is rendered against the MEAN
            # material — zeros in normalized space, i.e. tm after denorm. What
            # you see in that column is purely the sampled geometry.
            t_mean = tm.expand(f(x_mo).shape[0], -1).contiguous()
            if a.text:
                # Show the CAPTION, not the render — in text mode the model never
                # saw an image, and putting one in column 1 would misrepresent
                # what it was conditioned on.
                caps = r.get("captions") or []
                # The manifest's `captions` list is EMPTY for these rows and the
                # cache stores only encoded hidden states, so the caption text is
                # not recoverable here — say so rather than showing a blank cell.
                cap = (str(caps[min(a.cap, len(caps) - 1)]) if caps else
                       f"[caption t{a.cap:03d}: text not in manifest;\n"
                       f"cache holds encoded hidden only]")
                col0 = Image.new("RGB", (CELL, CELL), (255, 255, 255))
                dd = ImageDraw.Draw(col0)
                small = ImageFont.truetype(font.path, 20)
                y = 24
                for para in cap.split("\n"):
                    line = ""
                    for w in para.split():
                        if dd.textlength(line + " " + w, font=small) > CELL - 48:
                            dd.text((24, y), line, fill=(20, 20, 20), font=small)
                            y += 26; line = w
                        else:
                            line = (line + " " + w).strip()
                    dd.text((24, y), line, fill=(20, 20, 20), font=small); y += 26
                    if y > CELL - 40: break
                dd.text((24, CELL - 44), f"{a.cap and '' or ''}text-conditioned "
                        f"({np.load(os.path.join(ed, f't{a.cap:03d}.npz'))['hidden'].shape[0]} tokens)",
                        fill=(120, 120, 120), font=small)
            else:
                col0 = input_image(rdir)
            row = [col0,
                   R(s_raw, t_raw),                                   # GT both
                   R(s_raw, f(x_tm) * tsd + tm),                      # sampled tex, GT mesh
                   R(f(js) * ssd + sm, f(jt) * tsd + tm),             # joint: both sampled
                   R(f(cs) * ssd + sm, f(ct) * tsd + tm),             # cascade: tex on gen mesh
                   R(f(x_mo) * ssd + sm, t_mean)]                     # geo only, neutral
            for od in a.official:
                p = os.path.join(od, f"{sha}.png")
                if os.path.exists(p):
                    row.append(Image.open(p).convert("RGB"))
                else:
                    # a missing official render must LOOK missing — pasting the
                    # previous asset's cell or a blank white square would both
                    # read as a result
                    miss = Image.new("RGB", (CELL, CELL), (40, 40, 40))
                    ImageDraw.Draw(miss).text((24, CELL // 2 - 12), "no official render",
                                              fill=(230, 90, 90), font=font)
                    row.append(miss)
            for c_, im in enumerate(row):
                grid.paste(im.resize((CELL, CELL)), (c_ * CELL, HDRH + ri * CELL))

    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"\n{'MSE':14} {'':6} {mean['tm']:11.4f} {mean['jt']:10.4f} "
          f"{mean['jg']:10.4f} {mean['mo']:10.4f} {mean['ct']:9.4f} {mean['cg']:9.4f}")
    S = {k: float(np.mean(v)) for k, v in sd.items()}
    rt, rs = S["gt_t"], S["gt_s"]      # GT std per stream (tex / shape)
    print(f"{'std / GT':14} {'':6} {S['tm']/rt:10.0%} {S['jt']/rt:10.0%} "
          f"{S['jg']/rs:10.0%} {S['mo']/rs:10.0%} {S['ct']/rt:9.0%} {S['cg']/rs:9.0%}"
          "    <- near 100% = a sample; well under = collapsed to the mean")
    if grid is not None:
        grid.save(a.render)
        print(f"\n[overfit] grid -> {a.render}")
    print("\n1.0 = predicting the mean (latents are unit-variance in norm space).")
    print("An overfit that WORKED puts every column far below 1.0; a column stuck "
          "near 1.0 is that mode not learning, not just learning slowly.")


if __name__ == "__main__":
    main()
