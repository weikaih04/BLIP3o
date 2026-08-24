"""Synthetic identical-front/different-back benchmark — EVAL stage (1 GPU).

QUESTION: does a multi-view-conditioned SS flow actually READ non-frontal views?
16 synthetic assets = 8 pairs with IDENTICAL fronts and COARSELY different backs
(scripts/synth_backpair_build.py). A single front view is physically insufficient;
the back view disambiguates. Average IoU cannot show this (priors fill unseen surface);
the pair discrimination can.

MODELS
  IM  runs/s3_ss_im_mds/checkpoint-10000      (IM-trained, native 4-view m00 cond)
  I1  runs/fusion_ss_dpos_2n/checkpoint-22000 (deployed single-view baseline; its 4-view
      mode = TRELLIS-1 multidiffusion aggregation over per-view conds, t1_multiimage_infer)

CONDS (built by scripts/synth_backpair_conds.py)
  1v  front view only  : conds_pv v000 through the EXACT deployed single-view path
  4v  front+back+left+right : IM -> joint m00 (build_cond_im); I1 -> multidiffusion

METRICS per asset x model x mode (mean over seeds 0,1,2), GT = 64^3 shell occupancy:
  iou_own / iou_decoy (decoy = pair partner's GT), disc = own - decoy
  back-half (grid x<32, where the variants differ) b_own / b_dec / b_disc  <- THE signal
  aligned disc: best of 4 z-rotations chosen ONLY on the front half (shared within the
      pair -> unbiased); guards against azimuth misplacement (training render yaws are
      random per asset, so canonical azimuth is not deterministically inferable)
  pairdiv: IoU(occ | cond_X , occ | cond_Y) same model/mode/seed — alignment-free
      cond-sensitivity. 1v: conds identical -> 1.0 (determinism check). 4v << 1 means
      the non-frontal views actually change the output.

EXPECTED if fusion is ABSENT: disc(4v) ~= disc(1v) ~= 0 and pairdiv(4v) ~= 1.
If a model READS the back view: b_disc(4v) >> 0 while disc(1v) ~= 0 (structurally exact).

Out: table + runs/cache_logs/synth_backpair_eval.json + occ_samples.npz + grid png.
"""
import os, sys, json, time
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from trellis2_blip3o import _paths  # noqa
from trellis2 import models as t2models  # type: ignore
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # type: ignore
from scripts.export_glb_fullchain import load_ss_flow, SSDEC
# TRELLIS-1-style aggregation + per-view cond builder (proven t1 machinery)
from scripts.t1_multiimage_infer import MultiDiffusionSampler, build_cond_key

ROOT = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
SYNTH = "/fsx/home/weikai.huang/3dgen/im_probe/synth_backpair"
MANI = os.environ.get("MANI", os.path.join(SYNTH, "synth16.jsonl"))
COND_PV = os.path.join(SYNTH, "conds_pv")
COND_IM4 = os.path.join(SYNTH, "conds_im4")
HELDOUT = "/fsx/home/weikai.huang/3dgen/im_probe/heldout14.jsonl"
OUT_JSON = f"{ROOT}/runs/cache_logs/synth_backpair_eval.json"
MODELS = [("IM", os.environ.get("SS_CKPT_IM", f"{ROOT}/runs/s3_ss_im_mds/checkpoint-10000")),
          ("I1", os.environ.get("SS_CKPT_I1", f"{ROOT}/runs/fusion_ss_dpos_2n/checkpoint-22000"))]
SEEDS = [0, 1, 2]
# Official SS sampler settings (= eval_im_vs_i1_ss / diag_im_multiview / t1_multiimage_infer)
SS_OFF = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
              guidance_interval=[0.6, 1.0], rescale_t=5.0)
FONT = ("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")


def build_cond_im(conn, dve, entry_dir):
    """4-view joint fusion cond from an IM m00 entry — EXACT copy of
    scripts/eval_im_vs_i1_ss.build_cond_im (that module needs SS_CKPT_IM env at import)."""
    a = np.load(os.path.join(entry_dir, "m00.npz"))
    qwen = torch.from_numpy(a["hidden"]).float().cuda()
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()
    # trellis2_blip3o.eval_cond wraps build_unified_cond — the function the
    # training loop calls — so eval and training cannot disagree about the dino
    # segment, the view embedding, or which drop is CFG.
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    return cond_uncond_from_tensors(conn, qwen, qmask, dino, dmask, dve,
                                    dino_view_ids=vids)


@torch.no_grad()
def sample_occ(flow, sampler, ssdec, cond, uncond, seed):
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution,
                        generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = sampler.sample(flow, noise, cond=cond, neg_cond=uncond, verbose=False, **SS_OFF).samples
    return (ssdec(z) > 0)[0, 0].cpu().numpy()


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else -1.0


def proj(occ):
    views = [occ.max(ax) for ax in (2, 1, 0)]
    canvas = np.zeros((64, 64 * 3), np.uint8)
    for i, v in enumerate(views):
        canvas[:, i * 64:(i + 1) * 64] = (v * 255).astype(np.uint8)
    return Image.fromarray(canvas).resize((64 * 3 * 2, 64 * 2), Image.NEAREST).convert("RGB")


def gt_convention_check(ssdec):
    """Decode 3 real heldout GT ss latents: shell (hollow) or solid? -> pick synth GT key."""
    hollow = []
    for r in [json.loads(l) for l in open(HELDOUT)][:3]:
        gz = torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None]
        with torch.no_grad():
            occ = (ssdec(gz) > 0)[0, 0].cpu().numpy()
        filled = ndimage.binary_fill_holes(occ)
        h = float((filled & ~occ).sum()) / max(float(filled.sum()), 1.0)
        hollow.append(h)
        print(f"[gtcheck] {r['sha256'][:8]} occ={int(occ.sum())} filled={int(filled.sum())} "
              f"hollow_frac={h:.3f}", flush=True)
    mh = float(np.mean(hollow))
    key = "occ_shell" if mh > 0.10 else "occ_solid"
    print(f"[gtcheck] mean hollow_frac={mh:.3f} -> real GT is "
          f"{'SHELL' if key == 'occ_shell' else 'SOLID'} -> synth GT key = {key}", flush=True)
    return key


def main():
    recs = [json.loads(l) for l in open(MANI)]
    by_sha = {r["sha256"]: r for r in recs}
    print(f"[eval] {len(recs)} synth assets  seeds={SEEDS}  SS_OFF={SS_OFF}", flush=True)

    ssdec = t2models.from_pretrained(SSDEC).cuda().eval()
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    md_sampler = MultiDiffusionSampler(sigma_min=1e-5)
    gt_key = gt_convention_check(ssdec)

    flows = {}
    for mname, ck in MODELS:
        flow, conn, dve = load_ss_flow(ck)
        flows[mname] = (flow, conn, dve)
        print(f"[eval] {mname}={ck}  dve={None if dve is None else tuple(dve.shape)}  "
              f"pos_stamp={'yes' if getattr(conn, 'pos_stamp', None) is not None else 'no'}",
              flush=True)

    gts = {sha: np.load(r["gt"])[gt_key].astype(bool) for sha, r in by_sha.items()}
    gts_solid = {sha: np.load(r["gt"])["occ_solid"].astype(bool) for sha, r in by_sha.items()}
    print(f"[eval] synth GT[{gt_key}] voxels: "
          f"{[int(gts[r['sha256']].sum()) for r in recs[:4]]}...", flush=True)

    # ---------------- sampling ----------------
    occs = {}
    t0 = time.time()
    for r in recs:
        sha = r["sha256"]
        pv_dir = os.path.join(COND_PV, sha[:2], sha)
        im4_dir = os.path.join(COND_IM4, sha[:2], sha)
        for mname, _ in MODELS:
            flow, conn, dve = flows[mname]
            c1, u1 = build_cond_key(conn, dve, pv_dir, "v000")           # front only
            for seed in SEEDS:
                occs[(sha, mname, "1v", seed)] = sample_occ(flow, sampler, ssdec, c1, u1, seed)
            if mname == "IM":
                c4, u4 = build_cond_im(conn, dve, im4_dir)
                for seed in SEEDS:
                    occs[(sha, mname, "4v", seed)] = sample_occ(flow, sampler, ssdec, c4, u4, seed)
            else:  # I1: TRELLIS-1 multidiffusion over the 4 per-view conds
                pv = [build_cond_key(conn, dve, pv_dir, f"v{v:03d}") for v in range(4)]
                md_sampler.set_conds([c for c, _ in pv], [u for _, u in pv])
                md_sampler._log_once = False
                for seed in SEEDS:
                    occs[(sha, mname, "4v", seed)] = sample_occ(flow, md_sampler, ssdec,
                                                                pv[0][0], pv[0][1], seed)
        print(f"[sample] {r['name']} done ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(os.path.join(SYNTH, "occ_samples.npz"),
                        **{f"{by_sha[s]['name']}|{m}|{md}|s{sd}": o
                           for (s, m, md, sd), o in occs.items()})

    # ---------------- metrics ----------------
    B = 32  # back half = grid x < 32 (world x<0); front = x >= 32
    rows = []
    for r in recs:
        sha, psha = r["sha256"], r["partner_sha256"]
        own, dec = gts[sha], gts[psha]
        for mname, _ in MODELS:
            for mode in ("1v", "4v"):
                per_seed = []
                for seed in SEEDS:
                    o = occs[(sha, mname, mode, seed)]
                    po = occs[(psha, mname, mode, seed)]
                    # azimuth alignment on the SHARED front half only (unbiased)
                    ks = [iou(np.rot90(o, k, axes=(0, 1))[B:], own[B:]) for k in range(4)]
                    kstar = int(np.argmax(ks))
                    oa = np.rot90(o, kstar, axes=(0, 1))
                    per_seed.append(dict(
                        iou_own=iou(o, own), iou_dec=iou(o, dec),
                        iou_own_solid=iou(o, gts_solid[sha]),
                        f_own=iou(o[B:], own[B:]),
                        b_own=iou(o[:B], own[:B]), b_dec=iou(o[:B], dec[:B]),
                        kstar=kstar, a_front=ks[kstar],
                        a_own=iou(oa, own), a_dec=iou(oa, dec),
                        ab_own=iou(oa[:B], own[:B]), ab_dec=iou(oa[:B], dec[:B]),
                        pairdiv=iou(o, po)))
                m = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]
                     if k != "kstar"}
                m["kstars"] = [s["kstar"] for s in per_seed]
                m.update(disc=m["iou_own"] - m["iou_dec"], b_disc=m["b_own"] - m["b_dec"],
                         a_disc=m["a_own"] - m["a_dec"], ab_disc=m["ab_own"] - m["ab_dec"])
                rows.append(dict(name=r["name"], pair=r["pair"], variant=r["variant"],
                                 back=r["back_desc"], model=mname, mode=mode, **m))

    # ---------------- per-asset table ----------------
    hdr = (f"{'asset':<12}{'model':<4}{'mode':<4}{'own':>7}{'decoy':>7}{'disc':>8}"
           f"{'b_own':>7}{'b_dec':>7}{'b_disc':>8}{'ab_disc':>8}{'k*':>7}{'pairdiv':>8}")
    print("\n=== PER-ASSET (mean over seeds; b_* = back half; ab_disc = azimuth-aligned "
          "back disc; k* = chosen z-rot) ===")
    print(hdr)
    for row in rows:
        print(f"{row['name']:<12}{row['model']:<4}{row['mode']:<4}"
              f"{row['iou_own']:>7.3f}{row['iou_dec']:>7.3f}{row['disc']:>+8.3f}"
              f"{row['b_own']:>7.3f}{row['b_dec']:>7.3f}{row['b_disc']:>+8.3f}"
              f"{row['ab_disc']:>+8.3f}{str(row['kstars']):>7}{row['pairdiv']:>8.3f}",
              flush=True)

    # ---------------- summary ----------------
    summ = {"models": dict(MODELS), "seeds": SEEDS, "sampler": SS_OFF, "gt_key": gt_key,
            "n_assets": len(recs), "per_asset": rows, "groups": {}}
    print("\n=== SUMMARY (mean over 16 assets x 3 seeds) ===")
    print(f"{'model':<5}{'mode':<5}{'iou_own':>8}{'disc':>8}{'b_disc':>8}{'ab_disc':>8}"
          f"{'pairdiv':>8}{'n_disc>0':>9}{'n_bdisc>0':>10}")
    for mname, _ in MODELS:
        for mode in ("1v", "4v"):
            g = [row for row in rows if row["model"] == mname and row["mode"] == mode]
            gr = {k: float(np.mean([row[k] for row in g]))
                  for k in ("iou_own", "iou_dec", "disc", "b_own", "b_dec", "b_disc",
                            "a_disc", "ab_disc", "pairdiv", "f_own", "iou_own_solid")}
            gr["n_disc_pos"] = int(sum(row["disc"] > 0 for row in g))
            gr["n_bdisc_pos"] = int(sum(row["b_disc"] > 0 for row in g))
            summ["groups"][f"{mname}_{mode}"] = gr
            print(f"{mname:<5}{mode:<5}{gr['iou_own']:>8.3f}{gr['disc']:>+8.3f}"
                  f"{gr['b_disc']:>+8.3f}{gr['ab_disc']:>+8.3f}{gr['pairdiv']:>8.3f}"
                  f"{gr['n_disc_pos']:>7}/16{gr['n_bdisc_pos']:>8}/16", flush=True)

    g = summ["groups"]
    for mname, _ in MODELS:
        d1, d4 = g[f"{mname}_1v"], g[f"{mname}_4v"]
        gain = d4["b_disc"] - d1["b_disc"]
        reads = (d4["b_disc"] > 0.05 and d4["n_bdisc_pos"] >= 11 and
                 d4["pairdiv"] < 0.95)
        verdict = ("READS the back view (4v back-half discrimination >> 1v)" if reads else
                   "does NOT use the back view (4v ~= 1v ~= 0 discrimination)")
        summ["groups"][f"{mname}_verdict"] = verdict
        print(f"[verdict] {mname}: b_disc 1v={d1['b_disc']:+.3f} -> 4v={d4['b_disc']:+.3f} "
              f"(gain {gain:+.3f}), pairdiv 4v={d4['pairdiv']:.3f} => {verdict}", flush=True)
    if g["IM_1v"]["iou_own"] < 0.05 and g["I1_1v"]["iou_own"] < 0.05:
        print("[caveat] absolute own-IoU ~ 0: synthetic shapes may be too far OOD — "
              "treat verdicts as inconclusive", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(summ, open(OUT_JSON, "w"), indent=2)
    print(f"[eval] wrote {OUT_JSON}", flush=True)

    # ---------------- grid png (seed 0) ----------------
    cell, pw = 256, 64 * 3 * 2
    cols = ["front view", "GT", "IM 1v", "IM 4v", "I1 1v", "I1 4v(MD)"]
    W = cell + 5 * (pw + 10) + 20
    H = 40 + len(recs) * (cell // 2 + 40)
    grid = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    tf = ImageFont.truetype(FONT, 20)
    rf = ImageFont.truetype(FONT, 14)
    xs = [0, cell + 10]
    for _ in range(4):
        xs.append(xs[-1] + pw + 10)
    for x, lab in zip(xs, cols):
        d.text((x + 4, 8), lab, fill=(0, 0, 0), font=tf)
    yy = 40
    for r in recs:
        sha = r["sha256"]
        im = Image.open(os.path.join(r["renders_dir"], "000.webp")).convert("RGB")
        grid.paste(im.resize((cell // 2, cell // 2)), (xs[0], yy))
        d.text((xs[0] + 2, yy + cell // 2 + 2), r["name"], fill=(0, 0, 0), font=rf)
        grid.paste(proj(gts[sha]), (xs[1], yy))
        for ci, (mn, md) in enumerate([("IM", "1v"), ("IM", "4v"), ("I1", "1v"), ("I1", "4v")]):
            grid.paste(proj(occs[(sha, mn, md, 0)]), (xs[2 + ci], yy))
            row = next(x for x in rows if x["name"] == r["name"] and x["model"] == mn
                       and x["mode"] == md)
            d.text((xs[2 + ci], yy + 130), f"own {row['iou_own']:.2f} bd {row['b_disc']:+.2f}",
                   fill=(60, 60, 60), font=rf)
        yy += cell // 2 + 40
    gp = f"{ROOT}/runs/cache_logs/synth_backpair_grid.png"
    grid.save(gp)
    print(f"[grid] {gp}", flush=True)
    print("SYNTH_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
