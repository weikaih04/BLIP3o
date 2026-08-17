"""G1/G2 gate evals for the trained unified geo-tex DiT (needs runs/geotex_s1_*/checkpoint-*).

G2 (tex|mesh ≈ tex specialist) — geometry LOCKED to GT shape latent:
  unified sample_tex_given_mesh vs specialist sample_tex, same seed/steps/cfg,
  scored as latent-MSE vs the GT pbr latent (norm space) + a rendered side-by-side
  grid (input | GT | specialist | unified). Operating threshold (adjustable, ours):
  unified mean MSE ≤ 1.05× specialist.

G1 (joint-lag vs cascade) — α sweep {1, 4, 32, ∞}, same seed:
  α=∞ IS the cascade (sampler dispatches to full-geo-solve → tex|mesh). Renders
  one row per α for visual comparison + reports shape/tex latent stats. Numeric
  shape anchor: decoded-occupancy IoU vs the GT shape voxels.

Usage (compute node):
  python scripts/eval_geotex_g1g2.py --ckpt runs/geotex_s1_v1/checkpoint-3000 \
      --coupling union --mode g2 --n 6
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis2_blip3o import _paths  # noqa: F401
from trellis2.modules import sparse as sp

import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import (build_cond, good_view_b, cam_from_transforms,
                                     input_image, load_flow_and_connector, sample_shape)
from scripts.eval_tex_v22 import (load_tex_flow, sample_tex, render_textured,
                                  COND_ROOT, MANI, HDR)
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen,
                                         load_norm_stats, TEX_SLAT_CONFIG_PATH)
from trellis2_blip3o.unified_geotex import load_unified_inference, load_run_connectors
from trellis2_blip3o.geotex_sampler import GeoTexSampler

TEX_SPEC_CKPT = os.environ.get("G2_TEX_SPEC", "runs/s3_tex_t50b/checkpoint-8000")


# ── render-space metrics ───────────────────────────────────────────────────
# Latent MSE is a proxy with two known failure modes on this project: image-to-3D
# is multi-modal (a sharper-but-different texture scores WORSE), and MSE has
# already ranked a visually worse model higher here (n=2, 2026-08). These two
# measure what a person actually sees, and — critically — FIDELITY is measured
# against the INPUT IMAGE, not the GT, so a model that drifts toward a plausible
# object that no longer matches its conditioning is caught. That drift is the
# specific degeneration risk of letting a cond stream be rewritten by the voxels.
_LPIPS = None


def _lpips_net():
    global _LPIPS
    if _LPIPS is None:
        import lpips as _l
        _LPIPS = _l.LPIPS(net="alex").cuda().eval()
    return _LPIPS


def _to_t(im, size=256):
    a = np.asarray(im.convert("RGB").resize((size, size))).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].cuda()


def render_metrics(pred_im, gt_im, input_im=None):
    """PSNR + LPIPS vs the GT render, and LPIPS vs the INPUT view (fidelity)."""
    p, g = _to_t(pred_im), _to_t(gt_im)
    psnr = float(10.0 * torch.log10(1.0 / torch.clamp(((p - g) ** 2).mean(), min=1e-10)))
    with torch.no_grad():
        lp = float(_lpips_net()(p * 2 - 1, g * 2 - 1).flatten()[0])
        fid = (float(_lpips_net()(p * 2 - 1, _to_t(input_im) * 2 - 1).flatten()[0])
               if input_im is not None else float("nan"))
    return psnr, lp, fid
# G3 red line: mesh-only geometry must not regress vs the shape specialist the
# geo stream was warm-started from (S2b unfreezes geo, so this is the gate that
# says whether unfreezing broke what already worked).
SHAPE_SPEC_CKPT = os.environ.get("G3_SHAPE_SPEC", "runs/s3_shape_t50b/checkpoint-8000")


def pick_assets(n, max_vox=8192, shard=0, num_shards=1):
    """shard/num_shards: round-robin over the manifest so N GPUs can each take
    a disjoint slice of the SAME asset list — the union is exactly what a single
    n-asset run would have picked, so shards are poolable without double-counting."""
    recs = []
    idx = -1
    with open(MANI) as f:
        for line in f:
            r = json.loads(line)
            v = r.get("vlm") or {}
            # EVAL_ANY=1 skips the quality gate — required for held-out manifests,
            # whose rows carry no vlm scores (and whose whole point is to be the
            # assets the model never saw, not the prettiest ones).
            if os.environ.get("EVAL_ANY") != "1" and not (
                    v.get("part_complexity", 0) >= 7 and v.get("structural_score", 0) >= 7
                    and v.get("texture_score", 0) >= 6):
                continue
            ed = os.path.join(COND_ROOT, r["sha256"][:2], r["sha256"])
            if os.path.exists(os.path.join(ed, "v000.npz")) and r.get("pbr_latent_512") \
                    and r.get("shape_latent_512"):
                if np.load(r["shape_latent_512"])["coords"].shape[0] <= max_vox:
                    idx += 1
                    if idx >= n:
                        break
                    if idx % num_shards == shard:
                        recs.append((r, ed))
    return recs


def load_geotex(ckpt, coupling, bidir=True, fused=True):
    # MUST match the trained topology: the run is bidirectional and used the
    # FUSED attention path (absolute stream tag). strict load would fail on
    # t_mixer_s / cross_alpha_s if bidir were off.
    uni = load_unified_inference(ckpt, coupling=coupling, bidirectional=bidir).cuda().eval()
    uni.fused_attn = fused
    # GEO1WAY=1 → disable geo's tex-read at INFERENCE only (weights untouched:
    # union coupling adds no parameters for that read). Tests the hypothesis that
    # G1's joint collapse = frozen geo eating tex keys it was never trained on.
    if os.environ.get("GEO1WAY") == "1":
        uni.bidirectional = False
        print("[geotex] INFERENCE one-way: geo does NOT read tex", flush=True)
    conn_g, conn_x = load_run_connectors(ckpt)
    conn_g = conn_g.cuda().eval().float()
    conn_x = conn_x.cuda().eval().float()
    from safetensors.torch import load_file
    sd = load_file(os.path.join(ckpt, "model.safetensors"))
    dve = sd.get("dino_view_embed")
    dve = dve.cuda().float() if dve is not None else None
    print(f"[geotex] loaded {ckpt} coupling={coupling} "
          f"cross_alpha={float(uni.cross_alpha):.4f} dve={'yes' if dve is not None else 'no'}")
    return uni, conn_g, conn_x, dve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--coupling", default="union")
    ap.add_argument("--mode", choices=["g1", "g2", "g3", "g4"], default="g2")
    ap.add_argument("--n", type=int, default=6)
    # None = released TRELLIS.2-4B per-stream params. Passing either overrides
    # BOTH streams and deviates from the release (see geotex_sampler docstring).
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", default="1,4,32,inf")
    ap.add_argument("--out", default="runs/cache_logs/eval_geotex")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from trellis2.renderers import EnvMap

    recs = pick_assets(a.n, shard=a.shard, num_shards=a.num_shards)
    print(f"[{a.mode}] {len(recs)} held-out assets")
    uni, conn_g, conn_x, dve = load_geotex(a.ckpt, a.coupling)
    smp = GeoTexSampler(uni, steps=a.steps, cfg=a.cfg)

    sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    sm, ssd = sn["mean"].cuda(), sn["std"].cuda()
    tm, tsd = tn["mean"].cuda(), tn["std"].cuda()
    shape_dec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tex_dec = build_sc_vae_tex_decoder_frozen().cuda().eval()
    hdr = cv2.cvtColor(cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(hdr).cuda())
    font = ImageFont.truetype(
        "/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 40)

    cell, hdrh = 640, 64

    if a.mode == "g4":
        # END-TO-END PRODUCT COMPARISON, the one the other gates do not make:
        # unified joint (one model, geometry+texture together) vs the SHIPPING
        # cascade (shape specialist -> tex specialist). Neither side gets GT
        # geometry: the cascade's texture is conditioned on the geometry the
        # shape specialist just produced, exactly as in deployment. Scored on
        # BOTH latents, because joint's risk is texture, not geometry (G1'
        # already showed geometry is free).
        sflow, sconn, sdve = load_flow_and_connector(SHAPE_SPEC_CKPT)
        tflow, tconn, tdve = load_tex_flow(TEX_SPEC_CKPT)
        alphas = [float(x) if x != "inf" else float("inf") for x in a.alphas.split(",")]
        cols = ["input", "GT", "cascade (2 specialists)"] + [f"joint a={al}" for al in alphas]
        grid = Image.new("RGB", (cell * len(cols), hdrh + cell * len(recs)), (250, 250, 250))
        d = ImageDraw.Draw(grid)
        for c, lab in enumerate(cols):
            d.text((c * cell + 12, 12), lab, fill=(10, 10, 10), font=font)
        cs_s, cs_t = [], []
        js_s, js_t = {}, {}
        for ri, (r, ed) in enumerate(recs):
            sha = r["sha256"]
            EV._VIEW_FILE = good_view_b(sha)
            gt_s = np.load(r["shape_latent_512"]); gt_t = np.load(r["pbr_latent_512"])
            cx = torch.from_numpy(gt_s["coords"]).int()
            coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
            shape_raw = torch.from_numpy(gt_s["feats"]).float().cuda()
            shape_gt_norm = (shape_raw - sm) / ssd
            tex_gt_norm = (torch.from_numpy(gt_t["feats"]).float().cuda() - tm) / tsd

            c_s, u_s = build_cond(conn_g, dve, ed)
            c_x, u_x = build_cond(conn_x, dve, ed)
            c_ss, u_ss = build_cond(sconn, sdve, ed)
            c_xs, u_xs = build_cond(tconn, tdve, ed)

            # cascade: geometry first, then texture ON THAT geometry (not GT)
            s_c = sample_shape(sflow, c_ss, u_ss, coords, steps=a.steps, cfg=a.cfg, seed=a.seed)
            f = lambda z: (z.feats if hasattr(z, "feats") else z)
            t_c = sample_tex(tflow, c_xs, u_xs, coords, f(s_c),
                             steps=a.steps, cfg=a.cfg, seed=a.seed)
            e = lambda z, g: ((f(z) - g) ** 2).mean().item()
            cs_s.append(e(s_c, shape_gt_norm)); cs_t.append(e(t_c, tex_gt_norm))

            rdir = r.get("renders_dir") or r.get("renders_cond_dir")
            extr, intr = cam_from_transforms(rdir)
            row = [input_image(rdir),
                   render_textured(shape_dec, tex_dec, coords, shape_raw,
                                   torch.from_numpy(gt_t["feats"]).float().cuda(),
                                   extr, intr, envmap),
                   render_textured(shape_dec, tex_dec, coords, f(s_c) * ssd + sm,
                                   t_c * tsd + tm, extr, intr, envmap)]
            line = f"[g4] {sha[:12]} cascade: shape={cs_s[-1]:.4f} tex={cs_t[-1]:.4f}"
            for al in alphas:
                xs, xt = smp.sample_joint(coords, c_s, u_s, c_x, u_x, alpha=al, seed=a.seed)
                js_s.setdefault(al, []).append(e(xs, shape_gt_norm))
                js_t.setdefault(al, []).append(e(xt, tex_gt_norm))
                row.append(render_textured(shape_dec, tex_dec, coords, xs.feats * ssd + sm,
                                           xt * tsd + tm, extr, intr, envmap))
                line += f" | a={al}: shape={js_s[al][-1]:.4f} tex={js_t[al][-1]:.4f}"
            print(line, flush=True)
            for c, im in enumerate(row):
                grid.paste(im.resize((cell, cell)), (c * cell, hdrh + ri * cell))
        gp = os.path.join(a.out, f"g4_{os.path.basename(os.path.dirname(a.ckpt))}_sh{a.shard}.png")
        grid.save(gp)
        Cs, Ct = float(np.mean(cs_s)), float(np.mean(cs_t))
        print(f"[G4] cascade (shape spec -> tex spec): shape {Cs:.4f}  tex {Ct:.4f}")
        for al in alphas:
            S, T = float(np.mean(js_s[al])), float(np.mean(js_t[al]))
            print(f"[G4] joint a={al}: shape {S:.4f} ({S / Cs:.3f}x)  "
                  f"tex {T:.4f} ({T / Ct:.3f}x)")
        print(f"[G4] grid: {gp}")

    elif a.mode == "g3":
        # Geometry is scored two ways because the bidirectional model has two
        # legitimate mesh-only dispatches:
        #   structural  = geo lane alone (no tex tokens at all) — identical in
        #                 shape to the specialist's own trajectory;
        #   marginal    = tex lane pinned at t_x=1 with fresh noise each step,
        #                 which is what corner2 (20% of training) teaches and
        #                 what the deployed mesh-only path actually runs.
        # A gap between them is itself a finding: it means the geo stream leans
        # on tex tokens it should be able to ignore.
        sflow, sconn, sdve = load_flow_and_connector(SHAPE_SPEC_CKPT)
        cols = ["input", "GT geometry", "shape specialist", "unified mesh-only",
                "unified mesh-only (marginal)"]
        grid = Image.new("RGB", (cell * len(cols), hdrh + cell * len(recs)), (250, 250, 250))
        d = ImageDraw.Draw(grid)
        for c, lab in enumerate(cols):
            d.text((c * cell + 12, 12), lab, fill=(10, 10, 10), font=font)
        m_spec, m_str, m_mar = [], [], []
        for ri, (r, ed) in enumerate(recs):
            sha = r["sha256"]
            EV._VIEW_FILE = good_view_b(sha)
            gt_s = np.load(r["shape_latent_512"]); gt_t = np.load(r["pbr_latent_512"])
            cx = torch.from_numpy(gt_s["coords"]).int()
            coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
            shape_raw = torch.from_numpy(gt_s["feats"]).float().cuda()
            shape_gt_norm = (shape_raw - sm) / ssd
            tex_gt_raw = torch.from_numpy(gt_t["feats"]).float().cuda()

            c_s, u_s = build_cond(conn_g, dve, ed)
            c_x, u_x = build_cond(conn_x, dve, ed)
            c_ss, u_ss = build_cond(sconn, sdve, ed)

            s_spec = sample_shape(sflow, c_ss, u_ss, coords,
                                  steps=a.steps, cfg=a.cfg, seed=a.seed)
            s_str = smp.sample_mesh_only(coords, c_s, u_s, seed=a.seed)
            s_mar = smp.sample_mesh_only_marginal(coords, c_s, u_s, c_x, u_x, seed=a.seed)
            f = lambda z: (z.feats if hasattr(z, "feats") else z)
            e = lambda z: ((f(z) - shape_gt_norm) ** 2).mean().item()
            ms, mt, mm = e(s_spec), e(s_str), e(s_mar)
            m_spec.append(ms); m_str.append(mt); m_mar.append(mm)
            print(f"[g3] {sha[:12]}  spec={ms:.4f}  unified={mt:.4f}  marginal={mm:.4f}",
                  flush=True)

            # every column rendered with the SAME GT texture, so any visible
            # difference is geometry and nothing else (coords are shared — they
            # come from the GT sparse structure in all four cases).
            rdir = r.get("renders_dir") or r.get("renders_cond_dir")
            extr, intr = cam_from_transforms(rdir)
            imgs = [input_image(rdir)] + [
                render_textured(shape_dec, tex_dec, coords, z, tex_gt_raw, extr, intr, envmap)
                for z in (shape_raw,
                          f(s_spec) * ssd + sm,
                          f(s_str) * ssd + sm,
                          f(s_mar) * ssd + sm)]
            for c, im in enumerate(imgs):
                grid.paste(im.resize((cell, cell)), (c * cell, hdrh + ri * cell))
        gp = os.path.join(a.out, f"g3_{os.path.basename(os.path.dirname(a.ckpt))}_sh{a.shard}.png")
        grid.save(gp)
        ms, mt, mm = float(np.mean(m_spec)), float(np.mean(m_str)), float(np.mean(m_mar))
        v1 = "PASS" if mt <= 1.05 * ms else "FAIL"
        v2 = "PASS" if mm <= 1.05 * ms else "FAIL"
        print(f"[G3] shape specialist mean MSE {ms:.4f}\n"
              f"[G3] unified mesh-only  {mt:.4f}  ratio {mt / ms:.3f} -> {v1}\n"
              f"[G3] unified marginal   {mm:.4f}  ratio {mm / ms:.3f} -> {v2}\n"
              f"[G3] grid: {gp}")

    elif a.mode == "g2":
        tflow, tconn, tdve = load_tex_flow(TEX_SPEC_CKPT)
        cols = ["input", "GT textured", "tex specialist", "unified tex|mesh"]
        grid = Image.new("RGB", (cell * len(cols), hdrh + cell * len(recs)), (250, 250, 250))
        d = ImageDraw.Draw(grid)
        for c, lab in enumerate(cols):
            d.text((c * cell + 12, 12), lab, fill=(10, 10, 10), font=font)
        mses_spec, mses_uni = [], []
        rmet = {}
        for ri, (r, ed) in enumerate(recs):
            sha = r["sha256"]
            EV._VIEW_FILE = good_view_b(sha)
            gt_s = np.load(r["shape_latent_512"]); gt_t = np.load(r["pbr_latent_512"])
            cx = torch.from_numpy(gt_s["coords"]).int()
            coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
            shape_raw = torch.from_numpy(gt_s["feats"]).float().cuda()
            tex_gt_norm = ((torch.from_numpy(gt_t["feats"]).float().cuda() - tm) / tsd)
            shape_norm = (shape_raw - sm) / ssd

            c_s, _u_s = build_cond(conn_g, dve, ed)
            c_x, u_x = build_cond(conn_x, dve, ed)
            c_xs, u_xs = build_cond(tconn, tdve, ed)

            t_spec = sample_tex(tflow, c_xs, u_xs, coords, shape_norm,
                                steps=a.steps, cfg=a.cfg, seed=a.seed)
            t_uni = smp.sample_tex_given_mesh(coords, shape_norm, c_s, c_x, u_x, seed=a.seed)
            m_s = ((t_spec - tex_gt_norm) ** 2).mean().item()
            m_u = ((t_uni - tex_gt_norm) ** 2).mean().item()
            mses_spec.append(m_s); mses_uni.append(m_u)
            print(f"[g2] {sha[:12]}  spec_mse={m_s:.4f}  unified_mse={m_u:.4f}")

            rdir = r.get("renders_dir") or r.get("renders_cond_dir")
            extr, intr = cam_from_transforms(rdir)
            imgs = [input_image(rdir),
                    render_textured(shape_dec, tex_dec, coords, shape_raw,
                                    torch.from_numpy(gt_t["feats"]).float().cuda(),
                                    extr, intr, envmap),
                    render_textured(shape_dec, tex_dec, coords, shape_raw,
                                    t_spec * tsd + tm, extr, intr, envmap),
                    render_textured(shape_dec, tex_dec, coords, shape_raw,
                                    t_uni * tsd + tm, extr, intr, envmap)]
            # imgs = [input, GT, specialist, unified]; score both models in
            # render space against GT, and against the INPUT for fidelity.
            for tag, im in (("spec", imgs[2]), ("uni", imgs[3])):
                ps, lp, fid = render_metrics(im, imgs[1], imgs[0])
                rmet.setdefault(tag, []).append((ps, lp, fid))
                print(f"[g2r] {sha[:12]} {tag}: psnr={ps:.2f} lpips={lp:.4f} "
                      f"fidelity(vs input)={fid:.4f}", flush=True)
            for c, im in enumerate(imgs):
                grid.paste(im.resize((cell, cell)), (c * cell, hdrh + ri * cell))
        gp = os.path.join(a.out, f"g2_{os.path.basename(os.path.dirname(a.ckpt))}_sh{a.shard}.png")
        grid.save(gp)
        ms, mu = float(np.mean(mses_spec)), float(np.mean(mses_uni))
        verdict = "PASS" if mu <= 1.05 * ms else "FAIL"
        print(f"[G2] specialist mean MSE {ms:.4f}  unified {mu:.4f}  "
              f"ratio {mu / ms:.3f}  (op-threshold 1.05) → {verdict}")
        for tag in ("spec", "uni"):
            if tag in rmet:
                v = np.array(rmet[tag])
                print(f"[G2r] {tag}: PSNR {v[:, 0].mean():.2f}  LPIPS(vs GT) "
                      f"{v[:, 1].mean():.4f}  fidelity(vs input) {v[:, 2].mean():.4f}")
        print(f"[G2] grid: {gp}")

    else:  # g1
        alphas = [float("inf") if s == "inf" else float(s) for s in a.alphas.split(",")]
        cols = ["input", "GT textured"] + [f"joint a={('inf' if np.isinf(al) else int(al))}"
                                           for al in alphas]
        grid = Image.new("RGB", (cell * len(cols), hdrh + cell * len(recs)), (250, 250, 250))
        d = ImageDraw.Draw(grid)
        for c, lab in enumerate(cols):
            d.text((c * cell + 12, 12), lab, fill=(10, 10, 10), font=font)
        g1_solo, g1_joint = [], {}
        for ri, (r, ed) in enumerate(recs):
            sha = r["sha256"]
            EV._VIEW_FILE = good_view_b(sha)
            gt_s = np.load(r["shape_latent_512"]); gt_t = np.load(r["pbr_latent_512"])
            cx = torch.from_numpy(gt_s["coords"]).int()
            coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
            shape_raw = torch.from_numpy(gt_s["feats"]).float().cuda()
            shape_gt_norm = (shape_raw - sm) / ssd
            c_s, u_s = build_cond(conn_g, dve, ed)
            c_x, u_x = build_cond(conn_x, dve, ed)
            # G1' anchor: joint geometry must not be worse than the SAME model's
            # mesh-only geometry. Both are this model on this asset with this
            # seed, so the comparison isolates "does co-generating texture cost
            # geometry" — the only question that says whether unification paid.
            s_solo = smp.sample_mesh_only(coords, c_s, u_s, seed=a.seed)
            m_solo = ((s_solo.feats - shape_gt_norm) ** 2).mean().item()
            g1_solo.append(m_solo)
            print(f"[g1] {sha[:12]} mesh-only: shape_mse={m_solo:.4f}", flush=True)
            rdir = r.get("renders_dir") or r.get("renders_cond_dir")
            extr, intr = cam_from_transforms(rdir)
            row = [input_image(rdir),
                   render_textured(shape_dec, tex_dec, coords, shape_raw,
                                   torch.from_numpy(gt_t["feats"]).float().cuda(),
                                   extr, intr, envmap)]
            for al in alphas:
                xs, xt = smp.sample_joint(coords, c_s, u_s, c_x, u_x, alpha=al, seed=a.seed)
                row.append(render_textured(shape_dec, tex_dec, coords,
                                           xs.feats * ssd + sm, xt * tsd + tm,
                                           extr, intr, envmap))
                m_j = ((xs.feats - shape_gt_norm) ** 2).mean().item()
                g1_joint.setdefault(al, []).append(m_j)
                print(f"[g1] {sha[:12]} a={al}: shape_mse={m_j:.4f} "
                      f"(vs mesh-only {m_solo:.4f}, ratio {m_j / m_solo:.3f})  "
                      f"shape_std={xs.feats.std():.3f} tex_std={xt.std():.3f}", flush=True)
            for c, im in enumerate(row):
                grid.paste(im.resize((cell, cell)), (c * cell, hdrh + ri * cell))
        gp = os.path.join(a.out, f"g1_{os.path.basename(os.path.dirname(a.ckpt))}_sh{a.shard}.png")
        grid.save(gp)
        solo = float(np.mean(g1_solo))
        print(f"[G1] mesh-only geometry (same model, same seed): mean shape MSE {solo:.4f}")
        for al in alphas:
            v = float(np.mean(g1_joint[al]))
            tag = "PASS" if v <= 1.05 * solo else "FAIL"
            print(f"[G1] joint a={al}: mean shape MSE {v:.4f}  ratio {v / solo:.3f} -> {tag}")
        print(f"[G1] grid: {gp}  (cascade = the a=inf column; judge lag columns against it)")


if __name__ == "__main__":
    main()
