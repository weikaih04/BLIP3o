"""TRELLIS-1-style INFERENCE-TIME multi-image aggregation on the deployed I1 fusion SS model.

TRELLIS 1 (microsoft/TRELLIS) shipped multi-image on a single-image model as pure inference
aggregation — two modes:
  stochastic     : each denoising step uses ONE image's cond, cycling
                   (cond_indices = arange(num_steps) % num_images)
  multidiffusion : at each step, run the model once per image cond (CFG applied PER cond,
                   guidance interval respected) and average the guided velocity predictions.

Here: same trick on runs/fusion_ss_dpos_2n/checkpoint-22000 (deployed I1 fusion SS), on the
14 held-out assets, with FAITHFUL per-view single-view fusion conds (v22_heldout_perview,
built by scripts/t1_build_perview_conds.py over the pinned m00 4-view combos).

Variants (same seed-0 noise, official SS sampler settings):
  A       single good-view baseline == eval_im_vs_i1_ss.py's I1 baseline (must reproduce ~0.404)
  S1..S4  each of the 4 combo views singly
  MD      multidiffusion over the 4 per-view conds
  ST      stochastic cycling over the 4 per-view conds

Output: per-asset table + summary; JSON -> runs/cache_logs/t1mi_heldout.json
"""
import os, sys, json
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import numpy as np
import torch

from trellis2_blip3o import _paths  # noqa
from trellis2 import models as t2models  # type: ignore
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # type: ignore
import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import build_cond, good_view_b
from scripts.export_glb_fullchain import load_ss_flow, SSDEC

ROOT = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
SS_CKPT = os.environ.get("SS_CKPT", f"{ROOT}/runs/fusion_ss_dpos_2n/checkpoint-22000")
COND_I1 = os.environ.get("COND_I1", "/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_heldout")
COND_PV = os.environ.get("COND_PV", "/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_perview")
IM4 = os.environ.get("COND_IM", "/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_im4l")
MANI = os.environ.get("MANI", "/fsx/home/weikai.huang/3dgen/im_probe/heldout14.jsonl")
OUT_JSON = os.environ.get("OUT_JSON", f"{ROOT}/runs/cache_logs/t1mi_heldout.json")
# Official SS sampler settings (identical to eval_im_vs_i1_ss.py / fused_fullchain_eval)
SS_OFF = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
              guidance_interval=[0.6, 1.0], rescale_t=5.0)
SEED = 0


def build_cond_key(conn, dve, entry_dir, key):
    """EXACT copy of eval_fusion_v22.build_cond (fusion path), filename parameterized.
    Per-view conds go through the SAME transform as baseline A: connector + dpos pos_stamp
    (IMG_SPAN_FULL, uncond stamped too), dino + dve[0] (single-view slot-0, I1 training
    distribution), CFG uncond = [zeros-DINO ; connector(0)], keep-mask indexing."""
    a = np.load(os.path.join(entry_dir, f"{key}.npz"))
    qwen = torch.from_numpy(a["hidden"]).float().cuda()          # (Tq, 2048)
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()     # (Td, 1024)
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    # trellis2_blip3o.eval_cond wraps build_unified_cond — the function the
    # training loop calls — so eval and training cannot disagree about the dino
    # segment, the view embedding, or which drop is CFG.
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    return cond_uncond_from_tensors(conn, qwen, qmask, dino, dmask, dve,
                                    dino_view_ids=None)


class MultiDiffusionSampler(FlowEulerGuidanceIntervalSampler):
    """TRELLIS-1 'multidiffusion' mode: per step, one guided prediction PER cond
    (full CFG math per cond — interval + strength + rescale, via the unchanged mixin
    chain), then average the guided velocities. Top-level _inference_model is invoked
    exactly once per sampler step (recursion goes through super(), not self)."""

    def set_conds(self, conds, unconds):
        self._conds, self._unconds = conds, unconds
        self._log_once = True

    def _inference_model(self, model, x_t, t, cond, neg_cond, **kwargs):
        preds = [super(MultiDiffusionSampler, self)._inference_model(
                     model, x_t, t, c, neg_cond=u, **kwargs)
                 for c, u in zip(self._conds, self._unconds)]
        if self._log_once:
            print(f"[md-check] t={t:.3f}: queried {len(preds)} conds; per-view cond norms="
                  f"{['%.1f' % float(c.norm()) for c in self._conds]} guided-pred norms="
                  f"{['%.2f' % float(p.float().norm()) for p in preds]}", flush=True)
            self._log_once = False
        return sum(preds) / len(preds)


class StochasticSampler(FlowEulerGuidanceIntervalSampler):
    """TRELLIS-1 'stochastic' mode: step i uses cond[i % num_images] with normal CFG
    (== their cond_indices = arange(num_steps) % num_images; one top-level call/step)."""

    def set_conds(self, conds, unconds):
        self._conds, self._unconds = conds, unconds
        self._i = 0

    def _inference_model(self, model, x_t, t, cond, neg_cond, **kwargs):
        k = self._i % len(self._conds); self._i += 1
        return super()._inference_model(model, x_t, t, self._conds[k],
                                        neg_cond=self._unconds[k], **kwargs)


@torch.no_grad()
def sample_occ(flow, sampler, ssdec, cond, uncond, seed=SEED):
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution,
                        generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = sampler.sample(flow, noise, cond=cond, neg_cond=uncond, verbose=False, **SS_OFF).samples
    return (ssdec(z) > 0)[0, 0].cpu().numpy()


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else -1.0


def main():
    recs = [json.loads(l) for l in open(MANI)]
    print(f"[t1mi] {len(recs)} held-out assets  ckpt={SS_CKPT}", flush=True)
    print(f"[t1mi] SS_OFF={SS_OFF} seed={SEED}", flush=True)

    ssdec = t2models.from_pretrained(SSDEC).cuda().eval()
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    md_sampler = MultiDiffusionSampler(sigma_min=1e-5)
    st_sampler = StochasticSampler(sigma_min=1e-5)
    flow, conn, dve = load_ss_flow(SS_CKPT)
    print(f"[t1mi] dve={None if dve is None else tuple(dve.shape)} "
          f"pos_stamp={'yes' if getattr(conn, 'pos_stamp', None) is not None else 'no'}", flush=True)

    rows = []
    first = True
    for r in recs:
        sha = r["sha256"]; s8 = sha[:8]
        EV._VIEW_FILE = good_view_b(sha)                       # baseline-A parity
        ed_i1 = os.path.join(COND_I1, sha[:2], sha)
        ed_pv = os.path.join(COND_PV, sha[:2], sha)
        views = [int(v) for v in np.load(os.path.join(IM4, sha[:2], sha, "m00.npz"))["views"]]
        if not all(os.path.exists(os.path.join(ed_pv, f"v{v:03d}.npz")) for v in views):
            print(f"[skip] {s8}: missing per-view cond", flush=True); continue

        gz = torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None]
        with torch.no_grad():
            gt = (ssdec(gz) > 0)[0, 0].cpu().numpy()

        # A) baseline: deployed single good-view cond (EXACT eval_im_vs_i1_ss path)
        cA, uA = build_cond(conn, dve, ed_i1, qwen_only=False)
        ioA = iou(sample_occ(flow, sampler, ssdec, cA, uA), gt)

        # S1..S4) each combo view singly (same code path, per-view entry)
        pv = [build_cond_key(conn, dve, ed_pv, f"v{v:03d}") for v in views]
        if first:  # one-time distinctness check: the 4 conds must genuinely differ
            print(f"[t1mi-check] {s8} views={views} cond token counts="
                  f"{[c.shape[1] for c, _ in pv]} cond norms="
                  f"{['%.1f' % float(c.norm()) for c, _ in pv]} (must differ across views)",
                  flush=True)
        ioS = [iou(sample_occ(flow, sampler, ssdec, c, u), gt) for c, u in pv]

        # MD) multidiffusion over the 4 per-view conds
        md_sampler.set_conds([c for c, _ in pv], [u for _, u in pv])
        md_sampler._log_once = first
        ioMD = iou(sample_occ(flow, md_sampler, ssdec, pv[0][0], pv[0][1]), gt)

        # ST) stochastic cycling
        st_sampler.set_conds([c for c, _ in pv], [u for _, u in pv])
        ioST = iou(sample_occ(flow, st_sampler, ssdec, pv[0][0], pv[0][1]), gt)

        first = False
        rows.append(dict(sha8=s8, views=views, A=ioA, S=ioS, MD=ioMD, ST=ioST))
        print(f"[t1mi] {s8}  A={ioA:.3f}  S=[{' '.join('%.3f' % s for s in ioS)}]  "
              f"bestS={max(ioS):.3f}  MD={ioMD:.3f}  ST={ioST:.3f}", flush=True)

    A = np.array([r["A"] for r in rows])
    S = np.array([r["S"] for r in rows])           # (n, 4)
    MD = np.array([r["MD"] for r in rows])
    ST = np.array([r["ST"] for r in rows])
    bS = S.max(1)
    n = len(rows)
    summ = {
        "ckpt": SS_CKPT, "n": n, "seed": SEED, "sampler": SS_OFF,
        "mean_A": float(A.mean()), "median_A": float(np.median(A)),
        "mean_single": float(S.mean()), "median_single": float(np.median(S)),
        "mean_bestS": float(bS.mean()), "median_bestS": float(np.median(bS)),
        "mean_MD": float(MD.mean()), "median_MD": float(np.median(MD)),
        "mean_ST": float(ST.mean()), "median_ST": float(np.median(ST)),
        "delta_MD_minus_A": float(MD.mean() - A.mean()),
        "delta_ST_minus_A": float(ST.mean() - A.mean()),
        "delta_MD_minus_bestS": float(MD.mean() - bS.mean()),
        "delta_MD_minus_meanS": float(MD.mean() - S.mean()),
        "n_MD_gt_A": int((MD > A).sum()), "n_ST_gt_A": int((ST > A).sum()),
        "n_MD_gt_bestS": int((MD > bS).sum()), "n_MD_gt_ST": int((MD > ST).sum()),
        "per_asset": rows,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(summ, open(OUT_JSON, "w"), indent=2)

    print("\n=== T1 MULTI-IMAGE INFERENCE SUMMARY ===", flush=True)
    print(f"  n={n} seed={SEED} ckpt={os.path.basename(SS_CKPT.rstrip('/'))}", flush=True)
    print(f"  mean   A={summ['mean_A']:.3f}  singles={summ['mean_single']:.3f}  "
          f"bestS={summ['mean_bestS']:.3f}  MD={summ['mean_MD']:.3f}  ST={summ['mean_ST']:.3f}",
          flush=True)
    print(f"  median A={summ['median_A']:.3f}  singles={summ['median_single']:.3f}  "
          f"bestS={summ['median_bestS']:.3f}  MD={summ['median_MD']:.3f}  "
          f"ST={summ['median_ST']:.3f}", flush=True)
    print(f"  deltas: MD-A={summ['delta_MD_minus_A']:+.4f}  ST-A={summ['delta_ST_minus_A']:+.4f}  "
          f"MD-bestS={summ['delta_MD_minus_bestS']:+.4f}  "
          f"MD-meanS={summ['delta_MD_minus_meanS']:+.4f}", flush=True)
    print(f"  counts: MD>A {summ['n_MD_gt_A']}/{n}  ST>A {summ['n_ST_gt_A']}/{n}  "
          f"MD>bestS {summ['n_MD_gt_bestS']}/{n}  MD>ST {summ['n_MD_gt_ST']}/{n}", flush=True)
    print(f"  wrote {OUT_JSON}", flush=True)
    ok = abs(summ["mean_A"] - 0.404) <= 0.01
    print(f"  BASELINE CHECK: mean_A={summ['mean_A']:.4f} vs 0.404 -> "
          f"{'OK' if ok else 'MISMATCH — do NOT trust MD/ST numbers'}", flush=True)
    print("T1MI_DONE", flush=True)


if __name__ == "__main__":
    main()
