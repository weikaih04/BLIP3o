"""alpha sweep: is our 2-phase cascade worse than MF's single-rollout limit?

MF (flux_rgbd/depth/schedule.py) has NO cascade mode. To generate both
modalities it only ever runs `joint` with a Mobius warp f_alpha(t); large alpha
keeps the second stream noisy so the first resolves early, but both streams stay
in the SAME rollout and see each other every step.

Our alpha=inf special-cases to two separate rollouts and pins t_s=0 for the
texture phase — telling the model "this geometry is CLEAN GT" while handing it a
generated mesh. That combination (t_s=0 with imperfect geometry) never appears
in training. Large-but-finite alpha avoids the lie.

If finite alpha beats alpha=inf, the 2-phase cascade is the problem, not joint."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
from trellis2_blip3o import _paths
from scripts.eval_fusion_v22 import build_cond, good_view_b
import scripts.eval_fusion_v22 as EV
from scripts.eval_geotex_g1g2 import pick_assets
from trellis2_blip3o.tr2_modules import load_norm_stats, TEX_SLAT_CONFIG_PATH
from scripts.eval.eval_overfit10 import load_scratch
from trellis2_blip3o.geotex_sampler import GeoTexSampler

CK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ALPHAS = [1.0, 8.0, 32.0, 128.0, 512.0, float("inf")]
m, conn, dve = load_scratch(CK); smp = GeoTexSampler(m)
sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
sm, ssd = sn["mean"].cuda(), sn["std"].cuda(); tm, tsd = tn["mean"].cuda(), tn["std"].cuda()
f = lambda z: z.feats if hasattr(z, "feats") else z
mse = lambda z, g: float(((f(z) - g) ** 2).mean())
acc = {a: {"g": [], "t": []} for a in ALPHAS}
for r, ed in pick_assets(N):
    sha = r["sha256"]; EV._VIEW_FILE = good_view_b(sha)
    gs, gt = np.load(r["shape_latent_512"]), np.load(r["pbr_latent_512"])
    cx = torch.from_numpy(gs["coords"]).int()
    coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
    s_gt = (torch.from_numpy(gs["feats"]).float().cuda() - sm) / ssd
    t_gt = (torch.from_numpy(gt["feats"]).float().cuda() - tm) / tsd
    c, u = build_cond(conn, dve, ed)
    line = f"{sha[:12]}"
    for a in ALPHAS:
        with torch.no_grad():
            xs, xt = smp.sample_joint(coords, c, u, c, u, alpha=a, seed=0)
        acc[a]["g"].append(mse(xs, s_gt)); acc[a]["t"].append(mse(xt, t_gt))
        line += f"  a={a:<6}g {acc[a]['g'][-1]:.3f} t {acc[a]['t'][-1]:.3f}"
    print(line, flush=True)
print(f"\n{'alpha':>8} {'geo':>8} {'tex':>8}   (inf = our 2-phase cascade)")
for a in ALPHAS:
    print(f"{str(a):>8} {np.mean(acc[a]['g']):8.4f} {np.mean(acc[a]['t']):8.4f}")
