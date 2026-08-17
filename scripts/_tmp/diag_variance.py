"""Is joint's texture the CONDITIONAL MEAN rather than a sample?

The alpha=32 warp leaves 90% of the texture trajectory to the final Euler step.
One big step from t~0.9 to 0 is, in flow matching, essentially a direct x_0
prediction — i.e. the MMSE estimate. That has the LOWEST possible MSE and the
WRONG statistics: too little variance, washed-out detail. If true, joint should
show (a) the best MSE and (b) markedly lower std than GT and than cascade."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
from trellis2_blip3o import _paths
from scripts.eval_fusion_v22 import build_cond, good_view_b
import scripts.eval_fusion_v22 as EV
from scripts.eval_geotex_g1g2 import pick_assets
from trellis2_blip3o.tr2_modules import load_norm_stats, TEX_SLAT_CONFIG_PATH
from scripts.eval.eval_overfit10 import load_scratch
from trellis2_blip3o.geotex_sampler import GeoTexSampler

CK, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
m, conn, dve = load_scratch(CK); smp = GeoTexSampler(m)
sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
sm, ssd = sn["mean"].cuda(), sn["std"].cuda(); tm, tsd = tn["mean"].cuda(), tn["std"].cuda()
f = lambda z: z.feats if hasattr(z, "feats") else z
acc = {k: {"mse": [], "std": []} for k in ("GT", "tex|GTmesh", "joint", "cascade")}
for r, ed in pick_assets(N):
    sha = r["sha256"]; EV._VIEW_FILE = good_view_b(sha)
    gs, gt = np.load(r["shape_latent_512"]), np.load(r["pbr_latent_512"])
    cx = torch.from_numpy(gs["coords"]).int()
    coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
    s_gt = (torch.from_numpy(gs["feats"]).float().cuda() - sm) / ssd
    t_gt = (torch.from_numpy(gt["feats"]).float().cuda() - tm) / tsd
    c, u = build_cond(conn, dve, ed)
    with torch.no_grad():
        x_tm = f(smp.sample_tex_given_mesh(coords, s_gt, c, c, u, seed=0))
        _, jt = smp.sample_joint(coords, c, u, c, u, alpha=32.0, seed=0)
        _, ct = smp.sample_joint(coords, c, u, c, u, alpha=float("inf"), seed=0)
    for k, z in (("GT", t_gt), ("tex|GTmesh", x_tm), ("joint", f(jt)), ("cascade", f(ct))):
        acc[k]["std"].append(float(z.std()))
        acc[k]["mse"].append(float(((z - t_gt) ** 2).mean()))
print(f"\n{'':12} {'latent MSE':>11} {'latent std':>11}  (GT 的 std 是参照)")
for k, v in acc.items():
    print(f"{k:12} {np.mean(v['mse']):11.4f} {np.mean(v['std']):11.4f}"
          + ("   <- 目标" if k == "GT" else
             f"   = GT 的 {np.mean(v['std'])/np.mean(acc['GT']['std']):.0%}"))
print("\n若 joint 的 std 明显低于 GT 和 cascade,则它输出的是条件均值(MSE 最低但视觉最糊)")
