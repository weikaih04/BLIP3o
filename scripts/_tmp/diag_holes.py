"""Are the "holes" real geometry, or alpha rendering the surface transparent?

render_textured takes the "shaded" channel, and pbr_mesh_renderer.py:453-458
composites it as w = (1-alpha)*gb_alpha with a BLACK background — so a low
decoded PBR alpha makes a watertight mesh look full of holes.

"normal" and "mask" come straight off the rasterizer's first layer and never
touch alpha (pbr_mesh_renderer.py:318-321). Render both: if normal is solid
where shaded has holes, the mesh is fine and the alpha channel is the problem."""
import os, sys, numpy as np, torch, cv2
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
from trellis2_blip3o import _paths
from trellis2.renderers import EnvMap
from PIL import Image, ImageDraw, ImageFont
from scripts.eval_fusion_v22 import build_cond, good_view_b, cam_from_transforms, input_image
import scripts.eval_fusion_v22 as EV
from scripts.eval_geotex_g1g2 import pick_assets
import scripts.eval_tex_v22 as ET
from scripts.eval_tex_v22 import render_textured, HDR
from trellis2_blip3o.tr2_modules import (load_norm_stats, TEX_SLAT_CONFIG_PATH,
    build_sc_vae_shape_decoder_frozen, build_sc_vae_tex_decoder_frozen)
from scripts.eval.eval_overfit10 import load_scratch
from trellis2_blip3o.geotex_sampler import GeoTexSampler

CK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
m, conn, dve = load_scratch(CK); smp = GeoTexSampler(m)
sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
sm, ssd = sn["mean"].cuda(), sn["std"].cuda(); tm, tsd = tn["mean"].cuda(), tn["std"].cuda()
sdec, tdec = build_sc_vae_shape_decoder_frozen().cuda().eval(), build_sc_vae_tex_decoder_frozen().cuda().eval()
env = EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)).cuda())
f = lambda z: z.feats if hasattr(z, "feats") else z
CELL, H = 512, 56
cols = ["GT shaded", "GT normal", "GEN shaded (what we saw)", "GEN normal (alpha-free)"]
recs = pick_assets(N)
grid = Image.new("RGB", (CELL*len(cols), H+CELL*len(recs)), (250,250,250)); d = ImageDraw.Draw(grid)
fnt = ImageFont.truetype("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
                         "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 22)
for c, l in enumerate(cols): d.text((c*CELL+12, 14), l, fill=(10,10,10), font=fnt)
for ri, (r, ed) in enumerate(recs):
    sha = r["sha256"]; EV._VIEW_FILE = good_view_b(sha)
    gs, gt = np.load(r["shape_latent_512"]), np.load(r["pbr_latent_512"])
    cx = torch.from_numpy(gs["coords"]).int()
    coords = torch.cat([torch.zeros(cx.shape[0],1,dtype=torch.int32), cx], 1)
    s_raw = torch.from_numpy(gs["feats"]).float().cuda(); t_raw = torch.from_numpy(gt["feats"]).float().cuda()
    c, u = build_cond(conn, dve, ed)
    with torch.no_grad():
        xs, xt = smp.sample_joint(coords, c, u, c, u, alpha=32.0, seed=0)
    rdir = r.get("renders_dir") or r.get("renders_cond_dir"); extr, intr = cam_from_transforms(rdir)
    row = []
    for sf, tf in ((s_raw, t_raw), (f(xs)*ssd+sm, f(xt)*tsd+tm)):
        for ch in ("shaded", "normal"):
            row.append(render_textured(sdec, tdec, coords, sf, tf, extr, intr, env, channel=ch))
    for c_, im in enumerate(row): grid.paste(im.resize((CELL,CELL)), (c_*CELL, H+ri*CELL))
    print(f"{sha[:12]} ok", flush=True)
o = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/diag_holes.png"; grid.save(o); print("->", o)
