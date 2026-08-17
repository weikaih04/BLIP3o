"""诊断:tex|GTmesh 看起来好很多,是纹理更好,还是几何在背锅?
把 JOINT 生成的纹理贴到 GT 网格上 —— 如果这样就好看了,说明纹理没问题,
差距全在几何。这是唯一能把两者分开的对照。"""
import os, sys, numpy as np, torch, cv2
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
from trellis2_blip3o import _paths
from trellis2.modules import sparse as sp
from trellis2.renderers import EnvMap
from PIL import Image, ImageDraw, ImageFont
from scripts.eval_fusion_v22 import build_cond, good_view_b, cam_from_transforms, input_image
import scripts.eval_fusion_v22 as EV
from scripts.eval_geotex_g1g2 import pick_assets
from scripts.eval_tex_v22 import render_textured, HDR
from trellis2_blip3o.tr2_modules import (load_norm_stats, TEX_SLAT_CONFIG_PATH,
    build_sc_vae_shape_decoder_frozen, build_sc_vae_tex_decoder_frozen)
from scripts.eval.eval_overfit10 import load_scratch
from trellis2_blip3o.geotex_sampler import GeoTexSampler

CK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
m, conn, dve = load_scratch(CK)
smp = GeoTexSampler(m)
sn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
sm, ssd = sn["mean"].cuda(), sn["std"].cuda(); tm, tsd = tn["mean"].cuda(), tn["std"].cuda()
sdec, tdec = build_sc_vae_shape_decoder_frozen().cuda().eval(), build_sc_vae_tex_decoder_frozen().cuda().eval()
env = EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(HDR, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)).cuda())
CELL, H = 512, 56
# ASCII only — no CJK font on this box.
cols = ["input", "GT mesh + GT tex", "GT mesh + tex|mesh texture",
        "GT mesh + JOINT texture", "JOINT mesh + JOINT texture"]
recs = pick_assets(N)
grid = Image.new("RGB", (CELL*len(cols), H+CELL*len(recs)), (250,250,250))
d = ImageDraw.Draw(grid)
fnt = ImageFont.truetype("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
                         "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 26)
for c,l in enumerate(cols): d.text((c*CELL+12,12), l, fill=(10,10,10), font=fnt)
f = lambda z: z.feats if hasattr(z,"feats") else z
for ri,(r,ed) in enumerate(recs):
    sha = r["sha256"]; EV._VIEW_FILE = good_view_b(sha)
    gs, gt = np.load(r["shape_latent_512"]), np.load(r["pbr_latent_512"])
    cx = torch.from_numpy(gs["coords"]).int()
    coords = torch.cat([torch.zeros(cx.shape[0],1,dtype=torch.int32), cx],1)
    s_raw = torch.from_numpy(gs["feats"]).float().cuda(); t_raw = torch.from_numpy(gt["feats"]).float().cuda()
    s_gt = (s_raw-sm)/ssd
    c,u = build_cond(conn, dve, ed)
    with torch.no_grad():
        x_tm = smp.sample_tex_given_mesh(coords, s_gt, c, c, u, seed=0)
        js, jt = smp.sample_joint(coords, c, u, c, u, alpha=32.0, seed=0)
    rdir = r.get("renders_dir") or r.get("renders_cond_dir"); extr,intr = cam_from_transforms(rdir)
    R = lambda sf,tf: render_textured(sdec,tdec,coords,sf,tf,extr,intr,env)
    row = [input_image(rdir), R(s_raw,t_raw), R(s_raw, f(x_tm)*tsd+tm),
           R(s_raw, f(jt)*tsd+tm),                       # ← 关键对照
           R(f(js)*ssd+sm, f(jt)*tsd+tm)]
    for c_,im in enumerate(row): grid.paste(im.resize((CELL,CELL)), (c_*CELL, H+ri*CELL))
    print(f"{sha[:12]} ok", flush=True)
out = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/diag_tex_vs_geo.png"
grid.save(out); print("→", out)
