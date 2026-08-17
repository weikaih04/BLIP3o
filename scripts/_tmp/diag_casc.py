"""级联为什么差?tex|GTmesh 和级联走同一个 sample_tex_given_mesh,
唯一差别是条件几何(真值 vs 生成)。三列拆开:
  B = 用生成几何做条件,但渲染在 GT 网格上  ← 只看纹理本身受了多大影响
  C = 用生成几何做条件,渲染在生成网格上    ← 就是级联
A 和 B 的差 = 条件几何变差对纹理的伤害;B 和 C 的差 = 网格本身难看。"""
import os, sys, numpy as np, torch, cv2
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
from trellis2_blip3o import _paths
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

CK=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 6
m,conn,dve = load_scratch(CK); smp = GeoTexSampler(m)
sn=load_norm_stats(TEX_SLAT_CONFIG_PATH,"shape_slat_normalization")
tn=load_norm_stats(TEX_SLAT_CONFIG_PATH,"pbr_slat_normalization")
sm,ssd=sn["mean"].cuda(),sn["std"].cuda(); tm,tsd=tn["mean"].cuda(),tn["std"].cuda()
sdec,tdec=build_sc_vae_shape_decoder_frozen().cuda().eval(),build_sc_vae_tex_decoder_frozen().cuda().eval()
env=EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(HDR,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)).cuda())
CELL,H=512,56
# ASCII only: this box has no CJK font (fc-list is empty), so any Chinese
# label renders as tofu boxes.
# Every label must say what is GENERATED and what is GT. An earlier version
# said "render GT mesh" for A/B, which reads as "this column is ground truth" —
# but the TEXTURE in A/B/C is always generated; only col 2 is GT texture.
cols=["input","GT mesh + GT tex  (all GT)",
      "A: GEN tex (cond: GT geo)  on GT mesh",
      "B: GEN tex (cond: GEN geo)  on GT mesh",
      "C: GEN tex (cond: GEN geo)  on GEN mesh = cascade"]
recs=pick_assets(N)
grid=Image.new("RGB",(CELL*len(cols),H+CELL*len(recs)),(250,250,250)); d=ImageDraw.Draw(grid)
fnt=ImageFont.truetype("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
  "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",21)
for c,l in enumerate(cols): d.text((c*CELL+12,12),l,fill=(10,10,10),font=fnt)
f=lambda z: z.feats if hasattr(z,"feats") else z
mse=lambda z,g: float(((f(z)-g)**2).mean())
accA=[];accB=[]
for ri,(r,ed) in enumerate(recs):
    sha=r["sha256"]; EV._VIEW_FILE=good_view_b(sha)
    gs,gt=np.load(r["shape_latent_512"]),np.load(r["pbr_latent_512"])
    cx=torch.from_numpy(gs["coords"]).int()
    coords=torch.cat([torch.zeros(cx.shape[0],1,dtype=torch.int32),cx],1)
    s_raw=torch.from_numpy(gs["feats"]).float().cuda(); t_raw=torch.from_numpy(gt["feats"]).float().cuda()
    s_gt=(s_raw-sm)/ssd; t_gt=(t_raw-tm)/tsd
    c,u=build_cond(conn,dve,ed)
    with torch.no_grad():
        xs = smp.sample_mesh_only(coords,c,u,seed=0)              # 生成几何
        tA = smp.sample_tex_given_mesh(coords,s_gt,   c,c,u,seed=1)  # 条件=GT
        tB = smp.sample_tex_given_mesh(coords,f(xs),  c,c,u,seed=1)  # 条件=生成(同 seed!)
    accA.append(mse(tA,t_gt)); accB.append(mse(tB,t_gt))
    rdir=r.get("renders_dir") or r.get("renders_cond_dir"); extr,intr=cam_from_transforms(rdir)
    R=lambda sf,tf: render_textured(sdec,tdec,coords,sf,tf,extr,intr,env)
    row=[input_image(rdir), R(s_raw,t_raw), R(s_raw,f(tA)*tsd+tm),
         R(s_raw,f(tB)*tsd+tm), R(f(xs)*ssd+sm,f(tB)*tsd+tm)]
    for c_,im in enumerate(row): grid.paste(im.resize((CELL,CELL)),(c_*CELL,H+ri*CELL))
    print(f"{sha[:12]}  A(条件GT) {accA[-1]:.4f}   B(条件生成) {accB[-1]:.4f}",flush=True)
print(f"\nMEAN  A {np.mean(accA):.4f}   B {np.mean(accB):.4f}   "
      f"条件几何变差让纹理 MSE {'+' if np.mean(accB)>np.mean(accA) else ''}{100*(np.mean(accB)/np.mean(accA)-1):.1f}%")
o="/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/diag_cascade.png"; grid.save(o); print("→",o)
