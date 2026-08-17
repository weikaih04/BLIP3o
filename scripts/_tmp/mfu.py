"""MFU for MMDiT3D. FLOPs 解析式列明每一项,时间用实测。"""
import sys, time, torch
sys.path.insert(0,"/fsx/home/weikai.huang/3dgen/model/BLIP3o")
from trellis2_blip3o.mmdit3d import MMDiT3D
from trellis2.modules import sparse as sp
from trellis2.utils.elastic_utils import LinearMemoryController

PEAK = 989e12                      # H200 SXM bf16 稠密峰值 FLOP/s
D = int(__import__("os").environ.get("DIM",768)); H, MLP = D//128, 4.0
ND, NS = int(__import__("os").environ.get("ND",8)), int(__import__("os").environ.get("NS",16))                     # 三流块 / 共享块
import os
BS, NVOX, NCOND = int(os.environ.get("BS",4)), 2222, 1024    # 每卡 batch · SLAT 中位体素数 · cond token 数

def flops_fwd(bs, nvox, ncond):
    L = 2*nvox + ncond             # 联合序列:geo + tex + cond
    per_block_dense = 0
    for T in (nvox, nvox, ncond):
        per_block_dense += 2*T*D*3*D          # QKV
        per_block_dense += 2*T*D*D            # out proj
        per_block_dense += 2*2*T*D*(MLP*D)    # MLP 两个矩阵
    per_block_attn = 2*2*L*L*D                # QK^T + AV
    return bs * (ND+NS) * (per_block_dense + per_block_attn)

m = MMDiT3D(dim=D, num_heads=H, depth_double=ND, depth_single=NS).cuda().train()
import os as _o
ctrl = LinearMemoryController(target_ratio=float(_o.environ.get("RATIO",0.75))); m.register_memory_controller(ctrl)
g = torch.Generator(device="cuda").manual_seed(0)
co=[];fs=[];fx=[]
for b in range(BS):
    c=torch.randint(0,32,(NVOX,3),generator=g,device="cuda",dtype=torch.int32)
    co.append(torch.cat([torch.full((NVOX,1),b,dtype=torch.int32,device="cuda"),c],1))
    fs.append(torch.randn(NVOX,32,generator=g,device="cuda")); fx.append(torch.randn(NVOX,32,generator=g,device="cuda"))
co=torch.cat(co); x_s=sp.SparseTensor(torch.cat(fs),co); x_x=sp.SparseTensor(torch.cat(fx),co)
cond=[torch.randn(NCOND,1024,generator=g,device="cuda") for _ in range(BS)]
t_s=torch.rand(BS,device="cuda")*1000; t_x=torch.rand(BS,device="cuda")*1000

def bench(elastic, iters=12):
    for i in range(iters+4):
        if i==4: torch.cuda.synchronize(); t0=time.time()
        m.zero_grad(set_to_none=True)
        if elastic:
            with ctrl.record():
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    a,b=m(x_s,x_x,t_s,t_x,cond,cond,tex_concat_cond=x_s)
                (a.feats.float().pow(2).mean()+b.feats.float().pow(2).mean()).backward()
        else:
            m._ckpt_upto=0
            with torch.autocast("cuda",dtype=torch.bfloat16):
                a,b=m._forward_impl(x_s,x_x,t_s,t_x,cond,cond,x_s)
            (a.feats.float().pow(2).mean()+b.feats.float().pow(2).mean()).backward()
            m._ckpt_upto=None
    torch.cuda.synchronize(); return (time.time()-t0)/iters

F = flops_fwd(BS, NVOX, NCOND)
print(f"参数 {sum(p.numel() for p in m.parameters())/1e6:.0f}M · 配置 bs{BS}/卡 · 体素 {NVOX} · cond {NCOND} · 联合序列 {2*NVOX+NCOND} tok/样本 · {ND}+{NS} 块 · dim {D}")
print(f"单次前向 {F/1e12:.2f} TFLOP/卡  (稠密 {100*(F-BS*(ND+NS)*2*2*(2*NVOX+NCOND)**2*D)/F:.0f}% · 注意力 {100*BS*(ND+NS)*2*2*(2*NVOX+NCOND)**2*D/F:.0f}%)")
import os as _o2
for name, elastic in ((f"弹性 GC ratio={_o2.environ.get('RATIO',0.75)}", True),):
    dt = bench(elastic)
    model_fu = 3*F/dt/PEAK                      # 有效算力:前向+反向 = 3x 前向
    print(f"{name:22} {dt*1000:7.1f} ms/步  MFU {100*model_fu:5.1f}%  "
          f"峰值显存 {torch.cuda.max_memory_allocated()/2**30:5.2f} GiB")
    torch.cuda.reset_peak_memory_stats()
