"""Latency: joint (25 interleaved steps) vs cascade (25 geo + 25 tex, K/V cached).
FLOPs are ~equal; what differs is SEQUENTIAL DEPTH — the thing that matters when
the model is kernel-launch bound (measured: 33% GPU idle)."""
import os, sys, time, torch
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import trellis2_blip3o._paths  # noqa
from trellis2.modules import sparse as sp
from trellis2_blip3o.unified_geotex import load_unified_inference
from trellis2_blip3o.geotex_sampler import GeoTexSampler
DEV = "cuda:0"; torch.manual_seed(0)
CK = "runs/geotex_s1_v1/checkpoint-3000"
uni = load_unified_inference(CK, coupling="union", bidirectional=True).to(DEV).eval()
uni.fused_attn = True
smp = GeoTexSampler(uni)   # released per-stream params
N = int(os.environ.get("NVOX", "2176"))          # measured mean asset size
f = torch.randperm(32**3)[:N]
c = torch.stack([f//1024, (f//32)%32, f%32], 1).int()
coords = torch.cat([torch.zeros(N,1,dtype=torch.int32), c], 1).to(DEV)
c_s = torch.randn(1, 1200, 1024, device=DEV)
u_s = torch.randn(1, 1200, 1024, device=DEV)
c_x = torch.randn(1, 1200, 1024, device=DEV); u_x = torch.randn(1, 1200, 1024, device=DEV)
def timeit(fn, n=3):
    fn(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n
t_joint = timeit(lambda: smp.sample_joint(coords, c_s, u_s, c_x, u_x, alpha=4.0, seed=0))
t_casc  = timeit(lambda: smp.sample_joint(coords, c_s, u_s, c_x, u_x, alpha=float("inf"), seed=0))
print(f"voxels={N}  joint(a=4) {t_joint:6.2f}s   cascade(a=inf) {t_casc:6.2f}s   "
      f"speedup {t_casc/t_joint:.2f}x")
