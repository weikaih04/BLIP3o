"""Per-patch position code: the two init modes must differ in exactly the way
the design claims, and both must leave everything outside the image span alone."""
import os, sys
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import numpy as np, torch
from trellis2_blip3o import _paths  # noqa
from trellis2_blip3o.connector import TRELLIS2Connector
from trellis2_blip3o.flow_heads import build_unified_cond
from trellis2_blip3o.pos_stamp import DPOS_NPZ, IMG_SPAN_FULL

fails, done = [], []
def check(n, ok, m=""):
    (done if ok else fails).append(n)
    print(f"[{'ok ' if ok else 'FAIL'}] {n}{'' if ok else '  <- ' + m}", flush=True)

D = "cuda"; C = 1024
A, B_ = IMG_SPAN_FULL.start, IMG_SPAN_FULL.stop        # 10, 1034
Tq, Td = B_ + 16, 300                                  # full qwen seq incl. boilerplate
torch.manual_seed(0)
conn = TRELLIS2Connector(vlm_hidden_dim=2048, trellis_cond_dim=C).to(D).eval().float()
h = torch.randn(1, Tq, 2048, device=D)
km = torch.ones(1, Tq, dtype=torch.bool, device=D)
dh = torch.randn(1, Td, C, device=D); dkm = torch.ones(1, Td, dtype=torch.bool, device=D)
dvi = torch.zeros(1, Td, dtype=torch.long, device=D)
z = torch.zeros(1, dtype=torch.bool, device=D)
KW = dict(dino_hidden=dh, dino_key_mask=dkm, dino_view_ids=dvi,
          cond_max_length=10240, mask_drop_prob=0.0, dino_drop_prob=0.0,
          qwen_drop_prob=0.0, ext_drops=(z, z, z))

def run(pp=None):
    with torch.no_grad():
        c, k, _, _ = build_unified_cond(conn, h, km, cond_patch_pos=pp,
                                        cond_patch_span=(A, B_) if pp is not None else None, **KW)
    return c[0][k[0]]

base = run(None)
zero = run(torch.zeros(B_ - A, C, device=D))
sig = torch.from_numpy(np.load(DPOS_NPZ)["pos"].astype("float32")).to(D)
withsig = run(sig)

ok = torch.equal(base, zero)
check("zero-init is bitwise a no-op", ok)

ok = not torch.equal(base, withsig)
# only the image span moved; dino half and the qwen boilerplate are untouched
ok &= torch.equal(base[:Td], withsig[:Td])                       # dino half
ok &= torch.equal(base[Td:Td + A], withsig[Td:Td + A])           # qwen pre-image
ok &= torch.equal(base[Td + B_:], withsig[Td + B_:])             # qwen post-image
ok &= not torch.equal(base[Td + A:Td + B_], withsig[Td + A:Td + B_])
check("dino_sig moves ONLY the qwen image span", ok)

d = (withsig - base)[Td + A:Td + B_]
ok = torch.allclose(d, sig, atol=1e-5)
rel = float(sig.norm(dim=-1).mean() / dh.float().norm(dim=-1).mean())
print(f"        signature norm is {rel:.2f}x the DINO token norm "
      f"(the view-embed incident happened at 0.7x)")
check("the added delta IS the signature", ok)

# a short sequence must be left alone, not mis-indexed
short_h = torch.randn(1, 200, 2048, device=D)
short_km = torch.ones(1, 200, dtype=torch.bool, device=D)
with torch.no_grad():
    c1, k1, _, _ = build_unified_cond(conn, short_h, short_km, cond_patch_pos=sig,
                                      cond_patch_span=(A, B_), **KW)
    c2, k2, _, _ = build_unified_cond(conn, short_h, short_km, **KW)
check("a sequence shorter than the span is left alone (no mis-indexing)",
      torch.equal(c1[0][k1[0]], c2[0][k2[0]]))

print(f"\n{len(done)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
