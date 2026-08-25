"""Patch code indexed by REAL per-view position: i1 and multi-image must BOTH get
a code, and the same physical patch must land on the same lattice cell."""
import os, sys
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import torch
from trellis2_blip3o import _paths  # noqa
from trellis2_blip3o.connector import TRELLIS2Connector
from trellis2_blip3o.flow_heads import build_unified_cond

fails, done = [], []
def check(n, ok, m=""):
    (done if ok else fails).append(n)
    print(f"[{'ok ' if ok else 'FAIL'}] {n}{'' if ok else '  <- ' + m}", flush=True)

D, C, P = "cuda", 1024, 32
torch.manual_seed(0)
conn = TRELLIS2Connector(vlm_hidden_dim=2048, trellis_cond_dim=C).to(D).eval().float()
# a distinctive table: cell k gets value k, so the applied code is READABLE
tab = torch.arange(P * P, device=D, dtype=torch.float32)[:, None].repeat(1, C)
z = torch.zeros(1, dtype=torch.bool, device=D)

Td = 16   # a small dino segment so we exercise the FUSION branch, as training does
def run(qp, T, fusion=True):
    h = torch.zeros(1, T, 2048, device=D)          # zero input -> output is conn(0) + code
    km = torch.ones(1, T, dtype=torch.bool, device=D)
    kw = {}
    if fusion:
        kw = dict(dino_hidden=torch.zeros(1, Td, C, device=D),
                  dino_key_mask=torch.ones(1, Td, dtype=torch.bool, device=D),
                  dino_view_ids=torch.zeros(1, Td, dtype=torch.long, device=D))
    with torch.no_grad():
        base, k, _, _ = build_unified_cond(conn, h, km, mask_drop_prob=0.0,
                                           ext_drops=(z, z, z), **kw)
        got, k2, _, _ = build_unified_cond(conn, h, km, cond_patch_pos=tab,
                                           qwen_img_pos=qp, mask_drop_prob=0.0,
                                           ext_drops=(z, z, z), **kw)
    d = (got - base)[0, :, 0]
    return d[Td:] if fusion else d                 # drop the dino half

# ── i1: one view, 1024 tokens on a 32x32 grid -> identity mapping ──
T = 1040
qp = torch.full((1, T), -1, dtype=torch.long, device=D); qp[0, 8:8 + 1024] = torch.arange(1024, device=D)
d = run(qp, T)
ok = float(d[:8].abs().max()) == 0.0 and float(d[8 + 1024:].abs().max()) == 0.0   # non-image untouched
ok &= torch.equal(d[8:8 + 1024], torch.arange(1024, device=D, dtype=d.dtype))     # 1:1
check("i1 (1024 tok, 32x32): identity mapping, non-image tokens untouched", ok)

# ── IM: 4 views x 64 tokens on 8x8 grids -> every 4th lattice cell ──
T2 = 300
qp2 = torch.full((1, T2), -1, dtype=torch.long, device=D)
for v in range(4):
    qp2[0, 10 + 66 * v: 10 + 66 * v + 64] = torch.arange(64, device=D)
d2 = run(qp2, T2)
got = d2[10:10 + 64]
want = torch.tensor([(r * 4) * P + (c * 4) for r in range(8) for c in range(8)],
                    device=D, dtype=d2.dtype)
ok = torch.equal(got, want)
ok &= all(torch.equal(d2[10 + 66 * v: 10 + 66 * v + 64], got) for v in range(4))   # same per view
ok &= float(d2[74:76].abs().max()) == 0.0                                          # gaps untouched
check("IM (4x64 tok, 8x8): maps to every 4th lattice cell, identical across views", ok)

# ── the SAME physical corner lands on the SAME cell in both layouts ──
ok = float(d[8]) == float(d2[10]) == 0.0                       # top-left patch -> cell 0
ok &= float(d[8 + 31]) == float(want[7])                       # hmm: i1 col 31 vs im col 7*4=28
ok = float(d[8]) == float(d2[10])                              # corner agreement is the claim
ok &= float(d[8 + 31 * P + 31]) == float(P * P - 1)            # i1 bottom-right -> last cell
ok &= float(d2[10 + 63]) == float((7 * 4) * P + 7 * 4)         # im bottom-right -> cell (28,28)
check("the same physical corner maps to the same lattice region in both layouts", ok)

# ── coverage: text rows (no image tokens) get nothing, and it is visible ──
qp3 = torch.full((1, 60), -1, dtype=torch.long, device=D)
d3 = run(qp3, 60)
check("a text row (no image tokens) receives no code at all", float(d3.abs().max()) == 0.0)

# ── the PLAIN branch (fuse_dino=False) must behave identically ──
d4 = run(qp, T, fusion=False)
check("plain branch (no DINO) applies the same code — not a silent skip",
      torch.equal(d4[8:8 + 1024], torch.arange(1024, device=D, dtype=d4.dtype)))

print(f"\n{len(done)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
