"""The position code must survive tokens-per-view changing to anything: bigger,
smaller, non-square, and not a perfect square."""
import os, sys, json
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import torch
from trellis2_blip3o import _paths  # noqa
from trellis2_blip3o.flow_heads import _add_patch_pos

fails, done = [], []
def check(n, ok, m=""):
    (done if ok else fails).append(n)
    print(f"[{'ok ' if ok else 'FAIL'}] {n}{'' if ok else '  <- ' + m}", flush=True)

D, C, P = "cuda", 8, 32
# a table whose value encodes its own (row, col) so the sample is readable
tab = torch.zeros(P * P, C, device=D)
tab[:, 0] = torch.arange(P * P, device=D) // P          # row
tab[:, 1] = torch.arange(P * P, device=D) % P           # col

def rc_for(gh, gw):
    r = (torch.arange(gh * gw, device=D) // gw).float()
    c = (torch.arange(gh * gw, device=D) % gw).float()
    return torch.stack([(r + 0.5) / gh, (c + 0.5) / gw], -1)[None]

def sample(gh, gw):
    rc = rc_for(gh, gw)
    out = _add_patch_pos(torch.zeros(1, gh * gw, C, device=D), tab, rc)[0]
    return out[:, 0], out[:, 1]                          # sampled row, col

# ── 32x32: exact, interpolation degenerates to identity ──
r, c = sample(32, 32)
ok = torch.equal(r, (torch.arange(1024, device=D) // 32).float())
ok &= torch.equal(c, (torch.arange(1024, device=D) % 32).float())
check("32x32 (== table): identity, signature used verbatim", ok)

# ── 16x16: each token gets the MEAN of the 2x2 cells it covers ──
r, c = sample(16, 16)
want_r = torch.arange(16, device=D).repeat_interleave(16).float() * 2 + 0.5
ok = torch.allclose(r, want_r, atol=1e-4)
check("16x16 (coarser): mean of the 2x2 covered cells, not a corner", ok,
      f"{r[:3].tolist()} vs {want_r[:3].tolist()}")

# ── 64x64: FINER than the table — must not collide, must vary monotonically ──
r, c = sample(64, 64)
rr = r.view(64, 64)[:, 0]
ok = bool((rr[1:] >= rr[:-1]).all()) and float(rr.max()) <= P - 1 and float(rr.min()) >= 0
ok &= int(torch.unique(r).numel()) > 32          # more distinct values than a nearest-cell map
check("64x64 (finer than the table): interpolates instead of colliding", ok,
      f"{int(torch.unique(r).numel())} distinct row values")

# ── non-square, and a token count that is not a perfect square ──
for gh, gw in ((23, 58), (16, 64), (1, 1334)):
    r, c = sample(gh, gw)
    n = gh * gw
    ok = r.shape[0] == n and torch.isfinite(r).all() and torch.isfinite(c).all()
    ok &= float(r.max()) <= P - 1 and float(c.max()) <= P - 1
    ok &= float(r.min()) >= 0 and float(c.min()) >= 0
    # rows must be non-decreasing down the grid, cols must sweep within a row
    ok &= bool((r.view(gh, gw)[1:, 0] >= r.view(gh, gw)[:-1, 0]).all()) if gh > 1 else True
    ok &= bool((c.view(gh, gw)[0, 1:] >= c.view(gh, gw)[0, :-1]).all()) if gw > 1 else True
    check(f"{gh}x{gw} = {n} tokens (non-square / not a perfect square)", ok)

print(f"\n{len(done)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
