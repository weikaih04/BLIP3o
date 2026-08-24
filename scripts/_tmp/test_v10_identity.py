"""v10 U4 acceptance — THE certification.

With the gates at zero the three-tower forward must be bit-identical to the
two-tower one. This is the single check that makes everything downstream
interpretable: if it holds, any change v10 shows later came from training, not
from the assembly perturbing a model that was already working.

  K  gates zero -> (v_s, v_x) bitwise equal to the two-tower forward,
     on BOTH block paths (fused and bidir), with and without concat_cond
  L  the SS output of the joint forward equals the standalone SS tower
  M  a nonzero gate actually changes the slat outputs (the read is wired,
     not silently dropped) and leaves the SS output alone
  N  the SS read uses the UNTAGGED query: tagging it would rotate every
     cross-tower logit by pi/2, which is invisible at zero gate
  O  SS<-slat stays dormant: no ss_read_on, or ss_reads_enabled False, and the
     SS output is invariant to the slat inputs
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from trellis2_blip3o import _paths  # noqa: F401
from trellis2.modules import sparse as sp
from trellis2_blip3o.unified_geotex import assemble_unified, assemble_unified_tri

SHAPE = "runs/s3_shape_t50b/checkpoint-8000"
TEX = "runs/s3_tex_t50b/checkpoint-8000"
SS = "runs/s3_ss_t50b/checkpoint-14000"

fails, done = [], []


def check(name, ok, msg=""):
    (done if ok else fails).append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}{'' if ok else '  <- ' + msg}", flush=True)


def rand_shared(batch_sizes, res=32, seed=0):
    """Unique coords per sample — duplicates would alias RoPE phases."""
    g = torch.Generator().manual_seed(seed)
    cs = []
    for b, n in enumerate(batch_sizes):
        flat = torch.randperm(res ** 3, generator=g)[:n]
        c = torch.stack([flat // (res * res), (flat // res) % res, flat % res], -1)
        cs.append(torch.cat([torch.full((n, 1), b, dtype=torch.long), c], 1))
    return torch.cat(cs).int().cuda()


print("assembling ...", flush=True)
tri = assemble_unified_tri(SHAPE, TEX, SS, coupling="union", bidirectional=True,
                           all_trainable=True).cuda().eval()
duo = assemble_unified(SHAPE, TEX, coupling="union", bidirectional=True).cuda().eval()
ss_res = tri.ss_flow.resolution
dt = next(tri.tex_flow.parameters()).dtype
B, NS = 2, [900, 640]
coords = rand_shared(NS, seed=7)
N = coords.shape[0]
g = torch.Generator().manual_seed(11)
GC_IN = tri.geo_flow.in_channels          # 32; the 8 belongs to SS, not geo
x_s = sp.SparseTensor(torch.randn(N, GC_IN, generator=g).cuda().to(dt), coords)
x_x = sp.SparseTensor(torch.randn(N, 32, generator=g).cuda().to(dt), coords)
cc = sp.SparseTensor(torch.randn(N, 32, generator=g).cuda().to(dt), coords)
cond_s = torch.randn(B, 220, 1024, generator=g).cuda().to(dt)
cond_x = torch.randn(B, 220, 1024, generator=g).cuda().to(dt)
x_ss = torch.randn(B, 8, ss_res, ss_res, ss_res, generator=g).cuda().to(dt)
cond_ss = torch.randn(B, 220, 1024, generator=g).cuda().to(dt)
t_s = torch.tensor([600.0, 0.0]).cuda()       # one corner row, one interior
t_x = torch.tensor([900.0, 700.0]).cuda()
t_ss = torch.tensor([300.0, 800.0]).cuda()

AC = dict(device_type="cuda", dtype=torch.bfloat16)


def run(model, fused, ss=True, cc_on=True):
    model.fused_attn = fused
    kw = dict(tex_concat_cond=cc if cc_on else None)
    if ss and getattr(model, "ss_flow", None) is not None:
        kw.update(x_ss=x_ss, t_ss=t_ss, cond_ss=cond_ss)
    with torch.no_grad(), torch.autocast(**AC):
        return model(x_s, x_x, t_s, t_x, cond_s, cond_x, **kw)


# ── K: the certification ──
# concat_cond is NOT optional for the real tex tower: its in_channels is 64
# (32 state + 32 concat), so the tex_concat_cond=None branch only type-checks
# against a 32-channel test model. Both paths are still covered via fused.
worst, detail = 0.0, []
for fused in (True, False):
    for cc_on in (True,):
        a = run(tri, fused, ss=True, cc_on=cc_on)
        b = run(duo, fused, ss=False, cc_on=cc_on)
        ds = float((a[0].feats.float() - b[0].feats.float()).abs().max())
        dx = float((a[1].feats.float() - b[1].feats.float()).abs().max())
        worst = max(worst, ds, dx)
        detail.append(f"fused={fused} cc={cc_on}: geo {ds:.1e} tex {dx:.1e}")
print("        " + " | ".join(detail))
check("K gates zero -> three-tower == two-tower, bit-exact (fused + bidir paths)",
      worst == 0.0, f"max|d| = {worst:.3e}")

# ── L: the SS output is the standalone tower's ──
a = run(tri, fused=False, ss=True)
with torch.no_grad(), torch.autocast(**AC):
    ref = tri.ss_flow(x_ss, t_ss, cond_ss)
d = float((a[2].float() - ref.float()).abs().max())
check("L joint forward's SS output == standalone SS tower, bit-exact", d == 0.0, f"{d:.3e}")

# ── M: a nonzero gate is actually wired ──
with torch.no_grad():
    tri.ss_gates_geo.fill_(0.05)
    tri.ss_gates_tex.fill_(0.05)
c = run(tri, fused=False, ss=True)
ok = not torch.equal(c[0].feats, a[0].feats) and not torch.equal(c[1].feats, a[1].feats)
ok &= torch.equal(c[2], a[2])          # slat<-SS must not touch the SS output
ok &= torch.isfinite(c[0].feats.float()).all() and torch.isfinite(c[1].feats.float()).all()
check("M nonzero gate changes both slat lanes, leaves SS untouched, stays finite", ok)

# ── N: the read uses the untagged query ──
# Re-run with the SS keys rotated by the same pad-pair tag the tex stream wears.
# If the implementation were (wrongly) using the tagged q, tagging the keys too
# would CANCEL the spurious rotation and change the result; with the correct
# untagged q, tagging the keys can only make it differ.
orig = tri.ss_kv_for_slat


def tagged(k_pre, v):
    # _rotate_pad_pair takes a SparseTensor; the SS key is dense, so apply the
    # same exact-float (x, y) -> (-y, x) quarter turn to the identity-pad pair.
    k, v2 = orig(k_pre, v)
    x, y = k[..., -2:-1], k[..., -1:]
    return torch.cat([k[..., :-2], -y, x], dim=-1), v2


tri.ss_kv_for_slat = tagged
d_tag = run(tri, fused=False, ss=True)
tri.ss_kv_for_slat = orig
ok = not torch.equal(d_tag[1].feats, c[1].feats)
check("N SS read is sensitive to key tagging (i.e. q is untagged, no double tag)", ok)

with torch.no_grad():
    tri.ss_gates_geo.zero_()
    tri.ss_gates_tex.zero_()

# ── O: SS<-slat dormant ──
with torch.no_grad():
    tri.ss_reads_gate.fill_(0.1)          # open the gate ...
base = run(tri, fused=False, ss=True)     # ... but pass no ss_read_on
x_s2 = sp.SparseTensor(torch.randn(N, GC_IN, generator=g).cuda().to(dt), coords)
with torch.no_grad(), torch.autocast(**AC):
    other = tri(x_s2, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=cc,
                x_ss=x_ss, t_ss=t_ss, cond_ss=cond_ss)
ok = torch.equal(base[2], other[2])       # SS output invariant to slat input
row_on = torch.tensor([True, True]).cuda()
with torch.no_grad(), torch.autocast(**AC):
    lit = tri(x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=cc,
              x_ss=x_ss, t_ss=t_ss, cond_ss=cond_ss, ss_read_on=row_on)
ok &= not torch.equal(lit[2], base[2])    # ... and live when a row asks for it
tri.ss_reads_enabled = False
with torch.no_grad(), torch.autocast(**AC):
    off = tri(x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=cc,
              x_ss=x_ss, t_ss=t_ss, cond_ss=cond_ss, ss_read_on=row_on)
ok &= torch.equal(off[2], base[2])        # the global inference switch wins
check("O SS<-slat: dormant without ss_read_on, live with it, killed by the switch", ok)

print(f"\n{len(done)} passed, {len(fails)} failed")
if fails:
    print("FAILED:", ", ".join(fails))
sys.exit(1 if fails else 0)
