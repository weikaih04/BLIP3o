"""v10 U2 acceptance — the replicated dense SS lane must BE the SS tower.

G  ss_forward (the block-by-block replication) == ss_flow(...) bit-exact,
   with and without a cond mask, at several timesteps
H  the borrowed K/V is the pre-rope key, and re-roping it lands in the 32^3 frame
I  re-rope preserves RELATIVE position: the phase between an SS cell and a slat
   voxel matches what a native 32^3 embedder would give for (2c+0.5, p)
J  the SS<-slat injection point is live (a nonzero read changes the output) and
   dormant by default (None changes nothing)
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import torch

from trellis2_blip3o import _paths  # noqa: F401
from trellis2.modules.attention import RotaryPositionEmbedder
from trellis2_blip3o.unified_geotex import assemble_unified_tri

SHAPE = "runs/s3_shape_t50b/checkpoint-8000"
TEX = "runs/s3_tex_t50b/checkpoint-8000"
SS = "runs/s3_ss_t50b/checkpoint-14000"

fails, done = [], []


def check(name, ok, msg=""):
    (done if ok else fails).append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}{'' if ok else '  <- ' + msg}", flush=True)


uni = assemble_unified_tri(SHAPE, TEX, SS, coupling="union", bidirectional=True,
                           all_trainable=True).cuda().eval()
ss = uni.ss_flow
res, dt = ss.resolution, next(ss.parameters()).dtype
g = torch.Generator(device="cpu").manual_seed(2024)
B, T = 2, 300
x = torch.randn(B, ss.in_channels, res, res, res, generator=g).cuda().to(dt)
cnd = torch.randn(B, T, ss.cond_channels, generator=g).cuda().to(dt)

# ── G: bit-exact replication, masked and unmasked, across timesteps ──
worst = 0.0
for t_val in ([1000.0, 1000.0], [700.0, 250.0], [1.0, 999.0]):
    for mask in (None, torch.cat([torch.ones(B, 1, 1, T // 2, dtype=torch.bool),
                                  torch.zeros(B, 1, 1, T - T // 2, dtype=torch.bool)],
                                 dim=-1).cuda()):
        t = torch.tensor(t_val).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            a = uni.ss_forward(x, t, cnd, ss_cond_mask=mask).float()
            b = ss(x, t, cnd, cond_mask=mask).float()
        worst = max(worst, float((a - b).abs().max()))
check("G replicated SS lane == ss_flow(...), bit-exact (3 timesteps x masked/unmasked)",
      worst == 0.0, f"max|d| = {worst:.3e}")

# ── H: the export point is PRE-rope, and re-roping lands in the slat frame ──
# ss_prologue itself must run under autocast: TimestepEmbedder emits an fp32
# t_freq into a bf16 mlp, which is exactly why sparse_structure_flow's own
# forward is wrapped. The joint forward will be inside autocast for the same
# reason, so this is the production calling convention, not a test workaround.
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    h, mod, c = uni.ss_prologue(x, torch.tensor([500.0, 500.0]).cuda(), cnd)
    _, k_pre, v = uni._run_ss_block(0, h, mod, c, None, want_kv=True)
blk = ss.blocks[0]
# The reference recomputation must sit under autocast for the same reason the
# block itself does: LayerNorm32 promotes to fp32, and the bf16 to_qkv would
# reject that input outside an autocast region.
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    hn = blk.norm1(h)
    sh1, sc1, _, _, _, _ = (blk.modulation + mod).type(mod.dtype).chunk(6, dim=1)
    hn = hn * (1 + sc1.unsqueeze(1)) + sh1.unsqueeze(1)
    qkv = blk.self_attn.to_qkv(hn).reshape(B, res ** 3, 3, blk.self_attn.num_heads, -1)
    _, k_raw, v_raw = qkv.unbind(dim=2)
    k_rms = blk.self_attn.k_rms_norm(k_raw)
    k_roped = RotaryPositionEmbedder.apply_rotary_embedding(k_rms, ss.rope_phases)
ok = torch.equal(k_pre, k_rms)                       # post-rms
ok &= not torch.equal(k_pre, k_roped)                # and NOT roped
ok &= torch.equal(v, v_raw)                          # v untouched
k_slat, v_slat = uni.ss_kv_for_slat(k_pre, v)
ok &= not torch.equal(k_slat, k_pre) and not torch.equal(k_slat, k_roped)
ok &= torch.equal(v_slat, v)
check("H exported k is post-rms/pre-rope; re-rope gives a third, slat-frame key", ok)

# ── I: relative position is what the slat frame would give ──
hd = uni.geo_flow.blocks[0].self_attn.rope.head_dim
rp = RotaryPositionEmbedder(hd, 3, rope_freq=(1.0, 10000.0))
cell = torch.tensor([[3.0, 5.0, 2.0]])               # an SS cell
vox = torch.tensor([[11.0, 4.0, 27.0]])              # a slat voxel, 0..31
rel_v10 = (rp(vox) * rp(2.0 * cell + 0.5).conj())    # what q_slat . k_ss_reroped sees
rel_native = rp(vox - (2.0 * cell + 0.5))            # a native 32-frame relative phase
ok = torch.allclose(torch.angle(rel_v10), torch.angle(rel_native), atol=1e-5)
# and the SS lane's OWN attention still uses its native 16^3 phases
ok &= torch.equal(ss.rope_phases, rp_16 := RotaryPositionEmbedder(hd, 3)(
    torch.stack(torch.meshgrid(*[torch.arange(res, dtype=torch.float32)] * 3,
                               indexing="ij"), dim=-1).reshape(-1, 3)).to(ss.rope_phases.device))
check("I re-rope reproduces native 32-frame relative phase; SS self-attn keeps 16^3", ok)

# ── J: the SS<-slat injection point is live but dormant ──
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    h0, _, _ = uni._run_ss_block(0, h, mod, c, None)
    h1, _, _ = uni._run_ss_block(0, h, mod, c, None, ss_read=None)
    fake = torch.randn(B, res ** 3, ss.num_heads, hd, generator=g).cuda().to(h.dtype)
    h2, _, _ = uni._run_ss_block(0, h, mod, c, None, ss_read=fake)
ok = torch.equal(h0, h1)                              # None == absent
ok &= not torch.equal(h0, h2)                         # a real read does something
check("J SS<-slat injection point: dormant when None, live when fed", ok)

print(f"\n{len(done)} passed, {len(fails)} failed")
if fails:
    print("FAILED:", ", ".join(fails))
sys.exit(1 if fails else 0)
