"""v10 U1 acceptance — three-tower assembly on REAL s3_t50 weights.

What must hold before anything else is worth writing:
  A  the three towers load strictly and the parameter count is what we think
  B  every tower is trainable (all_trainable), and geo is NOT eval-pinned
  C  the gates exist, are exactly zero, and are distinct objects
  D  the re-rope table is complex64, unit-modulus, and in the slat frame
  E  the SS tower still reproduces the standalone specialist bit-for-bit
     (assembly must not perturb it — it has not been wired into the forward yet,
      so this is the "did loading break anything" check)
  F  rope_phases survived the bf16 cast with its imaginary part intact
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import torch

from trellis2_blip3o import _paths  # noqa: F401
from trellis2_blip3o.unified_geotex import assemble_unified_tri, load_tri_connectors

SHAPE = "runs/s3_shape_t50b/checkpoint-8000"
TEX = "runs/s3_tex_t50b/checkpoint-8000"
SS = "runs/s3_ss_t50b/checkpoint-14000"

fails, done = [], []


def check(name, ok, msg=""):
    (done if ok else fails).append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}{'' if ok else '  <- ' + msg}", flush=True)


print("assembling three towers from s3_t50 ...", flush=True)
uni = assemble_unified_tri(SHAPE, TEX, SS, coupling="union", bidirectional=True,
                           all_trainable=True)
cg, cx, cs = load_tri_connectors(SHAPE, TEX, SS)
print("assembled.", flush=True)

# ── A: parameter accounting ──
tw = uni.trainable_towers()
tot = sum(p.numel() for p in uni.parameters())
trn = sum(p.numel() for p in uni.parameters() if p.requires_grad)
print(f"        towers: " + "  ".join(f"{k} {v/1e9:.3f}B" if v > 1e8 else f"{k} {v/1e6:.2f}M"
                                      for k, v in tw.items()))
print(f"        total {tot/1e9:.3f}B  trainable {trn/1e9:.3f}B  "
      f"connectors {(sum(p.numel() for c in (cg,cx,cs) for p in c.parameters()))/1e6:.1f}M")
ok = 3.5e9 < tot < 4.3e9 and abs(trn - tot) < 1e6
check("A three towers ~3.9B, essentially all of it trainable", ok, f"{tot/1e9:.3f}B")

# ── B: nothing frozen, geo not eval-pinned ──
uni.train()
ok = all(m.training for m in (uni.ss_flow, uni.geo_flow, uni.tex_flow))
ok &= all(p.requires_grad for p in uni.geo_flow.parameters())
ok &= all(p.requires_grad for p in uni.ss_flow.parameters())
check("B all_trainable: three towers in train mode, geo not eval-pinned", ok)

# ── C: gates ──
g1, g2, g3 = uni.ss_gates_geo, uni.ss_gates_tex, uni.ss_reads_gate
nb, nh = len(uni.tex_flow.blocks), uni.tex_flow.blocks[0].self_attn.num_heads
ok = tuple(g1.shape) == (nb, nh) and tuple(g2.shape) == (nb, nh)
ok &= tuple(g3.shape) == (nb, 2, nh)
ok &= float(g1.abs().max()) == 0.0 and float(g2.abs().max()) == 0.0 and float(g3.abs().max()) == 0.0
ok &= g1 is not g2 and g1.data_ptr() != g2.data_ptr()      # distinct objects
ok &= g1.requires_grad and g2.requires_grad and g3.requires_grad
check(f"C gates ({nb},{nh}) x2 + ({nb},2,{nh}), exactly zero, distinct, trainable", ok)

# ── D: re-rope table ──
ph = uni.ss_phases_slat
res = uni.ss_flow.resolution
hd = uni.geo_flow.blocks[0].self_attn.rope.head_dim
ok = ph.dtype == torch.complex64 and tuple(ph.shape) == (res ** 3, hd // 2)
ok &= float((ph.abs() - 1).abs().max()) < 1e-5              # unit modulus
# the buffer is non-persistent: absent from state_dict
ok &= "ss_phases_slat" not in uni.state_dict()
# and it really is the 2c+0.5 frame, not 2c
from trellis2.modules.attention import RotaryPositionEmbedder
rp = RotaryPositionEmbedder(hd, 3, rope_freq=(1.0, 10000.0))
c = torch.stack(torch.meshgrid(*[torch.arange(res, dtype=torch.float32)] * 3,
                               indexing="ij"), dim=-1).reshape(-1, 3)
ok &= torch.equal(ph.cpu(), rp(2.0 * c + 0.5))
ok &= not torch.allclose(ph.cpu(), rp(2.0 * c))
check("D re-rope table complex64, unit-modulus, 2c+0.5 frame, non-persistent", ok)

# ── E: the SS tower still IS the specialist ──
ss_ref = None
try:
    from blip3o.model.multimodal_decoder.builder import build_ss_flow
    from trellis2_blip3o.unified_geotex import _load_prefixed_state

    class _C:
        trellis_ss_flow_ckpt = None
    ss_ref = build_ss_flow(_C())
    sd = _load_prefixed_state(SS, "ss_flow.")
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    ss_ref.load_state_dict(sd, strict=True)
    ss_ref = ss_ref.cuda().eval()
    uni_ss = uni.ss_flow.cuda().eval()
    g = torch.Generator(device="cpu").manual_seed(99)
    # build_ss_flow runs _uniform_bf16, which hard-casts EVERY layer incl. the
    # input_layer (that hard cast buys 2.1x training throughput and is why the
    # project must stay on DeepSpeed's BF16_Optimizer). So the inputs must be
    # bf16 and the call must sit under autocast, exactly as the training path does
    # — feeding fp32 here is a test artefact, not a model property.
    dt = next(uni_ss.parameters()).dtype
    x = torch.randn(2, uni_ss.in_channels, res, res, res, generator=g).cuda().to(dt)
    cnd = torch.randn(2, 300, uni_ss.cond_channels, generator=g).cuda().to(dt)
    t = torch.tensor([700.0, 250.0]).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = uni_ss(x, t, cnd).float()
        b = ss_ref(x, t, cnd).float()
    d = float((a - b).abs().max())
    check("E assembled SS lane == standalone specialist, bit-exact", d == 0.0, f"max|d|={d:.3e}")
except Exception as e:
    check("E assembled SS lane == standalone specialist", False, repr(e))

# ── F: complex rope survived ──
ok = uni.ss_flow.rope_phases.is_complex()
ok &= float(uni.ss_flow.rope_phases.imag.abs().max()) > 0
check("F ss_flow.rope_phases kept its imaginary part through the bf16 cast", ok)

print(f"\n{len(done)} passed, {len(fails)} failed")
if fails:
    print("FAILED:", ", ".join(fails))
sys.exit(1 if fails else 0)
