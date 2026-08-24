"""eval_cond must BE the training path, not a lookalike.

P  cond (no drops) is bitwise equal to what training produces for an undropped row
Q  uncond is the CFG drop: BOTH segments zero-valued, keys INTACT
R  uncond is NOT the ddrop regime (dino keys masked out) — the two differ in
   token count, which is the cheap way to tell them apart
S  the qwen-only arm is measurably different from the fusion arm (the trap that
   cost 0.156 vs 0.244), and eval_cond does not fall into it by default
T  drop_dino / drop_qwen expose the modality regimes and match training's rule
   that qdrop is suppressed by ddrop
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from trellis2_blip3o import _paths  # noqa: F401
from trellis2_blip3o.connector import TRELLIS2Connector
from trellis2_blip3o.eval_cond import cond_uncond
from trellis2_blip3o.flow_heads import build_unified_cond

fails, done = [], []


def check(name, ok, msg=""):
    (done if ok else fails).append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}{'' if ok else '  <- ' + msg}", flush=True)


torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
Tq, Td, VLM, C = 40, 60, 2048, 1024
conn = TRELLIS2Connector(vlm_hidden_dim=VLM, trellis_cond_dim=C).to(DEV).eval().float()
dve = torch.randn(8, C, device=DEV)

rec = {
    "cond_hidden": torch.randn(Tq, VLM, device=DEV),
    "cond_keep_mask": torch.ones(Tq, dtype=torch.bool, device=DEV),
    "dino_hidden": torch.randn(Td, C, device=DEV),
    "dino_keep_mask": torch.ones(Td, dtype=torch.bool, device=DEV),
    "dino_view_ids": torch.zeros(Td, dtype=torch.long, device=DEV),
    "qwen_view_ids": torch.zeros(Tq, dtype=torch.long, device=DEV),
}
h = rec["cond_hidden"].float()[None]
km = rec["cond_keep_mask"][None]
KW = dict(dino_hidden=rec["dino_hidden"].float()[None],
          dino_key_mask=rec["dino_keep_mask"][None],
          dino_view_ids=rec["dino_view_ids"][None],
          qwen_view_ids=rec["qwen_view_ids"][None],
          dino_view_embed=dve, cond_max_length=10240)
z = torch.zeros(1, dtype=torch.bool, device=DEV)
o = torch.ones(1, dtype=torch.bool, device=DEV)


def train_build(drop, ddrop, qdrop):
    with torch.no_grad():
        c, k, _, _ = build_unified_cond(conn, h, km, mask_drop_prob=0.0,
                                        dino_drop_prob=0.0, qwen_drop_prob=0.0,
                                        ext_drops=(drop, ddrop, qdrop), **KW)
    return c[0][k[0]][None]


c, u = cond_uncond(conn, rec, dino_view_embed=dve, device=DEV)

# ── P ──
ok = torch.equal(c, train_build(z, z, z))
ok &= c.shape[1] == Td + Tq            # both segments present
check(f"P cond == training's undropped row, bitwise ({Td}+{Tq} tokens)", ok,
      f"{tuple(c.shape)}")

# ── Q ──
ok = torch.equal(u, train_build(o, z, z))
ok &= u.shape[1] == Td + Tq            # keys INTACT — this is the whole point
# the dino half of uncond must be exactly zero (drop multiplies the segment)
ok &= float(u[0, :Td].abs().max()) == 0.0
# ... and the qwen half is connector(0), NOT zero
ok &= float(u[0, Td:].abs().max()) > 0.0
check("Q uncond == CFG drop: both segments zero-valued, keys kept, qwen=conn(0)", ok)

# ── R ──
u_ddrop = train_build(o, o, z)
ok = u_ddrop.shape[1] == Tq            # dino keys gone entirely
ok &= not torch.equal(u, u_ddrop)
check(f"R the ddrop regime is a DIFFERENT object ({Tq} tokens, not {Td+Tq})", ok,
      f"{tuple(u_ddrop.shape)}")

# ── S ──
qwen_only = train_build(z, o, z)
ok = qwen_only.shape[1] == Tq and c.shape[1] == Td + Tq
# The qwen HALF is identical in both arms — ddrop masks dino keys and does not
# touch cond_q. What the qwen-only arm loses is the entire dino segment, i.e.
# 60% of the tokens here (~1029 of ~2053 in production). That is the whole trap:
# the tensor still looks well-formed, the model still runs, the number is worse.
ok &= torch.equal(c[:, -Tq:], qwen_only)
ok &= c.shape[1] - qwen_only.shape[1] == Td
check(f"S qwen-only keeps the same {Tq} qwen tokens but loses all {Td} dino "
      "tokens; eval_cond defaults to fusion", ok,
      f"fusion {tuple(c.shape)} vs qwen-only {tuple(qwen_only.shape)}")

# ── T ──
cd, _ = cond_uncond(conn, rec, dino_view_embed=dve, device=DEV, drop_dino=True)
cq, _ = cond_uncond(conn, rec, dino_view_embed=dve, device=DEV, drop_qwen=True)
cb, _ = cond_uncond(conn, rec, dino_view_embed=dve, device=DEV,
                    drop_dino=True, drop_qwen=True)
ok = cd.shape[1] == Tq and cq.shape[1] == Td
# training rule: qdrop is suppressed by ddrop, so both-on == dino-dropped-only
ok &= torch.equal(cb, cd)
check("T modality regimes: drop_dino->qwen only, drop_qwen->dino only, "
      "both == drop_dino (qdrop suppressed by ddrop, as in training)", ok,
      f"{tuple(cd.shape)} {tuple(cq.shape)} {tuple(cb.shape)}")

print(f"\n{len(done)} passed, {len(fails)} failed")
if fails:
    print("FAILED:", ", ".join(fails))
sys.exit(1 if fails else 0)
