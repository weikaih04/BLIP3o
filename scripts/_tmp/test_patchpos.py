"""qwen_img_pos now carries the CANONICAL lattice index, computed in the encoder.
Verify on REAL records that i1 and multi-image land on the same physical lattice,
and that the consumer is a faithful gather."""
import os, sys, json
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import torch
from trellis2_blip3o import _paths  # noqa
from trellis2_blip3o.connector import TRELLIS2Connector
from trellis2_blip3o.eval_cond import encoder, good_view
from trellis2_blip3o.flow_heads import _add_patch_pos, build_unified_cond
from trellis2_blip3o.live_cond_batch import PATCH_LATTICE, prep_i1, prep_im

fails, done = [], []
def check(n, ok, m=""):
    (done if ok else fails).append(n)
    print(f"[{'ok ' if ok else 'FAIL'}] {n}{'' if ok else '  <- ' + m}", flush=True)

P = PATCH_LATTICE
r = json.loads(open("manifests/splits/val200.jsonl").readline())
e = encoder()
r1 = e.encode([prep_i1(r["renders_dir"], good_view(r["sha256"]))])[0]
rm = e.encode([prep_im(r["renders_dir"], [0, 4, 8, 12])])[0]

for tag, rec, side in (("i1", r1, 32), ("im", rm, 16)):
    qp = rec["qwen_img_pos"]; m = qp >= 0
    v = qp[m]
    ok = int(v.min()) >= 0 and int(v.max()) < P * P
    rr, cc = v // P, v % P
    # a side x side view must land on exactly side^2 DISTINCT cells, evenly spaced
    step = P // side
    ok &= set(rr.unique().tolist()) == set(range(0, P, step))
    ok &= set(cc.unique().tolist()) == set(range(0, P, step))
    ok &= int(v.unique().numel()) == side * side
    print(f"        {tag}: {int(m.sum())} image tokens -> {int(v.unique().numel())} distinct "
          f"cells, rows {sorted(rr.unique().tolist())[:4]}... step {step}")
    check(f"{tag} maps onto the {P}x{P} lattice with stride {step}", ok)

# the two layouts must AGREE on shared cells: im's cells are a subset of i1's
v1 = set(r1["qwen_img_pos"][r1["qwen_img_pos"] >= 0].unique().tolist())
vm = set(rm["qwen_img_pos"][rm["qwen_img_pos"] >= 0].unique().tolist())
check("multi-image cells are a strict subset of single-image cells "
      f"({len(vm)} of {len(v1)})", vm < v1)

# corners agree exactly
check("both layouts put the top-left patch on cell 0 and reach the far corner",
      0 in v1 and 0 in vm and (P * P - 1) in v1 and ((P - 2) * P + (P - 2)) in vm)

# the consumer is a faithful gather
D = "cuda"; C = 1024
tab = torch.arange(P * P, device=D, dtype=torch.float32)[:, None].repeat(1, C)
qp = r1["qwen_img_pos"].to(D)[None]
cq = torch.zeros(1, qp.shape[1], C, device=D)
out = _add_patch_pos(cq, tab, qp)[0, :, 0]
mm = (qp[0] >= 0)
check("gather is faithful: every image token got exactly its lattice index",
      torch.equal(out[mm], qp[0][mm].float()) and float(out[~mm].abs().max()) == 0.0)

print(f"\n{len(done)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
