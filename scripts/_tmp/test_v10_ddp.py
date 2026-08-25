"""v10 L3 — collective safety under RANK SKEW. Run with torchrun --nproc_per_node=2.

The thing this exists to catch cannot be caught on one GPU and does not raise:
compute_unified_geotex_loss makes three collectives per step (tex, geo, ss), and
if any of them becomes conditional on what a rank's batch happened to draw, the
ranks stop agreeing on how many all_reduces to issue and the job HANGS. Same for
the log keys: train_native's per_stage reducer builds a positionally-sorted value
vector from each rank's own key set and all_reduces it, so a rank-dependent key
set either mismatches shapes (hang) or silently reports one metric's number under
another's name.

Rank skew is forced, not hoped for: each rank's scheduler is patched to return a
single row class, so rank 0 can be all-clean (zero rows kept by the SS loss)
while rank 1 is all-solo (zero rows kept by the slat losses). That is exactly the
configuration a guarded collective dies on.

NCCL_TIMEOUT is set low on purpose — a hang should fail this test in a minute,
not hold a node for half an hour.
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import datetime

import torch
import torch.distributed as dist

from trellis2_blip3o import _paths  # noqa: F401
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.models.structured_latent_flow import SLatFlowModel
from trellis2.modules import sparse as sp
import trellis2_blip3o.flow_heads as FH
from trellis2_blip3o.connector import TRELLIS2Connector
from trellis2_blip3o.flow_heads import (CLS_CLEAN, CLS_LAG, CLS_SOLO,
                                        build_flow_loss_fns,
                                        compute_unified_geotex_loss)
from trellis2_blip3o.unified_geotex import UnifiedGeoTexFlow

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
DEV = f"cuda:{os.environ['LOCAL_RANK']}"


def r0(*a):
    if RANK == 0:
        print(*a, flush=True)


# ── tiny three-tower model (real code paths, toy sizes) ──
TINY = dict(resolution=16, model_channels=64, cond_channels=48, out_channels=8,
            num_blocks=2, num_heads=4, mlp_ratio=2, pe_mode="rope",
            dtype="float32", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True)
geo = SLatFlowModel(in_channels=8, **TINY)
tex = SLatFlowModel(in_channels=16, **TINY)
ss = SparseStructureFlowModel(in_channels=8, **{**TINY, "resolution": 4})
uni = UnifiedGeoTexFlow(geo, tex, coupling="union", bidirectional=True,
                        ss_flow=ss, all_trainable=True, cond_seg_embed=True,
                        cond_patch_pos="zero", cond_patch_lattice=8).to(DEV)
conns = [TRELLIS2Connector(vlm_hidden_dim=64, trellis_cond_dim=48).to(DEV).float()
         for _ in range(3)]


class _Cfg:
    logitnorm_mean, logitnorm_std, flow_sigma_min = 1.0, 1.0, 1e-5


loss_fn_ss, loss_fn_slat = build_flow_loss_fns(_Cfg())

B, NV = 2, 40
g = torch.Generator().manual_seed(100 + RANK)
coords = []
for b in range(B):
    c = torch.unique(torch.randint(0, 16, (NV, 3), generator=g), dim=0)
    coords.append(torch.cat([torch.full((c.shape[0], 1), b), c], 1))
coords = torch.cat(coords).int().to(DEV)
N = coords.shape[0]
x0_s = sp.SparseTensor(torch.randn(N, 8, generator=g).to(DEV), coords)
x0_x = sp.SparseTensor(torch.randn(N, 8, generator=g).to(DEV), coords)
cond_h = torch.randn(B, 30, 64, generator=g).to(DEV)
cond_m = torch.ones(B, 30, dtype=torch.bool, device=DEV)
ss_tgt = torch.randn(B, 8, 4, 4, 4, generator=g).to(DEV)

# ── force a single row class per rank ──
FORCE = [CLS_CLEAN, CLS_SOLO, CLS_LAG][RANK % 3]
_orig = FH.sample_timestep_triples


def forced(Bn, device, **kw):
    t_ss, t_s, t_x, cls = _orig(Bn, device, **kw)
    if FORCE == CLS_SOLO:
        return (torch.full_like(t_ss, 0.6), torch.ones_like(t_s), torch.ones_like(t_x),
                torch.full_like(cls, CLS_SOLO))
    if FORCE == CLS_CLEAN:
        return (torch.zeros_like(t_ss), t_s.clamp(0.1, 0.5), t_x.clamp(0.6, 0.9),
                torch.full_like(cls, CLS_CLEAN))
    return (torch.full_like(t_ss, 0.3), torch.full_like(t_s, 0.5),
            torch.full_like(t_x, 0.8), torch.full_like(cls, CLS_LAG))


FH.sample_timestep_triples = forced

fails = []
loss, logs = compute_unified_geotex_loss(
    unified_model=uni, connector_geo=conns[0], connector_tex=conns[1],
    loss_fn_slat=loss_fn_slat, cond_hidden=cond_h, cond_key_mask=cond_m,
    target_shape_slat_512=x0_s, target_tex_slat_512=x0_x,
    ss_flow_present=True, target_ss_latent=ss_tgt, connector_ss=conns[2],
    loss_fn_ss=loss_fn_ss, geo_loss_w=1.0, probe_every=0, mask_drop_prob=0.1)
loss.backward()
torch.cuda.synchronize()
r0(f"[L3] every rank completed a forward+backward under skew "
   f"(rank classes: {['CLEAN','SOLO','LAG']})")

# ── 1. the step completed on every rank ──
done = torch.tensor([1.0], device=DEV)
dist.all_reduce(done)
if int(done.item()) != WORLD:
    fails.append("not every rank reached the barrier")

# ── 2. IDENTICAL log-key LISTS (the reducer is positional, not by name) ──
keys = sorted(logs.keys())
gathered = [None] * WORLD
dist.all_gather_object(gathered, keys)
if any(k != gathered[0] for k in gathered):
    diff = set(gathered[0]) ^ set(gathered[1])
    fails.append(f"log key sets differ across ranks: {sorted(diff)[:6]}")
r0(f"[L3] {len(keys)} log keys, identical on every rank")

# ── 3. the masked-out side really is zero, and the OTHER side is not ──
if FORCE == CLS_CLEAN and logs["ss_flow_loss"] != 0.0:
    fails.append(f"clean rank has nonzero ss_flow_loss {logs['ss_flow_loss']}")
if FORCE == CLS_SOLO and logs["tex_flow_loss"] != 0.0:
    fails.append(f"solo rank has nonzero tex_flow_loss {logs['tex_flow_loss']}")

# ── 4. gradients exist on all three towers on EVERY rank, including the rank
#       whose own loss term was fully masked — that is what keeps DeepSpeed from
#       seeing an unused parameter and what proves the empty selection kept a
#       live grad_fn.
for nm in ("ss_flow", "geo_flow", "tex_flow"):
    m = getattr(uni, nm)
    if not any(p.grad is not None for p in m.parameters()):
        fails.append(f"{nm}: no gradient on rank {RANK} (class {FORCE})")

# ── 5. finite everywhere ──
if not torch.isfinite(loss).all():
    fails.append(f"non-finite loss on rank {RANK}")

bad = torch.tensor([float(len(fails))], device=DEV)
dist.all_reduce(bad)
if RANK == 0:
    print(f"\n[L3] {'PASS' if int(bad.item()) == 0 else 'FAIL'} "
          f"— {int(bad.item())} failure(s) across {WORLD} ranks", flush=True)
if fails:
    print(f"[rank{RANK}] " + "; ".join(fails), flush=True)
dist.barrier()
dist.destroy_process_group()
sys.exit(1 if int(bad.item()) else 0)
