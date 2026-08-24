"""occ_metrics unit tests — CPU, no models, hand-computed answers.

The point of every case here is that the expected number is derivable by hand, so
a passing run means the metric computes what the docstring claims and not merely
something self-consistent.

Run:  ATTN_BACKEND=sdpa python scripts/_tmp/test_occ_metrics.py
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "sdpa")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from trellis2_blip3o.occ_metrics import (
    SLAT_TOKEN_CAP, aggregate, chamfer_surface, coords_from_occ32, iou,
    occ32_from_occ64, occ_from_coords, occ_metrics, surface, trivial_baselines,
)

fails, done = [], []


def check(name, ok, msg=""):
    (done if ok else fails).append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}{'' if ok else '  <- ' + msg}", flush=True)


def box(lo, hi, res=64):
    o = torch.zeros(res, res, res, dtype=torch.bool)
    o[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return o


# ── 1. identity, disjoint, empty ──
def t_degenerate():
    a = box((10, 10, 10), (20, 20, 20))
    ok = iou(a, a) == 1.0
    ok &= iou(a, box((40, 40, 40), (50, 50, 50))) == 0.0
    e = torch.zeros(64, 64, 64, dtype=torch.bool)
    ok &= iou(e, e) == 1.0            # both empty: they agree
    ok &= iou(a, e) == 0.0 and iou(e, a) == 0.0
    m = occ_metrics(e, a)
    ok &= m["empty"] == 1.0 and m["iou64"] == 0.0 and m["prec64"] == 0.0
    ok &= np.isinf(m["chamfer64"])    # an empty generation is not "distance 0"
    ok &= all(np.isfinite(v) or k.startswith("chamfer") or k.startswith("vox_ratio")
              for k, v in m.items())
    check("1 degenerate cases (identity / disjoint / empty) are exact", ok)


# ── 2. hand-computed IoU on overlapping boxes ──
def t_hand_iou():
    # 10^3 = 1000 each; overlap [15,20)^3 = 5^3 = 125; union = 1875
    a = box((10, 10, 10), (20, 20, 20))
    b = box((15, 15, 15), (25, 25, 25))
    got, want = iou(a, b), 125 / 1875
    ok = abs(got - want) < 1e-12
    m = occ_metrics(a, b)
    ok &= abs(m["prec64"] - 125 / 1000) < 1e-12
    ok &= abs(m["rec64"] - 125 / 1000) < 1e-12
    ok &= m["nvox64_gen"] == 1000 and m["nvox64_gt"] == 1000
    check("2 IoU/prec/rec match hand arithmetic (125/1875, 0.125, 0.125)", ok,
          f"{got} vs {want}")


# ── 3. precision vs recall separate the two failure modes ──
def t_prec_rec_asymmetry():
    gt = box((20, 20, 20), (30, 30, 30))                     # 1000
    fat = box((18, 18, 18), (32, 32, 32))                    # 14^3 = 2744, contains gt
    thin = box((22, 22, 22), (28, 28, 28))                   # 6^3 = 216, inside gt
    mf, mt = occ_metrics(fat, gt), occ_metrics(thin, gt)
    ok = abs(mf["rec64"] - 1.0) < 1e-12 and abs(mf["prec64"] - 1000 / 2744) < 1e-12
    ok &= abs(mt["prec64"] - 1.0) < 1e-12 and abs(mt["rec64"] - 216 / 1000) < 1e-12
    # IoU alone cannot tell them apart at matched magnitude; prec/rec can
    ok &= abs(mf["iou64"] - 1000 / 2744) < 1e-12
    check("3 over-generation shows in prec, under-generation in rec", ok)


# ── 4. THE tolerance case: a 1-voxel shift wrecks IoU but not tol-1 ──
def t_shift_sensitivity():
    # a hollow shell 1 voxel thick — the geometry 64^3 IoU is worst at
    shell = box((20, 20, 20), (40, 40, 40)) & ~box((21, 21, 21), (39, 39, 39))
    shifted = torch.roll(shell, shifts=1, dims=0)
    m = occ_metrics(shifted, shell)
    # ~0.5 is the RIGHT answer, not a bug: shifting a hollow shell along one axis
    # leaves the two faces perpendicular to that axis fully overlapped, so half
    # the surface survives. That a visually-identical shape loses HALF its IoU is
    # exactly the brittleness this test exists to document.
    ok = 0.45 < m["iou64"] < 0.60                # exact IoU collapses ...
    ok &= m["prec64_t1"] > 0.99 and m["rec64_t1"] > 0.99   # ... tol-1 does not
    ok &= m["chamfer64"] <= 1.0                  # and chamfer says "under a voxel"
    # a structurally wrong shape of the same size must NOT pass the tol-1 screen
    elsewhere = torch.roll(shell, shifts=12, dims=0)
    m2 = occ_metrics(elsewhere, shell)
    ok &= m2["prec64_t1"] < 0.5 and m2["chamfer64"] > 5
    # THE DISCRIMINATOR, stated directly rather than as a derived gap: each
    # companion metric must separate the two cases by a wide margin on its own.
    # Measured: tol-1 precision 1.000 vs 0.347 (2.9x), chamfer 0.33 vs 5.04 (15x).
    # Note exact IoU separates them only 0.499 vs 0.163 (3.1x) while ALSO
    # condemning the visually-perfect shape — which is the point.
    ok &= m["prec64_t1"] > 2.5 * m2["prec64_t1"]
    ok &= m2["chamfer64"] > 10 * m["chamfer64"]
    print(f"        shift-1: iou {m['iou64']:.3f} tol1_p {m['prec64_t1']:.3f} "
          f"chamfer {m['chamfer64']:.2f}  |  shift-12: iou {m2['iou64']:.3f} "
          f"tol1_p {m2['prec64_t1']:.3f} chamfer {m2['chamfer64']:.2f}")
    check("4 tol-1 + chamfer separate 'half a voxel off' from 'structurally wrong'", ok)


# ── 5. pooling is bit-identical to the production chain ──
def t_pool_parity():
    torch.manual_seed(0)
    o = torch.rand(64, 64, 64) < 0.05
    mine = occ32_from_occ64(o)
    prod = F.max_pool3d(o.float()[None, None], 2, 2) > 0.5      # the shipped line
    ok = torch.equal(mine, prod[0, 0])
    # and coords round-trip
    c = coords_from_occ32(mine)
    ok &= torch.equal(occ_from_coords(c, 32), mine)
    ok &= c.dtype == torch.int32 and int(c.max()) <= 31
    check("5 occ32 pooling == production line; coords round-trip exact", ok)


# ── 6. surface extraction ──
def t_surface():
    solid = box((20, 20, 20), (30, 30, 30))            # 10^3
    s = surface(solid)
    # interior 8^3 = 512 removed -> 1000 - 512 = 488
    ok = int(s.sum()) == 1000 - 512
    # a shape touching the grid wall still has a surface there
    wall = box((0, 0, 0), (5, 5, 5))
    ok &= int(surface(wall).sum()) == 125 - 27       # interior 3^3
    ok &= chamfer_surface(solid, solid) == 0.0
    check("6 surface = occupied minus 6-connected interior (488 for a 10^3 cube)", ok,
          f"got {int(s.sum())}")


# ── 7. trivial baselines behave ──
def t_baselines():
    solid = box((20, 20, 20), (30, 30, 30))
    b = trivial_baselines(solid)
    ok = abs(b["iou_box"] - 1.0) < 1e-12               # a box IS its bounding box
    ok &= 0.0 < b["iou_sphere"] < 1.0                  # a ball is not a cube
    shell = box((20, 20, 20), (40, 40, 40)) & ~box((21, 21, 21), (39, 39, 39))
    b2 = trivial_baselines(shell)
    # exact: shell = 20^3 - 18^3 = 2168 voxels inside a 20^3 bbox -> 2168/8000
    ok &= abs(b2["iou_box"] - 2168 / 8000) < 1e-12     # a filled box is a bad shell
    ok &= b2["iou_sphere"] == 0.0                      # an equal-volume BALL at the
    #   centroid sits entirely inside the hollow -> zero overlap. The strongest
    #   statement this baseline can make: knowing size+position buys nothing here.
    print(f"        cube: box {b['iou_box']:.3f} sphere {b['iou_sphere']:.3f} | "
          f"shell: box {b2['iou_box']:.3f} sphere {b2['iou_sphere']:.3f}")
    check("7 trivial baselines: box==1 for a box, exactly 2168/8000 for a shell", ok)


# ── 8. failure flags stay in, and aggregate keeps them ──
def t_failures_counted():
    gt = box((20, 20, 20), (30, 30, 30))
    empty = torch.zeros(64, 64, 64, dtype=torch.bool)
    rows = [occ_metrics(gt, gt), occ_metrics(empty, gt)]
    agg = aggregate(rows, keys=["iou64", "empty", "chamfer64"])
    ok = abs(agg["iou64"]["mean"] - 0.5) < 1e-12       # the failure IS in the mean
    ok &= abs(agg["empty"]["mean"] - 0.5) < 1e-12
    ok &= agg["chamfer64"].get("inf_frac", 0.0) == 0.5  # inf tracked, not silently dropped
    # over_cap flags rather than filters
    dense = torch.rand(64, 64, 64) < 0.9
    ok &= occ_metrics(dense, gt)["over_cap"] == 1.0
    ok &= occ_metrics(gt, gt)["over_cap"] == 0.0
    check("8 empty/over-cap counted in the mean, inf tracked as inf_frac", ok)


# ── 9. 32^3 view is more forgiving than 64^3 (the resolution point) ──
def t_resolution():
    shell = box((20, 20, 20), (40, 40, 40)) & ~box((21, 21, 21), (39, 39, 39))
    m = occ_metrics(torch.roll(shell, 1, 0), shell)
    ok = m["iou32"] > m["iou64"]
    print(f"        shift-1 shell: iou64 {m['iou64']:.3f} -> iou32 {m['iou32']:.3f} "
          f"(coarser grids score higher for free — always state resolution)")
    check("9 iou32 > iou64 on the same error (resolution must be stated)", ok)


for fn in (t_degenerate, t_hand_iou, t_prec_rec_asymmetry, t_shift_sensitivity,
           t_pool_parity, t_surface, t_baselines, t_failures_counted, t_resolution):
    try:
        fn()
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(fn.__name__, False, repr(e))

print(f"\n{len(done)} passed, {len(fails)} failed")
if fails:
    print("FAILED:", ", ".join(fails))
sys.exit(1 if fails else 0)
