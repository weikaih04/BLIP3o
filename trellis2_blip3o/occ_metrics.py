"""Occupancy metrics for generated structure — the scoreboard v10 needs and v9 could not have.

WHY THIS DID NOT EXIST BEFORE. Every geotex eval to date pinned the voxel coords
to GT (scripts/eval_geotex_g1g2.py, scripts/_tmp/eval_all8.py), so occupancy was
identically correct and `occ_iou` was a comment, not a function. v10 generates its
own coords, which is the whole point of the third tower — and also the moment the
pinned-coords freebie disappears. The eval skill states the direction of that bias
outright: with GT coords "a tie means official is ahead". So G1/G2/T1 numbers from
the v9 era are NOT comparable to v10 numbers on generated coords, and a metric that
scores the structure itself is no longer optional.

THE REFERENCE IS dec(z_gt), NOT A "TRUE" 64^3 OCCUPANCY. The dataset stores the SS
latent, not the grid it was encoded from, so the only 64^3 ground truth available is
the frozen decoder's reconstruction. That is also the RIGHT reference: both arms
decode through the same frozen weights, so the VAE's own loss cancels and what is
left is the flow model's error. The ceiling is 1.0 by construction. (Verified
2026-08-24 on 64 val200 assets: pool(dec(z_gt)) vs the GT slat coords is IoU
median 1.0000 / mean 0.9999 / p10 0.9995, and the voxel counts match exactly — so
at the 32^3 resolution slat actually lives on, this reference IS the GT coords.)

WHY IoU ALONE IS NOT ENOUGH — three companions, each for a failure IoU hides:

  precision / recall  IoU collapses "hallucinated structure" and "missing
                      structure" into one number, and they are different bugs with
                      different fixes. A model that grows a fat blob and one that
                      drops the thin parts can score the same IoU.

  tolerance-1         At 64^3 a thin shell is ~1 voxel thick, so a half-voxel
                      systematic offset can halve IoU while changing nothing a
                      human would notice. tol-1 precision/recall count a voxel as
                      matched if the other set is within one voxel; a model that is
                      merely offset scores near 1 here while a model that is
                      structurally wrong does not. If iou64 is bad and tol-1 is
                      fine, look for a frame/rounding bug, not a quality problem.

  chamfer             Symmetric mean surface distance in voxel units. Bounded and
                      continuous where IoU is brittle: it degrades smoothly with
                      error magnitude instead of falling off a cliff at 1 voxel.

RESOLUTION IS PART OF THE NUMBER. _flow16_ref.py exists because an IoU@16 of 0.327
was being compared against IoU@64 numbers — coarse grids score higher for free.
Everything here is reported at BOTH 64^3 (the SS tower's own output space) and 32^3
(the coords slat actually receives, i.e. the resolution at which an error becomes
downstream damage). Recorded scale on this dataset: v2.3 AR rollout 0.089 IoU@64,
flow head 0.403 IoU@64 (_flow16_ref.py header).

TRIVIAL BASELINES SHIP WITH EVERY NUMBER. `trivial_baselines` scores a filled
bounding box and an equal-volume sphere at the GT centroid against the same GT.
Those are what "know the size and where it is, nothing else" already buys. An IoU
that does not clear them is not a result.

FAILURES COUNT. An empty generation scores 0 and stays in the mean; a generation
whose pooled 32^3 count exceeds `max_slat_tokens` is flagged rather than dropped,
because silently discarding the hard assets is how a mean turns into a press
release (skill geotex-model-eval: "失败率必须进表").
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

# trellis2_blip3o.data.tasks.threed: assets whose shape SLAT exceeds this are
# resampled during training, so a generation above it is one slat cannot consume.
SLAT_TOKEN_CAP = 8192


# ── production-parity conversions ────────────────────────────────────────────
def occ64_from_latent(ssdec, z: torch.Tensor) -> torch.Tensor:
    """(8,16,16,16) or (1,8,16,16,16) SS latent -> (64,64,64) bool occupancy.

    `> 0` on the decoder logits, verbatim from the shipped chain
    (scripts/export_glb_fullchain.py, scripts/trimodal_fullchain_eval.py). The SS
    latent is consumed RAW: the released SS config carries no `normalization` key
    (upstream design), so `load_norm_stats` returns None and every call site
    no-ops the (z-mean)/std. Passing a normalized latent here is a silent error.
    """
    if z.dim() == 4:
        z = z[None]
    with torch.no_grad():
        return (ssdec(z) > 0)[0, 0]


def occ32_from_occ64(occ64: torch.Tensor) -> torch.Tensor:
    """(64,64,64) -> (32,32,32), max-pool 2 with the > 0.5 threshold.

    Bit-identical to the production chain. Do not "simplify" to any_pool or to a
    reshape-max: the float pool + 0.5 compare is what every shipped script does,
    and a coords set that differs from production by even one voxel makes every
    number here describe a model nobody runs.
    """
    return F.max_pool3d(occ64.float()[None, None], 2, 2)[0, 0] > 0.5


def coords_from_occ32(occ32: torch.Tensor) -> torch.Tensor:
    """(32,32,32) bool -> (N,3) int32 coords, the slat support set."""
    return torch.argwhere(occ32).int()


def occ_from_coords(coords: torch.Tensor, res: int = 32) -> torch.Tensor:
    """(N,3) coords -> (res,res,res) bool. Coords out of range are dropped loudly."""
    occ = torch.zeros(res, res, res, dtype=torch.bool, device=coords.device)
    if coords.numel() == 0:
        return occ
    c = coords.long()
    assert int(c.min()) >= 0 and int(c.max()) < res, \
        f"coords out of range for res={res}: [{int(c.min())}, {int(c.max())}]"
    occ[c[:, 0], c[:, 1], c[:, 2]] = True
    return occ


# ── set metrics ──────────────────────────────────────────────────────────────
def _counts(a: torch.Tensor, b: torch.Tensor):
    inter = float((a & b).sum())
    return inter, float(a.sum()), float(b.sum())


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """|A ∩ B| / |A ∪ B|. Both empty -> 1.0 (they agree); one empty -> 0.0."""
    inter, na, nb = _counts(a, b)
    union = na + nb - inter
    if union == 0:
        return 1.0
    return inter / union


def _edt(occ: torch.Tensor) -> np.ndarray:
    """Exact Euclidean distance (in voxels) from every cell to the nearest True.

    scipy's EDT on the COMPLEMENT: distance_transform_edt measures distance to the
    nearest zero, so feeding ~occ gives distance to the nearest occupied cell.
    ~30 ms at 64^3. An all-empty set yields +inf everywhere, which is the correct
    answer and is handled by the callers.
    """
    from scipy import ndimage
    a = occ.detach().cpu().numpy()
    if not a.any():
        return np.full(a.shape, np.inf, dtype=np.float32)
    return ndimage.distance_transform_edt(~a).astype(np.float32)


def surface(occ: torch.Tensor) -> torch.Tensor:
    """Occupied cells with at least one empty 6-neighbour (grid boundary counts as
    empty, so a shape touching the wall still has a surface there)."""
    if not bool(occ.any()):
        return occ
    p = F.pad(occ[None, None].float(), (1, 1, 1, 1, 1, 1), value=0.0)[0, 0].bool()
    nb = (p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1]
          & p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1]
          & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:])
    return occ & ~nb


def chamfer_surface(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric mean nearest-neighbour distance between the two SURFACE sets, in
    voxels. Surfaces rather than volumes so a solid-vs-shell difference does not
    dominate; symmetric so neither over- nor under-generation is free. Either set
    empty -> inf (a real failure, not a small number)."""
    sa, sb = surface(a), surface(b)
    if not bool(sa.any()) or not bool(sb.any()):
        return float("inf")
    da, db = _edt(sb), _edt(sa)          # distance TO b, distance TO a
    ma = sa.detach().cpu().numpy()
    mb = sb.detach().cpu().numpy()
    return 0.5 * (float(da[ma].mean()) + float(db[mb].mean()))


def _tolerant(a: torch.Tensor, b: torch.Tensor, tol: float):
    """(precision, recall) counting a match when the other set is within `tol`."""
    na, nb = float(a.sum()), float(b.sum())
    if na == 0 or nb == 0:
        return (0.0 if na else 1.0), (0.0 if nb else 1.0)
    d_to_b, d_to_a = _edt(b), _edt(a)
    prec = float((d_to_b[a.detach().cpu().numpy()] <= tol).mean())
    rec = float((d_to_a[b.detach().cpu().numpy()] <= tol).mean())
    return prec, rec


def occ_metrics(gen64: torch.Tensor, gt64: torch.Tensor,
                tol: float = 1.0, cap: int = SLAT_TOKEN_CAP) -> Dict[str, float]:
    """Every occupancy number for one asset. gen64/gt64 are (64,64,64) bool.

    Read them together: iou64 is the headline, prec/rec say WHICH way it failed,
    the tol-1 pair says whether the failure is structural or a fraction of a voxel,
    chamfer says how far off in a unit that degrades smoothly, and iou32 says how
    much of it survives the pooling that decides slat's support set.
    """
    assert gen64.shape == gt64.shape == (64, 64, 64), (gen64.shape, gt64.shape)
    assert gen64.dtype == torch.bool and gt64.dtype == torch.bool

    inter, ng, nt = _counts(gen64, gt64)
    g32, t32 = occ32_from_occ64(gen64), occ32_from_occ64(gt64)
    i32, ng32, nt32 = _counts(g32, t32)
    p1, r1 = _tolerant(gen64, gt64, tol)

    return {
        # ── 64^3: the SS tower's own output space ──
        "iou64": iou(gen64, gt64),
        "prec64": (inter / ng) if ng else 0.0,          # of what we made, how much is real
        "rec64": (inter / nt) if nt else 1.0,           # of what is real, how much we made
        f"prec64_t{tol:g}": p1,
        f"rec64_t{tol:g}": r1,
        "chamfer64": chamfer_surface(gen64, gt64),
        "nvox64_gen": ng,
        "nvox64_gt": nt,
        "vox_ratio64": (ng / nt) if nt else float("inf"),
        # ── 32^3: the coords slat actually receives ──
        "iou32": iou(g32, t32),
        "prec32": (i32 / ng32) if ng32 else 0.0,
        "rec32": (i32 / nt32) if nt32 else 1.0,
        "n32_gen": ng32,
        "n32_gt": nt32,
        # ── failure flags: these stay IN the mean, never filtered out ──
        "empty": 1.0 if ng == 0 else 0.0,
        "over_cap": 1.0 if ng32 > cap else 0.0,
    }


def trivial_baselines(gt64: torch.Tensor) -> Dict[str, float]:
    """What "know the size and position, nothing else" already scores.

    box     the GT's filled axis-aligned bounding box
    sphere  a ball at the GT centroid with the GT's voxel count

    A generated IoU below these is worse than guessing a blob — the reason
    _flow16_ref.py was written. Both are computed per asset, not from dataset
    means, so they track the individual object's difficulty.
    """
    out = {"iou_box": 0.0, "iou_sphere": 0.0}
    if not bool(gt64.any()):
        return out
    idx = torch.argwhere(gt64)
    lo, hi = idx.min(0).values, idx.max(0).values
    box = torch.zeros_like(gt64)
    box[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
    out["iou_box"] = iou(box, gt64)

    n = int(gt64.sum())
    ctr = idx.float().mean(0)
    g = torch.stack(torch.meshgrid(*[torch.arange(64, device=gt64.device,
                                                  dtype=torch.float32)] * 3,
                                   indexing="ij"), dim=-1)
    d2 = ((g - ctr) ** 2).sum(-1)
    # radius = the n-th smallest squared distance, i.e. the ball with exactly the
    # GT's volume — the strongest form of the "knows only the size" baseline
    thr = torch.kthvalue(d2.flatten(), min(n, d2.numel())).values
    out["iou_sphere"] = iou(d2 <= thr, gt64)
    return out


def aggregate(rows, keys: Optional[list] = None) -> Dict[str, Dict[str, float]]:
    """mean / median / p10 per key over a list of per-asset dicts.

    Means INCLUDE failures (an empty generation contributes iou 0), which is the
    whole point; `chamfer64` can be inf, so its mean is reported over the finite
    subset with the infinite count kept alongside as `chamfer64_inf_frac`.
    """
    if not rows:
        return {}
    keys = keys or sorted(rows[0].keys())
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], dtype=np.float64)
        fin = v[np.isfinite(v)]
        out[k] = {
            "mean": float(fin.mean()) if fin.size else float("nan"),
            "median": float(np.median(fin)) if fin.size else float("nan"),
            "p10": float(np.percentile(fin, 10)) if fin.size else float("nan"),
            "n": int(v.size),
        }
        if fin.size != v.size:
            out[k]["inf_frac"] = float(1.0 - fin.size / v.size)
    return out
