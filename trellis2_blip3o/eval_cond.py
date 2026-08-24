"""The ONE way to build conditioning outside the training loop.

Every hand-rolled cond in this repo has been wrong in a way that does not crash.
Two examples, both measured:

  * `connector(cond_hidden)` alone silently runs the QWEN-ONLY arm. The s3 runs
    are fuse_dino=True, so the real cond is cat([dino_segment, qwen_segment])
    with a per-view embedding on the dino side; dropping the dino half costs
    ~1029 of ~2053 tokens and reads as a worse model — 0.156 vs 0.244 occ_iou@64
    on the same checkpoint.

  * The CFG uncond is NOT "zero everything". In build_unified_cond a `drop` row
    gets `cond_q = connector(cond_hidden * 0)` AND `dino_seg = dino_seg * 0`,
    i.e. both segments zero-VALUED with their keys still present. Masking the
    dino keys OUT instead is `ddrop`, a separate modality dropout that co-occurs
    with a CFG drop on ~1% of training rows. Using it as the uncond hands the
    sampler a guidance direction the model was hardly ever trained on.

So this module does not reimplement anything: it calls build_unified_cond — the
builder the training loop itself uses — with the drops FORCED through ext_drops.
One code path means a conditioning bug can no longer live in the gap between
training and eval, and forcing rather than sampling the drops makes eval
deterministic and RNG-neutral.

    from trellis2_blip3o.eval_cond import encode, cond_uncond
    rec = encode("i1", renders_dir=r["renders_dir"], view=v)
    c, u = cond_uncond(connector, rec, dino_view_embed=dve)
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from .flow_heads import build_unified_cond

_ENC = None


def encoder():
    """The resident TrainCondEncoder — the SAME encoder training uses, so eval and
    training cannot disagree about tokenisation, view layout or boilerplate."""
    global _ENC
    if _ENC is None:
        from .live_cond_batch import TrainCondEncoder
        _ENC = TrainCondEncoder()
    return _ENC


def encode(mode: str, *, renders_dir: str = None, view: int = 0,
           views: Sequence[int] = None, sha: str = None, caption: str = None,
           cap_idx: int = 0) -> Dict[str, torch.Tensor]:
    """One conditioning record. mode: "i1" | "im" | "t"."""
    from .live_cond_batch import prep_i1, prep_im, prep_t
    e = encoder()
    if mode == "i1":
        return e.encode([prep_i1(renders_dir, view)])[0]
    if mode == "im":
        return e.encode([prep_im(renders_dir, list(views))])[0]
    if mode == "t":
        return e.encode([prep_t(sha, cap_idx, caption)])[0]
    raise ValueError(f"unknown mode {mode!r}")


def good_view(sha: str) -> int:
    """The view index every eval in this repo uses (eval_fusion_v22.good_view_b
    returns the same choice as a filename)."""
    return 5 + (int(sha[:8], 16) + 3) % 7


def _b(t, device="cuda"):
    return None if t is None else t.to(device)[None]


def cond_uncond(connector, rec: Dict[str, torch.Tensor],
                dino_view_embed: Optional[torch.Tensor] = None,
                cond_max_length: int = 10240,
                device: str = "cuda",
                drop_dino: bool = False, drop_qwen: bool = False):
    """(cond, uncond), each (1, T, C), keep-masked — ready for a sampler.

    cond    no drops at all.
    uncond  the CFG drop ONLY: both segments zero-valued, keys intact. This is
            what `mask_drop_prob` produces in training and therefore the
            unconditional the model actually learned.

    drop_dino / drop_qwen expose the MODALITY dropout regimes (ddrop / qdrop) for
    ablations — "how much does this model rely on DINO?" — and are not part of
    CFG. drop_dino masks the dino keys out entirely; drop_qwen masks the qwen
    keys out (and is suppressed by drop_dino exactly as in training, since
    dropping both would leave no conditioning at all, which is the CFG drop's
    job).
    """
    h = rec["cond_hidden"].float().to(device)[None]
    km = rec["cond_keep_mask"].to(device).bool()[None]
    dh = _b(rec.get("dino_hidden"), device)
    if dh is not None:
        dh = dh.float()
    kw = dict(dino_hidden=dh,
              dino_key_mask=_b(rec.get("dino_keep_mask"), device),
              dino_view_ids=_b(rec.get("dino_view_ids"), device),
              qwen_view_ids=_b(rec.get("qwen_view_ids"), device),
              dino_view_embed=dino_view_embed,
              cond_max_length=cond_max_length)
    z = torch.zeros(1, dtype=torch.bool, device=device)
    o = torch.ones(1, dtype=torch.bool, device=device)

    def build(drop, ddrop, qdrop):
        c, k, _, _ = build_unified_cond(
            connector, h, km, mask_drop_prob=0.0, dino_drop_prob=0.0,
            qwen_drop_prob=0.0, ext_drops=(drop, ddrop, qdrop), **kw)
        return c[0][k[0]][None]

    dd = o if drop_dino else z
    qd = o if drop_qwen else z
    with torch.no_grad():
        cond = build(z, dd, qd)
        # The uncond stays in the SAME modality regime as the cond: guidance is
        # cond - uncond, so a uncond that carries keys the cond does not have
        # would put the difference partly in tokens the conditional never saw.
        uncond = build(o, dd, qd)        # CFG: values zeroed, keys as in cond
    return cond, uncond


def cond_uncond_from_tensors(connector, qwen, qwen_mask, dino=None, dino_mask=None,
                             dino_view_embed=None, dino_view_ids=None,
                             qwen_view_ids=None, **kw):
    """Tensor-level entry point, for callers holding a cond CACHE rather than a
    live record (the npz stores the same four/five arrays under other names).

    Exists so cache-based evals stop reimplementing the concat/view-embed/uncond
    logic. dino=None takes the plain (text) branch, which is why text
    conditioning no longer needs a hand-rolled path either.
    """
    rec = {"cond_hidden": qwen, "cond_keep_mask": qwen_mask}
    if dino is not None:
        rec["dino_hidden"] = dino
        rec["dino_keep_mask"] = (dino_mask if dino_mask is not None else
                                 torch.ones(dino.shape[0], dtype=torch.bool,
                                            device=dino.device))
        rec["dino_view_ids"] = (dino_view_ids if dino_view_ids is not None else
                                torch.zeros(dino.shape[0], dtype=torch.long,
                                            device=dino.device))
    if qwen_view_ids is not None:
        rec["qwen_view_ids"] = qwen_view_ids
    return cond_uncond(connector, rec, dino_view_embed=dino_view_embed,
                       device=str(qwen.device), **kw)
