"""Thin wrapper around TRELLIS.2's flow matching math.

We do NOT re-implement the diffuse / velocity / t-sampling formulas — we import
them from `trellis2.trainers.flow_matching.flow_matching.FlowMatchingTrainer`.
If upstream TRELLIS.2 changes the sigma schedule or t schedule, those updates
flow through automatically.

`FlowMatchingTrainer` itself can't be instantiated standalone (its
`__init__` chains into `BasicTrainer.__init__`, which requires
models/optimizer/dataset). Instead we bind the relevant unbound methods to
a lightweight holder object that exposes only the attributes those methods
read (`t_schedule`, `sigma_min`).

V3 condition-swap distillation (docs/V3_DISTILL_DESIGN.md) also lives here:
`__call__(..., teacher_cond=...)` runs a SECOND no_grad pass of the SAME flow on
the SAME (x_t, t) with the teacher (DINOv3) cond and adds
  kd_v_weight · ‖v_student − v_teacher‖²          (output-level KD, Scaling-Down)
  kd_f_weight · Σ_l ‖block_l(student) − block_l(teacher)‖²   (feature-level KD, PEA)
Teacher pass runs FIRST (activations freed before the student pass) → peak memory
≈ unchanged. KD terms are computed in fp32.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import _paths  # noqa: F401  — sets sys.path so trellis2 is importable
from trellis2.trainers.flow_matching.flow_matching import FlowMatchingTrainer as _FMT


def kd_block_indices(n_blocks: int, spec: str = "auto5") -> List[int]:
    """Which block outputs to align for feature-level KD.

    "autoK" → K evenly-spaced INNER blocks (skip the very first/last — their
    outputs are dominated by input/output projections, not cond integration).
    "3,9,15" → explicit indices."""
    spec = (spec or "auto5").strip()
    if spec.startswith("auto"):
        k = int(spec[4:] or 5)
        k = max(1, min(k, n_blocks))
        # evenly spaced in (0, n_blocks-1) exclusive-ish
        return sorted({round((i + 1) * (n_blocks - 1) / (k + 1)) for i in range(k)})
    return sorted({int(s) for s in spec.split(",") if s.strip() != ""})


class _BlockTap:
    """Forward hooks on `blocks[idx]` capturing outputs (SparseTensor → .feats).

    detach_fp32=True for the teacher pass (targets, freed of graph);
    False for the student pass (must stay in the autograd graph)."""

    def __init__(self, blocks, idxs: Sequence[int], detach_fp32: bool):
        self.captured: Dict[int, torch.Tensor] = {}
        self._handles = []
        for i in idxs:
            self._handles.append(blocks[i].register_forward_hook(self._mk(i, detach_fp32)))

    def _mk(self, idx: int, detach_fp32: bool):
        def hook(_mod, _inp, out):
            t = out.feats if hasattr(out, "feats") else out
            self.captured[idx] = t.detach().float() if detach_fp32 else t
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class _MathHolder:
    """Empty object that holds the two attrs upstream methods read off `self`."""
    pass


class TRELLIS2FlowMatchingLoss:
    """Per-stage flow-matching loss using TRELLIS.2's upstream math.

    Usage:
        # SS Flow (TRELLIS.2 trains it with logitNormal)
        loss_fn_ss = TRELLIS2FlowMatchingLoss(t_schedule="logitNormal", t_mean=1.0, t_std=1.0)

        # SLAT stages (TRELLIS.2 trains them with uniform)
        loss_fn_slat = TRELLIS2FlowMatchingLoss(t_schedule="uniform")

    Each instance binds upstream `sample_t` / `diffuse` / `get_v` to its own
    `_MathHolder` configured with the stage's correct t_schedule.
    """

    def __init__(
        self,
        t_schedule: str = "logitNormal",
        t_mean: float = 1.0,
        t_std: float = 1.0,
        sigma_min: float = 1e-5,
    ):
        h = _MathHolder()
        # TRELLIS.2 supports two t_schedule names: 'logitNormal' (uses mean+std)
        # and 'uniform' (ignores mean/std). See FlowMatchingTrainer.sample_t.
        if t_schedule == "logitNormal":
            h.t_schedule = {"name": "logitNormal", "args": {"mean": t_mean, "std": t_std}}
        elif t_schedule == "uniform":
            h.t_schedule = {"name": "uniform", "args": {}}
        else:
            raise ValueError(
                f"Unknown t_schedule '{t_schedule}'. TRELLIS.2 only supports "
                f"'logitNormal' or 'uniform'."
            )
        h.sigma_min = sigma_min
        self._holder = h
        # Bind the unbound upstream methods onto our holder.
        self.diffuse = _FMT.diffuse.__get__(h, _MathHolder)
        self.get_v = _FMT.get_v.__get__(h, _MathHolder)
        self.sample_t = _FMT.sample_t.__get__(h, _MathHolder)

    def __call__(
        self,
        flow_model,
        x_0,
        cond: torch.Tensor,
        cond_mask: torch.Tensor = None,
        teacher_cond: Optional[torch.Tensor] = None,
        kd_v_weight: float = 1.0,
        kd_f_weight: float = 0.5,
        kd_f_blocks: str = "auto5",
        **flow_kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute one-step flow MSE loss. Dispatches dense (SS Flow) vs sparse (SLAT).

        Args:
            flow_model: SparseStructureFlowModel (dense) OR SLatFlowModel (sparse).
            x_0: dense (B, C, D, H, W) torch.Tensor for SS, or sp.SparseTensor for SLAT.
            cond: (B, N_tokens, cond_dim) cross-attn conditioning (dense, padded).
            cond_mask: bool, either shape (B, N_tokens) [2D] or (B, 1, 1, N_tokens) [sdpa-ready].
                True = real token, False = padding.

                - For dense SS Flow: passed to model as `cond_mask=...` (sdpa attn_mask).
                - For sparse SLAT: converted to per-sample `cond_list`; SLatFlowModel auto-
                  builds VarLenTensor from list, flash_attn_varlen handles variable length
                  natively (no need to pass mask to the model).
            teacher_cond: optional (B, N_teacher, cond_dim) — V3 distillation. Runs a
                no_grad pass of the SAME flow on the SAME (x_t, t) with this cond and adds
                kd_v_weight·‖v_s − v_t‖² + kd_f_weight·Σ_l‖block_l^S − block_l^T‖².
                Teacher cond is assumed unpadded (DINOv3 features → no mask).

        Returns:
            loss: scalar tensor (flow MSE [+ KD terms when teacher_cond given]).
            logs: {flow_mse, t_mean[, kd_v, kd_f]}.
        """
        is_sparse = hasattr(x_0, "feats")
        if is_sparse:
            noise = x_0.replace(torch.randn_like(x_0.feats))
            x_0_dtype = x_0.feats.dtype
        else:
            noise = torch.randn_like(x_0)
            x_0_dtype = x_0.dtype
        # Cast t to x_0's dtype (matches TRELLIS.2 upstream training:
        #   `t = sample_t(...).to(x_0.device, x_0.dtype)`).
        # Critical: do NOT force float() here — that would upcast x_t to float32
        # via `x_t = x_0 * (1-t) + noise * t`, breaking dtype match with bf16
        # model weights when autocast isn't active.
        t = self.sample_t(x_0.shape[0]).to(x_0.device).to(x_0_dtype)
        x_t = self.diffuse(x_0, t, noise=noise)

        x_t_dtype = x_t.feats.dtype if is_sparse else x_t.dtype
        t_in = (t * 1000.0).to(x_t_dtype)

        distill = teacher_cond is not None
        kd_idx: List[int] = (
            kd_block_indices(len(flow_model.blocks), kd_f_blocks)
            if (distill and kd_f_weight > 0) else []
        )

        # TRELLIS.2's TimestepEmbedder builds an fp32 t_freq internally, which the
        # bf16 t-embedder MLP can only consume under autocast. HF Trainer + DeepSpeed
        # bf16 does NOT enable torch.autocast (DeepSpeed manages precision itself), so
        # we enter it explicitly here to avoid a Float/BFloat16 matmul mismatch.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # ── V3 distillation: TEACHER pass FIRST (no_grad → its activations are freed
            # before the student pass allocates, so peak memory ≈ a single pass). Same
            # flow, same (x_t, t); only the cond differs (condition-swap self-distill).
            v_teacher = None
            teacher_feats: Dict[int, torch.Tensor] = {}
            if distill:
                tap_t = _BlockTap(flow_model.blocks, kd_idx, detach_fp32=True) if kd_idx else None
                try:
                    with torch.no_grad():
                        tc = teacher_cond.to(x_t_dtype)
                        if is_sparse:
                            tc_list = [tc[b] for b in range(tc.shape[0])]
                            v_teacher = flow_model(x_t, t_in, tc_list, **flow_kwargs)
                            v_teacher = v_teacher.feats.float()
                        else:
                            v_teacher = flow_model(x_t, t_in, tc, cond_mask=None, **flow_kwargs)
                            v_teacher = v_teacher.float()
                finally:
                    if tap_t is not None:
                        teacher_feats = tap_t.captured
                        tap_t.remove()

            # ── STUDENT pass (the original flow loss path, unchanged) ──
            tap_s = _BlockTap(flow_model.blocks, kd_idx, detach_fp32=False) if kd_idx else None
            try:
                if is_sparse:
                    # Sparse path: unpack (cond, mask) → list-of-tensors per sample.
                    # SLatFlowModel.forward auto-wraps the list into a VarLenTensor; sparse
                    # cross-attn (flash_attn_varlen / xformers BlockDiagonalMask) handles
                    # variable length natively — no attn_mask param needed.
                    if cond_mask is not None:
                        m2d = cond_mask[:, 0, 0, :] if cond_mask.dim() == 4 else cond_mask
                        m2d = m2d.bool()
                        cond_list = [cond[b, m2d[b]] for b in range(cond.shape[0])]
                    else:
                        cond_list = [cond[b] for b in range(cond.shape[0])]
                    v_pred = flow_model(x_t, t_in, cond_list, **flow_kwargs)
                    v_target = self.get_v(x_0, noise, t)
                    v_pred_f = v_pred.feats.float()
                    loss = F.mse_loss(v_pred_f, v_target.feats.float())
                else:
                    v_pred = flow_model(x_t, t_in, cond, cond_mask=cond_mask, **flow_kwargs)
                    v_target = self.get_v(x_0, noise, t)
                    v_pred_f = v_pred.float()
                    loss = F.mse_loss(v_pred_f, v_target.float())
            finally:
                student_feats = tap_s.captured if tap_s is not None else {}
                if tap_s is not None:
                    tap_s.remove()

        logs = {
            "flow_mse": loss.detach().float().item(),
            "t_mean": float(t.mean().item()),
        }

        # ── KD terms (fp32, outside autocast — inputs already float) ──
        if distill:
            if kd_v_weight > 0 and v_teacher is not None:
                kd_v = F.mse_loss(v_pred_f, v_teacher)
                loss = loss + kd_v_weight * kd_v
                logs["kd_v"] = kd_v.detach().float().item()
            if kd_idx and teacher_feats:
                # RELATIVE per-block MSE (÷ teacher mean-square): raw block activations
                # have norms ~10-100× the velocity scale, which would let kd_f dominate
                # the gradient at any fixed λ. Normalizing makes kd_f an O(1) relative
                # error, comparable across blocks/stages → λ_f is scale-robust.
                kd_f = torch.stack([
                    F.mse_loss(student_feats[i].float(), teacher_feats[i])
                    / (teacher_feats[i].pow(2).mean() + 1e-6)
                    for i in kd_idx
                ]).mean()
                loss = loss + kd_f_weight * kd_f
                logs["kd_f"] = kd_f.detach().float().item()
        return loss, logs
