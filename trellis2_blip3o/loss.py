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
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from . import _paths  # noqa: F401  — sets sys.path so trellis2 is importable
from trellis2.trainers.flow_matching.flow_matching import FlowMatchingTrainer as _FMT


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

        Returns:
            loss: scalar tensor.
            logs: {flow_mse, t_mean}.
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

        # TRELLIS.2's TimestepEmbedder builds an fp32 t_freq internally, which the
        # bf16 t-embedder MLP can only consume under autocast. HF Trainer + DeepSpeed
        # bf16 does NOT enable torch.autocast (DeepSpeed manages precision itself), so
        # we enter it explicitly here to avoid a Float/BFloat16 matmul mismatch.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                loss = F.mse_loss(v_pred.feats.float(), v_target.feats.float())
            else:
                v_pred = flow_model(x_t, t_in, cond, cond_mask=cond_mask, **flow_kwargs)
                v_target = self.get_v(x_0, noise, t)
                loss = F.mse_loss(v_pred.float(), v_target.float())

        logs = {
            "flow_mse": loss.detach().float().item(),
            "t_mean": float(t.mean().item()),
        }
        return loss, logs
