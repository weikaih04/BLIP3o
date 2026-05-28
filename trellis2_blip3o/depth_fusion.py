"""Depth-wise Semantic Routing for the TRELLIS native-VLM variant.

Ports SemanticRouting's depth-wise fusion (Mode 3, arXiv 2602.03510 — verified vs
their `diffusion/models.py::_fuse_text_features`): each DiT block gets its OWN learnable
softmax gate over the L VLM layers, zero-init (→ uniform average at start). Per block:
    fused_d = Σ_l softmax(gate_d)_l · LayerNorm(hidden_l)         # convex, in VLM-hidden space
    cond_d  = connector(fused_d)                                  # our TRELLIS2Connector (dist-match)

Wiring into TRELLIS (whose flow we do NOT own): every flow block's forward is
`block(h, t_emb, cond, ...)` with cond at positional index 2 (verified for SS dense +
SLAT sparse). We wrap each block so it ignores the passed cond and uses its own cond_d.
No third-party code edited; no new DiT params beyond the tiny per-block gates.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

COND_ARG_IDX = 2  # block.forward(h, t_emb, cond, ...) — cond is the 3rd positional arg


class DepthFusionRouter(nn.Module):
    """Per-block convex fusion over L VLM layers → per-block cond via the shared connector."""

    def __init__(self, num_blocks: int, num_layers: int, connector: nn.Module):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.num_layers = int(num_layers)
        # connector is shared with the model; register as a (non-owned) ref so it is NOT
        # double-counted as a submodule of the router.
        object.__setattr__(self, "connector", connector)
        # zero-init → softmax(zeros) uniform (matches SemanticRouting Mode 3).
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.num_layers)) for _ in range(self.num_blocks)]
        )
        self._cond_per_block: List[torch.Tensor] | None = None  # stashed each forward

    def compute(self, layer_hiddens: List[torch.Tensor]) -> List[torch.Tensor]:
        """layer_hiddens: list of L tensors (B, T, C). → list of num_blocks tensors (B, T, 1024)."""
        assert len(layer_hiddens) == self.num_layers, \
            f"expected {self.num_layers} layers, got {len(layer_hiddens)}"
        stacked = torch.stack(layer_hiddens, dim=0).float()              # (L, B, T, C)
        stacked = F.layer_norm(stacked, (stacked.shape[-1],))           # per-layer LN
        w = torch.stack([F.softmax(g.float(), dim=0) for g in self.gates], dim=0)  # (K, L)
        fused = torch.einsum("kl,lbtc->kbtc", w, stacked)               # (K, B, T, C)
        dt = layer_hiddens[0].dtype
        return [self.connector(fused[d].to(dt)) for d in range(self.num_blocks)]   # K × (B,T,1024)

    def set_cond(self, layer_hiddens: List[torch.Tensor]):
        self._cond_per_block = self.compute(layer_hiddens)

    def clear(self):
        self._cond_per_block = None


class _CondInjectBlock(nn.Module):
    """Wraps a TRELLIS flow block; replaces the cond positional arg (idx 2) with the
    router's per-block cond. Signature-transparent (*args/**kwargs) so the flow's own
    forward loop calls it unchanged."""

    def __init__(self, block: nn.Module, get_cond, cond_arg_idx: int = COND_ARG_IDX):
        super().__init__()
        self.block = block                                  # keeps block params registered
        self.cond_arg_idx = cond_arg_idx
        object.__setattr__(self, "_get_cond", get_cond)     # plain ref, not a submodule

    def forward(self, *args, **kwargs):
        cond = self._get_cond()
        if cond is not None and len(args) > self.cond_arg_idx:
            args = list(args)
            args[self.cond_arg_idx] = cond
        return self.block(*args, **kwargs)


def install_depth_routing(flow: nn.Module, router: DepthFusionRouter):
    """Wrap every block of `flow.blocks` so block d uses router._cond_per_block[d].
    Idempotent-safe only on a fresh flow (call once)."""
    blocks = flow.blocks
    n = len(blocks)
    assert n == router.num_blocks, f"flow has {n} blocks, router built for {router.num_blocks}"
    def _getter(i):
        return lambda: (router._cond_per_block[i] if router._cond_per_block is not None else None)
    for i in range(n):
        blocks[i] = _CondInjectBlock(blocks[i], _getter(i))
    return flow
