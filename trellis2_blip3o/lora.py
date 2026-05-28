"""Minimal hand-rolled LoRA for the TRELLIS.2 flow DiTs — NO peft dependency.

Why hand-rolled: the env pins torch/transformers/flash_attn (see project notes);
`pip install peft` risks bumping transformers. This is ~1 small module instead.

`LoRALinear` is a DROP-IN wrapper around an existing nn.Linear: it freezes the
base weight and adds a low-rank update `scaling * (x A^T) B^T` (B zero-init, so at
init the wrapper == base). Crucially it keeps the plain `(tensor)->tensor` call
signature, so it works in BOTH flow paths:
  - dense MultiHeadAttention:           `self.to_q(x)`
  - sparse SparseMultiHeadAttention:    `self._linear(self.to_q, x)` → `to_q(x.feats)`

Target modules (default): the attention + MLP Linears inside the 3 trainable
flows (ss_flow / shape_slat_512 / tex_slat_512). MLP coverage matters — it's
~59% of each flow's params and carries most representational capacity.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 64, alpha: int = 128, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = int(r)
        self.scaling = alpha / float(r)
        w = base.weight
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays 0 → init == base
        self.lora_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x may be a plain Tensor (dense flow / attn via _linear(module, x.feats))
        # OR a SparseTensor (the MLP is called directly on the SparseTensor). Compute
        # the low-rank update in feature space and recombine accordingly.
        out = self.base(x)
        feats = x.feats if hasattr(x, "feats") else x
        upd = self.scaling * ((self.lora_drop(feats) @ self.lora_A.t()) @ self.lora_B.t())
        if hasattr(out, "replace") and hasattr(out, "feats"):   # SparseTensor
            return out.replace(out.feats + upd)
        return out + upd

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the LoRA update into base.weight and return the plain nn.Linear.
        After this the module behaves identically but has standard Linear keys —
        so the merged model saves/loads like the partial-FT runs (no infer change)."""
        delta = (self.lora_B @ self.lora_A) * self.scaling      # (out_f, in_f)
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def _is_default_target(name: str) -> bool:
    # attention (cross/self) + the feed-forward Linears (mlp.mlp.0 / mlp.mlp.2)
    return ("cross_attn" in name) or ("self_attn" in name) or (".mlp.mlp." in name)


def apply_lora_to_flows(
    model,
    r: int = 64,
    alpha: int = None,
    flow_tags=("ss_flow", "shape_slat_512", "tex_slat_512"),
    is_target=_is_default_target,
):
    """In-place wrap target nn.Linear modules of the 3 flows with LoRALinear.

    Call AFTER the freeze logic (the new lora_A/B default to requires_grad=True).
    Returns (n_wrapped, n_lora_params)."""
    if alpha is None:
        alpha = 2 * r
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(t in name for t in flow_tags) and is_target(name):
            targets.append(name)
    n_params = 0
    for name in targets:
        parent = model
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        base = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base, r=r, alpha=alpha))
        n_params += r * (base.in_features + base.out_features)
    try:
        from blip3o.utils import rank0_print
        rank0_print(f"[LoRA] wrapped {len(targets)} flow Linear (r={r}, alpha={alpha}) "
                    f"→ +{n_params/1e6:.1f}M trainable LoRA params")
    except Exception:
        pass
    return len(targets), n_params


@torch.no_grad()
def merge_lora_in_model(model):
    """Replace every LoRALinear in `model` with its merged plain nn.Linear, so the
    result saves/loads with standard keys. Use post-training before save_pretrained
    (or after loading a LoRA ckpt that was wrapped). Returns #merged."""
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            parent = model
            *path, attr = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, attr, mod.merge())
            n += 1
    return n
