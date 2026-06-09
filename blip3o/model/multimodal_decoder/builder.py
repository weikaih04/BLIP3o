"""Builders for the 3D-diffusion stack.

trellis2_blip3o forks BLIP3o-NEXT and swaps Sana DiT+VAE → TRELLIS.2 modules:
  - SS Flow (trainable, 1.3B sparse-structure DiT) — replaces Sana DiT
  - Shape SLAT + Tex SLAT + SC-VAE decoders (frozen) — replace Sana VAE

These builders are called by `blip3oMetaModel.__init__` and
`blip3oMetaModel.initialize_vision_modules`.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

import trellis2_blip3o._paths as _tr2_paths  # noqa: F401 — sets sys.path for trellis2 import


def _to_bf16_keep_complex(model: nn.Module) -> nn.Module:
    """`model.to(torch.bfloat16)` BUT preserve complex buffers.

    The dense SS Flow registers `rope_phases` as a **complex64** RoPE buffer. A blunt
    `model.to(bfloat16)` casts it to bf16, which silently DISCARDS the imaginary part
    ("Casting complex values to real discards the imaginary part") — destroying the rotary
    rotation → corrupted positions → attention diverges layer-by-layer → degraded geometry
    (thin/hollow shapes shatter). Upstream TRELLIS avoids this by casting only `self.blocks`
    (convert_module_to) and leaving rope_phases complex64. Mirror that: snapshot complex
    buffers, do the bf16 cast, then restore them.
    """
    complex_bufs = {n: b.clone() for n, b in model.named_buffers() if b is not None and b.is_complex()}
    model = model.to(torch.bfloat16)
    for name, buf in complex_bufs.items():
        mod = model
        *path, leaf = name.split(".")
        for p in path:
            mod = getattr(mod, p)
        # buffers registered via register_buffer live in _buffers; setattr re-registers correctly.
        # `.to(bfloat16)` is dtype-only (no device move), so the cloned complex buf is already on
        # the right device — re-attach it as-is, preserving complex64.
        setattr(mod, leaf, buf)
    return model


def _default_ss_flow_ckpt() -> str:
    return os.path.join(
        _tr2_paths.CHECKPOINTS_ROOT,
        "TRELLIS.2-4B",
        "ckpts",
        "ss_flow_img_dit_1_3B_64_bf16",
    )


def _default_shape_slat_ckpt() -> str:
    # 512 LR variant — output 32^3 sparse, paired with '512' pipeline_type.
    return os.path.join(
        _tr2_paths.CHECKPOINTS_ROOT,
        "TRELLIS.2-4B",
        "ckpts",
        "slat_flow_img2shape_dit_1_3B_512_bf16",
    )


def _default_tex_slat_ckpt() -> str:
    # 512 LR variant — output 32^3 sparse, paired with '512' pipeline_type.
    return os.path.join(
        _tr2_paths.CHECKPOINTS_ROOT,
        "TRELLIS.2-4B",
        "ckpts",
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    )


def _default_sc_vae_decoder_ckpt() -> str:
    # SC-VAE shape decoder. Tex decoder available separately.
    return os.path.join(
        _tr2_paths.CHECKPOINTS_ROOT,
        "TRELLIS.2-4B",
        "ckpts",
        "shape_dec_next_dc_f16c32_fp16",
    )


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


def build_ss_flow(cfg, **kwargs) -> nn.Module:
    """Build TRELLIS.2 SparseStructureFlowModel (trainable, 1.3B)."""
    from trellis2 import models

    ckpt = getattr(cfg, "trellis_ss_flow_ckpt", None) or _default_ss_flow_ckpt()
    model = models.from_pretrained(ckpt)
    return _to_bf16_keep_complex(model)


def build_shape_slat_512(cfg, **kwargs) -> nn.Module:
    """Build TRELLIS.2 Shape SLAT 512 (trainable, 1.3B). Sparse, output 32^3."""
    from trellis2 import models

    ckpt = getattr(cfg, "trellis_shape_slat_ckpt", None) or _default_shape_slat_ckpt()
    model = models.from_pretrained(ckpt)
    return _to_bf16_keep_complex(model)


def build_tex_slat_512(cfg, **kwargs) -> nn.Module:
    """Build TRELLIS.2 Tex SLAT 512 (trainable, 1.3B). Sparse, output 32^3."""
    from trellis2 import models

    ckpt = getattr(cfg, "trellis_tex_slat_ckpt", None) or _default_tex_slat_ckpt()
    model = models.from_pretrained(ckpt)
    return _to_bf16_keep_complex(model)


# === 1024 variants (HR cascade) — paired with 1024-encoder latents (64^3 sparse target) ===

def _default_shape_slat_1024_ckpt() -> str:
    return os.path.join(_tr2_paths.CHECKPOINTS_ROOT, "TRELLIS.2-4B", "ckpts",
                        "slat_flow_img2shape_dit_1_3B_1024_bf16")


def _default_tex_slat_1024_ckpt() -> str:
    return os.path.join(_tr2_paths.CHECKPOINTS_ROOT, "TRELLIS.2-4B", "ckpts",
                        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16")


def build_shape_slat_1024(cfg, **kwargs) -> nn.Module:
    """Build TRELLIS.2 Shape SLAT 1024 (trainable, 1.3B). Sparse, output 64^3."""
    from trellis2 import models
    ckpt = getattr(cfg, "trellis_shape_slat_1024_ckpt", None) or _default_shape_slat_1024_ckpt()
    return _to_bf16_keep_complex(models.from_pretrained(ckpt))


def build_tex_slat_1024(cfg, **kwargs) -> nn.Module:
    """Build TRELLIS.2 Tex SLAT 1024 (trainable, 1.3B). Sparse, output 64^3."""
    from trellis2 import models
    ckpt = getattr(cfg, "trellis_tex_slat_1024_ckpt", None) or _default_tex_slat_1024_ckpt()
    return _to_bf16_keep_complex(models.from_pretrained(ckpt))


def build_trellis_decoders(cfg, **kwargs) -> nn.ModuleDict:
    """Build the frozen decoder bundle (Shape SLAT + Tex SLAT + SC-VAE decoder).

    Wrapped in `nn.ModuleDict` for HF state_dict + .to(device) compatibility.
    Used only at INFERENCE — training consumes cached `ss_latent.npz` so we
    never call these forward during step().
    """
    from trellis2 import models

    bundle = nn.ModuleDict()
    for name, ckpt_attr, default_fn in [
        ("shape_slat", "trellis_shape_slat_ckpt", _default_shape_slat_ckpt),
        ("tex_slat", "trellis_tex_slat_ckpt", _default_tex_slat_ckpt),
        ("sc_vae_decoder", "trellis_sc_vae_ckpt", _default_sc_vae_decoder_ckpt),
    ]:
        ckpt = getattr(cfg, ckpt_attr, None) or default_fn()
        try:
            bundle[name] = _freeze(models.from_pretrained(ckpt))
        except Exception as e:
            print(f"[build_trellis_decoders] WARN: {name} ckpt not loadable ({e}); skipping")

    return bundle
