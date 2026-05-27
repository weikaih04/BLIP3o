"""Builders for TRELLIS.2 modules used inside the unified VLM+diffusion pipeline.

We reuse `trellis2.models.from_pretrained` — it reads `{path}.json` config and
`{path}.safetensors` weights side-by-side and instantiates the right class.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from . import _paths  # noqa: F401  (sys.path setup)


# Default local paths — symlinked from the TRELLIS.2-4B HF release.
DEFAULT_TRELLIS2_CKPTS = os.path.join(_paths.CHECKPOINTS_ROOT, "TRELLIS.2-4B", "ckpts")
DEFAULT_SS_FLOW = os.path.join(DEFAULT_TRELLIS2_CKPTS, "ss_flow_img_dit_1_3B_64_bf16")
DEFAULT_SHAPE_DEC = os.path.join(DEFAULT_TRELLIS2_CKPTS, "shape_dec_next_dc_f16c32_fp16")
DEFAULT_TEX_DEC = os.path.join(DEFAULT_TRELLIS2_CKPTS, "tex_dec_next_dc_f16c32_fp16")
DEFAULT_SHAPE_SLAT = os.path.join(DEFAULT_TRELLIS2_CKPTS, "slat_flow_img2shape_dit_1_3B_512_bf16")
DEFAULT_TEX_SLAT = os.path.join(DEFAULT_TRELLIS2_CKPTS, "slat_flow_imgshape2tex_dit_1_3B_512_bf16")


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def build_trellis_ss_flow(ckpt_path: Optional[str] = None, trainable: bool = True) -> nn.Module:
    """Load TRELLIS.2 SparseStructureFlowModel from local 4B ckpt.

    Args:
        ckpt_path: prefix for `{path}.json` + `{path}.safetensors`. Defaults to local
                   `checkpoints/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16`.
        trainable: if False, freeze all params (eval mode, requires_grad=False).
    """
    from trellis2 import models  # type: ignore
    model = models.from_pretrained(ckpt_path or DEFAULT_SS_FLOW)
    if not trainable:
        _freeze(model)
    return model


def build_shape_slat_flow_frozen(ckpt_path: Optional[str] = None) -> nn.Module:
    """Frozen Shape SLAT flow — used for *inference only* in Phase 1."""
    from trellis2 import models  # type: ignore
    return _freeze(models.from_pretrained(ckpt_path or DEFAULT_SHAPE_SLAT))


def build_tex_slat_flow_frozen(ckpt_path: Optional[str] = None) -> nn.Module:
    """Frozen Tex SLAT flow — inference only."""
    from trellis2 import models  # type: ignore
    return _freeze(models.from_pretrained(ckpt_path or DEFAULT_TEX_SLAT))


def build_sc_vae_shape_decoder_frozen(ckpt_path: Optional[str] = None) -> nn.Module:
    """Frozen SC-VAE shape decoder — inference only, decodes SLAT → mesh."""
    from trellis2 import models  # type: ignore
    return _freeze(models.from_pretrained(ckpt_path or DEFAULT_SHAPE_DEC))


def build_sc_vae_tex_decoder_frozen(ckpt_path: Optional[str] = None) -> nn.Module:
    """Frozen SC-VAE tex decoder — inference only, decodes PBR voxel volume."""
    from trellis2 import models  # type: ignore
    return _freeze(models.from_pretrained(ckpt_path or DEFAULT_TEX_DEC))


def build_all_frozen_modules() -> Dict[str, nn.Module]:
    """Convenience: build all the modules we keep frozen during training."""
    return {
        "shape_slat_flow": build_shape_slat_flow_frozen(),
        "tex_slat_flow": build_tex_slat_flow_frozen(),
        "shape_dec": build_sc_vae_shape_decoder_frozen(),
        "tex_dec": build_sc_vae_tex_decoder_frozen(),
    }


# ----------------------------------------------------------------------------
# Per-stage normalization stats loader.
#
# TRELLIS.2 trains each stage with a per-channel `(x - mean) / std` normalization
# (mean/std vectors live inside the stage's training config JSON, under the
# dataset args). We must apply the EXACT same normalization on our SLAT targets
# at training time so the pretrained checkpoints' decoders / flow models see the
# distribution they were trained on.
#
# Tex SLAT has TWO sets: `pbr_slat_normalization` (target) and
# `shape_slat_normalization` (concat_cond input that mirrors Shape SLAT's own
# normalization — they're the same numbers).
# ----------------------------------------------------------------------------

# TRELLIS.2 stage configs (the source of truth for normalization).
TRELLIS_REPO_ROOT = os.path.join(
    os.path.dirname(_paths.CHECKPOINTS_ROOT) if hasattr(_paths, "CHECKPOINTS_ROOT") else "",
    "third_party_3d_gen", "TRELLIS.2",
)
# Fall back to the absolute path we know works.
if not os.path.isdir(TRELLIS_REPO_ROOT):
    TRELLIS_REPO_ROOT = "/weka/oe-training-default/weikaih/world_explore/third_party_3d_gen/TRELLIS.2"

SS_FLOW_CONFIG_PATH    = os.path.join(TRELLIS_REPO_ROOT, "configs/gen/ss_flow_img_dit_1_3B_64_bf16.json")
SHAPE_SLAT_CONFIG_PATH = os.path.join(TRELLIS_REPO_ROOT, "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json")
TEX_SLAT_CONFIG_PATH   = os.path.join(TRELLIS_REPO_ROOT, "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json")


def load_norm_stats(config_path: str, key: str = "normalization") -> Optional[Dict[str, torch.Tensor]]:
    """Read per-channel normalization mean/std from a TRELLIS.2 stage config JSON.

    Args:
        config_path: path to TRELLIS.2 `configs/gen/<stage>.json`.
        key: which normalization key under `dataset.args` to read. Use
             "normalization" for SS / Shape SLAT, and "pbr_slat_normalization"
             or "shape_slat_normalization" for Tex SLAT (which has two sets).

    Returns:
        {"mean": Tensor, "std": Tensor} with shape (1, C) for per-channel broadcast,
        or None if the config doesn't have this key (e.g. SS Flow has no
        normalization, by upstream design).
    """
    import json

    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        cfg = json.load(f)
    args = cfg.get("dataset", {}).get("args", {})
    norm = args.get(key)
    if norm is None:
        return None
    # Reshape to (1, C) so it broadcasts against (N, C) feats — TRELLIS.2 does
    # the same: `torch.tensor(...).reshape(1, -1)` in SLatShape.__init__.
    return {
        "mean": torch.tensor(norm["mean"]).reshape(1, -1),
        "std":  torch.tensor(norm["std"]).reshape(1, -1),
    }
