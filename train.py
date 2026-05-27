"""trellis2_blip3o training entry-point.

Forked from blip3o.train.train. The BLIP3o source has been edited directly to
build TRELLIS.2 SS Flow (instead of Sana DiT) and compute the SS Flow loss via
`trellis2_blip3o.loss.TRELLIS2FlowMatchingLoss`. No runtime monkey-patches.

Usage:
    torchrun --nproc_per_node=8 train.py \\
        --model_name_or_path BLIP3o/BLIP3o-NEXT-Pretrain-3B \\
        --diffusion_name_or_path Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \\
        --data_path data/index_setup_A.jsonl \\
        --dataset_cls tr2_3d \\
        --output_dir runs/setup_A \\
        --setup_name A --cond_slice full --flow_weight 1.0 \\
        --num_image_tokens 0 --num_scale_tokens 0 --mm_use_im_start_end false \\
        --deepspeed configs/deepspeed_zero2.json --bf16 true \\
        --per_device_train_batch_size 2 ...
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pathlib
import torch
import transformers
from transformers import AutoTokenizer

# Set up sys.path (TRELLIS.2 + local BLIP3o fork) BEFORE importing blip3o.
from trellis2_blip3o import _paths  # noqa: F401

from blip3o.train.train import (  # type: ignore
    ModelArguments as BaseModelArguments,
    DataArguments as BaseDataArguments,
    TrainingArguments as BaseTrainingArguments,
    get_model,
    safe_save_model_for_hf_trainer,
)
from blip3o.train.blip3o_trainer import blip3oTrainer  # type: ignore
from blip3o.utils import rank0_print  # type: ignore
from blip3o.data.dataset import make_supervised_data_module  # type: ignore
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Extended dataclasses with our setup flags
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments(BaseModelArguments):
    trellis_ss_flow_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Path prefix for TRELLIS.2 SS Flow ckpt. None = default in builder."},
    )


@dataclass
class DataArguments(BaseDataArguments):
    use_codebook: bool = field(default=False, metadata={"help": "Setup B/C: inject SigLIP-2 codebook tokens"})
    num_views: int = field(default=4, metadata={"help": "Number of multi-view renders for codebook"})


@dataclass
class TrainingArguments(BaseTrainingArguments):
    setup_name: str = field(default="A", metadata={"help": "A | B | C — for run logging"})
    cond_slice: str = field(default="full", metadata={"help": "full | image_block"})
    flow_weight: float = field(default=1.0, metadata={"help": "Weight on flow_loss"})
    logitnorm_mean: float = field(default=1.0)
    logitnorm_std: float = field(default=1.0)
    detach_cond: bool = field(
        default=False,
        metadata={"help": "Detach VLM hidden before connector — flow loss does NOT "
                          "backprop into VLM (Setup A / MolmoAct2 style). Setup B/C "
                          "should leave this false to faithfully replicate BLIP3o-NEXT joint grad."},
    )
    cond_max_length: int = field(
        default=8192,
        metadata={"help": "Max cond length passed to SS Flow / SLAT Flow cross-attn. "
                          "Sized to cover 8-frame short video input + Setup B/C codebook block. "
                          "Cond longer than this gets truncated."},
    )
    flow_stage_weights: str = field(
        default="ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0",
        metadata={"help": "Per-stage relative weights w_i for the joint cascade loss. "
                          "Parsed to a dict; weights are normalized so ŵ_i sum to 1. "
                          "L_total = L_ce + flow_weight · Σ ŵ_i · L_flow_i. "
                          "Default = all 1.0 → arithmetic mean across active stages "
                          "(matches BLIP3o-NEXT's implicit 1:1 ratio between CE and flow)."},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = get_model(model_args, training_args)
    model.config.use_cache = False

    # ── Reload pretrained TRELLIS flow DiT weights ──
    # HF `from_pretrained` treats the TRELLIS flow keys (ss_flow.*, shape_slat_512.*,
    # tex_slat_512.*) as "missing" because they're absent from the BLIP3o checkpoint,
    # then runs `_init_weights` on them → NaN/zero garbage that overwrites the real
    # TRELLIS weights loaded in __init__. Reload them here (params are still real/CPU,
    # before the Trainer/DeepSpeed wraps & partitions the model).
    from blip3o.model.multimodal_decoder import builder as _flow_builder
    _mm = model.get_model()
    if getattr(_mm, "ss_flow", None) is not None:
        _mm.ss_flow.load_state_dict(_flow_builder.build_ss_flow(model.config).state_dict())
        _mm.shape_slat_512.load_state_dict(_flow_builder.build_shape_slat_512(model.config).state_dict())
        _mm.tex_slat_512.load_state_dict(_flow_builder.build_tex_slat_512(model.config).state_dict())
        rank0_print("Reloaded pretrained TRELLIS flow DiT weights (ss / shape_slat_512 / tex_slat_512).")

    # Plumb our setup flags onto model.config so forward() can read them.
    model.config.cond_slice = training_args.cond_slice
    model.config.flow_weight = training_args.flow_weight
    model.config.logitnorm_mean = training_args.logitnorm_mean
    model.config.logitnorm_std = training_args.logitnorm_std
    model.config.detach_cond = training_args.detach_cond
    model.config.cond_max_length = training_args.cond_max_length
    model.config.flow_stage_weights = training_args.flow_stage_weights
    if model_args.trellis_ss_flow_ckpt is not None:
        model.config.trellis_ss_flow_ckpt = model_args.trellis_ss_flow_ckpt

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _hook(_module, _input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_hook)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is not None:
        # Build vision tower + connector + TRELLIS modules.
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.diffusion_name_or_path = model_args.diffusion_name_or_path
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    else:
        # Setup A may run with no vision tower (text-only). Still set is_multimodal=False
        # so prepare_inputs_labels_for_multimodal skips image embedding.
        data_args.is_multimodal = False

    # ---- Decide trainable parts ----
    rank0_print(f"mm_tunable_parts: {model_args.mm_tunable_parts}")
    model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts

    model.requires_grad_(False)
    if model_args.vision_tower is not None:
        vision_tower.requires_grad_(False)
        vision_tower.eval()

    tunable_parts = [t.strip() for t in model_args.mm_tunable_parts.split(",")]

    def _is_trellis_decoder_param(name: str) -> bool:
        return "trellis_decoders" in name

    def _is_trainable_diffusion_param(name: str) -> bool:
        """SS Flow + Shape SLAT 512 + Tex SLAT 512 — the 3 trainable DiTs."""
        if _is_trellis_decoder_param(name):
            return False
        return ("ss_flow" in name) or ("shape_slat_512" in name) or ("tex_slat_512" in name)

    if "mm_language_model" in tunable_parts:
        for n, p in model.named_parameters():
            if "vision_tower" in n:
                continue
            if _is_trellis_decoder_param(n):
                continue  # frozen TRELLIS decoders
            if _is_trainable_diffusion_param(n):
                continue  # controlled by mm_diffusion flag below
            if "diffusion_connector" in n:
                continue  # controlled below (always on)
            p.requires_grad_(True)

    if "mm_embedding" in tunable_parts:
        for n, p in model.named_parameters():
            if "embed_tokens" in n or "lm_head" in n:
                p.requires_grad_(True)

    if "mm_diffusion" in tunable_parts:
        # Unfreezes all 3 trainable DiTs: SS Flow + Shape SLAT 512 + Tex SLAT 512.
        for n, p in model.named_parameters():
            if _is_trainable_diffusion_param(n):
                p.requires_grad_(True)

    # --- Partial-FT options: unfreeze only attention modules of the 3 flows ---
    # (connector is always trainable below). Lets us adapt the frozen TRELLIS
    # flows to the new BLIP3o-VLM cond without the full ~3.88B trainable that
    # forces CPU-offload. cross_attn = the cond pathway (KV comes from cond) →
    # ~712M w/ connector (18% of full); + self_attn → ~1.56B (40%).
    if "mm_flow_crossattn" in tunable_parts:
        for n, p in model.named_parameters():
            if _is_trainable_diffusion_param(n) and "cross_attn" in n:
                p.requires_grad_(True)

    if "mm_flow_selfattn" in tunable_parts:
        for n, p in model.named_parameters():
            if _is_trainable_diffusion_param(n) and "self_attn" in n:
                p.requires_grad_(True)

    # "mm_flow_lastNN" — unfreeze the LAST NN% of transformer blocks of each flow
    # FULLY (attn + MLP + modulation). MLP is ~59% of each flow's params and is
    # frozen by the attn-only modes above; this is the no-offload way to inject
    # MLP capacity (tex/ss lag without it). e.g. "mm_flow_last40" = last 40% blocks.
    import re as _re
    _last_frac = None
    for _t in tunable_parts:
        _m = _re.fullmatch(r"mm_flow_last(\d+)", _t)
        if _m:
            _last_frac = int(_m.group(1)) / 100.0
    if _last_frac is not None:
        _flow_tags = ("ss_flow", "shape_slat_512", "tex_slat_512")
        _maxblk = {}
        for n, _ in model.named_parameters():
            if not _is_trainable_diffusion_param(n):
                continue
            bm = _re.search(r"blocks\.(\d+)\.", n)
            if not bm:
                continue
            for tag in _flow_tags:
                if tag in n:
                    _maxblk[tag] = max(_maxblk.get(tag, -1), int(bm.group(1)))
        _thr = {tag: round((mx + 1) * (1.0 - _last_frac)) for tag, mx in _maxblk.items()}
        rank0_print(f"mm_flow_last{int(_last_frac*100)}: per-flow block counts={ {t: v+1 for t,v in _maxblk.items()} }, train blocks >= {_thr}")
        for n, p in model.named_parameters():
            if not _is_trainable_diffusion_param(n):
                continue
            bm = _re.search(r"blocks\.(\d+)\.", n)
            if not bm:
                continue
            for tag in _flow_tags:
                if tag in n and int(bm.group(1)) >= _thr[tag]:
                    p.requires_grad_(True)

    # Frozen TRELLIS decoder bundle (Shape SLAT + Tex SLAT + SC-VAE decoder).
    for n, p in model.named_parameters():
        if _is_trellis_decoder_param(n):
            p.requires_grad_(False)

    # Connector — always trainable (small, projects VLM hidden → TRELLIS cond).
    for n, p in model.named_parameters():
        if "diffusion_connector" in n:
            p.requires_grad_(True)

    if "mm_vision_tower" in tunable_parts and model_args.vision_tower is not None:
        for n, p in model.named_parameters():
            if "vision_tower" in n:
                p.requires_grad_(True)

    # "mm_flow_loraNN" — wrap the flows' attention + MLP Linears with LoRA (rank NN).
    # Scale-friendly alternative to full partial-FT: touches MLP (capacity) at low
    # rank, tiny trainable, no offload. Connector stays full (above). Applied AFTER
    # all freeze logic so the new lora_A/B params (default requires_grad=True) stick.
    _lora_r = None
    for _t in tunable_parts:
        _lm = _re.fullmatch(r"mm_flow_lora(\d+)", _t)
        if _lm:
            _lora_r = int(_lm.group(1))
    if _lora_r is not None:
        from trellis2_blip3o.lora import apply_lora_to_flows
        apply_lora_to_flows(model, r=_lora_r, alpha=2 * _lora_r)

    total = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable = sum(
        p.ds_numel if hasattr(p, "ds_numel") else p.numel()
        for p in model.parameters() if p.requires_grad
    )
    rank0_print(
        f"Total params: {total/1e6:.1f} M | "
        f"Trainable: {trainable/1e6:.1f} M ({100.0 * trainable / max(total, 1):.2f}%)"
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = blip3oTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # transformers>=4.55 reads `self.is_tp_enabled` in the training loop; the port's
    # trainer (written for an older transformers) never sets it. We don't use tensor
    # parallelism, so default it to False.
    if not hasattr(trainer, "is_tp_enabled"):
        trainer.is_tp_enabled = False

    if trainer.is_world_process_zero():
        stat = [(i, n, tuple(p.shape), p.requires_grad)
                for i, (n, p) in enumerate(trainer.model.named_parameters())]
        print(tabulate(stat[:40], headers=["idx", "name", "shape", "trainable"]))
        print(f"  (showing first 40 / {len(stat)} params)")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
