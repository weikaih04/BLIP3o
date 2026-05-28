"""Training entry for the native-VLM continuous variant (Phase 2 of QWEN35_VLM_DESIGN.md).

Self-contained (does NOT reuse the TA-Tok/codebook train.py). Builds TrellisNativeVLM
+ the native dataset/collator, runs HF Trainer (+ DeepSpeed via --deepspeed).

v1 default = SS-only (build_slat=False) + frozen VLM. SS target is a plain tensor, so
vanilla Trainer._prepare_inputs moves it to GPU fine. (Cascade/SLAT needs a
SparseTensor-aware _prepare_inputs — follow-up; see QWEN35_VLM_DESIGN.md.)

Launch (1 GPU):
  CUDA_VISIBLE_DEVICES=0 <env>/bin/python train_native.py \
    --vlm_model Qwen/Qwen3-VL-2B-Instruct --data_path data/overfit/imgtext.jsonl \
    --output_dir runs/native_q3vl_overfit --max_steps 200 --bf16 True \
    --per_device_train_batch_size 1 --learning_rate 1e-4 --logging_steps 10

Multi-GPU: torchrun --nproc_per_node=N train_native.py ... --deepspeed configs/deepspeed_zero2.json
Qwen3.5-2B: use the blip3o_trellis_qwen35 env + --vlm_model Qwen/Qwen3.5-2B.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import time

import transformers
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

import trellis2_blip3o._paths  # noqa: F401
from trellis2_blip3o.dataset_native import TR2NativeVLMDataset, NativeVLMCollator
from blip3o.model.language_model.trellis_native_vlm import (
    TrellisNativeVLMConfig, TrellisNativeVLMForConditionalGeneration,
)


class WandbFineGrainedCallback(TrainerCallback):
    """Configure wandb panels at train start + emit per-step wall-clock to wandb.

    - `define_metric` tells wandb how to aggregate each metric in the run-summary table
      (min for losses, max for cond_len/voxels, last for grad_norm/lr). Without this every
      summary defaults to "last" which is noisy for fluctuating per-stage losses.
    - `time/step_sec` lets us see throughput trends + correlate spikes with voxel counts.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
            if wandb.run is None:
                return
            wandb.define_metric("train/loss", summary="min")
            wandb.define_metric("train/per_stage/*", summary="min")
            wandb.define_metric("train/cond/*", summary="mean")
            wandb.define_metric("train/target/*", summary="max")
            wandb.define_metric("train/grad_norm", summary="last")
            wandb.define_metric("train/learning_rate", summary="last")
            wandb.define_metric("train/time/step_sec", summary="mean")
        except Exception as e:
            print(f"[WandbFineGrainedCallback] define_metric skipped: {e}")
        self._t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        try:
            import wandb
            if wandb.run is None:
                return
            now = time.time()
            wandb.log({"train/time/step_sec": now - self._t}, step=state.global_step, commit=False)
            self._t = now
        except Exception:
            pass


def _apply_flow_freeze(model, mode: str):
    """Partial-FT the 3 TRELLIS flows. connector stays trainable; VLM per cfg.freeze_vlm.
    mode: 'full' | 'crossattn' | 'crossattn,selfattn' | 'last{NN}' (last NN% of blocks, incl MLP)."""
    import re
    parts = [p.strip() for p in mode.split(",") if p.strip()]
    tags = ("ss_flow", "shape_slat_512", "tex_slat_512")
    def is_flow(n): return any(t in n for t in tags)
    if mode == "full" or not parts:
        return  # flows already fully trainable
    for n, p in model.named_parameters():       # freeze all flow params, then unfreeze selected
        if is_flow(n):
            p.requires_grad_(False)
    if "crossattn" in parts:
        for n, p in model.named_parameters():
            if is_flow(n) and "cross_attn" in n:
                p.requires_grad_(True)
    if "selfattn" in parts:
        for n, p in model.named_parameters():
            if is_flow(n) and "self_attn" in n:
                p.requires_grad_(True)
    lm = next((re.fullmatch(r"last(\d+)", x) for x in parts if re.fullmatch(r"last(\d+)", x)), None)
    if lm:
        frac = int(lm.group(1)) / 100.0
        maxb = {}
        for n, _ in model.named_parameters():
            if not is_flow(n):
                continue
            bm = re.search(r"blocks\.(\d+)\.", n)
            if bm:
                for t in tags:
                    if t in n:
                        maxb[t] = max(maxb.get(t, -1), int(bm.group(1)))
        thr = {t: round((mx + 1) * (1.0 - frac)) for t, mx in maxb.items()}
        for n, p in model.named_parameters():
            if not is_flow(n):
                continue
            bm = re.search(r"blocks\.(\d+)\.", n)
            if bm:
                for t in tags:
                    if t in n and int(bm.group(1)) >= thr[t]:
                        p.requires_grad_(True)
    for n, p in model.named_parameters():       # connector always trainable
        if "diffusion_connector" in n:
            p.requires_grad_(True)


class NativeTrainer(Trainer):
    """Trainer that (1) moves trellis2 SparseTensor SLAT targets to the device — vanilla
    Trainer._prepare_inputs only moves torch.Tensors, leaving SparseTensors on CPU and
    crashing the cascade — and (2) emits per-stage / cond / voxel-count diagnostics from
    model.forward into Trainer.log → wandb."""

    def _prepare_inputs(self, inputs):
        prepared = super()._prepare_inputs(inputs)
        for k, v in list(prepared.items()):
            if hasattr(v, "feats") and hasattr(v, "to"):  # SparseTensor
                prepared[k] = v.to(self.args.device)
        return prepared

    @staticmethod
    def _unwrap(m):
        """Peel DeepSpeed/DDP/etc. wrappers off to reach the original TrellisNativeVLM."""
        for _ in range(4):
            inner = getattr(m, "module", None)
            if inner is None:
                break
            m = inner
        return m

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # HF 4.4x+ added num_items_in_batch; pass through if supported.
        try:
            out = super().compute_loss(model, inputs, return_outputs=True,
                                       num_items_in_batch=num_items_in_batch)
        except TypeError:
            out = super().compute_loss(model, inputs, return_outputs=True)
        loss, model_out = out if isinstance(out, tuple) else (out, None)

        # Pull per-step diagnostics that the model stashed in forward; log on rank-0 only.
        diag = getattr(self._unwrap(model), "_last_diag", None)
        if diag and getattr(self.accelerator, "is_main_process", True):
            self.log({f"train/{k}": v for k, v in diag.items()})

        return (loss, model_out) if return_outputs else loss


@dataclass
class NativeArgs:
    vlm_model: str = field(default="Qwen/Qwen3-VL-2B-Instruct")
    data_path: str = field(default="data/overfit/imgtext.jsonl")
    freeze_vlm: bool = field(default=True)
    build_slat: bool = field(default=False)   # v1: SS-only
    ss_only: bool = field(default=True)
    num_cond_views: int = field(default=1)    # legacy schema only (multi_view_renders[:N])
    flow_weight: float = field(default=1.0)
    detach_cond: bool = field(default=False)
    cond_max_length: int = field(default=8192)
    # Flow partial-FT (connector ALWAYS trainable; VLM per freeze_vlm). Options:
    #   "full" | "crossattn" | "crossattn,selfattn" | "last40" (lastNN = last NN% blocks incl MLP)
    # Mirrors the blip3o_qwen capacity ladder; last40/crossattn,selfattn are no-offload-friendly.
    flow_tune: str = field(default="full")
    # cond conditioning: "none" (hidden[-1]) | "penultimate" ([-2]) | "depthwise" (Semantic Routing)
    cond_fusion: str = field(default="none")
    fusion_layers: int = field(default=0)   # depthwise: # VLM layer outputs to fuse (0 = all)
    slat_resolution: int = field(default=512)   # 512 or 1024 (HR cascade — use ABO 1024 latents)
    # ── Unified-schema knobs (only used when manifest rows contain `renders_dir`) ──
    # Task distribution per __getitem__: T (text-only), I1 (single image), IM (multi-image).
    # Empty/None → dataset default 'T:0.2,I1:0.4,IM:0.4'. Format: 'k:p,k:p,...'.
    task_mix: str = field(default="T:0.2,I1:0.4,IM:0.4")
    max_views: int = field(default=4)            # IM: n_views ~ U[2, max_views]
    crop_to_object: bool = field(default=False)  # tight alpha-bbox crop on RGBA renders
    min_aesthetic: Optional[float] = field(default=None)  # init-time filter
    # ── Mixture mode (Phase-1/2 multi-task infra) ──
    # When set, replaces --data_path. Wires up MixtureIterableDataset +
    # MultiTaskCollator from yaml (see configs/mix_*.yaml). Task weights and
    # per-task args live in the yaml; per-task class lives in
    # trellis2_blip3o/data/tasks/. Adding a new task = new file + new yaml row.
    mixture_config: Optional[str] = field(default=None)
    mixture_seed: int = field(default=0)


def main():
    parser = HfArgumentParser((NativeArgs, TrainingArguments))
    native_args, training_args = parser.parse_args_into_dataclasses()
    # The dataset emits non-forward columns (caption/images/...); Trainer must NOT
    # strip them before the collator runs.
    training_args.remove_unused_columns = False

    cfg = TrellisNativeVLMConfig(
        vlm_model=native_args.vlm_model,
        freeze_vlm=native_args.freeze_vlm,
        build_slat=native_args.build_slat,
        flow_weight=native_args.flow_weight,
        detach_cond=native_args.detach_cond,
        cond_max_length=native_args.cond_max_length,
        cond_fusion=native_args.cond_fusion,
        fusion_layers=native_args.fusion_layers,
        slat_resolution=native_args.slat_resolution,
    )
    model = TrellisNativeVLMForConditionalGeneration(cfg)
    _apply_flow_freeze(model, native_args.flow_tune)
    print(f"[train_native] flow_tune={native_args.flow_tune!r}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train_native] trainable {n_train/1e6:.1f}M / total {n_total/1e6:.1f}M  "
          f"(vlm frozen={cfg.freeze_vlm}, build_slat={cfg.build_slat})")

    processor = AutoProcessor.from_pretrained(native_args.vlm_model)

    # Choose data path:
    #   --mixture_config (new, multi-task)   → MixtureIterableDataset + MultiTaskCollator
    #   else (legacy, single-dataset)        → TR2NativeVLMDataset + NativeVLMCollator
    if native_args.mixture_config:
        from trellis2_blip3o.data import build_mixture
        per_dev_bs = max(1, int(training_args.per_device_train_batch_size))
        mix = build_mixture(
            native_args.mixture_config,
            processor=processor,
            batch_size=per_dev_bs,
            base_seed=native_args.mixture_seed,
        )
        print(f"[train_native] {mix.summary()}")
        train_ds = mix.dataset
        collator = mix.collator
        # IterableDataset is incompatible with HF Trainer's epoch-based length.
        # Driver must rely on --max_steps.
        if not training_args.max_steps or training_args.max_steps <= 0:
            raise ValueError(
                "--mixture_config uses IterableDataset; --max_steps must be set "
                "(epoch-based training is not meaningful in mixture mode)."
            )
    else:
        data_args = SimpleNamespace(
            use_codebook=False, num_views=1,
            num_cond_views=native_args.num_cond_views,           # legacy
            task_mix=native_args.task_mix,                       # unified
            max_views=native_args.max_views,
            slat_resolution=native_args.slat_resolution,
            crop_to_object=native_args.crop_to_object,
            min_aesthetic=native_args.min_aesthetic,
        )
        train_ds = TR2NativeVLMDataset(
            data_path=native_args.data_path, data_args=data_args, ss_only=native_args.ss_only,
        )
        collator = NativeVLMCollator(processor=processor)

    trainer = NativeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[WandbFineGrainedCallback()],
    )
    trainer.train(resume_from_checkpoint=bool(list(__import__("pathlib").Path(training_args.output_dir).glob("checkpoint-*"))) or None)
    trainer.save_state()


if __name__ == "__main__":
    main()
