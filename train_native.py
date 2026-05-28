"""Training entry for the native-VLM continuous variant (Phase 2 of QWEN35_VLM_DESIGN.md).

Self-contained (does NOT reuse the TA-Tok/codebook train.py). Builds TrellisNativeVLM
+ the native dataset/collator, runs HF Trainer (+ DeepSpeed via --deepspeed).

v1 default = SS-only (build_slat=False) + frozen VLM. SS target is a plain tensor, so
vanilla Trainer._prepare_inputs moves it to GPU fine. (Cascade/SLAT needs a
SparseTensor-aware _prepare_inputs — follow-up; see QWEN35_VLM_DESIGN.md.)

Only Qwen3.5-2B is supported. Use env `blip3o_trellis_qwen35` (transformers 5.2.0).

Launch (1 GPU):
  CUDA_VISIBLE_DEVICES=0 <env>/bin/python train_native.py \
    --vlm_model Qwen/Qwen3.5-2B --data_path data/overfit/imgtext.jsonl \
    --output_dir runs/native_q35_overfit --max_steps 200 --bf16 True \
    --per_device_train_batch_size 1 --learning_rate 1e-4 --logging_steps 10

Multi-GPU: torchrun --nproc_per_node=N train_native.py ... --deepspeed configs/deepspeed_zero2.json

# DEPRECATED (2026-05-28): Qwen3-VL-2B-Instruct and Qwen2.5-VL-3B-Instruct backbones
# are no longer supported. The model class remains backbone-agnostic so they CAN
# still be loaded for inspection, but they aren't a tested training path.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import time

import torch
import transformers
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

import trellis2_blip3o._paths  # noqa: F401
from trellis2_blip3o.dataset_native import TR2NativeVLMDataset, NativeVLMCollator
from blip3o.model.language_model.trellis_native_vlm import (
    TrellisNativeVLMConfig, TrellisNativeVLMForConditionalGeneration,
)


class WandbFineGrainedCallback(TrainerCallback):
    """Set wandb run-summary aggregation for the metrics we log: train/loss and the per-stage
    flow losses use `min` (best-so-far), lr/grad_norm use `last`. Everything goes through
    Trainer.log() on the main step (see NativeTrainer.log), so we do NOT call wandb.log()
    here — a separate per-step wandb.log() with its own step caused monotonic-step drops."""

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
            if wandb.run is None:
                return
            wandb.define_metric("train/loss", summary="min")
            wandb.define_metric("train/per_stage/*", summary="min")
            wandb.define_metric("train/grad_norm", summary="last")
            wandb.define_metric("train/learning_rate", summary="last")
        except Exception as e:
            print(f"[WandbFineGrainedCallback] define_metric skipped: {e}")


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
        # Wrap forward in each elastic SLAT controller's record() context, matching TRELLIS
        # official trainers/basic.py:run_step. record() resets `_last_input_size` per step so
        # variable voxel counts across assets don't trip controller's same-size assertion
        # (without this, step 2 with a different-sized asset raises ValueError).
        from contextlib import ExitStack
        base = self._unwrap(model)
        ctxs = []
        for name in ("shape_slat_512", "tex_slat_512"):
            m = getattr(base, name, None)
            ctrl = getattr(m, "_memory_controller", None) if m is not None else None
            if ctrl is not None:
                ctxs.append(ctrl.record())

        with ExitStack() as stack:
            for ctx in ctxs:
                stack.enter_context(ctx)
            # HF 4.4x+ added num_items_in_batch; pass through if supported.
            try:
                out = super().compute_loss(model, inputs, return_outputs=True,
                                           num_items_in_batch=num_items_in_batch)
            except TypeError:
                out = super().compute_loss(model, inputs, return_outputs=True)
        loss, model_out = out if isinstance(out, tuple) else (out, None)

        # Pull per-step diagnostics that the model stashed in forward; log on rank-0 only.
        diag = getattr(base, "_last_diag", None)
        if diag and getattr(self.accelerator, "is_main_process", True):
            self.log({f"train/{k}": v for k, v in diag.items()})

        return (loss, model_out) if return_outputs else loss


@dataclass
class NativeArgs:
    vlm_model: str = field(default="Qwen/Qwen3.5-2B")
    # DEPRECATED backbones (kept loadable but no longer a tested training path):
    #   "Qwen/Qwen3-VL-2B-Instruct"   — needs the `blip3o_trellis` env + transformers 4.57.6
    #   "Qwen/Qwen2.5-VL-3B-Instruct" — needs the `blip3o_trellis` env + transformers 4.57.6
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
    # SLAT voxel cap (matches TRELLIS official slat_flow_*_512 `max_tokens: 8192`).
    # Oversized assets are resampled to a different index in __getitem__. 0 disables.
    # Complements (does NOT replace) elastic SLAT GC: cap bounds the worst case at the
    # SOURCE; elastic GC handles normal voxel-count variation within the cap.
    max_slat_tokens: int = field(default=8192)
    # ── Mixture mode (Phase-1/2 multi-task infra) ──
    # When set, replaces --data_path. Wires up MixtureIterableDataset +
    # MultiTaskCollator from yaml (see configs/mix_*.yaml). Task weights and
    # per-task args live in the yaml; per-task class lives in
    # trellis2_blip3o/data/tasks/. Adding a new task = new file + new yaml row.
    mixture_config: Optional[str] = field(default=None)
    mixture_seed: int = field(default=0)

    # torch.compile on the dense SS flow. -18.8% step time at ~zero quality risk on
    # tests/profile_native_compile.py (BASELINE 234ms → 190ms; backward -30%).
    # ONLY ss_flow is compiled: VLM has FLA-custom-triton (Dynamo breaks), SLAT flows
    # are sparse + custom-triton + elastic-gc (Dynamo breaks). Cascade mode leaves the
    # SLAT flows uncompiled — the SS speedup still applies. Dynamic shapes are ON so
    # varying cond_len (single-image vs multi-image batches) doesn't recompile.
    compile_ss_flow: bool = field(default=True)
    compile_mode: str = field(default="default")   # default | reduce-overhead | max-autotune
    compile_dynamic: bool = field(default=True)
    # Elastic GC for SLAT flows — matches TRELLIS official slat_flow_*_512 training config.
    # A LinearMemoryController dynamically chooses how many blocks to checkpoint per step
    # so peak memory stays near target_ratio of GPU capacity. Cheaper than full GC, and
    # required to make BS>1 SLAT 512 fit on H100 (we saw OOM at BS=1 multi-asset w/o it).
    elastic_slat: bool = field(default=True)
    elastic_target_ratio: float = field(default=0.75)


def _enforce_no_offload(ds_cfg):
    """Repo policy: NO DeepSpeed CPU/NVMe offload. Offload masks real memory pressure
    by trading 3-4× step time for "fits". If your config doesn't fit, scale GPUs,
    cut batch, or freeze more — don't reach for offload. See OPTIMIZATIONS.md."""
    import json, pathlib
    if ds_cfg is None: return
    # HF Trainer accepts either a path (str / Path) or an already-parsed dict.
    if isinstance(ds_cfg, (str, pathlib.Path)):
        try:
            with open(ds_cfg) as f: cfg_dict = json.load(f)
        except Exception as e:
            raise RuntimeError(f"--deepspeed config unreadable: {ds_cfg}: {e}")
    elif isinstance(ds_cfg, dict):
        cfg_dict = ds_cfg
    else:
        return
    zo = (cfg_dict.get("zero_optimization") or {})
    off_opt = zo.get("offload_optimizer") or {}
    off_par = zo.get("offload_param") or {}
    bad = []
    if isinstance(off_opt, dict) and str(off_opt.get("device", "none")).lower() != "none":
        bad.append(f"offload_optimizer.device={off_opt.get('device')!r}")
    if isinstance(off_par, dict) and str(off_par.get("device", "none")).lower() != "none":
        bad.append(f"offload_param.device={off_par.get('device')!r}")
    if bad:
        raise RuntimeError(
            "REPO POLICY: NO OFFLOAD. The DeepSpeed config you passed enables: "
            f"{', '.join(bad)}. Offload trades 3-4× step time for 'fits' and "
            "masks real memory pressure. Use configs/deepspeed_zero2.json (no "
            "offload) and scale GPUs / cut batch / freeze more if it OOMs. "
            "See OPTIMIZATIONS.md §'No-offload policy'."
        )


def main():
    parser = HfArgumentParser((NativeArgs, TrainingArguments))
    native_args, training_args = parser.parse_args_into_dataclasses()
    # Enforce the no-offload policy BEFORE any deepspeed init happens.
    _enforce_no_offload(getattr(training_args, "deepspeed", None))
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

    # torch.compile must come AFTER _apply_flow_freeze (so Dynamo sees the final
    # requires_grad layout) and BEFORE Trainer wraps the model with DeepSpeed/FSDP.
    # SLAT flows are deliberately skipped — sparse + custom triton breaks Dynamo.
    if native_args.compile_ss_flow:
        model.ss_flow = torch.compile(
            model.ss_flow,
            mode=native_args.compile_mode,
            dynamic=native_args.compile_dynamic,
            fullgraph=False,   # cross_attn cond_mask path has data-dep branches
        )
        print(f"[train_native] torch.compile(ss_flow): mode={native_args.compile_mode!r} "
              f"dynamic={native_args.compile_dynamic}  (SLAT flows NOT compiled)")

    # Sync to TRELLIS official SLAT 512 config — register a LinearMemoryController so
    # each SLAT forward dynamically picks how many of the 30 blocks to checkpoint based
    # on observed memory pressure (target_ratio=0.75). Drop-in: SLatFlowModel has the
    # same body as ElasticSLatFlowModel, just missing the elastic mixin — rebind class.
    # NOTE: One controller PER SLAT model. Sharing one across shape+tex breaks because
    # the controller asserts `_last_input_size` matches across calls inside one forward;
    # shape and tex voxel counts can differ → ValueError.
    if native_args.elastic_slat and cfg.build_slat:
        from trellis2.models.structured_latent_flow import ElasticSLatFlowModel
        from trellis2.utils.elastic_utils import LinearMemoryController
        for name in ("shape_slat_512", "tex_slat_512"):
            m = getattr(model, name, None)
            if m is None:
                continue
            m.__class__ = ElasticSLatFlowModel       # add the elastic forward path
            m._memory_controller = None              # init mixin state (we bypassed __init__)
            controller = LinearMemoryController(
                buffer_size=1000, update_every=500,
                target_ratio=native_args.elastic_target_ratio,
                max_mem_ratio_start=0.5,
            )
            m.register_memory_controller(controller)
        print(f"[train_native] elastic SLAT GC ON  target_ratio={native_args.elastic_target_ratio} "
              f"max_mem_ratio_start=0.5  (one controller per SLAT, matches TRELLIS official)")

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
            max_slat_tokens=native_args.max_slat_tokens,
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
