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

import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import time

import torch
import transformers
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback


class EMACallback(TrainerCallback):
    """Exponential moving average of the TRAINABLE weights — matches TRELLIS official
    (ema_rate=0.9999; the released TRELLIS ckpts ARE EMA weights, and they SAMPLE from EMA).
    For multi-asset training EMA is the big lever: it averages out the per-batch gradient
    swing across diverse shapes, which raw weights at low step counts can't (→ collapsed geom).

    Saves the EMA shadow as a SEPARATE `ema.safetensors` per checkpoint (rank-0 only) instead
    of swapping params at save time — far less fragile under DeepSpeed's consolidated save.
    Inference overlays it on the base state dict (see tests/test_native_infer.py USE_EMA)."""

    def __init__(self, decay: float = 0.9999):
        self.decay = decay
        self.shadow = None      # {param_name: fp32 tensor on device}
        self._base = None

    @staticmethod
    def _unwrap(m):
        for _ in range(4):
            inner = getattr(m, "module", None)
            if inner is None:
                break
            m = inner
        return m

    def _trainable(self):
        return [(n, p) for n, p in self._base.named_parameters() if p.requires_grad]

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._base = self._unwrap(model)
        self.shadow = {n: p.detach().clone().float() for n, p in self._trainable()}
        print(f"[EMA] tracking {len(self.shadow)} trainable tensors, decay={self.decay}")

    def on_step_end(self, args, state, control, model=None, **kw):
        if self.shadow is None:
            return
        d = self.decay
        with torch.no_grad():
            for n, p in self._trainable():
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def on_save(self, args, state, control, **kw):
        if self.shadow is None or not state.is_world_process_zero:
            return
        from safetensors.torch import save_file
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt):
            save_file({k: v.detach().cpu() for k, v in self.shadow.items()},
                      os.path.join(ckpt, "ema.safetensors"))
            print(f"[EMA] wrote ema.safetensors → {ckpt}")

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
            wandb.define_metric("train/dino/align_loss", summary="last")
            wandb.define_metric("train/dual/gate_abs", summary="last")   # mean |gate| — anchor-lean
        except Exception as e:
            print(f"[WandbFineGrainedCallback] define_metric skipped: {e}")


def _apply_flow_freeze(model, mode: str):
    """Partial-FT the 3 TRELLIS flows. connector stays trainable; VLM per cfg.freeze_vlm.
    mode: 'full' | 'none' (connector-only — V3 Stage-1) | 'crossattn' |
    'crossattn,selfattn' | 'last{NN}' (last NN% of blocks, incl MLP)."""
    import re
    parts = [p.strip() for p in mode.split(",") if p.strip()]
    tags = ("ss_flow", "shape_slat_512", "tex_slat_512")
    def is_flow(n): return any(t in n for t in tags)
    if mode == "full" or not parts:
        return  # flows already fully trainable
    if mode == "none":   # V3 Stage-1: flows fully FROZEN; only connector (+VLM per freeze_vlm) trains
        for n, p in model.named_parameters():
            if is_flow(n):
                p.requires_grad_(False)
        return
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
    # Dual-branch params are NEW (Know3D Qwen cross-attn + zero-init gate + per-view embed) and
    # must train regardless of flow_tune mode — else e.g. last{NN} would freeze the Qwen branch
    # in early blocks (frozen by block index), leaving it stuck at zero-gate = dead. The original
    # DINOv3 anchor cross-attn (now under '...blocks.i.block.cross_attn') is still governed by the
    # normal partial-FT rules above.
    for n, p in model.named_parameters():
        if ("cross_attn_qwen" in n) or n.endswith(".gate") or ("norm_q" in n) or ("view_embed" in n):
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

        # Accumulate THIS rank's per-stage flow losses over the logging window. We do NOT
        # self.log() here (that logs at a wandb step behind Trainer's own logging step → wandb
        # drops it). Instead log() below reduces the window + all-GPUs and merges into the same
        # log call as train/loss, so per_stage/* shares train/loss's window-mean + cross-GPU
        # mean semantics exactly (not a single rank-0 sample).
        diag = getattr(base, "_last_diag", None)
        if diag:
            if not hasattr(self, "_stage_sum"):
                self._stage_sum, self._stage_n = {}, 0
            for k, v in diag.items():
                self._stage_sum[k] = self._stage_sum.get(k, 0.0) + v
            self._stage_n += 1

        return (loss, model_out) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # Merge windowed, cross-GPU-averaged per-stage losses into the train-loss log call so
        # they ride the SAME monotonic wandb step (no drops) and match train/loss semantics.
        if getattr(self, "_stage_n", 0) > 0 and "loss" in logs:
            keys = sorted(self._stage_sum)
            vec = torch.tensor([self._stage_sum[k] for k in keys],
                               dtype=torch.float32, device=self.args.device)
            n = self._stage_n
            world = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(vec, op=torch.distributed.ReduceOp.SUM)
                world = torch.distributed.get_world_size()
            vec = vec / (n * world)   # mean over (window steps × GPUs), matching train/loss
            for i, k in enumerate(keys):
                logs.setdefault(k, float(vec[i]))
            self._stage_sum, self._stage_n = {}, 0
        return super().log(logs, *args, **kwargs)


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
    # Randomly sample the cond view(s) per step from the asset's 16 renders (TRELLIS-style
    # viewpoint augmentation → view-robust). Default True for training; inference scripts
    # build the dataset without this so they stay deterministic on view 000.
    random_cond_view: bool = field(default=True)
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
    # EMA of trainable weights (TRELLIS official ema_rate=0.9999). 0 disables. Saved as a
    # separate ema.safetensors per checkpoint; inference overlays it (USE_EMA=1).
    ema_decay: float = field(default=0.9999)
    # ── DINOv3 conditioning alignment (REPA-inspired; see DINO_ALIGNMENT_DESIGN.md) ──
    # OFF by default. Distills DINOv3(render) into the connector via a train-only head;
    # image tasks only (text→3D contributes 0). Targets floaters/rough local geometry.
    dino_align: bool = field(default=False)
    dino_align_weight: float = field(default=0.5)      # λ; REPA default
    dino_align_mode: str = field(default="spatial")    # "spatial" (patch-wise) | "pooled"
    # Dual-branch conditioning (Know3D-style; see DUAL_COND_DESIGN.md). Keeps the original
    # TRELLIS DINOv3 cross-attn as an anchor + adds a zero-init-gated Qwen cross-attn. Replaces
    # dino_align. Init from PRETRAINED TRELLIS (not a Qwen-drifted ckpt) — see the doc.
    dual_cond: bool = field(default=False)
    dual_cond_max_views: int = field(default=8)        # per-view embedding table size
    dual_slat_qwen_stride: int = field(default=2)      # Qwen on every-Nth SLAT block (2=half → fits 512)
    dual_qwen_last_frac: float = field(default=0.0)     # >0: inject Qwen ONLY on last frac of blocks (0.2=last20%) → fits 16-GPU; overrides stride
    anchor_drop_prob: float = field(default=0.0)        # v2: prob drop ANCHOR-only (keep Qwen) → forces Qwen learning (text→3D)
    cfg_joint_drop_prob: float = field(default=0.0)      # v2: prob JOINT-drop anchor+Qwen (clean uncond) → TRELLIS-aligned CFG (image seed-stability)
    dual_anchor_pool: int = field(default=1)           # fixed avg-pool DINOv3 anchor grid by N
    dual_anchor_token_budget: int = field(default=0)   # >0: adaptive cap on TOTAL anchor tokens
    dual_ss_checkpoint: bool = field(default=False)    # gradient-checkpoint whole SS block (needs compile off)
    # ── V3 condition-swap distillation (docs/V3_DISTILL_DESIGN.md) ──
    # Teacher = the SAME frozen flow + single-view DINOv3 cond on the SAME (x_t,t);
    # student = connector(Qwen). L = L_flow + λ_v·‖v_s−v_t‖² + λ_f·Σ_l relMSE(block_l).
    # Stage-1 = I1-only data + connector-only (flow frozen). OFF by default.
    distill_dino: bool = field(default=False)
    distill_v_weight: float = field(default=1.0)       # λ_v output-level KD
    distill_f_weight: float = field(default=0.5)       # λ_f feature-level KD (relative MSE, O(1))
    distill_f_blocks: str = field(default="auto5")     # "autoK" evenly-spaced inner blocks | "3,9,15"
    # Stage-1.5 CFG-AWARE KD: >0 → kd_v matches the GUIDED velocity v_u+s(v_c−v_u), s~U[lo,hi].
    # Fixes the ×s amplification 'sand' (inference uses CFG; raw-v KD leaves the guided
    # quantity unsupervised). null_grad=False = lite (student null pass no_grad, ≈+40% step).
    distill_cfg_lo: float = field(default=3.0)
    distill_cfg_hi: float = field(default=0.0)
    distill_cfg_null_grad: bool = field(default=False)
    # V3 token raise: upscale each cond view so its Qwen vision tokens reach this count
    # (1024 = 32×32 grid, 1:1 with the DINOv3 teacher grid @512px). 0 = off (today's 256).
    # Set process-wide via vlm_collate → applies to ALL 3D task collates uniformly.
    target_tokens_per_view: int = field(default=0)
    # ── VLM-hidden cache + stage-split (vlm_cache.py / docs Stage-2 infra) ──
    # build_vlm=False: skip loading the 2B VLM (conds arrive precomputed via the
    # dataset's cached_hidden_root — set THAT in the mixture yaml task args).
    build_vlm: bool = field(default=True)
    # Train only one cascade component per job (TRELLIS-style split): all|ss|shape|tex.
    # Stages are GT-decoupled in training; each split job trains its own connector copy.
    train_stages: str = field(default="all")
    # fusion: cond = [raw DINOv3 tokens (cached d-keys); connector(Qwen)] — single cross-attn.
    fuse_dino: bool = field(default=False)
    dino_drop_prob: float = field(default=0.1)
    # Warm-start connector+flow from a prior run's checkpoint dir (loads model.safetensors,
    # strict=False, NO optimizer/step resume). For adding the dino head on trained weights.
    init_from_checkpoint: str = field(default="")
    # ── Mixture mode (Phase-1/2 multi-task infra) ──
    # When set, replaces --data_path. Wires up MixtureIterableDataset +
    # MultiTaskCollator from yaml (see configs/mix_*.yaml). Task weights and
    # per-task args live in the yaml; per-task class lives in
    # trellis2_blip3o/data/tasks/. Adding a new task = new file + new yaml row.
    mixture_config: Optional[str] = field(default=None)
    mixture_seed: int = field(default=0)
    # MDS / MosaicML-Streaming source (scripts/build_mds.py shards on local NVMe). When set,
    # use StreamingImageTo3D (node/rank-aware sharding + deterministic resume) instead of the
    # manifest+mixture path. Single-task image_to_3d (S1/S2 I1 fusion).
    mds_root: Optional[str] = field(default=None)
    mds_cache_limit: Optional[str] = field(default=None)   # e.g. "600gb" to bound local cache

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
        dino_align=native_args.dino_align,
        dino_align_weight=native_args.dino_align_weight,
        dino_align_mode=native_args.dino_align_mode,
        dual_cond=native_args.dual_cond,
        dual_cond_max_views=native_args.dual_cond_max_views,
        dual_slat_qwen_stride=native_args.dual_slat_qwen_stride,
        dual_qwen_last_frac=native_args.dual_qwen_last_frac,
        anchor_drop_prob=native_args.anchor_drop_prob,
        cfg_joint_drop_prob=native_args.cfg_joint_drop_prob,
        dual_anchor_pool=native_args.dual_anchor_pool,
        dual_anchor_token_budget=native_args.dual_anchor_token_budget,
        dual_ss_checkpoint=native_args.dual_ss_checkpoint,
        distill_dino=native_args.distill_dino,
        distill_v_weight=native_args.distill_v_weight,
        distill_f_weight=native_args.distill_f_weight,
        distill_f_blocks=native_args.distill_f_blocks,
        distill_cfg_lo=native_args.distill_cfg_lo,
        distill_cfg_hi=native_args.distill_cfg_hi,
        distill_cfg_null_grad=native_args.distill_cfg_null_grad,
        # vision-token contract → saved in ckpt config.json so inference auto-matches
        target_tokens_per_view=native_args.target_tokens_per_view,
        build_vlm=native_args.build_vlm,
        train_stages=native_args.train_stages,
        fuse_dino=native_args.fuse_dino,
        dino_drop_prob=native_args.dino_drop_prob,
    )
    model = TrellisNativeVLMForConditionalGeneration(cfg)
    _apply_flow_freeze(model, native_args.flow_tune)
    print(f"[train_native] flow_tune={native_args.flow_tune!r}")
    # V3 token raise: one process-wide knob, consumed by vlm_collate (all 3D collates).
    if native_args.target_tokens_per_view:
        from trellis2_blip3o.vlm_collate import set_default_target_tokens_per_view
        set_default_target_tokens_per_view(native_args.target_tokens_per_view)
        print(f"[train_native] target_tokens_per_view={native_args.target_tokens_per_view} "
              f"(views upscaled to a {int(native_args.target_tokens_per_view ** 0.5)}²-token grid)")

    # --init_from_checkpoint: warm-start connector+flow weights from a prior run's
    # model.safetensors, but DO NOT resume optimizer/step/EMA. Use this to add a NEW
    # module (e.g. dino_align's projection head) on top of trained weights and start a
    # fresh short run — resuming can't add params (optimizer-state mismatch). strict=False
    # loads matching keys (connector/flow/vlm), leaves the new head at init. Strips a
    # torch.compile '_orig_mod.' prefix if present (see project_compile_origmod_save_bug).
    if native_args.init_from_checkpoint:
        from safetensors.torch import load_file as _load_sft
        import os as _os
        sd_path = _os.path.join(native_args.init_from_checkpoint, "model.safetensors")
        sd = _load_sft(sd_path)
        if any("_orig_mod." in k for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        new_keys = [k for k in missing if "dino_aligner" in k]
        print(f"[train_native] init_from_checkpoint={sd_path}: loaded "
              f"{len(sd)} tensors; missing={len(missing)} (new head: {len(new_keys)}), "
              f"unexpected={len(unexpected)}")

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
    per_dev_bs = max(1, int(training_args.per_device_train_batch_size))
    if native_args.mds_root:
        # MDS path: StreamingDataset does its own node/rank-aware sharding + resume.
        from trellis2_blip3o.data.streaming_task import StreamingImageTo3D
        from trellis2_blip3o.data.mixture import MultiTaskCollator
        train_ds = StreamingImageTo3D(
            native_args.mds_root,
            fuse_dino=native_args.fuse_dino,
            ss_only=native_args.ss_only,
            max_slat_tokens=native_args.max_slat_tokens,
            shuffle=True, batch_size=per_dev_bs,
            cache_limit=native_args.mds_cache_limit,
        )
        collator = MultiTaskCollator(processor=processor)
        print(f"[train_native] MDS source {native_args.mds_root} "
              f"({len(train_ds.ds)} samples, fuse_dino={native_args.fuse_dino}, ss_only={native_args.ss_only})")
        if not training_args.max_steps or training_args.max_steps <= 0:
            raise ValueError("--mds_root uses IterableDataset; --max_steps must be set.")
        # StreamingDataset is already rank-aware → each rank iterates its own loader (same as
        # the mixture path; accelerate must NOT dispatch-from-rank0-and-broadcast).
        try:
            training_args.accelerator_config.dispatch_batches = False
        except Exception:
            training_args.dispatch_batches = False
    elif native_args.mixture_config:
        from trellis2_blip3o.data import build_mixture
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
        # CRITICAL: each rank must iterate its OWN dataloader. MixtureIterableDataset
        # is already rank-aware (rank-synced task choice + per-rank within-task index),
        # so accelerate must NOT dispatch-from-rank0-and-broadcast. Besides defeating
        # the per-rank data-parallel sampling, broadcasting the batch fails outright on
        # the `_task` str field (accelerate's _gpu_broadcast_one only handles tensors).
        try:
            training_args.accelerator_config.dispatch_batches = False
        except Exception:
            # Older HF: the flag lives directly on TrainingArguments.
            training_args.dispatch_batches = False
    else:
        data_args = SimpleNamespace(
            use_codebook=False, num_views=1,
            num_cond_views=native_args.num_cond_views,           # legacy
            random_cond_view=native_args.random_cond_view,       # viewpoint augmentation
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

    callbacks = [WandbFineGrainedCallback()]
    if native_args.ema_decay and native_args.ema_decay > 0:
        callbacks.append(EMACallback(decay=native_args.ema_decay))
    trainer = NativeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=bool(list(__import__("pathlib").Path(training_args.output_dir).glob("checkpoint-*"))) or None)
    trainer.save_state()


if __name__ == "__main__":
    main()
