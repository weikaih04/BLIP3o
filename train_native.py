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

import numpy as np
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
        self._n = 0             # EMA updates so far — drives the warmup below

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
        if not state.is_world_process_zero:
            # on_save writes from rank 0 only, so every other rank was keeping an
            # fp32 shadow of the whole model for nothing: 802.7M x 4B = 3.2 GB of
            # GPU memory per rank, 31 ranks of it on a 4-node run.
            self.shadow = None
            return
        self.shadow = {n: p.detach().clone().float() for n, p in self._trainable()}
        print(f"[EMA] tracking {len(self.shadow)} trainable tensors, decay={self.decay}")

    def _decay_at(self, n: int) -> float:
        """Warmed-up decay: min(decay, (1+n)/(10+n)).

        A bare 0.9999 has a time constant of 10k updates, so on a 10k-step run the
        shadow is still 0.9999**10000 = 36.8% the RANDOM INIT when it is written
        out — v7's ema.safetensors is that. Official uses 0.9999 too, but over 1M
        steps, where the init leaks e^-100. The warmup makes the shadow track the
        model from update 0 and reach the nominal decay by ~n=1e5, so the same
        number stays correct at both run lengths."""
        return min(self.decay, (1.0 + n) / (10.0 + n))

    def on_step_end(self, args, state, control, model=None, **kw):
        if self.shadow is None:
            return
        d = self._decay_at(self._n)
        self._n += 1
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

class AdaptiveGradClipCallback(TrainerCallback):
    """TRELLIS.2's AdaptiveGradClipper (utils/grad_clip_utils.py:7-81), wired into
    the DeepSpeed path.

    The released 512 shape/tex configs both clip with
    AdaptiveGradClipper(max_norm=1.0, clip_percentile=95): a rolling 1000-step
    buffer of pre-clip norms, after which the threshold becomes
    min(p95(buffer), 1.0). By construction that clips about 5% of steps. Our flat
    1.0 clipped 0.6% of v7's steps, and the p95 of its last 1000 was ~0.44 — so
    the released recipe clips considerably HARDER than we were, and this is the
    one trainer knob that was still off.

    DeepSpeed owns the clip under bf16 + ZeRO-1 (bf16_optimizer.py:311 reads
    self.clip_grad fresh on every step), so the threshold is written there rather
    than by calling clip_grad_norm_ ourselves, and the norm read back is the
    PRE-clip one DeepSpeed stashes at :308 — the same quantity the official
    buffer stores."""

    def __init__(self, max_norm: float = 1.0, percentile: float = 95.0,
                 buffer_size: int = 1000):
        self.max_norm, self.percentile, self.buffer_size = max_norm, percentile, buffer_size
        self._buf = np.zeros(buffer_size, dtype=np.float32)
        self._ptr = self._len = 0
        self._engine = None
        self._cur = max_norm

    def state_dict(self):
        return {"buf": self._buf, "ptr": self._ptr, "len": self._len, "cur": self._cur}

    def load_state_dict(self, sd):
        self._buf, self._ptr = sd["buf"], sd["ptr"]
        self._len, self._cur = sd["len"], sd["cur"]

    def _opt(self):
        e = self._engine
        return getattr(e, "optimizer", None) if e is not None else None

    def on_train_begin(self, args, state, control, **kw):
        if self._engine is None:
            print("[gradclip] no DeepSpeed engine attached — flat clipping stays in force")
        else:
            print(f"[gradclip] adaptive: p{self.percentile:.0f} of the last "
                  f"{self.buffer_size} steps, capped at {self.max_norm}")

    def on_step_end(self, args, state, control, **kw):
        e = self._engine
        if e is None or not hasattr(e, "get_global_grad_norm"):
            return
        gn = e.get_global_grad_norm()
        if gn is None:
            return
        gn = float(gn)
        if not np.isfinite(gn):
            return                                  # official skips non-finite too
        self._buf[self._ptr] = gn
        self._ptr = (self._ptr + 1) % self.buffer_size
        self._len = min(self._len + 1, self.buffer_size)
        if self._len < self.buffer_size:
            return                                  # warm-up: flat max_norm, as upstream
        self._cur = min(float(np.percentile(self._buf, self.percentile)), self.max_norm)
        opt = self._opt()
        if opt is not None and hasattr(opt, "clip_grad"):
            opt.clip_grad = self._cur
        if state.global_step % 500 == 0 and state.is_world_process_zero:
            print(f"[gradclip] step {state.global_step}: threshold {self._cur:.4f}", flush=True)

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




def _pack_live_conds(conds, dev):
    """Per-sample encoder output -> the padded batch tensors the model expects.

    Mirrors vlm_collate's cached branch, including the two things that are NOT uniform
    across tasks and cost a 32-GPU crash to learn:
      * T carries NO DINO at all (no image), so the dino pack must be conditional —
        flow_heads takes its non-fusion branch for text and a dino segment there would be
        a condition the cached run never had.
      * IM carries qwen_view_ids, padded with -1 (never 0, which is a real view ordinal).

    collate_cached builds its mask with a bare torch.zeros, i.e. ALWAYS on CPU; on the
    cached path _prepare_inputs moves it afterwards, and here nothing would.
    """
    from trellis2_blip3o.vlm_cache import collate_cached
    q = collate_cached(conds)
    out = {"cond_hidden": q["cond_hidden"].to(dev),
           "cond_keep_mask": q["cond_keep_mask"].to(dev)}
    if "dino_hidden" in conds[0]:
        d = collate_cached([{"cond_hidden": c["dino_hidden"],
                             "cond_keep_mask": c["dino_keep_mask"]} for c in conds])
        out["dino_hidden"] = d["cond_hidden"].to(dev)
        out["dino_keep_mask"] = d["cond_keep_mask"].to(dev)
        T = d["cond_keep_mask"].shape[1]
        vid = torch.zeros(len(conds), T, dtype=torch.long)
        for i, c in enumerate(conds):
            v = c.get("dino_view_ids")
            if v is not None:
                vid[i, :v.shape[0]] = v.cpu()
        out["dino_view_ids"] = vid.to(dev)
    if "qwen_view_ids" in conds[0]:
        Tq = out["cond_hidden"].shape[1]
        qv = torch.full((len(conds), Tq), -1, dtype=torch.long)
        for i, c in enumerate(conds):
            v = c["qwen_view_ids"]
            qv[i, :v.shape[0]] = v.cpu()
        out["qwen_view_ids"] = qv.to(dev)
    return out


def _mixture_uses_live_cond(trainer) -> bool:
    """True when any task in the mixture was built with live_cond=True.

    Read off the DATASETS, not the CLI: live_cond is a per-task yaml arg (mixture.py passes
    only `s.args` to the task constructor), so there is no flag on the command line to
    check and a --live_cond would be silently dropped like --max_slat_tokens is.
    """
    ds = getattr(trainer, "train_dataset", None)
    return any(getattr(t, "live_cond", False) for t in getattr(ds, "tasks", []) or [])


class _LiveCondPrefetch:
    """One-batch-deep prefetch that makes the live-cond encode overlap the training step.

    The encode is ~382 ms of GPU work per batch (bs8) against a ~6.5 s step. Run inline in
    _prepare_inputs it is 5.9% of wall clock, serialized in front of the forward. Run here
    it is enqueued on a side stream for batch N+1 BEFORE batch N is yielded, so it executes
    while the main stream is busy with step N.

    ORDER MATTERS AND IS EASY TO GET WRONG. Creating a side stream inside _prepare_inputs
    and joining it two lines later overlaps NOTHING — the main stream has no work queued at
    that moment, so it just adds two syncs. The encode has to be launched a full iteration
    ahead of the step it hides behind, which is why this is a dataloader wrapper and not a
    trainer hook.

    This does NOT reduce GPU work; the encode and the step contend for SMs. What it
    recovers is the step's idle SM time (ZeRO allreduce, low-occupancy sparse kernels), so
    the win is real but bounded — measure it, do not assume it is free.
    """

    def __init__(self, dl, enc, device):
        self.dl, self.enc, self.device = dl, enc, device
        self.stream = torch.cuda.Stream(device=device)

    def __len__(self):
        return len(self.dl)

    def __getattr__(self, k):                       # dataset / sampler / set_epoch / ...
        return getattr(self.__dict__["dl"], k)

    def _launch(self, batch):
        """Enqueue this batch's encode on the side stream. Returns (batch, event)."""
        if batch is None:
            return None
        preps = batch.pop("_live_prep", None) if isinstance(batch, dict) else None
        if preps is None:
            return (batch, None)
        cur = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(cur)                # side stream must see prior main work
        with torch.cuda.stream(self.stream):
            conds = self.enc.encode(preps)
            packed = _pack_live_conds(conds, self.device)
        # The allocator frees a tensor when ITS stream is done with it; these are consumed
        # on the main stream, so tell it that too or the blocks can be recycled early.
        for v in packed.values():
            v.record_stream(cur)
        batch.update(packed)
        ev = torch.cuda.Event()
        ev.record(self.stream)
        return (batch, ev)

    def __iter__(self):
        it = iter(self.dl)
        cur = self._launch(next(it, None))
        while cur is not None:
            batch, ev = cur
            if ev is not None:
                torch.cuda.current_stream(self.device).wait_event(ev)   # GPU-side, no host stall
            nxt_raw = next(it, None)
            nxt = self._launch(nxt_raw)             # <- N+1 goes on the wire BEFORE N is yielded
            yield batch
            cur = nxt


class NativeTrainer(Trainer):
    """Trainer that (1) moves trellis2 SparseTensor SLAT targets to the device — vanilla
    Trainer._prepare_inputs only moves torch.Tensors, leaving SparseTensors on CPU and
    crashing the cascade — and (2) emits per-stage / cond / voxel-count diagnostics from
    model.forward into Trainer.log → wandb."""

    # LIVE conditioning (live_cond: true in the mixture yaml): the dataloader workers ship
    # the CPU half of the Qwen/DINO preprocessing and the two towers run HERE, once per
    # batch, SERIALLY in front of the step's forward. Measured 8 x 54.8 ms = 0.44 s against
    # a 6.5 s step, i.e. a ~6.7% tax.
    #
    # It is serial ON PURPOSE for now. Hiding it needs a one-batch-deep prefetch that
    # enqueues batch N+1's encode on a side stream BEFORE yielding batch N (so the encode
    # overlaps step N's own kernels) — a side stream created and joined inside this method
    # overlaps nothing, it just adds two syncs. That wrapper belongs on get_train_dataloader
    # and has to hand tensors across streams with record_stream; not worth the allocator
    # risk until 6.7% is the thing standing in the way.
    _live_enc = None

    def _live_encoder(self):
        if self._live_enc is None:
            from trellis2_blip3o.live_cond_batch import TrainCondEncoder
            self._live_enc = TrainCondEncoder(device=str(self.args.device))
            self._live_enc.warmup(int(self.args.per_device_train_batch_size))
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[live_cond] cond VLM = {self._live_enc.vlm_path}", flush=True)
                print(f"[live_cond] encoder resident on {self.args.device} "
                      f"({torch.cuda.memory_allocated(self.args.device)/2**30:.1f} GB)",
                      flush=True)
        return self._live_enc

    def get_train_dataloader(self):
        """Wrap the prepared dataloader so the live-cond encode runs a batch ahead.

        LIVE_COND_PREFETCH=0 falls back to encoding inline in _prepare_inputs — same
        numbers, ~5.9% slower, and the one to use if a stream/allocator problem is ever
        suspected. Non-live runs get the dataloader untouched.
        """
        dl = super().get_train_dataloader()
        if not _mixture_uses_live_cond(self):
            return dl
        if os.environ.get("LIVE_COND_PREFETCH") == "0":
            print("[live_cond] prefetch DISABLED — encoding inline in _prepare_inputs",
                  flush=True)
            return dl
        print("[live_cond] prefetch ON — batch N+1 encodes on a side stream during step N",
              flush=True)
        return _LiveCondPrefetch(dl, self._live_encoder(), self.args.device)

    def _encode_live(self, prepared):
        """Inline fallback — only reached when the prefetch wrapper is disabled."""
        preps = prepared.pop("_live_prep", None)
        if preps is None:
            return prepared
        _prof = os.environ.get("LIVE_COND_PROF") == "1"
        if _prof:
            torch.cuda.synchronize(); _t0 = __import__("time").perf_counter()
            _gap = _t0 - getattr(self, "_live_tprev", _t0)
        prepared.update(_pack_live_conds(self._live_encoder().encode(preps),
                                         self.args.device))
        if _prof:
            torch.cuda.synchronize(); _t1 = __import__("time").perf_counter()
            self._live_n = getattr(self, "_live_n", 0) + 1
            print(f"[live_cond_prof] step {self._live_n:3d}  gap(fetch+step) {_gap*1e3:7.1f} ms"
                  f"  encode {(_t1-_t0)*1e3:7.1f} ms ({(_t1-_t0)*1e3/len(preps):5.1f}/img)",
                  flush=True)
            self._live_tprev = __import__("time").perf_counter()
        return prepared

    def _prepare_inputs(self, inputs):
        live = inputs.pop("_live_prep", None)   # a list of dicts — Trainer would choke
        prepared = super()._prepare_inputs(inputs)
        for k, v in list(prepared.items()):
            if hasattr(v, "feats") and hasattr(v, "to"):  # SparseTensor
                prepared[k] = v.to(self.args.device)
        if live is not None:
            prepared["_live_prep"] = live
            prepared = self._encode_live(prepared)
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
        # cross-attn anneal: drive the unified model's xattn_scale 1 -> 0 over
        # [start, end] steps so the run ENDS as a pure three-stream MMDiT (one
        # conditioning pathway) while STARTING bit-exact with the warm start.
        _uni = getattr(base, "unified_geotex", None)
        if _uni is not None and getattr(_uni, "cond_mode", "") == "stream":
            _a = int(getattr(base.config, "geotex_xattn_anneal_start", 0))
            _b = int(getattr(base.config, "geotex_xattn_anneal_end", 0))
            if _b > _a:
                _st = int(self.state.global_step)
                _uni.xattn_scale = float(
                    1.0 if _st <= _a else 0.0 if _st >= _b else 1.0 - (_st - _a) / (_b - _a))
        ctxs = []
        for name in ("shape_slat_512", "tex_slat_512", "unified_geotex"):
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

        # Per-TASK loss (multitask mixture runs). Mixture batches are homogeneous
        # (MixtureIterableDataset granularity="batch") AND rank-synced, so `_task` is a
        # single label for the whole global batch — accumulating per label over the logging
        # window gives an exact per-task loss curve. Kept as a device tensor (no .item())
        # so this adds no extra GPU sync per micro-step. No-op for single-task runs
        # (one key → identical numbers to train/loss).
        t = inputs.get("_task") if isinstance(inputs, dict) else None
        # S3_TASK_DEBUG=1 → one line per micro-batch per rank, so a 2-rank run can be
        # diffed to PROVE the mixture's task draw is rank-synced. (Ranks that disagree
        # would train mismatched used-parameter sets → DDP hang / silent grad corruption.)
        if os.environ.get("S3_TASK_DEBUG") == "1":
            self._dbg_i = getattr(self, "_dbg_i", 0) + 1
            print(f"[taskdbg] rank={os.environ.get('RANK','0')} micro={self._dbg_i} "
                  f"task={t} cond={tuple(inputs['cond_hidden'].shape)} "
                  f"dino={tuple(inputs['dino_hidden'].shape) if 'dino_hidden' in inputs else None}",
                  flush=True)
        if t in self._TASK_IDX:
            if not hasattr(self, "_task_sum"):
                self._task_sum = [None] * len(self._TASK_NAMES)
                self._task_n = [0] * len(self._TASK_NAMES)
            i = self._TASK_IDX[t]
            d = loss.detach().float()
            self._task_sum[i] = d if self._task_sum[i] is None else self._task_sum[i] + d
            self._task_n[i] += 1

        return (loss, model_out) if return_outputs else loss

    # Fixed order so the cross-rank all_reduce below has an identical-length vector on
    # every rank even if a rank happened to see no batch of some task in the window.
    _TASK_NAMES = ("image_to_3d", "multi_image_to_3d", "text_to_3d")
    _TASK_IDX = {n: i for i, n in enumerate(_TASK_NAMES)}

    def _merge_task_logs(self, logs):
        if not any(getattr(self, "_task_n", []) or []):
            return
        dev = self.args.device
        s = torch.stack([(v if v is not None else torch.zeros((), device=dev))
                         for v in self._task_sum]).float()
        n = torch.tensor(self._task_n, dtype=torch.float32, device=dev)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(s, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
        tot = float(n.sum())
        for i, name in enumerate(self._TASK_NAMES):
            if float(n[i]) > 0:
                logs.setdefault(f"train/task/{name}/loss", float(s[i] / n[i]))
            # running mix fraction — the live check that we are actually at 0.5/0.3/0.2
            logs.setdefault(f"train/task/{name}/frac", float(n[i]) / max(tot, 1.0))
        self._task_sum = [None] * len(self._TASK_NAMES)
        self._task_n = [0] * len(self._TASK_NAMES)

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            self._merge_task_logs(logs)
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
    # Train only one cascade component per job (TRELLIS-style split): all|ss|shape|tex,
    # or "geotex" = the unified geo-tex dual-stream DiT (docs/UNIFIED_GEOTEX_DIT_DESIGN.md).
    # Stages are GT-decoupled in training; each split job trains its own connector copy.
    train_stages: str = field(default="all")
    # ── geotex (unified) knobs — only read when train_stages="geotex" ──
    geotex_shape_init: str = field(default="")   # shape-specialist run ckpt (EMA overlaid)
    geotex_tex_init: str = field(default="")     # tex-specialist run ckpt (EMA overlaid)
    geotex_coupling: str = field(default="union")  # union (MF-bare, G0-cleared) | gated
    geotex_cond_mode: str = field(default="cross_attn")  # cross_attn (A) | stream (B: three-stream MMDiT)
    geotex_cond_stream_blocks: int = field(default=10)   # cond stream on the LAST N blocks; <=0 = all 30
    # cross-attn anneal window (steps). end>start enables it; the run finishes
    # with cross-attn at exactly 0 = a true single-pathway MMDiT.
    geotex_xattn_anneal_start: int = field(default=0)
    geotex_xattn_anneal_end: int = field(default=0)
    # FROM SCRATCH: three-stream sparse MMDiT (trellis2_blip3o/mmdit3d.py), no
    # TRELLIS.2 weights. shape_init/tex_init are ignored; the VAEs are unchanged.
    # Sizing is these four knobs only (defaults = the 298M pilot).
    geotex_from_scratch: bool = field(default=False)
    geotex_dim: int = field(default=768)
    geotex_heads: int = field(default=6)
    geotex_depth_double: int = field(default=8)    # triple-stream blocks (own weights)
    geotex_depth_single: int = field(default=16)   # shared-weight blocks (2x double, owner's 1:2)
    # released 512 config values (slat_flow_img2shape_dit_1_3B_512_bf16.json:13,16);
    # we shipped 4.0 + vanilla xavier by mistake, audited 2026-08-17
    geotex_mlp_ratio: float = field(default=5.3334)
    geotex_init: str = field(default="scaled")     # "scaled" (released) | "vanilla"
    geotex_bidir: bool = field(default=False)  # corner-masked bidirectional geo<->tex
    geotex_fused: bool = field(default=True)   # fused MMDiT attention (MFU 18.7->30.0%, G0-fused certified)
    geotex_gc: float = field(default=1.0)      # fraction of block pairs gradient-checkpointed (1=all, 0=none)
    # S2b — unfreeze geo (three-pack: geo loss + self-distill + G3 red line)
    geotex_unfreeze_geo: bool = field(default=False)
    # ── v10: the third tower. Setting geotex_ss_init is what turns v10 on. ──
    geotex_ss_init: str = field(default="")            # s3_ss run ckpt; "" = two-tower
    geotex_ss_loss_w: float = field(default=1.0)
    geotex_cond_seg_embed: bool = field(default=False)  # image/text segment code on cond
    geotex_cond_patch_pos: str = field(default="off")   # off | zero | dino_sig
    geotex_p_solo: float = field(default=0.20)         # SS-solo rows
    geotex_p_lag: float = field(default=0.20)          # lag-band rows
    geotex_k0_lo: int = field(default=3)               # t_ss<=t_s fails below 2
    geotex_k0_hi: int = field(default=11)              # k0=steps leaves no slat step
    geotex_geo_loss_w: float = field(default=1.0)
    geotex_p_corner: float = field(default=0.4)  # user 2026-08-11: flagship tex|mesh mass; 0.2 = A1
    geotex_p_corner2: float = field(default=0.2)  # t_x=1 corner (mesh-only marginal, bidir design)
    # A6 sampler arms (default OFF = the S1/S2b recipe, bit-exact):
    geotex_concat_cond: bool = field(default=False)  # cascade-legacy shape concat into tex
    geotex_pooled_cond: bool = field(default=True)   # SD3/FLUX pooled cond -> adaLN modulation
    geotex_mismatch_w: float = field(default=0.0)    # mismatched-image hinge (experiment)
    geotex_mismatch_margin: float = field(default=0.15)
    # ── dual-teacher output KD (s3_t50b specialists; all default OFF) ──
    # WARM path: replay cond_x's realized drops onto cond_s (both towers drop
    # together = inference's joint uncond). Legacy S1/S2b behavior = False.
    geotex_joint_cond_drop: bool = field(default=False)
    adaptive_grad_clip: bool = field(default=True)   # TRELLIS.2's p95 rolling clip
    # fusion: cond = [raw DINOv3 tokens (cached d-keys); connector(Qwen)] — single cross-attn.
    fuse_dino: bool = field(default=False)
    dino_drop_prob: float = field(default=0.1)
    qwen_drop_prob: float = field(default=0.0)  # mirror of dino_drop; blocks the flow's escape to qwen
    view_embed_mode: str = field(default="learned")  # "learned"(zero-init dve) | "sincos"(HY3D-mv fixed view enc)
    # ── REPA-style SS aux alignment (trellis2_blip3o/repa.py; official sihyun-yu/REPA) ──
    # repa_root = VGGT target cache root ({root}/{sha[:2]}/{sha}/vggt16.npz); "" = OFF.
    # Also exported as env REPA_ROOT so the streaming dataset loads the targets.
    # REQUIRES --compile_ss_flow False (forward hooks don't survive torch.compile) —
    # enforced below. CFG-dropped samples get aux weight 0 (deliberate deviation).
    repa_root: str = field(default="")
    repa_coeff: float = field(default=0.5)     # λ on the negative-cosine proj loss
    repa_depth: int = field(default=0)         # 1-indexed SS block tap; 0 = num_blocks//3
    repa_zdim: int = field(default=2049)       # target dim (density 1 + RAW 2048-d VGGT feat)
    # Targets below this builder quality score load as aux-weight-0 (gate finding:
    # thin scan-like shells → garbage VGGT clouds). Exported as env REPA_MIN_QUALITY.
    repa_min_quality: float = field(default=0.5)
    cond_pos_stamp: bool = field(default=False)  # DINO position signature on qwen cond (pos_stamp.py)
    cond_adapter: str = field(default="xf2")      # cond connector: "mlp" (3.1M) | "xf2" (i1 2-block xformer, ~27M).
    # DEFAULT=xf2 (weikaih 2026-07-15): kept for text-to-3D (i1's deep text-adapter finding — the
    # language→generation interface is the bottleneck there). NOTE: on image/qwen-only SS the xf2
    # ablation measured WORSE (IoU 0.229 vs MLP 0.256) — position is the bottleneck there, not the
    # interface, so image-conditioned runs (I1/IM fusion) should pass --cond_adapter mlp explicitly.
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
    # NODE-SHARDED MDS: comma-separated shard dirs THIS node holds locally (built by
    # scripts/stage_mds_v3.sh). When set, StreamingImageTo3D shards node-locally (each node's
    # owned shards split across its own ranks; disjoint shards across nodes tile the full set).
    # Overrides --mds_root. See scripts/stage_mds_v3.sh + s1_v3_launch.sh.
    mds_shards: Optional[str] = field(default=None)

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

    # REPA (SS aux alignment): forward hooks on the SS blocks are skipped inside
    # torch.compile'd regions, so repa requires the uncompiled SS flow. Fail EARLY
    # and loudly (compile_ss_flow defaults True — repa runs must pass it False, which
    # the split launcher does via COMPILE_SS=0). Export REPA_ROOT before the dataset
    # is built so streaming_task loads the targets (workers inherit the env).
    if native_args.repa_root:
        if native_args.compile_ss_flow:
            raise RuntimeError(
                "--repa_root requires --compile_ss_flow False (COMPILE_SS=0): the SS "
                "block forward hook that taps the REPA hidden is skipped inside "
                "torch.compile'd blocks. IM SS runs already use COMPILE_SS=0."
            )
        os.environ["REPA_ROOT"] = native_args.repa_root
        os.environ["REPA_MIN_QUALITY"] = str(native_args.repa_min_quality)
        print(f"[train_native] REPA on: root={native_args.repa_root} "
              f"coeff={native_args.repa_coeff} depth={native_args.repa_depth or 'auto'} "
              f"zdim={native_args.repa_zdim} min_quality={native_args.repa_min_quality}")

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
        geotex_shape_init=native_args.geotex_shape_init,
        geotex_tex_init=native_args.geotex_tex_init,
        geotex_coupling=native_args.geotex_coupling,
        geotex_cond_mode=native_args.geotex_cond_mode,
        geotex_cond_stream_blocks=native_args.geotex_cond_stream_blocks,
        geotex_xattn_anneal_start=native_args.geotex_xattn_anneal_start,
        geotex_xattn_anneal_end=native_args.geotex_xattn_anneal_end,
        geotex_from_scratch=native_args.geotex_from_scratch,
        geotex_dim=native_args.geotex_dim,
        geotex_heads=native_args.geotex_heads,
        geotex_depth_double=native_args.geotex_depth_double,
        geotex_depth_single=native_args.geotex_depth_single,
        geotex_mlp_ratio=native_args.geotex_mlp_ratio,
        geotex_init=native_args.geotex_init,
        geotex_bidir=native_args.geotex_bidir,
        geotex_fused=native_args.geotex_fused,
        geotex_gc=native_args.geotex_gc,
        geotex_unfreeze_geo=native_args.geotex_unfreeze_geo,
        geotex_ss_init=native_args.geotex_ss_init,
        geotex_ss_loss_w=native_args.geotex_ss_loss_w,
        geotex_cond_seg_embed=native_args.geotex_cond_seg_embed,
        geotex_cond_patch_pos=native_args.geotex_cond_patch_pos,
        geotex_p_solo=native_args.geotex_p_solo,
        geotex_p_lag=native_args.geotex_p_lag,
        geotex_k0_lo=native_args.geotex_k0_lo,
        geotex_k0_hi=native_args.geotex_k0_hi,
        geotex_geo_loss_w=native_args.geotex_geo_loss_w,
        geotex_p_corner=native_args.geotex_p_corner,
        geotex_p_corner2=native_args.geotex_p_corner2,
        geotex_concat_cond=native_args.geotex_concat_cond,
        geotex_pooled_cond=native_args.geotex_pooled_cond,
        geotex_mismatch_w=native_args.geotex_mismatch_w,
        geotex_joint_cond_drop=native_args.geotex_joint_cond_drop,
        geotex_mismatch_margin=native_args.geotex_mismatch_margin,
        fuse_dino=native_args.fuse_dino,
        dino_drop_prob=native_args.dino_drop_prob,
        qwen_drop_prob=native_args.qwen_drop_prob,
        view_embed_mode=native_args.view_embed_mode,
        cond_pos_stamp=native_args.cond_pos_stamp,
        cond_adapter=native_args.cond_adapter,
        repa_root=native_args.repa_root,
        repa_coeff=native_args.repa_coeff,
        repa_depth=native_args.repa_depth,
        repa_zdim=native_args.repa_zdim,
    )
    model = TrellisNativeVLMForConditionalGeneration(cfg)
    _apply_flow_freeze(model, native_args.flow_tune)
    print(f"[train_native] flow_tune={native_args.flow_tune!r}")

    if native_args.train_stages == "geotex":
        # geotex warm-start/freeze are owned by the model __init__ (two-ckpt EMA
        # assembly); the single-ckpt and compile/elastic paths don't apply.
        assert not native_args.init_from_checkpoint, \
            "[geotex] warm-start via --geotex_{shape,tex}_init, not --init_from_checkpoint"
        assert not native_args.compile_ss_flow, "[geotex] no SS flow to compile"
        # elastic GC: TRELLIS's OWN adaptive controller (audit 2026-08-12 — the
        # first version hard-coded a static fraction, throwing away the policy
        # that decides how many blocks to checkpoint from a fitted memory model).
        # --elastic_slat True enables it; --geotex_gc stays as the static
        # kill-switch. NativeTrainer enters the controller's record() context.
        if native_args.elastic_slat:
            from trellis2.utils.elastic_utils import LinearMemoryController
            model.unified_geotex.register_memory_controller(LinearMemoryController(
                buffer_size=1000, update_every=500,
                target_ratio=native_args.elastic_target_ratio,
                max_mem_ratio_start=0.5))
            print(f"[geotex] elastic GC ON (TRELLIS LinearMemoryController, "
                  f"target_ratio={native_args.elastic_target_ratio})")
        else:
            # Say it out loud. `elastic_slat` DEFAULTS TO TRUE (:842, and
            # scripts/train_native_geotex.sh's ELASTIC:-True), so a relaunch that
            # forgets ELASTIC=False silently changes the memory policy of a run
            # that was tuned without it — and the failure mode is not a slowdown:
            # mmdit3d.py:585-590 pins _ckpt_upto=0 in its finally, so once the
            # elastic context has run, static --geotex_gc is permanently overridden
            # OFF. A bad memory fit therefore degrades to NO checkpointing, i.e. OOM,
            # not to safe-and-slow. Printing both states makes the policy visible in
            # the log instead of inferable from the absence of a line.
            print(f"[geotex] elastic GC OFF — static checkpointing at "
                  f"geotex_gc={native_args.geotex_gc}")
        # freeze audit: trainables must be EXACTLY {tex flow, t-mixer, cross_alpha,
        # c_gates, tex connector} — a stray geo/VLM param here would silently train.
        _allowed = ("unified_geotex.tex_flow.", "unified_geotex.t_mixer.",
                    "unified_geotex.t_mixer_s.",
                    "unified_geotex.cond_proj.", "unified_geotex.cond_blocks.",
                    "diffusion_connector.")
        if native_args.geotex_unfreeze_geo:
            # S2b: geo joins. The geo CONNECTOR stays frozen on purpose — one
            # variable at a time; geo's blocks 0-23 have never seen our cond, so
            # let them adapt to the existing cond distribution first.
            _allowed = _allowed + ("unified_geotex.geo_flow.",)
        if getattr(getattr(model, "unified_geotex", None), "ss_flow", None) is not None:
            # v10: three towers, three connectors, and the cross-tower gates.
            # Listed EXPLICITLY rather than by widening the audit to a bare
            # "unified_geotex." prefix — the audit's whole value is that it fails
            # when something unintended became trainable, and there are now three
            # towers to keep straight.
            _allowed = _allowed + (
                "unified_geotex.ss_flow.", "unified_geotex.geo_flow.",
                "unified_geotex.ss_gates_geo", "unified_geotex.ss_gates_tex",
                "unified_geotex.ss_reads_gate", "unified_geotex.cond_seg_embed", "unified_geotex.cond_patch_pos",
                "geo_connector.", "ss_connector.")
        if getattr(getattr(model, "unified_geotex", None), "from_scratch", False):
            # From-scratch MMDiT3D: there is no frozen warm start, so EVERY
            # unified param is meant to train (geo_flow, tex_flow, cond_flow,
            # cond_t, shared_blocks). The audit still earns its keep — it keeps
            # catching a VLM or geo-connector param leaking into the optimizer.
            _allowed = _allowed + ("unified_geotex.",)
        _exact = {"unified_geotex.cross_alpha", "unified_geotex.cross_alpha_s",
                  "unified_geotex.c_gates", "unified_geotex.b_gates",
                  "unified_geotex.cond_gates_tex", "unified_geotex.cond_gates_geo",
                  "unified_geotex.cond_reads_gate"}
        _bad = [n for n, p in model.named_parameters()
                if p.requires_grad and not (n.startswith(_allowed) or n in _exact)]
        assert not _bad, f"[geotex] unexpected trainable params: {_bad[:8]}"
        print(f"[geotex] freeze audit OK — coupling={native_args.geotex_coupling} "
              f"p_corner={native_args.geotex_p_corner} p_corner2={native_args.geotex_p_corner2}")
        # dino_view_embed is a FIXED buffer whose scale is an env var at build time
        # (t50b ckpts: sincos × VIEW_EMBED_SCALE=0.2, L2/row 4.531). Restore the tex
        # run's exact table so the warm-started connector sees the values it was
        # trained with — env drift here would silently shift every IM cond.
        # ...but only for a warm start. From scratch there is no ckpt to be
        # consistent WITH — the fresh connector's table is built here, at this
        # run's VIEW_EMBED_SCALE, and is the only definition. Reading the tex
        # ckpt anyway would crash a machine that has no such ckpt, and would
        # otherwise silently overwrite the table this run just built.
        from safetensors.torch import load_file as _lf_ve
        _sd_ve = {} if native_args.geotex_from_scratch else \
            _lf_ve(os.path.join(native_args.geotex_tex_init, "model.safetensors"))
        if "dino_view_embed" in _sd_ve and getattr(model, "dino_view_embed", None) is not None:
            with torch.no_grad():
                model.dino_view_embed.copy_(
                    _sd_ve["dino_view_embed"].to(model.dino_view_embed.dtype))
            print("[geotex] dino_view_embed restored from tex ckpt "
                  f"(L2/row={model.dino_view_embed.norm(dim=1).mean():.3f})")
        del _sd_ve
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
        # sincos view-embed is a FIXED buffer: a warm-start ckpt trained with the old zero-init
        # LEARNED dino_view_embed would overwrite the sincos values with zeros → drop that key so
        # the fresh sincos buffer survives (this is the whole point of the mode).
        if native_args.view_embed_mode == "sincos":
            sd = {k: v for k, v in sd.items() if "dino_view_embed" not in k}
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
        # accelerate>=1.x: extract_model_from_parallel() does `has_compiled_regions(model)` and,
        # if any SUBMODULE is compiled (our ss_flow), tries `model = model._orig_mod` on the TOP
        # model — which is NOT compiled → AttributeError inside Trainer.__init__'s unwrap_model.
        # We deliberately compile only the ss_flow submodule and keep it compiled through training,
        # so neutralize that top-level unwrap probe (resume re-compiles → checkpoint keys still match).
        try:
            import accelerate.utils.other as _aother
            if getattr(_aother, "has_compiled_regions", None) is not None:
                _aother.has_compiled_regions = lambda *a, **k: False
                print("[train_native] patched accelerate.has_compiled_regions→False (partial-compile unwrap fix)")
        except Exception as _e:
            print(f"[train_native] WARN: could not patch has_compiled_regions: {_e}")

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
    if native_args.mds_root or native_args.mds_shards:
        # MDS path: StreamingDataset does its own node/rank-aware sharding + resume.
        from trellis2_blip3o.data.streaming_task import StreamingImageTo3D
        from trellis2_blip3o.data.mixture import MultiTaskCollator
        _shard_dirs = [d for d in (native_args.mds_shards or "").split(",") if d] or None
        train_ds = StreamingImageTo3D(
            native_args.mds_root,
            shard_dirs=_shard_dirs,
            fuse_dino=native_args.fuse_dino,
            ss_only=native_args.ss_only,
            max_slat_tokens=native_args.max_slat_tokens,
            shuffle=True, batch_size=per_dev_bs,
            cache_limit=native_args.mds_cache_limit,
        )
        collator = MultiTaskCollator(processor=processor)
        _src = f"shards={_shard_dirs}" if _shard_dirs else native_args.mds_root
        print(f"[train_native] MDS source {_src} "
              f"({len(train_ds.ds)} samples this node, fuse_dino={native_args.fuse_dino}, ss_only={native_args.ss_only})")
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
    gclip = None
    if native_args.adaptive_grad_clip:
        gclip = AdaptiveGradClipCallback(max_norm=training_args.max_grad_norm)
        callbacks.append(gclip)
    trainer = NativeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    if gclip is not None:
        # model_wrapped is the DeepSpeedEngine and is not handed to callbacks;
        # it only exists once accelerate has prepared it, i.e. inside train(),
        # so bind lazily on the first step instead of here.
        _orig_ts = trainer.training_step

        def _bind_then_step(*a, **k):
            if gclip._engine is None:
                gclip._engine = trainer.model_wrapped
            return _orig_ts(*a, **k)
        trainer.training_step = _bind_then_step
    trainer.train(resume_from_checkpoint=bool(list(__import__("pathlib").Path(training_args.output_dir).glob("checkpoint-*"))) or None)
    trainer.save_state()


if __name__ == "__main__":
    # Bind this rank's CUDA device BEFORE any collective. torchrun sets LOCAL_RANK; without an
    # explicit set_device, early NCCL collectives (e.g. StreamingDataset's barrier, which runs
    # before accelerate/DeepSpeed bind the device) warn "devices ... currently unknown" and can
    # HANG cross-node (the 2-node S1 jobs timed out on an allreduce). Idempotent w/ accelerate.
    import os as _os, torch as _torch
    if _os.environ.get("LOCAL_RANK") is not None and _torch.cuda.is_available():
        _torch.cuda.set_device(int(_os.environ["LOCAL_RANK"]))
    main()
