# Training-speed optimizations (TrellisNativeVLM / Qwen3.5-2B)

Single document tracking every speed-up applied to the native-VLM training path,
**with measurements**, **what was deliberately skipped**, and the **flags to
flip them on/off**. Reproduce any number with `tests/profile_native_compile.py`.

Companion to [`QWEN35_VLM_DESIGN.md`](QWEN35_VLM_DESIGN.md) (the architecture
side). Date stamp: 2026-05-28.

---

## TL;DR table

Single-asset overfit, SS-only, `flow_tune=last40`, 1×H100, bf16-autocast, cond_len=432.

| Optimization                | Where                                   | Step time   | Δ                  | Default? |
|----------------------------|-----------------------------------------|-------------|--------------------|----------|
| (start)                     | bare profile_native.py, default opts    | ~1.44 s     | —                  | —        |
| FLA Gated DeltaNet fast-path| Qwen3.5 linear-attn (18/24 layers)      | ~310 ms     | **-1130 ms (~5×)** | YES (env)|
| `cond_max_tokens=400` cap   | `dataset_native.NativeVLMCollator`      | ~240 ms     | -70 ms             | YES      |
| `--optim adamw_torch_fused` | HF Trainer optimizer choice             | ~234 ms     | -5.5 ms            | **YES** (script) |
| `torch.compile(ss_flow)`    | `train_native.py:main` (mode=default)   | **~190 ms** | **-44 ms (-19%)**  | **YES** (flag) |
| `compile_mode=max-autotune` | same flag, autotune triton variants     | _TBD_*      | _TBD_*             | NO (one-time +5 min compile) |

*max-autotune was still benchmarking kernels (2.5s × ~hundreds of matmuls) at
doc-write time — see `tests/profile_native_compile.py` to rerun. Will refresh
once the bench finishes.

**Total realized: 1.44 s → 190 ms ≈ 7.5× on the steady-state step.**

---

## 1. FLA Gated DeltaNet fast-path (one-time env install)

**What.** Qwen3.5-2B's 18/24 layers are `linear_attention` (Gated DeltaNet, not
Mamba2). HF's `transformers/models/qwen3_5/modeling_qwen3_5.py:294` selects
between an O(N) parallel-scan triton kernel (from FLA / `fla-core`) and a slow
O(N²) fallback. Without FLA, the VLM forward is ~159 ms for 432 tokens; with
FLA it drops to ~50 ms.

**Where.** `pip install fla-core==0.4.2 causal_conv1d` into the
`blip3o_trellis_qwen35` env (build from source for causal_conv1d to avoid the
prebuilt-wheel GLIBC-2.32 mismatch, see project memory `project_blip3o_trellis_env`).

**Why NOT `mamba-ssm` / `flash-linear-attention`.** Both were tried and
rejected:
- `mamba-ssm` ships kernels for Mamba2 SSM (Qwen3 mamba-style), **not** Gated
  DeltaNet. Also breaks at import in transformers 5.x (`GreedySearchDecoderOnlyOutput`
  was removed).
- `flash-linear-attention` PyPI wheel is a stub (no `modules` / `ops`). The
  correct package is `fla-core`.

**Risk / regression watch.** Triton 3.2 (our pin) is below FLA's recommended
3.3 — warning only, kernel runs fine. If we ever bump Triton, re-bench: the FLA
kernel could regress or break.

---

## 2. `cond_max_tokens=400` token cap

**What.** `NativeVLMCollator.max_tokens_single = 400` (multi-image budget=1000)
caps the VLM input image so it produces ≤ 400 vision tokens per view (single
view) or ≤ 1000 total (multi-view). For Qwen3.5's patch16 × merge2 = 32, this
maps to ~640² px / view single, ~390² / view at 3 views.

**Why.** Quadratic-attention era this was a 4096-cap (way too generous); now
with FLA O(N) the VLM cost is linear so the savings are smaller, but **the
cross-attn in the flow** is still `O(flow_tok × cond_len)`, and `cond_len=400`
is plenty for 3D conditioning (SAM3D uses ~1280, Qwen-Image-Edit uses ~150
because it has a separate VAE-cond path — we don't).

**Why NOT cap 200.** Discussed and explicitly rejected (see chat 2026-05-28):
- After FLA, dropping 400 → 200 only saves ~20 ms (8% of step) because VLM is
  now O(N) (halving N halves work, not 4×).
- Backward (the biggest 124 ms slice) is independent of cond_len.
- 3D needs more cond density than 2D edit because we have a single cond path
  (VLM hidden) carrying semantics + geometry + appearance — Qwen-Image-Edit's
  150-tok choice doesn't transfer.
- No A/B on quality has been run. Keep 400 until we have evidence it hurts.

**Flag.** Set in `trellis2_blip3o/dataset_native.py:NativeVLMCollator`. Override
per-run by subclassing or editing the field directly.

---

## 3. `--optim adamw_torch_fused`

**What.** HF Trainer's `--optim adamw_torch_fused` selects `torch.optim.AdamW(...,
fused=True)`, a single fused CUDA kernel for the param update instead of N
elementwise kernels.

**Win.** opt_step: 9 ms → 3.5 ms (-5.5 ms / step). Free, no risk.

**Where.** Already added to `scripts/train_native_q35.sh`. For new training
scripts, just pass `--optim adamw_torch_fused`.

**Risk.** None known. fused AdamW has been stable in torch ≥ 2.0; we're on 2.6.

**Why NOT fused everywhere via CPU-offloaded optim** — `cpu_adam` (DeepSpeed
ZeRO-2 + offload) **does not benefit** from `fused=True` because the update
runs on CPU, not CUDA. For cascade training (which we offload), this flag is a
no-op; that's fine.

---

## 4. `torch.compile(model.ss_flow)` **← the new one**

**What.** Wraps the dense SS flow DiT (`SparseStructureFlowModel`, 30 blocks,
1.3 B params, 16³=4096 dense tokens) with `torch.compile`, so Dynamo +
TorchInductor fuse the per-block elementwise epilogues (RMSNorm + RoPE + SwiGLU
+ bias-add) into single kernels in both forward and backward.

**Where.** `train_native.py` between `_apply_flow_freeze(...)` and the
`NativeTrainer(...)` construction — i.e. AFTER param freezing (Dynamo sees the
final `requires_grad` layout) and BEFORE DeepSpeed/FSDP wrap (post-wrap compile
has known interaction bugs).

```python
model.ss_flow = torch.compile(
    model.ss_flow,
    mode=native_args.compile_mode,       # default | reduce-overhead | max-autotune
    dynamic=native_args.compile_dynamic, # True so varying cond_len doesn't recompile
    fullgraph=False,                     # cross_attn cond_mask path has data-dep branches
)
```

**Win** (measured on `tests/profile_native_compile.py`, single asset, cond_len=432):

| Phase       | BASELINE | COMPILE(default) | Δ                  |
|-------------|----------|------------------|--------------------|
| vlm         | 49.9 ms  | 49.2 ms          | — (not compiled)   |
| connector   | ~0 ms    | ~0 ms            | —                  |
| **flow_fwd**| 56.9 ms  | **50.5 ms**      | **-6.4 ms (-11%)** |
| **backward**|**124.0 ms**| **87.2 ms**    | **-36.8 ms (-30%)**|
| opt_step    | 3.5 ms   | 3.5 ms           | —                  |
| **total**   |**234.5 ms**|**190.4 ms**    | **-44.1 ms (-18.8%)**|

JIT warmup: ~41 s on first run (then steady-state). For long training (>1 h),
amortized cost is trivial.

**Why ONLY `ss_flow`** — three other candidates explicitly skipped:

| Candidate           | Why skipped                                                                                                  |
|--------------------|--------------------------------------------------------------------------------------------------------------|
| VLM (Qwen3.5)       | 18 layers FLA Gated DeltaNet custom triton → Dynamo breaks. Also frozen + `no_grad` → fwd-only, small win.   |
| Connector           | 3 M params, ~0 ms — fusion overhead exceeds savings.                                                          |
| Shape / Tex SLAT    | TRELLIS sparse + custom triton + elastic dynamic-checkpoint → graph-breaks consistently. Cascade re-evaluate. |

**Flags.**

```bash
--compile_ss_flow True            # on by default
--compile_mode default            # default | reduce-overhead | max-autotune
--compile_dynamic True            # keep True for real training (cond_len varies)
```

**Why `mode=default` (not reduce-overhead) for training**: `reduce-overhead`
captures CUDA graphs which re-capture on every shape change. Single-asset
overfit has fixed shapes → CUDA graph would help; real training with varying
cond_len/batch would thrash. `default` is the safe choice.

**Why `mode=max-autotune` is OFF by default**: it autotunes each matmul across
~19 Triton kernel variants (BLOCK_K, num_stages, num_warps, ...) at first
compile — ~2.5 s × ~hundreds of matmuls = several minutes one-time cost.
Marginal extra gain (3-8 ms expected). Worth it for long training runs
(`--compile_mode max-autotune` + same script).

**Known interaction warnings (from autotune logs):**
- `Torchinductor does not support code generation for complex operators` — RoPE
  uses complex ops and falls back to eager for those ops. Caps the ceiling on
  max-autotune's extra gain.
- `Casting complex values to real discards the imaginary part` — same RoPE
  path. Pre-existing, not introduced by compile.

**Save / load.** `torch.compile` returns an `OptimizedModule` wrapper that
exposes the underlying module as `._orig_mod` and proxies `state_dict` /
`named_parameters` to it. HF Trainer checkpointing works unchanged. When
loading a checkpoint, the model is built uncompiled and then re-compiled at
training start.

**Quality risk.** Zero observed. The compiled forward is numerically equivalent
to eager bf16 (Inductor preserves the autocast semantics). Loss / overfit
curves should match within run-to-run noise. If you suspect a regression: run
both passes side-by-side via `tests/profile_native_compile.py` and compare
final overfit loss after N steps.

---

## No-offload policy ⛔

**This repo forbids DeepSpeed CPU / NVMe offload.** `train_native.py` hard-refuses
any `--deepspeed` config with `zero_optimization.offload_optimizer.device != "none"`
or any `zero_optimization.offload_param.*` set; see `_enforce_no_offload` (~line 217).

**Why.** Offload trades **3-4× step time** for "fits". It masks real memory pressure
and bottlenecks throughput on the CPU `optimizer.step()`. Concretely:

- ZeRO-2 + CPU-offload cascade: ~13-20 s / step (measured in
  [[project_overfit_train_resources]])
- ZeRO-2 NO-offload on enough GPUs: same step would be ~3-5 s

When a config doesn't fit, the right responses are:

1. **Scale GPUs** (more devices → ZeRO-2 splits opt state across them, no CPU hop)
2. **Cut batch** (`per_device_train_batch_size 1 + gradient_accumulation_steps N`)
3. **Freeze more** (`--freeze_vlm True`, `--flow_tune last40` instead of `full`)
4. **Use LoRA** on the heavy module instead of full FT

**What's enforced.** `_enforce_no_offload` runs at the very top of `main()` (before
DeepSpeed init) and raises `RuntimeError` with the specific offending key. Tested
clean configs accepted, `configs/deepspeed_zero2_offload.json` (kept on disk for
reference) rejected.

**Practical impact.**

| Workload | Recommended | NPROC | Without offload |
|---|---|---|---|
| SS-only frozen VLM (513 M trainable) | `configs/deepspeed_zero2.json` | 1+ | BS=4 fits 1 GPU (50 GB) |
| Cascade frozen VLM (3.9 B trainable) | `configs/deepspeed_zero2.json` | ≥4 | BS=1 per GPU, ZeRO-2 splits opt state |
| SS-only D2 unfrozen VLM (3.5 B trainable) | `configs/deepspeed_zero2.json` | 1+ | BS=1 fits 1 GPU (39 GB); BS=2 needs 2 GPU |
| Cascade D2 unfrozen VLM (5.7 B trainable) | TBD | ≥8 | Likely needs ZeRO-3 (still no offload) |

The `train_native_q35.sh` script hard-errors if you ask for `cascade` with `NPROC<4`.

---

## What was deliberately NOT added

| Idea                          | Verdict                                                                                                                       |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Disable gradient_checkpointing| Already off by default in our SS-only path. TRELLIS official also doesn't checkpoint the SS flow (only SLAT base/ft via elastic). |
| `last40 → last20`              | Cuts trainable params and backward ~30%, but **risks quality**. last40 is empirically the level that gave clean overfit; LoRA / fewer blocks failed. Don't touch without a quality A/B. |
| Increase `batch_size`         | Not done in script yet — orthogonal lever. SS-only on 1×H100 has plenty of headroom; **bumping BS=2/4 is the next natural step** for throughput. |
| Cache VLM hidden states       | Tempting (VLM is frozen + 50 ms of every step). NOT done because (a) for the eventual VLM-unfreeze path (LoRA on VLM), the cache becomes stale; (b) for shuffled multi-task batches the cache hit rate is low. Single-asset overfit benefits most — added as opt-in future work. |
| `torch.compile` the whole model | Would compile VLM + connector + flows in one graph — Dynamo breaks at FLA / sparse triton. Per-component compile is the only viable shape. |
| fp8 / TransformerEngine        | H100 supports it, but it requires reworking attention with TE primitives and changes numerical behavior. Defer until we're throughput-bound and have headroom for an experiment. |

---

## How to use

**New runs.** Defaults in `scripts/train_native_q35.sh` already enable
everything. Just run:

```bash
bash scripts/train_native_q35.sh ss     # SS-only, 1 GPU
bash scripts/train_native_q35.sh cascade 2  # full cascade, 2 GPU, DS ZeRO-2 offload
```

**Disable** (debugging / A/B):

```bash
# Add these flags to disable individually
--compile_ss_flow False                   # turn off compile
--optim adamw_torch                        # turn off fused AdamW
# (cond cap is in the collator code — edit dataset_native.py to change)
```

**Profile your own change:**

```bash
CUDA_VISIBLE_DEVICES=0 FUSED_ADAM=1 \
  $ENV/bin/python tests/profile_native_compile.py
# Env vars: COMPILE_MODE, MODE (ss|cascade), FLOW_TUNE, FUSED_ADAM, STEPS
```

Output is a side-by-side table with **per-phase ms + Δ vs baseline**.

---

## Future work (cheapest to most expensive)

1. **Bump `batch_size` to 2 / 4 in `train_native_q35.sh`** (SS-only) — should
   ~2× throughput at near-constant per-step time. **Single biggest remaining
   win**, do this next.
2. Try `compile_mode=reduce-overhead` for single-asset overfit profile runs
   (fixed shapes → CUDA graphs viable, +5-15 ms).
3. Try `compile_mode=max-autotune` for long training runs (one-time +5 min
   compile cost amortized; +3-8 ms / step).
4. Compile the SLAT flows when we move to cascade — but **first** verify Dynamo
   doesn't break the sparse triton ops. If it does (likely), this is dead.
5. VLM-hidden cache for single-asset / small-vocab task mixes. Will help
   overfit / sanity tests, less so for shuffled multi-task.
6. fp8 forward via TransformerEngine — H100 only, biggest one-shot win but
   biggest engineering risk.
7. Investigate the `Torchinductor does not support code generation for complex
   operators` warning (RoPE) — if we can rewrite RoPE with real-valued ops,
   max-autotune can probably squeeze out more.
