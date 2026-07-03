# Training Handoff — BLIP3o-NEXT × TRELLIS.2 (connector warm-up → fusion split → multitask split)

> Owner-handoff doc. Written 2026-06-25, **flow rewritten 2026-06-26 from ground truth**
> (origin commit `96a0b26`, `experiment/qwen35_trellis2_training/*.yaml` = the actual
> Beaker jobs weikaih ran — superseded the earlier reconstruction in §1-§2). fsx box
> `/fsx/sfr/weikaih/3dgen/model/BLIP3o` (branch `BLIP3o-NEXT`). Claims are read from
> source / verified by running the env; **"⚠️ GAP"** = does NOT exist or work yet.
> Design rationale: `QWEN35_VLM_DESIGN.md`, `DUAL_COND_DESIGN.md`, `V3_DISTILL_DESIGN.md`,
> `OPTIMIZATIONS.md`, `MULTI_TASK_DATA.md`, `RESULTS.md`, `TRAINING_PLAN.md`.

## 0. TL;DR — the live method

Frozen **Qwen3.5-2B** VLM → per-stage **connector** (MLP, 2048→1024, identity-init
LayerNorm) → one of three **TRELLIS.2 flows** (`ss` / `shape_slat_512` / `tex_slat_512`).
Image conditioning is `[ raw DINOv3 tokens (d-key) ; connector(Qwen hidden) (v-key) ]`
concatenated into ONE cross-attn stream ("**fusion**"), read from an offline cache.
Training is a **3-link curriculum**, and the flows are trained as **separate per-flow
jobs** (each its own connector copy, GT-decoupled, assembled at inference):

```
① connector warm-up      ─►   ② FUSION split (×3)      ─►   ③ S3 multitask split (×3)
   train_connector_warmup        train_native_split            train_native_split
   mix_i1_only (LIVE vlm)        mix_i1_cached_fusion          mix_s3_multitask (I1.5/IM.3/T.2)
   no cache, no fusion           cached + fusion               cached + fusion
   flow_tune=none                flow_tune=last20              flow_tune=last20
   connector only, flows ❄       train_stages ss|shape|tex     train_stages ss|shape|tex
   ema 0.9999                    ema 0.998, dino_drop 0.3      ema 0.998, dino_drop 0.3, cond_max 10240
   3000 steps, eff BS 256        init ← ①@3000                init ← ②fusion_<stage>@3000
```
Every link: 3000 steps, eff BS 256 (bs4×ga8×8gpu), lr 1e-4, warmup 100, wd 0.01,
betas 0.9/0.95, grad-clip 1.0, zero2 no-offload. **Source of truth = commit `96a0b26`.**

## 1. The 3-link curriculum (ground truth)

These are TWO different "threes" — don't conflate:
- **3 tasks** = T / I1 / IM (text / image / multi-image → 3D) — a *data* axis (the mixture).
- **3 stages** = ss / shape / tex (the three flows) — a *per-flow-split* axis (`--train_stages`).

The curriculum chains them:

| link | job(s) | data (mixture) | trainable | warm-start from |
|---|---|---|---|---|
| **① connector warm-up** | 1 job | `mix_i1_only` (live VLM, no cache, no fusion) | connector only (`flow_tune none`, all 3 flows ❄) | TRELLIS pretrained |
| **② fusion split** | 3 jobs (ss/shape/tex) | `mix_i1_cached_fusion` (I1, cached+fusion) | connector + flow tail (`flow_tune last20`) | ① `checkpoint-3000` |
| **③ s3 multitask split** | 3 jobs (ss/shape/tex) | `mix_s3_multitask` (I1+IM+T, cached+fusion) | connector + flow tail (`flow_tune last20`) | ② `fusion_<stage>/checkpoint-3000` |

① warms a SINGLE connector with no DINO; ②/③ add the fusion DINO segment (zero-init
`dino_view_embed` loads cleanly over ①) and split into per-flow jobs, each chaining its
own connector copy. ③ is where the 3 tasks (T/IM) finally enter.

## 2. How to launch (the chain)

Two launchers wrap `train_native.py` with the ground-truth flags + the H200 overrides
agreed 2026-06-26 (**`elastic_target_ratio 0.1→0.75`**, **`dino_drop_prob→0.3`** both rounds).
Both auto-compute `grad_accum` to hold **eff BS 256** when you change `NPROC`/`PER_GPU_BS`.

```bash
conda activate blip3o_trellis          # see §6 — env is ready on this box

# ① connector warm-up → runs/p1_connector_warmup/checkpoint-3000
bash scripts/train_connector_warmup.sh 8

# ② FUSION split (init = ①). All three init from the SAME ① ckpt → can run in parallel.
S1=runs/p1_connector_warmup/checkpoint-3000
bash scripts/train_native_split.sh ss    fusion $S1 8
bash scripts/train_native_split.sh shape fusion $S1 8
bash scripts/train_native_split.sh tex   fusion $S1 8     # → runs/fusion_<stage>/checkpoint-3000

# ③ S3 multitask split (each inits from its own ② fusion ckpt)
bash scripts/train_native_split.sh ss    s3 runs/fusion_ss/checkpoint-3000    8
bash scripts/train_native_split.sh shape s3 runs/fusion_shape/checkpoint-3000 8
bash scripts/train_native_split.sh tex   s3 runs/fusion_tex/checkpoint-3000   8
```
`train_native_split.sh STAGE ROUND INIT_CKPT [NPROC]`. Env overrides: `PER_GPU_BS`(4),
`ELASTIC_RATIO`(0.75), `DINO_DROP`(0.3), `MAX_STEPS`(3000), `LR`(1e-4), `RUN_TAG`.

**Per-stage wiring** (matches ground truth; `flow_heads.py:209-233`):
- `ss`: `build_slat False`, `ss_only True`, **`compile_ss_flow True`** (only flow compile can handle).
- `shape`/`tex`: `build_slat True`, `ss_only False`, **`compile_ss_flow False`** (ss_flow=None → `torch.compile(None)` throws), `elastic_slat True --elastic_target_ratio 0.75`.

**Per-round wiring**: `fusion` → `mix_i1_cached_fusion`, workers 4; `s3` → `mix_s3_multitask`,
`--cond_max_length 10240` (4-view fusion ≈ 8225 tok), workers 8.

**GPU note (H200, 141G)**: 8/stage reproduces eff BS 256 exactly. `ss` is light (513M,
`build_slat False`) — fine on 2–4 GPU (launcher rebalances `grad_accum`). For speed you can
`PER_GPU_BS=8 NPROC=8` (ga auto → 4). `elastic 0.75` only affects shape/tex; if they OOM,
drop to 0.5. The original Beaker jobs are 8×H200 (`gpuCount: 8`, `ai2/jupiter|ceres`).

## 3. The three flows (verified: `flow_heads.py:209-233`, `builder.py`)

| stage | builds | loads TRELLIS ckpt | target it needs | t-schedule | special |
|---|---|---|---|---|---|
| `ss` | `ss_flow` (dense, 1.3B) | `ckpts/ss_flow_img_dit_1_3B_64_bf16` | `target_ss_latent` | logitNormal | the ONLY flow torch.compile can handle |
| `shape` | `shape_slat_512` (sparse) | `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16` | `target_shape_slat_512` | uniform | conditions on **GT coords** |
| `tex` | `tex_slat_512` (sparse) | `ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16` | `target_tex_slat_512` + `tex_concat_cond` | uniform | **GT shape SLAT teacher-forced** as concat_cond |

ckpts live under `checkpoints/TRELLIS.2-4B/ckpts/` (present on this box). Decoders
(`shape_dec`/`tex_dec`/`shape_enc`/`tex_enc`) are frozen, inference-only. Each split job
trains its **own connector copy** — at inference you load all three (connector+flow) sets.

**None-flow guards** (`flow_heads.py:211-237`): a stage with its flow or its target = None
is silently skipped; only ALL-None raises. So a wrong `--train_stages`/`--build_slat`
combo can "run" while training nothing — watch the per-stage loss actually appears in wandb.

## 4. Data + cache contract (what must exist on disk)

Two independent inputs (do not conflate):
1. **Cond cache** — `cached_hidden_root` in the yaml. Tree: `{root}/{sha[:2]}/{sha}/v{NNN}.npz`,
   plus `_meta.json` (records `vlm`, `target_tokens_per_view`, `crop_to_object`, `max_views`,
   and for fusion `dino_model`/`dino_max_views`). A merged v-entry npz holds `hidden (T,1024) fp16`,
   `keep_mask`, `dino_hidden (N_d,1024)`, `dino_keep_mask`. (`vlm_cache.py`, `data/tasks/threed.py:78-115,314-352`)
2. **GT latents** — paths inside the **manifest** jsonl, NOT in the cache. Per record:
   `sha256`, `n_views`, `ss_latent_64`, `shape_latent_512`, `pbr_latent_512` (+ `_1024`
   variants), `renders_dir`, `captions`, `aesthetic_score`. Latent npz = `{coords:(N,3) int32, feats:(N,C) f32}` for SLAT, dense for ss.

Build recipe (sharded; `scripts/build_vlm_cache.py` → `build_dino_cache.py` → `merge_vd_cache.py`):
```bash
python scripts/build_vlm_cache.py  --manifest M --out_root R --target_tokens_per_view 1024 --crop_to_object 1 --max_views 4 --shard i --num_shards N
python scripts/build_dino_cache.py --manifest M --out_root R --crop_to_object 1 --max_views 4 --shard i --num_shards N
python scripts/merge_vd_cache.py   --root R --manifest M --max_views 4 --shard i --num_shards N   # +51% step-time fix: one open/sample
```
`fuse_dino=True` with no DINO entries in `_meta.json` → hard error (`threed.py:109-113`).

## 5. ⚠️ Current blocker on THIS box: no data

- `/weka` is **not mounted** here → the manifest (`ready_v1`, ~80k assets) and the cache
  referenced by `configs/mix_i1_cached_fusion.yaml` are **unreachable**.
- `BLIP3o/data/` does not exist. `runs/` has only a `smoke/` dir that is a **tokenizer**
  smoke (shape/codebook/occ_iou), unrelated to flow training.
- So nothing real has trained here. To run: either sync `ready_v1` + the cache from weka,
  or rebuild the cache locally (needs the raw renders + manifest + GT latents first).

## 6. Environment (verified by running it)

Env **`blip3o_trellis`** at `/fsx/sfr/weikaih/miniconda3/envs/blip3o_trellis` is ready:
`transformers 5.2.0`, `torch 2.6.0+cu124`, `deepspeed 0.19.0`, `flash_attn 2.7.3`,
`fla 0.4.2` + `causal_conv1d 1.6.2` (the FLA Gated-DeltaNet kernel — the big VLM speedup, see `OPTIMIZATIONS.md`).
- ⚠️ `requirements.txt` (transformers 4.51.3) and `requirements-freeze.txt` (4.52.4) are
  **STALE** — the live env is 5.2.0. Trust the env, not those files.
- ⚠️ Docs sometimes name `blip3o_trellis_qwen35`; on this box the working env is just
  `blip3o_trellis`. (The split launcher just says `conda activate blip3o_trellis`.)
- **No-offload policy** is enforced in code (`train_native.py:375-406`): any deepspeed
  config with `offload_optimizer.device != none` is hard-rejected. Use `configs/deepspeed_zero2.json`.

## 7. Checkpoints & monitoring (verified: `train_native.py`)

- Save dir `runs/<run>/checkpoint-<step>/`: `model.safetensors` + (separate) `ema.safetensors`
  + `config.json` + optimizer/scheduler/trainer_state. EMA (decay 0.9999) is saved
  **separately**, overlaid at inference via `USE_EMA=1`, not folded into model.safetensors.
- Resume is automatic if `checkpoint-*` exists in `output_dir` (`train_native.py:587`).
- `--init_from_checkpoint DIR` = warm-start connector+flow from `model.safetensors`
  (strict=False, strips `_orig_mod.` compile prefix), **no** optimizer/step resume.
- Watch in wandb: per-stage `flow_mse` (the stage you're training must show up and trend
  down), `grad_norm` (not 0 / not exploding), `learning_rate`. `NativeTrainer.log()`
  all-reduces per-stage losses across ranks.

## 8. ⚠️ GAPS / open items for the new owner

1. **No runnable native-3D inference script.** `inference.py` at repo root is the OLD
   BLIP3o **text-to-image** path (`T2IConfig`, `blip3oQwenForInferenceLM.generate_image`)
   — unrelated. There is no driver that loads the 3 split (connector+flow) ckpts, assembles
   the cascade (SS→shape→tex→decoders), and samples a mesh. The model-side pieces exist
   (`builder.py`, `blip3o_qwen.py`, `tr2_modules.py`) but the assembly/sampling entrypoint
   must be written/ported (TRELLIS.2 `sample_shape_slat_cascade` per `TRAINING_PLAN.md` §3).
   → **verification of a trained ckpt is currently not possible without writing this.**
2. **`tests/` is empty.** Any `tests/test_*.py` / `tests/generate_native.py` referenced in
   docs no longer exist. There is no smoke/forward test to run as-is.
3. **Data not on this box** (§5).
4. **1024 cascade / self-forcing** (`TRAINING_PLAN.md` Phase 2-4) not coded; 512 only.

## 9. Rejected approaches — do NOT resurrect (see `RESULTS.md`)

Code is still present but gated off by default; these were tested and dropped:
`dual_cond` (OOM + gate-starvation), `dino_align`/REPA (single-seed artifact),
`distill_dino`/KD (same ceiling as no-KD, +17% step + CFG fragility), discrete codebook
(continuous path subsumes it), `cond_fusion=depthwise` routing (fusion-concat is simpler).
The converged recipe is deliberately the *simple* one: frozen Qwen + offline cache +
fusion-concat + 3-stage split + pure flow-MSE + 1024 tok.

## 10. Key files

| area | file |
|---|---|
| model | `blip3o/model/language_model/trellis_native_vlm.py` |
| flow loss | `trellis2_blip3o/flow_heads.py` |
| flow builders | `blip3o/model/multimodal_decoder/builder.py` |
| norm stats / rope | `trellis2_blip3o/tr2_modules.py` |
| connector | `trellis2_blip3o/connector.py` |
| dataset / cache read | `trellis2_blip3o/data/tasks/threed.py`, `trellis2_blip3o/vlm_cache.py` |
| mixture / collate | `trellis2_blip3o/data/mixture.py`, `trellis2_blip3o/vlm_collate.py` |
| trainer entry | `train_native.py` |
| ① connector warm-up | `scripts/train_connector_warmup.sh` |
| ②③ per-flow split | `scripts/train_native_split.sh` (STAGE ROUND INIT_CKPT [NPROC]) |
| **ground-truth launch configs** | `experiment/qwen35_trellis2_training/train_{v3_stage1_nokd,fusion,s3}_*.yaml` (Beaker; origin commit `96a0b26`, weka/jasonr paths) |
| cache build | `scripts/build_vlm_cache.py`, `build_dino_cache.py`, `merge_vd_cache.py` (+ Beaker `experiment/.../build_*cache_readyv1.yaml`) |
| configs | `configs/mix_i1_only.yaml`, `mix_i1_cached_fusion.yaml`, `mix_s3_multitask.yaml`, `deepspeed_zero2.json` |
