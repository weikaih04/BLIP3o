# Multi-Task Data Infrastructure

This document covers the multi-task data system that drives `train_native.py`:
how to mix multiple task families (`text_to_3d`, `image_to_3d`, `multi_image_to_3d`,
`vqa`, `grounding`, `text_sft`) at configurable ratios, how to add a new task,
and the efficiency design choices that make it work well under DeepSpeed ZeRO.

**TL;DR**

```bash
# Pure 3D mix (matches the legacy task_mix='T:0.2,I1:0.4,IM:0.4')
torchrun --nproc_per_node=4 train_native.py \
  --mixture_config configs/mix_3d_only.yaml \
  --vlm_model Qwen/Qwen3.5-2B  ...

# Full mix: 3D + VQA + grounding + pure-text SFT (anti-catastrophic-forget)
torchrun --nproc_per_node=4 train_native.py \
  --mixture_config configs/mix_full.yaml \
  --freeze_vlm False   # required for LM-loss tasks
  --vlm_model Qwen/Qwen3.5-2B  ...
```

To change the mix ratio: **edit a YAML file**. No code changes.
To add a new task: **one new Python file + one new YAML row**. Model and trainer untouched.

---

## 1. Why

Earlier the dataset did task sampling **inside one `__getitem__`** (`TR2NativeVLMDataset`
with `--task_mix 'T:0.2,I1:0.4,IM:0.4'`). That works for 3 tightly-related 3D
tasks but breaks the moment a 4th task arrives — VQA needs `labels`, 3D needs
`target_ss_latent`, pure NLP has no image, etc. Different shape signatures,
different loss heads, different data sources. Stuffing all of those into one
class is unmaintainable.

The new design treats each task as a **plug-in**:
* its own Dataset class with its own `__getitem__` and `collate_fn`
* register via `@register_task("name")`
* list it in a YAML mixture config with a weight
* model's `forward` dispatches on a `_task` field

Adding pure-text SFT (`text_sft`) to a 3D run becomes a 1-line YAML edit —
which is exactly the anti-catastrophic-forget regularizer you want when
fine-tuning a VLM like Qwen3.5 on heavy 3D / VQA data.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  YAML mixture config  configs/mix_*.yaml                          │
│  tasks: [(name, weight, args), ...]   sampling: granularity, temp │
└────────────────────────────────┬─────────────────────────────────┘
                                  │ build_mixture(yaml, processor, BS)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  MixtureIterableDataset                                           │
│  • wraps N per-task Dataset objects + normalized weights          │
│  • each __iter__: pick task by weight, yield BS items same task   │
│  • rank-synced task choice (Gap-1 fix; see §5)                    │
│  • per-batch param hook (Gap-3 fix; see §5)                       │
└────────────────────────────────┬─────────────────────────────────┘
                                  │ (HF DataLoader pulls items)
                                  ▼
              ┌────────────────────┴────────────────────┐
              ▼                    ▼                    ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ ImageTo3DDataset │  │   VQADataset     │  │ TextSFTDataset   │
   │  reads           │  │  reads chat      │  │  reads chat      │
   │  ready_v1.jsonl  │  │  JSONL + image   │  │  JSONL (no img)  │
   │  emits dict      │  │  emits dict      │  │  emits dict      │
   │  _task=          │  │  _task="vqa"     │  │  _task="text_sft"│
   │  "image_to_3d"   │  │                  │  │                  │
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            └──────────────┬──────┴──────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  MultiTaskCollator(processor)                                     │
│  • batch[0]["_task"] → dispatches to cls.collate_fn(batch, proc)  │
│  • each task implements its OWN tokenization / padding            │
│  • emits dict stamped with _task                                  │
└────────────────────────────────┬─────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  TrellisNativeVLM.forward(**batch)                                │
│  • if _task ∈ {"vqa","grounding","text_sft"} → _forward_lm        │
│      (vlm.lm_head + CE on `labels`)                               │
│  • else → flow path (3-stage cascade MSE: SS / Shape / Tex)       │
└──────────────────────────────────────────────────────────────────┘
```

### Files

```
trellis2_blip3o/
├── configs/
│   ├── mix_3d_only.yaml           # 3-task mix (parity with legacy --task_mix)
│   └── mix_full.yaml              # 6-task mix (3D + LM tasks)
├── trellis2_blip3o/data/
│   ├── registry.py                # @register_task decorator + lookup
│   ├── base.py                    # TaskDataset / collator contract
│   ├── mixture.py                 # MixtureIterableDataset + MultiTaskCollator
│   └── tasks/
│       ├── threed.py              # text_to_3d / image_to_3d / multi_image_to_3d
│       └── chat.py                # vqa / grounding / text_sft (LM-loss)
└── blip3o/model/language_model/trellis_native_vlm.py
                                   # forward dispatches on _task → flow or LM path
```

---

## 3. YAML config schema

```yaml
tasks:
  - name: <registered_task_name>   # e.g. "text_to_3d"
    weight: <float >= 0>           # raw weight; framework normalizes
    args:                          # forwarded as kwargs to Dataset.__init__
      manifest: /abs/path/to.jsonl
      ...task-specific kwargs...
  - name: ...
    weight: ...
    args: ...

sampling:
  granularity: batch               # batch | item  (batch is recommended; see §5)
  temperature: 1.0                 # p ∝ weight^(1/τ); τ<1 to suppress big tasks
```

### Per-task args reference

**`text_to_3d` / `image_to_3d` / `multi_image_to_3d`** (`data/tasks/threed.py`):
```yaml
args:
  manifest: data/manifests/ready_v1.jsonl       # unified ready_v1 schema
  slat_resolution: 512                          # 512 | 1024
  crop_to_object: false                         # alpha-bbox crop on RGBA renders
  max_views: 4                                  # IM: n_views ∈ [2, max_views]
  min_aesthetic: 4.5                            # init-time filter (None to disable)
  require_caption: false                        # default True for text_to_3d
  ss_only: false                                # skip shape/tex targets
```

**`vqa` / `grounding`** (`data/tasks/chat.py`):
```yaml
args:
  manifest: data/manifests/llava_instruct.jsonl
  image_root: /optional/image/root              # joined with relative image paths
  require_image: true                           # drop rows without `image`
  max_image_tokens: 4096                        # per-image cap (Qwen3.5: 4096 ≈ 2048²)
```

**`text_sft`** (`data/tasks/chat.py`):
```yaml
args:
  manifest: data/manifests/tulu_chat.jsonl
  require_image: false                          # pure NLP — no image processing
```

### JSONL schemas

**3D tasks** — see `build_index.py` for canonical schema; one record per asset:
```json
{
  "sha256": "...", "subset": "ABO",
  "ss_latent_64": "...", "shape_latent_512": "...", "shape_latent_1024": "...",
  "pbr_latent_512": "...",  "pbr_latent_1024": "...",
  "renders_dir": "...", "n_views": 16,
  "captions": ["full caption...", "shorter...", "Chair."],
  "aesthetic_score": 5.5
}
```

**Chat tasks** — flexible; supports two formats:
```json
// canonical
{"image": "/path.jpg",
 "conversations": [{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}]}

// LLaVA-style (auto-converted)
{"image": "/path.jpg",
 "conversations": [{"from": "human", "value": "..."},
                   {"from": "gpt",   "value": "..."}]}
```

`image` is optional (omit for pure NLP). Multi-turn supported; ALL assistant
turns are supervised (matches LLaMA-Factory `qwen3` default).

---

## 4. How to add a new task

Example: a `depth_to_3d` task taking a depth map → 3D.

**1. Write the task** (`trellis2_blip3o/data/tasks/depth_3d.py`):
```python
from torch.utils.data import Dataset
from ..registry import register_task

@register_task("depth_to_3d")
class DepthTo3DDataset(Dataset):
    def __init__(self, manifest, ...):
        self.records = [json.loads(l) for l in open(manifest)]
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        return {
            "_task": "depth_to_3d",
            "depth": ...,                  # torch.Tensor
            "target_ss_latent": ...,
            ...
        }

    @staticmethod
    def collate_fn(batch, processor):
        return {
            "_task": "depth_to_3d",
            "pixel_values": ...,            # processed depth
            "target_ss_latent": torch.stack(...),
            ...
        }
```

**2. Register on package import** (`trellis2_blip3o/data/tasks/__init__.py`):
```python
from . import depth_3d   # triggers @register_task
```

**3. Use in YAML**:
```yaml
tasks:
  - name: depth_to_3d
    weight: 0.10
    args: {manifest: data/manifests/depth.jsonl}
```

**4. (LM-loss tasks only)** add the name to `TrellisNativeVLM._LM_TASKS` in
`blip3o/model/language_model/trellis_native_vlm.py`. For 3D / flow-loss tasks,
no model change is needed — the existing flow path handles them.

That's it. `train_native.py`, `MixtureIterableDataset`, and `MultiTaskCollator`
require no changes.

---

## 5. Efficiency design — fixes applied

Multi-task mixing introduces 4 efficiency hazards. The mixture is built to
defuse them; the design choices are documented below.

### 5.1 Batch-homogeneous vs mixed-per-example — and why we chose homogeneous

**Industry context first**: the bulk of open-source VLM SFT (LLaVA, LLaVA-NeXT,
Qwen-VL / Qwen2-VL, InternVL, Molmo, PaliGemma, DeepSeek-VL, Janus, …) uses
**mixed-per-example** sampling — each row of a batch can be a different task /
dataset, drawn by mixture weight. That's the default for ~99% of multi-task
VLM training. Papers typically find mixed-per-example **slightly better**
(< 1 pt) than batch-homogeneous, mostly via lower per-step gradient variance
and implicit task balancing.

**We deliberately chose batch-homogeneous** (`granularity="batch"`, default):
every batch contains items of one task only; tasks rotate per step.

**Why we deviate from the industry default**: the LLaVA-family precedent works
because all their tasks share the SAME batch shape signature (image + chat →
text). Our task families do not:

| Task family | Batch fields present | Loss head |
|---|---|---|
| `image_to_3d` / `multi_image_to_3d` | `pixel_values`, `target_ss_latent`, `target_shape_slat_512:SparseTensor`, `target_tex_slat_512:SparseTensor`, `tex_concat_cond:SparseTensor` | flow MSE |
| `text_to_3d` | (no `pixel_values`), `target_ss_latent`, sparse SLAT targets | flow MSE |
| `vqa` / `grounding` | `pixel_values`, `labels` (no 3D targets) | CE |
| `text_sft` | (no `pixel_values`), `labels` | CE |

Mixed-per-example would require:
* a collator that unions all task-shapes per batch, padding the missing fields
  per row, and correctly stacking SparseTensors only on the sub-rows that
  carry them (~150-200 LOC);
* a model forward that slices the batch by `_task`, runs the flow path on the
  3D sub-rows and the LM path on the chat sub-rows, then weighted-sums two
  losses of incompatible units (~100 LOC plus careful normalization).

Homogeneous batching collapses both into:
* per-task `collate_fn` (~50 LOC each, each owns ONE shape signature),
* a 15-line `MultiTaskCollator` that just dispatches by `_task`,
* a `forward` that branches on `_task` and runs ONE loss.

**Performance impact** (per the industry literature):

* Final-task-accuracy delta: **< 1 pt** (most papers).
* Gradient variance per step: higher (single-task signal), **mitigated by
  `gradient_accumulation_steps > 1`**: with grad_accum=4, four micro-batches
  see four (re-rolled) task draws, so the *optimizer* step gradient is
  effectively mixed across ~4 tasks even though each micro-batch is
  single-task. This is the natural setting under our NO-OFFLOAD memory
  policy where large tasks already need grad_accum.
* Padding waste within a batch: minimal (same shape signature → no
  cross-task pad).
* Wall-clock throughput: slightly better than mixed (no cross-task padding).

**When you'd want mixed instead**: if all your tasks shared a single shape
signature (Molmo with five 3D-detection variants is the canonical case),
mixed is the cleaner choice — it costs nothing extra in collator code and
buys back the < 1 pt accuracy. The choice here is *shape diversity* driving
the implementation; **shape-homogeneous mixtures should switch to mixed**.

**What homogeneous batching does NOT mean**: it does NOT mean different ranks
see different tasks per step (that would be the *worst* of all worlds — see
§5.2 Gap-1). All DDP/DeepSpeed ranks pick the SAME task on the SAME step.
Mixing is across STEPS, not across ranks within a step.

### 5.2 Rank-synced task choice (Gap-1 fix)

Under DDP / DeepSpeed ZeRO, all ranks must finish backward before the
reduce-scatter / all-reduce barrier. If rank 0 happens to pick a fast task
(`text_sft`, ~0.5s/step) while rank 1 picks a slow task
(`multi_image_to_3d`, ~3s/step), **rank 0 idles for 2.5s waiting on rank 1**.

Effective throughput collapses to the slowest task's speed even when "easy"
tasks dominate the mixture.

**Fix**: the per-step task-choice RNG is seeded WITHOUT `rank`. All ranks
draw the same task on the same step. Per-task index sampling still includes
`rank` (data-parallel — each rank sees a different slice of the same task's
data). See `MixtureIterableDataset.__iter__`.

Verified by `tests/test_mixture_efficiency.py::test_gap1_rank_synced_task_choice`.

### 5.3 Per-batch n_views (Gap-3 fix)

`multi_image_to_3d.__getitem__` picks `n_views ∈ [2, max_views]` per item.
With per-item independent sampling, one batch can contain a 2-view item and a
4-view item — the 2-view item gets padded out to 4-view length, wasting ~50%
of the vision-token budget on that row.

**Fix**: `MixtureIterableDataset` calls `dataset.set_batch_params(rng)` ONCE
before the per-task batch is drawn. For `MultiImageTo3DDataset`, this samples
`n_views` for the whole batch and stashes it on `self._batch_n_views`. Every
`__getitem__` in that batch reads the shared value → zero intra-batch padding
on vision tokens.

The hook is opt-in: any task can implement `set_batch_params(rng)` + optional
`clear_batch_params()`. Tasks that don't expose it are unaffected.

Verified by `tests/test_mixture_efficiency.py::test_gap3_per_batch_n_views`.

### 5.4 num_workers safety (Gap-4 guard)

PyTorch's `DataLoader` with `IterableDataset + num_workers > 1` round-robins
items from different workers when assembling batches. If worker 0 is on
`text_sft` and worker 1 is on `vqa`, the assembled batch could mix tasks → the
`MultiTaskCollator`'s homogeneity assertion fires.

**Fix**: at `__iter__` time we warn loudly if `num_workers > 1`. HF Trainer
defaults `dataloader_num_workers=0`, so the bug is dormant by default — but
the warning fires the moment someone enables workers for IO throughput.

A proper fix requires changing the contract to yield pre-batched lists +
identity collator (currently a TODO).

Verified by `tests/test_mixture_efficiency.py::test_gap4_num_workers_warning`.

---

## 6. Known gaps / future work

### 6.1 Per-task batch size (DEFERRED)

The HF Trainer's `--per_device_train_batch_size` is a single global value.
With NO-OFFLOAD memory policy (see `feedback_no_offload_policy.md`), BS is
capped by the largest task — `multi_image_to_3d` BS=2 might use 65GB while
`text_sft` BS=2 uses 8GB on the same card.

Ideal:
```yaml
tasks:
  - {name: multi_image_to_3d, weight: 0.30, per_device_batch_size: 2}
  - {name: text_sft,          weight: 0.10, per_device_batch_size: 16}
```

Implementing this requires inverting the data-flow contract: `IterableDataset`
yields **pre-collated batches** as single items, HF Trainer's `batch_size=1`,
collator becomes identity. ~30 LOC change, planned with the upcoming rename
refactor.

### 6.2 Cost-balanced sampling (NOT IMPLEMENTED)

If `image_to_3d` (1.5s/step) and `text_sft` (0.5s/step) both have weight 0.5,
wall-clock-wise the model sees 3× more `text_sft` steps. To get
"training-budget 1:1" you'd reverse-weight by per-task step time:
```yaml
sampling:
  cost_balance: true     # not implemented; manual weight tuning works fine for now
```

Profile your config first; until you have measured per-task step times,
hand-tuning `weight` is the right tool.

### 6.3 Per-turn supervision in multi-turn chat (DONE)

Chat collator's `_build_supervised_labels` scans for every
`<|im_start|>assistant\n` marker and unmasks until the next `<|im_start|>`.
All assistant turns in a multi-turn conversation get supervised (matches
LLaMA-Factory `qwen3` default `mask_history=False`). Verified
`tests/test_chat_multiturn.py`.

---

## 7. Tests

```bash
ENV=/weka/.../envs/blip3o_trellis_qwen35/bin/python

# Core multi-task wiring (6 tasks register, mixture produces right-shape batches)
$ENV tests/test_mixture.py

# Chat label masking (Qwen3.5 official chat template; bit-identical to LF for single-turn)
$ENV tests/test_chat_mask.py
$ENV tests/test_chat_multiturn.py

# Efficiency fixes (Gap-1 rank sync, Gap-3 per-batch n_views, Gap-4 worker warning)
$ENV tests/test_mixture_efficiency.py
```

All four are <30s on 1 GPU.

---

## 8. Quick reference

| You want to … | Where |
|---|---|
| Change mix ratios | `configs/mix_*.yaml` `weight:` |
| Drop a task from the mix | Comment its block or `weight: 0` |
| Use a different manifest for same task | Add another `name: <same>` block with new `args.manifest` |
| Add a brand-new task | `data/tasks/<new>.py` + register, import in `tasks/__init__.py`, add YAML row |
| Add an LM-loss task | Above + add name to `TrellisNativeVLM._LM_TASKS` |
| Suppress an over-represented task | `sampling.temperature < 1.0` |
| Run mixture training | `train_native.py --mixture_config configs/X.yaml` |
| Verify a config | The trainer prints the resolved mix at startup |
