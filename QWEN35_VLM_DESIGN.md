# Qwen3.5-2B Native-VLM Variant — Design Doc

> **DEPRECATED sections (2026-05-28)**: Phase-2 ran a Qwen3-VL-2B vs Qwen3.5-2B
> A/B. The result picked **Qwen3.5-2B** (Option α; see memory
> `project_qwen35_vlm_variant.md`). All Qwen3-VL-2B and Qwen2.5-VL-3B references
> below are kept for historical context, but only Qwen3.5-2B is supported going
> forward. Code, scripts, and tests for the deprecated backbones are commented
> out / labeled DEPRECATED in-place.

Status: **Phase-1 implemented + smoke-tested (both backbones)**. Author-facing design
note for a new conditioning backbone variant of the TRELLIS↔BLIP3o port.

## 0. Verification status (implemented)

Files: `trellis2_blip3o/flow_heads.py` (shared cascade loss), `blip3o/model/language_model/
trellis_native_vlm.py` (backbone-agnostic model), `tests/test_native_vlm.py` (Tier A/B/C).

Envs (key finding — Qwen3-VL works in the EXISTING env, only Qwen3.5 needs a bump):
- **Qwen3-VL-2B** → existing **`blip3o_trellis`** env (transformers 4.57.6 already has `qwen3_vl`).
- **Qwen3.5-2B** → cloned **`blip3o_trellis_qwen35`** env = `cp -a blip3o_trellis` +
  `pip install transformers==5.2.0` (earliest with `qwen3_5`). torch stayed 2.6.0,
  trellis2 + flash_attn + the blip3o import chain all still import. huggingface-hub →1.16.4 (fine).

Tests passing (1×H100, bf16, autocast):
- **Tier A** — `compute_cascade_flow_loss` w/ real SS flow + dummy cond → finite loss, grad→connector.
- **Tier B** — full `TrellisNativeVLM` forward, real backbone + dummy 3D target, **frozen AND unfrozen**
  (unfrozen → grad reaches the VLM). Qwen3-VL-2B: loss≈2.5; Qwen3.5-2B: loss≈1.8 (frozen).
- **Tier C** — overfit a fixed (cond,target) 40 steps → loss 4.3→2.0 (the train loop learns).

Run: `CUDA_VISIBLE_DEVICES=0 <env>/bin/python tests/test_native_vlm.py all`
(Qwen3.5: prefix `TEST_VLM_MODEL=Qwen/Qwen3.5-2B` + use the `_qwen35` env).

**Phase 2/3 DONE + real-data overfit verified:**
- `trellis2_blip3o/dataset_native.py` — `TR2NativeVLMDataset` (reuses the 3D-target loaders;
  emits raw caption+PIL) + `NativeVLMCollator` (native AutoProcessor → VLM inputs; stacks targets).
- `train_native.py` — HF Trainer entry (SS-only + frozen VLM default; `remove_unused_columns=False`;
  DeepSpeed via `--deepspeed`). + `scripts/train_native_q3vl.sh` / `train_native_q35.sh`.
- Model got a `build_slat` flag → SS-only skips loading shape/tex flows + decoders (lighter).
- **Real-data single-asset overfit (loveseat, `data/overfit/imgtext.jsonl`, SS-only, frozen VLM,
  1295M trainable = connector+SS flow) LOSS DECREASES on BOTH backbones**: Qwen3-VL-2B 0.67→0.42,
  Qwen3.5-2B 0.76→0.38 (`tests/overfit_native.py`). `train_native.py` also runs clean via HF Trainer.

**FULL CASCADE (SS + Shape + Tex together) trains** — `NativeTrainer._prepare_inputs` moves
SparseTensor SLAT targets to GPU. **Under the repo NO-OFFLOAD policy** (see OPTIMIZATIONS.md
§"No-offload policy"), cascade needs ≥4 GPUs with `configs/deepspeed_zero2.json`. The 1×80GB
CPU-offload recipe used during initial verification is forbidden by `train_native.py`'s
runtime guard (`_enforce_no_offload`). Cascade also currently has a sparse-tensor broadcast
bug at BS>1 (see chat 2026-05-28) that needs fixing before cascade training is reliable.

**GENERATION works end-to-end** — `tests/generate_native.py`: front half = native VLM
`encode_cond` + CFG-correct `cond_and_null` (neg=connector(0)); back half IDENTICAL to the
original `test_overfit_infer.py` (Trellis2ImageTo3DPipeline: SS → shape → tex → decode → GLB +
8-view render). Verified: fresh/untrained connector → EMPTY SS (0 voxels, expected); after a
250-step in-memory SS overfit → NON-EMPTY (3220 voxels) → 1.18M-vert textured mesh + renders.
Geometry still ROUGH/blocky (250 steps, SS-only, shape/tex frozen) — clean geometry needs
more steps + full cascade + (per the discrete recipe) unfreezing flow cross+self-attn.

Remaining (next): overfit-to-CONVERGENCE for clean geometry (more steps, cascade, unfreeze flow
attn); multi-GPU; unfreeze VLM (D2); register AutoModel + composite save/load.

---

## 1. Goal

Replace the current `Qwen3 (text) + TA-Tok + discrete <I*> codebook` stack with a
**native VLM encoder** (`Qwen/Qwen3.5-2B`) and a **purely continuous** conditioning
path (no discrete codebook, no CE on `<I*>`), then run **two ablations** that differ
in ONE knob — whether the flow gradient updates the VLM encoder.

This isolates the question: *"for 3D generation, is the discrete-AR blueprint
(BLIP3o-NEXT style) actually needed, or is continuous VLM-hidden conditioning
(original-BLIP3o / Qwen-Image style) enough?"*

## 2. Backbones: two 2B native VLMs to compare (verified from config.json)

The **primary ablation is the backbone**: two byte-identical builds that differ ONLY in
`--vlm_model`. One backbone-agnostic model class loads either via its
`*ForConditionalGeneration` class, takes `output_hidden_states[-1]` (2048-d), feeds the
same connector + TRELLIS flows.

| | `Qwen/Qwen3-VL-2B-Instruct` | `Qwen/Qwen3.5-2B` |
|---|---|---|
| class | `Qwen3VLForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` |
| model_type | `qwen3_vl` (2025-10) | `qwen3_5` (2026-03) |
| **LLM hidden** | **2048** | **2048** |
| vision | SigLIP2 1024 + deepstack | 1024 + deepstack |
| native multimodal | ✓ (vision_config) | ✓ (vision_config) |
| Instruct variant | yes (`-Instruct`) | none — use `Qwen/Qwen3.5-2B` / `-Base` |

**Lucky fact:** both have LLM hidden = **2048 = current `TRELLIS2Connector` input dim** →
connector input dim needs NO change for either backbone.

**What this ablation isolates:** newer unified arch (`qwen3_5`) vs prior gen (`qwen3_vl`)
as a 3D-conditioning encoder, everything downstream held fixed.

## 3. Data flow (proposed variant)

```
text prompt (+ optional reference image)
        │
        ▼  native VLM (vision baked in), ONE forward, take output_hidden_states[-1]
 ┌─────────────────────────────┐
 │ Qwen3.5-2B  (hidden 2048)   │   ← encoder ONLY. No AR decode, no <I*>, no CE.
 └────┬────────────────────────┘
      │ full hidden (B, T, 2048)              [cond_slice = "full"]
      ▼
 ┌─────────────────────────────┐
 │ TRELLIS2Connector           │   2048 → 1024  (MLP, unchanged)
 └────┬────────────────────────┘
      │ (B, T, 1024)  + key mask
      ▼   *** CROSS-ATTENTION *** (TRELLIS native — weights preserved)
 ┌──────────────────────────────────────┐
 │ TRELLIS SS Flow / Shape SLAT / Tex SLAT│  → SC-VAE decode → 3D
 └──────────────────────────────────────┘

Loss = flow_MSE(SS) + flow_MSE(shape SLAT) + flow_MSE(tex SLAT)   # NO CE term
```

## 4. ⚠️ Is the conditioning the same as Qwen-Image-Edit? **NO — and on purpose.**

This variant is **NOT** a faithful Qwen-Image-Edit clone. It deliberately keeps
TRELLIS's native conditioning so the pretrained TRELLIS flow weights stay reusable.
The two differ on **both** the injection mechanism and the connector design:

| Aspect | **This variant (weight-reuse)** | **Qwen-Image-Edit (diffusers, verified)** |
|---|---|---|
| Injection | **Cross-attention**: cond = K/V only, does **not** update through layers | **Dual-stream MMDiT joint attention**: text is a full stream, **updates** every layer (`Attention(cross_attention_dim=None, added_kv_proj_dim=dim)`, `QwenDoubleStreamAttnProcessor`) |
| Connector | `TRELLIS2Connector` = **Linear(2048→1024) → GELU → Linear → LayerNorm** (2-layer MLP; LayerNorm init = identity to match DINOv3 layer-norm dist the TRELLIS cross-attn was pretrained on) | **`txt_norm` (RMSNorm) + `txt_in` (single Linear 3584→3072)** — minimal, because the dual-stream blocks do the adapting |
| Appearance/reconstructive path | **none yet** (semantic VLM hidden only) | reference image **VAE latent concatenated into the image token stream** (in-context self-attn), 1024² res; VLM semantic at 384² |
| Why different | **keep TRELLIS pretrained flow weights** (TRELLIS is a cross-attn DiT; switching to MMDiT breaks clean weight reuse) | built MMDiT from scratch, no such constraint |

**Key tradeoff (the reason for the divergence):**
- Full Qwen-MMDiT alignment ⇒ must restructure TRELLIS attention (cross-attn → joint
  dual-stream) ⇒ TRELLIS's cross-attn weights don't map 1:1, text-stream params
  (txt Q / txt MLP / txt mod) are new ⇒ **partial weight reuse + retrain conditioning pathway**.
- Keeping cross-attn + `TRELLIS2Connector` ⇒ **100% TRELLIS weight reuse**, but the
  conditioning is *not* Qwen-identical.

### 4.1 Connector requirement (Option α): preserve TRELLIS DINOv3 cond distribution

**Hard requirement, carried over from the current `TRELLIS2Connector`.** TRELLIS SS-Flow's
cross-attn was pretrained on `F.layer_norm(dinov3_features, [1024])` → per-token mean=0,
std=1 (see `trellis2/modules/image_feature_extractor.py:92`). For the frozen-ish cross-attn
to behave at step 0, the connector output **must match that distribution**.

The existing connector already guarantees this **structurally**: it ends with
`nn.LayerNorm(1024)` initialized to identity (weight=1, bias=0), so its output is per-token
mean0/std1 = DINOv3's `F.layer_norm` distribution **at step 0**, then the affine may drift.

**Why this matters for the backbone swap:** because the match comes from the *output*
LayerNorm (not the input stats), it is **backbone-agnostic** — feeding Qwen3.5-2B's hidden
instead of Qwen3+TA-Tok's hidden still lands on the DINOv3 distribution at init. So:

- ✅ **Distribution match at init: preserved for free** (structural; keep the identity-init
  output LayerNorm — do NOT replace it with RMSNorm or drop it).
- ⚠️ **Content mapping is NOT preserved**: the two `Linear` weights were trained on the old
  VLM's hidden directions. With a new encoder they must **retrain** (init-from-port-ckpt is
  shape-compatible since both VLMs are hidden=2048, but the learned directions are stale →
  treat connector as needing training in both D1 and D2; D1 trains connector even though VLM
  is frozen).
- Keep `fc1 = Linear(2048→1024)` (dim already matches), `GELU`, `fc2 = Linear(1024→1024)`,
  `out_norm = LayerNorm(1024)` identity-init. **Unchanged from current code.**

(Option β would instead use Qwen's bare `RMSNorm + Linear` and a *fresh*, MMDiT-pretrained
target distribution — i.e. this DINOv3-matching requirement only applies to α.)

### Two architectural options (pick explicitly)

- **Option α — weight-reuse (this doc's default).** TRELLIS cross-attn + `TRELLIS2Connector`
  MLP. Diverges from Qwen. Fast to stand up; inherits all TRELLIS weights.
- **Option β — Qwen-identical.** Convert TRELLIS flow blocks to dual-stream MMDiT joint
  attention + bare `RMSNorm+Linear` connector + reference-latent concat. Aligns with
  Qwen-Image-Edit, but sacrifices clean TRELLIS weight reuse and needs retraining of the
  attention pathway. (The "reconstructive path" for 3D editing — concatenating a reference
  asset's sparse SLAT latent into the token stream — is the hard, unsolved part here.)

> **Decision needed.** Current plan = **α**. If the goal is true Qwen-Image-Edit parity
> (esp. editing), that's **β** and is a much larger change.

## 5. Ablation axes

**Axis 1 (primary) — backbone:** `Qwen3-VL-2B` vs `Qwen3.5-2B` (see §2). Two builds,
identical except `--vlm_model`.

**Axis 2 (secondary) — encoder training (`detach_cond`):**

| | `detach_cond` | VLM gradient | Meaning |
|---|---|---|---|
| **D1 (detach / frozen encoder)** | `true` | none (flow detached, no CE) ⇒ VLM **frozen** | train connector + TRELLIS flows only; cheap |
| **D2 (joint encoder)** | `false` | flow grad reaches VLM ⇒ encoder **jointly trained** | VLM adapts to 3D conditioning |

(With no CE, `detach=true` ⇒ VLM gets zero gradient ⇒ effectively frozen. So D1 vs D2 =
"frozen encoder" vs "flow-trained encoder".)

**Recommended run plan:**
1. **First: frozen + in-loop overfit smoke-test** (one backbone) — cheapest, fits 2×H100
   (frozen VLM = forward only, no optimizer state), and validates the full wiring
   VLM→`encode_cond`→connector→flow. This is a wiring test, not the final setting.
   (Frozen also matches Qwen-Image-Edit, which freezes Qwen2.5-VL.)
2. **Then backbone A/B** (`qwen3_vl` vs `qwen3_5`), still frozen + in-loop.
3. **Then unfreeze** (`requires_grad=True`, flag flip) IF capacity is short — but full 2B
   joint-train likely needs ≥4 GPU or VLM-LoRA, so gate on resources.

## 6. File-level changes

| Action | File |
|---|---|
| 🆕 add | `blip3o/model/language_model/trellis_native_vlm.py` — **backbone-agnostic** main model: load VLM by `--vlm_model` (`AutoModelForVision2Seq` / specific class) → `output_hidden_states[-1]` (2048) → `diffusion_connector` → TRELLIS flows (cross-attn). Parallel to `blip3o_qwen.py`; do **not** overwrite it. Backbone chosen by config, not hardcoded. |
| 🆕 add | `scripts/train_q3vl_D2.sh`, `scripts/train_q35_D2.sh` (byte-identical except `--vlm_model`); optional `_D1` variants for the 2×2 grid |
| 🔧 modify | `blip3o/model/blip3o_arch.py` — `prepare_inputs_labels_for_multimodal`: native VLM brings its own vision; route image through Qwen3.5's processor/ViT instead of TA-Tok. Keep ss_flow / connector wiring. |
| 🔧 modify | `blip3o/model/builder.py` + `blip3o/model/__init__.py` — register new model class |
| 🔧 modify | `blip3o/train/train.py` — drop codebook freeze logic; add encoder freeze toggle (D1 freeze / D2 train); `lambda_ce` unused |
| ✅ reuse as-is | `trellis2_blip3o/{tr2_modules,connector,loss,dataset}.py` (connector dim 2048→1024 already matches) |
| 🗑️ unused | `tok/`, `multimodal_encoder/ta_tok_encoder.py`, `trellis2_blip3o/codebook_prep.py`, `blip3o_qwen_grpo.py`, `trl/`, `scripts/prep_data_codebook.sh` (all discrete-only) |

## 7. Risks / prerequisites

- 🔴 **transformers version**: `qwen3_5` is a brand-new arch (2026-03); `qwen3_vl` is
  2025-10. A transformers new enough for `qwen3_5` should also cover `qwen3_vl`, so **one
  new env serves both backbones.** The pinned `blip3o_trellis` env almost certainly lacks
  `qwen3_5`. **Must validate a separate env** (`cp -a` clone → bump transformers → confirm
  BOTH `Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")` and
  `Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-2B")` load, run one
  forward, and don't break the pinned torch / flash_attn). **Gate everything on this.**
- Connector input dim 2048 matches — but verify Qwen3.5's `output_hidden_states[-1]` is
  the 2048-d LLM hidden (not a vision-merger dim).
- Native VLM image-token handling differs from TA-Tok; the prompt template + processor
  must follow Qwen3.5's format.

## 8. Open questions

- Option α vs β (see §4) — do we need true Qwen parity, or is weight-reuse continuous
  conditioning enough? The D1/D2 ablation answers a related but narrower question.
- If we ever want 3D *editing* (Qwen's reconstructive path): how to concatenate a
  reference asset's **sparse SLAT** latent into the flow token stream (sparse 3D ≠ dense
  2D `torch.cat`). This is the genuine hard part, deferred.

---

## 9. Implementation plan

Design principle: **compose, don't inherit.** The new model is a thin `nn.Module` that
HAS-A native VLM + the existing TRELLIS flow modules + the existing connector. We do NOT
inherit `blip3oMetaModel` (it's tangled with TA-Tok / codebook / `prepare_inputs_labels`).
We reuse the flow-loss loop verbatim from `blip3o_qwen.py`. **Connector unchanged.**

**v1 = VLM runs IN-LOOP, NO caching.** Running the VLM is irreducible (caching would still
have to run it in an offline script), so caching saves no code — it only ADDS a storage/load
layer. Therefore v1 just runs the VLM in `forward()`; mask comes naturally from the
processor's `attention_mask`. Freeze vs train = a flag (`no_grad`+`requires_grad`) around the
shared `encode_cond()`. **Caching is fully deferred** — a later offline wrapper around the
*same* `encode_cond()` if compute/env decoupling ever demands it (purely additive).

### Phase 0 — env validation  🔴 GATE (do before anything else)
- `cp -a` the `blip3o_trellis` env → bump `transformers` to a version exposing `qwen3_5`
  (same version also covers `qwen3_vl`).
- Smoke test BOTH: `from_pretrained("Qwen/Qwen3-VL-2B-Instruct")` and `("Qwen/Qwen3.5-2B")`
  → run one `forward(..., output_hidden_states=True)` → assert `hidden_states[-1].shape[-1]==2048`
  → confirm torch / flash_attn don't break.
- **Exit criteria:** both load + forward clean in the new env. Nothing downstream until this passes.

### Phase 1 — model skeleton  `blip3o/model/language_model/trellis_native_vlm.py`
Holds (all by composition):
- `self.vlm` = backbone (`Qwen3VLForConditionalGeneration` / `Qwen3_5ForConditionalGeneration`,
  chosen by `config.vlm_model` — not hardcoded).
- `self.diffusion_connector = TRELLIS2Connector(2048, 1024)`  ← **reuse, unchanged**.
- `self.ss_flow / shape_slat_512 / tex_slat_512 / sc_vae_*` = `tr2_modules.build_*`  ← reuse.
- loss fns = `TRELLIS2FlowMatchingLoss` (ss=logitNormal, slat=uniform)  ← reuse.

`forward(input_ids, attention_mask, pixel_values, image_grid_thw, target_ss_latent,
target_shape_slat_512, target_tex_slat_512, tex_concat_cond, ...)`:
1. `out = self.vlm(input_ids, attention_mask, pixel_values, image_grid_thw, output_hidden_states=True)`
2. `hidden = out.hidden_states[-1]`  # (B, T, 2048) — includes text + image tokens
3. `cond_hidden = hidden`; `cond_key_mask = attention_mask.bool()`   (cond_slice = "full" only)
4. `if self.config.detach_cond: cond_hidden = cond_hidden.detach()`
5. `cond = self.diffusion_connector(_mask_drop(cond_hidden))`; `sdpa_mask = cond_key_mask[:,None,None,:]`
6. 3-stage flow loss — **copy the loop from `blip3o_qwen.py:223–340` as-is** (cond_max_length
   cap, dtype cast, ss/shape/tex stages, weights). **Drop the CE block entirely.**
7. `loss = Σ flow_losses`  (no CE term).

**Smoke test:** dummy batch → returns finite loss; grads reach connector (+ VLM iff not detached).

### Phase 2 — freeze logic + config
- New config/args: `vlm_model` (str), `detach_cond` (reuse), `train_vlm` (bool: D1=false / D2=true).
- `blip3o/train/train.py`: delete codebook/CE freeze branches; set `self.vlm.requires_grad_(train_vlm)`;
  connector always trainable; TRELLIS flows per existing `mm_flow_crossattn/selfattn/last40` flags
  (keep that machinery — it's how SS flow gets partially unfrozen).
- D1: VLM frozen + (recommended) not in optimizer → big memory save. D2: VLM trainable —
  under the repo NO-OFFLOAD policy this requires ≥2 GPUs with ZeRO-2 (no CPU offload);
  the 1×80GB unfreeze probe (`tests/test_fla_backward.py`) measures 39 GB peak at BS=1
  so D2 fits 1 GPU at BS=1 (340 ms/step) but BS=2 requires DDP across 2 GPUs.

### Phase 3 — data / collator (REUSE the native processor — do NOT hand-roll)
**Principle: use `AutoProcessor.from_pretrained(vlm_model)` (`Qwen3VLProcessor` /
`Qwen3_5Processor`) for ALL vision/text→tensor work.** It natively handles chat-template,
vision-placeholder insertion, `<image_pad>` expansion by grid, **multi-image, video,
padding, `image_grid_thw` / `video_grid_thw`**. The collator is a thin wrapper, NOT new
preprocessing code.

Thin collator:
1. Build native `messages` per example — `content` = list of `{"type":"image"/"video"}`
   items (0 / 1 / N) + `{"type":"text", "text": caption}`. **Do not hand-write "Picture N:"**
   (that's Qwen-Image-Edit's private convention; the native template inserts
   `<vision_start><image_pad><vision_end>` itself).
2. `texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
   then `out = processor(text=texts, images=imgs, videos=vids, padding=True, return_tensors="pt")`
   → `input_ids / attention_mask / pixel_values / image_grid_thw (/ video_grid_thw)`.
3. Attach 3D targets from the existing prep: `out["target_ss_latent"] = …` etc.

- **One code path covers text→3D / single-img→3D / multi-view→3D / video→3D** (just vary the
  `content` items; processor dispatches). Multi-image & video are **free** on the semantic path.
- **Reuse** 3D target prep (`prep_data_ss_latents`); **drop** `codebook_prep`.
- Cond extraction (in model fwd): `hidden_states[-1]`. **v1 = keep FULL hidden, drop nothing.**
  `drop_idx` (dropping the constant system-prompt prefix) is **deferred** — it's downstream of
  the template choice and pure polish; a frozen VLM's cross-attn tolerates the constant prefix.
  Revisit later by auto-computing the prefix length from the tokenizer (never hardcode).
  Note: using native `apply_chat_template` (needed for vision-placeholder insertion) means a
  template prefix WILL exist — that's fine; keep any system prompt minimal.

### Phase 4 — scripts + overfit validation
- `scripts/train_q3vl_D2.sh`, `scripts/train_q35_D2.sh` — byte-identical except `--vlm_model`,
  `--detach_cond false`, `--train_vlm true`.
- **Validate on single-asset overfit FIRST** (mirror existing overfit recipe: ZeRO-2 NO-offload,
  small `save_steps`). Confirm clean geometry before scaling — same bar as
  [[project_image_overfit_ss_todo]].

### Phase 5 — inference
- `inference.py` variant: VLM forward → `hidden[-1]` → connector → **TRELLIS flow sampling**
  (reuse TRELLIS sampler) → SC-VAE decode → mesh/PBR. **No AR decode.**

### Phase 6 — backbone A/B
- Run `q3vl×D2` vs `q35×D2`; compare geometry/fidelity. Expand to 2×2 (`×D1`) only if the
  frozen-encoder comparison is wanted.

### Reuse / New / Drop (summary)
- **Reuse unchanged:** `connector.py`, `tr2_modules.py`, `loss.py`, the flow-loss loop, TRELLIS
  sampler, Trainer/ZeRO scaffolding, 3D target prep.
- **New:** `trellis_native_vlm.py`, VLM-processor collator, 2 scripts.
- **Drop/bypass:** TA-Tok, `<I*>` codebook + CE, `prepare_inputs_labels_for_multimodal` (TA-Tok
  injection), `codebook_prep`, GRPO/trl.

### Top risks
1. 🔴 env / transformers (Phase 0 gate).
2. D2 memory: 2B VLM joint-trained ⇒ 39 GB / GPU at BS=1 (measured); fits 1 GPU but BS=2
   needs DDP across 2 GPUs (NO offload allowed per repo policy). D1 much cheaper.
3. Verify `hidden_states[-1]` is the 2048-d LLM hidden (not a vision-merger output) for BOTH backbones.
4. Processor plumbing into the collator (image_grid_thw, padding) — main new-code surface.

### Build decisions (native-reuse audit — don't reinvent)
Adopted (use native / existing, don't hand-roll):
- **Collator** → native `AutoProcessor` (`apply_chat_template` + processor); covers
  template, vision-placeholder insertion, multi-image, video, grid_thw, padding.
- **VLM loading** → `AutoModelForImageTextToText.from_pretrained(vlm_model)` (backbone-agnostic;
  one line for both backbones). Fallback to the specific class if auto-mapping misses `qwen3_5`.
- **Model class** → subclass **`PreTrainedModel`** ⇒ free `save_pretrained` / `from_pretrained` /
  Trainer / DeepSpeed integration (no hand-rolled checkpointing).
- **3-stage flow loss** → factor into a **shared helper** (`flow_heads.py`) imported by both
  `blip3o_qwen.py` and `trellis_native_vlm.py`; do NOT copy-paste (avoids drift).
- **Inference sampling** → TRELLIS native sampler + guidance (match training schedule); no custom
  diffusion loop.
- **`encode_cond()`** → a single shared method (load VLM + processor inputs → `hidden_states[-1]`
  + mask). The model `forward()` calls it in-loop; any future caching script calls the SAME method.
- **CFG negative-cond fix (review #2)**: the training-time unconditional is `connector(0)`
  (mask_drop zeros the connector INPUT; identity-init LayerNorm ⇒ `connector(0) ≠ 0`). Old
  inference used `zeros_like(cond)` — **109 units** off (test_cfg_null_cond.py). Fix: canonical
  `flow_heads.null_cond_like(connector, hidden)`; `test_overfit_infer.py` fixed; native model has
  `cond_and_null()` so any `generate()` is correct by construction. (blip3o_qwen_inference.py / Sana
  reference was already correct — it nulls the connector input.)
- **flow_heads is now the TRUE single source**: `blip3o_qwen.py` migrated off its inline loop to
  `flow_heads.compute_cascade_flow_loss` — verified NUMERICALLY IDENTICAL (tests/test_flow_heads_equiv.py
  diff=0), import chain + native overfit still green. So both models share one cascade-loss impl.

Deferred (NOT in v1):
- **Cond caching** — adds no savings in code (VLM must run regardless) and a storage/load layer.
  v1 runs the VLM **in-loop**; caching is a purely-additive offline wrapper around `encode_cond()`
  added later ONLY if compute or env-decoupling demands it.
- **D2 / training the VLM** (joint full-FT or PEFT-LoRA) — deferred. First overfit smoke-test runs
  the VLM **frozen + in-loop** (Qwen-Image-Edit also freezes Qwen2.5-VL); unfreeze is a flag flip
  but needs ≥4 GPU or VLM-LoRA, so not the first run.
- **`drop_idx` / template-prefix dropping** — v1 keeps full hidden; revisit as cheap polish.
