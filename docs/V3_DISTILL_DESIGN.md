# V3 — Pure-Qwen conditioning via condition-swap distillation

**Goal.** Remove DINOv3: condition the TRELLIS.2 cascade purely on `connector(Qwen3.5-VL hidden)`
(image / multi-image / text → 3D in one cond path). v1-dual failed by gradient starvation
(DINOv3 never dropped → Qwen residual ≈0 → gate ≈0); v2c (anchor-dropout) fixed the competition
but kept the sparse task-loss-only signal. v3 changes the learning signal itself: **distill the
DINOv3-conditioned behavior of the frozen flow into the Qwen-conditioned path.**

**Decided 2026-06-10 (weikaih):** UniVideo's staged schedule (skeleton) + PEA-Diffusion's two-level
KD (engine) + our ground-truth flow loss. KD active in Phase A only (default).

---

## 1. Recipe

One training step (image/multi-image task), same sample, same noise, same `t`, same frozen flow:

```
                     ┌─ teacher cond: DINOv3(renders)            (V×1029 tok, the flow's "native tongue")
 same rendered views ┤
                     └─ student cond: connector(Qwen-VL hidden)  (V×1024 tok after token raise)

 pass 1 (no_grad):  flow(x_t, t, dino_cond) → v_t  + hook'd block outputs h_l^T
 pass 2 (grad):     flow(x_t, t, qwen_cond) → v_s  + h_l^S

 L = L_flow + λ_v·L_v + λ_f·L_f
   L_flow = ‖v_s − v_gt‖²                       ground truth (lets student EXCEED teacher; OUR addition —
                                                 PEA/Scaling-Down are pure-KD, they had no target-domain GT)
   L_v    = ‖v_s − v_t‖²                        output-level KD   [Scaling-Down 2503.19897]
   L_f    = Σ_l ‖h_l^S − h_l^T‖²                feature-level KD on 4–6 evenly-spaced blocks
                                                 [PEA 2311.17086 — its ablation: THIS is the workhorse;
                                                  output-level alone is marginal]
```

Defaults: `λ_v=1.0, λ_f=0.5` (all MSE, same scale; sweep later). Applied per stage (SS + shape + tex).

Mechanical notes:
- **Teacher pass runs FIRST** (no_grad, activations freed) → peak memory ≈ unchanged; ~+35% step time.
- KD terms computed in **fp32** (teacher runs bf16; don't let precision noise into the loss).
- **CFG-dropped samples skip KD** (uncond direction is the same flow for both — no signal); they keep L_flow.
- **text→3D batches have no teacher** (no render → no DINOv3) → L_flow only. Pipeline naturally compatible.
- Two passes share `x_t`/`t`/coords → teacher/student block outputs align **element-wise**; token-count
  mismatch (1029 vs 1024/view) is irrelevant (cross-attn output shape follows voxels, not cond tokens).
- SLAT blocks output SparseTensor → compare `.feats`.
- Per-layer distill curves = free diagnostics: early-block gap = low-level structure not getting through
  (token/capacity); late-block gap = refinement issues. `train/distill/{v,f}_{ss,shape,tex}` in wandb.

## 2. Architecture

- **No dual_router** — α-variant shape: `Qwen-VL (frozen) → last-hidden → connector (trained) → the flow's
  ORIGINAL cross-attn`. DINOv3 exists only inside the teacher pass (and is deleted at inference).
- **Token raise: 1024/view** via `target_tokens_per_view=1024` (knob already in `vlm_collate.py`;
  upscales 512² renders → 1024² → 32×32 Qwen grid = 1:1 with the DINOv3 teacher grid). mv4 ≈ 4k tok +
  text < 8192 flow cap. Validated 2026-06-10: the dual ckpt handled 4×1024-tok cond cleanly at inference.
- **Teacher = single-view DINOv3 only** (S1 is I1-only; KD never coexists with IM/T — see §3). The
  multi-view-concat-teacher question is moot for v3.
- Init: flow = pretrained TRELLIS.2 (NOT a dual ckpt); connector = fresh.

## 3. Stages (= UniVideo S1→S2→S3; FINAL per weikaih 2026-06-10: S1/S2 are SINGLE-IMAGE-ONLY)

| | data | trains | losses | teacher |
|---|---|---|---|---|
| **Stage 1** (align) | **I1 only** | connector ONLY (flow ❄, Qwen ❄) | `L_flow + 1.0·L_v + 0.5·L_f` | SAME frozen flow + **single-view** DINOv3 (2 passes, zero extra memory) |
| **Stage 2** (quality) | **I1 only** | connector + flow last-N | `L_flow` (KD off, λ=0) | none in training; dino-teacher = drift-monitor EVAL |
| **Stage 3** (tasks) | I1 + IM + T mix | connector + flow last-N | `L_flow` | none — IM/T ride the already-aligned Qwen path |

**Why I1-only for S1/S2 (weikaih's call):** nail the core capability — Qwen single-image conditioning —
in the cleanest possible setup first; add multi-view and text as task extensions afterwards.
**Clean side-effect: the multi-view-teacher question VANISHES.** KD is only active in Stage 1, and Stage 1
is I1-only → the teacher is always single-view DINOv3 (the stock flow's exact pretraining distribution,
zero OOD risk). By the time IM enters (Stage 3) KD is off, so an IM teacher is never needed. The Step-1
pre-experiment (4-view concat teacher on stock flow) is therefore MOOT for v3 — optional curiosity only.
(Its single-view control already ran clean — warrior 1975 coords / 850k verts — which validated the
manual teacher-cond plumbing itself.)

Why KD off from Stage 2 on: S1's problem is "connector can't speak the flow's language" → dense teacher
signal; S2's problem is "flow must adapt beyond mimicry" → KD would pin the student inside the teacher's
shadow. Distill-then-finetune is the standard pattern; no KD ⇒ **no frozen teacher copy ever needed**
(S1 shares the module; a separate ~8 GB copy only if KD is re-enabled on drift evidence).

Safety nets for S2/S3: (i) drift monitor — every N k steps render dino-teacher vs qwen-student
side-by-side (eval-only); (ii) on drift, re-enable small λ (~0.1) as a regularizer — only then build the
frozen-teacher-copy path; (iii) λ annealing available as a middle ground. Plus last-N-only unfreezing + EMA.

Expectation calibrated by literature: UniVideo ultimately fine-tunes its DiT — **Stage 2 is likely
needed.** Connector-only-forever precedents are only ELLA (large timestep-aware resampler) and PEA
(6M+KD, semantics-only). Stage 1's job: align the connector + measure how far KD stretches it.
Text→3D note: deferred to S3 by design; the VLM is frozen throughout so no text understanding is lost —
the connector picks up the text task in S3 on top of an already-aligned image mapping.

## 4. Paper grounding

| paper | what we take | what we verified |
|---|---|---|
| **UniVideo** (2510.08377) | the staged schedule + the architecture shape (frozen MLLM → last-hidden → MLP connector → DiT) + full encoder REPLACEMENT end-state | 3 stages: S1 connector-only → S2/S3 connector+MMDiT FT; Hunyuan's native text encoder REMOVED entirely |
| **PEA-Diffusion** (2311.17086) | the KD engine (`λ_F·intermediate-feature + λ_L·output`), adapter-only training | ablation: feature-KD >> output-KD; "aligning the text encoder is insufficient" |
| **Scaling-Down** (2503.19897) | output-level KD legitimacy; warning that naive ENCODER-feature alignment mode-collapses | frozen backbone, teacher/student cond, match predictions; 50× smaller encoder ≈ original |
| **ELLA** (2403.05135) | the connector+task-loss baseline; TSC (timestep-aware connector) as a Phase-A capacity upgrade option | connector-only forever, no distill |
| our α variant | proof that `connector + flow last40 + flow-loss` is CLEAN on our task | = the Phase-B fallback's known-good region |
| our capacity ladder | connector-only + task-loss-only = GARBAGE | = why Phase A needs the KD engine at all |

## 5. Implementation plan (3 files)

1. **`trellis2_blip3o/flow_heads.py`** — `compute_cascade_flow_loss(..., teacher_cond=None, distill_v_weight,
   distill_f_weight, distill_f_blocks)`: per stage, teacher pass first (no_grad, fp32-detached hook
   captures), student pass, three losses, per-stage logging. Forward hooks on selected blocks (dense SS:
   tensor outputs; SLAT: `.feats`).
2. **`blip3o/model/language_model/trellis_native_vlm.py`** — build teacher cond from `dino_images` (the
   unified collator already emits it) via the existing `_dino_extractor`; v3 config: dual OFF, `distill_dino`
   ON; pass teacher_cond through.
3. **`train_native.py`** — flags `--distill_dino --distill_v_weight 1.0 --distill_f_weight 0.5
   --distill_f_blocks <spec>` + `--target_tokens_per_view 1024` (DataArguments → collator) + config echo.

Phase A: NO torch.compile on flows (hooks vs compile — correctness first). NO-OFFLOAD policy stands.

## 6. Validation ladder (cheap → expensive, each with go/no-go)

1. **Step 1 — multi-view-teacher pre-experiment (30 min GPU):** frozen pretrained flow + 4-view concat
   DINOv3 cond → clean output? decides IM teacher form.
2. **3a forward smoke** (1 GPU): losses finite, teacher truly no_grad, memory/step-time as expected.
3. **3b invariant test:** feed DINOv3 cond to BOTH paths → `L_v ≈ 0` and `L_f ≈ 0` exactly. One line;
   catches every mask/order/hook bug.
4. **3c killer overfit (7 min):** connector-only + KD on the single-asset overfit. History: connector-only
   + task loss = GARBAGE. If KD flips it to clean → thesis proven for pennies. Either way informative:
   garbage again → capacity is real → Phase A starts at connector+last15 instead.
5. **Phase A real run** (ready_v1 80k, 2–8 GPU): watch distill curves (the direct "distance to DINOv3"
   metric), teacher-vs-student renders every N k steps, **≥3-seed rule** for any quality claim, log to
   `docs/RESULTS.md`.

## 7. Risks

- Multi-view teacher OOD → step 1; fallback single-view teacher.
- Connector capacity (3.1M vs spatial structure) → 3c decides; fallback last-N; ELLA-style timestep-aware
  TSC as a connector upgrade before touching the flow.
- Hooks × compile → compile off in Phase A.
- bf16 KD noise → fp32 loss computation.
- Teacher copy memory (Phase B + KD fallback only) → ~8 GB bf16 for 3 stages; only built on drift evidence.
