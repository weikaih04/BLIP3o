# VLM-side 3D Pretraining (MolmoAct2-inspired) — 3D-understanding + editing pretrain stages

**Decision (2026-06-20, weikaih):** add **two new VLM-side pretraining stages** before/around the
flow training — **(VP1) 3D-understanding pretraining** and **(VP2) editing pretraining** — so the
Qwen backbone becomes *3D-literate* instead of a frozen 2D reader. Motivated by MolmoAct2's recipe
(discrete-token VLM + flow-matching expert via per-layer KV). This doc is the design + the honest
open questions; the conditioning-mechanism half (per-layer KV) is also the immediate next experiment.

Related: [[QWEN_HANDOFF_DESIGN]] (DINOv3→Qwen handoff, the unified-model §7/§8), [[3D_EDITING_ROADMAP]]
(Plan-1 renders vs Plan-2 native-3D), [[QWEN35_VLM_DESIGN]] (D1/D2 freeze axis), [[RESULTS]].

---

## 0. TL;DR
- The real question is **"does VLM-based 3D pretraining help final gen?"** — NOT a literal
  "depth-token = SS" mapping. SS / compressed-SLAT / depth are just *choices of which token to
  pretrain on*; the hypothesis is "**train the VLM on a 3D objective → 3D-literate KV → better gen**."
- **Status: plausible + MolmoAct2 ships it, but NOT yet proven by an isolating ablation** (we did not
  find a MolmoAct2 "discrete-pretrain on/off → final continuous gen" number). Treat as a hypothesis to
  earn its keep, per our `feedback_results_log_convention`.
- **Cheapest test first:** the *conditioning mechanism* (per-layer KV, frozen Qwen) is independent of
  the pretraining and much cheaper. Run it first (Step 1). Only add VP1/VP2 if frozen-Qwen KV
  underdelivers.
- **The strategic payoff:** a discrete 3D tokenizer + VLM-3D pretraining is **one piece of infra that
  pays double** — (a) 3D-literate KV for *generation*, and (b) a native 3D *input* modality for
  *editing* (the Plan-2 bridge). That reframes the Plan-1-vs-Plan-2 bet: if VP1 helps gen, it also
  unlocks native editing.

---

## 1. The MolmoAct2 lesson (what actually transfers)
MolmoAct2 = **discrete-token VLM (Molmo2-ER) + flow-matching continuous-action expert, joined by
per-layer KV-cache conditioning**, dual loss `L_LM + L_flow`. Verified details (arXiv 2605.02881):

| MolmoAct2 piece | What it is | Transfers to us? |
|---|---|---|
| **OpenFAST action tokenizer** | DCT (frequency transform) → quantize → BPE, 2048 vocab, on 1-s action chunks | **NO** — FAST exploits *temporal* smoothness of trajectories; our 3D output is *spatial/sparse*. Use **VQ-VAE on SS / compressed SLAT**, not FAST. |
| **Discrete action-token pretraining** (Stage 3, 200K steps, next-token `L_LM`, 90% robot) | trains the VLM backbone on the output modality → 3D/action-aware representations | **YES — this is the core idea** (= our VP1). |
| **Per-layer KV conditioning** | expert layer ℓ cross-attends to VLM layer-ℓ K,V via learned adapters `P_K/P_V`; **detached** (knowledge insulation) | **YES — directly replaces our read-once cross-attn** (the multi-view bottleneck). |
| **Flow expert** (DiT, 36 layers = VLM depth), `L_flow` on velocity | continuous high-precision output, the only path at inference | We already have this (TRELLIS flow). |
| **Depth reasoning tokens** (10×10×128 VQ, AR) | coarse geometric reasoning trace | optional; our SS occupancy is the natural analog if we want a reasoning trace. |

**Inference fact (verified):** at deployment the **discrete path is dropped** — the flow expert
generates from the VLM's per-layer KV; the VLM does **not** decode action tokens. So the discrete
tokenizer's *only* job is **training-time representation shaping**. That is exactly why VP1 (below)
is a *pretraining* stage, not an inference component.

---

## 2. Architecture pieces

### 2a. Per-layer KV conditioning (the gen mechanism — independent of VP1/VP2)
Today BLIP3o feeds `connector(Qwen final-hidden)` to TRELLIS's cross-attn — **one static K/V, read
once per block**. Our own IM diagnostic (RESULTS 2026-06-16) blamed exactly this for multi-view
mushiness ("read-once cross-attn → blurs views"; cited Qwen-RobotWorld / MolmoAct2 per-layer KV /
EVA-01). The fix:
- For each TRELLIS DiT block `b`, cross-attend to **Qwen layer ℓ(b)'s K,V** (projected by learned
  `P_K/P_V` into the block's cross-attn width), instead of the single connector output.
- **Layer mapping**: Qwen3.5-2B (~28 layers) vs SS flow (30 blocks) — depths differ, so map evenly
  (MolmoAct2 sidestepped this by designing the expert to match VLM depth; we can't, so this is a
  design knob).
- **Detach**: Qwen is frozen → knowledge insulation is satisfied for free (flow loss never enters Qwen).
- **Cache**: extend `trellis2_blip3o/vlm_cache.py` from "cache final hidden" → "cache per-layer KV".
- **Compose with FUSION**: today's FUSION (`[raw DINO; Qwen]` in one cross-attn) becomes the
  per-layer version — inject Qwen KV per layer + keep the DINO structural rail.

### 2b. Discrete 3D tokenizer — the **Kyvo-style route** (verified against literature + Kyvo code, NOT FAST)
The plan is a **second-stage latent tokenizer on top of the FROZEN TRELLIS.2 VAE**:
`x → E_VAE → z (SLAT) → E_q → quantize → q → D_q → ẑ → D_VAE → x̂`. The AR model only predicts the
middle discrete tokens. The *architecture* (densify → 3D conv → VQ → ~512 tokens) is from **Kyvo
(arXiv 2506.08002)**'s complex-shape enrichment, plus **UniLat3D (2509.25079)** for densify+unified-latent.

> **Reality check on Kyvo's code (github.com/AadSah/kyvo) — NOT a turnkey clone:** its *released*
> VQGAN is the **stock 2D-image taming-transformers** (CompVis) used to tokenize RENDERS; its main 3D
> scenes are serialized JSON → **categorical/numeric** tokens (shape category + numeric pose/size), and
> **color/material are categorical — there is NO texture VQ-VAE**. The "TRELLIS-SLAT → 512-token 3D
> VQ-VAE" is the *paper's* complex-shape enrichment, not a ready module. **Implication:** we build the
> tokenizer ourselves by **adapting taming-transformers' VQGAN 2D→3D** (reuse its battle-tested
> `Encoder/Decoder/VectorQuantizer/codebook/recon+commitment loss`; swap `Conv2d→Conv3d`, `H×W→D×H×W`),
> fed the densified TRELLIS SLAT. **Texture: not in Kyvo → mirror the shape-VQ ourselves.** Pretrained
> HF codebooks are 2D-image only, not reusable for our 3D shape/tex.

**Recipe (densify → dense 3D conv VQ-VAE), with our REAL TRELLIS.2-512 numbers:**
```
sparse SLAT (active voxels in a 32³ grid, ×32 ch)        # NOTE: 512/1024 = INPUT voxel res;
  → densify → dense 32³×32 (+ occupancy-mask channel)    # SLAT grid = res/16 (f16 VAE):
  → 3D conv U-Net, 2× stride-2: 32³→16³→8³               #   512→32³, 1024→64³  (NOT 64³×8 — that's Kyvo's TRELLIS-1)
  → VQ (codebook ~8192) at 8³                            # plain VQ is the demonstrated baseline (Kyvo)
  → 8³ = 512 discrete tokens / modality                  # Kyvo: 512 tok, ~40× reduction
```
- **Densify = scatter** sparse (coord, feat) into a zeros grid + an **occupancy-mask channel** (so the
  conv distinguishes "real 0 feature" from "empty"). Decode side knows the coords from the **SS stage**,
  so the feature tokenizer only (de)compresses features on given coords; **structure is SS's job**.
- **Per modality, separate codebooks:** SS (structure/occupancy — near-discrete, may skip the VAE
  tokenizer), **shape SLAT**, **tex SLAT**. Coarse-to-fine AR stream `<ss>…<shape>…<tex>…`.
- **Token budget (empirical, per-modality):** 512 (8³) is the Kyvo-proven start and is **fine for VP1's
  purpose** (tokens are training-only, dropped at inference — see §1). **Geometry compresses well at 512;
  texture is higher-entropy → 512 likely captures only coarse appearance** → for faithful texture
  (VP2/gen) expect to need more (16³=4096 / bigger codebook / RVQ). Decide by a **rate-distortion curve
  on final-decode quality**, not latent MSE.

**Corrections to earlier drafts (verified wrong):** NOT sparse-conv (densify + dense conv is the field
standard — Kyvo, UniLat3D both densify to a small grid); NOT naive average-pooling the learned latent
(off-manifold); **plain VQ-8192 is the demonstrated baseline** (FSQ only as a stability hedge if VQ
collapses, not the default).

**Losses:** `λ_z‖z−ẑ‖₁ + λ_x·(SAMPLED final-decode distance) + λ_q·commitment`. Target = `D_VAE(z)`
(the frozen VAE's own decode), not raw `x` — don't ask the tokenizer to fix the VAE's own error. The
final-decode loss is expensive in 3D (mesh/PBR extraction) → **sample it / low weight**, latent-L1 carries.
**Optional robustness — token corruption:** randomly corrupt ~5–10% of tokens when training the latent
decoder so it tolerates AR prediction errors (a single wrong token shouldn't blow up the decode).
Precedent: MolmoAct2 injects 10% noise into teacher-forced depth tokens. Not in Kyvo → treat as a hedge,
not established necessity.

**Open fork (geometry+texture):** Path A (this section) tokenizes shape-SLAT and tex-SLAT **separately**
(reuses TRELLIS.2, closest to Kyvo; risk = geometry-texture alignment across two streams). Path B = a
**UniLat3D-style unified geometry+appearance latent** (16³×32 dense) → ONE token stream (no geo-tex
mismatch, fewer tokens) but needs a new unified encoder. **VP1 uses Path A** (tokens are training-only,
no need for gen-grade unification); reserve Path B for VP2/real generation.

---

## 3. The two new VLM-side stages (the decision)

### VP1 — 3D-understanding pretraining  (= MolmoAct2 Stage 3 analog)
**Make Qwen 3D-literate.** Next-token `L_LM` on the discrete 3D tokens (§2b) for each asset; no flow.
- Inputs: render(s)/caption → predict the asset's 3D tokens (image/text→3D-token), reusing `ready_v1`
  (renders + 3D latents already prepped).
- Output: a **"Qwen-3D" checkpoint** (analog of Molmo2-ER) whose per-layer KV now carries geometry.
- Cheap (next-token loss, no sampling), and the *stable* way to train the VLM vs backprop-flow-into-VLM
  (D2). Interleave original Molmo/Tulu-style text (anti-forget) as `MULTI_TASK_DATA.md` already supports.

### VP2 — editing pretraining  (the Plan-2 bridge)
**Give Qwen a native 3D *input* modality for editing.** Builds on VP1's 3D-literate Qwen.
- Feed the **input asset's 3D tokens** (encode mesh → voxelize → SS/SLAT → VQ tokens, via `data_toolkit`)
  **into the VLM** alongside the edit instruction; train it to understand/produce the edited target
  (token-level supervision and/or as conditioning).
- This is exactly the **Plan-2 "3D tokens into the VLM"** that `3D_EDITING_ROADMAP` listed as the
  north star. VP1's tokenizer + 3D-literacy is the prerequisite that makes VP2 feasible.

Then the **flow training** (per-layer KV, §2a) conditions on this 3D-literate Qwen; the high-fidelity
output is still the continuous cascade.

---

## 4. How it slots into the existing pipeline
```
                 NEW (VLM side)                          EXISTING (flow side)
  Qwen3.5  ──VP1: 3D-understanding pretrain──►  Qwen-3D ──per-layer KV──► TRELLIS flow cascade
              (discrete 3D-token next-token)        │     (§2a, detached)     SS→shapeSLAT→texSLAT
                                                     │
            ──VP2: editing pretrain───────────►  Qwen-3D-edit ─(input-asset 3D tokens)→ edit
              (input-asset tokens + instruction)
```
- VP1/VP2 produce the conditioning backbone; the cascade + FUSION + cascade-training (V3/Stage-2/3)
  stay as-is but now read a 3D-literate Qwen via per-layer KV.

## 5. Validation order (cheap-first — do NOT skip to the big build)
1. **Step 1 (cheapest, do first): per-layer KV with *frozen* Qwen** (§2a), SS-only minimal repro
   (mirror the dual_cond bring-up). Measure multi-view + text→3D. **If this alone fixes it → no VP1
   needed.** (Honors V3's "pure flow works; extras must earn their place.")
2. **Step 2 (only if Step 1 plateaus): VP1** 3D-understanding pretrain (SS tokens first, §2b-1) →
   per-layer KV. ≥3-seed eval vs Step 1, log to `RESULTS.md`.
3. **Step 3: add shape+tex tokens** (Kyvo densify+VQ, §2b) — finer geometry + appearance if SS-only
   grounding is too coarse. Texture likely needs a bigger budget than 512 (set by rate-distortion).
4. **Step 4: VP2** editing pretrain → native-3D editing (Plan-2). North-star-scale; gate on Step 2 win.

## 6. Reuse / New / Drop
- **Reuse:** `vlm_cache.py` (extend to per-layer KV), `data_toolkit` voxelize + SS/SLAT encode,
  TRELLIS SLAT-VAE (the latent to VQ), `MULTI_TASK_DATA.md` mixture (anti-forget text), the cascade.
- **New:** a 2nd-stage 3D tokenizer — **adapt taming-transformers VQGAN (CompVis) 2D→3D**, fed densified
  FROZEN-TRELLIS SLAT (§2b); build a SEPARATE one for texture (not in Kyvo). Plus per-layer KV adapters
  `P_K/P_V` + layer map, VP1/VP2 data recipes + next-token heads
  (extended vocab: SS/shape/tex codes + `<ss>/<shape>/<tex>` markers, MolmoAct2-style train only new
  embeddings + output head).
- **Drop/avoid:** FAST-style tokenizer (action-specific); **backprop flow-loss into Qwen (D2)** —
  MolmoAct2's knowledge-insulation deliberately avoids it; use VP1's next-token loss instead.

## 7. Open decisions (for weikaih)
- (i) Tokenizer — **RESOLVED (Kyvo route, §2b):** frozen TRELLIS VAE → densify → dense 3D conv VQ-8192
  → ~512 tok/modality, shape+tex **separately** (Path A). Remaining sub-decision: texture token budget
  (512 vs 16³) set by rate-distortion; unified geo+tex latent (Path B, UniLat3D) deferred to VP2.
- (ii) VP1 supervision: predict 3D tokens from **renders** (image→3D-token) vs **text** vs both?
- (iii) Per-layer KV: unfreeze a *little* of Qwen during VP1 (so KV truly shifts) vs keep frozen +
  rely on the connector/adapters? (D1/D2 axis, but via next-token not flow loss.)
- (iv) VP2 injection: input-asset 3D tokens **in the VLM token stream** (Plan-2) vs keep the
  SLAT rail **DiT-only** (Plan-1) and only render-condition the VLM — i.e. is VP2 worth it over the
  already-decided render route, given renders miss interior/occluded geometry?

## 8. Honest caveats
- **Unproven:** "VLM-based 3D pretraining helps gen" has no isolating ablation we could find. Step 1
  (per-layer KV) may already capture most of the win without VP1.
- **North-star cost:** native-3D editing (VP2/Plan-2) is the heavy, long-horizon bet; `3D_EDITING_ROADMAP`
  already argues renders (Plan-1) suffice for most edits. VP2 earns its place mainly on hard
  interior/occluded edits that 2D renders can't see.
- All quality claims follow the `RESULTS.md` ≥3-seed + dated convention.
