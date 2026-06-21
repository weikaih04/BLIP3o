# DINOv3 → Qwen handoff: how to reach pure-Qwen conditioning

**Goal.** Today: DINOv3 anchor + Qwen VL branch condition the TRELLIS.2 cascade *simultaneously*,
with Qwen *aligning to* DINOv3 (the anchor/teacher). End state: **drop DINOv3, condition purely on
Qwen** (a real VLM → image / multi-image / text in one cond path). This doc analyzes how to get
there cleanly, including the two framings weikaih raised: **(B) follow Qwen-Image-Edit**, and
**(A) keep TRELLIS.2's form but eventually remove DINOv3**.

Status grounding (2026-06-09): both-CFG'd packed inference validated clean (warrior coords 2144 /
792k verts; teapot coords 2226 / 657k verts). Trained gates (EMA, ckpt-30000) are tiny —
SS mean|g|=0.009 (max 0.054), shape 0.003, tex 0.001 → **Qwen contributes ~1%; the model is still
~99% DINOv3-driven.**

---

## 0. The core obstacle: why "pure Qwen" is not automatic

The gate is stuck near 0 **not because of a bug** — it's a training dynamic:

- Each dual block computes `h' = h + cross_attn_dino(h, K_dino) + gate·cross_attn_qwen(h, K_qwen)`.
- DINOv3's cross-attn is **pretrained** (TRELLIS.2, ~800k steps) and `K_dino` is fed **on every
  step, never dropped** (`set_dino` in `trellis_native_vlm.py:570` is unconditional; only the Qwen
  cond gets `mask_drop` p=0.1 in `flow_heads.py`).
- So DINOv3 already explains the denoising target almost perfectly → **the residual left for Qwen
  to explain is ~0 → the gradient into the Qwen branch is ~0 → gate barely grows.** Rich-get-richer.

**Consequence:** if we just delete DINOv3 at inference today, we fall off a cliff — the main path
expects `K_dino` and Qwen (gate 0.01) is far too weak to drive generation.

**Linchpin (true for every route below):** to give Qwen a gradient, we must **reduce DINOv3's
explanatory power during training → DINOv3 dropout.** When `K_dino` is dropped, the only way to
lower the loss is through Qwen → large gradient → gate grows, and `cross_attn_qwen` learns *real*
conditioning instead of a cosmetic correction.

---

## 1. The structure-capacity question (the real quality risk)

DINOv3 and Qwen carry **different kinds** of information:

| source            | tokens (this run)     | nature                                          |
|-------------------|-----------------------|-------------------------------------------------|
| DINOv3 ViT-L/16   | 1029 / view (1024+5)  | **dense spatial** grid — strong geometry prior  |
| Qwen VL connector | 282 (sv) / 809 (mv4)  | **semantic** tokens — fewer, not a clean grid   |

For pure-Qwen, Qwen must convey the **structure** DINOv3 currently provides. Qwen2.5-VL *can* (it
keeps 2D position info + many vision tokens at high res), but our connector compresses to ~282
tokens — likely **structure-starved** for fine geometry. Levers:

- (a) **more Qwen vision tokens / higher-res visual input** (let Qwen carry structure);
- (b) keep a **cheap learned structural anchor** (Qwen-Image-Edit's VAE-latent route) instead of
  DINOv3 — smaller than DINOv3 but still a structural rail;
- (c) accept Qwen-only and measure.

This is independent of the dropout curriculum and is what decides whether pure-Qwen *quality*
matches TRELLIS, vs merely *runs*.

---

## 2. Qwen-Image-Edit — what it actually does, and what to borrow

Qwen-Image-Edit conditions on (source image, text) and encodes the **source image twice**:
1. **VAE latent**, concatenated to the noisy latent → low-level appearance / structure;
2. **Qwen2.5-VL vision tokens** → high-level semantics. Text also via the VL.

CFG: it uses **true CFG** (`true_cfg_scale>1` + a negative prompt). The "骚操作": the **source image
is fed to BOTH the positive and negative pass** (image *not* dropped) — CFG sharpens only the **text
instruction** difference; recommended CFG≈1 for edits (high CFG distorts the source you want to
preserve).

**Mapping to us:** their *VAE-latent = structure* and *VL = semantics* is **exactly** our
*DINOv3 = structure* and *Qwen = semantics*. The difference in intent:

- Qwen-Image-Edit **keeps both forever** (editing = preserve source) → image stays in both CFG
  passes, only text is CFG'd.
- We want to **remove the structural one (DINOv3)** → the structural role must be *absorbed by Qwen*
  (more tokens) or handed to a *lighter* anchor.

So "follow Qwen-Image-Edit" literally ⇒ Qwen-VL is the **main** conditioner and the structural rail
is auxiliary/preserved — the **opposite** of our current "DINOv3 main, Qwen gated add-on."

---

## 3. The two routes (weikaih's two framings)

### Route A — keep TRELLIS.2's form, transition K/V source DINOv3 → Qwen

The "form" = cross-attention to image features; we only change *which* features feed K/V.

- **A1 — gated-additive handoff (current arch, cheapest).** Keep dino cross-attn + gated Qwen
  branch. Add a **DINOv3-dropout curriculum** (§4) + let the gate grow. At the end, DINOv3 is always
  null → delete it; the Qwen branch carries everything as a (now large) residual on the
  unconditional base. *Pros:* minimal change, reuses all our infra, the dino cross-attn's
  "unconditional base" (cross_attn_dino(null)) is valid because stock TRELLIS trained with image
  dropout. *Cons:* Qwen stays a *residual* form rather than the primary attention.
  **CFG:** joint (drop both) while both on → done; drop Qwen once pure.

- **A2 — feature-align & swap (cleanest reuse).** Train the Qwen connector so `K_qwen` lands in
  **DINOv3's feature space** (distill `K_qwen ≈ K_dino`, plus flow loss). Then removing DINOv3 =
  feed the Qwen-connector output straight into the **pretrained** cross-attn — no gated branch
  needed. *Pros:* reuses the strong pretrained attention head directly → best handoff quality.
  *Cons:* **token-structure mismatch** (dino 1029 dense-spatial vs Qwen 282 semantic) — the
  connector must emit ~1029 spatial-ish tokens; hardest engineering of the three.

### Route B — re-architect to Qwen-Image-Edit form (Qwen = main)

Qwen-VL = the **main** cross-attn (the driver, trained to be). The structural rail (DINOv3 now, or a
VAE-latent) = auxiliary, **kept in both CFG passes** (preserve, low CFG). To *remove* the rail,
Qwen must carry structure (more vision tokens, §1). *Pros:* the "correct" long-term form if Qwen is
the destiny; matches a proven editing model. *Cons:* heaviest — Qwen-as-main is ~from-scratch
conditioning, forfeiting the DINOv3-pretrained attention head start; longest training.

---

## 4. The linchpin recipe (DINOv3-dropout curriculum) — applies to A1 & A2

| phase | DINOv3 dropout p | what happens                                                            | watch                                  |
|-------|------------------|-------------------------------------------------------------------------|----------------------------------------|
| P1 (≈now) | 0.0          | dino always on; Qwen "aligns", gate creeps (≈0.01)                      | gate, both-on quality                  |
| P2    | ramp 0 → 0.3 → 0.5 | dino randomly absent → Qwen forced to carry cond → **gate grows fast**, qwen attn learns structure | gate↑, **dino-dropped eval** quality↑  |
| P3    | 0.8 → 1.0        | mostly/always pure-Qwen; short clean-up FT; then delete dino             | pure-Qwen ≈ both-on?                    |

Key eval to add: at each checkpoint, generate **with DINOv3 forced off** and track quality climbing
toward the dino-on number. That curve *is* the handoff. (CFG in the dino-dropped regime = drop Qwen,
standard.)

---

## 5. CFG per regime (settled)

- **Both-on (today):** joint CFG — pack `[K_dino, K_qwen]`, drop **both** in the neg pass; wrapper
  splits at `_n_dino` so each cross-attn gets its own null (`zeros` for dino, `connector(0)` for
  Qwen). *Validated clean.* Two **separate** nulls, concat→split — **not** merged.
- **Pure-Qwen (end state):** standard CFG — drop Qwen.
- **If we keep a structural rail (B / Qwen-Image-Edit):** rail in **both** passes (preserve), CFG
  only Qwen. For independent control of structure vs semantics, **InstructPix2Pix 2-scale CFG**
  (3 passes: ∅ / rail-only / rail+Qwen, two guidance terms).

---

## 6. Recommendation (for sign-off — not yet implemented)

1. **Start with A1 + the §4 dropout curriculum.** It's the minimal delta from where we are, keeps
   TRELLIS.2's form, reuses our dual infra, and is the cheapest path to *test whether Qwen can drive
   alone at all*. The only training change is: dropout `K_dino` with a ramping probability (mirror
   `mask_drop`, but on the anchor) + add the dino-dropped eval.
2. **In parallel, raise Qwen's visual token budget** (§1a) before judging pure-Qwen *quality* — else
   we'll wrongly conclude "Qwen can't do structure" when it was just token-starved.
3. **Hold B (Qwen-as-main) as the fallback** if A1's pure-Qwen quality plateaus below TRELLIS, or if
   we decide a permanent lightweight structural rail (VAE-latent) is acceptable (then it's a feature,
   not a crutch).
4. **A2 (feature-align)** only if we specifically want to reuse the pretrained cross-attn head and
   are willing to pay the token-structure-matching cost.

**Open decisions for weikaih:** (i) A1 vs B as the primary track; (ii) whether pure-Qwen is a hard
requirement or a permanent lightweight structural rail is acceptable; (iii) Qwen visual-token budget.

---

## 7. The unified model: any of {image, 3D asset, text} → 3D asset

Target = a single model like **Qwen-Image-2.0** (2026-02-10, unified gen+edit), but 3D-native. Inputs
are any interleaved combination of **text + image(s) + an input 3D asset**; output is always a 3D
asset (SS coords → shape SLAT → tex SLAT). Two routes, mirroring Qwen-Image-Edit's *VL + VAE-latent*:

**Route 1 — universal semantic = Qwen3-VL (the unifier).** Everything becomes VL tokens: text,
image(s), and **rendered views of the input 3D asset** (the VLM "sees" a 3D asset through a few
renders). VLMs are natively multi-image + interleaved-text, so "any combination of the three inputs"
is just "feed whatever tokens you have." This is the route that makes the model *unified*.

**Route 2 — structural rail = a latent that is the 3D analog of `image_latents`, source-dependent &
DROPOUT-able:**
- input **image** → DINOv3 features (2D structure anchor, current), injected via **cross-attn**;
- input **3D asset** → its **SLAT latent** (encode the asset through the TRELLIS VAE), injected
  **in-context** (concatenated alongside the noisy output SLAT — they share the SLAT space), exactly
  the role `image_latents` plays in Qwen-Image-Edit;
- **text only** → no rail.

**Injection-method principle (why image uses cross-attn but a 3D asset uses concat).** Qwen-Image-Edit
can use a single VAE-latent-concat for the structure because it is *image→image* (input & output share
the VAE space). The rule: **same-space input → encode + in-context concat; cross-domain input →
cross-attn lifting.** So 3D-asset→3D (both SLAT space) → SLAT-latent concat ✓; image→3D (image vs 3D,
cross-domain) → DINOv3 cross-attn (you cannot concat an image latent into the 3D SLAT). The two
structural-injection methods coexist, chosen by whether the input shares the output space. The semantic
route (VL) is uniform across all inputs. (Aside: in QIE the input image is encoded **twice** — Qwen2.5-VL's
own ViT for semantics + the VAE for the concat latent; the ViT is Qwen's, not CLIP/SigLIP.)

**Training: stochastically drop the structural rail** → one model covers every mode:
text→3D (rail dropped = pure gen) · image/multi-image→3D (DINOv3 rail) · 3D-asset edit (input-SLAT
rail kept + CFG the Qwen instruction) · any mixture (Qwen reads all tokens; add a rail per available
structural source). This is the *same* dropout move as §4 — it simultaneously buys removable-DINOv3,
unified gen+edit, AND multi-input.

**CFG auto-switches by rail presence** (like the unified image model): no rail → CFG the cond
(generation); rail present → keep rail in both passes, CFG the instruction (editing). "Gen vs edit"
is inferred from "is a structural rail present," exactly as Qwen-Image-2.0 infers it from "is an
image present."

**3D-specific wrinkle — coords.** Output = SS coords → shape SLAT → tex SLAT. For *editing*:
- *material/appearance only* (coords fixed): feed the input SLAT as rail, condition the **tex** (and
  optionally shape) SLAT stage, **reuse the input asset's coords** for SS → simple & faithful;
- *structural* (coords change — add/remove/reshape): the **SS flow must re-generate coords**
  conditioned on (input-asset structure + Qwen instruction) → harder, same principle. Maps to the
  routes in [[project_trellis2_editing_constraints]].

**This is a SUPERSET of current infra, not a rewrite.** We already have Qwen-VL + DINOv3 rail
(image→3D, text→3D). To complete the unification, add only: (a) the **SLAT-latent rail** for 3D-asset
inputs (a new encode path + in-context concat); (b) the **dropout curriculum** (already the linchpin).
Multi-image is already half-supported (per-view DINOv3 + Qwen multi-image).

**Multi-image precedent (Qwen-Image-Edit-2509, 2025-09):** official doc says only "image
concatenation, further-trained." Inferred mechanism: Qwen2.5-VL handles multiple `<image>` tokens
natively (semantic route, ~free) + per-image VAE latents concatenated into the cond sequence with
per-image position/index embeddings (latent route). Exact details not published.

**Precedent — InstructX (arXiv 2510.08485, unified image+video editing, MLLM-guided) VALIDATES the
two-channel design.** It encodes the source TWICE by two different encoders, both branches take it:
MLLM (Qwen2.5-VL-3B) takes the source via its **native vision encoder** (video = 13 sampled frames) →
learnable queries (256 img / 512 video) → MLP connector → **replace the DiT's text embeddings**; AND the
source's **VAE latent is added to the noisy latent** (DiT, faithfulness). Key lesson for us: **the VLM
should take an input 3D asset via RENDERS** (its native ViT, like InstructX's frames), **not** by ingesting
the SLAT/native-3D latent — the SLAT-VAE stays DiT-only. So Plan-1 (two-channel, VLM-sees-renders) is what
SOTA unified editing actually does; you do NOT need to teach the VLM a 3D modality for editing. Janus
(2410.13848) "decouple visual encoding" (SigLIP-understanding vs VQ-VAE-generation) reinforces: semantic
encoder ≠ reconstruction encoder. Native-3D-into-VLM (Plan-2) = the Emu3/Chameleon/Janus AR direction
(heavier, north star). Two details worth copying: InstructX's learnable-query→connector→replace-text-embeds
(cleaner than `connector(all VL hidden)`); query budget 256/512 (our ~282 ≈ image budget; push 512+ for 3D).

**Open decisions added:** (iv) structural rail = droppable crutch (→ pure-Qwen) vs permanent
(always-anchored, more faithful)? (v) 3D-edit first ship "material-only (coords locked)" vs full
"structural edit"? (vi) input-3D-asset rail via in-context concat vs cross-attn?

---

## 8. Recommended roadmap (synthesis — pending weikaih's go on Phase 1)

**Settled architecture: two channels.** Semantic = Qwen3-VL (all inputs as VL tokens incl. rendered
views of a 3D asset) — the unifier, long-term primary cond. Structural rail = droppable, injected by
input type: image→DINOv3 cross-attn (cross-domain lift), 3D-asset→SLAT in-context concat (same-space),
text→none. Train with rail-dropout → one model = gen (rail off) + edit (rail on) + removable-DINOv3;
CFG auto-switches by rail presence.

**Opinionated calls:** (1) **A1** (gated handoff) over B — cheapest, reuses pretrained cross-attn;
B only if A1 plateaus. (2) **Don't force pure-Qwen now** — dropout makes DINOv3 *optional* (drop-or-keep
decided later by quality), not a now-or-never bet. (3) **Material/appearance edit first** (coords locked,
reuse input coords, condition tex) before coords-changing structural edit. (4) **Plan-1 (two-channel)**
now; Plan-2 (3D tokens into VLM) is the north star.

**Phases:**
- **P0 (done):** dual cond + BOTH-CFG'd inference validated.
- **P1 (linchpin — START HERE, minimal):** DINOv3-dropout curriculum — mirror `mask_drop` on the anchor
  with a ramping p (0→0.3→0.5→0.8) + add a "force-dino-off" eval. A few lines + one eval. Solves three
  things at once: gate finally gets gradient & grows; DINOv3 becomes removable (→ pure-Qwen); it's the
  same mechanism as unified gen+edit. De-risks the core hypothesis "can Qwen drive alone?"
  - **P1b (only if needed):** if the dino-off eval shows Qwen is structure-starved, raise Qwen's vision
    token budget (currently ~282).
- **P2:** add the 3D-asset rail (SLAT in-context concat) + train material edits (coords locked) — proves
  the SLAT-rail + VLM-instruction loop on the easy case.
- **P3:** structural edits (SS regenerates coords from input-asset + instruction) + full multi-input
  mixture training.
- **P4 (north star):** 3D tokens into the VLM (Plan-2) if render-semantics prove too coarse.

Start = P1 (highest-leverage, cheapest, directly tests "can Qwen drive"). Do not begin until weikaih
signs off (per the approve-design-before-implementing rule).

---

## 9. Update (2026-06-20): VLM-side 3D pretraining + per-layer KV
A separate, complementary track lands in **`VLM_3D_PRETRAIN_DESIGN.md`** (MolmoAct2-inspired): make
Qwen *3D-literate* via two new VLM-side stages — **VP1 (3D-understanding pretrain)** and **VP2
(editing pretrain)** — and replace the read-once connector cross-attn with **per-layer KV
conditioning** (the fix for the multi-view mushiness in RESULTS 2026-06-16). The discrete-token
route there is the cheap (next-token, not flow-loss) way to make Qwen drive — directly relevant to
this doc's "can Qwen drive alone?" question. Cheapest first step = per-layer KV with frozen Qwen.
