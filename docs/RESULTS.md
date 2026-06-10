# RESULTS — experiment & ablation log

Living record of what we ran, what we saw, and what is **validated**. Design rationale lives in
the `*_DESIGN.md` / `*_PLAN.md` docs; this file is **outcomes only**.

## Conventions
- **Validated ✅** = reproduced / visually confirmed working → gets a **date** (`YYYY-MM-DD`) the day it
  was confirmed. Only dated rows are "入库" (committed as a real result).
- **Observed 🔍** = a single run / impression, not yet confirmed across seeds or reruns. No date.
- **Negative ❌** = tried and did not work (kept so we don't repeat it).
- Inference is seed-dependent (`SEED` env, default 42). **A single-seed impression is NOT a result** —
  see the 2026-06-01 variance finding below.

---

## Validated results (入库)

| Date | Result | Evidence |
|------|--------|----------|
| 2026-06-10 🔍 | **Stage-1 A/B @ckpt-1000, 3 MORE held-out assets (eval_clear8 idx0/4/7): KD wins 4/4 overall.** Village-on-terrain: KD keeps green ground+buildings+fence, no-KD browns the ground and loses buildings. White dome: KD clean dome+base, no-KD melted with flaps. Blue machine: KD detailed (parts/texture), no-KD dark blob+pipe. Same direction as the warrior → KD's held-out advantage is systematic (color fidelity/layout/detail), not single-example luck. Single-seed, random input views — quantitative verdict at 3000 + multi-seed. | runs/native_infer_v3s1_{kd,nokd}_1000_ec{0,4,7} |
| 2026-06-10 🔍 | **Stage-1 A/B @ckpt-1000: KD stays a length ahead; both arms converging toward the asset.** KD: full golden warrior (helmet/armor/red sash/wings forming), coords 3139→2333 (→ dual baseline 2144); 'right thing getting cleaner'. no-KD: grew gold+wings from the @500 bald mannequin but spiky/fragmented wings, floating debris, melted-wax body, coords 846→1945; 'getting right, but rougher'. Single-seed observed. | runs/native_infer_v3s1_{kd,nokd}_1000_warrior |
| 2026-06-10 🔍 | **Stage-1 A/B @ckpt-500 (held-out warrior, single-seed OBSERVED): opposite failure modes.** KD arm = asset STRUCTURE present (wings/cape/armor/face, coords 3139 ≈ dual-baseline ballpark) but rough (stranded wings, blobby legs). no-KD arm = clean surfaces but SEMANTIC COLLAPSE to a generic bald T-pose mannequin (coords 846; zero asset-specific structure). Complements 3c: overfit(memorization)→no-KD wins; held-out generalization→KD's structure transfer shows immediately. 1/6 of training; trajectory check at ckpt-1000+. | runs/native_infer_v3s1_{kd,nokd}_500_warrior |
| 2026-06-10 | **V3 condition-swap KD pipeline works end-to-end + connector-only@1024tok is NOT garbage (3c killer overfit).** Two-arm single-asset overfit (loveseat, 300 steps, BS2, `--flow_tune none` = connector-only 3.1M, `--target_tokens_per_view 1024`): **BOTH arms produce clean complete loveseats** (KD arm 3717 coords/2.49M verts; no-KD 3506/1.26M; renders visually clean, zero shatter). Historical connector-only=garbage was at **256 tok/view** → **the 1024-token raise is the unlock** (capacity, not only signal sparsity). KD arm's distill metrics dropped ~95% in 300 steps (kd_v_ss 0.26→0.010, kd_f_ss 0.40→0.019) = connector demonstrably matches the DINOv3-conditioned flow's behavior; KD overhead +17% step time (1.18→1.38 s/it). Overfit cannot discriminate KD's value (both pass) — that test moves to Stage-1 scale. Unit layer: loss-level KD invariant EXACT 0 (same cond both paths, dense+sparse); model-level smoke: grads connector-only, flows frozen, no-teacher path clean. | runs/v3_overfit_{kd,nokd} + runs/native_infer_v3overfit_{kd,nokd} + tests/test_kd_loss.py + tests/test_distill_forward.py |
| 2026-06-05 | **Dual-branch 512 cascade TRAINS STABLE on 16-GPU/2-node via `--dual_qwen_last_frac 0.2` (Qwen cross-attn injected only on the LAST 20% of blocks).** Full-dual (Qwen on all 90 blocks) OOMs on EVERY topology incl 8-GPU single-node — the +9.5GB is the new Qwen cross-attn activation across all blocks; flow_tune last15 only trains the tail so early blocks have gate≈0 = wasted. Injecting only last 20% cuts dual cost +9.5GB→~+2GB. **Smoke: 2-GPU DDP 0 OOM, peak 61.2GB** (vs full-dual ~80.7); **cluster: 16-GPU 2-node passed step 1237+, ckpt-1000+EMA saved, 0 OOM, ~1.7 s/step (30k≈14h), loss ~0.34, gate_abs growing 5e-5→2.4e-4.** Zero compromise: full 4-view multi_image, full data, full-throughput deepspeed. Exp `01KTB4AMWDGEWZEWM3YB5EZ53Y`. | smoke 2-GPU + 16-GPU run past tail+ckpt |
| 2026-06-01 | **Dual-branch (DINOv3 anchor + scalar-gated Qwen cross-attn) SS path runs end-to-end on the real train stack.** Build OK; gate → `dual/gate_abs=0` at init (behaviour == pretrained flow) then grows; loss finite, no NaN; compile steady-state ~1 s/it. 1-GPU BS=2 OOMs, **2-GPU BS=2 fits**. SLAT sparse wrapper forward-validated (ran SS+shape SLAT). | SS smoke 1-GPU BS1 + 2-GPU BS2 |
| 2026-06-01 | **512 + dual FITS on 2-GPU with `--flow_tune last20`** (8/8 steps, 135s, gate_abs 0→1.67e-4). last40 OOMs (step 5), last20 fits — partial-FT DOES help (fewer trained layers → smaller optimizer states + less backward-stored activation; the extra 20% is the OOM↔fit margin). Dual raw cost ~10GB (SS BS1: 32.5→41.6GB: frozen fp32 DINOv3 ~1.2GB + per-block Qwen cross-attn). **last20 suits dual**: anchor preserves the original flow's input dist (needs little retraining), while the Qwen branch (cross-attn+gate) is always fully trained. No mem-optimization needed for 2-GPU; 8-GPU allows more FT. | 512+dual last20/last40 (2-GPU), SS ±dual peak |
| 2026-06-01 | **dino-align "2k→5k degradation" was a single-seed artifact, NOT real checkpoint regression.** The collapse was seed-42 instability, not a monotonic trend. | 4-seed × 2-ckpt violin sweep (below) |
| (prior) | **Single-asset text→3D overfit works.** | 6 port bugs fixed + 512 latent prep + ZeRO-2/CPU-offload recipe (see `project_overfit_training_fixes`) |
| (prior) | **Single-image overfit works via partial-FT.** Capacity ladder: connector 3.1M = garbage < LoRA-r64 227M ≈ cross-attn 712M = rough < cross+self 1562M ≈ last40 1533M = CLEAN (≈ text quality). Full-rank partial-FT beats low-rank LoRA. | `project_image_overfit_ss_todo` |
| (prior) | **Qwen3.5-2B native-VLM continuous-cond variant validated.** Frozen Qwen3.5-2B + connector + flow last40 → clean loveseat ≈ blip3o quality. | `project_qwen35_vlm_variant`, `QWEN35_VLM_DESIGN.md` |
| (prior) | **Multi-asset 30k baseline trained successfully**, quality improved 12k→16k→29k (violin→real 3D, aircraft→recognizable, lamp→refined). Residual: floaters / rough surfaces. | 30k run |

> Backfill note: rows marked `(prior)` were validated before this log existed; exact dates unknown.
> All new entries must carry the confirmation date.

---

## Experiment log (newest first)

### 2026-06-08 — rope-fix baseline (v1 ropefix ckpt-30000 FINAL) across 3 cond modes 🔍
**Question:** how does the rope-fixed image→3D baseline (v1 ropefix run, COMPLETED 30000/30000,
loss_ss healthy ~0.10 throughout) do on single-view / multi-view / text→3D?

**Setup:** `eval_clear8` assets 0 (flat site model) + 1 (3D walled building), forced each mode via
new `TASK_MIX` env in `test_native_infer.py` (`I1:1.0` / `IM:1.0` / `T:1.0`), `MAX_VIEWS=4`,
USE_EMA=1, FIX_ROPE=1, SEED=42. 6 runs (2 assets × 3 modes). Renders
`runs/native_infer_ropefix30k_{I1,IM,T}_a{0,1}/`.

**Finding (🔍, seed 42, 2 assets — NOT yet ≥3-seed validated):**
- **single-view (I1):** ✓ clean — asset1 = solid 3D walled building; asset0 = correct flat base +
  buildings. rope fix HOLDS at 30k (coherent meshes, NO shatter/rubble).
- **multi-view (IM, 2–3 views):** ⚠️ **MIXED / object-dependent** (verified across 4 assets 2026-06-08).
  vert count is a bad proxy (initial "multi better" claim was wrong). Single-vs-multi by object —
  house(a1): single coherent vs multi sprawled → SINGLE; jar(a3): single clean neck+lip vs multi squashed
  pot+floating fragment → SINGLE; dome(a4): single flat disc (MISSED dome) vs multi clean grooved
  hemisphere → MULTI (multi disambiguates where one view reads flat); wheel(a2): both blobby (hard
  thin/hollow). So multi-view CAN help (resolve 3D-extent ambiguity) but also over-generates/sprawls/
  drops fine features. NOT consistently worse — object-dependent & unreliable. Both modes have distinct
  failure modes. **BUT heavily CONFOUNDED by random input-VIEW quality** (inputs shown 2026-06-08): dome's
  single input was a dark near-top-down view (dome reads flat → flat-disc output was reasonable); multi
  caught a clear side view. Input view angle+lighting is the DOMINANT factor; a clean architectural
  single-vs-multi verdict needs a CONTROLLED-VIEW test (same view(s) to both). Don't over-conclude from
  random-view-draw comparisons.
- **UPDATE (bigger multi-view batch, 6 diverse objects 2026-06-08): multi-view→3D is GENERALLY GOOD.**
  Held-out teapot ✓✓, human figure ✓✓, villa ✓; train-seen winged trophy ✓✓, rocket ✓, gondola ✓. The
  earlier "multi worse than single" was OVER-GENERALIZED from 2 unlucky cases (house a1 + jar a3). NET:
  both single & multi image→3D usable; quality ∝ object + input-view quality; multi NOT generally worse.
  Held-out wins confirm real generalization. text→3D is the only truly broken mode (gate≈0).
- **text→3D (T):** ❌ **collapses to a flat sheet** on BOTH assets (asset1 building → flat+box;
  asset0 → bare wavy sheet, lost the buildings). Mechanism: caption(13 tok)→Qwen branch ×gate≈0.0009
  ≈ 0, anchor=null → flow gets ~zero conditioning → unconditional SS prior = flat.

**Conclusions:**
1. rope-fix SINGLE-view image→3D is genuinely usable (clean, coherent). ✅ (further confirms the
   rope-fix validation at the FINAL ckpt). multi-view is MIXED/object-dependent (verified 4 assets) —
   helps on single-view-ambiguous objects (dome), hurts on others (house/jar over-generate) — not a
   reliable win, both modes have failure modes.
2. text→3D is **completely untrained** at v1 (gate≈0) — flat-sheet collapse. This is the gap the next
   phase must close. Quantifies "why text→3D needs dedicated training."
3. NOTE: asset0 is a genuinely flat object → poor text discriminator; asset1 (3D building) is the
   clean one. Per ≥3-seed convention, re-run multi-seed + more categories before calling a trend.

### 2026-06-01 — dino-align multi-seed variance test 🔍→✅
**Question:** the earlier impression that dino-align (weight 0.25, from-30k) *degraded* from
checkpoint-2000 → 5000 — is it real, or seed noise / under-training?

**Setup:** violin asset, `tests/test_native_infer.py` with `SEED ∈ {42,7,123,99}`, on
checkpoint-4000 and checkpoint-5000. Front-view (`gen_shaded_00`) montage:
`runs/_variance_montage.png`. Mesh vert counts in `tasks/b7g4zq83g.output`.

**Finding (✅):**
- **ck-5000:** s42 = total collapse (flat wavy ribbon); s7 / s123 / s99 = recognizable instrument body.
- **ck-4000:** s42 / s7 / s99 = good violin body; s123 = body + detached floating scroll.
- → The "5000 is worse" call came **entirely from seed 42**, which happens to be a bad seed for
  ck-5000. On 3 of 4 seeds the two checkpoints are comparable.

**Conclusions:**
1. The 2k→5k "degradation" is **not a real monotonic regression** — it is **seed-variance / instability**.
   Single-seed (42) comparison gave a false signal. → always sweep ≥3 seeds before calling a trend.
2. Both checkpoints are **high-variance**: output ranges clean-violin → floater → ribbon across seeds.
   This under-determined cond→shape mapping is consistent with the "replace DINOv3 with a from-scratch
   connector" disruption hypothesis (see architecture note below).

**Action items:**
- Do not over-interpret single-seed inference. Tag all future quality calls with their seed(s).
- Architecture redesign candidate (Know3D-style additive branch) expected to *reduce* this variance by
  keeping the pretrained DINOv3 path as a geometry anchor — see `reference_know3d_paper` + below.

### (prior) — dino-align (REPA, weight 0.25, spatial), resumed from 30k 🔍
- ck-2000 looked like a sweet spot (cleaner, fewer floaters, aircraft clean) on seed 42.
- ck-5000 looked degraded **on seed 42** → SUPERSEDED by the 2026-06-01 variance finding above
  (that was seed noise, not a real ckpt trend).
- Design: `DINO_ALIGNMENT_DESIGN.md`, code `trellis2_blip3o/dino_align.py`.

---

## Open architecture decision (2026-06-01, in discussion)
Whether to replace the current **DINOv3→Qwen replacement** cond (unstable, needs dino-align crutch)
with a **Know3D-style additive design**: keep TRELLIS2's original DINOv3 cross-attn as a frozen
geometry **anchor** (image / multi-image only) + add a parallel zero-init Qwen cross-attn (all tasks;
sole cond for text→3D). Would borrow TRELLIS2's pretrained flow without disturbing it. See
`reference_know3d_paper`, `reference_cgmllm_paper`. Not yet implemented.
