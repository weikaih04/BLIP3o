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
| 2026-06-11 | **Stage-1 CLOSED: no-KD@3000 final evals — monotone improvement to the end, artifact-free; recipe (pure flow loss + 1024 tok, ELLA route) VALIDATED.** Warrior@3000: cleanest action-pose golden swordsman (1678 coords/582k). Pinned-view 4-asset finals: dome ≈ TEACHER (best student dome), farm ≥ teacher richness, village/machine still behind on compactness/color. Remaining ceiling = identity-level fidelity (connector-only capacity) → Stage-2. | runs/native_infer_v3s1_nokd_3000_* |
| 2026-06-11 | **VLM-hidden cache + stage-split infra VALIDATED (tests/test_vlm_cache.py ALL PASS):** producer == live encode_cond fp16-EXACT; cached-path training-step loss == live Δ=0.00e+00; build_vlm=False works (VLM not loaded); per-stage jobs build only their component (ss=1.3B total vs ~10B) with healthy grads. Meta-contract refuses mismatched caches. Cache build for ready_v1×4views running on Beaker (01KTTN70YJTKYH0Q86C6DY2W1Q, ~1.7TB). | trellis2_blip3o/vlm_cache.py + scripts/build_vlm_cache.py |
| 2026-06-11 | **STAGE-1 A/B FINAL VERDICT: KD works mechanically but is NOT WORTH IT in our setting → V3 recipe = pure flow loss + 1024 tok (ELLA route).** KD delivered ~2× early convergence (KD@1000≈noKD@2000) and real structure transfer, but did NOT raise the ceiling (no-KD caught up by 2000-2500, monotone, artifact-free; both arms hit the same connector-capacity fidelity wall — 'ghost-warrior' identity errors), and cost CFG-robustness degradation + per-ckpt calibration. Literature (PEA/Scaling-Down) not contradicted: their setting = backbone frozen forever (KD is the only dense signal there); ours proceeds to Stage-2 (flow opens) where stability > early speed. Stage-2 init = no-KD@3000. KD code stays (default off) + endpoint-pair formulation documented. PAIRED re-eval also confirmed: @2500 'regression' was VIEW NOISE (pinned-view paired_eval4.jsonl now standard; degenerate top-down random views polluted earlier grids). | full A/B trail: runs/native_infer_v3s1_* + docs/V3_DISTILL_DESIGN.md |
| 2026-06-11 | **Stage-1.5 CFG-aware KD (LITE variant) FAILED — degrades the model at ALL guidance scales; run stopped @~500.** S1.5@500 warrior: speckled rainbow noise at BOTH CFG 7.5 and 5.0 (worse than the original @1500 sand; warmstart point KD@1000 was clean). Root cause of the failure: lite mode freezes the student null pass → the guided target forces the COND direction to compensate the uncorrectable null-gap → systematic cond distortion; plus the guided objective's ×s gradient amplification (grad_norm 1-2 vs 0.1-0.3) destabilized the connector. kd_v never broke below the s²-envelope (~0.5-0.6) in 500 steps — the early warning. **Math lesson: v_guided is LINEAR in (v_c, v_u) → matching both ENDPOINTS (plain MSE each, null pass WITH grads) ⟺ matching all guided combos, without amplification/coupling. 'Endpoint-pair KD' is the correct formulation if retried (docs/V3_DISTILL_DESIGN.md).** | runs/native_infer_v3s15_500_warrior{75,50}; exp 01KTSSQABQ0W4JCG8S5E8T4R61 (stopped) |
| 2026-06-11 🔍 | **9-asset held-out scorecard, KD@1000 vs no-KD@2000: ≈4 win / 2 loss / 3 tie for KD (at HALF the steps).** New 5: ec1 house KD✓ (closed body+roof vs roofless floorplan), ec2 metal parts tie (both blob), ec3 pot KD✓ (handles+shape; has a noise-strip artifact), ec5 farm no-KD✓ (house/trees/fence richer), ec6 standing person tie — **both collapse input pose to T-pose mannequin (shared prior bias, pose conditioning ignored — worth tracking)**. Net: KD = fidelity edge at 2× speed; no-KD stronger on scene-type; ceiling verdict pending no-KD@3000 + S1.5. | runs/native_infer_v3s1_{kd_1000,nokd_2000}_ec{1,2,3,5,6} |
| 2026-06-11 🔍 | **no-KD arm @ckpt-2000 largely CATCHES UP to KD@1000 on all 4 held-out probes** (village: green terrain+buildings back; dome: clean; blue machine: box+details; warrior: its best — golden, formed wings, sword, NO sand @CFG7.5). Updated narrative: KD's real value = ~2× convergence speed (KD@1000 ≈ no-KD@2000), but pure-flow-loss (ELLA route) does get there and is artifact-free; KD introduced CFG-sensitivity (sand) as a side effect. CFG sensitivities are OPPOSITE: KD-arm better @CFG5, no-KD-arm better @CFG7.5. Final verdict = no-KD@3000 vs Stage-1.5 CFG-aware-KD (ceiling comparison, not speed). | runs/native_infer_v3s1_nokd_2000_{warrior,warrior_cfg5,ec0,ec4,ec7} |
| 2026-06-10 | **KD-arm @1500 'sand' regression root-caused: inference-CFG amplification, NOT weight regression (hypothesis CONFIRMED via controlled CFG ablation).** 3 seeds @CFG7.5 all sandy/spiky (verts 3.0M noise-inflated) while train curves stayed perfectly healthy; SAME ckpt+seed @CFG5.0 → clean solid warrior (wings as panels, 1.58M verts). Mechanism: student distills RAW v_cond; its residual high-freq error vs teacher gets amplified ×7.5 in v_u+s(v_c−v_u) — and the closer the student tracks the teacher, the more structured the amplified artifact. ⇒ (a) student inference CFG must be recalibrated (~5, not DINOv3-tuned 7.5); (b) CFG-aware distillation promoted from optional to needed; (c) EMA diag crashed as predicted (EMA@1500 = 86% init for decay 0.9999 — useless on short runs, batch-size-independent arithmetic). Milestone evals now dual-CFG {5, 7.5}. | runs/native_infer_v3s1_kd_1500_{warrior,warrior_s43,warrior_s44,cfg5} |
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

## 2026-06-11 — V3 Stage-2 split-training mid-run eval (@1500/3000)
**Observed (1 seed, pinned views, CFG 7.5):** S2 SPLIT@1500 (3 independent 8-GPU jobs: ss/shape/tex, cached VLM hidden, init=no-KD@3000) ≈ S1 joint@3000 on paired_eval4 + pinned warrior. Color fidelity better on paired3 (cyan vs S1's purple drift); one identity drift (paired1 dome grew a doorway; S1 faithful). Grid: runs/eval_grids/v3s2_1500_vs_s1_vs_teacher.png.
- Infra validated: SPLIT inference mode (CKPT_SS/SHAPE/TEX, per-stage private connectors); token-contract fallback to split-ckpt config (base S1 ckpt pre-contract → cond silently 491 tok without it).
- Eval hygiene fix: warrior now pinned via data/overfit/pinned_warrior.jsonl (first run drew a degenerate dark back view from fresh-entropy rng — invalid row, rerun).
- Negative (archived): first grid impression "S2 paired0 = flat slab / paired1 = broken dome" was thumbnail+view-angle artifact; full-res zoom both fine.

## 2026-06-11 — V3 Stage-2 FINAL eval (@3000, dual CFG {7.5, 5})
**Observed (1 seed, pinned views, 10/10 evals):** S2 split@3000 ≈ S1 joint@3000 — split+cache infra carries NO quality cost at full length (3 jobs, ~4.5h each, all exit 0). 1500→3000 gains mild (paired2 diorama richer). Grid: runs/eval_grids/v3s2_3000_final.png.
- Residual failure modes = cond information ceiling (fusion motivation): paired1 dome doorway-drift persists at 3000; paired3 tex grows a camo-textured face at CFG 7.5 (clean at CFG 5 → student CFG sweet spot ≈5 reconfirmed); warrior wing-pose drift.
- Verdict: V3 recipe validated end-to-end on split infra; next lever = cond information (DINO+Qwen token-concat fusion) + data scale, not more steps on same data.

## 2026-06-11 — FUSION first signal (smoke@30 steps)
**Observed (1 seed, paired0):** input = L-shaped lot. TEACHER reproduces the L; S2@3000 (pure Qwen, 6000 steps total) gives a SQUARE lot (geometry lost); FUSION (cond=[raw DINO; Qwen], init=S2@3000 smoke) restores the L-shape + corner buildings after 30 STEPS. Geometric alignment is cond-information-bound, not step-bound. /tmp/fusion_first_signal.png; runs/native_infer_fusion_smoke30_paired0.
- Caveat: at 30 steps the model is ~the pretrained DINOv3 reader; the 3000-step question is whether the Qwen segment stays alive under dino_drop_prob=0.1 (watch text→3D / dropout-regime quality).
- Infra: fusion train (tests/test_fusion.py 4/4) + SPLIT×FUSION inference both validated end-to-end.

## 2026-06-11 — FUSION mid-run eval (@1500/3000) — SWEEP
**Observed (1 seed, pinned views, CFG 7.5):** fusion@1500 beats S2@3000 on EVERY row, reaching teacher-level fidelity: paired0 L-lot restored; paired1 dome SOLID (doorway drift cured); paired2 faithful empty lot (S2 hallucinated a dense diorama); paired3 clean cyan (camo face gone); warrior folded-wing pose ≈ input. Grid: runs/eval_grids/fusion_1500_vs_s2_vs_teacher.png.
- Qwen-segment aliveness (FUSION_NO_DINO=1 ablation): paired0 ≈ S2-quality square-lot+house; warrior rough but not collapsed → dino_drop_prob=0.1 sufficient so far.
- Throughput note: fusion ≈ +8% compute (controlled local A/B); Beaker +51% traced to 2× small-file random-read contention at 8-rank scale → next-run fix: merge v+d npz per view.

## 2026-06-12 — FUSION final eval (@3000) + segment ablations — VERDICT
**Observed (1 seed, pinned views, 12+4 evals):** fusion@3000 holds the sweep at BOTH CFGs (solid dome, L-lot, clean cyan teapot, folded-wing warrior; no CFG-7.5 texture hallucination). Grids: runs/eval_grids/fusion_3000_final.png, fusion_3000_ablations.png.
- Segment ablations (the Qwen-marginal question): FUSION ≈ DINO-only on ALL I1 rows → Qwen marginal ≈ 0 for single-image reconstruction (DINO carries fidelity); Qwen-only ≈ S2 (path fully alive, not harmed by fusion training). Scenario "2": Qwen is a harmless passenger on I1; earns its seat at Stage-3 (T/IM/edit).
- Consequence: Stage-3 can raise dino_drop_prob to 0.3+ aggressively — I1 fidelity is DINO-guaranteed.
- Day summary: S2 split 3-job (infra, 5×) → fusion 3-job (architecture) → 3-way ablation (attribution), all in ~24h.

## 2026-06-14 — STAGE-3 multitask final eval (@3000) — I1 + IM + T
**Observed (1 seed, pinned/multi-view):** all 3 tasks produce coherent 3D from one set of split ckpts (init=fusion@3000; I1 0.5 / IM 0.3 / T 0.2; dino_drop 0.3). Grid: runs/eval_grids/s3_3task_final.png.
- I1: fusion-level fidelity MAINTAINED (dome solid, teapot clean, warrior) — no regression from adding IM+T.
- IM (true 3-view, DINO@416² budget tier + view_embed, cond 4082 = m01 exact): coherent multi-view reconstructions; first working multi-image→3D.
- T (text→3D, Qwen-only, DINO segment absent): path ALIVE — produces real 3D — but text-fidelity WEAK: t3 scene/diorama HIT; t0 golden-warrior→generic humanoid (body ok, no armor/wings) PARTIAL; t1 motorcycle→humanoid MISS; t2 tall-box→diorama MISS. ~1/4 clear hit. Expected: frozen Qwen + small connector + last20, T weight 0.2 ≈ 600 effective text steps. Qwen earns semantics (humanoid/scene category) not fine geometry.
- Eval-harness fix: IM needs renders_dir records (multi_view) + TASK_MIX=IM + IM_TOKEN_BUDGET/IM_DINO_SIZE matching trained tier; paired_eval4 is single-image (can't test IM). im_eval4.jsonl built from ready_v1.
- Next levers for T: raise T mixture weight + more steps; or unfreeze later Qwen layers for text (currently frozen).

## 2026-06-16 — IM multi-view diagnostic (6 assets × 3 arms, IDENTICAL views [0,4,8])
**Observed:** A=3v@416²(trained) | B=3v@512²(fullres) | C=1view@512². Grid: runs/eval_grids/im_diag.png.
- A ≈ B → resolution is NOT the cause (full-res barely sharper, still mushy). weikaih's intuition confirmed.
- C (single view) ≥ A/B (multi-view) on most rows; #2 motorcycle decisive (single = crisp bike, 3-view = red blob). Multi-view HURTS, not helps.
- Verdict: IM mushiness is ARCHITECTURAL — TRELLIS cross-attn reads concatenated view tokens once as static K/V, no mechanism to register/reconcile views → optimum = blur the views into a union; more views = more conflict = mushier. Not data/resolution/training-amount.
- Aligns with the 3-paper convergence (Qwen-RobotWorld layer-wise joint attn / MolmoAct2 per-layer KV / EVA-01 concat self-attn): good multi-modal/multi-view fusion needs condition tokens in per-layer JOINT attention, not read-once cross-attn.
- Next: (a) add camera pose to view tokens (cheap, lets cross-attn geometrically place views) vs (b) joint-attention/double-stream surgery (TRELLIS is cross-attn DiT, big). Test (a) first.
