# 3D Editing Roadmap

Unified **{image(s) · 3D asset · text} → 3D asset** model that does **generation AND editing in one
model** (the 3D analog of Qwen-Image-2.0), built on the TRELLIS.2 cascade + Qwen3-VL. Route decided
(2026-06-09, weikaih): **Plan-1 — two-channel, VLM-takes-renders.** This README is the canonical plan;
deeper design rationale + the DINOv3→pure-Qwen handoff analysis live in `QWEN_HANDOFF_DESIGN.md`.

**Status:** P0 done (dual-cond + BOTH-CFG'd inference validated — warrior 792k / teapot 657k verts clean).
**Editing (P2–P3) PARKED (2026-06-09)** — full design captured in this doc (two-channel, VLM-takes-renders,
SS-dense structural edits, two-segment rail, mesh→SS+SLAT encoding); resume after the generation track.
**Active focus = generation:** P1 (DINOv3-dropout curriculum) toward removable-DINOv3 / pure-Qwen.

---

## 1. Vision

One model, any combination of inputs, a 3D asset out:
- **text → 3D** (pure generation)
- **image(s) → 3D** (single / multi-view lift)
- **3D asset (+ text/image) → 3D** (editing)
- mixtures of the above

Editing and generation are the *same model* — which mode you're in is inferred from *which inputs are
present*, exactly as Qwen-Image-2.0 infers gen-vs-edit from "is a source image present." Long-term,
DINOv3 becomes an *optional* rail (droppable → pure-Qwen) rather than a hard dependency.

---

## 2. Architecture (DECIDED: two channels)

```
                 ┌──────────────── SEMANTIC channel — "understand" ────────────────┐
  text ──────────┤                                                                 │
  image(s) ──────┼──►  Qwen3-VL  ──last-hidden──►  connector  ──► cond ──┐          │
  3D asset ─render┘    (native ViT)                                      │          │
                                                                  (cross-attn)      │
                 ┌────────────── STRUCTURAL rail — "be faithful" (droppable) ───────┘
  image ─────────►  DINOv3  ───────────────────── cross-attn ──┐                    │
  3D asset ──────►  SLAT-VAE ── in-context concat ─────────────┤                    ▼
  text ──────────►  (no rail)                                  └──►  TRELLIS DiT cascade ──► 3D
                                                                     SS → shapeSLAT → texSLAT
                                                                     (denoises from noise)
```

**Semantic channel = Qwen3-VL (the universal unifier, the "brain").** EVERY input becomes VL tokens:
text, image(s), and **rendered views of an input 3D asset** (the VLM sees a 3D asset through renders —
its native modality, NOT the SLAT latent). The VLM's last-layer hidden states → a trainable connector →
the conditioning the DiT cross-attends to. (This is our current `connector(VL hidden)` path = the
UniVideo recipe; InstructX's learnable-queries is the alternative.)

**Structural rail = a droppable, source-dependent latent (the "be-faithful" channel).** Injection method
follows the **same-space-vs-cross-domain** rule:
- input **image** → DINOv3 features → **cross-attn** (cross-domain 2D→3D *lift*; can't concat an image
  latent into 3D);
- input **3D asset** → its **SLAT-VAE latent** → **in-context concat** alongside the noisy output SLAT
  (same SLAT space → concat, à la Qwen-Image-Edit's `image_latents`);
- **text** → no rail.

**Unification via dropout.** Train with the structural rail **stochastically dropped** → one model covers
text→3D (rail off), image→3D (DINOv3 rail), 3D-edit (SLAT rail), and mixtures. The same dropout makes
DINOv3 *removable* (→ pure-Qwen). **CFG auto-switches by rail presence:** no rail → CFG the cond
(generation); rail present → keep rail in both passes, CFG the instruction (editing).

---

## 3. Decisions locked

| # | Fork | Decision | Why |
|---|------|----------|-----|
| 1 | Handoff arch | **A1 — gated additive** (not B "Qwen-as-main") | cheapest, reuses DINOv3-pretrained cross-attn; B forfeits the head start |
| 2 | Pure-Qwen | **Don't force it now** — dropout makes DINOv3 *optional* | one training move = "drop-or-keep decided later by quality", not a now-or-never bet |
| 3 | VLM takes 3D asset via… | **multi-view renders** (not native SLAT) | native modality → reuses pretrained understanding + zero-shot transfer; proven by InstructX/UniVideo; reuses our render infra |
| 4 | Structural channel split | **SLAT-VAE → DiT only** (not into VLM) | DiT needs lossless geometry; VLM's job is semantic (renders suffice). Janus: decouple understanding vs reconstruction encoders |
| 5 | Edit start point | **from pure noise**, input SLAT = clean in-context cond | Qwen-Image-Edit style; SDEdit-from-input only does mild variants |
| 6 | First edit type | **material/appearance (coords locked)** before structural | reuse input coords → easiest, highest-success first win; validates the loop |
| 7 | Channels | **Plan-1 two-channel** (not Plan-2 all-through-VLM) | proven by InstructX/UniVideo; Plan-2 (3D tokens into VLM) = north star |

---

## 4. The editing forward pass (concrete)

**Encoding the input asset (SS and SLAT are DIFFERENT latents — two encoders).** A mesh → voxelize
(`data_toolkit/voxelize_pbr.py`) → ① occupancy → `SparseStructureEncoder` (`sparse_structure_vae.py`) =
**SS latent** (dense, → SS rail); ② voxelized PBR → pbr/structured-latent encoder
(`data_toolkit/encode_pbr_latent.py`) = **SLAT** (sparse coords+feats, → SLAT rail); ③ multi-view render
= the VLM input (separate, not from a latent). This is TRELLIS's standard training-data prep — reuse it.
*Nuance:* if editing a just-generated asset, its SS+SLAT latents already exist → reuse them, skip the
decode→re-encode round-trip; only external/arbitrary meshes need the voxelize+encode pass.

For **3D-asset editing** (`3D asset + instruction → edited 3D`):
1. Render the input asset to **N multi-view images** → Qwen3-VL (native vision) + the text instruction
   → last-hidden → connector → **semantic cond** (what to change).
2. Encode the input asset to its **SLAT latent** (TRELLIS VAE) → **in-context concat** to the noisy
   output SLAT → **faithfulness** (what to preserve).
3. Output SLAT **denoises from pure noise**, cross-attending the semantic cond + attending the in-context
   input SLAT. The VLM does **not** ingest the SLAT.
4. **Coords / structural edits — the rail is TWO segments.** Key fact (verified in
   `sparse_structure_flow.py`): **the SS stage is DENSE** — it flows a `randn(B, C, reso³)` low-res volume,
   which the SS-VAE decodes into the sparse coords; only the *downstream* SLAT is sparse. So:
   - *material/appearance edit* → coords fixed: **reuse the input asset's coords**, condition only the
     tex/shape SLAT (SLAT-level rail) — no SS edit.
   - *structural edit* → edit at the **dense SS stage** (you CAN add/remove there, like editing an image
     latent): condition the SS flow on (input asset's dense SS latent + instruction) → new occupancy/coords;
     global ("make it taller") = re-condition+regen, local ("add a handle here") = **mask-inpaint** (fix
     structure outside a 3D mask, generate inside — cf. VACE masked-v2v). Then the SLAT-level rail fills the
     new coords, using the input SLAT on overlapping coords (inpainting-style preservation) — **sparse-attn
     bridges input-coords ↔ new-coords**. The "sparse → can't change" limit is only the SLAT layer; we edit
     upstream at the dense SS, SLAT just follows.
   - So the **structural rail = SS-level rail (input dense SS latent → SS flow) + SLAT-level rail (input
     SLAT → SLAT flow).** P2 uses only the SLAT rail (coords locked); P3 uses both.
5. **CFG:** keep both the input SLAT and (optionally) DINOv3 in both passes, CFG the instruction.

The same machinery, with the rail **off**, is plain text→3D / image→3D generation.

---

## 5. Precedents (grounding)

- **Qwen-Image-Edit / Qwen-Image-2.0** — dual-encoding (VL semantic + VAE latent); unified gen+edit via
  stochastic source-image presence; source image kept in both CFG passes, CFG only the text
  (`true_cfg_scale` default 4.0).
- **InstructX** (2510.08485) — MLLM (Qwen2.5-VL) takes source via **frames** + learnable queries → connector
  → **replace DiT text embeddings**; source VAE latent **added to noisy latent**. Two encoders, dual-encode.
- **UniVideo** (2510.08377) — **frozen** MLLM last-hidden → trainable MLP connector → MMDiT (= our setup);
  **zero-shot transfer** from image-editing data to unseen video editing → editing may transfer cheaply.
- **Janus** (2410.13848) — *decouple visual encoding*: understanding-encoder ≠ generation/reconstruction-encoder.
- **VACE** (2503.07598) — pure-diffusion camp: source → DiT only (no MLLM). Contrast point.

---

## 6. Phased roadmap

- [x] **P0** — dual-cond (DINOv3 + gated Qwen) + **BOTH-CFG'd packed inference** validated (no retrain).
- [ ] **P1 — anchor-dropout (ALREADY IMPLEMENTED — re-evaluate, then resume).** NOT greenfield: the
      3-way coordinated CFG dropout already exists (`anchor_drop_prob` / `cfg_joint_drop_prob` in
      `train_native.py` + `trellis_native_vlm.py:548-568`: joint-drop / anchor-only-drop / both-on; the
      anchor-only branch's comment is literally "forces the Qwen branch to learn → fixes gate≈0"). It was
      **already trained as the v2c run** (`--anchor_drop_prob 0.15 --cfg_joint_drop_prob 0.1`, warmstart
      v1-ropefix-ckpt24000) but reached only **ckpt-4000/15000** before being stopped — that stop *was* the
      "v2c image→3D degradation" investigation, which we traced to the **BOTH-CFG inference bug (now fixed)**.
      So P1 = **(a) re-evaluate v2c-ckpt4000 with the fixed BOTH-CFG inference** (gate growth? text→3D fixed?
      image→3D no longer degraded?), then **(b) resume/extend** the dropout training if on track. (EMA gate
      readout lags badly <10k steps — `EMA@0.9999` barely moves — judge by raw gate + the dino-off eval, not
      EMA.) Add a **"force-DINOv3-off" eval** to watch pure-Qwen quality climb.
  - [ ] **P1b** (only if the dino-off eval shows Qwen structure-starved) — raise Qwen's vision-token budget
        (currently ~282; InstructX uses 256 img / 512 video → push toward 512+).
- [ ] **P2** — add the **3D-asset rail** (SLAT-VAE in-context concat) + render→VLM path; train **material
      edits** (coords locked). Proves the SLAT-rail + VLM-instruction loop on the easy case.
- [ ] **P3** — **structural edits** (SS regenerates coords from input-asset + instruction) + full
      multi-input mixture training.
- [ ] **P4 (north star)** — **SLAT → VLM** (3D-native MLLM) if render-semantics prove too coarse for
      occlusion/interior edits.

---

## 7. Phase 1 — immediate next step (RE-EVALUATE the existing v2c, then resume)

The anchor-dropout is **already built and already trained** (see §6 P1) — don't re-implement it. The plan:

1. **Re-eval v2c-ckpt4000 with the fixed BOTH-CFG inference** (no retrain). Run image→3D (warrior/teapot —
   confirm the "degradation" was the now-fixed bug, not the dropout) AND **text→3D** (`TASK_MIX=T:1.0` —
   the mode anchor-dropout targets; v1 had it broken at gate≈0). Compare to ropefix-ckpt30000
   (image: warrior 792k / teapot 657k clean).
2. **Judge by raw gate + dino-off quality, NOT EMA** (`EMA@0.9999` barely moves <10k steps; v2c-4000 EMA
   gate ≈ ropefix's, which is uninformative). Read the raw `model.safetensors` gate; add a **"force-DINOv3-off"
   eval** to watch pure-Qwen quality climb toward dino-on.
3. **If on track → resume/extend v2c** (it was 4k/15k steps, interrupted). If not → diagnose (schedule,
   warmstart point, or whether the BOTH-CFG fix changes the training calculus).

Config knobs already exist: `--anchor_drop_prob` (0.15 in v2c), `--cfg_joint_drop_prob` (0.1). Honor the
**NO DeepSpeed CPU-offload** policy; reuse the v2c launch yaml
(`experiment/qwen35_trellis2_training/train_dual_v2_anchordrop_uptext_2node_molmo3.yaml`).
**Blocker (2026-06-09):** the 2 local GPUs are occupied by an unrelated `thinkmorph` training job — re-eval
needs free GPUs or a Beaker job (terminal-side auth).

---

## 8. Open decisions (still to settle, later phases)

- (v) ship "material-only edit (coords locked)" first vs jump to structural edit — *leaning material-first*.
- Permanent lightweight structural anchor vs fully removing DINOv3 — *deferred; dropout keeps it optional*.
- 3D-asset rail injection: in-context concat (chosen) vs cross-attn — *concat, but revisit if coords-mismatch attention is hard*.
- Number/selection of render views fed to the VLM (coverage vs token budget).

---

## 9. Pointers

- Design rationale + handoff analysis: `QWEN_HANDOFF_DESIGN.md`
- Dual-cond wrappers: `trellis2_blip3o/dual_cond.py` (`DualCondInjectBlock` dense / `DualCondSparseInjectBlock` sparse; `_packed_cond` BOTH-CFG path)
- Inference harness: `tests/test_native_infer.py` (BOTH-CFG default; `DINO_ONLY=1` / `LEGACY_SS=1` A/B)
- Model: `blip3o/model/language_model/trellis_native_vlm.py` (dual_router, `set_dino`, connector)
- Results log: `RESULTS.md`
