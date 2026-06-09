# TRELLIS.2 → BLIP3o-NEXT: Staged Training Plan (512 → 1024 cascade → self-forcing)

## Context

The port currently implements **only the 512 flow stages** (`ss`, `shape_slat_512`,
`tex_slat_512`). We want to add the full **1024 cascade** and, as a final
stage, a **self-forcing** finetune that closes the train↔inference gap.

Key facts established during analysis:

- TRELLIS.2 trains each flow model **independently** (decoupled flow-matching on
  precomputed GT latents). The **cascade is inference-only**.
- The 1024 DiT is **not trained from scratch** — it is **finetuned from the 512
  ckpt** (`finetune_ckpt = 512`). Architecture is byte-identical (1536d / 30
  blocks / rope); only the latent coord distribution differs.
- The architecture is **resolution-agnostic** because it is **sparse**
  (per-token, no fixed grid tensor) + **RoPE** (position = continuous function
  of integer coords, not a size-locked lookup table). So the same weights run on
  32³ / 64³ / 96³ coords.
- **1536 needs no training**: inference-time cascade quantizes proposed coords
  to 96³ (`1536//16`) and reuses the 1024 weights via RoPE extrapolation.
- Data: `dual_grid_1024` / `pbr_voxels_1024` raw voxels **already exist** (Beaker
  CPU prep). The **64³ SLAT latents do NOT yet exist** — they require the GPU
  Phase-B encode step.

### The three train↔inference gaps (motivation for self-forcing)

Current training teacher-forces at three points; inference does not have these:

1. **SS→shape coord gap** — shape flow trained on **GT** active-voxel coords;
   at inference the coords come from the **SS flow's own** generated occupancy.
2. **Shape cascade coord gap** — `shape_slat_1024` trained on **GT 64³** coords;
   at inference the coords are **512-LR → `decoder.upsample` proposed** coords.
3. **Tex concat gap** — `tex_slat` trained with **GT shape SLAT** as `concat_cond`
   (explicit teacher-forcing, `blip3o_qwen.py:343`); at inference the concat is
   the **generated** shape latent.

Self-forcing (Phase 4) runs the cascade end-to-end during a short final finetune
so each stage conditions on the **model's own** upstream outputs, removing
exposure bias.

---

## Phase 1 — 512 setup (current; finish + lock the A/B/C ablation)

**Goal**: validate the BLIP3o→TRELLIS conditioning path; pick the conditioning
formate (A/B/C) before spending compute on 1024.

- Models (joint training run): LM (+CE) + `ss` + `shape_slat_512` + `tex_slat_512`.
- A/B/C ablation (already wired via `configs/setup_{A,B,C}.json`):
  - A = full LM hidden cond, no codebook, `detach_cond=true`
  - B = discrete `<I*>` cond (`cond_slice=image_block`)
  - C = full hidden + codebook (BLIP3o-NEXT dual anchor)
  - **All use A1 (`cond_slice="full"`) already** — no change needed.
- `flow_stage_weights = "ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0"` (current default).
- Token-bandwidth note: cond ≈ 730 (single view) — fine for 512³. For faithful
  recon at higher res, plan multi-view (N×730) or DINOv3 hybrid (separate lever,
  not gating Phase 1).

**Deliverable**: a converged 512 ckpt per setup; ablation decision (A vs C).

---

## Phase 2 — add 1024 stages, finetune from 512

**Gating dependency (must finish first)**: GPU Phase-B encode of 64³ latents.

```bash
# Shape 64³ latents from dual_grid_1024 (per subset)
python data_toolkit/encode_shape_latent.py <SUBSET> --root <DATA_ROOT> \
    --resolution 1024 \
    --enc_pretrained microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16
# → shape_latents/shape_enc_next_dc_f16c32_fp16_1024/<sha>.npz   (64³ sparse)

# Tex 64³ latents from pbr_voxels_1024
python data_toolkit/encode_pbr_latent.py  <SUBSET> --root <DATA_ROOT> \
    --resolution 1024 \
    --enc_pretrained microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16
# → pbr_latents/tex_enc_next_dc_f16c32_fp16_1024/<sha>.npz
```

**Code changes (mirror the 512 wiring; 1024 DiT ckpts already on disk):**

| File | Add |
|---|---|
| `blip3o/model/multimodal_decoder/builder.py` | `build_shape_slat_1024()` / `build_tex_slat_1024()` → load `slat_flow_img2shape_dit_1_3B_1024_bf16` / `...imgshape2tex...1024` |
| `blip3o/model/blip3o_arch.py` | `self.shape_slat_1024` / `self.tex_slat_1024` + getters |
| `blip3o/model/language_model/blip3o_qwen.py` | two more flow stages `shape_slat_1024` / `tex_slat_1024` in the forward loop (copy the 512 blocks at lines ~317-346) |
| `trellis2_blip3o/dataset.py` | `process_target_shape_slat_1024` + collate; manifest keys `target_shape_slat_1024`, `target_tex_slat_1024` |
| `trellis2_blip3o/tr2_modules.py` | 1024 config paths + 1024 normalization stats (from `..._1024_bf16.json`) |
| `train.py` + `configs/*.json` | `flow_stage_weights += "shape_slat_1024=1.0,tex_slat_1024=1.0"` |

**Training**: init `shape_slat_1024` ← `shape_slat_512` ckpt, `tex_slat_1024` ←
`tex_slat_512` ckpt (finetune, not from scratch). Train on GT 64³ latents with
**GT coords** (TRELLIS.2 recipe — accepts the cascade coord gap; closed in Phase 4).

- `shape_slat_512` **must stay trained** — it is the cascade LR step.
- `tex_slat_512` optional (only for the `'512'` pipeline). For 1024_cascade you
  need: `ss`, `shape_slat_512`, `shape_slat_1024`, `tex_slat_1024`.

---

## Phase 3 — inference cascade (port `sample_shape_slat_cascade`)

Port from `TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py`:

```
SS flow → SS-VAE-Dec → 64³ occ → maxpool → coords_32
shape_slat_512 (LR) on coords_32 → shape_z_lr
shape_dec.upsample(shape_z_lr, upsample_times=4) → propose coords (quantize to N³)
shape_slat_1024 (HR) on coords_64 → shape_z
tex_slat_1024 on coords_64, concat_cond=shape_z → tex_z
shape_dec(shape_z) → mesh+subs ; tex_dec(tex_z, guide_subs=subs) → PBR
```

**Parameterize the quantization divisor** so the same code serves both:
```python
quant = ((hr_coords + 0.5) / lr_resolution * (target_res // 16)).int()
#   target_res = 1024 → 64³ ;  target_res = 1536 → 96³  (FREE, reuses 1024 weights + RoPE extrapolation)
```
→ **1536 comes for free** — no extra training, just pass `target_res=1536`.

---

## Phase 4 — self-forcing (final coherent end-to-end finetune)

**Goal**: remove exposure bias from the 3 teacher-forcing points so the cascade
is self-consistent at inference. Run as a **short finetune on top of Phase 2/3**,
not from scratch (sampling-in-the-loop is expensive).

For each training step, run a (few-step) cascade rollout and condition each
downstream stage on the model's **own** upstream output instead of GT:

1. **SS-forced coords**: sample SS flow → decode occupancy → coords (instead of
   GT coords). Train shape on these.
2. **Cascade-forced HR coords**: run `shape_slat_512` LR + `decoder.upsample` to
   propose coords_64 (instead of GT 64³ coords). Train `shape_slat_1024` on these.
3. **Self-forced tex concat**: generate `shape_z` from `shape_slat_1024`, use it as
   `tex_slat`'s `concat_cond` (instead of GT shape SLAT at `blip3o_qwen.py:343`).

**Loss target alignment**: since the proposed coords differ from GT coords, align
GT latents to the proposed coord set (nearest/scatter from GT) or use a
distribution/render loss. Decide per stage:
- Geometry: re-encode GT mesh at proposed coords, or use the decoder render loss
  (mask/normal/depth — already in TRELLIS.2 shape VAE trainer) as a coord-agnostic
  signal.
- Tex: PBR L1 on the intersection of proposed ∩ GT coords.

**Cost control**: few-step rollout (e.g. 4–8 Euler steps, not 50), short schedule
(~few k steps), low LR. Optionally only self-force the **last** cascade link (tex
concat) first — it's the cheapest and most impactful (the GT-shape teacher-forcing
is the largest gap).

**Implementation knobs to add**:
- `self_forcing: {none, tex_only, full}` config flag
- `rollout_steps` (Euler steps for in-loop sampling)
- gradient policy: detach the rollout sampler steps (treat as fixed context) vs
  backprop-through-sampling (more expensive, usually detach).

---

## Dependency / ordering summary

```
Phase 1 (512 train + A/B/C)  ──┐
                                ├──► Phase 2 (1024 ft)  ──► Phase 3 (cascade infer, 1536 free)  ──► Phase 4 (self-forcing)
Phase-B encode 64³ latents  ──┘        ▲ init from 512 ckpt
   ▲
dual_grid_1024 / pbr_voxels_1024  (✅ exist from Beaker CPU prep)
```

Critical path right now: **(1)** converge a 512 ckpt + decide A vs C; **(2)** in
parallel kick off Phase-B 64³ latent encode. Both done → 1024 ft is incremental.

---

## Verification per phase

- **P1**: 512 flow_mse decreasing; sample preview via `tests/test_trellis_cascade_512.py`
  / `test_trellis_preview.py` → renders look reasonable; A vs C eval on held-out images.
- **P2**: 1024 stages' flow_mse converge from the 512-init (should start lower than
  scratch); `test_forward_setups.py`-style smoke for 1024 stages.
- **P3**: end-to-end `run(pipeline_type='1024_cascade')` produces a 1024³ mesh; then
  `target_res=1536` produces a 1536³ mesh without retraining.
- **P4**: inference-quality gap vs GT-coord training shrinks (measure recon
  fidelity: rendered LPIPS/normal error of cascade output vs single-stage GT-coord
  output on the same images).
