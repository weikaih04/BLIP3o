# DINOv3 Conditioning Alignment (REPA-inspired) — Design

**Motivation:** ckpt-12000 image→3D samples (violin/aircraft/lamp) show the
conditioning IS working (object-specific geometry tracks the input) but geometry
is rough (jagged/holey/fragmented). Root cause is the known cond-side divergence:
TRELLIS's flow cross-attn was pretrained on **DINOv3 ViT-L/16** features; we feed
it `connector(Qwen)`, trained only by indirect flow-MSE backprop. Give the
connector a **direct DINOv3 regression target** → stronger signal → the frozen
pretrained cross-attn gets in-distribution cond → better/faster geometry.

## What REPA actually does (verified from arXiv 2410.06940)
- Aligns the **denoiser's own internal hidden states** (NOT the conditioning),
  at the **first ~8 of 24 layers**, to **DINOv2** patch features of the clean image.
- Loss: `L = -E[ (1/N) Σ_n cos( y*[n], h_φ(h_t[n]) ) ]` — patch-wise cosine,
  maximize; `h_φ` = 3-layer MLP + SiLU projection head.
- Patch correspondence: denoiser & encoder share the same 32×32 grid → align by
  patch index. λ=0.5 (robust 0.25–1.0). DINOv2-L/g best. ~17.5× SiT speedup.

## Why literal REPA doesn't transfer cleanly to us
1. REPA is uncond/class-cond generation (no image conditioning); we already have
   an image cond.
2. REPA relies on a **2D patch grid correspondence**. Our denoiser denoises a
   **3D voxel latent** (SS 16³ dense / SLAT 64³ sparse) — no 2D-patch↔3D-voxel
   spatial correspondence, so REPA's patch-index alignment on denoiser internals
   doesn't apply directly.

## Our adaptation: cond-side DINOv3 distillation
Align the **conditioning features** (connector output, the thing the cross-attn
consumes) to DINOv3(render), instead of denoiser internals.

**Privileged target:** REPA picks DINOv2 as a generically-good target. We have the
EXACT encoder TRELLIS's frozen cross-attn was pretrained on (DINOv3 ViT-L/16). So
distilling to DINOv3 is strictly more principled here than vanilla REPA.

### Shapes (verified)
- `DinoV3FeatureExtractor(render@512)` → `(B, N_dino, 1024)`, N_dino = 1024 patches
  (32×32) + CLS/reg. Final LayerNorm applied. (trellis2/modules/image_feature_extractor.py:59)
- `connector(Qwen_hidden)` → `(B, N_qwen, 1024)`. Linear(2048→1024)→GELU→Linear→LayerNorm.
- **Dims already match (1024 = 1024)** → no dim projection needed; a light MLP head optional.
- **Token counts differ** (N_qwen ≠ N_dino) → need correspondence handling.

### Token-correspondence options (pick one)
1. **Pooled (simplest):** mean-pool both over tokens → (B,1024), cosine align.
   Coarsest; aligns global semantic only. Good first cut, low risk.
2. **Spatial-resampled (most REPA-faithful):** Qwen vision tokens have a spatial
   grid (from `image_grid_thw` → h×w); bilinearly resize that grid to DINOv3's
   32×32, then patch-wise cosine. Preserves spatial structure → should help local
   geometry (the jagged/holey problem) more than pooled.
3. **Learned cross-attn head:** small head maps N_qwen→1024 tokens in DINOv3 layout,
   then patch cosine. Most flexible, most params.

Recommendation: start with **(2) spatial-resampled patch cosine** — it directly
targets the local-geometry roughness; fall back to (1) if the grid bookkeeping is
fiddly.

### Loss
```
L_align = - (1/N) Σ_n cos( dino[n], h_φ(cond[n]) )           # maximize cosine
L_total = L_flow + λ_align · L_align
```
- `cond` = connector output (or pre-LayerNorm — ablate); `dino` = DINOv3(render) detached.
- `h_φ` = 3-layer MLP+SiLU (REPA uses one). **Keep the head even though dims match**
  — it's the text-protection mechanism (see below): align `h_φ(cond)`, so `cond`
  only needs to be DINOv3-*mappable*, not DINOv3-*equal*. Head is train-only,
  dropped at inference.
- λ_align: start **0.5** (REPA's robust default); ablate 0.25–1.0.
- DINOv3 frozen, `@torch.no_grad()`; render@512 to match TRELLIS's SS/lowres cond.

### Protecting text→3D (key concern — don't sacrifice text-only)
Worry: pushing `connector(Qwen)` to mimic DINOv3(render) could distort the shared
connector and hurt text→3D (which has no render). Three protections:

1. **Align via the projection head, not the connector itself** (this is exactly
   why REPA uses h_φ). Loss = `cos( DINOv3, h_φ(cond) )`, NOT `cos(DINOv3, cond)`.
   The connector output only needs to be **DINOv3-mappable**, not DINOv3-equal →
   it keeps freedom to also serve text. h_φ is **train-only, dropped at inference**
   (no cond-path cost, no constraint on the actual cond used). ⇒ With a head, do
   NOT start identity — the head is the text-protection mechanism, keep it.
2. **Alignment only fires on image-task steps.** No render → no DINOv3 target →
   `L_align=0`. On text_to_3d steps the connector gets pure text-flow gradient,
   fully unaffected. text_to_3d weight (0.2 in mix_3d_only) keeps training the
   text pathway throughout.
3. **Safety net:** sample BOTH text→3D and image→3D periodically during the run;
   if text degrades, lower λ_align or raise text weight.

### Scope / caveats
- Helps **image_to_3d / multi_image_to_3d** directly (text_to_3d has no render →
  no DINOv3 target; protected as above). For multi-image, align each view's Qwen
  tokens to that view's DINOv3 features.
- Adds one frozen DINOv3-L forward per step (cheap vs 3.9B flow).
- Does NOT replace more steps / possible unfreezing — but should lift the cond
  pathway and likely sharpen geometry. It's the most on-target single lever.
- Watch: don't over-weight λ_align or the connector collapses to "mimic DINOv3"
  and loses any Qwen-semantic advantage for text. Keep flow loss dominant.

### Minimal implementation sketch
- Add `DinoV3FeatureExtractor` (reuse TRELLIS's) to the model, frozen.
- In `compute_cascade_flow_loss` (or the native forward), when a render is present:
  compute `dino = extractor(render@512)`, `cond = connector(vlm_hidden)`,
  resample cond grid → 32×32, add `λ·(1 - cos)` to the loss.
- Gate on task: only image/multi-image. Log `align_cos` to wandb.
- Cheapest A/B: fork a short run from ckpt-15000 with alignment ON vs OFF, sample
  at +5k steps, compare violin/aircraft/lamp geometry.

## Open questions to decide with weikaih
- Pooled (1) vs spatial-resampled (2) first?
- Align connector output (post-LayerNorm) or the pre-norm features?
- Add the MLP head or start identity (dims match)?
- Run as a fresh experiment vs resume-from-ckpt-15000 + turn alignment on?
