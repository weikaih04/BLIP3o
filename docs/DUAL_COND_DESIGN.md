# Dual-Branch Conditioning + Multi-Image — Design & Plan

Status: **in progress** (core module landed; wiring + SLAT + training TODO). Created 2026-06-01.
Supersedes the "replace DINOv3 with Qwen" cond and the `dino_align` aux-loss crutch.

Related: `reference_know3d_paper`, `reference_cgmllm_paper`, `RESULTS.md` (Open architecture
decision), `DINO_ALIGNMENT_DESIGN.md` (the approach this replaces), `MULTI_TASK_DATA.md`.

---

## 1. Problem

Our current path **replaces** TRELLIS2's pretrained DINOv3 image cross-attn cond with a
from-scratch Qwen connector. The pretrained flow then receives a cond distribution it never
saw → unstable: high seed variance, collapse on some seeds (validated 2026-06-01, RESULTS.md).
`dino_align` (REPA aux loss) was a crutch to pull Qwen toward the DINOv3 manifold; it is itself
unstable (2k→5k seed-variance) and does nothing for text→3D.

Separately, **multi-image** input has a known failure (official + community TRELLIS2): the model
treats K views as K *different objects* and fuses junk. Existing community impls (ComfyUI-Trellis2,
OpsiClear) concat DINOv3 tokens but add **no per-view marker** — DeepWiki explicitly calls this
"a limitation not addressed." Issue #103 "multi-image inputs is worse."

## 2. Fix (Know3D-style additive dual branch)

Keep TRELLIS2's original DINOv3 cross-attn as a **geometry anchor**; **add** a parallel,
zero-init Qwen cross-attn so the pretrained flow is never disrupted. Per DiT block:

```
x_out  = block(h, t, H_dino, cond_mask=None)        # ORIGINAL cross-attn ← DINOv3 anchor
x_out += gate · cross_attn_qwen(norm(x_out), H_qwen)  # NEW branch; gate = per-block scalar, zero-init
```

- **H_dino** = raw DINOv3 features (1024-d) — *exactly* the distribution the pretrained TRELLIS
  cross-attn was trained on (`get_cond` feeds `image_cond_model(image)` straight in, no projection).
  So the anchor works immediately, no retraining of the visible-geometry path.
- **H_qwen** = `connector(Qwen hidden)` — the unified cond present on every task; **sole cond for
  text→3D** (where H_dino is a null token).
- **gate = per-block scalar** (one learnable scalar per DiT block / depth), zero-init ⇒ ΔF=0 at start
  (behaviour == pretrained flow), grows as the branch learns. NOT a linear gate — a linear after the
  cross-attn's own `to_out` is redundant (two linears, no nonlinearity ⇒ collapses to one). The scalar
  is the non-redundant, directly-readable "how much this depth uses Qwen" knob (logged `dual/gate_abs`).

### Per-task conditioning
| task | H_dino (original cross-attn) | H_qwen (new branch) |
|------|------------------------------|---------------------|
| image_to_3d | DINOv3(input view), 1024 tok | connector(Qwen: img+text) |
| multi_image_to_3d | concat of V views' DINOv3 + per-view embed | connector(Qwen: V imgs+text) — Qwen separates views natively |
| text_to_3d | null (single zero token) → unconditional anchor | connector(Qwen: text) — carries everything |

### Why dual branch (not concat into one memory)
Separate cross-attns → each source has its own Wk/Wv + `qk_rms_norm_cross` → no cross-source
norm domination; independent token budgets (each ≤ 8192, not shared). Compute: cross-attn is
`L_3d × L_ctx` (linear); SS block attention +~20%; SLAT bottleneck (voxel self-attn) untouched.

## 3. Multi-image + per-view embedding (the part nobody else has)

Concat is the proven mechanism (ComfyUI does it in `get_cond`, fuse-once-before-cross-attn). Our
addition: a **learned per-view embedding** `view_embed[v]` (1024-d, zero-init) added to view v's
whole DINOv3 block before concat, so the flow can group tokens by view and learn "K views of ONE
object" (not K objects) from multi-view training data. Zero-init ⇒ single-view is unaffected at start.

- Only the **DINOv3 anchor** branch needs this — Qwen separates images natively via
  `<|vision_start|>/<|vision_end|>` + its own position ids.
- View-index embedding, **not** camera pose: TRELLIS is pose-free / canonical-space; the
  "same object, different angle" semantics is learned from data. Pose is a later upgrade if needed.
- 4 views @512 = 4×1024 = 4096 anchor tokens < 8192 cap.

## 4. Implementation status & files

**Done**
- `trellis2_blip3o/dual_cond.py` — `DualCondRouter` (stashes H_dino + per-view embed table,
  `build_dino_anchor`), `DualCondInjectBlock` (dense SS wrapper: re-purposes the cascade's cond
  arg as H_qwen, substitutes H_dino into the original cross-attn, adds zero-init Qwen cross-attn),
  `install_dual_routing`. No third-party TRELLIS code touched.

**TODO**
1. **Wire into `trellis_native_vlm.py`** (behind `config.dual_cond`):
   - build the frozen DINOv3 extractor (reuse `dino_align.DinoV3FeatureExtractor`, 512px).
   - `__init__`: build `DualCondRouter`, `install_dual_routing(self.ss_flow, router)`.
   - `forward`: from `dino_images` (M,3,512,512) and `V = M // B`, run DINOv3 per view →
     `router.build_dino_anchor([feat_v for v])`; text (no dino_images) → `H_dino = zeros(B,1,1024)`;
     `router.set_dino(H_dino)`; call `compute_cascade_flow_loss` as usual (H_qwen rides the normal
     connector path; wrappers inject); `router.clear()`.
   - drop/disable `dino_align` when `dual_cond` is on.
2. **SLAT (sparse) dual branch** — SLAT blocks operate on `sp.SparseTensor`; the new Qwen
   cross-attn must be a sparse cross-attn. Either (a) wrapper owning a sparse cross-attn, or
   (b) edit the sparse block class (we already edited `modulated.py` for cond_mask). Decide after SS works.
3. **Inference** (`tests/test_native_infer.py`): produce H_dino from the input view(s), set router,
   run. Multi-image inference path.
4. **Training flags** (`train_native.py`): `--dual_cond`, `--dual_cond_max_views`,
   `--init_from_pretrained_trellis` (NOT our 30k Qwen ckpt — see §6).
5. **ZeRO-2 graph consistency**: H_qwen is non-None on every task (text included) ⇒ the Qwen branch
   always runs ⇒ no graph-structure drift across image/text steps (no `_zero_touch` needed). The
   DINOv3 branch is *inside* the original block (always called); for text H_dino is a zero token, so
   the block runs identically. Verify on the 2-GPU local DeepSpeed repro before cluster.

## 5. Data — mostly already in place

`data/tasks/threed.py` already provides everything:
- view-sampling modes **T / I1 / IM** (text / 1 view / n∈[2,max_views]); `set_batch_params` locks
  `n_views` per batch ⇒ homogeneous V within a batch ⇒ H_dino has uniform token count, **no padding/mask**.
- emits `dino_images` (M,3,512,512) at DINOv3's 512px in the **same flat order** as the cond views.

**Data work needed: minimal.**
- Confirm `dino_images` flat order is view-major per sample so `V = M // B` and reshape `(B,V,N,1024)`
  is correct (it is — emitted in cond-view order).
- Ensure the multi-image mixture weight is non-trivial so `view_embed[1..]` actually gets gradient
  (single-view-only training leaves higher view slots at zero-init).
- (Optional later) carry camera elevation/azimuth per view if we upgrade to pose embeddings.

## 6. Training plan

- **Init from the ORIGINAL pretrained TRELLIS.2-4B flow** (DINOv3-native cross-attn) + fresh
  zero-init Qwen branch + fresh per-view embed + connector. Do **NOT** resume our 30k Qwen ckpt —
  its original cross-attn has already drifted toward Qwen and is no longer DINOv3-native, which
  defeats the anchor.
- **Stage order**: SS dual-branch first (dense, tractable) → verify init behaviour == pretrained
  (ΔF_qwen=0) and that image overfit re-converges → add SLAT → enable text + multi-image mixture.
- **Loss**: unchanged cascade flow-matching (`compute_cascade_flow_loss`); no dino_align aux loss.
- **CFG**: H_qwen keeps mask_drop CFG dropout (existing). H_dino CFG dropout (drop to null) is a
  later refinement; for v1 keep anchor always on for image tasks.
- **Eval**: per `feedback_results_log_convention` — sweep ≥3 seeds; log to `RESULTS.md` with date.
  Expect lower seed variance than the replace-DINOv3 baseline (the anchor stabilizes visible geometry).

## 7. Open risks
- **Anchor lean**: image tasks may over-rely on the anchor → Qwen branch undertrained → text→3D
  (no anchor) suffers. Mitigation: random anchor-drop on image tasks, and/or up-weight text in the
  mixture. Monitor text→3D quality vs image→3D.
- **SLAT sparse cross-attn** complexity (§4.2).
- **Two-cond CFG** semantics (independent vs joint dropout) — defer.

## 8. Task checklist
- [x] `dual_cond.py` core (DualCondRouter / DualCondInjectBlock / install) — SS dense, separate zero-init gate
- [x] Wire `trellis_native_vlm.py` (`_build_dino_anchor` from dino_images V=M//B; install_dual_routing on ss_flow; set_dino/clear in forward; frozen DINOv3 extractor as non-child)
- [x] `--dual_cond` / `--dual_cond_max_views` flags in `train_native.py` + force dual params trainable in `_apply_flow_freeze` (else last{NN} freezes the Qwen branch in early blocks)
- [ ] Smoke: build w/ dual_cond=True, 1 fwd — assert gate_norm≈0 at init (behaviour==pretrained) + loss finite
- [ ] Local 2-GPU ZeRO-2 smoke (mixed image+text, graph consistency)
- [ ] SS-only image overfit re-converge check (≥3 seeds → RESULTS.md)
- [ ] **Inference** (`test_native_infer.py`): build H_dino from the input image at 512 (the inference dataset `TR2NativeVLMDataset` does NOT emit dino_images — build directly via `model._dino_extractor`), `model.dual_router.set_dino(H_dino)` before `sample_sparse_structure`. **CFG decision: anchor always-ON for both cond and neg passes** (matches training — H_dino is never dropped; only H_qwen has mask_drop CFG dropout), so CFG guides the Qwen/semantic part while the anchor is always-on reconstruction guidance. text→3D inference → H_dino = null token.
- [x] SLAT sparse dual branch (`DualCondSparseInjectBlock`; install dispatches dense vs sparse by class name; installed on shape+tex too). Forward-validated: dual ran through SS+shape SLAT (30 sparse blocks) before tex OOM.
- [x] **512+dual backward VALIDATED on 2-GPU with `--flow_tune last20`** (8/8 steps, 135s, gate_abs 0→1.67e-4). last40 OOMs (step 5) but last20 fits — partial-FT helps (fewer trained layers → smaller optimizer states + less backward-stored activation). Dual raw cost ~10GB (SS: 32.5→41.6GB). **last20 suits dual** (anchor preserves the original flow → little retraining needed; Qwen branch always fully trained via the `_apply_flow_freeze` force-trainable clause). No mem-optimization needed for 2-GPU.
- [ ] Multi-image per-view embed gradient check (needs multi-image in the mixture)
- [ ] Full mixture run (image+multi-image+text), seed sweep, RESULTS.md entry

**Note (init source):** when training, point `--init_from_checkpoint` at PRETRAINED TRELLIS weights
(or no init_from_checkpoint → builders already load pretrained TRELLIS), NOT the 30k Qwen run.
