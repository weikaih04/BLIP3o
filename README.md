# WilD3DGen

**WilD3DGen** (formerly referred to internally by the repo dir name **BLIP3o**; NOT Salesforce's **BLIP3o-NEXT** — that is a separate upstream project) — a VLM-conditioned 3D generation stack (3D-VLM v2.x + TRELLIS.2 flows).

> Note on naming: the repository directory (`.../3dgen/model/BLIP3o`) and the git branch (`BLIP3o-NEXT`) intentionally keep their historical names — dozens of scripts and active tooling hardcode these paths. Only the project name changed. `docs/README.md` is the upstream Salesforce BLIP3o-NEXT readme kept for reference.

## Where to start

- **Training handoff / main docs:** [`docs/TRAINING_HANDOFF.md`](docs/TRAINING_HANDOFF.md) (see also `docs/TRAINING_PLAN.md`, `docs/RESULTS.md`, and the design docs in `docs/`)
- **Key entry points:**
  - `train_native.py` — native (single-process-group) trainer
  - `scripts/train_native_split.sh` — split 3-stage launcher (ss / shape / tex)
  - `inference.py` — inference driver
  - `scripts/` — ablation, caching, eval, and data-build tooling (e.g. `build_mds.py`, `build_vlm_cache*.py`, `ablate_*`)
- **Configs:** `configs/`

## Conventions that change between checkpoints

### Multi-image conditioning: qwen tokens per view

`IM_QWEN_TOK_PER_VIEW` is **64 for `s3_shape_t50` and every checkpoint before it, and 128 from the
next one onward.** At 64 a view carries 256²px and four of them total 256 qwen tokens — one
sixteenth of what the single-image path gets for a single view (1024), which is the leading
suspect for why the IM arm ignores view *content* (four different views score the same as four
copies of one). 128 gives 362²px/view and 512 tokens for four views.

Because this straddles checkpoints, three things must move together and none of them is inferable
from the others:

| where | value today | note |
|---|---|---|
| cache `_meta.json` (`vlm_hidden_cache/v22_im4r`) | `im_qwen_tok_per_view: 64` | the only IM cache that exists; **the source of truth** |
| `trellis2_blip3o/live_cond.py:60` | `IM_QWEN_TOK_PER_VIEW` | must equal the cache the checkpoint trained on |
| `scripts/diag_im_multiview.py` | hardcoded `10 + 66*v … +64` | silently mis-indexes if the cache is rebuilt at 128 — **no error, just wrong token slices** |

**Rule: never bump one without the other two.** Running a 128 constant against the 64 cache puts
the multi-image path off-distribution for `s3_shape_t50`, and the diagnostic will keep reporting
confidently on the wrong token spans. Ideally these constants should be *read from the cond root's
`_meta.json`* rather than copied into source — a copied constant is a value that expires silently
the next time the cache is rebuilt.
