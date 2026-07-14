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
