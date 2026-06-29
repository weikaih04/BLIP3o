"""Build a unified `ready_v1.jsonl` for unified text/image/multi-image → 3D training.

Joins `data/trellis2/manifests/ready_v1/MANIFEST.csv` with per-subset caption + aesthetic
sources, then emits one JSON line per asset containing only PER-ASSET FACTS:

    sha256, subset,
    ss_latent_64,
    shape_latent_512, shape_latent_1024,
    pbr_latent_512,   pbr_latent_1024,
    renders_dir, n_views,
    captions (list[str]),
    aesthetic_score (float or null)

Sampling/policy decisions (which task, which view, which caption variant, crop, aesthetic
threshold) live in `dataset_native.py`, NOT in this file — so changing the task mix never
requires rebuilding the manifest.

Caption sources:
  TexVerse                                          → caption.json keyed by file_identifier
                                                      (joined to sha256 via metadata.csv)
  ABO / HSSD / Toys4k / ObjaverseXL_{sf,gh}         → metadata.csv `captions` column
                                                      (JSON list of 11 length variants)

Usage:
    python build_index.py \
        --out data/trellis2/manifests/ready_v1/ready_v1.jsonl \
        [--filter_trainable] [--min_aesthetic 4.5] [--require_caption]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

DATA_ROOT = Path(os.environ.get(
    "TRELLIS2_DATA_ROOT",
    "/weka/oe-training-default/weikaih/world_explore/data/trellis2"))
DEFAULT_MANIFEST = DATA_ROOT / "manifests/ready_v1/MANIFEST.csv"

# Directory names (match HANDOFF.md §1 + actual on-disk layout)
SHAPE_DIR = {
    512:  "shape_latents/shape_enc_next_dc_f16c32_fp16_512",
    1024: "shape_latents/shape_enc_next_dc_f16c32_fp16_1024",
}
PBR_DIR = {
    512:  "pbr_latents/tex_enc_next_dc_f16c32_fp16_512",
    1024: "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024",
}
SS_DIR      = "ss_latents/ss_enc_conv3d_16l8_fp16_64"
RENDERS_DIR = "renders_cond"
N_VIEWS     = 16   # data_toolkit/render_cond.py renders 16 views per asset


def _read_metadata_captions(path: Path):
    """Returns {sha256: {"captions": list[str], "aesthetic_score": float|None}}.
    Skips rows where the captions column is empty or not valid JSON."""
    out: Dict[str, Dict] = {}
    if not path.exists():
        return out
    csv.field_size_limit(sys.maxsize)  # captions field can be very long
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sha = row.get("sha256")
            if not sha:
                continue
            aes_raw = row.get("aesthetic_score") or ""
            try:
                aes = float(aes_raw) if aes_raw else None
            except ValueError:
                aes = None
            caps_raw = row.get("captions") or ""
            caps: List[str] = []
            if caps_raw:
                try:
                    parsed = json.loads(caps_raw)
                    if isinstance(parsed, list):
                        caps = [str(c) for c in parsed if c]
                except (json.JSONDecodeError, ValueError):
                    caps = []
            out[sha] = {"captions": caps, "aesthetic_score": aes}
    return out


def _read_texverse_captions(subset_root: Path):
    """TexVerse-specific: metadata.csv has (sha256, file_identifier, local_path) only;
    caption.json is keyed by file_identifier. Join to recover {sha: [caption]}."""
    out: Dict[str, Dict] = {}
    meta_path = subset_root / "metadata.csv"
    cap_path = subset_root / "caption.json"
    if not meta_path.exists() or not cap_path.exists():
        return out
    print(f"  loading TexVerse caption.json (~856K entries)...", flush=True)
    with open(cap_path) as f:
        cap_by_fid: Dict[str, str] = json.load(f)
    csv.field_size_limit(sys.maxsize)
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            sha = row.get("sha256")
            fid = row.get("file_identifier")
            if not sha or not fid:
                continue
            cap = cap_by_fid.get(fid)
            out[sha] = {
                "captions": [cap] if cap else [],
                "aesthetic_score": None,  # TexVerse metadata has no aesthetic_score
            }
    return out


def load_subset_caption_index(subset: str) -> Dict[str, Dict]:
    """Dispatch: TexVerse uses caption.json + UUID join; others use metadata.csv `captions`."""
    subset_root = DATA_ROOT / subset
    if subset == "TexVerse":
        return _read_texverse_captions(subset_root)
    return _read_metadata_captions(subset_root / "metadata.csv")


def build_record(subset: str, sha: str, cap_info: Dict) -> Dict:
    base = DATA_ROOT / subset

    def _maybe(path: Path) -> Optional[str]:
        # We return the path string regardless; let the dataset (or --filter_trainable)
        # decide whether the file exists. This avoids 5×83K stat calls in the common case.
        return str(path)

    return {
        "sha256": sha,
        "subset": subset,
        "ss_latent_64":      _maybe(base / SS_DIR / f"{sha}.npz"),
        "shape_latent_512":  _maybe(base / SHAPE_DIR[512]  / f"{sha}.npz"),
        "shape_latent_1024": _maybe(base / SHAPE_DIR[1024] / f"{sha}.npz"),
        "pbr_latent_512":    _maybe(base / PBR_DIR[512]    / f"{sha}.npz"),
        "pbr_latent_1024":   _maybe(base / PBR_DIR[1024]   / f"{sha}.npz"),
        "renders_dir":       str(base / RENDERS_DIR / sha),
        "n_views":           N_VIEWS,
        "captions":          cap_info.get("captions", []),
        "aesthetic_score":   cap_info.get("aesthetic_score"),
    }


def _trainable(rec: Dict, resolutions=(512, 1024)) -> bool:
    """All 4 latents (ss + shape@res + pbr@res for each res requested) + renders dir present."""
    if not os.path.isfile(rec["ss_latent_64"]):
        return False
    if not os.path.isdir(rec["renders_dir"]):
        return False
    for res in resolutions:
        for k in (f"shape_latent_{res}", f"pbr_latent_{res}"):
            if not os.path.isfile(rec[k]):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="Path to MANIFEST.csv (subset,sha256 rows). "
                             f"Default: {DEFAULT_MANIFEST}")
    parser.add_argument("--out", default=str(DATA_ROOT / "manifests/ready_v1/ready_v1.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--filter_trainable", action="store_true",
                        help="Stat every latent + renders dir; drop rows that aren't fully encoded yet")
    parser.add_argument("--require_caption", action="store_true",
                        help="Drop rows whose caption list is empty (TexVerse misses + some ObjxL_sf)")
    parser.add_argument("--min_aesthetic", type=float, default=None,
                        help="Drop rows with aesthetic_score < this (TexVerse has no score → kept)")
    parser.add_argument("--resolutions", default="512,1024",
                        help="Comma-separated resolutions used by --filter_trainable. Default: 512,1024")
    args = parser.parse_args()

    resolutions = tuple(int(x) for x in args.resolutions.split(",") if x.strip())

    # 1. Read MANIFEST.csv, collect unique subsets
    rows: List[Dict[str, str]] = []
    subsets = set()
    with open(args.manifest, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            subsets.add(r["subset"])
    print(f"[build_index] {len(rows)} rows across {len(subsets)} subsets: {sorted(subsets)}",
          flush=True)

    # 2. Load caption indices per subset
    caption_index: Dict[str, Dict[str, Dict]] = {}
    for s in sorted(subsets):
        print(f"[build_index] loading captions: {s}", flush=True)
        caption_index[s] = load_subset_caption_index(s)
        print(f"  → {len(caption_index[s])} caption entries", flush=True)

    # 3. Build records and apply filters
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_no_caption = 0
    n_low_aesthetic = 0
    n_missing_files = 0
    per_subset: Dict[str, int] = {s: 0 for s in subsets}

    with open(out_path, "w") as out:
        for r in rows:
            sha, subset = r["sha256"], r["subset"]
            cap_info = caption_index[subset].get(sha) or {"captions": [], "aesthetic_score": None}
            rec = build_record(subset, sha, cap_info)

            if args.require_caption and not rec["captions"]:
                n_no_caption += 1
                continue
            aes = rec["aesthetic_score"]
            if args.min_aesthetic is not None and aes is not None and aes < args.min_aesthetic:
                n_low_aesthetic += 1
                continue
            if args.filter_trainable and not _trainable(rec, resolutions=resolutions):
                n_missing_files += 1
                continue

            out.write(json.dumps(rec) + "\n")
            n_written += 1
            per_subset[subset] = per_subset.get(subset, 0) + 1

    print(f"\n[build_index] wrote {n_written} records → {out_path}")
    print("  per subset:")
    for s in sorted(subsets):
        print(f"    {s:<24s} {per_subset[s]}")
    if args.require_caption:
        print(f"  dropped (no caption):       {n_no_caption}")
    if args.min_aesthetic is not None:
        print(f"  dropped (aesthetic<{args.min_aesthetic}): {n_low_aesthetic}")
    if args.filter_trainable:
        print(f"  dropped (files missing):    {n_missing_files}")


if __name__ == "__main__":
    main()
