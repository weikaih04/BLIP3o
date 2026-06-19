#!/usr/bin/env bash
# Build a JSONL manifest for Setup A from trellis2_sam3d's already-prepared SS latents.
#
# Usage:
#   bash scripts/prep_data_ss_latents.sh [DATA_DIR=<absolute path>] [OUT_JSONL=<rel path>]
#
# Defaults: data lives at /weka/.../world_explore/data/objxl_sketchfab,
#           manifest written to <project>/data/index_setup_A.jsonl
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${1:-/weka/oe-training-default/weikaih/world_explore/data/objxl_sketchfab}"
OUT_REL="${2:-data/index_setup_A.jsonl}"
OUT_ABS="${PROJECT_ROOT}/${OUT_REL}"

mkdir -p "$(dirname "${OUT_ABS}")"

DATA_DIR="${DATA_DIR}" OUT_ABS="${OUT_ABS}" python3 - <<'PY'
import json, os, glob
import pandas as pd

DATA = os.environ["DATA_DIR"]
OUT = os.environ["OUT_ABS"]
SS_DIR = f"{DATA}/ss_latents/ss_enc_conv3d_16l8_fp16"
REND_DIR = f"{DATA}/renders_cond"
META_CSV = f"{DATA}/metadata.csv"

print(f"DATA = {DATA}")
print(f"SS_DIR exists: {os.path.isdir(SS_DIR)}")
print(f"REND_DIR exists: {os.path.isdir(REND_DIR)}")
print(f"metadata.csv: {'loaded' if os.path.exists(META_CSV) else 'MISSING'}")

meta = None
if os.path.exists(META_CSV):
    meta = pd.read_csv(META_CSV, dtype={'sha256': str}, low_memory=False)
    meta = meta.set_index("sha256")

records = []
shas = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{SS_DIR}/*.npz"))
print(f"found {len(shas)} SS latents")

for sha in shas:
    ss_path = f"{SS_DIR}/{sha}.npz"
    img_path = f"{REND_DIR}/{sha}/000.png"
    if not os.path.exists(img_path):
        continue
    caption = "a 3D object"
    if meta is not None and sha in meta.index:
        cap = meta.loc[sha].get("captions", "")
        try:
            if isinstance(cap, str) and cap.startswith("["):
                arr = json.loads(cap)
                if arr:
                    caption = arr[0]
        except Exception:
            caption = str(cap)[:200] or caption
    records.append({
        "id": sha,
        "type": "I_2_3D",
        "image": img_path,
        "txt": caption,
        "target_ss_latent": ss_path,
        "multi_view_renders": [f"{REND_DIR}/{sha}/{v:03d}.png" for v in (0, 1, 2, 3)],
    })

with open(OUT, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(records)} samples to {OUT}")
PY
