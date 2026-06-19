#!/usr/bin/env bash
# Setup B/C: Pre-compute SigLIP-2 + VQ codebook IDs for multi-view renders.
#
# Usage:
#   bash scripts/prep_data_codebook.sh [DATA_DIR=data/objxl_sketchfab] [TA_TOK_CKPT=<path>]
set -euo pipefail
DATA_DIR="${1:-../data/objxl_sketchfab}"
TA_TOK_CKPT="${2:-$BLIP3O_TA_TOK_CKPT}"

if [ -z "${TA_TOK_CKPT}" ]; then
    echo "ERROR: TA_TOK_CKPT not set. Either pass as 2nd arg or export BLIP3O_TA_TOK_CKPT=<path>."
    echo "       This is the BLIP3o-NEXT vision_tower checkpoint for TextAlignedTokenizer."
    exit 1
fi

cd "$(dirname "$0")/.."

python3 -m trellis2_blip3o.codebook_prep \
    --renders-root "${DATA_DIR}/renders_cond" \
    --out-root     "${DATA_DIR}/siglip_codebook_ids" \
    --views 0 1 2 3 \
    --ta-tok-ckpt  "${TA_TOK_CKPT}"

# Now augment the index_setup_BC.jsonl with codebook paths
python3 - <<PY
import json, os, glob
DATA="${DATA_DIR}"
SS_DIR=f"{DATA}/ss_latents/ss_enc_conv3d_16l8_fp16"
REND_DIR=f"{DATA}/renders_cond"
CB_DIR=f"{DATA}/siglip_codebook_ids"

records = []
shas = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{SS_DIR}/*.npz"))
for sha in shas:
    cb = f"{CB_DIR}/{sha}.npz"
    img = f"{REND_DIR}/{sha}/000.png"
    if not (os.path.exists(cb) and os.path.exists(img)):
        continue
    records.append({
        "id": sha,
        "type": "I_2_3D",
        "image": img,
        "txt": "a 3D object",
        "target_ss_latent": f"{SS_DIR}/{sha}.npz",
        "multi_view_renders": [f"{REND_DIR}/{sha}/{v:03d}.png" for v in (0,1,2,3)],
        "siglip_codebook_ids": cb,
    })

out = "data/index_setup_BC.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(records)} samples to {out}")
PY
