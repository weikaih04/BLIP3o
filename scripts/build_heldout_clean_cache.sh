#!/bin/bash
# Build ALL THREE conditioning modalities for the content-deduplicated held-out set
# (scripts/build_heldout_clean.py output).  Runs on ONE already-held node, NGPU shards.
#
#   I1  v000.npz (+d000.npz, then merged in-place)  -> $CR
#         phase V : build_vlm_cache_v22.py --mode views, VIEW_SET=B
#         phase D : build_dino_cache.py --max_views 1 --image_size 512 --crop_to_object 1
#         phase M : merge_vd_cache.py  (fusion cond reads ONE npz)
#       VIEW_SET=B is not cosmetic: v22_3dvlm_tok1024_mv1 (the I1 cond cache the model
#       trained on) is set B, and eval_fusion_v22.good_view_b assumes it.
#   T   t000..t003.npz -> $CR   (--mode captions, over the capT manifest)
#   IM  m00.npz        -> $IMR  (--mode im4 --im4_view_sampling WEIGHTED,
#                                --im_tok_per_view 64 --dino_image_size 320)
#       These three settings are copied from v22_im4r/_meta.json, i.e. the cache the model
#       ACTUALLY trained on.  The old held-out cache v22_heldout_im4l used the legacy
#       `fixed` good-band pick -> a train/eval mismatch that had to be corrected once
#       already.  Do not "simplify" back to the builder defaults (256/512/fixed).
#
# Usage:  NGPU=4 GPUS="0,1,2,3" bash scripts/build_heldout_clean_cache.sh
#         Check nvidia-smi first; leave headroom for the neighbours on the node.
set -uo pipefail
REPO=/fsx/home/weikai.huang/3dgen/model/BLIP3o
source /fsx/home/weikai.huang/miniconda3/bin/activate blip3o_trellis
cd "$REPO"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export HF_HOME=/fsx/home/weikai.huang/.cache/huggingface
export ATTN_BACKEND=sdpa

HO=/fsx/home/weikai.huang/3dgen/im_probe/heldout_clean
MANI=${MANI:-$HO/heldout_clean.jsonl}
MANI_CAP=${MANI_CAP:-$HO/heldout_clean_capT.jsonl}
# NOT v22_heldout: merge_vd_cache.py hard-refuses any d_root whose path contains
# "v22_heldout" or "v22_3dvlm_tok1024_mv1" (STALE-DINO GUARD, 2026-07-13 — renders_cond was
# re-rendered ~07-12 and the d-files in those roots came from the OLD renders).  Our d-files
# are built fresh from the CURRENT renders seconds earlier, but the guard is name-based, so
# phase M silently no-ops and v000 ships without its DINO segment.  A separate root is also
# safer: v22_heldout is read live by other people's evals.
CR=${CR:-/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_hoclean}
IMR=${IMR:-/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_clean_im4r}
GPUS=${GPUS:-0,1,2,3}
IFS=',' read -ra GARR <<< "$GPUS"
NG=${#GARR[@]}
LOGD=$REPO/runs/cache_logs/heldout_clean; mkdir -p "$LOGD"

CK_SRC=/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000
CK_LOCAL=/dev/shm/v22ckpt_hoclean
mkdir -p "$CK_LOCAL"
rsync -a --exclude 'rng_state_*.pth' --exclude 'global_step*' --exclude '*.optim*' \
      --exclude 'latest' "$CK_SRC/" "$CK_LOCAL/" 2>&1 | tail -1
echo "[hoclean] node=$(hostname) gpus=$GPUS n=$(wc -l < "$MANI") ckpt staged $(date)"

# $CR is a SHARED cache root that other people's evals read right now.  `--mode views`
# shard 0 calls vlm_cache.write_meta, which REWRITES _meta.json wholesale (dropping the
# dino_* contract until phase D puts it back).  Snapshot it and re-merge at the end so no
# key can be lost by an interrupted run.
[ -f "$CR/_meta.json" ] && cp "$CR/_meta.json" "$LOGD/meta_before.json"

run_shards () {   # $1 = phase tag, rest = command template with %S/%N placeholders
  local tag=$1; shift
  local i=0
  for g in "${GARR[@]}"; do
    local cmd=${*//%S/$i}; cmd=${cmd//%N/$NG}
    CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > "$LOGD/${tag}_$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  echo "[hoclean] phase $tag done $(date)"; tail -n1 "$LOGD/${tag}"_*.log
}

echo "===== Phase V (I1 qwen, VIEW_SET=B) ====="
run_shards v "VIEW_SET=B python scripts/build_vlm_cache_v22.py --mode views \
  --manifests $MANI --out_root $CR --vlm $CK_LOCAL --vlm_name $CK_SRC \
  --shard %S --num_shards %N --min_free_tb 0"

echo "===== Phase D (DINOv3 512, GOOD_VIEWS=1 VIEW_SET=B) ====="
run_shards d "GOOD_VIEWS=1 VIEW_SET=B python scripts/build_dino_cache.py \
  --manifest $MANI --out_root $CR --crop_to_object 1 --max_views 1 --image_size 512 \
  --shard %S --num_shards %N"

echo "===== Phase M (merge v+d) ====="
run_shards m "python scripts/merge_vd_cache.py --root $CR --manifest $MANI \
  --max_views 1 --shard %S --num_shards %N"

echo "===== Phase T (captions t000..t003) ====="
run_shards t "python scripts/build_vlm_cache_v22.py --mode captions \
  --manifests $MANI_CAP --out_root $CR --vlm $CK_LOCAL --vlm_name $CK_SRC \
  --shard %S --num_shards %N --min_free_tb 0"

echo "===== Phase IM (m00, WEIGHTED 4-of-16, tok64/dino320 = v22_im4r contract) ====="
run_shards im "python scripts/build_vlm_cache_v22.py --mode im4 --im4_view_sampling weighted \
  --im_tok_per_view 64 --dino_image_size 320 \
  --manifests $MANI --out_root $IMR --vlm $CK_LOCAL --vlm_name $CK_SRC \
  --shard %S --num_shards %N --min_free_tb 0"

if [ -f "$LOGD/meta_before.json" ]; then
  python - <<PY
import json
before=json.load(open("$LOGD/meta_before.json"))
cur=json.load(open("$CR/_meta.json"))
lost={k:v for k,v in before.items() if k not in cur}
if lost:
    cur.update(lost); json.dump(cur, open("$CR/_meta.json","w"), indent=2)
print(f"[hoclean] meta guard: restored {len(lost)} key(s) {sorted(lost)}")
PY
fi

echo "===== coverage check ====="
python - <<PY
import json, os
CR="$CR"; IMR="$IMR"
recs=[json.loads(l) for l in open("$MANI")]
def has(root, sha, k): return os.path.exists(os.path.join(root, sha[:2], sha, k))
miss={k:[] for k in ("v000","t000","t001","t002","t003","m00")}
for r in recs:
    s=r["sha256"]
    for k in ("v000","t000","t001","t002","t003"):
        if not has(CR,s,k+".npz"): miss[k].append(s[:8])
    if not has(IMR,s,"m00.npz"): miss["m00"].append(s[:8])
import numpy as np
print(f"[coverage] n={len(recs)}")
for k,v in miss.items(): print(f"  {k}: missing {len(v)} {v[:8]}")
# dino really merged into v000?
nod=[r["sha256"][:8] for r in recs
     if has(CR,r["sha256"],"v000.npz") and
        "dino_hidden" not in np.load(os.path.join(CR,r["sha256"][:2],r["sha256"],"v000.npz")).files]
print(f"  v000 without merged dino segment: {len(nod)} {nod[:8]}")
PY
echo "[hoclean] ALL DONE $(date)"
