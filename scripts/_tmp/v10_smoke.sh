#!/bin/bash
source /fsx/home/weikai.huang/miniconda3/etc/profile.d/conda.sh
conda activate blip3o_trellis
cd /fsx/home/weikai.huang/3dgen/model/BLIP3o
# GPU 0 is where the molmo assist workload lives; it does not honour the
# workload reservation (that only stops the benchmark), so leave it alone
# rather than fight it — 7 ranks is just as good a test of the collectives.
NP=${NP:-7}
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,4,5,6,7}
echo "SMOKE on $NP GPUs (devices $CUDA_VISIBLE_DEVICES)"
export NNODES=1 COMPILE_SS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GEOTEX_IM_TOK_PER_VIEW=64
export HF_HOME=/fsx/home/weikai.huang/hf_cache
# v2.2 VLM (COND_VLM_CKPT deliberately UNSET) + s3_t50 three-tower warm start
RUN_TAG=v10smoke FROM_SCRATCH=False BIDIR=True FUSED=True COUPLING=union \
SHAPE_INIT=runs/s3_shape_t50b/checkpoint-8000 TEX_INIT=runs/s3_tex_t50b/checkpoint-8000 \
SS_INIT=runs/s3_ss_t50b/checkpoint-14000 \
MIX=configs/mix_s3_splits_live1800k.yaml MAX_STEPS=40 PER_GPU_BS=${BS:-1} EFF_BS=$((${BS:-1}*NP)) LR=5e-5 \
WORKERS=4 ELASTIC=False GC=1.0 SAVE_STEPS=100000 REPORT_TO=none \
UNFREEZE_GEO=True GEO_LOSS_W=1.0 \
P_CORNER=0.1 P_CORNER2=0.2 DINO_DROP=0.1 QWEN_DROP=0.1 \
P_SOLO=0.34 P_LAG=0.33 \
bash scripts/train_native_geotex.sh "$NP"
RC=$?
echo "SMOKE-EXIT $RC"
exit $RC
