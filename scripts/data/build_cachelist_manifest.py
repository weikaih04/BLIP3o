import json, os
IMG="/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1"
IM4="/fsx/home/weikai.huang/3dgen/vlm_hidden_cache/v22_im4r"
def shas(root):
    s=set()
    for p in os.listdir(root):
        d=os.path.join(root,p)
        if os.path.isdir(d): s.update(os.listdir(d))
    return s
img=shas(IMG); im4=shas(IM4)
print(f"图像 cond 缓存 {len(img):,} · 多视角 IM 缓存 {len(im4):,} · 交集 {len(img&im4):,}")
M="/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v5_vlm_filtered/vlm_filtered_all_matclean.jsonl"
O="/fsx/home/weikai.huang/3dgen/model/BLIP3o/manifests"
os.makedirs(O, exist_ok=True)
import hashlib
ho=lambda s:int(hashlib.md5(s.encode()).hexdigest()[:8],16)%100==0   # 1% 同分布留出(skill §5 要求)
ftr=open(f"{O}/v5_matclean_cached_train.jsonl","w"); fho=open(f"{O}/v5_matclean_cached_heldout.jsonl","w")
tot=k=h=0
for line in open(M):
    tot+=1; s=json.loads(line)["sha256"]
    if s not in img: continue
    if ho(s): fho.write(line); h+=1
    else: ftr.write(line); k+=1
ftr.close(); fho.close()
print(f"{tot:,} → 有图像缓存 {k+h:,} → 训练 {k:,} + 留出 {h:,}")
