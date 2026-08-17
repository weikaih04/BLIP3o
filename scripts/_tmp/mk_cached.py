import json, os, hashlib
M="/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v5_vlm_filtered/vlm_filtered_all_matclean.jsonl"
C="/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1"
O="/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v5_vlm_filtered"
heldout=lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8],16)%100==0   # 按 sha 稳定切 1% 同分布留出
tr=open(f"{O}/vlm_filtered_all_matclean_cached.jsonl","w")
ho=open(f"{O}/vlm_filtered_all_matclean_cached_heldout.jsonl","w")
tot=hit=k=h=0
for line in open(M):
    tot+=1; r=json.loads(line); s=r["sha256"]
    if not os.path.exists(f"{C}/{s[:2]}/{s}/v000.npz"): continue
    hit+=1
    if heldout(s): ho.write(line); h+=1
    else: tr.write(line); k+=1
tr.close(); ho.close()
print(f"{tot:,} → 有 cond 缓存 {hit:,} ({hit/tot:.1%}) → 训练 {k:,} + 同分布留出 {h:,}")
