import json, os, hashlib
VD="/fsx/home/weikai.huang/3dgen/data/_decgt/material_verdict_v5.jsonl"
SRC="/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_capT.jsonl"
IMG="/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1"
O="/fsx/home/weikai.huang/3dgen/model/BLIP3o/manifests"
BAD={"BUG","BUG_partial"}          # 判定值大小写混用,原样比较([a-z_]+ 会静默匹配不到
bad=set(); seen=set()
for line in open(VD):
    d=json.loads(line); s=d.get("sha") or d.get("sha256"); seen.add(s)
    if d.get("verdict") in BAD: bad.add(s)
cached=set()
for p in os.listdir(IMG):
    d=os.path.join(IMG,p)
    if os.path.isdir(d): cached.update(os.listdir(d))
ho=lambda s:int(hashlib.md5(s.encode()).hexdigest()[:8],16)%100==0   # 与图像路完全同一条规则
tr=open(f"{O}/v4_capT_matclean_cached_train.jsonl","w")
hh=open(f"{O}/v4_capT_matclean_cached_heldout.jsonl","w")
tot=cov=drop=nocache=k=h=0
for line in open(SRC):
    tot+=1; s=json.loads(line)["sha256"]
    if s in seen: cov+=1
    if s in bad: drop+=1; continue
    if s not in cached: nocache+=1; continue
    if ho(s): hh.write(line); h+=1
    else: tr.write(line); k+=1
tr.close(); hh.close()
assert cov > 0.9*tot, f"判定文件只覆盖 {cov}/{tot},key 对不上,结果不可信"
print(f"capT {tot:,} · 判定覆盖 {cov/tot:.1%} → 剔材质坏图 {drop:,} ({drop/tot:.2%}) "
      f"· 剔无缓存 {nocache:,} → 训练 {k:,} + 留出 {h:,}")
