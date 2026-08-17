import json, hashlib
VD="/fsx/home/weikai.huang/3dgen/data/_decgt/material_verdict_v5.jsonl"
ROOT="/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered"
BAD={"BUG","BUG_partial"}          # 判定值大小写混用,原样比较
bad=set(); allv=set(); n=0
for line in open(VD):
    n+=1; d=json.loads(line); s=d.get("sha") or d.get("sha256")
    allv.add(s)
    if d.get("verdict") in BAD: bad.add(s)
print(f"判定文件 {n:,} 行 · 坏图 {len(bad):,}")
heldout=lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8],16) % 100 == 0
for name in ("vlm_filtered_all","vlm_filtered_capT"):
    tr=open(f"{ROOT}/{name}_matclean.jsonl","w"); ho=open(f"{ROOT}/{name}_matclean_heldout.jsonl","w")
    tot=drop=hn=kn=cov=0
    for line in open(f"{ROOT}/{name}.jsonl"):
        tot+=1; s=json.loads(line)["sha256"]
        if s in allv: cov+=1                      # join 覆盖率,防止 key 对不上却静默算出 0 坏图
        if s in bad: drop+=1; continue
        if heldout(s): ho.write(line); hn+=1
        else: tr.write(line); kn+=1
    tr.close(); ho.close()
    assert cov > 0.9*tot, f"{name}: 判定文件只覆盖 {cov}/{tot},key 对不上,结果不可信"
    print(f"{name}: {tot:,} · 判定覆盖 {cov/tot:.1%} → 剔坏图 {drop:,} ({drop/tot:.2%}) "
          f"→ 训练 {kn:,} + 留出 {hn:,}")
