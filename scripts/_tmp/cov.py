import json, os, itertools
from collections import Counter
M="/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v5_vlm_filtered/vlm_filtered_all_matclean.jsonl"
C="/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1"
print("存在:", os.path.exists(M))
tot=0; hit=0; sub=Counter(); subhit=Counter(); pbr=0
for line in open(M):
    r=json.loads(line); s=r["sha256"]; tot+=1; sub[r["subset"]]+=1
    if r.get("pbr_latent_512") and r["pbr_latent_512"]!="None": pbr+=1
    if os.path.exists(f"{C}/{s[:2]}/{s}/v000.npz"): hit+=1; subhit[r["subset"]]+=1
print(f"总行 {tot:,} · 有 pbr {pbr:,} ({pbr/tot:.1%}) · 有 cond 缓存 {hit:,} ({hit/tot:.1%})")
for k,v in sub.most_common():
    print(f"  {k:26} {v:8,}  缓存 {subhit[k]:8,} ({subhit[k]/v:6.1%})")
