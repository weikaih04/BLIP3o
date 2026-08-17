import json, os, numpy as np, itertools
SRC="/fsx/home/weikai.huang/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/vlm_filtered_all_matclean.jsonl"
CACHE="/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1"
OUT="/fsx/home/weikai.huang/3dgen/model/BLIP3o/manifests/overfit10.jsonl"
picked=[]
for line in itertools.islice(open(SRC), 60000):
    if len(picked)>=10: break
    r=json.loads(line); s=r["sha256"]
    if not r.get("pbr_latent_512") or r["pbr_latent_512"]=="None": continue
    if float(r.get("aesthetic_score") or 0) < 4.5: continue
    if not os.path.exists(f"{CACHE}/{s[:2]}/{s}/v000.npz"): continue
    try:                                   # 体素数必须 < 8192,否则会被 loader 重采样掉
        n=int(np.load(r["shape_latent_512"])["coords"].shape[0])
    except Exception: continue
    if not (800 <= n <= 6000): continue     # 避开极端,取中段
    picked.append((r, n))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,"w") as f:
    for r,_ in picked: f.write(json.dumps(r)+"\n")
print(f"选出 {len(picked)} 个 · 体素数 {[n for _,n in picked]}")
print(f"子集分布 { {r['subset'] for r,_ in picked} } → {OUT}")
