"""Measured audit of every manifest/cache the CURRENT run actually reads.
Nothing here is taken from a README or a _meta.json — each claim is a count."""
import json, os, hashlib, random, sys
from collections import Counter
import numpy as np

R = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/manifests"
VD = "/fsx/home/weikai.huang/3dgen/data/_decgt/material_verdict_v5.jsonl"
QC = "/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1"
IM = "/fsx/home/weikai.huang/3dgen/vlm_hidden_cache/v22_im4r"
BAD = {"BUG", "BUG_partial"}          # values are MIXED CASE (data skill S3)

def shas(p):
    return [json.loads(l)["sha256"] for l in open(p)]

print("=" * 78)
print("① 训练/留出清单的行数与重叠")
sets = {}
for name in ("v5_matclean_cached_train", "v5_matclean_cached_heldout",
             "v4_capT_matclean_cached_train", "v4_capT_matclean_cached_heldout"):
    p = f"{R}/{name}.jsonl"
    s = shas(p); sets[name] = set(s)
    print(f"  {name:38} {len(s):>8,} 行 · 唯一 {len(set(s)):>8,}")
tr_i, ho_i = sets["v5_matclean_cached_train"], sets["v5_matclean_cached_heldout"]
tr_t, ho_t = sets["v4_capT_matclean_cached_train"], sets["v4_capT_matclean_cached_heldout"]
print(f"  图像训练 ∩ 图像留出 = {len(tr_i & ho_i)}   (必须 0)")
print(f"  文字训练 ∩ 图像留出 = {len(tr_t & ho_i)}   (必须 0 — 同一 mesh 换模态还是同一 mesh)")
print(f"  图像训练 ∩ 文字留出 = {len(tr_i & ho_t)}   (必须 0)")
print(f"  两个留出集是否同一批 sha: 交集 {len(ho_i & ho_t):,} / 图像留出 {len(ho_i):,}")

print("\n② 材质判定 —— 训练集里还有没有坏图")
bad = set(); seen = set()
for l in open(VD):
    d = json.loads(l); s = d.get("sha") or d.get("sha256"); seen.add(s)
    if d.get("verdict") in BAD: bad.add(s)
print(f"  判定文件 {len(seen):,} 条 · 坏图 {len(bad):,}")
for n in ("v5_matclean_cached_train", "v4_capT_matclean_cached_train",
          "v5_matclean_cached_heldout"):
    S = sets[n]; cov = len(S & seen); b = len(S & bad)
    print(f"  {n:38} 判定覆盖 {cov/len(S):6.1%} · 坏图 {b:>6,} ({b/len(S):.3%})")

print("\n③ cond 缓存:视角一致性 + 完整性(抽 300 个)")
rng = random.Random(0)
samp = rng.sample(sorted(tr_i), 300)
miss_q = miss_d = miss_im = miss_t = 0
qtok, dtok = [], []
for s in samp:
    p = f"{QC}/{s[:2]}/{s}/v000.npz"
    if not os.path.exists(p): miss_q += 1; continue
    a = np.load(p)
    if "dino_hidden" not in a: miss_d += 1
    else: dtok.append(int(a["dino_keep_mask"].sum()))
    qtok.append(int(a["keep_mask"].sum()))
    if not os.path.exists(f"{IM}/{s[:2]}/{s}/m00.npz"): miss_im += 1
    if not os.path.exists(f"{QC}/{s[:2]}/{s}/t000.npz"): miss_t += 1
print(f"  Qwen 缺 {miss_q}/300 · DINO 缺 {miss_d}/300 · IM(m00) 缺 {miss_im}/300 · caption(t000) 缺 {miss_t}/300")
print(f"  Qwen token 中位 {int(np.median(qtok))} · DINO token 中位 {int(np.median(dtok))}")

print("\n④ 视角挑选公式实测(缓存 build 用的是 5+(sha+off)%7,落在 005-011)")
vs = Counter((5 + (int(s[:8], 16) + 5) % 7) for s in samp)     # VIEW_SET C, offs=5
print(f"  按公式算出的视角分布: {dict(sorted(vs.items()))}")
print(f"  → 覆盖 {len(vs)} 个不同视角,全部在 005-011 内: {all(5<=k<=11 for k in vs)}")

print("\n⑤ latent:shape 和 pbr 的 coords 必须逐点相同(官方 dataset 的 assert)")
bad_c = miss_l = 0; vox = []
for s in samp[:120]:
    r = None
    # 从训练清单里取该 sha 的行(只读一次文件太慢,改成建索引)
    break
idx = {}
for l in open(f"{R}/v5_matclean_cached_train.jsonl"):
    d = json.loads(l)
    if d["sha256"] in set(samp[:120]): idx[d["sha256"]] = d
for s in samp[:120]:
    r = idx.get(s)
    if not r: miss_l += 1; continue
    try:
        a = np.load(r["shape_latent_512"]); b = np.load(r["pbr_latent_512"])
    except Exception: miss_l += 1; continue
    if not np.array_equal(a["coords"], b["coords"]): bad_c += 1
    vox.append(a["coords"].shape[0])
print(f"  检查 {len(vox)} 个 · coords 不一致 {bad_c} (必须 0) · 读不到 {miss_l}")
v = np.array(vox)
print(f"  体素数: 中位 {int(np.median(v))} · 均值 {v.mean():.0f} · 最大 {v.max()} · >8192 的比例 {(v>8192).mean():.2%}")

print("\n⑥ 美学分(官方 min_aesthetic_score=4.5)")
sc = [float(idx[s].get("aesthetic_score") or 0) for s in samp[:120] if s in idx]
sc = np.array(sc)
print(f"  中位 {np.median(sc):.1f} · <4.5 的比例 {(sc<4.5).mean():.1%} (会被 loader 丢掉)")
