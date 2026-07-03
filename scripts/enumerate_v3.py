"""Enumerate ready_v3 (= ready_v2 5 non-TexVerse subsets + reprocessed TexVerse) trainable assets
from on-disk latents, and report the cond-cache gap. Trainable = shape@512 ∩ ss@64 ∩ renders_cond.
Same structure across all 6 subsets (verified). No R2, no GPU — pure /fsx enumeration.
Output: _v3prep/ready_v3_shas.json {ready: sha->subset, pbr_ok: [...], need_cache: [...]}.
"""
import os, json, collections
TREE = "/fsx/sfr/weikaih/3dgen/data/trellis2"
CACHE = "/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/qwen35-2b_tok1024_crop_mv1"
SUBS = ["ObjaverseXL_github", "ObjaverseXL_sketchfab", "ABO", "HSSD", "Toys4k", "TexVerse"]  # + TexVerse
SHAPE = "shape_latents/shape_enc_next_dc_f16c32_fp16_512"
SS    = "ss_latents/ss_enc_conv3d_16l8_fp16_64"
PBR   = "pbr_latents/tex_enc_next_dc_f16c32_fp16_512"
REND  = "renders_cond"

def shas_npz(d):
    try: return {f[:-4] for f in os.listdir(d) if f.endswith(".npz")}
    except FileNotFoundError: return set()
def shas_dir(d):
    try: return set(os.listdir(d))
    except FileNotFoundError: return set()

print("enumerating per-subset (shape ∩ ss ∩ renders) ...", flush=True)
ready = {}; pbr_ok = set()
for s in SUBS:
    shp = shas_npz(f"{TREE}/{s}/{SHAPE}")
    ss  = shas_npz(f"{TREE}/{s}/{SS}")
    rnd = shas_dir(f"{TREE}/{s}/{REND}")
    pbr = shas_npz(f"{TREE}/{s}/{PBR}")
    inter = shp & ss & rnd
    for sha in inter: ready[sha] = s
    pbr_ok |= (inter & pbr)
    print(f"  {s:24s} shape {len(shp):7d}  ss {len(ss):7d}  rend {len(rnd):7d}  → ready {len(inter):7d}  (pbr {len(inter&pbr):7d})", flush=True)
print(f"TOTAL ready_v3 (ss∩shape∩render): {len(ready)}  | with pbr: {len(pbr_ok)}", flush=True)

print("\nlisting cond cache shas ...", flush=True)
cached = set()
for p in os.listdir(CACHE):
    pd = os.path.join(CACHE, p)
    if len(p) == 2 and os.path.isdir(pd):
        cached |= set(os.listdir(pd))
print(f"cond cache entries: {len(cached)}", flush=True)

ready_set = set(ready); have = ready_set & cached; need = ready_set - cached
print(f"\n=== COND CACHE GAP (for I1 stage-1/2) ===", flush=True)
print(f"  ready_v3 assets:         {len(ready_set)}", flush=True)
print(f"  already have cond cache: {len(have)}", flush=True)
print(f"  NEED cond cache built:   {len(need)}", flush=True)
for k, v in collections.Counter(ready[s] for s in need).most_common(): print(f"     {k:24s} {v}", flush=True)

os.makedirs("/fsx/sfr/weikaih/3dgen/data/_v3prep", exist_ok=True)
with open("/fsx/sfr/weikaih/3dgen/data/_v3prep/ready_v3_shas.json", "w") as f:
    json.dump({"ready": ready, "pbr_ok": sorted(pbr_ok), "need_cache": sorted(need)}, f)
print("\nwrote /fsx/sfr/weikaih/3dgen/data/_v3prep/ready_v3_shas.json", flush=True)
