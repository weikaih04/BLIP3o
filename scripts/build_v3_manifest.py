"""Build ready_v3 manifest (= ready_v2 5 non-TexVerse + reprocessed TexVerse) from on-disk latents.
Source = enumerate_v3.py output (_v3prep/ready_v3_shas.json). Reuses aesthetic + captions from the
OLD ready_v2 manifest where sha matches (non-TexVerse); TexVerse + new assets get aesthetic=null,
captions=[] — fine for I1 stage-1/2 (needs neither; captions re-joined later for S3).
Caveat handled: TexVerse pbr_latents has only @512 (no @1024) → pbr_latent_1024 only emitted for
non-TexVerse subsets. Schema matches the training ImageTo3DDataset records.
Output: data/trellis2/manifests/ready_v3/ready_v3_clean.jsonl
"""
import json, os
TREE = "/fsx/sfr/weikaih/3dgen/data/trellis2"
PREP = "/fsx/sfr/weikaih/3dgen/data/_v3prep/ready_v3_shas.json"
OLD  = "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v2/ready_v2.jsonl"
OUTD = "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v3"
OUT  = f"{OUTD}/ready_v3_clean.jsonl"
SHAPE512 = "shape_latents/shape_enc_next_dc_f16c32_fp16_512"
SHAPE1024= "shape_latents/shape_enc_next_dc_f16c32_fp16_1024"
SS64     = "ss_latents/ss_enc_conv3d_16l8_fp16_64"
PBR512   = "pbr_latents/tex_enc_next_dc_f16c32_fp16_512"
PBR1024  = "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024"
REND     = "renders_cond"

print("loading enumerate output ...", flush=True)
prep = json.load(open(PREP)); ready = prep["ready"]; pbr_ok = set(prep["pbr_ok"])

print("joining aesthetic + captions from old ready_v2 manifest (non-TexVerse) ...", flush=True)
aes, caps = {}, {}
for line in open(OLD):
    r = json.loads(line); s = r.get("sha256")
    if not s: continue
    if r.get("aesthetic_score") is not None: aes[s] = r["aesthetic_score"]
    if r.get("captions"): caps[s] = r["captions"]
print(f"  old manifest provides aesthetic for {len(aes)}, captions for {len(caps)}", flush=True)

os.makedirs(OUTD, exist_ok=True)
n = n_pbr = n_tv = 0
with open(OUT, "w") as o:
    for sha, sub in ready.items():
        is_tv = (sub == "TexVerse"); n_tv += is_tv
        rec = {
            "sha256": sha, "subset": sub,
            "ss_latent_64":      f"{TREE}/{sub}/{SS64}/{sha}.npz",
            "shape_latent_512":  f"{TREE}/{sub}/{SHAPE512}/{sha}.npz",
            "shape_latent_1024": f"{TREE}/{sub}/{SHAPE1024}/{sha}.npz",
            "renders_dir":       f"{TREE}/{sub}/{REND}/{sha}",
            "n_views": 16,
            "captions": caps.get(sha, []),
            "aesthetic_score": aes.get(sha),
        }
        if sha in pbr_ok:
            rec["pbr_latent_512"] = f"{TREE}/{sub}/{PBR512}/{sha}.npz"
            if not is_tv:   # TexVerse has no pbr@1024
                rec["pbr_latent_1024"] = f"{TREE}/{sub}/{PBR1024}/{sha}.npz"
            n_pbr += 1
        o.write(json.dumps(rec) + "\n"); n += 1
print(f"\nwrote {OUT}", flush=True)
print(f"  records: {n}  (TexVerse {n_tv} / non-TV {n - n_tv})  | with pbr: {n_pbr}", flush=True)
