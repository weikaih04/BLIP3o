"""Build the CORRECT ready_v2 manifest from on-disk recovered latents (non-TexVerse only).
Source of truth = scripts/enumerate_v2.py output (_v2prep/ready_v2_shas.json: the 455,617
assets with shape∩ss∩render). Reuses aesthetic_score + captions from the OLD manifest where
the sha matches (so we don't lose them); new assets get aesthetic=null, captions=[] (fine for
I1 stage-1/2 which need neither; only the cond-cache build + geometry training use this).

Schema matches the training ImageTo3DDataset records.
Output: data/trellis2/manifests/ready_v2/ready_v2_clean.jsonl
"""
import json, os
TREE = "/fsx/sfr/weikaih/3dgen/data/trellis2"
PREP = "/fsx/sfr/weikaih/3dgen/data/_v2prep/ready_v2_shas.json"
OLD  = "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v2/ready_v2.jsonl"
OUT  = "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v2/ready_v2_clean.jsonl"
SHAPE512 = "shape_latents/shape_enc_next_dc_f16c32_fp16_512"
SHAPE1024= "shape_latents/shape_enc_next_dc_f16c32_fp16_1024"
SS64     = "ss_latents/ss_enc_conv3d_16l8_fp16_64"
PBR512   = "pbr_latents/tex_enc_next_dc_f16c32_fp16_512"
PBR1024  = "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024"
REND     = "renders_cond"

print("loading enumerate output ...", flush=True)
prep = json.load(open(PREP))
ready = prep["ready"]                 # sha -> subset
pbr_ok = set(prep["pbr_ok"])          # shas that also have pbr@512

print("joining aesthetic + captions from old manifest ...", flush=True)
aes, caps = {}, {}
for line in open(OLD):
    r = json.loads(line); s = r.get("sha256")
    if not s: continue
    if r.get("aesthetic_score") is not None: aes[s] = r["aesthetic_score"]
    if r.get("captions"): caps[s] = r["captions"]
print(f"  old manifest provides aesthetic for {len(aes)}, captions for {len(caps)}", flush=True)

n = n_aes = n_cap = n_pbr = 0
with open(OUT, "w") as o:
    for sha, sub in ready.items():
        rec = {
            "sha256": sha, "subset": sub,
            "ss_latent_64":      f"{TREE}/{sub}/{SS64}/{sha}.npz",
            "shape_latent_512":  f"{TREE}/{sub}/{SHAPE512}/{sha}.npz",
            "shape_latent_1024": f"{TREE}/{sub}/{SHAPE1024}/{sha}.npz",
            "renders_dir":       f"{TREE}/{sub}/{REND}/{sha}",
            "n_views": 16,
            "captions": caps.get(sha, []),
            "aesthetic_score": aes.get(sha),     # may be null for new assets
        }
        if sha in pbr_ok:
            rec["pbr_latent_512"]  = f"{TREE}/{sub}/{PBR512}/{sha}.npz"
            rec["pbr_latent_1024"] = f"{TREE}/{sub}/{PBR1024}/{sha}.npz"
            n_pbr += 1
        o.write(json.dumps(rec) + "\n")
        n += 1; n_aes += sha in aes; n_cap += sha in caps
print(f"\nwrote {OUT}", flush=True)
print(f"  records: {n}  | with pbr: {n_pbr}  | with aesthetic: {n_aes}  | with captions: {n_cap}", flush=True)
print(f"  (new assets w/o aesthetic: {n - n_aes}  w/o caption: {n - n_cap})", flush=True)
