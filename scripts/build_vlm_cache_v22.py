"""V-phase cond cache for Stage-2, using the v2.2 3D-VLM as an ENCODER (image+text →
last-hidden), NOT a code generator. Replaces build_vlm_cache.py's V phase for v2.2.

Per asset (single view 000): feed the EXACT v2.2 training prompt
    [3D Gen] <image>\nReconstruct this object in 3D.
with add_generation_prompt=True (NO assistant codes) through v2.2, one forward, save the
last-layer hidden + keep_mask (chat-boilerplate filtered). Image is the raw render webp,
UN-cropped, matching v2.2 training (extract_taps_v22 did the same).

D phase (build_dino_cache.py) + M phase (merge_vd_cache.py) are REUSED unchanged — DINO is
VLM-independent.

Output (vlm_cache format): {out_root}/{sha[:2]}/{sha}/v000.npz  (hidden fp16, keep_mask bool)
Shardable: --shard i --num_shards N (one GPU each). Idempotent (skips existing).
"""
import os, sys, json, argparse, time
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/fsx/sfr/weikaih/3dgen/model/BLIP3o")
import numpy as np
import torch
from PIL import Image

from trellis2_blip3o import vlm_cache
from trellis2_blip3o.vlm_collate import boiler_ids

V22 = "/fsx/sfr/weikaih/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000"
PROMPT = "[3D Gen] <image>\nReconstruct this object in 3D."
IMG = "<|vision_start|><|image_pad|><|vision_end|>"
# per-asset GOOD view (renders_cond is elevation-sorted LOW->HIGH; 000 = below-ground.
# Deterministic per-sha pick from 005-011 (eye-level..3/4). See memory renders-cond-view-ordering-trap.
VIEW_SET = os.environ.get("VIEW_SET", "A")          # A/B/C → 同资产三个互异好视角
def pick_view(sha):
    base = int(sha[:8], 16)
    offs = {"A": 0, "B": 3, "C": 5}[VIEW_SET]
    return f"{5 + (base + offs) % 7:03d}.webp"
FORCE = os.environ.get("FORCE_REEXTRACT", "0") == "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--vlm", default=V22)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--reverse", action="store_true",
                    help="iterate the record list back-to-front (extra idempotent workers "
                         "start from the tail so they converge with forward workers)")
    a = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.vlm)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        a.vlm, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    boiler = boiler_ids(tok, include_system=False)
    text = proc.apply_chat_template(
        [{"role": "user", "content": PROMPT.replace("<image>", IMG)}],
        tokenize=False, add_generation_prompt=True)

    if a.shard == 0:
        os.makedirs(a.out_root, exist_ok=True)
        vlm_cache.write_meta(a.out_root, vlm=a.vlm, target_tokens_per_view=1024,
                             crop_to_object=False, max_views=1)

    # record list (sha, render view path), round-robin by shard
    recs = []
    for mf in a.manifests:
        for line in open(mf):
            r = json.loads(line)
            recs.append((r["sha256"], os.path.join(r["renders_dir"], pick_view(r["sha256"]))))
    mine = [recs[i] for i in range(a.shard, len(recs), a.num_shards)]
    if a.reverse:
        mine.reverse()
    print(f"[w{a.shard}] {len(mine)} assets  vlm={os.path.basename(a.vlm)}"
          f"{' REVERSE' if a.reverse else ''}", flush=True)

    done = skip = err = 0
    t0 = time.time()
    for sha, img_path in mine:
        try:
            if not FORCE and os.path.exists(vlm_cache.entry_path(a.out_root, sha, "v000")):
                skip += 1
                continue
            if not os.path.exists(img_path):
                cand = sorted(f for f in os.listdir(os.path.dirname(img_path)) if f.endswith(".webp"))
                if not cand:
                    err += 1
                    continue
                img_path = os.path.join(os.path.dirname(img_path), cand[0])
            inputs = proc(text=[text], images=[Image.open(img_path).convert("RGB")],
                          return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1][0]                 # (T, 2048) last layer
            ids = inputs["input_ids"][0]
            am = inputs["attention_mask"][0].bool()
            keep = am & ~torch.tensor([int(t) in boiler for t in ids.tolist()],
                                      device=ids.device)
            vlm_cache.save_entry(a.out_root, sha, "v000", hidden=hidden, keep_mask=keep,
                                 view=np.array(int(os.path.basename(img_path)[:3])))
            done += 1
            if done % 200 == 0:
                r = done / (time.time() - t0)
                print(f"[w{a.shard}] {done} done ({skip} skip, {err} err) {r:.1f}/s", flush=True)
        except Exception as e:
            err += 1
            if err <= 5 or err % 200 == 0:
                print(f"[w{a.shard}] ERR#{err} {type(e).__name__}: {str(e)[:100]}", flush=True)
    print(f"[w{a.shard}] DONE {done} new, {skip} skip, {err} err, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
