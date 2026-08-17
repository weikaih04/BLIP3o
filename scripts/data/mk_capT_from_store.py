#!/usr/bin/env python
"""Rebuild the text-to-3D manifests from the STANDALONE caption store, so the
text task stops depending on the ready_v4 layer.

THE BUG THIS FIXES. `manifests/splits/capT_train.jsonl` descends from
`ready_v4_vlm_filtered/vlm_filtered_capT.jsonl`, which was produced on
2026-07-13 by joining the caption store onto the **v4** pool. Nothing about the
captions is v4-specific — but because the join was materialised into a v4
manifest, the text task inherited v4's ceiling: it can never cover the assets
v5 added (SketchfabV1, SWH), and it dies if that one file goes away. Measured
against that manifest, caption coverage looked like 400k 99.9% / 800k 49.1% /
1.8M 22.9%.

Measured against the STORE itself it is 100% / 100% / 100% (1,741,113 assets
covered, more than the 1.8M pool holds). So there was never a caption shortage —
only a stale join.

STORE (survived the /fsx/sfr/weikaih -> /fsx/home/weikai.huang migration):
    /fsx/home/weikai.huang/hyperpod/weikaih_cap/FINAL_*_captions.json
    format {"n":..,"results":[{sha, subset, raw}, ..]}
    raw is a MARKDOWN-FENCED json string (```json ... ```), not json — parsing
    it directly raises JSONDecodeError at char 0.
    holistic raw: object / caption_long / caption_medium / caption_short / tags
    texture  raw: object / appearance / global_material / parts / lighting_note
                  / texture_caption / confidence

OUTPUT keeps the historical 4-element ladder so nothing downstream changes:
    captions = [long, medium, short, long + " " + texture_caption]
(`scripts/data/build_capT_manifest.py` built exactly this from v4.)

Usage (compute node, needs ~120 GB RAM to hold both stores):
  srun --jobid=<hold> --overlap --mem=200G python scripts/mk_capT_from_store.py
"""
import glob
import json
import os
import sys

STORE = "/fsx/home/weikai.huang/hyperpod/weikaih_cap"
SPLITS = "manifests/splits"
# (input pool, output name) — the text task now reads the SAME pools as the
# image tasks, which is the whole point: one asset set, three modalities.
JOBS = [("pool400k_train", "capT400k_train"),
        ("pool800k_train", "capT800k_train"),
        ("val", "capT_val_v5"),
        ("test", "capT_test_v5")]


def unfence(s: str):
    """raw is ```json\n{...}\n``` — strip the fence, then parse."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    try:
        return json.loads(s)
    except Exception:
        return None


def load(patterns, keys):
    """sha -> tuple(keys). Counts unparseable records instead of dropping them
    silently — a caption store that half-fails would otherwise look like a
    coverage problem in the pools."""
    out, bad, n = {}, 0, 0
    seen_files = []
    for pat in patterns:
        for p in sorted(glob.glob(f"{STORE}/{pat}")):
            if p in seen_files:
                continue
            seen_files.append(p)
            d = json.load(open(p))
            for r in d.get("results", []):
                n += 1
                raw = unfence(r.get("raw") or "")
                if not raw:
                    bad += 1
                    continue
                out.setdefault(r["sha"], tuple(raw.get(k) or "" for k in keys))
            print(f"  loaded {os.path.basename(p):46} running unique={len(out)}",
                  flush=True)
    print(f"  -> {len(out)} unique sha, {bad}/{n} records unparseable", flush=True)
    return out


def main():
    print("[capT] holistic store", flush=True)
    hol = load(["FINAL_holistic_captions.json", "FINAL_tv_holistic_captions.json",
                "FINAL_sfv1_holistic_captions.json",
                "FINAL_backfill_holistic_captions.json"],
               ("caption_long", "caption_medium", "caption_short"))
    print("[capT] texture store", flush=True)
    tex = load(["FINAL_texture_captions.json", "FINAL_tv_texture_captions.json",
                "FINAL_sfv1_texture_captions.json",
                "FINAL_matbackfill_texture_captions.json"],
               ("texture_caption",))

    for src, dst in JOBS:
        sp, dp = f"{SPLITS}/{src}.jsonl", f"{SPLITS}/{dst}.jsonl"
        if not os.path.exists(sp):
            print(f"[capT] SKIP {src}: not found", flush=True)
            continue
        n = kept = 0
        with open(sp) as fi, open(dp, "w") as fo:
            for line in fi:
                n += 1
                r = json.loads(line)
                h = hol.get(r["sha256"])
                if not h:
                    continue          # no caption -> the row cannot serve the T task
                lo, me, sh = h
                t = (tex.get(r["sha256"]) or ("",))[0]
                r["captions"] = [lo, me, sh, (lo + " " + t).strip() if t else lo]
                kept += 1
                fo.write(json.dumps(r) + "\n")
        print(f"[capT] {dst}: {kept}/{n} rows  ({kept / max(n, 1) * 100:.1f}% covered)"
              f" -> {dp}", flush=True)


if __name__ == "__main__":
    main()
