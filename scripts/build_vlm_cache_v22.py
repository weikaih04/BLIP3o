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

S3 MULTITASK EXTENSION (2026-07-13) — two more modes, same v2.2-encoder discipline:
  --mode captions : T task t-keys. One entry PER caption-list index of the manifest record
      (threed.py T samples caption_key(idx) over rec["captions"] order — build over a
      captioned manifest, e.g. vlm_filtered_capT.jsonl whose captions order is FIXED:
      t000=caption_long, t001=caption_medium, t002=caption_short,
      t003=caption_long+" "+texture_caption (concat; absent for ~0.005% w/o texture cap).
      Prompt = "[3D Gen] " + TXT_PROMPTS[(int(sha[:8],16)+idx) % 3].format(c=cap)
      (the exact gen_txt v2.2 training format), add_generation_prompt=True, no image.
      KEEP-ONLY storage (--keep_only 1, default for this mode): only keep_mask==True
      positions are stored (chat boilerplate dropped; identical cross-attn result —
      masked keys never contribute). Sampling weights live in _meta.json
      (t_sampling_weights; t003 falls back to t000 when absent).
  --mode im4      : IM task pinned 4-view combo (m00, meta im_combo_sizes=[4]).
      Views: --im4_view_sampling picks the scheme.
        fixed    (legacy, v22_im4l): 4 distinct GOOD views (elevation trap): offsets
          {A:0,B:3,C:5} + a 4th sha-picked from {1,2,4,6}, view = 5+(base+off)%7.
        weighted (v22_im4r, 2026-07-24): 4 distinct views sampled WITHOUT replacement
          from ALL 16 renders, p ∝ IM4_VIEW_WEIGHTS (good band 1.0, low 0.6, high 0.5,
          below-ground 0.15), rng seeded per-sha → deterministic / idempotent /
          resumable. Rationale: `fixed` drew all 4 views from the same good band, so
          every training sample saw 4 near-identical viewpoints → ZERO multi-view
          fusion pressure. Weighted sampling covers the full elevation range (incl.
          the degenerate bottom views, rarely) while keeping good angles most likely.
      Both schemes feed the 4 views in ascending view order; the chosen indices are
      stored in the npz as BOTH `views` (legacy name) and `view_indices` (audit).
      Qwen = ONE joint 4-image forward, each view downscaled to --im_tok_per_view
      (default 256 → 512², qwen-light) → hidden ≈ 1.1k tok. DINO = 4 × full 512²
      (1029 tok/view, spatial priority) via the LIVE crop pipeline, inline in the
      m-entry (fp16) with dino_view_ids ordinals — matches threed.py IM fusion path.
"""
import os, sys, json, argparse, time
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image

from trellis2_blip3o import vlm_cache
from trellis2_blip3o.vlm_collate import boiler_ids

V22 = "/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000"
PROMPT = "[3D Gen] <image>\nReconstruct this object in 3D."
IMG = "<|vision_start|><|image_pad|><|vision_end|>"
# gen_txt templates — EXACT copies from vlm3d_stage1/build_messages_jsonl_v22.py
TXT_PROMPTS = ["Generate a 3D asset: {c}", "Create a 3D model of: {c}", "Make this in 3D: {c}"]
IM4_OFF4 = (1, 2, 4, 6)   # 4th good-view offset candidates (distinct mod 7 from {0,3,5})


def pick_im4_views(sha: str):
    """4 distinct good views (005-011): VIEW_SET A/B/C offsets + a sha-picked 4th."""
    base = int(sha[:8], 16)
    offs = [0, 3, 5, IM4_OFF4[int(sha[8:12], 16) % len(IM4_OFF4)]]
    return sorted(5 + (base + o) % 7 for o in offs)


# ── weighted-random view sampling (v22_im4r, 2026-07-24) ──────────────────────
# renders_cond has 16 views ELEVATION-SORTED LOW→HIGH:
#   000-002 below-ground / bottom-view (degenerate)   → w 0.15 (rare, not excluded)
#   003-004 low                                       → w 0.6
#   005-011 good band (eye-level .. 3/4)              → w 1.0
#   012-015 high / top-down                           → w 0.5
# 4 distinct views, sampled WITHOUT replacement with p ∝ w. Covers the whole
# elevation range (diversity/realism at inference) while keeping good angles most
# likely ("提高好视角的概率"). Contrast with pick_im4_views above, which drew all 4
# from 005-011 → 4 near-identical viewpoints → no multi-view fusion pressure.
IM4_N_VIEWS_TOTAL = 16
IM4_VIEW_WEIGHTS = tuple(
    [0.15] * 3 +          # 000,001,002
    [0.6] * 2 +           # 003,004
    [1.0] * 7 +           # 005..011
    [0.5] * 4             # 012..015
)
assert len(IM4_VIEW_WEIGHTS) == IM4_N_VIEWS_TOTAL


def im4_rng(sha: str) -> "np.random.Generator":
    """Per-sha deterministic RNG → the same sha always yields the same 4 views, so
    the build is reproducible and a resumed/re-run worker reproduces prior picks."""
    return np.random.default_rng(int(sha[:16], 16))


def pick_im4_views_weighted(sha: str, n_views: int = IM4_N_VIEWS_TOTAL):
    """4 DISTINCT views from all `n_views` renders, p ∝ IM4_VIEW_WEIGHTS, per-sha seed.
    Returned in ascending view order (same convention as pick_im4_views)."""
    w = np.asarray(IM4_VIEW_WEIGHTS[:n_views], dtype=np.float64)
    k = min(4, n_views)
    idx = im4_rng(sha).choice(n_views, size=k, replace=False, p=w / w.sum())
    return sorted(int(v) for v in idx)


def pick_views_for(sha: str, scheme: str, n_views: int = IM4_N_VIEWS_TOTAL):
    return (pick_im4_views_weighted(sha, n_views) if scheme == "weighted"
            else pick_im4_views(sha))
# per-asset GOOD view (renders_cond is elevation-sorted LOW->HIGH; 000 = below-ground.
# Deterministic per-sha pick from 005-011 (eye-level..3/4). See memory renders-cond-view-ordering-trap.
VIEW_SET = os.environ.get("VIEW_SET", "A")          # A/B/C → 同资产三个互异好视角
def pick_view(sha):
    base = int(sha[:8], 16)
    offs = {"A": 0, "B": 3, "C": 5}[VIEW_SET]
    return f"{5 + (base + offs) % 7:03d}.webp"
FORCE = os.environ.get("FORCE_REEXTRACT", "0") == "1"


def _encode(model, proc, boiler, text, images=None):
    """One v2.2-encoder forward (bs1, no padding) → (hidden (T,2048), keep_mask (T,))."""
    kw = dict(text=[text], return_tensors="pt")
    if images:
        kw["images"] = images
    inputs = proc(**kw).to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[-1][0]                 # (T, 2048) last layer
    ids = inputs["input_ids"][0]
    am = inputs["attention_mask"][0].bool()
    keep = am & ~torch.tensor([int(t) in boiler for t in ids.tolist()], device=ids.device)
    return hidden, keep


def _encode_batch_text(model, proc, boiler, texts):
    """Batched TEXT-ONLY forwards (captions mode). RIGHT padding: real tokens keep
    positions 0..L-1 (identical RoPE to unpadded) and — causal model — trailing pads
    can never influence earlier positions, so per-row [:L] hiddens match bs1 up to
    bf16 batched-reduction noise. (LEFT padding would SHIFT positions in a plain
    forward → wrong hiddens.) Returns list of (hidden (L,H), keep (L,))."""
    tok = proc.tokenizer
    tok.padding_side = "right"
    inputs = proc(text=texts, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[-1]                        # (B, T, H)
    res = []
    for i in range(len(texts)):
        am = inputs["attention_mask"][i].bool()
        L = int(am.sum())
        ids = inputs["input_ids"][i, :L]
        keep = ~torch.tensor([int(t) in boiler for t in ids.tolist()], device=ids.device)
        res.append((hs[i, :L], keep))
    return res


def _free_tb(path: str) -> float:
    """Free TB actually usable at `path` = min(filesystem avail, lustre quota headroom).
    A 1.85 TB cache on a shared /fsx can hit EITHER wall, and Lustre reports ENOSPC
    only after the fact — so we check both and stop the build ourselves."""
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize / 1e12
    try:                                                     # quota headroom (TB)
        import re, subprocess
        out = subprocess.run(["lfs", "quota", "-u", os.environ.get("USER", ""), path],
                             capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"^\s*\S*%s\S*\s+(\d+)\s+(\d+)\s+(\d+)" % re.escape(path.split("/")[1]),
                      out, re.M)
        if m:
            used_kb, _soft_kb, hard_kb = (int(m.group(i)) for i in (1, 2, 3))
            if hard_kb > 0:
                free = min(free, (hard_kb - used_kb) * 1024 / 1e12)
    except Exception:
        pass
    return free


def _check_space(out_root: str, min_free_tb: float, shard: int) -> None:
    if min_free_tb <= 0:
        return
    f = _free_tb(out_root)
    if f < min_free_tb:
        print(f"\n{'!' * 78}\n[w{shard}] FATAL: OUT OF DISK — only {f:.2f} TB free at "
              f"{out_root} (< --min_free_tb {min_free_tb}). ABORTING so the cache is not "
              f"silently truncated. Free space or lower the threshold, then re-run "
              f"(the build is idempotent and resumes).\n{'!' * 78}\n", flush=True)
        sys.stdout.flush()
        os._exit(17)


def _open_view(renders_dir: str, view: int):
    for ext in ("webp", "png", "jpg"):
        p = os.path.join(renders_dir, f"{view:03d}.{ext}")
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--vlm", default=V22)
    ap.add_argument("--vlm_name", default=None,
                    help="value recorded as _meta.json 'vlm' — pass the CANONICAL /fsx "
                         "checkpoint path when --vlm points at a node-local stage copy "
                         "(/dev/shm/...), so the cache contract stays meaningful")
    ap.add_argument("--mode", default="views", choices=["views", "captions", "im4"])
    ap.add_argument("--im_tok_per_view", type=int, default=256,
                    help="im4: qwen vision tokens per view (256 → 512² input; qwen-light)")
    ap.add_argument("--dino_image_size", type=int, default=512,
                    help="im4: per-view DINOv3 input size (512 → 1029 tok/view, full)")
    ap.add_argument("--keep_only", type=int, default=None,
                    help="store ONLY keep_mask==True positions (compact; default 1 for "
                         "captions mode, 0 otherwise)")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="captions mode: captions per batched text forward (right-pad)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--im4_view_sampling", default="fixed", choices=["fixed", "weighted"],
                    help="im4 view pick: 'fixed' = legacy good-band offsets (v22_im4l); "
                         "'weighted' = 4 distinct of all 16, p ∝ IM4_VIEW_WEIGHTS, "
                         "per-sha seeded (v22_im4r)")
    ap.add_argument("--min_free_tb", type=float, default=2.0,
                    help="abort LOUDLY when free space (min of fs-avail and lustre "
                         "quota headroom) drops below this many TB; 0 disables")
    ap.add_argument("--max_assets", type=int, default=0,
                    help="debug: stop after this many assets of this shard (0 = all)")
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
    keep_only = bool(a.keep_only) if a.keep_only is not None else (a.mode == "captions")
    vlm_name = a.vlm_name or a.vlm            # what goes into _meta.json's contract
    text = proc.apply_chat_template(
        [{"role": "user", "content": PROMPT.replace("<image>", IMG)}],
        tokenize=False, add_generation_prompt=True)

    dino_ext = ds = None
    if a.mode == "im4":
        from trellis2_blip3o.vlm_collate import cap_image, px_per_tok
        from trellis2_blip3o.dino_align import DinoV3FeatureExtractor, TRELLIS_DINOV3_NAME
        from trellis2_blip3o.data.tasks.threed import ImageTo3DDataset
        # live crop pipeline for the DINO side (same as build_dino_cache.py / prod d-entries)
        ds = ImageTo3DDataset(manifest=a.manifests[0], ss_only=True, crop_to_object=True)
        dino_ext = DinoV3FeatureExtractor(TRELLIS_DINOV3_NAME, image_size=a.dino_image_size)
        dino_ext.model.eval().to("cuda")
        for p_ in dino_ext.model.parameters():
            p_.requires_grad_(False)
        text_m = proc.apply_chat_template(
            [{"role": "user",
              "content": "[3D Gen] " + IMG * 4 + "\nReconstruct this object in 3D."}],
            tokenize=False, add_generation_prompt=True)
        qwen_max_px = a.im_tok_per_view * px_per_tok(proc)

    # NEVER mix view-sampling schemes inside one cache root: entries are keyed m00
    # regardless of scheme, so a scheme flip on a resume would silently produce a
    # half-fixed / half-weighted cache that nothing downstream can tell apart.
    if a.mode == "im4":
        _mp = os.path.join(a.out_root, "_meta.json")
        if os.path.isfile(_mp):
            _prev = json.load(open(_mp)).get("im_view_sampling")
            if _prev is not None and _prev != a.im4_view_sampling:
                raise SystemExit(
                    f"[w{a.shard}] REFUSING: {a.out_root} was built with "
                    f"im_view_sampling={_prev!r} but this run asks for "
                    f"{a.im4_view_sampling!r}. Use a different --out_root.")

    if a.shard == 0:
        os.makedirs(a.out_root, exist_ok=True)
        if a.mode == "views":
            vlm_cache.write_meta(a.out_root, vlm=vlm_name, target_tokens_per_view=1024,
                                 crop_to_object=False, max_views=1)
        else:
            # captions/im4 keep any existing contract fields and add their own
            mp = os.path.join(a.out_root, "_meta.json")
            meta = json.load(open(mp)) if os.path.isfile(mp) else {
                "schema": 1, "vlm": vlm_name, "target_tokens_per_view": 1024,
                "crop_to_object": False, "max_views": 1}
            if a.mode == "captions":
                meta.update({"has_captions": True, "txt_prompts": TXT_PROMPTS,
                             "txt_prompt_pick": "(int(sha[:8],16)+idx) % 3",
                             "keep_only_storage": keep_only,
                             "caption_order": ["caption_long", "caption_medium",
                                               "caption_short",
                                               "caption_long+texture_caption"],
                             # user-decided training-time sampling over t-keys
                             # (t003 falls back to t000 when the asset has no texture cap)
                             "t_sampling_weights": {"t000": 0.35, "t001": 0.20,
                                                    "t002": 0.10, "t003": 0.35},
                             "t_sampling_fallback": {"t003": "t000"}})
            else:  # im4
                from trellis2_blip3o.dino_align import TRELLIS_DINOV3_NAME as _DN
                meta.update({"im_combo_sizes": [4],
                             "im_qwen_tok_per_view": int(a.im_tok_per_view),
                             "im_view_sampling": a.im4_view_sampling,
                             "im_view_weights": (list(IM4_VIEW_WEIGHTS)
                                                 if a.im4_view_sampling == "weighted" else None),
                             "im_view_seed": ("np.random.default_rng(int(sha[:16],16))"
                                              if a.im4_view_sampling == "weighted" else None),
                             "im_view_offsets": ([0, 3, 5] + list(IM4_OFF4)
                                                 if a.im4_view_sampling == "fixed" else None),
                             "dino_model": _DN,
                             "dino_image_size": int(a.dino_image_size),
                             "dino_max_views": 4, "dino_crop_to_object": True,
                             "max_views": 4})
            with open(mp, "w") as f:
                json.dump(meta, f, indent=2)

    # record list, round-robin by shard
    recs = []
    for mf in a.manifests:
        for line in open(mf):
            recs.append(json.loads(line))
    mine = [recs[i] for i in range(a.shard, len(recs), a.num_shards)]
    if a.reverse:
        mine.reverse()
    print(f"[w{a.shard}] mode={a.mode} {len(mine)} assets  vlm={os.path.basename(a.vlm)}"
          f"{' sampling=' + a.im4_view_sampling if a.mode == 'im4' else ''}"
          f"{' REVERSE' if a.reverse else ''}", flush=True)
    if a.mode == "im4":
        _check_space(a.out_root, a.min_free_tb, a.shard)

    done = skip = err = 0
    t0 = time.time()

    if a.mode == "captions":
        # ── batched text path: gather pending (sha, key, text), forward in batches ──
        def cap_text(sha, ci, cap):
            tmpl = TXT_PROMPTS[(int(sha[:8], 16) + ci) % len(TXT_PROMPTS)]
            return proc.apply_chat_template(
                [{"role": "user", "content": "[3D Gen] " + tmpl.format(c=cap)}],
                tokenize=False, add_generation_prompt=True)

        # self-check: batched(right-pad) vs bs1 on this shard's first asset
        for rec in mine:
            caps0 = [c for c in (rec.get("captions") or []) if c]
            if caps0:
                sha0 = rec["sha256"]
                txts = [cap_text(sha0, ci, c) for ci, c in enumerate(caps0)]
                # 3-way diagnostic: (a) bs1 determinism baseline, (b) SAME-LENGTH batch
                # (4× identical text → zero padding), (c) padded batch of the 4 variants.
                # Padding is INNOCENT iff rel(c) ≈ rel(b); rel(b) itself is generic
                # batched-GEMM bf16 reduction noise (unavoidable when batching).
                def _rel(hx, hy):
                    fx, fy = hx.float(), hy.float()
                    return float((fx - fy).abs().max()) / max(float(fx.abs().max()), 1e-8)
                h1a, _ = _encode(model, proc, boiler, txts[0])
                h1b, _ = _encode(model, proc, boiler, txts[0])
                rel_a = _rel(h1a, h1b)
                hsame = _encode_batch_text(model, proc, boiler, [txts[0]] * 4)
                rel_b = _rel(h1a, hsame[0][0])
                bres = _encode_batch_text(model, proc, boiler, txts)
                rel_c = 0.0
                for ci, t in enumerate(txts):
                    h1, k1 = _encode(model, proc, boiler, t)
                    h2, k2 = bres[ci]
                    assert h1.shape == h2.shape and bool((k1 == k2).all()), "self-check shape/mask"
                    rel_c = max(rel_c, _rel(h1, h2))
                print(f"[w{a.shard}] SELF-CHECK rel: bs1-vs-bs1={rel_a:.5f} "
                      f"samelen-batch={rel_b:.5f} padded-batch={rel_c:.5f}", flush=True)
                # benign iff EITHER small in absolute relative terms OR no worse than
                # the same-length batching noise (padding-independent). Measured fleet-
                # wide 2026-07-13: a=0.0 everywhere, b≈0.009-0.015, c≈0.012-0.025.
                assert rel_c < 0.035 or rel_c < max(2 * rel_b, 0.005) + 0.01, \
                    f"padding-specific divergence: a={rel_a} b={rel_b} c={rel_c}"
                break

        pend = []   # (sha, key, text)
        def flush():
            nonlocal done, err
            if not pend:
                return
            try:
                res = _encode_batch_text(model, proc, boiler, [t for _, _, t in pend])
                for (sha_, key_, _), (hidden, keep) in zip(pend, res):
                    if keep_only:
                        hidden, keep = hidden[keep], keep[keep]
                    vlm_cache.save_entry(a.out_root, sha_, key_, hidden=hidden, keep_mask=keep)
                    done += 1
            except Exception as e:
                err += len(pend)
                print(f"[w{a.shard}] BATCH ERR {type(e).__name__}: {str(e)[:120]}", flush=True)
            pend.clear()
            if done % (a.batch_size * 16) < a.batch_size:
                print(f"[w{a.shard}] {done} done ({skip} skip, {err} err) "
                      f"{done/max(1e-9, time.time()-t0):.1f}/s", flush=True)

        for rec in mine:
            sha = rec["sha256"]
            for ci, cap in enumerate([c for c in (rec.get("captions") or []) if c]):
                key = vlm_cache.caption_key(ci)
                if not FORCE and os.path.exists(vlm_cache.entry_path(a.out_root, sha, key)):
                    skip += 1
                    continue
                pend.append((sha, key, cap_text(sha, ci, cap)))
                if len(pend) >= a.batch_size:
                    flush()
        flush()
        print(f"[w{a.shard}] DONE {done} new, {skip} skip, {err} err, "
              f"{time.time()-t0:.0f}s", flush=True)
        return

    for rec in mine:
        sha = rec["sha256"]
        try:

            if a.mode == "im4":
                key = vlm_cache.combo_key(0)          # m00 = THE pinned 4-view combo
                # idempotent resume: skip only entries that exist AND are plausibly
                # complete. A real m00 is ~4.5 MB (qwen ~292×2048 + dino ~1620×1024,
                # fp16); anything under 1 MB is a truncated/aborted write → rebuild.
                # (save_entry is tmp+os.replace atomic, so this is belt-and-braces.)
                _p = vlm_cache.entry_path(a.out_root, sha, key)
                if not FORCE and os.path.exists(_p) and os.path.getsize(_p) >= (1 << 20):
                    skip += 1
                    continue
                nv = int(rec.get("n_views") or IM4_N_VIEWS_TOTAL)
                views = pick_views_for(sha, a.im4_view_sampling,
                                       min(nv, IM4_N_VIEWS_TOTAL))
                assert len(set(views)) == len(views), f"duplicate views {views} for {sha}"
                imgs_q = [_open_view(rec["renders_dir"], v) for v in views]
                if any(im is None for im in imgs_q):
                    err += 1
                    continue
                if done < 3:
                    print(f"[w{a.shard}] sha={sha[:12]} sampling={a.im4_view_sampling} "
                          f"views={views}", flush=True)
                imgs_q = [cap_image(im, qwen_max_px) for im in imgs_q]   # qwen-light
                hidden, keep = _encode(model, proc, boiler, text_m, images=imgs_q)
                if keep_only:
                    hidden, keep = hidden[keep], keep[keep]
                dimgs = ds._load_views(rec["renders_dir"], views)        # crop pipeline
                if len(dimgs) != 4:
                    err += 1
                    continue
                with torch.no_grad():
                    dfeats = dino_ext(dimgs)                             # (4, N_d, 1024)
                nd = dfeats.shape[1]
                dino_h = dfeats.reshape(-1, dfeats.shape[-1]).to(torch.float16).cpu()
                dino_ids = torch.arange(4, dtype=torch.long).repeat_interleave(nd)
                vlm_cache.save_entry(
                    a.out_root, sha, key, hidden=hidden, keep_mask=keep,
                    views=np.asarray(views, dtype=np.int64),          # legacy name
                    view_indices=np.asarray(views, dtype=np.int64),   # audit (same data)
                    dino_hidden=dino_h,
                    dino_keep_mask=torch.ones(dino_h.shape[0], dtype=torch.bool),
                    dino_view_ids=dino_ids)
                done += 1
                if done % 200 == 0:
                    print(f"[w{a.shard}] {done} done ({skip} skip, {err} err) "
                          f"{done/max(1e-9, time.time()-t0):.1f}/s", flush=True)
                if done % 500 == 0:
                    _check_space(a.out_root, a.min_free_tb, a.shard)
                if a.max_assets and done >= a.max_assets:
                    break
                continue

            # --- mode == "views" (original single-view behavior, unchanged) ---
            img_path = os.path.join(rec["renders_dir"], pick_view(sha))
            if not FORCE and os.path.exists(vlm_cache.entry_path(a.out_root, sha, "v000")):
                skip += 1
                continue
            if not os.path.exists(img_path):
                cand = sorted(f for f in os.listdir(os.path.dirname(img_path)) if f.endswith(".webp"))
                if not cand:
                    err += 1
                    continue
                img_path = os.path.join(os.path.dirname(img_path), cand[0])
            hidden, keep = _encode(model, proc, boiler, text,
                                   images=[Image.open(img_path).convert("RGB")])
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
