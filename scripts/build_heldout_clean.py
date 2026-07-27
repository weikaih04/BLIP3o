"""Build a CONTENT-DEDUPLICATED held-out evaluation set for WilD3DGen.

WHY (2026-07-27): the incumbent held-out set (im_probe/heldout14*.jsonl, n=14) is
(a) contaminated — `edade2fc304d4d76bde7f42f820890b1` (SketchfabV1 cow) reaches SS-IoU
    1.000 from a 57-token text-only cond, i.e. a CONTENT duplicate of it sits in the
    training corpus under a DIFFERENT sha (sha-level exclusion cannot see it), and
(b) too small — the 4-distinct-vs-4-copy statistic has sd ≈ ±0.023 at n=14, the same
    size as the effect (~+0.02), so conclusions flip sign on noise alone.

WHAT THIS BUILDS
  A held-out manifest of >=100 assets that are
    * absent from the training manifest by sha  AND by CONTENT (two independent axes),
    * complete in all three conditioning modalities (I1 / IM / T),
    * verified on-disk (the /fsx/sfr/weikaih -> /fsx/home/weikai.huang migration left
      dead paths in several manifests; every path here is stat()'d, never trusted).

DEDUP CRITERION — three scores against the FULL 422k training corpus; an asset is dropped
if ANY of them fires.  renders_cond uses a FIXED 16-camera rig shared by every asset
(transforms.json is byte-identical across assets), so view k of a duplicate mesh is the
same camera as view k of the original and everything can be compared VIEW-ALIGNED.
  Per asset, for views VIEWS=(2,6,9,13): alpha-crop to the object bbox (square, 4% pad),
  resize to 24x24, keep RGB-on-white and alpha RAW -> a 4*(1728+576) = 9216-D descriptor.
  1. RENDER (appearance)  full RGB+alpha, mean-centred on the training corpus, cosine
     >= --render_cos.  Catches a re-export that kept its textures.
  2. SIL (silhouette)     the alpha planes only, same treatment, cosine >= --sil_cos.
     Survives recolouring/retexturing, which axis 1 does not.
  3. GEOM (voxels)        ss_latent_64 z (8,16,16,16 f32) — the exact tensor the SS flow
     is trained to produce.  Flatten (32768), L2-normalise, project with a fixed seed-0
     Gaussian to 512 dims (JL inner-product error sd ~1/sqrt(512)=0.044); cosine >=
     --geom_cos_screen shortlists, then the pair is re-scored EXACTLY from both raw z
     tensors and >= --geom_cos marks a duplicate.  This is the SHARPEST axis: real
     duplicates score 1.000 and everything else sits below 0.9, so the threshold is not
     delicate.  It catches re-exports whose renders differ completely (recoloured,
     relit, different exporter) — the case a render hash cannot see.

TWO THINGS MEASURED THE HARD WAY, recorded so they are not re-learned:
  * A plain dHash/pHash of the UNCROPPED render is useless: the frames are ~90% white
    background, so unrelated assets land within 0-4 bits of each other (min pairwise dHash
    distance across the incumbent held-out 14 = 0 bits, between a cow and a sports car).
  * The first cropped descriptor normalised the gray and alpha blocks separately, which
    threw away colour and absolute brightness.  At cos>=0.90 it removed 266/1452
    candidates and eyeballing showed they were SAME-CATEGORY, not same-asset (a red
    sphere matched a black sphere at 0.986; every sword matched every other sword).
    Hence: raw RGBA, centre once globally, and a much higher threshold.
  KNOWN LIMITATION: all three axes are orientation-sensitive.  A duplicate re-exported
  with a different up-axis or a global rotation is NOT caught.  Nothing cheap covers that;
  it is recorded here rather than papered over.

TIERS (every manifest record carries `tier`, so any statistic can be reported A-only)
  A  the EXACT training gate: judge validity=='single_object' AND recommended=='yes'.
     Verified: vlm_filtered_all.jsonl == precisely the 419,999 gate-passing assets.  The
     training corpus consumed 419,999 of the 425,526 gate-passers, so tier A is what is
     LEFT: only ~240 assets with complete artefacts + a caption.  That scarcity is the
     honest ceiling on a distribution-matched held-out set, not a choice.
  B  single_object AND aesthetic>=5 AND structural>=6 but recommended=='no' — good assets
     the judge declined.  Sampled to the TRAINING subset mix to top tier A up.  Their
     aesthetic is ~5.0 vs 5.7 in training (nothing above 5 is left outside training), so
     tier B is a REAL quality skew and must be reported as such.

STAGES (each is idempotent and writes into --work)
  pool      training shas (all U capT), judge scores, captions, on-disk enumeration,
            tier A/B selection, sentinel injection
  feats     render + geometry features for a sha list (multiprocessing)   [the slow one]
  dedup     candidate x training nearest-neighbour search on both axes -> survivors
  manifest  sentinel audit, then emit the final jsonl (+ capT jsonl), paths stat()'d

USAGE
  python scripts/build_heldout_clean.py --stage pool
  python scripts/build_heldout_clean.py --stage feats --which train   # ~422k, ~26 min
  python scripts/build_heldout_clean.py --stage feats --which cand
  python scripts/build_heldout_clean.py --stage dedup
  python scripts/build_heldout_clean.py --stage manifest
"""
import argparse
import json
import os
import re
import time

# MUST precede numpy: every worker does a (1,32768)@(32768,512) GEMM, and an unpinned
# OpenBLAS spawns one thread per core PER WORKER (measured: 40 workers -> load avg 1000+
# on a 96-core box, which is antisocial on a shared node).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# ── fixed layout ──────────────────────────────────────────────────────────────
DATA = "/fsx/home/weikai.huang/3dgen/data/trellis2"
MDIR = f"{DATA}/manifests/ready_v4_vlm_filtered"
# configs/mix_s3_multitask_xgenmm_ss.yaml: image_to_3d + multi_image_to_3d read
# vlm_filtered_all.jsonl, text_to_3d_weighted reads vlm_filtered_capT.jsonl.  The
# contamination reference is therefore the UNION (422,281 shas), not `all` alone —
# using `all` alone would have admitted the 2,282 capT-only assets, every one of which
# the T task trains on.
TRAIN_MANIFESTS = [f"{MDIR}/vlm_filtered_all.jsonl", f"{MDIR}/vlm_filtered_capT.jsonl"]
TRAIN_MANIFEST = TRAIN_MANIFESTS[0]
# built 2026-07-24, unreferenced by any training script today, but it is plainly the next
# training pool — candidates in it are FLAGGED (and excluded by default) so this held-out
# set does not rot the moment ready_v5 is adopted.
FUTURE_MANIFEST = f"{DATA}/manifests/ready_v5_vlm_filtered/vlm_filtered_all.jsonl"
CAPDIR = "/fsx/home/weikai.huang/hyperpod/weikaih_cap"
JUDGE = f"{CAPDIR}/judge_scores.jsonl"     # 1,039,752 VLM quality judgements
OUTDIR = "/fsx/home/weikai.huang/3dgen/im_probe/heldout_clean"
SUBSETS = ["ABO", "HSSD", "ObjaverseXL_github", "ObjaverseXL_sketchfab",
           "SketchfabV1", "SWH", "TexVerse", "Toys4k"]
SS_ENC = "ss_latents/ss_enc_conv3d_16l8_fp16_64"
SHAPE_ENC = "shape_latents/shape_enc_next_dc_f16c32_fp16_512"
PBR_ENC = "pbr_latents/tex_enc_next_dc_f16c32_fp16_512"
VIEWS = (2, 6, 9, 13)          # fixed rig -> comparable across assets
N_VIEWS_REQ = 16
PROJ_DIM = 512
THUMB = 24                     # per-view crop thumbnail edge (rgb + alpha)
VSTRIDE = THUMB * THUMB * 4    # per-view descriptor length: 3 colour planes + alpha
KNOWN_CONTAMINATED = {           # must be caught by the criterion or the criterion is wrong
    "edade2fc304d4d76bde7f42f820890b1": "cow: SS-IoU 1.000 from text-only cond",
    "6fea7992af7c5338c21fe671860beeadd0cab14d2b2372e86e81110ac53d3d99": "white sports coupe twin",
}


def ss_path(sub, sha):
    return f"{DATA}/{sub}/{SS_ENC}/{sha}.npz"


def shape_path(sub, sha):
    return f"{DATA}/{sub}/{SHAPE_ENC}/{sha}.npz"


def pbr_path(sub, sha):
    return f"{DATA}/{sub}/{PBR_ENC}/{sha}.npz"


def renders_dir(sub, sha):
    return f"{DATA}/{sub}/renders_cond/{sha}"


# ── descriptors ───────────────────────────────────────────────────────────────
_PROJ = None


def proj_matrix():
    """Fixed seed-0 Gaussian JL matrix (32768 -> PROJ_DIM). Built in the PARENT before
    fork so the 67 MB is shared copy-on-write across workers."""
    global _PROJ
    if _PROJ is None:
        rng = np.random.default_rng(0)
        _PROJ = (rng.standard_normal((8 * 16 * 16 * 16, PROJ_DIM), dtype=np.float32)
                 / np.sqrt(PROJ_DIM))
    return _PROJ


def crop_thumb(path):
    """alpha-crop -> square -> THUMBxTHUMB, returns (rgb_on_white [T,T,3], alpha [T,T]),
    RAW values in [0,1].  Deliberately NOT per-block normalised: an earlier version
    mean-centred and L2-normalised the gray and alpha blocks separately, which threw away
    colour and absolute brightness — a RED sphere then matched a BLACK sphere at cos 0.986,
    and every sword matched every other sword.  Keep the raw signal; centre once, globally,
    at compare time."""
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im, dtype=np.float32)
    al = a[..., 3]
    ys, xs = np.nonzero(al > 8)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    half = max(y1 - y0, x1 - x0) * 0.52 + 1          # square + ~4% pad
    H, W = al.shape
    y0 = int(max(0, cy - half)); y1 = int(min(H, cy + half))
    x0 = int(max(0, cx - half)); x1 = int(min(W, cx + half))
    crop = a[y0:y1, x0:x1]
    im2 = Image.fromarray(crop.astype(np.uint8), "RGBA").resize((THUMB, THUMB), Image.BILINEAR)
    c = np.asarray(im2, dtype=np.float32) / 255.0
    alv = c[..., 3]
    comp = c[..., :3] * alv[..., None] + (1.0 - alv[..., None])
    return comp, alv


def feats_one(arg):
    """(sha, subset) -> (sha, render_desc, geom_proj[PROJ_DIM]) or None.

    render_desc layout (float16): per view, [rgb (THUMB^2*3) | alpha (THUMB^2)], views in
    VIEWS order -> stride VSTRIDE.  The alpha slice alone is a retexture-invariant
    silhouette descriptor; dedup scores appearance and silhouette separately."""
    sha, sub = arg
    rd = renders_dir(sub, sha)
    try:
        parts = []
        for v in VIEWS:
            t = crop_thumb(os.path.join(rd, f"{v:03d}.webp"))
            if t is None:
                return None
            parts.append(t[0].ravel())
            parts.append(t[1].ravel())
        desc = np.concatenate(parts).astype(np.float32)
        z = np.load(ss_path(sub, sha))["z"].astype(np.float32).ravel()
        n = np.linalg.norm(z)
        if not np.isfinite(n) or n == 0:
            return None
        g = ((z / n) @ proj_matrix()).astype(np.float32)
    except Exception:
        return None
    return sha, desc, g


def run_feats(items, out_npz, procs, tag):
    t0 = time.time()
    proj_matrix()                                      # build before fork
    shas, D, G = [], [], []
    with Pool(procs) as pool:
        for k, r in enumerate(pool.imap_unordered(feats_one, items, chunksize=32)):
            if r is not None:
                shas.append(r[0]); D.append(r[1]); G.append(r[2])
            if (k + 1) % 20000 == 0:
                el = time.time() - t0
                print(f"[{tag}] {k+1}/{len(items)} ok={len(shas)} {(k+1)/el:.0f}/s "
                      f"eta={(len(items)-k-1)/max((k+1)/el,1e-9)/60:.1f}min", flush=True)
    np.savez(out_npz, shas=np.array(shas),
             desc=np.array(D, dtype=np.float16), geom=np.array(G, dtype=np.float32),
             views=np.array(VIEWS), thumb=np.array(THUMB))
    print(f"[{tag}] wrote {out_npz}: {len(shas)}/{len(items)} in {time.time()-t0:.0f}s", flush=True)


# ── stages ────────────────────────────────────────────────────────────────────
def stage_pool(a):
    w = a.work
    t0 = time.time()
    # 1. training shas (the contamination reference set) = all U capT
    def manifest_items(p):
        out = []
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                out.append((r["sha256"], r["subset"]))
        return out
    tr_items = {}
    for p in TRAIN_MANIFESTS:
        for s, sub in manifest_items(p):
            tr_items.setdefault(s, sub)
    tr = set(tr_items)
    print(f"[pool] training shas (all U capT): {len(tr)}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(sorted(tr), open(f"{w}/train_shas.json", "w"))
    json.dump([[s, sub] for s, sub in tr_items.items()], open(f"{w}/train_items.json", "w"))
    fut = set()
    if os.path.isfile(FUTURE_MANIFEST):
        fut = {s for s, _ in manifest_items(FUTURE_MANIFEST)}
    print(f"[pool] ready_v5 (future pool) shas: {len(fut)}", flush=True)
    json.dump(sorted(fut), open(f"{w}/future_shas.json", "w"))

    # 1b. VLM judge scores. `vlm_filtered_all` == exactly the assets with
    #     validity=='single_object' AND recommended=='yes' (verified: 419,999/419,999),
    #     so re-applying that gate to the leftovers reproduces the training quality bar.
    J = {}
    for line in open(JUDGE):
        d = json.loads(line)
        J[d["sha"]] = d
    print(f"[pool] judge scores: {len(J)}  ({time.time()-t0:.0f}s)", flush=True)

    # 2. captions (holistic + texture) from the FINAL_* caption dumps
    def pull(raw, key):
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % key, raw)
        if not m:
            return None
        try:
            return json.loads('"' + m.group(1) + '"')
        except Exception:
            return None
    hol, tex = {}, {}
    for fn in ["FINAL_holistic_captions.json", "FINAL_sfv1_holistic_captions.json",
               "FINAL_tv_holistic_captions.json"]:
        for r in json.load(open(f"{CAPDIR}/{fn}"))["results"]:
            s, raw = r.get("sha"), r.get("raw", "") or ""
            cl, cm, cs = pull(raw, "caption_long"), pull(raw, "caption_medium"), pull(raw, "caption_short")
            if s and cl and cm and cs:
                hol[s] = [cl, cm, cs]
    for fn in ["FINAL_texture_captions.json", "FINAL_sfv1_texture_captions.json",
               "FINAL_tv_texture_captions.json"]:
        for r in json.load(open(f"{CAPDIR}/{fn}"))["results"]:
            s = r.get("sha")
            tc = pull(r.get("raw", "") or "", "texture_caption")
            if s and tc:
                tex[s] = tc
    print(f"[pool] captions holistic={len(hol)} texture={len(tex)}  ({time.time()-t0:.0f}s)", flush=True)

    # 3. on-disk enumeration, minus training shas, requiring every artefact
    pool_by_sub = {}
    for sub in SUBSETS:
        d = f"{DATA}/{sub}/{SS_ENC}"
        if not os.path.isdir(d):
            continue
        have_ss = {f[:-4] for f in os.listdir(d) if f.endswith(".npz")}
        have_sh = {f[:-4] for f in os.listdir(f"{DATA}/{sub}/{SHAPE_ENC}") if f.endswith(".npz")} \
            if os.path.isdir(f"{DATA}/{sub}/{SHAPE_ENC}") else set()
        have_pb = {f[:-4] for f in os.listdir(f"{DATA}/{sub}/{PBR_ENC}") if f.endswith(".npz")} \
            if os.path.isdir(f"{DATA}/{sub}/{PBR_ENC}") else set()
        have_rd = set(os.listdir(f"{DATA}/{sub}/renders_cond")) \
            if os.path.isdir(f"{DATA}/{sub}/renders_cond") else set()
        cand = (have_ss & have_sh & have_pb & have_rd & set(hol)) - tr
        pool_by_sub[sub] = sorted(cand)
        print(f"[pool] {sub}: ss={len(have_ss)} shape={len(have_sh)} pbr={len(have_pb)} "
              f"renders={len(have_rd)} -> candidates={len(cand)}", flush=True)
    json.dump(pool_by_sub, open(f"{w}/pool_by_subset.json", "w"))

    # 4. TIERS.
    #  A "matched"  — validity=single_object AND recommended=yes: the EXACT gate that
    #                 defined the training corpus, so tier A is drawn from the same
    #                 population the model was trained on.  It is nearly exhausted: the
    #                 training set consumed 419,999 of the 425,526 gate-passing assets.
    #  B "relaxed"  — single_object AND aesthetic>=5 AND structural>=6, i.e. good assets
    #                 the judge declined to recommend.  Used only to top tier A up; every
    #                 record carries `tier` so any statistic can be reported A-only.
    #  S "sentinel" — the incumbent held-out 14.  Never shipped in the manifest; carried
    #                 through feats/dedup ONLY so the criterion can be checked against the
    #                 known-contaminated cow.
    sub_of = {s: sub for sub, v in pool_by_sub.items() for s in v}
    A, Bpool = [], []
    for s, sub in sub_of.items():
        d = J.get(s)
        if not d or d.get("validity") != "single_object":
            continue
        if d.get("recommended") == "yes":
            A.append(s)
        elif d.get("aesthetic_score", 0) >= 5 and d.get("structural_score", 0) >= 6:
            Bpool.append(s)
    A.sort(); Bpool.sort()
    print(f"[pool] tier-A (training gate, unused): {len(A)}   tier-B pool: {len(Bpool)}",
          flush=True)
    rng = np.random.default_rng(20260727)
    # tier-B is sampled proportional to the TRAINING subset mix (not the pool mix), so
    # topping up does not import the pool's own subset skew on top of tier A's.
    tr_mix = {}
    for s, sub in tr_items.items():
        tr_mix[sub] = tr_mix.get(sub, 0) + 1
    ntr = sum(tr_mix.values())
    by_sub = {}
    for s in Bpool:
        by_sub.setdefault(sub_of[s], []).append(s)
    B = []
    for sub, v in sorted(by_sub.items()):
        n = min(len(v), int(round(a.n_tierb * tr_mix.get(sub, 0) / ntr)))
        if n:
            B += [v[i] for i in rng.choice(len(v), size=n, replace=False)]
    B.sort()
    print(f"[pool] tier-B sample: {len(B)}  (target {a.n_tierb}, training subset mix)",
          flush=True)

    sent = []
    if os.path.isfile(a.sentinels):
        for line in open(a.sentinels):
            r = json.loads(line)
            sent.append((r["sha256"], r["subset"]))
    tier = {}
    items = []
    for s in A:
        tier[s] = "A"; items.append((s, sub_of[s]))
    for s in B:
        tier[s] = "B"; items.append((s, sub_of[s]))
    for s, sub in sent:
        if s not in tier:
            tier[s] = "S"; items.append((s, sub))
    # verify 16 renders actually on disk (manifests lie; the migration left dead paths)
    keep = []
    for sha, sub in items:
        try:
            n = sum(1 for f in os.listdir(renders_dir(sub, sha)) if f.endswith(".webp"))
        except Exception:
            continue
        if n >= N_VIEWS_REQ:
            keep.append((sha, sub))
    print(f"[pool] candidates with >={N_VIEWS_REQ} verified renders: {len(keep)}/{len(items)}",
          flush=True)
    json.dump(keep, open(f"{w}/cand_items.json", "w"))
    json.dump(tier, open(f"{w}/cand_tier.json", "w"))
    json.dump({s: {"hol": hol.get(s), "tex": tex.get(s)} for s, _ in keep},
              open(f"{w}/cand_captions.json", "w"))
    json.dump({s: J[s] for s, _ in keep if s in J}, open(f"{w}/cand_judge.json", "w"))
    json.dump({s: J[s] for s in tr if s in J}, open(f"{w}/train_judge.json", "w"))
    print(f"[pool] DONE {time.time()-t0:.0f}s", flush=True)


def stage_feats(a):
    w = a.work
    if a.which == "train":
        items = [tuple(x) for x in json.load(open(f"{w}/train_items.json"))]
        out = f"{w}/feats_train.npz"
    else:
        items = [tuple(x) for x in json.load(open(f"{w}/cand_items.json"))]
        out = f"{w}/feats_cand.npz"
    if a.limit:
        items = items[:a.limit]
        out = out.replace(".npz", f"_lim{a.limit}.npz")
    run_feats(items, out, a.procs, a.which)


def _topk_cos(qc, tb_iter, k):
    """Streaming top-k cosine. qc (Nq,D) L2-normalised; tb_iter yields (offset, block)."""
    Nq = qc.shape[0]
    bestv = np.full((Nq, k), -2.0, dtype=np.float32)
    besti = np.full((Nq, k), -1, dtype=np.int64)
    for off, tb in tb_iter:
        sim = qc @ tb.T                                    # (Nq, B)
        allv = np.concatenate([bestv, sim], 1)
        alli = np.concatenate([besti, np.broadcast_to(off + np.arange(tb.shape[0]),
                                                      (Nq, tb.shape[0]))], 1)
        idx = np.argpartition(-allv, k - 1, axis=1)[:, :k]
        bestv = np.take_along_axis(allv, idx, 1)
        besti = np.take_along_axis(alli, idx, 1)
    order = np.argsort(-bestv, axis=1)
    return np.take_along_axis(bestv, order, 1), np.take_along_axis(besti, order, 1)


def _exact_geom_cos(pairs, sub_of):
    """pairs: list of (sha_a, sha_b). Returns dict[(a,b)] = exact cosine of the raw z."""
    cache = {}

    def z(s):
        if s not in cache:
            v = np.load(ss_path(sub_of[s], s))["z"].astype(np.float32).ravel()
            cache[s] = v / max(np.linalg.norm(v), 1e-8)
        return cache[s]
    return {p: float(z(p[0]) @ z(p[1])) for p in pairs}


def stage_dedup(a):
    w = a.work
    C = np.load(f"{w}/feats_cand.npz", allow_pickle=True)
    T = np.load(f"{w}/feats_train.npz", allow_pickle=True)
    cs, ts = [str(x) for x in C["shas"]], [str(x) for x in T["shas"]]
    print(f"[dedup] cand={len(cs)} train={len(ts)}", flush=True)
    res = {s: {} for s in cs}
    t0 = time.time()
    BLK = 40000

    # ---- render axis: two scores off the same descriptor ----
    #   appearance = full RGB+alpha  (a re-export with the same textures matches)
    #   silhouette = alpha only      (survives retexturing/recolouring)
    # Both are mean-centred on the training corpus, then view-aligned cosine.
    dt = T["desc"]
    thumb = int(T["thumb"]) if "thumb" in T.files else THUMB
    vstride = thumb * thumb * 4
    nv = dt.shape[1] // vstride
    aidx = np.concatenate([np.arange(v * vstride + thumb * thumb * 3,
                                     (v + 1) * vstride) for v in range(nv)])
    mu = np.zeros(dt.shape[1], dtype=np.float64)
    for s in range(0, dt.shape[0], BLK):
        mu += dt[s:s + BLK].astype(np.float32).sum(0)
    mu = (mu / dt.shape[0]).astype(np.float32)

    def norm(x, idx=None):
        x = x.astype(np.float32) - mu
        if idx is not None:
            x = x[:, idx]
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    res_axes = {}
    for name, idx in (("render", None), ("sil", aidx)):
        qc = norm(C["desc"], idx)

        def blocks(idx=idx):
            for s in range(0, dt.shape[0], BLK):
                yield s, norm(dt[s:s + BLK], idx)
        v, i = _topk_cos(qc, blocks(), a.topk)
        res_axes[name] = (v, i)
        print(f"[dedup] {name} cos: max={v[:,0].max():.4f} p99={np.percentile(v[:,0],99):.4f} "
              f"p50={np.percentile(v[:,0],50):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    rv, ri = res_axes["render"]
    sv, si = res_axes["sil"]

    # ---- geometry axis (JL screen, then exact re-score of the shortlist) ----
    gt = T["geom"].astype(np.float32)
    gmu = gt.mean(0, keepdims=True)
    gt = gt - gmu
    gt /= np.maximum(np.linalg.norm(gt, axis=1, keepdims=True), 1e-8)
    gq = C["geom"].astype(np.float32) - gmu
    gq /= np.maximum(np.linalg.norm(gq, axis=1, keepdims=True), 1e-8)
    gv, gi = _topk_cos(gq, ((s, gt[s:s + BLK]) for s in range(0, gt.shape[0], BLK)), a.topk)
    print(f"[dedup] geom(JL) cos: max={gv[:,0].max():.4f} p99={np.percentile(gv[:,0],99):.4f} "
          f"p50={np.percentile(gv[:,0],50):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    sub_of = {s: sub for s, sub in json.load(open(f"{w}/train_items.json"))}
    sub_of.update({s: sub for s, sub in json.load(open(f"{w}/cand_items.json"))})
    # exact re-score: every (cand, train) pair that either axis puts in the shortlist
    pairs = set()
    for i, s in enumerate(cs):
        for j in range(a.topk):
            if gv[i, j] >= a.geom_cos_screen and gi[i, j] >= 0:
                pairs.add((s, ts[int(gi[i, j])]))
            if rv[i, j] >= a.render_cos - 0.05 and ri[i, j] >= 0:
                pairs.add((s, ts[int(ri[i, j])]))     # exact geom for every render hit too
            if sv[i, j] >= a.sil_cos - 0.05 and si[i, j] >= 0:
                pairs.add((s, ts[int(si[i, j])]))
    print(f"[dedup] exact geom re-score on {len(pairs)} pairs", flush=True)
    exact = _exact_geom_cos(sorted(pairs), sub_of)
    best_exact = {}
    for (ca, tb), v in exact.items():
        if v > best_exact.get(ca, (-2, None))[0]:
            best_exact[ca] = (v, tb)

    for i, s in enumerate(cs):
        r = res[s]
        r["render_cos"] = float(rv[i, 0])
        r["render_match"] = ts[int(ri[i, 0])] if ri[i, 0] >= 0 else None
        r["render_top5"] = [[ts[int(ri[i, j])], float(rv[i, j])] for j in range(min(5, a.topk))]
        r["sil_cos"] = float(sv[i, 0])
        r["sil_match"] = ts[int(si[i, 0])] if si[i, 0] >= 0 else None
        r["geom_cos_jl"] = float(gv[i, 0])
        r["geom_match_jl"] = ts[int(gi[i, 0])] if gi[i, 0] >= 0 else None
        ev, et = best_exact.get(s, (-2.0, None))
        r["geom_cos_exact"] = ev
        r["geom_match_exact"] = et
        r["dup_render"] = r["render_cos"] >= a.render_cos
        r["dup_sil"] = r["sil_cos"] >= a.sil_cos
        r["dup_geom"] = ev >= a.geom_cos
        r["dup"] = r["dup_render"] or r["dup_geom"] or r["dup_sil"]
    n_r = sum(r["dup_render"] for r in res.values())
    n_s = sum(r["dup_sil"] for r in res.values())
    n_g = sum(r["dup_geom"] for r in res.values())
    n_a = sum(r["dup"] for r in res.values())
    n_ronly = sum(r["dup_render"] and not r["dup_geom"] for r in res.values())
    print(f"[dedup] thresholds render>={a.render_cos} sil>={a.sil_cos} geom>={a.geom_cos} "
          f"(screen {a.geom_cos_screen}): render-dup={n_r} sil-dup={n_s} geom-dup={n_g} "
          f"union={n_a} survivors={len(cs)-n_a}  (render caught {n_ronly} that geom missed)",
          flush=True)
    json.dump({"thresholds": {"render_cos": a.render_cos, "sil_cos": a.sil_cos,
                              "geom_cos": a.geom_cos,
                              "geom_cos_screen": a.geom_cos_screen, "topk": a.topk,
                              "views": list(VIEWS), "thumb": THUMB, "proj_dim": PROJ_DIM},
               "n_cand": len(cs), "n_train": len(ts),
               "n_dup_render": n_r, "n_dup_sil": n_s, "n_dup_geom": n_g, "n_dup_union": n_a,
               "n_render_only": n_ronly,
               "per_candidate": res},
              open(f"{w}/dedup_report.json", "w"), indent=1)
    print(f"[dedup] wrote {w}/dedup_report.json", flush=True)


def stage_manifest(a):
    w = a.work
    rep = json.load(open(f"{w}/dedup_report.json"))["per_candidate"]
    items = {s: sub for s, sub in json.load(open(f"{w}/cand_items.json"))}
    tier = json.load(open(f"{w}/cand_tier.json"))
    caps = json.load(open(f"{w}/cand_captions.json"))
    judge = json.load(open(f"{w}/cand_judge.json"))
    fut = set(json.load(open(f"{w}/future_shas.json")))

    # sentinel audit — the criterion is only trustworthy if it re-derives what we already
    # know.  Every KNOWN_CONTAMINATED asset must be flagged dup.
    audit = {}
    for sha, why in KNOWN_CONTAMINATED.items():
        r = rep.get(sha)
        audit[sha] = {"why": why, "found": r is not None,
                      "dup": bool(r and r["dup"]),
                      "render_cos": r and r["render_cos"], "render_match": r and r["render_match"],
                      "geom_cos_exact": r and r["geom_cos_exact"],
                      "geom_match_exact": r and r["geom_match_exact"]}
        print(f"[audit] {sha[:12]} ({why}): dup={audit[sha]['dup']} "
              f"render_cos={audit[sha]['render_cos']} geom_exact={audit[sha]['geom_cos_exact']}",
              flush=True)
    if not all(v["dup"] for v in audit.values()):
        print("[audit] *** A KNOWN-CONTAMINATED SENTINEL WAS NOT CAUGHT — the criterion "
              "is wrong; not writing a manifest. ***", flush=True)
        if not a.force:
            raise SystemExit(2)

    # threed.py applies min_aesthetic=4.5 AT LOAD, so the model never saw an asset scored
    # below that even though such rows sit in vlm_filtered_all.  Apply the same cut here or
    # the held-out set contains assets from outside the trained distribution.
    def ok(s):
        if rep[s]["dup"] or s in fut:
            return False
        aes = judge.get(s, {}).get("aesthetic_score")
        return aes is None or float(aes) >= a.min_aesthetic
    survA = sorted(s for s in items if tier.get(s) == "A" and ok(s))
    survB = sorted(s for s in items if tier.get(s) == "B" and ok(s))
    print(f"[manifest] survivors (dedup + aesthetic>={a.min_aesthetic} + not-in-v5): "
          f"tierA={len(survA)} tierB_pool={len(survB)}", flush=True)
    rng = np.random.default_rng(20260727)
    nb = min(len(survB), a.n_final_b)
    pickB = sorted(survB[i] for i in rng.choice(len(survB), size=nb, replace=False)) if nb else []
    pick = survA + pickB
    print(f"[manifest] final: {len(pick)}  (A={len(survA)} B={len(pickB)})", flush=True)

    def rec(sha, with_caps):
        sub = items[sha]
        c = caps[sha]["hol"]
        tc = caps[sha]["tex"]
        # caption order is CONTRACTUAL — build_vlm_cache_v22.py --mode captions maps
        # t000=long t001=medium t002=short t003=long+texture
        captions = [c[0], c[1], c[2], (c[0] + " " + tc) if tc else c[0]]
        j = judge.get(sha, {})
        return {"sha256": sha, "subset": sub,
                "ss_latent_64": ss_path(sub, sha),
                "shape_latent_512": shape_path(sub, sha),
                "pbr_latent_512": pbr_path(sub, sha),
                "renders_dir": renders_dir(sub, sha),
                "n_views": N_VIEWS_REQ,
                "captions": captions if with_caps else [],
                "aesthetic_score": j.get("aesthetic_score"),
                "vlm": {k: j[k] for k in ("structural_score", "texture_score",
                                          "aesthetic_score", "part_complexity",
                                          "detail_complexity", "color_richness",
                                          "style", "category", "issues") if k in j},
                "tier": tier[sha],
                "dedup": {k: rep[sha][k] for k in ("render_cos", "sil_cos",
                                                   "geom_cos_exact", "geom_cos_jl")}}

    os.makedirs(OUTDIR, exist_ok=True)
    base = f"{OUTDIR}/heldout_clean.jsonl"
    capt = f"{OUTDIR}/heldout_clean_capT.jsonl"
    nbad = 0
    with open(base, "w") as f1, open(capt, "w") as f2:
        for sha in pick:
            r = rec(sha, False)
            for k in ("ss_latent_64", "shape_latent_512", "pbr_latent_512"):
                if not os.path.isfile(r[k]):
                    nbad += 1
                    print(f"[manifest] MISSING {k} for {sha[:12]}", flush=True)
            if not os.path.isdir(r["renders_dir"]):
                nbad += 1
                print(f"[manifest] MISSING renders_dir for {sha[:12]}", flush=True)
            f1.write(json.dumps(r) + "\n")
            f2.write(json.dumps(rec(sha, True)) + "\n")
    json.dump(audit, open(f"{OUTDIR}/sentinel_audit.json", "w"), indent=1)
    print(f"[manifest] wrote {base} and {capt}  (path failures: {nbad})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pool", "feats", "dedup", "manifest"])
    ap.add_argument("--work", default=f"{OUTDIR}/_work")
    ap.add_argument("--which", default="cand", choices=["train", "cand"])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_tierb", type=int, default=1200,
                    help="size of the tier-B candidate pool fed to dedup (final tier-B "
                         "count is set by --n_final_b after dedup)")
    ap.add_argument("--sentinels", default="/fsx/home/weikai.huang/3dgen/im_probe/"
                                           "heldout14_newpaths.jsonl")
    ap.add_argument("--n_final_b", type=int, default=300)
    ap.add_argument("--min_aesthetic", type=float, default=4.5,
                    help="same cut threed.py applies at load time")
    ap.add_argument("--force", action="store_true",
                    help="write the manifest even if a sentinel audit fails")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--render_cos", type=float, default=0.97)
    ap.add_argument("--sil_cos", type=float, default=0.99)
    ap.add_argument("--geom_cos", type=float, default=0.90)
    ap.add_argument("--geom_cos_screen", type=float, default=0.70)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    {"pool": stage_pool, "feats": stage_feats,
     "dedup": stage_dedup, "manifest": stage_manifest}[a.stage](a)


if __name__ == "__main__":
    main()
