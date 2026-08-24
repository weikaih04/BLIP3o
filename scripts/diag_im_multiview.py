"""Multi-view diagnostic for the IM-trained SS flow.

QUESTION: does the IM SS model actually USE the distinct information in the 4
conditioning views, or does it only condition on "there are 4 image slots" and
ignore the per-view content?

For EACH of the 14 held-out assets we build 3 cond variants for the SAME IM model
(SS_CKPT_IM), sample SS occupancy with the SAME seed (0), and IoU vs GT@64:

  (1) 4-distinct : native m00 cond (all 4 dino view segments are distinct).
                   == the proven build_cond_im.
  (2) 4-copy     : same token count/layout as (1), but the dino segment is replaced
                   so all 4 view-slots carry view-ordinal-0's content (tiled 4x).
                   dino_view_ids kept as [0,1,2,3] blocks so the per-view dve is
                   STILL applied — the ONLY difference from (1) is the raw dino
                   content. Isolates: does DISTINCT view content matter vs just
                   having 4 image-token slots + per-view embeddings.
  (3) 1-view     : only view-0's dino segment (fewer dino tokens). qwen joint kept.

The qwen-joint segment (a["hidden"]) is IDENTICAL across all 3 variants, so the
differences are pure dino/multi-view-content effects.

KEY: mean IoU(4-distinct) vs mean IoU(4-copy).
  4d >> 4c  -> the model genuinely fuses distinct views (multi-view works).
  4d ~= 4c  -> the model IGNORES distinct view content (multi-view fusion NOT learned).

Env:
  SS_CKPT_IM = IM SS ckpt dir (e.g. runs/s3_ss_im_mds/checkpoint-10000)   [required]
  COND_IM    = IM cond cache root (default /fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_im4l)
  MANI       = held-out manifest (default /fsx/home/weikai.huang/3dgen/im_probe/heldout14.jsonl)
  SEED       = sampler seed (default 0)
  SHARD/NUM_SHARDS = split the manifest across GPUs; each shard writes its own json and
               the caller reduces (scripts/reduce_diag_im.py). Default 0/1 = old behaviour.
  OUT_TAG    = output json suffix (default: ckpt basename [+ _s{SEED}] [+ _sh{SHARD}])
"""
import os, sys, json
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import numpy as np
import torch

from trellis2_blip3o import _paths  # noqa
from trellis2 import models as t2models  # type: ignore
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # type: ignore
# reuse the EXACT proven loaders/decoder path (copied refs, not importing main())
from scripts.export_glb_fullchain import load_ss_flow, SSDEC

ROOT = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
SS_CKPT_IM = os.environ["SS_CKPT_IM"]
COND_IM = os.environ.get("COND_IM", "/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_im4l")
MANI = os.environ.get("MANI", "/fsx/home/weikai.huang/3dgen/im_probe/heldout14.jsonl")
# Official SS sampler settings (identical to eval_im_vs_i1_ss.py / fused_fullchain_eval)
SS_OFF = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
              guidance_interval=[0.6, 1.0], rescale_t=5.0)
SEED = int(os.environ.get("SEED", "0"))
SHARD = int(os.environ.get("SHARD", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
_tag = os.path.basename(SS_CKPT_IM.rstrip("/"))
if SEED:
    _tag += f"_s{SEED}"
if NUM_SHARDS > 1:
    _tag += f"_sh{SHARD}of{NUM_SHARDS}"
OUT_TAG = os.environ.get("OUT_TAG", _tag)


def make_qwen_cond(conn, a):
    """connector(qwen-joint) + uncond connector(0). IDENTICAL across all variants."""
    qwen = torch.from_numpy(a["hidden"]).float().cuda()   # (Tq, 2048) joint 4-view
    with torch.no_grad():
        cq = conn(qwen[None])
        c0 = conn(torch.zeros_like(qwen)[None])
        if getattr(conn, "pos_stamp", None) is not None:
            from trellis2_blip3o.pos_stamp import IMG_SPAN_FULL
            cq = conn.pos_stamp(cq, IMG_SPAN_FULL)
            c0 = conn.pos_stamp(c0, IMG_SPAN_FULL)
    return cq, c0


def build_variant(mode, a, dve, cq, c0):
    """Return (cond, uncond, dino_content_used) for one variant.

    dino_content_used = the RAW dino segment (pre-dve, post keep-select) used, so the
    caller can verify the 4-copy tiling really made all view-slots identical."""
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()   # (Td, 1024)
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()  # (Td,) per-token view ordinal

    if mode == "4distinct":
        d_dino, d_vids, d_dmask = dino, vids, dmask

    elif mode == "4copy":
        # view blocks must be equal-length & contiguous for the tile to line up with vids
        uids = torch.unique(vids)
        counts = [(int(vids.eq(u).sum())) for u in uids]
        assert len(set(counts)) == 1, f"view blocks unequal length {counts} -> cannot tile"
        sel0 = (vids == 0)
        v0 = dino[sel0]                       # (n0, 1024) view-0 content
        v0m = dmask[sel0]                     # (n0,)
        nblk = int(vids.max().item()) + 1     # 4
        d_dino = v0.repeat(nblk, 1)           # [v0;v0;v0;v0] -> same content in every slot
        d_dmask = v0m.repeat(nblk)
        d_vids = vids.clone()                 # KEEP [0..0,1..1,2..2,3..3] so per-view dve still applied
        # (relies on equal contiguous blocks so d_dino block-j content == v0 == block-0)

    elif mode == "1view":
        sel0 = (vids == 0)
        d_dino = dino[sel0]                   # only view-0 tokens (fewer)
        d_dmask = dmask[sel0]
        d_vids = torch.zeros(d_dino.shape[0], dtype=torch.long, device=dino.device)  # dve[0]

    else:
        raise ValueError(mode)

    # QWEN-segment view identity — mirror of training (flow_heads adds the SAME dve to the
    # qwen tokens). Identical in every mode (it is a slot identity, not content), so it does
    # not confound the 4-distinct vs 4-copy contrast; omitting it would instead evaluate the
    # model missing a signal it was trained with. Cache layout: 292 tokens, four 64-token
    # image blocks at 10/76/142/208 (stride 66); text/structural tokens get no code.
    if dve is not None and cq.shape[1] >= 272:
        import torch as _t
        _qv = _t.full((cq.shape[1],), -1, dtype=_t.long, device=cq.device)
        for _v in range(4):
            _qv[10 + 66 * _v: 10 + 66 * _v + 64] = _v
        cq = cq + (dve[_qv.clamp_min(0)].float() * (_qv >= 0).unsqueeze(-1).float())[None]
    # trellis2_blip3o.eval_cond wraps build_unified_cond — the training loop's own
    # builder — so eval and training cannot disagree about the dino segment, the
    # view embedding, or which drop is CFG.
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    cond, uncond = cond_uncond_from_tensors(conn, qwen, qmask, d_dino, d_dmask, dve,
                                            dino_view_ids=d_vids, qwen_view_ids=_qv)
    return cond, uncond, d_dino, d_vids


@torch.no_grad()
def sample_occ(flow, sampler, ssdec, cond, uncond, seed=SEED):
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution,
                        generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = sampler.sample(flow, noise, cond=cond, neg_cond=uncond, verbose=False, **SS_OFF).samples
    return (ssdec(z) > 0)[0, 0].cpu().numpy()


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else -1.0


def verify_4copy(a, dve):
    """One-time structural sanity: 4-copy slots are content-identical; 4-distinct are not;
    token counts line up with native."""
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()
    # build the same tiled content the variant uses
    sel0 = (vids == 0)
    v0 = dino[sel0]
    nblk = int(vids.max().item()) + 1
    tiled = v0.repeat(nblk, 1)
    n0 = v0.shape[0]
    print("[verify] --- 4-copy structural check ---", flush=True)
    print(f"[verify] native dino tokens={dino.shape[0]}  views={list(int(v) for v in a['views'])}  "
          f"per-view={n0}  nblk={nblk}", flush=True)
    print(f"[verify] 4copy tiled dino tokens={tiled.shape[0]} (== native {dino.shape[0]}? "
          f"{tiled.shape[0] == dino.shape[0]})", flush=True)
    # slot1 == slot0 in the tiled (4-copy) content?
    same_c = torch.allclose(tiled[n0:2*n0], tiled[:n0])
    print(f"[verify] 4copy: slot1_content == slot0_content ? {bool(same_c)}", flush=True)
    same_c2 = torch.allclose(tiled[2*n0:3*n0], tiled[:n0]) and torch.allclose(tiled[3*n0:4*n0], tiled[:n0])
    print(f"[verify] 4copy: slot2 & slot3 also == slot0 ? {bool(same_c2)}", flush=True)
    # native (4-distinct): slot1 != slot0 (views genuinely differ)?
    dist = not torch.allclose(dino[n0:2*n0], dino[:n0])
    print(f"[verify] 4distinct(native): slot1_content != slot0_content ? {bool(dist)} "
          f"(maxdiff={float((dino[n0:2*n0]-dino[:n0]).abs().max()):.3f})", flush=True)
    # confirm per-view dve is still applied differently across slots (identity kept)
    if dve is not None:
        d01 = float((dve[0] - dve[1]).abs().max())
        print(f"[verify] dve[0] vs dve[1] differ (per-view embed still applied on 4copy): "
              f"maxdiff={d01:.3f}", flush=True)
    print("[verify] --- end check ---", flush=True)


def main():
    recs = [json.loads(l) for l in open(MANI)]
    if NUM_SHARDS > 1:                       # round-robin so every shard sees a mixed slice
        recs = recs[SHARD::NUM_SHARDS]
    print(f"[diag] {len(recs)} held-out assets (shard {SHARD}/{NUM_SHARDS}); "
          f"IM ckpt={SS_CKPT_IM}", flush=True)
    print(f"[diag] SS_OFF={SS_OFF}  seed={SEED}", flush=True)

    ssdec = t2models.from_pretrained(SSDEC).cuda().eval()
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    flow, conn, dve = load_ss_flow(SS_CKPT_IM)
    print(f"[diag] dve shape={None if dve is None else tuple(dve.shape)}  "
          f"pos_stamp={'yes' if getattr(conn,'pos_stamp',None) is not None else 'no'}", flush=True)

    verified = False
    rows = []
    for r in recs:
        sha = r["sha256"]; s8 = sha[:8]
        ed = os.path.join(COND_IM, sha[:2], sha)
        if not os.path.exists(os.path.join(ed, "m00.npz")):
            print(f"[skip] {s8}: no IM cond entry", flush=True); continue
        a = np.load(os.path.join(ed, "m00.npz"))

        if not verified:
            verify_4copy(a, dve); verified = True

        # GT occupancy @64
        gz = torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None]
        with torch.no_grad():
            gt = (ssdec(gz) > 0)[0, 0].cpu().numpy()

        cq, c0 = make_qwen_cond(conn, a)   # shared across variants

        c4d, u4d, dc4d, _ = build_variant("4distinct", a, dve, cq, c0)
        c4c, u4c, dc4c, _ = build_variant("4copy",     a, dve, cq, c0)
        c1v, u1v, dc1v, _ = build_variant("1view",     a, dve, cq, c0)

        occ4d = sample_occ(flow, sampler, ssdec, c4d, u4d)
        occ4c = sample_occ(flow, sampler, ssdec, c4c, u4c)
        occ1v = sample_occ(flow, sampler, ssdec, c1v, u1v)

        io4d, io4c, io1v = iou(occ4d, gt), iou(occ4c, gt), iou(occ1v, gt)
        rows.append((s8, io4d, io4c, io1v, c4d.shape[1], c4c.shape[1], c1v.shape[1]))
        print(f"[diag] {s8}  4d={io4d:.3f}  4c={io4c:.3f}  1v={io1v:.3f}  "
              f"(cond_tok 4d={c4d.shape[1]} 4c={c4c.shape[1]} 1v={c1v.shape[1]})", flush=True)

    # ---- summary ----
    D = np.array([x[1] for x in rows]); C = np.array([x[2] for x in rows]); V = np.array([x[3] for x in rows])
    m = (D >= 0) & (C >= 0) & (V >= 0)
    n = int(m.sum())
    summ = {
        "im_ckpt": SS_CKPT_IM, "n": n, "seed": SEED, "mani": MANI, "cond_im": COND_IM,
        "shard": SHARD, "num_shards": NUM_SHARDS,
        "mean_4distinct": float(D[m].mean()), "median_4distinct": float(np.median(D[m])),
        "mean_4copy": float(C[m].mean()), "median_4copy": float(np.median(C[m])),
        "mean_1view": float(V[m].mean()), "median_1view": float(np.median(V[m])),
        "delta_4d_minus_4c": float(D[m].mean() - C[m].mean()),
        "delta_4d_minus_1v": float(D[m].mean() - V[m].mean()),
        "n_assets_4d_gt_4c": int(((D > C) & m).sum()),
        "per_asset": [{"sha8": s, "4distinct": d, "4copy": c, "1view": v}
                      for (s, d, c, v, *_ ) in rows],
    }
    out_json = f"{ROOT}/runs/cache_logs/diag_im_mv_{OUT_TAG}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(summ, open(out_json, "w"), indent=2)

    print("\n=== DIAG SUMMARY (IM multi-view usage) ===", flush=True)
    print(f"  n={n}  seed={SEED}", flush=True)
    print(f"  mean   4distinct={summ['mean_4distinct']:.3f}  4copy={summ['mean_4copy']:.3f}  "
          f"1view={summ['mean_1view']:.3f}", flush=True)
    print(f"  median 4distinct={summ['median_4distinct']:.3f}  4copy={summ['median_4copy']:.3f}  "
          f"1view={summ['median_1view']:.3f}", flush=True)
    print(f"  KEY delta 4d-4c = {summ['delta_4d_minus_4c']:+.4f}   "
          f"(4d>4c on {summ['n_assets_4d_gt_4c']}/{n} assets)", flush=True)
    print(f"      delta 4d-1v = {summ['delta_4d_minus_1v']:+.4f}", flush=True)
    verdict = ("USES multi-view (distinct content helps)"
               if summ['delta_4d_minus_4c'] > 0.01 and summ['n_assets_4d_gt_4c'] > n // 2
               else "IGNORES distinct view content (4-slot conditioning only)")
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  wrote {out_json}", flush=True)
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
