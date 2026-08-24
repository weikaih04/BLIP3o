"""Back-face recovery test for the IM-trained SS flow (runs/s3_ss_im_mds/checkpoint-10000).

USER QUESTION (verbatim): on objects where the back differs from the front, if we give
the model views that SEE the back, does the back of the generated 3D improve vs giving
ONLY front-ish views?  This isolates whether SS uses multi-view for OCCLUDED geometry
(vs the earlier 4d~=4c null, which was about good-view-band redundancy).

Per asset (14 held-out):
  1. Decode GT occupancy @64 (ssdec(ss_latent_64)). Compute per-view visibility for the 4
     conditioning views via mv_visibility_analysis.visible_mask (z-buffer splat) on the 64^3
     GT voxels. Camera poses from {renders_dir}/transforms.json (OpenGL c2w, look-at origin).
       front view = the combo view with highest single-view GT coverage.
       back voxels = GT voxels NOT visible from the front view.
       recoverable back = back voxels visible from >=1 of the other 3 combo views (the region
                          multi-view COULD recover).  best-back view = the single other view
                          seeing the most back.
  2. Qualify: |back|/|surface| >= 0.15  AND  |recoverable|/|back| >= 0.30.
  3. Sample SS occupancy (same seed) under 3 conds, qwen-joint held IDENTICAL, only DINO varies:
       A FRONT-ONLY : DINO = front view's block alone (relabelled slot-0, dve[0]).
       B ALL-4      : native 4-view DINO cond (build_cond_im).
       C BEST-BACK  : DINO = best-back view's block alone (slot-0) -- sanity ceiling that the
                      back-seeing view individually carries the back info.
  4. Metrics: overall IoU@64, and the KEY ones restricted to the recoverable-back region:
       back_recall = |pred & back| / |back|            (direct back recovery)
       back_IoU    = IoU(pred, GT) over a dilated back region (penalises hallucination too).
     If SS uses back-seeing views: B's back >> A's back. If it ignores them: B ~= A.

The ss-occ 64^3 numpy grid is aligned to the (known-good) shape-coord world frame ONCE by an
axis perm/flip search that maximises summed IoU(occ32, shape_occ32) over all assets; that
mapping is used only to turn grid indices into world coords for the cameras (the region masks
themselves stay in native decoder index space, identical to the sampled predictions).

Env:  SS_CKPT_IM (default runs/s3_ss_im_mds/checkpoint-10000)
Outputs: runs/cache_logs/ss_backface/{summary.json, grid.png}
"""
import os, sys, json, itertools
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FUSED_MODULATE", "1")
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from trellis2_blip3o import _paths  # noqa
from trellis2 import models as t2models  # type: ignore
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # type: ignore
from scripts.export_glb_fullchain import load_ss_flow, SSDEC
import scripts.mv_visibility_analysis as MV
from scripts.mv_visibility_analysis import load_cameras

ROOT = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
SS_CKPT_IM = os.environ.get("SS_CKPT_IM", f"{ROOT}/runs/s3_ss_im_mds/checkpoint-10000")
COND_IM = os.environ.get("COND_IM", "/fsx/home/weikai.huang/3dgen/im_probe/v22_heldout_im4l")
MANI = os.environ.get("MANI", "/fsx/home/weikai.huang/3dgen/im_probe/heldout14.jsonl")
OUT_DIR = f"{ROOT}/runs/cache_logs/ss_backface"
SS_OFF = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
              guidance_interval=[0.6, 1.0], rescale_t=5.0)
SEED = 0
RES = 64
BACK_FRAC_MIN = 0.15      # |back|/|surface| threshold to have a "meaningful back"
RECOV_FRAC_MIN = 0.30     # |recoverable|/|back| : some other view must see a good chunk
FONT_PATH = ("/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/lib/python3.10/"
             "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")

# run visibility on the 64^3 GT grid (mv_visibility defaults are for 32^3)
MV.RES = RES
MV.VOX = 1.0 / RES


# ----------------------------- cond builders (qwen-joint held constant) -----------------------------
def make_qwen_cond(conn, a):
    qwen = torch.from_numpy(a["hidden"]).float().cuda()
    with torch.no_grad():
        cq = conn(qwen[None]); c0 = conn(torch.zeros_like(qwen)[None])
        if getattr(conn, "pos_stamp", None) is not None:
            from trellis2_blip3o.pos_stamp import IMG_SPAN_FULL
            cq = conn.pos_stamp(cq, IMG_SPAN_FULL); c0 = conn.pos_stamp(c0, IMG_SPAN_FULL)
    return cq, c0


def build_all4(a, dve, cq, c0):
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    # trellis2_blip3o.eval_cond wraps build_unified_cond — the training loop's own
    # builder — so eval and training cannot disagree about the dino segment, the
    # view embedding, or which drop is CFG.
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    return cond_uncond_from_tensors(conn, qwen, qmask, dino, dmask, dve,
                                    dino_view_ids=vids)


def build_1view(a, dve, cq, c0, ordinal):
    """Single view's DINO block, relabelled to slot-0 (dve[0]) exactly like diag_im 1view."""
    dino = torch.from_numpy(a["dino_hidden"]).float().cuda()
    dmask = torch.from_numpy(a["dino_keep_mask"]).cuda()
    vids = torch.from_numpy(a["dino_view_ids"]).long().cuda()
    qmask = torch.from_numpy(a["keep_mask"]).cuda()
    sel = (vids == ordinal)
    d_dino = dino[sel]; d_dmask = dmask[sel]
    d_vids = torch.zeros(d_dino.shape[0], dtype=torch.long, device=dino.device)
    from trellis2_blip3o.eval_cond import cond_uncond_from_tensors
    return cond_uncond_from_tensors(conn, qwen, qmask, d_dino, d_dmask, dve,
                                    dino_view_ids=d_vids)


@torch.no_grad()
def sample_occ(flow, sampler, ssdec, cond, uncond, seed=SEED):
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution,
                        generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = sampler.sample(flow, noise, cond=cond, neg_cond=uncond, verbose=False, **SS_OFF).samples
    return (ssdec(z) > 0)[0, 0].cpu().numpy()


# ----------------------------- geometry helpers -----------------------------
def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else -1.0


def recall(pred, region):
    region = region.astype(bool); n = int(region.sum())
    return float((pred.astype(bool) & region).sum()) / n if n else -1.0


def dilate(m):
    out = m.copy()
    for ax in range(3):
        out |= np.roll(m, 1, ax); out |= np.roll(m, -1, ax)
    return out


def apply_align(coords, perm, signs):
    """coords (N,3) int grid indices -> aligned grid indices (shape-coord world frame)."""
    out = coords[:, list(perm)].copy()
    for a in range(3):
        if signs[a]:
            out[:, a] = (RES - 1) - out[:, a]
    return out


def maxpool2(occ):
    o = occ.reshape(RES // 2, 2, RES // 2, 2, RES // 2, 2)
    return o.any(axis=(1, 3, 5))


def _score_align(gt32_coords, shape_occ32s, perm, signs):
    tot = 0.0
    for c32, sh in zip(gt32_coords, shape_occ32s):
        ca = c32[:, list(perm)].copy()
        for a in range(3):
            if signs[a]:
                ca[:, a] = 31 - ca[:, a]
        g = np.zeros((32, 32, 32), bool)
        g[ca[:, 0], ca[:, 1], ca[:, 2]] = True
        tot += iou(g, sh)
    return tot


def choose_alignment(gt_occs, shape_occ32s):
    gt32_coords = [np.argwhere(maxpool2(o)) for o in gt_occs]
    scored = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((0, 1), repeat=3):
            scored.append((_score_align(gt32_coords, shape_occ32s, perm, signs), perm, signs))
    scored.sort(key=lambda x: -x[0])
    n = len(gt_occs)
    print(f"[align] top-3 (mean IoU over {n} assets): "
          + "  ".join(f"perm{p}flip{s}={sc/n:.3f}" for sc, p, s in scored[:3]), flush=True)
    ident = [sc for sc, p, s in scored if p == (0, 1, 2) and s == (0, 0, 0)][0]
    print(f"[align] identity mean IoU = {ident/n:.3f}", flush=True)
    return scored[0][1], scored[0][2], scored[0][0] / n, ident / n


def load_shape_occ32(rec):
    coords = np.load(rec["shape_latent_512"])["coords"].astype(np.int64)
    g = np.zeros((32, 32, 32), bool)
    g[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return g


# ----------------------------- rendering -----------------------------
def colored_proj(occ, highlight, hi_color):
    """3 orthographic max-projections; occ in gray, highlight-overlap in hi_color. -> RGB."""
    canvas = np.zeros((RES, RES * 3, 3), np.uint8)
    hl = occ.astype(bool) & highlight.astype(bool)
    for i, ax in enumerate((2, 1, 0)):
        base = occ.max(ax).astype(bool)
        hi = hl.max(ax).astype(bool)
        tile = np.zeros((RES, RES, 3), np.uint8)
        tile[base] = (150, 150, 150)
        tile[hi] = hi_color
        canvas[:, i * RES:(i + 1) * RES] = tile
    im = Image.fromarray(canvas).resize((RES * 3 * 2, RES * 2), Image.NEAREST)
    return im


def load_view_img(renders_dir, v, size):
    for ext in ("webp", "png", "jpg"):
        p = os.path.join(renders_dir, f"{int(v):03d}.{ext}")
        if os.path.isfile(p):
            return Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
    return Image.new("RGB", (size, size), (230, 230, 230))


# ----------------------------- main -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = [json.loads(l) for l in open(MANI)]
    print(f"[bf] {len(recs)} held-out assets; IM ckpt={SS_CKPT_IM}", flush=True)
    print(f"[bf] SS_OFF={SS_OFF} seed={SEED} back_frac_min={BACK_FRAC_MIN} "
          f"recov_frac_min={RECOV_FRAC_MIN}", flush=True)

    ssdec = t2models.from_pretrained(SSDEC).cuda().eval()
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    flow, conn, dve = load_ss_flow(SS_CKPT_IM)
    print(f"[bf] dve={None if dve is None else tuple(dve.shape)}  "
          f"pos_stamp={'yes' if getattr(conn,'pos_stamp',None) is not None else 'no'}", flush=True)

    # --- pass 0: decode all GT occ + shape occ, choose global alignment ---
    valid, gt_occs, shape32 = [], [], []
    for r in recs:
        sha = r["sha256"]
        ed = os.path.join(COND_IM, sha[:2], sha, "m00.npz")
        if not os.path.exists(ed):
            print(f"[skip] {sha[:8]}: no IM cond entry", flush=True); continue
        gz = torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None]
        with torch.no_grad():
            occ = (ssdec(gz) > 0)[0, 0].cpu().numpy()
        assert occ.shape == (RES, RES, RES), occ.shape
        valid.append(r); gt_occs.append(occ); shape32.append(load_shape_occ32(r))
    perm, signs, best_mean, ident_mean = choose_alignment(gt_occs, shape32)
    print(f"[align] CHOSEN perm={perm} signs={signs} (mean IoU {best_mean:.3f} vs identity "
          f"{ident_mean:.3f})", flush=True)

    # --- determinism check on asset 0 ---
    a0 = np.load(os.path.join(COND_IM, valid[0]["sha256"][:2], valid[0]["sha256"], "m00.npz"))
    cq0, c00 = make_qwen_cond(conn, a0)
    cA0, uA0 = build_1view(a0, dve, cq0, c00, 0)
    o1 = sample_occ(flow, sampler, ssdec, cA0, uA0)
    o2 = sample_occ(flow, sampler, ssdec, cA0, uA0)
    print(f"[verify] same-seed determinism: identical={np.array_equal(o1, o2)} "
          f"(occ voxels {int(o1.sum())})", flush=True)

    rows, grid_rows = [], []
    for idx, (r, gt) in enumerate(zip(valid, gt_occs)):
        sha = r["sha256"]; s8 = sha[:8]
        a = np.load(os.path.join(COND_IM, sha[:2], sha, "m00.npz"))
        combo = [int(v) for v in a["views"]]
        c2w, fov, resid = load_cameras(r["renders_dir"])

        coords = np.argwhere(gt)                         # (N,3) native index space
        coords_w = apply_align(coords, perm, signs)      # -> shape/world frame indices
        centers_w = (coords_w.astype(np.float64) + 0.5) * MV.VOX - 0.5
        vis = {v: MV.visible_mask(centers_w, c2w[v], fov[v]) for v in combo}   # per-view bool (N,)
        cov = {v: float(vis[v].mean()) for v in combo}
        front = max(combo, key=lambda v: cov[v])
        others = [v for v in combo if v != front]

        back = ~vis[front]                                # GT voxels not seen by front (N,)
        seen_by_other = np.zeros(len(coords), bool)
        for v in others:
            seen_by_other |= vis[v]
        recover = back & seen_by_other                   # multi-view COULD recover these
        best_back = max(others, key=lambda v: int((back & vis[v]).sum()))
        n = len(coords)
        back_frac = back.sum() / n
        recov_frac = (recover.sum() / back.sum()) if back.sum() else 0.0
        best_back_cov = float((back & vis[best_back]).sum()) / max(1, int(back.sum()))

        qualifies = (back_frac >= BACK_FRAC_MIN) and (recov_frac >= RECOV_FRAC_MIN)

        # region masks in NATIVE index space (aligned with predictions)
        back_grid = np.zeros_like(gt); back_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = back
        recov_grid = np.zeros_like(gt); recov_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = recover
        R = dilate(recov_grid)                            # spatial back zone for back_IoU

        # ordinals within the m00 (a["views"] order == dino_view_ids 0..3)
        ord_front = combo.index(front); ord_back = combo.index(best_back)

        cq, c0 = make_qwen_cond(conn, a)
        cA, uA = build_1view(a, dve, cq, c0, ord_front)   # FRONT-ONLY
        cB, uB = build_all4(a, dve, cq, c0)               # ALL-4
        cC, uC = build_1view(a, dve, cq, c0, ord_back)    # BEST-BACK alone
        occA = sample_occ(flow, sampler, ssdec, cA, uA)
        occB = sample_occ(flow, sampler, ssdec, cB, uB)
        occC = sample_occ(flow, sampler, ssdec, cC, uC)

        def gtmask(pred):  # IoU over dilated recoverable-back region
            return iou(pred & R, gt & R)
        row = dict(
            sha8=s8, subset=r["subset"], combo=combo, front=front, best_back=best_back,
            n_surf=n, back_frac=round(float(back_frac), 3), recov_frac=round(float(recov_frac), 3),
            best_back_cov=round(best_back_cov, 3), front_cov=round(cov[front], 3),
            qualifies=bool(qualifies),
            overall_iou=dict(A=round(iou(occA, gt), 3), B=round(iou(occB, gt), 3),
                             C=round(iou(occC, gt), 3)),
            back_recall=dict(A=round(recall(occA, recov_grid), 3), B=round(recall(occB, recov_grid), 3),
                             C=round(recall(occC, recov_grid), 3)),
            back_iou=dict(A=round(gtmask(occA), 3), B=round(gtmask(occB), 3),
                          C=round(gtmask(occC), 3)),
        )
        rows.append(row)
        print(f"[{'Q' if qualifies else '.'}] {s8} front=v{front} bestback=v{best_back} "
              f"|back|/surf={back_frac:.2f} recov/back={recov_frac:.2f} bbcov={best_back_cov:.2f} "
              f"| overallIoU A/B/C={row['overall_iou']['A']:.2f}/{row['overall_iou']['B']:.2f}/"
              f"{row['overall_iou']['C']:.2f} | backRecall A/B/C="
              f"{row['back_recall']['A']:.2f}/{row['back_recall']['B']:.2f}/{row['back_recall']['C']:.2f} "
              f"| backIoU A/B/C={row['back_iou']['A']:.2f}/{row['back_iou']['B']:.2f}/"
              f"{row['back_iou']['C']:.2f}", flush=True)

        if qualifies:
            grid_rows.append(dict(row=row, rd=r["renders_dir"], front=front, best_back=best_back,
                                  occA=occA, occB=occB, occC=occC, gt=gt, recov=recov_grid))

    # --- summary over qualifying assets ---
    Q = [r for r in rows if r["qualifies"]]

    def arr(cond, metric):
        return np.array([r[metric][cond] for r in Q], float)
    summ = {
        "im_ckpt": SS_CKPT_IM, "seed": SEED, "align_perm": list(perm), "align_signs": list(signs),
        "align_mean_iou": round(best_mean, 3), "align_identity_iou": round(ident_mean, 3),
        "n_total": len(rows), "n_qualifying": len(Q),
        "qualifying_shas": [r["sha8"] for r in Q],
        "nonqualifying_shas": [r["sha8"] for r in rows if not r["qualifies"]],
    }
    if Q:
        for metric in ("back_recall", "back_iou", "overall_iou"):
            summ[f"mean_{metric}"] = {c: round(float(arr(c, metric).mean()), 3) for c in "ABC"}
        summ["mean_back_recall_delta_B_minus_A"] = round(
            float((arr("B", "back_recall") - arr("A", "back_recall")).mean()), 3)
        summ["mean_back_iou_delta_B_minus_A"] = round(
            float((arr("B", "back_iou") - arr("A", "back_iou")).mean()), 3)
        summ["n_B_gt_A_backrecall_by_0.05"] = int(
            ((arr("B", "back_recall") - arr("A", "back_recall")) > 0.05).sum())
        summ["n_B_gt_A_backiou_by_0.05"] = int(
            ((arr("B", "back_iou") - arr("A", "back_iou")) > 0.05).sum())
        dR = summ["mean_back_recall_delta_B_minus_A"]; dI = summ["mean_back_iou_delta_B_minus_A"]
        summ["verdict"] = ("BACK RECOVERED by back-seeing views (SS CAN use occluded-view info)"
                           if (dR > 0.05 and summ["n_B_gt_A_backrecall_by_0.05"] > len(Q) // 2)
                           else "NULL: back NOT recovered (SS ignores occluded-view info)")
    summ["per_asset"] = rows
    json.dump(summ, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)

    print("\n=== BACK-FACE RECOVERY SUMMARY ===", flush=True)
    print(f"  alignment perm={perm} signs={signs} (meanIoU {best_mean:.3f} vs id {ident_mean:.3f})")
    print(f"  qualifying assets: {len(Q)}/{len(rows)}  -> {summ['qualifying_shas']}", flush=True)
    if Q:
        print(f"  mean back_recall  A(front)={summ['mean_back_recall']['A']:.3f}  "
              f"B(all4)={summ['mean_back_recall']['B']:.3f}  "
              f"C(bestback)={summ['mean_back_recall']['C']:.3f}", flush=True)
        print(f"  mean back_IoU     A={summ['mean_back_iou']['A']:.3f}  "
              f"B={summ['mean_back_iou']['B']:.3f}  C={summ['mean_back_iou']['C']:.3f}", flush=True)
        print(f"  mean overall_IoU  A={summ['mean_overall_iou']['A']:.3f}  "
              f"B={summ['mean_overall_iou']['B']:.3f}  C={summ['mean_overall_iou']['C']:.3f}", flush=True)
        print(f"  KEY delta back_recall (B-A) = {summ['mean_back_recall_delta_B_minus_A']:+.3f}  "
              f"(B>A by .05 on {summ['n_B_gt_A_backrecall_by_0.05']}/{len(Q)})", flush=True)
        print(f"      delta back_IoU    (B-A) = {summ['mean_back_iou_delta_B_minus_A']:+.3f}  "
              f"(B>A by .05 on {summ['n_B_gt_A_backiou_by_0.05']}/{len(Q)})", flush=True)
        print(f"  VERDICT: {summ['verdict']}", flush=True)
    print(f"  wrote {os.path.join(OUT_DIR,'summary.json')}", flush=True)

    # --- PNG grid: rows=qualifying; cols=[front img | back-view img | A(back-hi) | B(back-hi) | GT(back-hi)] ---
    if grid_rows:
        cell = 224; occ_w = RES * 3 * 2; occ_h = RES * 2
        cols_w = [cell, cell, occ_w, occ_w, occ_w]
        gap = 10; hdr = 40; rowh = max(cell, occ_h) + 26
        xs = [0]
        for w in cols_w[:-1]:
            xs.append(xs[-1] + w + gap)
        W = xs[-1] + cols_w[-1] + 6
        H = hdr + rowh * len(grid_rows)
        grid = Image.new("RGB", (W, H), (250, 250, 250))
        d = ImageDraw.Draw(grid)
        tfont = ImageFont.truetype(FONT_PATH, 18); rfont = ImageFont.truetype(FONT_PATH, 15)
        labels = ["front view", "back-seeing view", "A front-only (back=green)",
                  "B all-4 (back=green)", "GT (back=red)"]
        for x, lab in zip(xs, labels):
            d.text((x + 4, 12), lab, fill=(10, 10, 10), font=tfont)
        for ri, g in enumerate(grid_rows):
            y = hdr + ri * rowh
            grid.paste(load_view_img(g["rd"], g["front"], cell), (xs[0], y))
            grid.paste(load_view_img(g["rd"], g["best_back"], cell), (xs[1], y))
            grid.paste(colored_proj(g["occA"], g["recov"], (40, 210, 60)), (xs[2], y + (cell - occ_h) // 2))
            grid.paste(colored_proj(g["occB"], g["recov"], (40, 210, 60)), (xs[3], y + (cell - occ_h) // 2))
            grid.paste(colored_proj(g["gt"], g["recov"], (230, 50, 50)), (xs[4], y + (cell - occ_h) // 2))
            rr = g["row"]
            d.text((xs[0] + 3, y + cell - 20), f"{rr['sha8']} v{g['front']}", fill=(0, 0, 0), font=rfont)
            d.text((xs[1] + 3, y + cell - 20), f"v{g['best_back']}", fill=(0, 0, 0), font=rfont)
            d.text((xs[2] + 3, y + occ_h + 4),
                   f"backR {rr['back_recall']['A']:.2f} IoU {rr['overall_iou']['A']:.2f}",
                   fill=(120, 0, 0), font=rfont)
            d.text((xs[3] + 3, y + occ_h + 4),
                   f"backR {rr['back_recall']['B']:.2f} IoU {rr['overall_iou']['B']:.2f}",
                   fill=(0, 100, 0), font=rfont)
            d.text((xs[4] + 3, y + occ_h + 4),
                   f"|back|/surf {rr['back_frac']:.2f}", fill=(0, 0, 0), font=rfont)
        out_png = os.path.join(OUT_DIR, "grid.png")
        grid.save(out_png)
        print(f"[grid] saved {out_png} ({len(grid_rows)} rows)", flush=True)
    print("BACKFACE_DONE", flush=True)


if __name__ == "__main__":
    main()
