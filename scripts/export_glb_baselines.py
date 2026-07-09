"""Export the 2 missing GLB variants per held-out asset for the 4-way viewer:
  trellis2_<sha>.glb  — pretrained TRELLIS.2 flows (native DINOv3-only conditioning)
  qwenonly_<sha>.glb  — our good-view fusion, v2.2-hidden-only cond (dino_drop regime)
(gt_ and gen_(=qwen+dino) already exported by export_glb_v22.py)
Full gen chain in all cases: generated shape SLAT → shape-conditioned tex SLAT, GT coords."""
import os, sys, json, argparse
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("FUSED_MODULATE", "1")
os.environ.setdefault("EVAL_COND_ROOT", "/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/v22_heldout")
os.environ.setdefault("EVAL_MANI", "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/heldout_eval.jsonl")
sys.path.insert(0, "/fsx/sfr/weikaih/3dgen/model/BLIP3o")
import numpy as np
import torch

from trellis2_blip3o import _paths  # noqa
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen,
                                         build_shape_slat_flow_frozen,
                                         build_tex_slat_flow_frozen,
                                         load_norm_stats, SHAPE_SLAT_CONFIG_PATH,
                                         TEX_SLAT_CONFIG_PATH)
import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import (load_flow_and_connector, build_cond, sample_shape,
                                     good_view_b)
from scripts.eval_tex_v22 import load_tex_flow, sample_tex
from scripts.export_glb_v22 import build_mw, export_glb

CR = os.environ["EVAL_COND_ROOT"]


def dino_only_cond(entry_dir):
    """Native TRELLIS.2 conditioning: raw DINOv3 tokens (1024-d), uncond = zeros."""
    d = np.load(os.path.join(entry_dir, "v000.npz"))
    dh = torch.from_numpy(d["dino_hidden"]).float().cuda()[None]        # (1,Td,1024)
    dm = torch.from_numpy(d["dino_keep_mask"]).bool().cuda()
    cond = dh[:, dm]
    return cond, torch.zeros_like(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="runs/cache_logs/glb_viewer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for l in open(os.environ["EVAL_MANI"])][:a.n]

    # ours (qwen-only mode)
    flow, conn, dve = load_flow_and_connector("runs/fusion_shape_v22gv/checkpoint-16000")
    tflow, tconn, tdve = load_tex_flow("runs/fusion_tex_v22gv/checkpoint-16000")
    # pretrained TRELLIS.2
    t2_shape = build_shape_slat_flow_frozen().cuda().eval()
    t2_tex = build_tex_slat_flow_frozen().cuda().eval()
    sdec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tdec = build_sc_vae_tex_decoder_frozen().cuda().eval()
    sn = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    tsn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    sm_, ssd_ = sn["mean"].cuda(), sn["std"].cuda()
    tm_, tsd_ = tn["mean"].cuda(), tn["std"].cuda()
    xm_, xsd_ = tsn["mean"].cuda(), tsn["std"].cuda()

    for r in recs:
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        gt = np.load(r["shape_latent_512"])
        cx = torch.from_numpy(gt["coords"]).int()
        coords = torch.cat([torch.zeros(cx.shape[0], 1, dtype=torch.int32), cx], 1)
        ed = os.path.join(CR, sha[:2], sha)

        for tag, (sf, tf, cpair) in {
            "trellis2": (t2_shape, t2_tex, dino_only_cond(ed)),
            "qwenonly": (flow, tflow, build_cond(conn, dve, ed, qwen_only=True)),
        }.items():
            cond, uncond = cpair
            print(f"[{tag}] {sha[:8]} cond tokens={cond.shape[1]}", flush=True)
            slat = sample_shape(sf, cond, uncond, coords)
            gen_shape_raw = slat.feats.float() * ssd_ + sm_
            if tag == "qwenonly":
                tcond, tuncond = build_cond(tconn, tdve, ed, qwen_only=True)
            else:
                tcond, tuncond = cond, uncond
            tex_n = sample_tex(tf, tcond, tuncond, coords, (gen_shape_raw - xm_) / xsd_)
            gen_tex_raw = tex_n.cuda() * tsd_ + tm_
            export_glb(build_mw(sdec, tdec, coords, gen_shape_raw, gen_tex_raw),
                       os.path.join(a.out, f"{tag}_{sha[:12]}.glb"))
        print(f"[baselines] {sha[:12]} done", flush=True)
    print("[baselines] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
