"""FULL pipeline export (zero GT): image cond → SS flow (64³ occupancy) → max_pool→32³ coords
→ shape flow → shape-conditioned tex flow → textured GLB. Also exports per-stage viz GLBs:
  ssvox_<sha>.glb   — generated 64³ occupancy as voxel cubes (surface voxels only)
  shapemesh_<sha>.glb — generated shape mesh, untextured
  full_<sha>.glb    — final textured asset
Prints SS-vs-GT occupancy IoU + token counts for sanity."""
import os, sys, json, argparse
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("FUSED_MODULATE", "1")
os.environ.setdefault("EVAL_COND_ROOT", "/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/v22_heldout")
sys.path.insert(0, "/fsx/sfr/weikaih/3dgen/model/BLIP3o")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from trellis2_blip3o import _paths  # noqa
from trellis2 import models as t2models  # type: ignore
from trellis2_blip3o.tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                         build_sc_vae_tex_decoder_frozen,
                                         load_norm_stats, SHAPE_SLAT_CONFIG_PATH,
                                         TEX_SLAT_CONFIG_PATH, SS_FLOW_CONFIG_PATH,
                                         DEFAULT_SS_FLOW)
from benchmarks.checkpoint import load_connector, load_state_dict
import scripts.eval_fusion_v22 as EV
from scripts.eval_fusion_v22 import (load_flow_and_connector, build_cond, sample_shape,
                                     good_view_b, input_image)
from scripts.eval_tex_v22 import load_tex_flow, sample_tex
from scripts.export_glb_v22 import build_mw, export_glb

CR = os.environ["EVAL_COND_ROOT"]
SSDEC = ("/fsx/home/weikai.huang/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/"
         "snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16")


def load_ss_flow(ckpt_dir, use_ema=True):
    flow = t2models.from_pretrained(DEFAULT_SS_FLOW)
    sd = load_state_dict(ckpt_dir, use_ema=use_ema)
    fsd = {k[len("ss_flow."):]: v for k, v in sd.items() if k.startswith("ss_flow.")}
    fsd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in fsd.items()}
    assert fsd, "no ss_flow keys"
    missing, _ = flow.load_state_dict(fsd, strict=False)
    conn, dve, ckcfg = load_connector(ckpt_dir, state=sd,
                                      vlm_hidden_dim=2048,
                                      cond_dim=flow.cond_channels)
    assert len(missing) == 0, f"ss flow load missing {len(missing)} keys!"
    print(f"[load-ss] flow {len(fsd)} (missing={len(missing)}), "
          f"conn={ckcfg['cond_adapter']}, pos_stamp={ckcfg.get('cond_pos_stamp', False)}",
          flush=True)
    return flow.cuda().eval(), conn.cuda().eval().float(), (dve.cuda() if dve is not None else None)


@torch.no_grad()
def sample_ss(flow, cond, uncond, steps=25, cfg=3.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    reso, C = flow.resolution, flow.in_channels
    x = torch.randn(1, C, reso, reso, reso, generator=g, device="cuda")
    ts = np.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t, tp = float(ts[i]), float(ts[i + 1])
        tt = torch.tensor([t * 1000.0], device="cuda")
        vp = flow(x, tt, cond).float()
        vn = flow(x, tt, uncond).float()
        x = x - (t - tp) * (cfg * vp + (1 - cfg) * vn)
    return x


def voxel_cubes_glb(occ64, path):
    """occ64: (64,64,64) bool numpy → surface voxel cubes GLB."""
    occ = occ64.astype(bool)
    pad = np.pad(occ, 1)
    nb = (pad[:-2, 1:-1, 1:-1] & pad[2:, 1:-1, 1:-1] & pad[1:-1, :-2, 1:-1]
          & pad[1:-1, 2:, 1:-1] & pad[1:-1, 1:-1, :-2] & pad[1:-1, 1:-1, 2:])
    surface = occ & ~nb
    idx = np.argwhere(surface)
    centers = (idx + 0.5) / 64.0 - 0.5
    centers = np.stack([centers[:, 0], centers[:, 2], -centers[:, 1]], 1)  # match to_glb y/z swap
    mesh = trimesh.voxel.ops.multibox(centers, pitch=1.0 / 64)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[90, 140, 220, 255], metallicFactor=0.0, roughnessFactor=0.9))
    mesh.export(path)
    print(f"[glb] {path} ({os.path.getsize(path)/1e6:.1f}MB, {len(idx)} surface vox)", flush=True)


def shape_mesh_glb(mesh_t, path):
    v = mesh_t.vertices.detach().cpu().numpy().copy()
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)  # match to_glb y/z swap
    f = mesh_t.faces.detach().cpu().numpy()
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[200, 200, 205, 255], metallicFactor=0.0, roughnessFactor=0.6))
    m.export(path)
    print(f"[glb] {path} ({os.path.getsize(path)/1e6:.1f}MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mani", default="/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v4_vlm_filtered/heldout_eval.jsonl")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--out", default="runs/cache_logs/glb_fullchain")
    ap.add_argument("--qwen_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for l in open(a.mani)][a.skip:a.skip + a.n]

    ssflow, ssconn, ssdve = load_ss_flow("runs/fusion_ss_v22gv/checkpoint-16000")
    ssdec = t2models.from_pretrained(SSDEC).cuda().eval()
    flow, conn, dve = load_flow_and_connector("runs/fusion_shape_v22gv/checkpoint-16000")
    tflow, tconn, tdve = load_tex_flow("runs/fusion_tex_v22gv/checkpoint-16000")
    sdec = build_sc_vae_shape_decoder_frozen().cuda().eval()
    tdec = build_sc_vae_tex_decoder_frozen().cuda().eval()
    ss_norm = load_norm_stats(SS_FLOW_CONFIG_PATH, "normalization")
    sn = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
    tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    tsn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    sm_, ssd_ = sn["mean"].cuda(), sn["std"].cuda()
    tm_, tsd_ = tn["mean"].cuda(), tn["std"].cuda()
    xm_, xsd_ = tsn["mean"].cuda(), tsn["std"].cuda()
    ssm = ss_norm["mean"].cuda().view(1, -1, 1, 1, 1) if ss_norm else 0
    sss = ss_norm["std"].cuda().view(1, -1, 1, 1, 1) if ss_norm else 1

    for r in recs:
        sha = r["sha256"]
        EV._VIEW_FILE = good_view_b(sha)
        ed = os.path.join(CR, sha[:2], sha)
        input_image(r["renders_dir"]).save(os.path.join(a.out, f"input_{sha[:12]}.webp"))
        # --- SS stage ---
        qo = a.qwen_only
        pref = "qo_" if qo else ""
        c_ss, u_ss = build_cond(ssconn, ssdve, ed, qwen_only=qo)
        z = sample_ss(ssflow, c_ss, u_ss)
        z = z * sss + ssm
        occ = (ssdec(z) > 0)[0, 0]                                   # (64,64,64) bool
        gt_occ = None
        try:
            gz = torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None]
            gt_occ = (ssdec(gz) > 0)[0, 0]
            iou = (occ & gt_occ).sum().item() / max(1, (occ | gt_occ).sum().item())
        except Exception:
            iou = -1
        voxel_cubes_glb(occ.cpu().numpy(), os.path.join(a.out, f"{pref}ssvox_{sha[:12]}.glb"))
        # 64³ → 32³ coords
        occ32 = F.max_pool3d(occ.float()[None, None], 2, 2) > 0.5
        cc = torch.argwhere(occ32[0, 0]).int()
        coords = torch.cat([torch.zeros(cc.shape[0], 1, dtype=torch.int32, device="cuda"), cc], 1).cpu()
        print(f"[ss] {sha[:8]} vox64={int(occ.sum())} IoU_gt={iou:.3f} coords32={len(cc)}", flush=True)
        # --- shape stage (on GENERATED coords) ---
        c_sh, u_sh = build_cond(conn, dve, ed, qwen_only=qo)
        slat = sample_shape(flow, c_sh, u_sh, coords)
        gen_shape_raw = slat.feats.float() * ssd_ + sm_
        from trellis2.modules import sparse as sp  # type: ignore
        sdec.set_resolution(512)
        meshes, _ = sdec(sp.SparseTensor(gen_shape_raw, coords.cuda()), return_subs=True)
        meshes[0].simplify(300000)
        shape_mesh_glb(meshes[0], os.path.join(a.out, f"{pref}shapemesh_{sha[:12]}.glb"))
        # --- tex stage (on generated shape) ---
        c_tx, u_tx = build_cond(tconn, tdve, ed, qwen_only=qo)
        tex_n = sample_tex(tflow, c_tx, u_tx, coords, (gen_shape_raw - xm_) / xsd_)
        gen_tex_raw = tex_n.cuda() * tsd_ + tm_
        export_glb(build_mw(sdec, tdec, coords, gen_shape_raw, gen_tex_raw),
                   os.path.join(a.out, f"{pref}full_{sha[:12]}.glb"))
        print(f"[fullchain] {sha[:12]} done", flush=True)
    print("[fullchain] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
