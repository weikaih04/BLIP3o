"""Resident tri-modal cascade for the interactive demo.

Same chain the offline evals run (`scripts/trimodal_fullchain_eval.py`,
`scripts/export_glb_fullchain.py`), with two demo-specific properties:

  * EVERYTHING STAYS LOADED. Cold start is minutes (three flows, three decoders, the v2.2
    VLM and DINOv3); a request must not pay that. One process, one GPU, models built once.
  * ONE REQUEST AT A TIME. The flows are stateful under sampling and the sparse decoders
    keep resolution state, so concurrent requests would interleave and corrupt each other.
    `Pipeline.generate` takes a process-wide lock; Gradio's queue is also pinned to
    concurrency 1. The lock is the real guarantee — the queue setting is belt-and-braces.

Cascade (zero GT anywhere):
    cond -> SS flow (64³ occupancy, official sampler) -> maxpool 32³ coords
         -> shape SLAT-512 flow on GENERATED coords
         -> shape-conditioned tex SLAT-512 flow on the GENERATED shape
         -> textured GLB
Optional per-stage GLBs (SS voxel cubes, untextured shape mesh) are cheap and make the
cascade visible.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import _paths  # noqa: F401
from .live_cond import CondPack, LiveCondEncoder, V22_CKPT, build_stage_cond

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the S3 tri-modal 40k family — ONE training run, three stages, 37 000 steps each
SS_CKPT = os.environ.get("DEMO_SS_CKPT", f"{ROOT}/runs/s3_ss_40k/checkpoint-37000")
SHAPE_CKPT = os.environ.get("DEMO_SHAPE_CKPT", f"{ROOT}/runs/s3_shape_40k/checkpoint-37000")
TEX_CKPT = os.environ.get("DEMO_TEX_CKPT", f"{ROOT}/runs/s3_tex_40k/checkpoint-37000")

# official SS sampler (fused_fullchain_eval / eval_im_vs_i1_ss / trimodal_fullchain_eval)
SS_SAMPLER = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
                  guidance_interval=[0.6, 1.0], rescale_t=5.0)
SLAT_STEPS, SLAT_CFG = 25, 3.0


@dataclass
class Result:
    glb: Optional[str] = None
    ss_glb: Optional[str] = None
    shape_glb: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)


def _voxel_cubes_glb(occ64: np.ndarray, path: str) -> str:
    """Generated 64³ occupancy as surface voxel cubes (export_glb_fullchain.voxel_cubes_glb)."""
    import trimesh
    occ = occ64.astype(bool)
    pad = np.pad(occ, 1)
    interior = (pad[:-2, 1:-1, 1:-1] & pad[2:, 1:-1, 1:-1] & pad[1:-1, :-2, 1:-1]
                & pad[1:-1, 2:, 1:-1] & pad[1:-1, 1:-1, :-2] & pad[1:-1, 1:-1, 2:])
    idx = np.argwhere(occ & ~interior)
    if len(idx) == 0:
        raise RuntimeError("empty occupancy")
    c = (idx + 0.5) / 64.0 - 0.5
    c = np.stack([c[:, 0], c[:, 2], -c[:, 1]], 1)          # match to_glb's y/z swap
    mesh = trimesh.voxel.ops.multibox(c, pitch=1.0 / 64)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[90, 140, 220, 255], metallicFactor=0.0, roughnessFactor=0.9))
    mesh.export(path)
    return path


def _shape_mesh_glb(mesh_t, path: str) -> str:
    import trimesh
    v = mesh_t.vertices.detach().cpu().numpy().copy()
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
    m = trimesh.Trimesh(v, mesh_t.faces.detach().cpu().numpy(), process=False)
    m.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[200, 200, 205, 255], metallicFactor=0.0, roughnessFactor=0.6))
    m.export(path)
    return path


class Pipeline:
    """Load once, generate many. Thread-safe by exclusion (see module docstring)."""

    def __init__(self, ss_ckpt: str = SS_CKPT, shape_ckpt: str = SHAPE_CKPT,
                 tex_ckpt: str = TEX_CKPT, vlm: str = V22_CKPT):
        self.ss_ckpt, self.shape_ckpt, self.tex_ckpt, self.vlm = (
            ss_ckpt, shape_ckpt, tex_ckpt, vlm)
        self._lock = threading.Lock()
        self.loaded = False
        self.load_seconds = 0.0

    # ─────────────────────────────── cold start ───────────────────────────────
    def load(self, log: Callable[[str], None] = print) -> None:
        if self.loaded:
            return
        t0 = time.time()
        from trellis2 import models as t2models  # type: ignore
        from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # type: ignore
        from .tr2_modules import (build_sc_vae_shape_decoder_frozen,
                                  build_sc_vae_tex_decoder_frozen, load_norm_stats,
                                  SHAPE_SLAT_CONFIG_PATH, TEX_SLAT_CONFIG_PATH,
                                  SS_FLOW_CONFIG_PATH)
        import sys
        sys.path.insert(0, ROOT)
        from scripts.export_glb_fullchain import load_ss_flow, SSDEC
        from scripts.eval_fusion_v22 import load_flow_and_connector
        from scripts.eval_tex_v22 import load_tex_flow

        log("[load] v2.2 VLM + DINOv3 (live conditioning) ...")
        self.enc = LiveCondEncoder(self.vlm)
        log(f"[load] SS flow {self.ss_ckpt} ...")
        # load_ss_flow reads cond_adapter/cond_pos_stamp from the checkpoint config, so the
        # connector architecture is never guessed from the current training default.
        self.ss_flow, self.ss_conn, self.ss_dve = load_ss_flow(self.ss_ckpt)
        self.ss_sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        self.ss_dec = t2models.from_pretrained(SSDEC).cuda().eval()
        log(f"[load] shape flow {self.shape_ckpt} ...")
        self.sh_flow, self.sh_conn, self.sh_dve = load_flow_and_connector(self.shape_ckpt)
        log(f"[load] tex flow {self.tex_ckpt} ...")
        self.tx_flow, self.tx_conn, self.tx_dve = load_tex_flow(self.tex_ckpt)
        log("[load] SC-VAE shape + tex decoders ...")
        self.shape_dec = build_sc_vae_shape_decoder_frozen().cuda().eval()
        self.tex_dec = build_sc_vae_tex_decoder_frozen().cuda().eval()

        ssn = load_norm_stats(SS_FLOW_CONFIG_PATH, "normalization")
        sn = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
        tn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
        tsn = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
        self.sm, self.ssd = sn["mean"].cuda(), sn["std"].cuda()
        self.tm, self.tsd = tn["mean"].cuda(), tn["std"].cuda()
        self.xm, self.xsd = tsn["mean"].cuda(), tsn["std"].cuda()
        self.ss_mean = ssn["mean"].cuda().view(1, -1, 1, 1, 1) if ssn else 0
        self.ss_std = ssn["std"].cuda().view(1, -1, 1, 1, 1) if ssn else 1

        self.loaded = True
        self.load_seconds = time.time() - t0
        log(f"[load] cascade resident in {self.load_seconds:.0f}s")

    # ─────────────────────────────── conditioning ───────────────────────────────
    def encode(self, modality: str, **kw) -> CondPack:
        if modality == "i1":
            return self.enc.encode_image(kw["image"], remove_bg=kw.get("remove_bg", True))
        if modality == "im":
            return self.enc.encode_views(kw["images"], remove_bg=kw.get("remove_bg", True))
        if modality == "t":
            return self.enc.encode_text(kw["text"], template=kw.get("template", 0))
        raise ValueError(f"unknown modality {modality!r}")

    # ─────────────────────────────── the cascade ───────────────────────────────
    @torch.no_grad()
    def _sample_ss(self, cond, uncond, seed: int, guidance: float, steps: int):
        cfg = dict(SS_SAMPLER, guidance_strength=guidance, steps=steps)
        g = torch.Generator(device="cuda").manual_seed(seed)
        r, c = self.ss_flow.resolution, self.ss_flow.in_channels
        noise = torch.randn(1, c, r, r, r, generator=g, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.ss_sampler.sample(self.ss_flow, noise, cond=cond, neg_cond=uncond,
                                       verbose=False, **cfg).samples
        return (self.ss_dec(z * self.ss_std + self.ss_mean) > 0)[0, 0]

    def generate(self, pack: CondPack, out_dir: str, tag: str, *, seed: int = 0,
                 ss_guidance: float = SS_SAMPLER["guidance_strength"],
                 ss_steps: int = SS_SAMPLER["steps"],
                 slat_cfg: float = SLAT_CFG, slat_steps: int = SLAT_STEPS,
                 want_stages: bool = True,
                 progress: Callable[[float, str], None] = lambda f, m: None) -> Result:
        from trellis2.modules import sparse as sp  # type: ignore
        import sys
        sys.path.insert(0, ROOT)
        from scripts.eval_fusion_v22 import sample_shape
        from scripts.eval_tex_v22 import sample_tex
        from scripts.export_glb_v22 import build_mw, export_glb

        os.makedirs(out_dir, exist_ok=True)
        res = Result()
        # ONE request at a time — see module docstring.
        with self._lock:
            torch.cuda.empty_cache()
            t0 = time.time()

            # ── stage 1: structured sparse occupancy @64³ ──
            progress(0.10, "structure (SS flow, 64³ occupancy)")
            c, u = build_stage_cond(self.ss_conn, self.ss_dve, pack)
            res.stats["cond_tokens"] = int(c.shape[1])
            occ = self._sample_ss(c, u, seed, ss_guidance, ss_steps)
            res.timings["ss"] = time.time() - t0
            n_vox = int(occ.sum())
            if n_vox == 0:
                raise RuntimeError(
                    "the structure stage produced an empty object. Try another seed, a "
                    "lower SS guidance, or a cleaner / more object-centred input.")
            res.stats["voxels_64"] = n_vox
            if want_stages:
                res.ss_glb = _voxel_cubes_glb(occ.cpu().numpy(),
                                              os.path.join(out_dir, f"{tag}_ss.glb"))

            # 64³ -> 32³ coords for the SLAT stages
            occ32 = F.max_pool3d(occ.float()[None, None], 2, 2) > 0.5
            cc = torch.argwhere(occ32[0, 0]).int()
            coords = torch.cat([torch.zeros(cc.shape[0], 1, dtype=torch.int32,
                                            device="cuda"), cc], 1).cpu()
            res.stats["coords_32"] = int(cc.shape[0])

            # ── stage 2: shape SLAT-512 on the GENERATED coords ──
            t1 = time.time()
            progress(0.40, "shape (SLAT-512 flow on generated coords)")
            c, u = build_stage_cond(self.sh_conn, self.sh_dve, pack)
            slat = sample_shape(self.sh_flow, c, u, coords, steps=slat_steps,
                                cfg=slat_cfg, seed=seed)
            shape_raw = slat.feats.float() * self.ssd + self.sm
            res.timings["shape"] = time.time() - t1
            if want_stages:
                t_ = time.time()
                self.shape_dec.set_resolution(512)
                meshes, _ = self.shape_dec(sp.SparseTensor(shape_raw, coords.cuda()),
                                           return_subs=True)
                meshes[0].simplify(300000)
                res.shape_glb = _shape_mesh_glb(meshes[0],
                                                os.path.join(out_dir, f"{tag}_shape.glb"))
                res.timings["shape_preview"] = time.time() - t_

            # ── stage 3: shape-conditioned texture SLAT-512 ──
            t2 = time.time()
            progress(0.70, "texture (shape-conditioned SLAT-512 flow)")
            c, u = build_stage_cond(self.tx_conn, self.tx_dve, pack)
            tex_n = sample_tex(self.tx_flow, c, u, coords,
                               (shape_raw - self.xm) / self.xsd,
                               steps=slat_steps, cfg=slat_cfg, seed=seed)
            tex_raw = tex_n.cuda() * self.tsd + self.tm
            res.timings["tex"] = time.time() - t2

            # ── decode + textured GLB ──
            t3 = time.time()
            progress(0.85, "decoding mesh + baking texture")
            mw = build_mw(self.shape_dec, self.tex_dec, coords, shape_raw, tex_raw)
            res.glb = os.path.join(out_dir, f"{tag}.glb")
            export_glb(mw, res.glb)
            res.timings["export"] = time.time() - t3
            res.timings["total"] = time.time() - t0
            progress(1.0, "done")
        return res
