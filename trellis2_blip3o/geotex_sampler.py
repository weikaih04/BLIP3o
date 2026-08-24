"""3-mode inference for the unified geo-tex DiT (docs/UNIFIED_GEOTEX_DIT_DESIGN.md).

Modes (the product definition):
  mesh_only       — geo stream alone ≡ the shape specialist (identical trajectory:
                    the geo lane never reads tex, and G0 certified it bit-exact).
  tex_given_mesh  — t_s≡0, clean shape latent LOCKED (never integrated); geo K/V
                    computed in ONE pass and reused for every tex step + both CFG
                    branches (CFG negates only the tex cond — cascade parity:
                    eval_tex_v22.sample_tex keeps the shape concat in both branches).
  joint           — geo leads on the α-warped grid t_s = t_x/(α−(α−1)t_x) (the MF
                    f_α inverse — the SAME map the A6 band sampler uses, flow_heads
                    sample_timestep_pairs). α=∞ dispatches to the literal cascade
                    (full geo solve → tex_given_mesh): the exact production
                    semantics, not a degenerate 1-step warp.

SAMPLER = TRELLIS.2's OWN, per stream, at the SHIPPED parameters.

Until 2026-08-16 this file used the repo's older eval scripts
(eval_fusion_v22.sample_shape / eval_tex_v22.sample_tex, both from 2026-07-09):
a bare Euler walk on a uniform grid with cfg=3.0 for BOTH streams and no
guidance interval. Those scripts predate this work and were copied without
checking them against the release. They do not match TRELLIS.2-4B's published
pipeline on a single parameter:

  stream   sampler                             steps  strength  rescale  interval    rescale_t
  SS       FlowEulerGuidanceIntervalSampler      12      7.5      0.7    [0.6,1.0]      5.0
  shape    FlowEulerGuidanceIntervalSampler      12      7.5      0.5    [0.6,1.0]      3.0
  tex      FlowEulerGuidanceIntervalSampler      12      1.0      0.0    [0.6,0.9]      3.0
  (ours, before)  bare Euler                     25      3.0      —      all t          1.0

Two of those are not tuning differences but sign errors of intent: texture ran
at 3x guidance where the release uses NONE (strength 1.0 — and Modality Forcing
independently uses cfg 1.0 for its refine pass, runner.py:326), and geometry ran
at 3.0 where the release uses 7.5. Every held-out number reported before
2026-08-16 was measured through that sampler.

This module now ports the release path verbatim from
  pipelines/samplers/flow_euler.py               (t_seq rescale, Euler update)
  pipelines/samplers/classifier_free_guidance_mixin.py  (CFG + CFG-rescale)
  pipelines/samplers/guidance_interval_mixin.py         (strength=1 outside the interval)
with per-stream defaults taken from the released TRELLIS.2-4B pipeline config.
"""
import numpy as np
import torch

import trellis2_blip3o._paths  # noqa: F401
from trellis2.modules import sparse as sp

# ── released TRELLIS.2-4B pipeline params (models--microsoft--TRELLIS.2-4B) ──
SHAPE_PARAMS = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5,
                    guidance_interval=(0.6, 1.0), rescale_t=3.0)
TEX_PARAMS = dict(steps=12, guidance_strength=1.0, guidance_rescale=0.0,
                  guidance_interval=(0.6, 0.9), rescale_t=3.0)
# SS (sparse-structure) stage. Same released-config provenance as the two above;
# it lived only in the eval scripts (benchmarks/run_wild3dgen.py, eval_ss_*.py,
# demo_pipeline.py) until v10's lag scheduler needed the grid at TRAIN time.
# Retyping the constant is how the two arms drift apart, so it lives here now.
SS_PARAMS = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
                 guidance_interval=(0.6, 1.0), rescale_t=5.0)
SIGMA_MIN = 1e-5


def _t_seq(steps: int, rescale_t: float):
    """flow_euler.py:113-115 verbatim: uniform grid, then the shift transform
    t' = r·t / (1 + (r−1)·t). r>1 pushes nodes toward the noisy end."""
    t = np.linspace(1, 0, steps + 1)
    return (rescale_t * t / (1 + (rescale_t - 1) * t)).tolist()


def _pred_to_xstart(x_t, t, pred):
    """flow_euler.py:41."""
    return (1 - SIGMA_MIN) * x_t - (SIGMA_MIN + (1 - SIGMA_MIN) * t) * pred


def _xstart_to_pred(x_t, t, x_0):
    """flow_euler.py:44."""
    return ((1 - SIGMA_MIN) * x_t - x_0) / (SIGMA_MIN + (1 - SIGMA_MIN) * t)


def _guided(v_pos, v_neg, x_t, t, strength, rescale, interval):
    """classifier_free_guidance_mixin.py + guidance_interval_mixin.py verbatim.

    Outside the interval the mixin calls the model with strength=1, i.e. the
    CONDITIONAL prediction alone — not "no CFG blend", the same thing here.
    Tensors are (N, C) rather than (B, C, ...); std is taken over every axis but
    the first, which matches the mixin's `dim=list(range(1, ndim))`."""
    if not (interval[0] <= t <= interval[1]) or strength == 1:
        return v_pos
    pred = strength * v_pos + (1 - strength) * v_neg
    if rescale > 0:
        x0_pos = _pred_to_xstart(x_t, t, v_pos)
        x0_cfg = _pred_to_xstart(x_t, t, pred)
        # ONE SCALAR for the whole latent, not one per voxel. The mixin writes
        # `.std(dim=list(range(1, ndim)))`, but it is called on a SparseTensor
        # whose `.shape` is (B, C) — ndim 2, so dim=[1] — and SparseTensor.std
        # ALSO segment-reduces the token axis (basic.py:267-299). The result is a
        # population std over every voxel AND channel, one number per sample.
        # Copying the expression onto a plain (N, C) tensor silently made dim=[1]
        # mean "per voxel", which flattens the voxel-to-voxel amplitude structure
        # — a different operator entirely, applied on 9 of the shape stream's 12
        # steps. Verified by two independent audits, 2026-08-17.
        gstd = lambda z: (z.pow(2).mean() - z.mean() ** 2).clamp_min(0).sqrt()
        x0 = rescale * (x0_cfg * (gstd(x0_pos) / gstd(x0_cfg))) + (1 - rescale) * x0_cfg
        pred = _xstart_to_pred(x_t, t, x0)
    return pred


def warp_tx(t_s: float, alpha: float) -> float:
    """MF f_α (schedule.py:48-53) VERBATIM: f_α(t) = αt/(1+(α−1)t) ≥ t for α>1,
    i.e. it maps a time to a NOISIER one. The LEADER walks the uniform grid and
    the FOLLOWER's time is warped to lag behind: t_tex = f_α(t_geo).

    BUG FIXED 2026-08-12 (G1 collapse): the first implementation had the roles
    inverted — tex uniform, geo = f_α^{-1}(t_x) — which handed GEO the compressed
    grid. At α=32 that made geo's FIRST Euler step jump t_s 1.00 → 0.43, and the
    resulting garbage geometry is what the tex stream then read. Symptoms matched
    exactly: worse with larger α, and α=∞ unaffected because it dispatches to the
    cascade and never uses this grid.
    """
    return alpha * t_s / (1.0 + (alpha - 1.0) * t_s)


class GeoTexSampler:
    """Per-stream sampling at the released TRELLIS.2-4B parameters.

    `steps`/`cfg` are kept ONLY so existing callers still construct; passing
    them now overrides BOTH streams and is a deliberate deviation from the
    release. Leave them None for the shipped behaviour."""

    def __init__(self, unified, steps=None, cfg=None,
                 shape_params: dict = None, tex_params: dict = None):
        self.m = unified.eval()
        self.shape_p = dict(SHAPE_PARAMS, **(shape_params or {}))
        self.tex_p = dict(TEX_PARAMS, **(tex_params or {}))
        if steps is not None:
            self.shape_p["steps"] = self.tex_p["steps"] = int(steps)
        if cfg is not None:
            self.shape_p["guidance_strength"] = self.tex_p["guidance_strength"] = float(cfg)
        # legacy attributes some callers read
        self.steps = self.shape_p["steps"]
        self.cfg = self.shape_p["guidance_strength"]

    def _grid(self, p=None):
        """The SHAPE stream's node grid (the leader in joint mode)."""
        p = p or self.shape_p
        return np.array(_t_seq(p["steps"], p["rescale_t"]))

    # ── mode ③: mesh-only ───────────────────────────────────────────────────
    @torch.no_grad()
    def sample_mesh_only(self, coords, cond_s, uncond_s, seed: int = 0):
        """Returns the shape latent (norm space) as a SparseTensor — trajectory
        identical to eval_fusion_v22.sample_shape on the same seed."""
        if getattr(self.m, "from_scratch", False):
            # A jointly-trained MMDiT has no standalone geo function to call:
            # mesh-only IS the marginal (tex pinned at its t=1 noise corner, the
            # config P_CORNER2 trains). Dispatching here rather than letting
            # geo_flow's compatibility forward do it keeps the noise on THIS
            # method's seeded generator, so the seed-reproducibility promised
            # above still holds. Single cond stream ⇒ cond_s doubles as cond_x.
            return self.sample_mesh_only_marginal(coords, cond_s, uncond_s,
                                                  cond_s, uncond_s, seed=seed)
        geo = self.m.geo_flow
        p = self.shape_p
        g = torch.Generator(device="cuda").manual_seed(seed)
        N = coords.shape[0]
        x = sp.SparseTensor(torch.randn(N, geo.in_channels, generator=g, device="cuda",
                                        dtype=torch.float32), coords.cuda())
        ts = self._grid(p)
        for i in range(p["steps"]):
            t, tp = float(ts[i]), float(ts[i + 1])
            tt = torch.tensor([t * 1000.0], device="cuda")
            vp = geo(x, tt, cond_s).feats.float()
            vn = geo(x, tt, uncond_s).feats.float()
            v = _guided(vp, vn, x.feats.float(), t, p["guidance_strength"],
                        p["guidance_rescale"], p["guidance_interval"])
            x = x.replace(x.feats - (t - tp) * v)
        return x

    # ── mode ③b: mesh-only MARGINAL (bidirectional models) ──────────────────
    @torch.no_grad()
    def sample_mesh_only_marginal(self, coords, cond_s, uncond_s, cond_x,
                                  uncond_x, seed: int = 0):
        """Mesh-only for a BIDIR-trained model: tex lane pinned at t_x=1 with
        fresh pure noise each step (MF marginal: the other modality is FED
        noise, not dropped — trained at the t_x=1 corner). Only geo integrates.
        The pure-specialist sample_mesh_only remains available for comparison
        (b_gates≈0 ⇒ the two coincide)."""
        geo = self.m.geo_flow
        g = torch.Generator(device="cuda").manual_seed(seed)
        N = coords.shape[0]
        x_s = sp.SparseTensor(torch.randn(N, geo.in_channels, generator=g, device="cuda",
                                          dtype=torch.float32), coords.cuda())
        p = self.shape_p
        tt_x = torch.tensor([1000.0], device="cuda")
        ts = self._grid(p)
        for i in range(p["steps"]):
            t, tp = float(ts[i]), float(ts[i + 1])
            tt_s = torch.tensor([t * 1000.0], device="cuda")
            xn = sp.SparseTensor(torch.randn(N, 32, generator=g, device="cuda",
                                             dtype=torch.float32), coords.cuda())
            v_sp, _ = self.m(x_s, xn, tt_s, tt_x, cond_s, cond_x, tex_concat_cond=x_s)
            v_sn, _ = self.m(x_s, xn, tt_s, tt_x, uncond_s, uncond_x, tex_concat_cond=x_s)
            v_s = _guided(v_sp.feats.float(), v_sn.feats.float(), x_s.feats.float(), t,
                          p["guidance_strength"], p["guidance_rescale"],
                          p["guidance_interval"])
            x_s = x_s.replace(x_s.feats - (t - tp) * v_s)
        return x_s

    # ── mode ②: tex | mesh (exact geometry lock) ────────────────────────────
    @torch.no_grad()
    def sample_tex_given_mesh(self, coords, shape_norm_feats, cond_s, cond_x,
                              uncond_x, seed: int = 0):
        """shape_norm_feats: (N,32) CLEAN shape latent in shape-norm space (≡ the
        tex concat_cond space — stats measured bit-identical). Returns (N,32)
        tex latent feats (tex-norm space).
        BIDIR-safe: t_s≡0 is the corner config where the mask turns geo's tex
        read OFF ⇒ geo is pure-self and time-invariant ⇒ the one-pass K/V cache
        below remains exactly valid (that's what the corner mask is FOR)."""
        N = coords.shape[0]
        cc = sp.SparseTensor(shape_norm_feats.float().cuda(), coords.cuda())
        t0 = torch.tensor([0.0], device="cuda")
        kv, _ = self.m.precompute_geo_kv(cc, t0, cond_s)
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(N, 32, generator=g, device="cuda", dtype=torch.float32)
        p = self.tex_p
        ts = self._grid(p)
        for i in range(p["steps"]):
            t, tp = float(ts[i]), float(ts[i + 1])
            tt = torch.tensor([t * 1000.0], device="cuda")
            xin = sp.SparseTensor(x, coords.cuda())
            vp = self.m.tex_forward_cached(xin, tt, t0, cond_x, cc, kv).feats.float()
            # At the released strength of 1.0 the negative branch is never used
            # (the CFG mixin short-circuits), so skip the forward entirely.
            vn = (vp if p["guidance_strength"] == 1 else
                  self.m.tex_forward_cached(xin, tt, t0, uncond_x, cc, kv).feats.float())
            v = _guided(vp, vn, x, t, p["guidance_strength"],
                        p["guidance_rescale"], p["guidance_interval"])
            x = x - (t - tp) * v
        return x

    # ── mode ①: joint (geo leads by α) ──────────────────────────────────────
    @torch.no_grad()
    def sample_joint(self, coords, cond_s, uncond_s, cond_x, uncond_x,
                     alpha: float = 32.0, seed: int = 0, refine_tex: bool = True):
        """Returns (shape SparseTensor, tex feats). Within a step the tex read
        uses the CURRENT geo state (kv + concat_cond at t_s,i), then geo
        advances — matching the training distribution (cc = x_{t_s}).

        refine_tex (DEFAULT ON, Modality Forcing's `--refine-depth`,
        runner.py:311-326): after the joint rollout, RE-DERIVE the texture from
        the finished geometry with a full-length tex_given_mesh pass. MF makes
        this its demo default and gives the reason — conditioning the follower on
        a fully formed leader instead of a co-evolving noisy one.

        We need it for a sharper reason: the Mobius warp t_x = f_alpha(t_s)
        leaves the texture almost stationary and then drops it in one Euler step.
        At the released 12 steps with alpha=32, the FINAL step covers 90% of the
        texture trajectory (t_x: 1.000 .. 0.897 -> 0). One big flow step is
        effectively a direct x_0 prediction, i.e. the MMSE estimate — measured on
        checkpoint-48000, joint texture had the LOWEST latent MSE (1.043) and the
        LOWEST std, 66% of GT's, against 73% for the cascade and 81% for
        tex|GT-mesh. That is a conditional MEAN, not a sample: best possible MSE,
        visibly washed out. MF has the same arithmetic (their own shipped
        mu=1.1 / alpha=32 / 50 steps puts 66% of the depth trajectory in the last
        step), which is presumably why the refine pass exists at all.

        The joint rollout still earns its keep: it is what lets the texture shape
        the GEOMETRY through the bidirectional attention. Only the texture is
        re-derived."""
        if alpha == float("inf"):
            xs = self.sample_mesh_only(coords, cond_s, uncond_s, seed=seed)
            xt = self.sample_tex_given_mesh(coords, xs.feats, cond_s, cond_x,
                                            uncond_x, seed=seed + 1)
            return xs, xt
        geo = self.m.geo_flow
        g = torch.Generator(device="cuda").manual_seed(seed)
        N = coords.shape[0]
        x_s = sp.SparseTensor(torch.randn(N, geo.in_channels, generator=g, device="cuda",
                                          dtype=torch.float32), coords.cuda())
        x_x = torch.randn(N, 32, generator=g, device="cuda", dtype=torch.float32)
        # geo LEADS on ITS OWN released grid (rescale_t=3 shift); tex's node
        # times are the f_alpha warp OF THAT GRID — MF does the same
        # (schedule.py:93, t_depth = f_alpha(t_rgb) applied AFTER the shift), so
        # tex inherits the leader's shift rather than getting a second one.
        ps, px = self.shape_p, self.tex_p
        ts_s = self._grid(ps)
        ts_x = np.array([warp_tx(float(t), float(alpha)) for t in ts_s])
        # full unified forward per step (one call = both velocities): required
        # under the bidirectional topology (geo reads tex's CURRENT state, so
        # geo K/V cannot be built without tex); for the one-way model this is
        # mathematically identical to the old kv-reuse path.
        for i in range(ps["steps"]):
            tx, txp = float(ts_x[i]), float(ts_x[i + 1])
            tss, tsp = float(ts_s[i]), float(ts_s[i + 1])
            tt_x = torch.tensor([tx * 1000.0], device="cuda")
            tt_s = torch.tensor([tss * 1000.0], device="cuda")
            xin = sp.SparseTensor(x_x, coords.cuda())
            v_sp, v_xp = self.m(x_s, xin, tt_s, tt_x, cond_s, cond_x,
                                tex_concat_cond=x_s)
            v_sn, v_xn = self.m(x_s, xin, tt_s, tt_x, uncond_s, uncond_x,
                                tex_concat_cond=x_s)
            # each stream is guided by ITS OWN released params, at ITS OWN time
            v_x = _guided(v_xp.feats.float(), v_xn.feats.float(), x_x, tx,
                          px["guidance_strength"], px["guidance_rescale"],
                          px["guidance_interval"])
            x_x = x_x - (tx - txp) * v_x
            v_s = _guided(v_sp.feats.float(), v_sn.feats.float(), x_s.feats.float(), tss,
                          ps["guidance_strength"], ps["guidance_rescale"],
                          ps["guidance_interval"])
            x_s = x_s.replace(x_s.feats - (tss - tsp) * v_s)
        if refine_tex:
            # Stage 2 (MF runner.py:311-326): the geometry from stage 1 is final,
            # so re-derive the texture over its OWN full 12-step grid. Fresh seed
            # so this is a new draw rather than a continuation of the collapsed
            # one. Geometry is NOT touched — only the follower is re-sampled.
            x_x = self.sample_tex_given_mesh(coords, x_s.feats, cond_s, cond_x,
                                             uncond_x, seed=seed + 1)
        return x_s, x_x
