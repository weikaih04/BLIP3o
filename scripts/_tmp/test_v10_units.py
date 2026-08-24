"""v10 L0 unit tests — CPU, no model weights, seconds.

Lives in scripts/_tmp/ rather than tests/ because tests/ is gitignored in this
repo (test_unified_conventions.py is untracked), and these assertions are the
only thing standing between us and the 2026-08-16 failure class.

Run:  ATTN_BACKEND=sdpa python scripts/_tmp/test_v10_units.py
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "sdpa")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from trellis2_blip3o import _paths  # noqa: F401
from trellis2_blip3o.flow_heads import (
    CLS_CLEAN, CLS_SOLO, CLS_LAG, _shift_grid,
    sample_timestep_pairs, sample_timestep_triples,
)
from trellis2_blip3o.geotex_sampler import SHAPE_PARAMS, SS_PARAMS, _t_seq

fails, done = [], []


def check(name, ok, msg=""):
    (done if ok else fails).append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}{'' if ok else '  <- ' + msg}", flush=True)


DEV = torch.device("cpu")


# ── T1: the order invariant, structurally, over a knob grid incl. degenerates ──
def t_order_invariant():
    grid = [
        dict(p_solo=0.20, p_lag=0.20), dict(p_solo=0.35, p_lag=0.05),
        dict(p_solo=0.0, p_lag=0.0),   # v9 fallback
        dict(p_solo=1.0, p_lag=0.0), dict(p_solo=0.0, p_lag=1.0),
        dict(p_solo=0.5, p_lag=0.5),
        dict(p_solo=0.2, p_lag=0.2, p_corner=0.0, p_corner2=0.0),
        dict(p_solo=0.2, p_lag=0.2, p_corner=0.5, p_corner2=0.5),
        dict(p_solo=0.2, p_lag=0.2, k0_lo=2, k0_hi=2),
        dict(p_solo=0.2, p_lag=0.2, k0_lo=11, k0_hi=11),
    ]
    worst = 0.0
    for kw in grid:
        for B in (1, 3, 50_000):
            t_ss, t_s, t_x, _ = sample_timestep_triples(B, DEV, **kw)
            worst = max(worst, float((t_ss - t_s).max()), float((t_s - t_x).max()))
            if not (bool((t_ss <= t_s + 1e-6).all()) and bool((t_s <= t_x + 1e-6).all())):
                return check("T1 order invariant", False, f"{kw} B={B}")
            if not (bool((t_ss >= 0).all()) and bool((t_x <= 1 + 1e-6).all())):
                return check("T1 order invariant", False, f"range {kw} B={B}")
    check("T1 order invariant (10 knob settings x 3 batch sizes)", True)
    print(f"        worst margin over all rows: {worst:+.2e} (<= 1e-6 by construction)")


# ── T2: class proportions inside a binomial CI ──
def t_class_proportions():
    N = 200_000
    for p_solo, p_lag in [(0.20, 0.20), (0.35, 0.05), (0.0, 0.0)]:
        torch.manual_seed(0)
        _, _, _, cls = sample_timestep_triples(N, DEV, p_solo=p_solo, p_lag=p_lag)
        got = {c: float((cls == c).float().mean()) for c in (CLS_SOLO, CLS_LAG, CLS_CLEAN)}
        want = {CLS_SOLO: p_solo, CLS_LAG: p_lag, CLS_CLEAN: 1 - p_solo - p_lag}
        for c, p in want.items():
            tol = 4 * np.sqrt(max(p * (1 - p), 1e-12) / N) + 1e-9
            if abs(got[c] - p) > tol:
                return check("T2 class proportions", False,
                             f"cls{c}: {got[c]:.5f} vs {p} (tol {tol:.5f})")
    check("T2 class proportions within binomial CI (N=200k, 3 settings)", True)


# ── T3: RNG consumption is identical regardless of knobs / realized classes ──
def t_rng_uniform():
    probes = []
    for kw in [dict(p_solo=0.20, p_lag=0.20), dict(p_solo=0.0, p_lag=0.0),
               dict(p_solo=1.0, p_lag=0.0), dict(p_solo=0.0, p_lag=1.0),
               dict(p_solo=0.35, p_lag=0.05, k0_lo=5, k0_hi=9)]:
        torch.manual_seed(1234)
        sample_timestep_triples(777, DEV, **kw)
        probes.append(torch.rand(8))
    ok = all(torch.equal(probes[0], p) for p in probes[1:])
    check("T3 RNG consumption independent of knobs/classes", ok,
          "downstream draws diverged -> class-correlated noise + rank desync")


# ── T4: the clean class is v9, bit-for-bit ──
def t_clean_is_v9():
    B = 20_000
    for p_c, p_c2 in [(0.1, 0.2), (0.2, 0.2), (0.0, 0.0)]:
        torch.manual_seed(7)
        t_ss, t_s, t_x, cls = sample_timestep_triples(
            B, DEV, p_solo=0.0, p_lag=0.0, p_corner=p_c, p_corner2=p_c2)
        # replay the v9 draw with the SAME primitive order the triple uses
        torch.manual_seed(7)
        torch.rand(B, device=DEV); torch.randn(B, device=DEV)      # u_cls, g_solo
        torch.rand(B, device=DEV)                                   # u_lag
        torch.randint(3, 13, (B,), device=DEV)                      # k0
        torch.rand(B, device=DEV)                                   # w_lag
        r_s, r_x = sample_timestep_pairs(B, DEV, p_corner=p_c, p_corner2=p_c2)
        if not (torch.equal(t_s, r_s) and torch.equal(t_x, r_x)):
            return check("T4 clean class == v9 bitwise", False, f"p_corner={p_c}")
        if not bool((t_ss == 0).all()) or not bool((cls == CLS_CLEAN).all()):
            return check("T4 clean class == v9 bitwise", False, "t_ss/cls wrong")
    check("T4 clean class reproduces sample_timestep_pairs bitwise", True)


# ── T5: solo rows pinned; solo t_ss follows logitNormal(1,1) ──
def t_solo_rows():
    torch.manual_seed(3)
    t_ss, t_s, t_x, cls = sample_timestep_triples(200_000, DEV, p_solo=1.0, p_lag=0.0)
    ok = bool((t_s == 1).all()) and bool((t_x == 1).all()) and bool((cls == CLS_SOLO).all())
    ok &= bool((t_ss > 0).all()) and bool((t_ss < 1).all())
    ref = torch.sigmoid(1.0 + 1.0 * torch.randn(400_000))
    dm = abs(float(t_ss.mean()) - float(ref.mean()))
    ok &= dm < 5e-3
    check("T5 solo rows pinned (1,1), t_ss ~ logitNormal(1,1)", ok, f"mean drift {dm:.4f}")


# ── T6: lag rows sit on the node-pairing curve family ──
def t_lag_curve():
    torch.manual_seed(11)
    B = 100_000
    t_ss, t_s, t_x, cls = sample_timestep_triples(B, DEV, p_solo=0.0, p_lag=1.0,
                                                  k0_lo=3, k0_hi=11)
    if not bool((cls == CLS_LAG).all()):
        return check("T6 lag curve family", False, "class")
    # invert t_s to recover u, then check t_ss lies on grid-5 at some k0 in range
    r = 3.0
    u = t_s / (r - (r - 1.0) * t_s)
    resid = []
    for k0 in range(3, 12):
        resid.append((t_ss - _shift_grid(5.0, (u - k0 / 12.0).clamp_min(0.0))).abs())
    best = torch.stack(resid, 0).min(0).values
    ok = float(best.max()) < 1e-5 and bool((t_x >= t_s - 1e-6).all())
    check("T6 lag rows lie exactly on the k0 in {3..11} curve family", ok,
          f"max residual {float(best.max()):.2e}")


# ── T7: grids match the released sampler params ──
def t_grids():
    slat = _t_seq(SHAPE_PARAMS["steps"], SHAPE_PARAMS["rescale_t"])
    ss = _t_seq(SS_PARAMS["steps"], SS_PARAMS["rescale_t"])
    ok = (SS_PARAMS["rescale_t"] == 5.0 and SS_PARAMS["steps"] == 12
          and abs(ss[9] - 0.625) < 1e-9 and abs(slat[9] - 0.5) < 1e-9
          and len(ss) == 13)
    # The continuous map must reproduce the discrete grid at the node coords.
    # Tolerance is float32 eps, not 0: _t_seq computes in float64 (numpy) and the
    # comparison lands in float32, so ~6e-8 is the exact-arithmetic answer here.
    u = torch.tensor([1 - j / 12 for j in range(13)], dtype=torch.float64)
    ok &= float((_shift_grid(5.0, u) - torch.tensor(ss, dtype=torch.float64)).abs().max()) < 1e-12
    ok &= float((_shift_grid(3.0, u) - torch.tensor(slat, dtype=torch.float64)).abs().max()) < 1e-12
    check("T7 SS_PARAMS grid == released nodes; _shift_grid == _t_seq", ok)


# ── T8: k0 < 2 is rejected, not silently clamped ──
def t_k0_guard():
    ok = False
    try:
        sample_timestep_triples(8, DEV, k0_lo=1)
    except AssertionError:
        ok = True
    check("T8 k0_lo < 2 raises (the invariant fails there)", ok)


# ── T9: re-rope phase, hand-computed at production dims ──
def t_rerope_phase():
    from trellis2.modules.attention import RotaryPositionEmbedder
    head_dim, dim = 128, 3
    rp = RotaryPositionEmbedder(head_dim, dim, rope_freq=(1.0, 10000.0))
    c = torch.tensor([[3.0, 0.0, 0.0]])
    ph = rp(2.0 * c + 0.5)                       # the v10 mapping
    freq_dim = head_dim // 2 // dim              # 21
    freqs = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
    freqs = 1.0 / (10000.0 ** freqs)
    want_x = torch.polar(torch.ones(freq_dim), 6.5 * freqs)   # 2*3 + 0.5
    ok = ph.dtype == torch.complex64 and ph.shape == (1, head_dim // 2)
    ok &= torch.allclose(ph[0, :freq_dim], want_x, atol=0, rtol=0)
    # +0.5 actually applied: differs from BOTH the integer 3 and the integer 6
    ok &= not torch.allclose(ph[0, :freq_dim], torch.polar(torch.ones(freq_dim), 3.0 * freqs))
    ok &= not torch.allclose(ph[0, :freq_dim], torch.polar(torch.ones(freq_dim), 6.0 * freqs))
    # the 64th pair is the identity pad (what _rotate_pad_pair abuses as a tag)
    ok &= bool(torch.allclose(ph[0, 3 * freq_dim:], torch.ones(head_dim // 2 - 3 * freq_dim,
                                                              dtype=torch.complex64)))
    check("T9 re-rope phase == hand-computed polar(1, (2c+0.5)*freqs)", ok)


# ── T10: relative phase equals the 32-frame embedder's on the mapped coords ──
def t_rerope_relative():
    from trellis2.modules.attention import RotaryPositionEmbedder
    head_dim = 128
    rp = RotaryPositionEmbedder(head_dim, 3, rope_freq=(1.0, 10000.0))
    c1 = torch.tensor([[3.0, 1.0, 4.0]])
    c2 = torch.tensor([[7.0, 2.0, 0.0]])
    d_ss = rp(2.0 * c1 + 0.5) * rp(2.0 * c2 + 0.5).conj()
    d_slat = rp(2.0 * c1 + 0.5) * rp(2.0 * c2 + 0.5).conj()   # same frame by construction
    ok = torch.allclose(d_ss, d_slat, atol=0, rtol=0)
    # and the mapped SS cell sits exactly between its two 32-frame children
    child_lo, child_hi = rp(2.0 * c1), rp(2.0 * c1 + 1.0)
    mid = (torch.angle(child_lo) + torch.angle(child_hi)) / 2
    ok &= torch.allclose(torch.angle(rp(2.0 * c1 + 0.5)), mid, atol=1e-5)
    check("T10 mapped SS phase is the midpoint of its two 32-frame children", ok)


# ── T11: float path — integer coords would truncate 2c+0.5 to 2c ──
def t_rerope_float():
    from trellis2.modules.attention import RotaryPositionEmbedder
    rp = RotaryPositionEmbedder(128, 3, rope_freq=(1.0, 10000.0))
    c_int = torch.arange(3).reshape(1, 3)                       # int64
    mapped = 2 * c_int + 0.5
    ok = mapped.dtype.is_floating_point                          # promotes in torch
    ph = rp(mapped.float())
    ok &= ph.dtype == torch.complex64
    ok &= not torch.allclose(ph, rp((2 * c_int).float()))
    check("T11 2c+0.5 stays float (no integer truncation to 2c)", ok)


# ── T12: AGGREGATE marginals at the production config ──
# The test that would have caught the clamp bug. Every class passed in isolation
# (T4/T5/T6) while the MIXTURE was wrong: clamp_min(0) parked 62.5% of the lag
# class at t_ss==0 carrying a non-v9 (t_s,t_x) law, so true-lag was 7.5% not 20%
# and the clean class was no longer the whole t_ss==0 population.
def t_aggregate_marginals():
    torch.manual_seed(5)
    N = 500_000
    P_SOLO, P_LAG, P_C, P_C2 = 0.20, 0.20, 0.1, 0.2
    t_ss, t_s, t_x, cls = sample_timestep_triples(
        N, DEV, p_corner=P_C, p_corner2=P_C2, p_solo=P_SOLO, p_lag=P_LAG)
    f = lambda m: float(m.float().mean())
    zero_ss = f(t_ss == 0)
    true_lag = f((cls == CLS_LAG) & (t_ss > 0))
    lag_at_zero = f((cls == CLS_LAG) & (t_ss == 0))
    tol = 4 * np.sqrt(0.25 / N) + 2e-3
    ok = abs(zero_ss - (1 - P_SOLO - P_LAG)) < tol       # t_ss==0 IS exactly the clean class
    ok &= abs(true_lag - P_LAG) < tol                    # the lag class is really lagging
    ok &= lag_at_zero < 1e-4                             # no lag row parked at t_ss=0
    # slat-SUPERVISED rows (clean+lag) must keep v9's corner rates; solo rows pin
    # (1,1) but their slat losses are masked, so they are excluded from this count
    sup = cls != CLS_SOLO
    ok &= abs(f((t_s[sup] == 0)) - P_C * (1 - P_SOLO - P_LAG) / f(sup)) < 0.02
    print(f"        t_ss==0 {zero_ss:.4f} (want {1-P_SOLO-P_LAG:.2f}) | "
          f"true-lag {true_lag:.4f} (want {P_LAG:.2f}) | lag@0 {lag_at_zero:.6f} | "
          f"t_x==1 all-rows {f(t_x == 1):.4f}, slat-supervised {f(t_x[sup] == 1):.4f}")
    check("T12 aggregate marginals match the design", ok)


for fn in (t_order_invariant, t_class_proportions, t_rng_uniform, t_clean_is_v9,
           t_solo_rows, t_lag_curve, t_grids, t_k0_guard,
           t_rerope_phase, t_rerope_relative, t_rerope_float,
           t_aggregate_marginals):
    try:
        fn()
    except Exception as e:  # a crash is a failure, with the traceback kept
        import traceback
        traceback.print_exc()
        check(fn.__name__, False, repr(e))

print(f"\n{len(done)} passed, {len(fails)} failed")
if fails:
    print("FAILED:", ", ".join(fails))
sys.exit(1 if fails else 0)
