"""Build a self-contained HTML report (base64-embedded figures) of the molmo3-3d
VLM-conditioning exploration — English, for meeting presentation. One file, opens anywhere."""
import base64, os

R = "/weka/oe-training-default/weikaih/world_explore/trellis2_blip3o"
G = f"{R}/runs/eval_grids"


def img(path, max_w=980):
    if not os.path.exists(path):
        return f'<div class="missing">[missing: {os.path.basename(path)}]</div>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img style="max-width:{max_w}px" src="data:image/png;base64,{b}"/>'


F = {
    "arch_cascade": f"{G}/arch_cascade.png",
    "arch_evo": f"{G}/arch_evolution.png",
    "fusion_final": f"{G}/fusion_3000_final.png",
    "fusion_abl": f"{G}/fusion_3000_ablations.png",
    "s3": f"{G}/s3_3task_final.png",
    "im_res": f"{G}/im_resolution_test.png",
    "warrior": f"{G}/warrior_indist_vs_ood.png",
    "coverage": f"{G}/train_view_coverage.png",
    "im_v2": f"{G}/im_v2_full.png",
}

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>molmo3-3d · VLM-Native 3D Conditioning</title>
<style>
:root{{--fg:#1a1a1a;--mut:#666;--bd:#e2e2e2;--ac:#d4571c;--gd:#0aa05a;--bl:#3b7bf5}}
*{{box-sizing:border-box}}
body{{font:16px/1.7 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--fg);max-width:1040px;margin:0 auto;padding:48px 28px 120px}}
h1{{font-size:30px;margin:0 0 4px;letter-spacing:-.5px}}
h2{{font-size:23px;margin:52px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--ac)}}
h3{{font-size:18px;margin:30px 0 8px}}
.sub{{color:var(--mut);font-size:15px;margin-bottom:8px}}
.tag{{display:inline-block;background:#f4f4f4;border:1px solid var(--bd);border-radius:5px;
padding:1px 9px;font-size:13px;color:#444;margin:2px 4px 2px 0}}
figure{{margin:18px 0 26px;text-align:center}}
figure img{{width:100%;border:1px solid var(--bd);border-radius:8px}}
figcaption{{color:var(--mut);font-size:13.5px;margin-top:8px;text-align:left}}
.key{{background:#fff8f3;border-left:4px solid var(--ac);padding:12px 16px;margin:16px 0;border-radius:0 6px 6px 0}}
.win{{background:#f1faf4;border-left:4px solid var(--gd)}}
.warn{{background:#fbf7ec;border-left:4px solid #d9a300}}
ul{{margin:8px 0}}li{{margin:5px 0}}
code{{background:#f4f4f4;padding:1px 5px;border-radius:4px;font-size:13.5px}}
.tl{{border-left:2px solid var(--bd);margin-left:8px;padding-left:22px}}
.tl .step{{position:relative;margin:0 0 22px}}
.tl .step::before{{content:"";position:absolute;left:-29px;top:6px;width:11px;height:11px;
border-radius:50%;background:var(--ac);border:2px solid #fff;box-shadow:0 0 0 2px var(--ac)}}
.muted{{color:var(--mut)}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14.5px}}
th,td{{border:1px solid var(--bd);padding:7px 11px;text-align:left;vertical-align:top}}
th{{background:#fafafa}}
.foot{{margin-top:60px;color:var(--mut);font-size:13px;border-top:1px solid var(--bd);padding-top:14px}}
</style></head><body>

<h1>molmo3-3d &mdash; Conditioning 3D Generation directly with a VLM</h1>
<div class="sub">Exploration summary · TRELLIS.2-4B × Qwen3.5-2B · June 2026</div>
<span class="tag">condition swap</span><span class="tag">cache + split-training infra</span>
<span class="tag">DINO × Qwen fusion</span><span class="tag">multi-task I1/IM/T</span><span class="tag">multi-view diagnosis</span>

<h2>1 · Background &amp; Goal</h2>
<p>The backbone is <b>TRELLIS.2-4B</b>: a cascade 3D generator
(SS sparse-structure flow → shape SLAT@512 → texture SLAT@512), natively conditioned on
<b>DINOv3</b> visual features through a <b>cross-attention DiT</b> (the 3D latent is the main
sequence; the condition is read as K/V by cross-attention).</p>
<p>Goal: replace the condition with representations from <b>Qwen3.5-2B (a VLM)</b>, to drive
3D generation from a single unified interface — <b>{{single-image, multi-image, text}} → 3D</b> —
and lay the groundwork for 3D editing / conversational generation.</p>

<h2>2 · Model Architecture</h2>
<figure>{img(F['arch_cascade'])}<figcaption>
The cascade backbone. The condition enters every DiT block once via cross-attention (as K/V).
Key property: the condition is a <b>read-only dictionary</b> — it does not co-evolve with the
3D latent across layers (contrast with MMDiT/joint-attention, §6).</figcaption></figure>
<figure>{img(F['arch_evo'])}<figcaption>
How we evolved the conditioning. (1) native DINOv3; (2) Qwen-only via a small connector (V3);
(3) the current best: concatenate raw DINOv3 tokens with connector(Qwen) into one cross-attn,
plus DINO-dropout and (for multi-image) a per-view embedding.</figcaption></figure>

<h2>3 · Exploration Timeline</h2>
<div class="tl">
<div class="step"><b>V3 · Condition swap</b><br>Qwen last-layer hidden → small connector → replace
DINOv3. Works, but geometric alignment is limited (Qwen's last layer is semantic, weak on geometry).</div>
<div class="step"><b>Infra · VLM cache + split training</b><br>Precompute the frozen VLM hiddens to
disk (no VLM at train time); train the three cascade components separately. <b>≈5× throughput.</b></div>
<div class="step"><b>Fusion · DINO × Qwen in one cross-attn</b><br>
cond = [raw DINOv3 tokens ; connector(Qwen)] concatenated into the same cross-attn, with a
DINO-dropout curriculum. <b>Beats Qwen-only across the board; reaches teacher-level fidelity.</b></div>
<div class="step"><b>Multi-task · I1 + IM + T</b><br>One set of weights handles three input types
(single-image / multi-image / text). I1 fidelity preserved; IM works; T (text→3D) path alive but weak.</div>
<div class="step"><b>Multi-view deep dive</b><br>Diagnosed IM "mushiness": ruled out resolution;
isolated narrow training view-coverage + OOD viewpoints; in-distribution multi-view is in fact effective.</div>
</div>

<h2>4 · Key Results</h2>

<h3>4.1 Fusion reaches teacher-level fidelity</h3>
<div class="key win"><b>Result:</b> feeding raw DINOv3 tokens and Qwen together into the same
cross-attn (DINO bypasses the connector, keeping its native distribution) beats Qwen-only at
half the steps — the dome becomes solid, the L-shaped lot reappears, texture hallucinations vanish.
Stable at both CFG settings.</div>
<figure>{img(F['fusion_final'])}<figcaption>
Fusion @3000 final. Columns: input | TEACHER (DINOv3) | Qwen-only (S2) | Fusion | Fusion (CFG5) |
Qwen-only ablation. Fusion matches TEACHER fidelity; the Qwen-only column shows the dome opening a
hole, camo texture on the teapot, etc.</figcaption></figure>

<h3>4.2 Ablation: who carries the quality?</h3>
<div class="key"><b>On single-image:</b> Fusion ≈ DINO-only (Qwen's marginal ≈ 0), while
Qwen-only ≈ the Qwen baseline (the path stays alive, not damaged by fusion training). So single-image
fidelity is carried by DINO; Qwen is a "harmless passenger" on single-image and earns its keep on
multi-image / text tasks.</div>
<figure>{img(F['fusion_abl'])}<figcaption>
Segment ablation. Columns: input | Fusion (both segments) | DINO-only | Qwen-only | TEACHER.
Fusion ≈ DINO-only on single-image; Qwen-only still produces an object (the path is alive).</figcaption></figure>

<h3>4.3 Multi-task: one model, three input types</h3>
<figure>{img(F['s3'])}<figcaption>
Stage-3, three tasks. Top: I1 (single-image, fidelity preserved) / middle: IM (multi-image) /
bottom: T (text → 3D). T produces coherent 3D from text alone (non-trivial), but text-to-geometry
alignment is still weak (frozen Qwen + only last-20 tuned + text weight 0.2).</figcaption></figure>

<h2>5 · Multi-view (IM) — see the dedicated report</h2>
<p>The multi-image / multi-view investigation (why IM looked "mushy", and what actually causes it)
is written up separately so it can stand on its own: <b>"Multi-view (IM) Diagnosis"</b>
(on the hub dashboard). In short: it is <b>not</b> resolution, and <b>not</b> "multi-view is broken" —
the bottleneck is training view-coverage / OOD robustness (a data issue). In-distribution, multi-view
helps.</p>

<h2>6 · Where this sits in the literature</h2>
<p>Three "frozen-VLM → generative model" works point to the same signal:</p>
<table>
<tr><th>Work</th><th>How the condition is injected</th></tr>
<tr><td>MolmoAct2</td><td>Per-layer KV-cache: the expert reads the VLM's K/V at the same depth, every layer</td></tr>
<tr><td>EVA-01</td><td>Concatenated sequence + global self-attention (3D tokens see vision tokens at every layer)</td></tr>
<tr><td>Qwen-RobotWorld</td><td>Double-stream MMDiT, layer-wise joint attention (frozen Qwen + VAE, two streams)</td></tr>
<tr><td><b>Ours (current)</b></td><td>Last-layer hidden → small connector → single cross-attn (condition is a read-only dictionary)</td></tr>
</table>
<p class="muted">Common thread: the ones that work let the condition tokens <b>participate in per-layer
joint attention</b>, rather than being queried once by cross-attention. This is the clear direction for
our next architecture upgrade (cost: TRELLIS is a cross-attn DiT; moving to joint-attention is major surgery).</p>

<h2>7 · Conclusions &amp; Next Steps</h2>
<ul>
<li><b>Fusion (DINO × Qwen) is the current best recipe</b>: teacher-level single-image fidelity +
a live Qwen pathway + a unified multi-task interface.</li>
<li><b>End-to-end pure Qwen</b>: marginal on single-image; its value must come from T / IM / editing,
which still need investment.</li>
<li><b>IM is not "broken"</b>: effective in-distribution; the bottleneck is <b>view coverage
(raise max_views) + OOD robustness</b>.</li>
<li><b>T is weak</b>: limited by a frozen Qwen + a small training budget; next is to raise its weight /
unfreeze later Qwen layers.</li>
<li><b>Long term</b>: evolve toward per-layer joint attention (MMDiT / double-stream) so the condition
truly participates in generation.</li>
</ul>

<div class="foot">molmo3-3d · auto-generated exploration summary (images are real evaluation renders,
single seed, pinned views). Infra: VLM-hidden cache + 3-component split training (~5× speedup).</div>
</body></html>"""

out = f"{R}/runs/molmo3_exploration_report.html"
open(out, "w").write(HTML)
# also update the served copy
serve = "/tmp/report_serve/index.html"
if os.path.isdir("/tmp/report_serve"):
    open(serve, "w").write(HTML)
print("wrote", out, "+ served copy" if os.path.isdir("/tmp/report_serve") else "")
