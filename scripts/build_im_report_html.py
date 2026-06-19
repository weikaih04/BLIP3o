"""Standalone HTML report: multi-image (IM) multi-view diagnosis. Self-contained."""
import base64, os
R = "/weka/oe-training-default/weikaih/world_explore/trellis2_blip3o"
G = f"{R}/runs/eval_grids"


def img(path, max_w=980):
    if not os.path.exists(path):
        return f'<div class="missing">[missing: {os.path.basename(path)}]</div>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img style="max-width:{max_w}px" src="data:image/png;base64,{b}"/>'


F = {
    "im_res": f"{G}/im_resolution_test.png",
    "coverage": f"{G}/train_view_coverage.png",
    "warrior": f"{G}/warrior_indist_vs_ood.png",
    "im_v2": f"{G}/im_v2_full.png",
}

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>molmo3-3d · Multi-view (IM) Diagnosis</title>
<style>
:root{{--fg:#1a1a1a;--mut:#666;--bd:#e2e2e2;--ac:#d4571c;--gd:#0aa05a}}
*{{box-sizing:border-box}}
body{{font:16px/1.7 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--fg);max-width:1040px;margin:0 auto;padding:48px 28px 120px}}
h1{{font-size:28px;margin:0 0 4px;letter-spacing:-.5px}}
h2{{font-size:22px;margin:46px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--ac)}}
.sub{{color:var(--mut);font-size:15px;margin-bottom:8px}}
.tag{{display:inline-block;background:#f4f4f4;border:1px solid var(--bd);border-radius:5px;padding:1px 9px;font-size:13px;color:#444;margin:2px 4px 2px 0}}
figure{{margin:18px 0 26px;text-align:center}}figure img{{width:100%;border:1px solid var(--bd);border-radius:8px}}
figcaption{{color:var(--mut);font-size:13.5px;margin-top:8px;text-align:left}}
.key{{background:#fff8f3;border-left:4px solid var(--ac);padding:12px 16px;margin:16px 0;border-radius:0 6px 6px 0}}
.win{{background:#f1faf4;border-left:4px solid var(--gd)}}.warn{{background:#fbf7ec;border-left:4px solid #d9a300}}
ul{{margin:8px 0}}li{{margin:5px 0}}code{{background:#f4f4f4;padding:1px 5px;border-radius:4px;font-size:13.5px}}
.foot{{margin-top:54px;color:var(--mut);font-size:13px;border-top:1px solid var(--bd);padding-top:14px}}
a{{color:#c24f17}}
</style></head><body>

<h1>Multi-view (IM) Diagnosis</h1>
<div class="sub">molmo3-3d · why multi-image → 3D looked "mushy", and what actually causes it · June 2026</div>
<span class="tag">controlled ablation</span><span class="tag">resolution</span>
<span class="tag">view coverage / OOD</span><span class="tag">view-count</span>

<h2>Question</h2>
<p>In the multi-task model, <b>multi-image (IM) → 3D</b> reconstructions initially looked soft /
"mushy" compared to single-image. Is multi-view fundamentally not working, or is something else
going on? We ran controlled tests to isolate the cause rather than guess.</p>

<h2>1 · It is NOT resolution</h2>
<figure>{img(F['im_res'])}<figcaption>
Same object, same views: 3v@416² (the trained tier) vs 3v@512² (full resolution) vs single@512².
Full resolution barely changes anything → <b>resolution is not the bottleneck</b>.</figcaption></figure>

<h2>2 · The real issue: narrow training view-coverage + OOD</h2>
<div class="key warn"><b>Finding:</b> the cache was built with <code>max_views=4</code>, so IM training
only ever conditioned on views 0–3 of each object. At inference, feeding angles the model never
trained on (e.g. 90° / 180° side and back views) breaks it. Much of the original "multi-view is
worse" impression was this <b>out-of-distribution (OOD) artifact</b>, not multi-view per se.</div>
<figure>{img(F['coverage'])}<figcaption>
Views 0–3 (green, used in training) vs 4 / 8 / 12 (orange, never seen) for 6 objects. Training
view coverage is narrow.</figcaption></figure>

<h2>3 · With in-distribution views, multi-view actually helps</h2>
<div class="key win"><b>Reversed conclusion:</b> using in-distribution views (subsets of 0–3),
multi-view is <b>better</b> than single-image — the warrior's bow and wings are complete with 4
views and broken with 1. The bottleneck is <b>view coverage / OOD robustness (a data issue)</b>,
not "multi-view fusion itself does not work."</div>
<figure>{img(F['warrior'])}<figcaption>
Warrior: single | 4 views [0-3] (in-distribution) | 3 views [0,4,8] (OOD). In-distribution
multi-view is the most complete (bowstring intact); OOD is the messiest.</figcaption></figure>
<figure>{img(F['im_v2'])}<figcaption>
View-count comparison on in-distribution views (8 objects). Columns: 1v | 2v | 3v | 4v (all within
0-3) | [0,4,8] OOD. In-distribution multi-view holds up or improves; the OOD column is consistently
worst.</figcaption></figure>

<h2>4 · Takeaways</h2>
<ul>
<li><b>Not resolution</b> — full-res per-view barely moves the result.</li>
<li><b>Not "multi-view is broken"</b> — in-distribution, more views = equal or better.</li>
<li><b>The fix is on the data side</b>: widen training view coverage (raise <code>max_views</code>)
and improve OOD-viewpoint robustness.</li>
<li>Separately, the literature (MolmoAct2 / EVA-01 / Qwen-RobotWorld) suggests per-layer joint
attention would fuse views more sharply than a single read-once cross-attention — a longer-term
architecture lever.</li>
</ul>

<div class="foot">molmo3-3d · multi-view diagnosis (real evaluation renders, single seed, pinned views).
See the main exploration report for the full project arc.</div>
</body></html>"""

out = f"{R}/runs/molmo3_im_diagnosis.html"
open(out, "w").write(HTML)
print("wrote", out)
