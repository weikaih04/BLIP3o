"""Interactive tri-modal demo for the S3 40k model — image / multi-view / text -> textured GLB.

One checkpoint family (runs/s3_{ss,shape,tex}_40k/checkpoint-37000), three conditioning
modalities trained jointly (single image 0.5 / 4 views 0.3 / text 0.2). Unlike every offline
eval in this repo, conditioning is computed LIVE (see trellis2_blip3o/live_cond.py) so the
model can be pointed at an arbitrary user image or sentence.

    python scripts/demo_app.py --port 7860

The whole cascade stays resident (cold start is minutes) and requests are serialized — see
trellis2_blip3o/demo_pipeline.py.
"""
import os
import sys
import argparse
import random
import time
import traceback

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("FUSED_MODULATE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# scripts/export_glb_v22.py reads this at import time; the demo never touches a cond cache.
os.environ.setdefault("EVAL_COND_ROOT",
                      "/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_heldout")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gradio as gr
from PIL import Image

from trellis2_blip3o.demo_pipeline import (Pipeline, SS_CKPT, SHAPE_CKPT, TEX_CKPT,
                                           SLAT_CFG, SLAT_STEPS, SS_SAMPLER)
from trellis2_blip3o.live_cond import TXT_PROMPTS

OUT_DIR = os.environ.get("DEMO_OUT", os.path.join(ROOT, "runs", "demo_outputs"))
ASSETS = os.path.join(ROOT, "scripts", "demo_assets")
PIPE = Pipeline()

EXAMPLE_CAPTIONS = [
    "A sleek silver luxury sedan featuring a low-slung aerodynamic silhouette, a large "
    "front grille, dark tinted windows, and multi-spoke alloy wheels.",
    "A futuristic assault rifle rendered in a uniform matte grey finish, featuring a sleek, "
    "angular silhouette with a prominent top-mounted optical sight and a long barrel.",
    "A wooden dining chair with a tall slatted back, four straight tapered legs and a "
    "woven rush seat.",
    "A red fire hydrant with a domed cap and two side outlets.",
]

HEADER = """
# BLIP3o-3D · tri-modal 3D generation

One model, three ways in. A single 37 000-step training run conditions the same
structure -> shape -> texture cascade on **a single image**, **2-4 views of one object**, or
**a text description**, and returns a textured GLB you can spin in the browser.

**How it works.** Your input goes through a 3D-tuned Qwen VLM (and DINOv3 for images) to
produce conditioning tokens; a flow model samples a 64<sup>3</sup> occupancy grid, a second
flow fills in shape latents on those voxels, and a third paints shape-conditioned PBR
texture. Nothing is retrieved — every asset is sampled from noise.
"""

CAVEATS = """
### What this model is and is not good at

- **Single image is the strongest mode.** It is half the training mixture and it is the one
  that pins down pose, scale and silhouette. Start here.
- **Text gives you the right *kind* of object, not a specific instance.** With no image,
  pose and scale are unconstrained and the description only has to be satisfied loosely — ask
  for "a wooden dining chair" and you get a plausible wooden dining chair, not *your* chair.
  On held-out assets text scores roughly half the geometric IoU of single-image.
- **Multi-view is not better than single-image on our metrics** (mean SS IoU 0.388 vs 0.407
  over 14 held-out assets). It does demonstrably *use* the extra views — four distinct views
  beat four copies of one view — but the gain does not yet show up as fidelity. Included here
  because it is honest to show it, not because it wins.
- **Clean, object-centred inputs work best.** Training saw single objects rendered on a plain
  dark background. The demo tries to segment your object and re-frame it the same way; the
  "what the model actually saw" panel shows you the result. Busy photos, several objects, or
  heavy occlusion will degrade it.
- Textures are 512-resolution PBR baked from a sparse latent — expect the gist of the
  material, not photographic detail.
"""


# ─────────────────────────────── helpers ───────────────────────────────
def _tag(kind: str) -> str:
    return f"{kind}_{time.strftime('%H%M%S')}_{random.randint(1000, 9999)}"


def _examples():
    if not os.path.isdir(ASSETS):
        return []
    return [os.path.join(ASSETS, f) for f in sorted(os.listdir(ASSETS))
            if f.startswith("ex_") and f.endswith(".png")]


def _mv_example():
    d = os.path.join(ASSETS, "mv")
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]


def _resolve_seed(seed, randomize):
    return random.randint(0, 2 ** 31 - 1) if randomize else int(seed)


def _report(pack, res, seed, mode_note=""):
    t = res.timings
    stage = " · ".join(f"{k} {t[k]:.1f}s" for k in ("ss", "shape", "tex", "export") if k in t)
    lines = [
        f"**Generated in {t['total']:.1f}s** &nbsp;·&nbsp; {stage}",
        f"conditioning: **{pack.n_tokens} tokens** "
        f"(qwen {pack.info.get('n_qwen', '?')}"
        + (f" + dino {pack.info['n_dino']}" if pack.dino is not None else ", no DINO segment")
        + f") &nbsp;·&nbsp; seed **{seed}**",
        f"structure: **{res.stats['voxels_64']}** occupied voxels @64³ -> "
        f"**{res.stats['coords_32']}** SLAT coords @32³",
    ]
    if mode_note:
        lines.append(mode_note)
    return "\n\n".join(lines)


def _fail(e: Exception):
    """A real message, not a stack trace — but keep the trace in the server log."""
    print("[demo] request failed:\n" + traceback.format_exc(), flush=True)
    msg = str(e).strip() or e.__class__.__name__
    if isinstance(e, torch_oom_types()):
        msg = ("ran out of GPU memory on this request — try again, or reduce the number "
               "of views.")
    raise gr.Error(msg[:400])


def torch_oom_types():
    import torch
    return (torch.cuda.OutOfMemoryError,)


def _run(pack, kind, seed, ss_guidance, ss_steps, slat_cfg, slat_steps, stages, progress,
         mode_note=""):
    res = PIPE.generate(pack, OUT_DIR, _tag(kind), seed=seed, ss_guidance=ss_guidance,
                        ss_steps=int(ss_steps), slat_cfg=slat_cfg,
                        slat_steps=int(slat_steps), want_stages=stages,
                        progress=lambda f, m: progress(f, desc=m))
    return (res.glb, res.ss_glb, res.shape_glb,
            pack.info.get("preview") or None,
            _report(pack, res, seed, mode_note))


# ─────────────────────────────── callbacks ───────────────────────────────
def gen_image(img, seed, randomize, ss_guidance, ss_steps, slat_cfg, slat_steps, remove_bg,
              stages, progress=gr.Progress()):
    if img is None:
        raise gr.Error("drop an image first.")
    seed = _resolve_seed(seed, randomize)
    try:
        progress(0.02, desc="conditioning (v2.2 VLM + DINOv3)")
        pack = PIPE.encode("i1", image=img, remove_bg=remove_bg)
        note = ("" if pack.info["segmented"] else
                "_note: no clear object/background separation was found, so the whole frame "
                "was used. A cut-out or a plain background will do better._")
        return _run(pack, "img", seed, ss_guidance, ss_steps, slat_cfg, slat_steps, stages,
                    progress, note)
    except gr.Error:
        raise
    except Exception as e:
        _fail(e)


def gen_views(files, seed, randomize, ss_guidance, ss_steps, slat_cfg, slat_steps, remove_bg,
              stages, progress=gr.Progress()):
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    if len(paths) < 2:
        raise gr.Error("drop at least 2 views of the SAME object (up to 4).")
    if len(paths) > 4:
        raise gr.Error(f"this model was trained on at most 4 views, you gave {len(paths)}.")
    seed = _resolve_seed(seed, randomize)
    try:
        progress(0.02, desc="conditioning (one joint VLM forward over all views)")
        imgs = [Image.open(p) for p in paths]
        pack = PIPE.encode("im", images=imgs, remove_bg=remove_bg)
        note = ""
        if pack.info["n_views"] != 4:
            note = (f"_note: the multi-view task was trained on exactly 4 views; "
                    f"{pack.info['n_views']} still works but is extrapolation._")
        return _run(pack, "mv", seed, ss_guidance, ss_steps, slat_cfg, slat_steps, stages,
                    progress, note)
    except gr.Error:
        raise
    except Exception as e:
        _fail(e)


def gen_text(text, template, seed, randomize, ss_guidance, ss_steps, slat_cfg, slat_steps,
             stages, progress=gr.Progress()):
    if not (text or "").strip():
        raise gr.Error("type a description first.")
    seed = _resolve_seed(seed, randomize)
    try:
        progress(0.02, desc="conditioning (text-only VLM forward)")
        pack = PIPE.encode("t", text=text, template=TXT_PROMPTS.index(template)
                           if template in TXT_PROMPTS else 0)
        note = ("_text mode has no image, so pose and scale are unconstrained — expect the "
                "right kind of object rather than a specific one._")
        return _run(pack, "txt", seed, ss_guidance, ss_steps, slat_cfg, slat_steps, stages,
                    progress, note)
    except gr.Error:
        raise
    except Exception as e:
        _fail(e)


# ─────────────────────────────── UI ───────────────────────────────
def advanced(with_bg: bool = True):
    """Seed / CFG / step controls shared by all three tabs."""
    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            seed = gr.Number(value=0, label="Seed", precision=0, scale=1)
            randomize = gr.Checkbox(value=True, label="Randomize seed", scale=1)
        with gr.Row():
            ss_g = gr.Slider(1.0, 15.0, value=SS_SAMPLER["guidance_strength"], step=0.5,
                             label="Structure guidance (CFG)")
            ss_s = gr.Slider(6, 40, value=SS_SAMPLER["steps"], step=1,
                             label="Structure steps")
        with gr.Row():
            sl_c = gr.Slider(1.0, 8.0, value=SLAT_CFG, step=0.5,
                             label="Shape/texture guidance (CFG)")
            sl_s = gr.Slider(10, 50, value=SLAT_STEPS, step=1,
                             label="Shape/texture steps")
        bg = gr.Checkbox(value=True, label="Auto-segment the object and re-frame it "
                                           "(recommended — matches training)",
                         visible=with_bg)
        stages = gr.Checkbox(value=True, label="Also return the intermediate stages "
                                               "(occupancy + untextured shape)")
    return seed, randomize, ss_g, ss_s, sl_c, sl_s, bg, stages


def outputs():
    glb = gr.Model3D(label="Textured result (.glb)", height=460, clear_color=(0.1, 0.1, 0.12, 1))
    status = gr.Markdown()
    with gr.Accordion("The cascade, stage by stage", open=False):
        gr.Markdown("Left to right: what the model was actually conditioned on, the raw "
                    "64³ occupancy the structure flow sampled, and the mesh before any "
                    "texture. Every stage is generated — no ground truth is used anywhere.")
        with gr.Row():
            seen = gr.Gallery(label="What the model saw", height=240, columns=4,
                              object_fit="contain")
            ss = gr.Model3D(label="Stage 1 — occupancy @64³", height=240,
                            clear_color=(0.1, 0.1, 0.12, 1))
            shp = gr.Model3D(label="Stage 2 — shape, untextured", height=240,
                             clear_color=(0.1, 0.1, 0.12, 1))
    return glb, ss, shp, seen, status


def build():
    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate")
    with gr.Blocks(title="BLIP3o-3D · tri-modal 3D generation", theme=theme) as demo:
        gr.Markdown(HEADER)

        with gr.Tabs():
            # ── 1. image ──
            with gr.Tab("Image → 3D"):
                gr.Markdown("Drop one photo or render of a **single object**. This is the "
                            "model's strongest mode.")
                with gr.Row():
                    with gr.Column(scale=4):
                        # image_mode RGBA: a cut-out's alpha channel is the exact signal
                        # ImageTo3DDataset._alpha_crop used at training time — keep it.
                        img = gr.Image(type="pil", label="Input image", height=320,
                                       image_mode="RGBA",
                                       sources=["upload", "clipboard"])
                        a = advanced()
                        btn = gr.Button("Generate 3D", variant="primary")
                        if _examples():
                            gr.Examples(examples=[[p] for p in _examples()], inputs=[img],
                                        label="Examples (held-out assets, never trained on)")
                    with gr.Column(scale=6):
                        o = outputs()
                btn.click(gen_image, inputs=[img, a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                                             a[7]],
                          outputs=list(o), concurrency_limit=1)

            # ── 2. multi-view ──
            with gr.Tab("Multi-view → 3D"):
                gr.Markdown("Drop **2-4 views of the same object**. They go through the VLM "
                            "in one joint forward, each view tagged with its own view code. "
                            "Trained on 4; fewer works but is extrapolation.")
                with gr.Row():
                    with gr.Column(scale=4):
                        files = gr.File(file_count="multiple", file_types=["image"],
                                        label="2-4 views of one object")
                        prev = gr.Gallery(label="Views", height=180, columns=4,
                                          object_fit="contain")
                        files.change(lambda fs: [f if isinstance(f, str) else f.name
                                                 for f in (fs or [])],
                                     inputs=files, outputs=prev)
                        a2 = advanced()
                        btn2 = gr.Button("Generate 3D", variant="primary")
                        if _mv_example():
                            gr.Examples(examples=[[_mv_example()]], inputs=[files],
                                        label="Example: 4 views of one held-out asset")
                    with gr.Column(scale=6):
                        o2 = outputs()
                btn2.click(gen_views, inputs=[files, a2[0], a2[1], a2[2], a2[3], a2[4],
                                              a2[5], a2[6], a2[7]],
                           outputs=list(o2), concurrency_limit=1)

            # ── 3. text ──
            with gr.Tab("Text → 3D"):
                gr.Markdown("Describe one object. The training captions look like the "
                            "examples below — a noun phrase plus shape, material and colour "
                            "detail. **No image means pose and scale are unconstrained**: "
                            "you get the right kind of object, not a specific one.")
                with gr.Row():
                    with gr.Column(scale=4):
                        txt = gr.Textbox(lines=4, label="Description",
                                         value=EXAMPLE_CAPTIONS[0],
                                         placeholder="A wooden dining chair with a tall "
                                                     "slatted back ...")
                        gr.Examples(examples=[[c] for c in EXAMPLE_CAPTIONS], inputs=[txt],
                                    label="Training-style captions")
                        tmpl = gr.Dropdown(TXT_PROMPTS, value=TXT_PROMPTS[0],
                                           label="Prompt template (all three were used in "
                                                 "training)")
                        a3 = advanced(with_bg=False)
                        btn3 = gr.Button("Generate 3D", variant="primary")
                    with gr.Column(scale=6):
                        o3 = outputs()
                btn3.click(gen_text, inputs=[txt, tmpl, a3[0], a3[1], a3[2], a3[3], a3[4],
                                             a3[5], a3[7]],
                           outputs=list(o3), concurrency_limit=1)

        gr.Markdown(CAVEATS)
        gr.Markdown(
            f"<sub>checkpoints: `{os.path.relpath(SS_CKPT, ROOT)}` · "
            f"`{os.path.relpath(SHAPE_CKPT, ROOT)}` · `{os.path.relpath(TEX_CKPT, ROOT)}` "
            f"— one 37 000-step tri-modal run. Requests are served one at a time on a single "
            f"H200; if someone else is generating, yours queues.</sub>")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    PIPE.load()
    demo = build()
    demo.queue(default_concurrency_limit=1, max_size=24)
    demo.launch(server_name=a.host, server_port=a.port, share=a.share,
                allowed_paths=[OUT_DIR, ASSETS], show_error=True)


if __name__ == "__main__":
    main()
