"""LIVE tri-modal conditioning — the demo-side twin of the offline cond cache.

Every eval in this repo reads *precomputed* conds from `vlm_hidden_cache/...`. A demo takes
arbitrary user input, so the v2.2-VLM and DINOv3 forwards have to happen at request time.
This module reproduces, token for token, what `scripts/build_vlm_cache_v22.py` +
`scripts/build_dino_cache.py` wrote for each of the three modalities the S3 model was
trained on. Get any of it wrong and the model is evaluated off-distribution.

Contract reproduced (verified against the live caches — see `assert_matches_cache`):

  I1  single image   root data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1, key v000
      qwen : "[3D Gen] <image>\\nReconstruct this object in 3D.", add_generation_prompt=True,
             image = the FULL 1024² render (uncropped) -> 1024 vision tok, 1054 total
      dino : DINOv3 @512 on the ALPHA-CROPPED render -> 1029 tok, view code dve[0]

  IM  2-4 views      root vlm_hidden_cache/v22_im4r, key m00
      qwen : ONE joint forward over N images, each capped to 64 vision tok (256² px)
             -> 292 tok for N=4. Per-token view code on the image blocks (flow_heads
             applies the SAME dve to the qwen segment for mode IM).
      dino : DINOv3 @320 on the ALPHA-CROPPED renders -> 405 tok/view, code dve[view_id]

  T   text           same root as I1, keys t000..t003
      qwen : "[3D Gen] " + one of TXT_PROMPTS, add_generation_prompt=True, NO image
      dino : ABSENT ENTIRELY (flow_heads takes the non-fusion branch; no view embedding)

THE BACKGROUND TRAP: the training renders are 1024² RGBA whose RGB under alpha=0 is ~(1,1,1),
i.e. `.convert("RGB")` puts the object on BLACK, and the object's longest side spans ~0.51 of
the frame (measured over the held-out renders). A user photo on a white desk is a different
distribution. `object_frame()` normalizes any input into that geometry: segment -> square box
-> composite on black -> 0.52 extent. The demo shows the result so the user sees what the
model actually got.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from . import _paths  # noqa: F401 — puts trellis2 on sys.path
from .vlm_collate import boiler_ids, cap_image, px_per_tok
from .dino_align import DinoV3FeatureExtractor, TRELLIS_DINOV3_NAME

# ── the exact strings the v2.2 cache was built with (build_vlm_cache_v22.py) ──
_V22_DEFAULT = ("/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/"
                "v0-20260703-051346/checkpoint-1000")
# COND_VLM_CKPT swaps the conditioning VLM for every consumer that defaults to V22_CKPT
# (TrainCondEncoder, prep_i1/prep_im, LiveCond). It must be a LOCAL dir whose processor
# uses the same <|vision_start|><|image_pad|><|vision_end|> convention (any Qwen VL).
# Unset → the v2.2 finetune, bit-identical to every run launched before this knob existed.
V22_CKPT = os.environ.get("COND_VLM_CKPT", _V22_DEFAULT)
IMG_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPT_I1 = "[3D Gen] " + IMG_TOKEN + "\nReconstruct this object in 3D."
PROMPT_IM = "[3D Gen] {imgs}\nReconstruct this object in 3D."
TXT_PROMPTS = ["Generate a 3D asset: {c}",
               "Create a 3D model of: {c}",
               "Make this in 3D: {c}"]

# per-modality token budgets, from the cache _meta.json of the roots the 40k run trained on
DINO_SIZE_I1 = 512          # v22_3dvlm_tok1024_mv1: dino_image_size 512 -> 1029 tok
DINO_SIZE_IM = 320          # v22_im4r:              dino_image_size 320 ->  405 tok/view
IM_QWEN_TOK_PER_VIEW = 128  # 362² px/view; 4 views -> 512 qwen tokens
# Raised from 64 (v22_im4r). At 64 a view carried 256x256 pixels and four of them totalled 256
# tokens — one SIXTEENTH of what the single-image path gets for one view (1024). That is the
# most likely physical reason the IM arm ignores view CONTENT (four different views scored the
# same as four copies of one): the information never entered the conditioning.
#
# This invalidates the v22_im4r cond cache and any flow trained on it — the cache has to be
# rebuilt and the flows retrained. That is affordable only because the final run is from
# scratch anyway; do NOT change this constant against an already-trained flow.
I1_QWEN_CANVAS = 1024       # the render resolution -> 1024 vision tok

# render geometry (measured over the held-out renders_cond webps)
RENDER_BG = (1, 1, 1)       # RGB under alpha=0 in the training renders (~black)
OBJECT_EXTENT = 0.52        # object longest side / frame width, median 0.542 over 30 renders


# ────────────────────────────── image preprocessing ──────────────────────────────
def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest blob — kills JPEG speckle and stray background objects."""
    try:
        import cv2
    except Exception:
        return mask
    m = (mask.astype(np.uint8)) * 255
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == biggest


def _border_bg_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Foreground mask for an object on a roughly uniform backdrop (studio / product shot).

    Estimates the background colour from the image border and keeps pixels far from it.
    Deliberately self-limiting: on a busy photo the border itself is not uniform, the
    threshold blows up, the mask degenerates and we return None so the caller falls back
    to 'no crop' rather than mangling the image.
    """
    h, w = rgb.shape[:2]
    b = max(2, min(h, w) // 48)
    border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                             rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    spread = float(np.median(np.abs(border - bg).sum(axis=1)))
    if spread > 90:                       # background is not uniform -> refuse
        return None
    dist = np.abs(rgb.astype(np.float32) - bg).sum(axis=-1)
    mask = dist > max(36.0, spread * 4.0)
    frac = float(mask.mean())
    if frac < 0.004 or frac > 0.94:       # found nothing, or "everything is foreground"
        return None
    mask = _largest_component(mask)
    if not mask.any():
        return None
    return mask


# ── learned matting (BiRefNet — the SAME model TRELLIS.2 uses for rembg). Segments a real
#    photo into the object-on-black setup the training renders used, instead of the border
#    heuristic (which refuses on busy backgrounds → raw photo w/ background fed to the
#    encoders = OOD). Lazy singleton, loaded once on first real photo. Env: LIVE_REMBG=off
#    forces the border fallback; BIREFNET_MODEL overrides the HF id. ──
_BIREFNET = None            # (model, transform, device) once loaded; "FAILED" if unusable


def _get_birefnet():
    global _BIREFNET
    if _BIREFNET is not None:
        return None if _BIREFNET == "FAILED" else _BIREFNET
    if os.environ.get("LIVE_REMBG", "").lower() in ("off", "border", "0"):
        _BIREFNET = "FAILED"; return None
    try:
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        name = os.environ.get("BIREFNET_MODEL", "ZhengPeng7/BiRefNet")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageSegmentation.from_pretrained(name, trust_remote_code=True)
        model.eval().to(dev)
        tfm = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _BIREFNET = (model, tfm, dev)
        print(f"[live_cond] BiRefNet loaded ({name}) on {dev} for background removal", flush=True)
        return _BIREFNET
    except Exception as e:
        print(f"[live_cond] BiRefNet unavailable ({type(e).__name__}: {str(e)[:120]}); "
              f"falling back to border-heuristic matting", flush=True)
        _BIREFNET = "FAILED"; return None


def _birefnet_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Foreground mask via BiRefNet (TRELLIS.2-parity rembg). Boolean HxW mask, or None if
    the model can't load / the matte is degenerate (→ caller falls back to the heuristic)."""
    got = _get_birefnet()
    if got is None:
        return None
    model, tfm, dev = got
    try:
        pil = Image.fromarray(rgb.astype(np.uint8)[:, :, :3])
        pdtype = next(model.parameters()).dtype           # match model dtype (transformers may load half)
        x = tfm(pil).unsqueeze(0).to(dev, pdtype)
        with torch.no_grad():
            matte = model(x)[-1].sigmoid().float().cpu()[0].squeeze()     # (1024,1024) ∈ [0,1]
        matte = Image.fromarray((matte.numpy() * 255).astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.BILINEAR)
        mask = np.array(matte) > 128                                      # threshold → object
        frac = float(mask.mean())
        if frac < 0.001 or frac > 0.999:            # empty or everything → treat as failure
            return None
        return _largest_component(mask)
    except Exception as e:
        print(f"[live_cond] BiRefNet inference failed ({type(e).__name__}: {str(e)[:100]})", flush=True)
        return None


def _square_box(mask: np.ndarray):
    """The alpha-crop box of ImageTo3DDataset._alpha_crop: square, centred on the bbox."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0, 1) / 2.0
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


@dataclass
class FramedImage:
    """One user image normalized into the training-render geometry."""
    full: Image.Image     # 1024² RGB, object at ~0.52 extent on black  -> the QWEN input
    tight: Image.Image    # square crop, object edge-to-edge on black   -> the DINO input
    segmented: bool       # True if a foreground mask was actually found


def object_frame(img: Image.Image, canvas: int = I1_QWEN_CANVAS,
                 extent: float = OBJECT_EXTENT, remove_bg: bool = True) -> FramedImage:
    """Any user image -> (full render-like frame, tight object crop), both RGB on black.

    Mirrors the two views of an asset the caches were built from:
      * QWEN saw the whole 1024² render (object ~0.51 of the frame, black background)
      * DINO saw `ImageTo3DDataset._alpha_crop` of that render (object edge-to-edge)
    """
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
    mask = None
    if img.mode == "RGBA":
        a = np.array(img.getchannel("A"))
        if (a == 0).mean() > 0.01:            # a real alpha channel, not a fully-opaque one
            mask = a > 0
        img_rgb = np.array(img.convert("RGB"))
    else:
        img_rgb = np.array(img)
    if mask is None and remove_bg:
        mask = _birefnet_mask(img_rgb)          # learned matting (TRELLIS.2-parity rembg)
        if mask is None:                        # weights/net unavailable, or degenerate matte
            mask = _border_bg_mask(img_rgb)     # fallback: uniform-backdrop heuristic
    segmented = mask is not None
    if mask is None:
        mask = np.ones(img_rgb.shape[:2], dtype=bool)

    # object on black (exactly what .convert("RGB") does to a training render)
    obj = img_rgb.copy()
    obj[~mask] = RENDER_BG
    box = _square_box(mask) or (0, 0, img_rgb.shape[1], img_rgb.shape[0])
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0, 8)
    tight = np.full((side, side, 3), RENDER_BG, dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img_rgb.shape[1], x0 + side), min(img_rgb.shape[0], y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        tight[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = obj[sy0:sy1, sx0:sx1]
    tight_im = Image.fromarray(tight)

    # full frame: object at `extent` of a square black canvas, centred
    s = max(8, int(round(canvas * extent)))
    full = Image.new("RGB", (canvas, canvas), RENDER_BG)
    full.paste(tight_im.resize((s, s), Image.LANCZOS), ((canvas - s) // 2, (canvas - s) // 2))
    return FramedImage(full=full, tight=tight_im, segmented=segmented)


# ─────────────────────────────── the live encoder ───────────────────────────────
@dataclass
class CondPack:
    """Raw (pre-connector) conditioning for one request — the in-memory twin of a cache npz.

    `dino` is None for text: that is not an optimization, it is the T task's definition
    (flow_heads only takes the fusion branch when dino_hidden is not None).
    """
    qwen: torch.Tensor                              # (Tq, 2048) float32 cuda
    qwen_keep: torch.Tensor                         # (Tq,) bool cuda
    dino: Optional[torch.Tensor] = None             # (Td, 1024) float32 cuda
    dino_keep: Optional[torch.Tensor] = None        # (Td,) bool cuda
    dino_view_ids: Optional[torch.Tensor] = None    # (Td,) long cuda — IM only
    qwen_view_ids: Optional[torch.Tensor] = None    # (Tq,) long cuda, -1 = no code — IM only
    # VLM-as-narrator stream: the model's OWN caption of the input, encoded through the
    # text->3D path. None = absent (the SAM3D-style "this modality is dropped" state).
    cap_qwen: Optional[torch.Tensor] = None         # (Tc, 2048) float32 cuda
    cap_keep: Optional[torch.Tensor] = None         # (Tc,) bool cuda
    caption: Optional[str] = None                   # the readable text, for UI / debugging
    modality: str = "i1"
    info: Dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        n = int(self.qwen_keep.sum())
        if self.dino_keep is not None:
            n += int(self.dino_keep.sum())
        return n


class LiveCondEncoder:
    """v2.2 VLM + DINOv3, resident, producing cache-identical conds from live input."""

    def __init__(self, vlm_path: str = V22_CKPT, device: str = "cuda"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.device = device
        self.proc = AutoProcessor.from_pretrained(vlm_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            vlm_path, torch_dtype=torch.bfloat16, device_map={"": device}).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = self.proc.tokenizer
        self.boiler = boiler_ids(self.tok, include_system=False)
        self.px_per_tok = px_per_tok(self.proc)
        # one DINOv3 instance serves both sizes: image_size only drives the PIL resize,
        # the ViT itself is resolution-agnostic (RoPE is computed from the actual input).
        self.dino = DinoV3FeatureExtractor(TRELLIS_DINOV3_NAME, image_size=DINO_SIZE_I1)
        self.dino.model.eval().to(device)
        for p in self.dino.model.parameters():
            p.requires_grad_(False)
        self.image_token_id = self._resolve_image_token_id()

    def _resolve_image_token_id(self) -> int:
        for name in ("<|image_pad|>", "<|image|>"):
            tid = self.tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0:
                return tid
        cfg = getattr(self.model, "config", None)
        tid = getattr(cfg, "image_token_id", None)
        if tid is None:
            tid = getattr(getattr(cfg, "text_config", cfg), "image_token_id", None)
        if tid is None:
            raise RuntimeError("cannot resolve the Qwen image-pad token id")
        return int(tid)

    # ── the single VLM forward used by all three modalities (build_vlm_cache_v22._encode) ──
    @torch.no_grad()
    def _encode(self, text: str, images: Optional[Sequence[Image.Image]] = None):
        kw: Dict = dict(text=[text], return_tensors="pt")
        if images:
            kw["images"] = list(images)
        inputs = self.proc(**kw).to(self.device)
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1][0]                       # (T, 2048) last layer
        ids = inputs["input_ids"][0]
        am = inputs["attention_mask"][0].bool()
        keep = am & ~torch.tensor([int(t) in self.boiler for t in ids.tolist()],
                                  device=ids.device)
        return hidden.float(), keep, ids, inputs.get("image_grid_thw")

    def _chat(self, content: str) -> str:
        return self.proc.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True)

    @torch.no_grad()
    def _dino(self, images: Sequence[Image.Image], image_size: int) -> torch.Tensor:
        self.dino.image_size = image_size
        return self.dino(list(images)).float()                  # (M, N, 1024)

    def _qwen_view_ids(self, ids: torch.Tensor, grid_thw, n_views: int) -> torch.Tensor:
        """Per-token view ordinal for the QWEN segment (-1 = text/structural token).

        `eval_im_vs_i1_ss.build_cond_im` hardcodes the cache's fixed layout (292 tokens, four
        64-token blocks at 10/76/142/208). We derive the same thing from the actual
        image_pad positions so 2 and 3 views work too; `assert_matches_cache` checks the
        derived layout still reproduces 10/66/64 on the 4-view case.
        """
        qv = torch.full((ids.shape[0],), -1, dtype=torch.long, device=ids.device)
        pos = (ids == self.image_token_id).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            return qv
        ms = getattr(self.proc.image_processor, "merge_size", 2)
        if grid_thw is not None:
            counts = [int(g[0] * g[1] * g[2]) // (ms * ms) for g in grid_thw]
        else:
            counts = [pos.numel() // n_views] * n_views
        p = 0
        for v, n in enumerate(counts[:n_views]):
            qv[pos[p:p + n]] = v
            p += n
        return qv

    # ─────────────────────────── I1: single image ───────────────────────────
    def encode_image(self, img: Image.Image, remove_bg: bool = True) -> CondPack:
        fr = object_frame(img, remove_bg=remove_bg)
        qwen, keep, _, _ = self._encode(self._chat(PROMPT_I1), images=[fr.full])
        dino = self._dino([fr.tight], DINO_SIZE_I1)[0]          # (1029, 1024)
        return CondPack(
            qwen=qwen, qwen_keep=keep,
            dino=dino, dino_keep=torch.ones(dino.shape[0], dtype=torch.bool,
                                            device=dino.device),
            modality="i1",
            info={"n_qwen": int(qwen.shape[0]), "n_dino": int(dino.shape[0]),
                  "segmented": fr.segmented, "preview": [fr.full]})

    # ─────────────────────────── IM: 2-4 views ───────────────────────────
    def encode_views(self, imgs: Sequence[Image.Image], remove_bg: bool = True) -> CondPack:
        if not 1 <= len(imgs) <= 4:
            raise ValueError(f"multi-view takes 1-4 images, got {len(imgs)}")
        frames = [object_frame(im, remove_bg=remove_bg) for im in imgs]
        n = len(frames)
        # QWEN: ONE joint forward, every view capped to 64 vision tokens (im_qwen_tok_per_view)
        max_px = IM_QWEN_TOK_PER_VIEW * self.px_per_tok
        q_imgs = [cap_image(f.full, max_px) for f in frames]
        text = self._chat(PROMPT_IM.format(imgs=IMG_TOKEN * n))
        qwen, keep, ids, grid = self._encode(text, images=q_imgs)
        qv = self._qwen_view_ids(ids, grid, n)
        # DINO: per-view @320 with per-token view ordinals
        feats = self._dino([f.tight for f in frames], DINO_SIZE_IM)   # (n, 405, 1024)
        nd = feats.shape[1]
        dino = feats.reshape(-1, feats.shape[-1])
        vids = torch.arange(n, device=dino.device).repeat_interleave(nd)
        return CondPack(
            qwen=qwen, qwen_keep=keep,
            dino=dino, dino_keep=torch.ones(dino.shape[0], dtype=torch.bool,
                                            device=dino.device),
            dino_view_ids=vids, qwen_view_ids=qv, modality="im",
            info={"n_views": n, "n_qwen": int(qwen.shape[0]), "n_dino": int(dino.shape[0]),
                  "dino_per_view": int(nd),
                  "qwen_blocks": [int((qv == v).sum()) for v in range(n)],
                  "segmented": all(f.segmented for f in frames),
                  "preview": [f.full for f in frames]})

    # ─────────────────────────── T: text ───────────────────────────
    # ───────────────── VLM-as-narrator: image(s) -> caption -> conditioning ─────────────────
    # The v2.2 VLM was trained on BOTH halves of this loop with the SAME caption strings
    # (vlm3d_stage1/build_messages_jsonl_v22.py):
    #     "[VQA] <image>\n" + CAP_PROMPTS   -> caption      (the v2.1 "captioning heal" task)
    #     "[3D Gen] " + TXT_PROMPTS(c=cap)  -> 3D codes     (the text->3D task)
    # so a caption the model writes itself is IN-DISTRIBUTION for the text conditioning path.
    # That is what makes this composable with zero training.
    #
    # Do NOT prefix a caption request with "[3D Gen]": that tag routes the model into 3D-token
    # generation and it emits <3DSTART><SS_...> instead of prose (measured 2026-07-30).
    CAPTION_PROMPTS = {
        # exactly the trained strings — the safest, in-distribution default
        "trained":   "[VQA] <image>\nDescribe this image in detail.",
        "trained_short": "[VQA] <image>\nDescribe this image.",
        # same tag, steered at the two axes the flow actually needs. Mildly OOD in wording,
        # in-distribution in format; measured to produce accurate material words.
        "material":  "[VQA] <image>\nDescribe this object's material and surface finish: "
                     "what it is made of, whether it is metallic or non-metallic, glossy or "
                     "matte, rough or smooth.",
        "geometry":  "[VQA] <image>\nDescribe this object's shape, structure and parts in detail.",
        # two-step "CoT": reason over shape then material, then commit to one sentence. The most
        # OOD of the set for a 2B model — keep it behind an A/B, do not assume it wins.
        "cot":       "[VQA] <image>\nFirst think about what the object is and how it is built, "
                     "then about what it is made of, then give one detailed description covering "
                     "shape, parts, colour, material and surface finish.",
    }

    @torch.no_grad()
    def caption_images(self, imgs: Sequence[Image.Image], strategy: str = "trained",
                       max_new_tokens: int = 96, remove_bg: bool = True) -> str:
        """Let the VLM describe the input image(s) in its own words.

        Single image uses the trained `[VQA] <image>` form verbatim. For multiple views the
        <image> placeholder is repeated — the captioning task was trained single-image, but the
        IM generation task fed several views through one joint forward, so the processor layout
        is familiar even though this exact combination is an extrapolation.
        """
        if strategy not in self.CAPTION_PROMPTS:
            raise ValueError(f"unknown caption strategy {strategy!r}; "
                             f"have {sorted(self.CAPTION_PROMPTS)}")
        frames = [object_frame(im, remove_bg=remove_bg).full for im in imgs]
        prompt = self.CAPTION_PROMPTS[strategy]
        if len(frames) > 1:                       # one placeholder per view
            prompt = prompt.replace("<image>", "<image>" * len(frames), 1)
        text = self._chat(prompt.replace("<image>", IMG_TOKEN))
        inputs = self.proc(text=[text], images=frames, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen.strip()

    @torch.no_grad()
    def encode_caption(self, caption: str, template: int = 0):
        """Encode a caption through the TEXT->3D path and return (hidden, keep).

        Identical to `encode_text`'s encoding, but returned raw so the caller can attach it to
        an image pack as an EXTRA conditioning stream rather than as a standalone T request.
        """
        caption = (caption or "").strip()
        if not caption:
            return None, None
        text = self._chat("[3D Gen] " + TXT_PROMPTS[template % len(TXT_PROMPTS)].format(c=caption))
        qwen, keep, _, _ = self._encode(text)
        return qwen[keep], keep[keep]

    @torch.no_grad()
    def attach_self_caption(self, pack: CondPack, imgs: Sequence[Image.Image],
                            strategy: str = "trained", caption: Optional[str] = None,
                            remove_bg: bool = True) -> CondPack:
        """Add the narrator stream to an existing image pack, in place.

        `caption=None` → the VLM writes it; pass a string to override (a user-edited caption,
        or a GT caption for a controlled A/B). Failure to caption leaves the pack untouched,
        so the image path keeps working.
        """
        try:
            cap = caption if caption is not None else self.caption_images(
                imgs, strategy=strategy, remove_bg=remove_bg)
            h, k = self.encode_caption(cap)
            if h is None:
                return pack
            pack.cap_qwen, pack.cap_keep, pack.caption = h, k, cap
            pack.info["caption_strategy"] = "user" if caption is not None else strategy
            pack.info["n_cap"] = int(h.shape[0])
        except Exception as e:                       # narrator is additive — never fatal
            print(f"[live_cond] self-caption failed ({type(e).__name__}: {str(e)[:90]})",
                  flush=True)
        return pack

    def encode_text(self, caption: str, template: int = 0) -> CondPack:
        caption = (caption or "").strip()
        if not caption:
            raise ValueError("empty description")
        text = self._chat("[3D Gen] " + TXT_PROMPTS[template % len(TXT_PROMPTS)]
                          .format(c=caption))
        qwen, keep, _, _ = self._encode(text)                   # NO images -> no DINO segment
        # the t-cache stores keep-only; the connector is an MLP (token-wise) so filtering
        # before vs after is identical, but mirror the cache anyway.
        qwen, keep = qwen[keep], keep[keep]
        return CondPack(qwen=qwen, qwen_keep=keep, modality="t",
                        info={"n_qwen": int(qwen.shape[0]),
                              "prompt": TXT_PROMPTS[template % len(TXT_PROMPTS)]
                              .format(c=caption)})


# ─────────────────────────── per-stage conditioning ───────────────────────────
@torch.no_grad()
def build_stage_cond(conn, dve, pack: CondPack, cond_mode: str = "fusion"):
    """(cond, uncond) for ONE flow stage, through THAT stage's connector + view table.

    Merges the three cache-side builders of `scripts/trimodal_fullchain_eval.py`
    (build_cond_i1 / build_cond_im / build_cond_t) into one function keyed off what the
    pack carries — which is exactly how flow_heads.py branches at training time.

    cond_mode selects which conditioning segments reach the flow:
      "fusion"    — [DINO ; connector(qwen)]  (default; the trained deploy regime)
      "qwen_only" — drop the DINO segment      (the dino_drop training regime — supported)
      "dino_only" — drop the qwen segment      (EXPERIMENTAL: the model rarely saw DINO
                    without qwen in training, so expect degraded / OOD behaviour)
    """
    qwen = pack.qwen
    cq = conn(qwen[None])
    c0 = conn(torch.zeros_like(qwen)[None])
    if getattr(conn, "pos_stamp", None) is not None:      # not used by the 40k ckpts
        from .pos_stamp import IMG_SPAN_FULL
        cq = conn.pos_stamp(cq, IMG_SPAN_FULL)
        c0 = conn.pos_stamp(c0, IMG_SPAN_FULL)
    # IM applies the view code to BOTH segments; I1 to DINO only; T to neither.
    if pack.qwen_view_ids is not None and dve is not None:
        qv = pack.qwen_view_ids
        add = dve[qv.clamp_min(0)].float() * (qv >= 0).unsqueeze(-1).float()
        cq = cq + add[None]
    # ── narrator stream: the VLM's own caption, encoded through the text->3D path ──
    # Appended along the SEQUENCE (the same axis DINO and qwen already share) so no shape
    # anywhere else changes; absent caption = the stream is simply not concatenated, which is
    # the "modality dropped" state. CFG mirrors the qwen convention: connector(zeros).
    cap_c = cap_u = None
    if cond_mode != "dino_only" and pack.cap_qwen is not None:
        cap_c = conn(pack.cap_qwen[None])[:, pack.cap_keep]
        cap_u = conn(torch.zeros_like(pack.cap_qwen)[None])[:, pack.cap_keep]

    # qwen-only: DINO segment absent — same path TEXT takes, and the dino_drop regime.
    if cond_mode == "qwen_only" or pack.dino is None:
        c, u = cq[:, pack.qwen_keep], c0[:, pack.qwen_keep]
        if cap_c is not None:
            c, u = torch.cat([c, cap_c], 1), torch.cat([u, cap_u], 1)
        return c, u
    dseg = pack.dino[None]
    if dve is not None:
        dseg = dseg + (dve[pack.dino_view_ids][None].float()
                       if pack.dino_view_ids is not None else dve[0][None, None].float())
    # dino-only: drop the qwen segment (EXPERIMENTAL). CFG uncond = zeros-DINO, as in fusion.
    if cond_mode == "dino_only":
        return dseg[:, pack.dino_keep], torch.zeros_like(dseg)[:, pack.dino_keep]
    cond = torch.cat([dseg, cq], 1)
    uncond = torch.cat([torch.zeros_like(dseg), c0], 1)    # flow_heads CFG convention
    keep = torch.cat([pack.dino_keep, pack.qwen_keep])
    cond, uncond = cond[:, keep], uncond[:, keep]
    if cap_c is not None:                                  # narrator appended last
        cond, uncond = torch.cat([cond, cap_c], 1), torch.cat([uncond, cap_u], 1)
    return cond, uncond


# ─────────────────────────── parity self-check ───────────────────────────
def assert_matches_cache(enc: LiveCondEncoder, renders_dir: str, sha: str,
                         cache_i1: str, cache_im: str, verbose: bool = True) -> Dict:
    """Re-encode a cached asset LIVE from its renders and diff against the stored npz.

    Not a unit test of numerics (live preprocessing reframes the render, so features
    legitimately differ) — it verifies the SHAPES and the qwen view-block LAYOUT, which is
    where a live re-implementation actually goes wrong.
    """
    out: Dict = {}
    a = np.load(os.path.join(cache_i1, sha[:2], sha, "v000.npz"))
    view = int(a["view"]) if "view" in a.files else 8
    src = None
    for ext in ("webp", "png", "jpg"):
        p = os.path.join(renders_dir, f"{view:03d}.{ext}")
        if os.path.isfile(p):
            src = p
            break
    p1 = enc.encode_image(Image.open(src))
    out["i1"] = {"live": (p1.qwen.shape[0], p1.dino.shape[0]),
                 "cache": (a["hidden"].shape[0], a["dino_hidden"].shape[0])}
    b = np.load(os.path.join(cache_im, sha[:2], sha, "m00.npz"))
    views = [int(v) for v in b["views"]]
    imgs = []
    for v in views:
        for ext in ("webp", "png", "jpg"):
            p = os.path.join(renders_dir, f"{v:03d}.{ext}")
            if os.path.isfile(p):
                imgs.append(Image.open(p))
                break
    p2 = enc.encode_views(imgs)
    qv = p2.qwen_view_ids
    starts = [int((qv == v).nonzero()[0]) for v in range(len(imgs))]
    out["im"] = {"live": (p2.qwen.shape[0], p2.dino.shape[0]),
                 "cache": (b["hidden"].shape[0], b["dino_hidden"].shape[0]),
                 "block_starts": starts, "block_sizes": p2.info["qwen_blocks"]}
    out["im"]["layout_ok"] = (starts == [10, 76, 142, 208]
                              and p2.info["qwen_blocks"] == [64] * 4) if len(imgs) == 4 else None
    if verbose:
        print(f"[parity] I1 live qwen/dino {out['i1']['live']} vs cache {out['i1']['cache']}")
        print(f"[parity] IM live qwen/dino {out['im']['live']} vs cache {out['im']['cache']}"
              f"  blocks@{starts} sizes={p2.info['qwen_blocks']} "
              f"layout_ok={out['im']['layout_ok']}")
    return out
