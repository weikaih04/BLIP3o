"""LIVE conditioning for TRAINING — batched twin of the offline cond cache.

`live_cond.LiveCondEncoder` is the *demo* twin: it takes an arbitrary user photo and
normalizes it (`object_frame`: segment → square box → composite on black) into the render
distribution. That reframing is exactly what we must NOT do at training time — we already
HAVE the renders the cache was built from, so we reproduce the cache's own preprocessing
instead, and parity is checkable to the last token (see `scripts/_tmp/lc_parity.py`).

Reproduced, per I1 sample (build_vlm_cache_v22.py mode=views + build_dino_cache.py):

    qwen : apply_chat_template("[3D Gen] <image>\\nReconstruct this object in 3D.")
           over the FULL, UNCROPPED render, `.convert("RGB")`   → 1054 tok, fp16
    dino : DINOv3 @512 over the ALPHA-CROPPED render (the exact
           `ImageTo3DDataset._load_views` path, crop_to_object=1) → 1029 tok, fp16

WHY THIS EXISTS — the cache is frozen at ONE view per asset (`max_views: 1`, because a
second view costs 3.86 TB and there are 2.1 TB free). The loader already draws a random
view id, but `_cache_views == 1` clamps every draw to view `pick_view(sha)`. Encoding live
reads a 21 KB webp instead of an 8.5 MB npz (400× less I/O) and unlocks all 16 views.

SPLIT — the 253 ms/image of the naive path is 88 ms CPU (processor) + 99 ms GPU + misc.
`prep()` is the CPU half and is safe to call in a dataloader worker; `encode()` is the GPU
half and does ONE Qwen forward and ONE DINOv3 forward for the whole batch. Batching is what
pays: 67.3 → 27.2 ms/image for Qwen at bs8. All renders are 1024², so every sample tokenizes
to exactly 1054 and the batch needs NO padding (asserted).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from . import _paths  # noqa: F401 — puts trellis2 on sys.path
from .vlm_collate import boiler_ids
from .dino_align import DinoV3FeatureExtractor, TRELLIS_DINOV3_NAME
from .live_cond import V22_CKPT, PROMPT_I1, IMG_TOKEN, DINO_SIZE_I1, I1_QWEN_CANVAS

# ── IM contract, read off v22_im4r/_meta.json — NOT off live_cond.py ──
# live_cond.IM_QWEN_TOK_PER_VIEW was raised 64 -> 128 for a future from-scratch run, and its
# own comment says doing that against an already-trained flow is wrong. v8 has 21k steps on
# 64 (292 total qwen tokens for N=4); using 128 would silently move the IM conditioning
# off-distribution mid-run. The number that matters is the cache's, so it lives here.
# 64 was a STORAGE limit, not a modelling choice: 420k assets x (292x2048 qwen +
# 1620x1024 dino) fp16 is already 1.9 TB, and 256 tok/view would have made it 3.2 TB
# against a 1.9 TB free quota. Live encoding stores nothing, so the constraint is gone.
#
# What 64 cost: each IM view carried 64 vision tokens against the single-image path's
# 1024 — one SIXTEENTH per view — which is the most likely physical reason the IM arm was
# measured to ignore view CONTENT (four distinct views scored the same as four copies of
# one). 256/view puts IM's total vision budget at 4x256+36 = 1060 against I1's 1054, i.e.
# the two arms get the SAME conditioning bandwidth; that is the reason for this number
# rather than any other. Set GEOTEX_IM_TOK_PER_VIEW=64 to reproduce the cache exactly.
IM_QWEN_TOK_PER_VIEW = int(os.environ.get("GEOTEX_IM_TOK_PER_VIEW", "256"))
DINO_SIZE_IM = 320                    # -> 405 tok/view, 1620 for N=4
IM_N_VIEWS = 4                        # v22_im4r ships one combo size
IM_VIEW_WEIGHTS = tuple([0.15] * 3 +  # 000-002 below-ground
                        [0.6] * 2 +   # 003-004
                        [1.0] * 7 +   # 005-011 eye-level..3/4, the good ones
                        [0.5] * 4)    # 012-015
# THE POINT OF GOING LIVE FOR IM: the cache draws these with np.random.default_rng(
# int(sha[:16],16)) — a per-sha seed, so every asset has ONE frozen 4-view combo for the
# life of the cache. Live draws from the same weights with a fresh rng per sample.

TXT_PROMPTS = ["Generate a 3D asset: {c}", "Create a 3D model of: {c}", "Make this in 3D: {c}"]


def im_prompt(n: int) -> str:
    return "[3D Gen] " + IMG_TOKEN * n + "\nReconstruct this object in 3D."


def txt_prompt(sha: str, cap_idx: int, caption: str) -> str:
    """build_vlm_cache_v22.cap_text: the template is picked by (sha, caption index)."""
    return "[3D Gen] " + TXT_PROMPTS[(int(sha[:8], 16) + cap_idx) % len(TXT_PROMPTS)].format(c=caption)

# PARITY AGAINST THE FROZEN CACHE (scripts/_tmp/lc_parity.py, 48 val assets, same view):
#   qwen  cos 0.99979 median                     — prompt/template/image/tokenizer confirmed
#   dino  cos 0.9937  mean, deterministic         — see below
# The DINO residual has been narrowed but not closed. Ruled OUT by measurement:
#   nondeterminism (same input twice -> cos 1.00000 exactly), the 2026-08-18 crop-rule
#   change (old rule scores the same), dtype (DINOv3 runs fp32 both sides), and the view
#   (a wrong view scores 0.89-0.97 and the sweep peaks on the right one).
#   NOT cropping at all scores 0.88-0.93, which is what `crop_to_object: false` in the
#   cache's _meta.json would suggest — that field describes the QWEN contract only;
#   build_dino_cache.py passes crop_to_object=1 independently.
# So: ~0.5% of deterministic cosine, an order of magnitude below the stale-entry signal
# (0.83) and thirty times below the wrong-framing signal. Small enough to ship, big
# enough not to call it noise.
#
# SEPARATELY, and worth knowing before trusting any cached number: 1 of 48 sampled cache
# entries (2.1%) has a QWEN half that reproduces at cos 0.83 FLAT ACROSS ALL 16 VIEWS —
# i.e. it does not correspond to any view of its own asset. The builder skips an asset
# whose v000 already exists, so an entry from an earlier build survives forever. Live
# encoding cannot have stale entries by construction.

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


# ────────────────────────── CPU half (dataloader worker) ──────────────────────────
_PROC_CACHE: Dict[str, object] = {}
_TEXT_CACHE: Dict[str, str] = {}


def _worker_proc(vlm_path: str):
    """Per-process processor. Built lazily so the encoder object stays picklable and each
    dataloader worker gets its own (the fast image processor is not fork-safe to share)."""
    p = _PROC_CACHE.get(vlm_path)
    if p is None:
        from transformers import AutoProcessor
        p = AutoProcessor.from_pretrained(vlm_path)
        _PROC_CACHE[vlm_path] = p
        _TEXT_CACHE[vlm_path] = p.apply_chat_template(
            [{"role": "user", "content": PROMPT_I1}],
            tokenize=False, add_generation_prompt=True)
    return p


def _view_path(renders_dir: str, view: int) -> Optional[str]:
    for ext in ("webp", "png", "jpg"):
        c = os.path.join(renders_dir, f"{view:03d}.{ext}")
        if os.path.isfile(c):
            return c
    return None


def _dino_image(img: Image.Image, min_px: Optional[int] = None) -> Image.Image:
    """`ImageTo3DDataset._load_views` with crop_to_object=1, for ONE already-open image.

    Delegates to the dataset's own staticmethods rather than copying them: an inlined copy
    drifted by a pixel within an hour of being written (`(x0+x1)//2` vs `/2.0` rounds the
    box differently on odd extents), and a one-pixel crop shift is exactly the kind of
    thing that shows up as an unexplained 0.99 cosine three days later.
    """
    from .data.tasks.threed import _ThreeDTaskBase as _T
    min_px = _T.MIN_IMG_PX if min_px is None else min_px
    if img.mode == "RGBA":
        m = max(img.size)
        if m > 1024:
            sc = 1024 / m
            img = img.resize((int(img.width * sc), int(img.height * sc)), Image.LANCZOS)
        img = _T._composite_black(_T._alpha_crop(img))
    else:
        img = img.convert("RGB")
    w, h = img.size
    if w < min_px or h < min_px:
        s = min_px / max(1, min(w, h))
        img = img.resize((max(min_px, int(round(w * s))), max(min_px, int(round(h * s)))),
                         Image.LANCZOS)
    return img


def prep_i1(renders_dir: str, view: int, vlm_path: str = V22_CKPT,
            dino_size: int = DINO_SIZE_I1) -> Dict[str, torch.Tensor]:
    """CPU-only preprocessing for one I1 sample. Returns worker→main transferable tensors.

    Qwen and DINO get DIFFERENT images from the same render: Qwen the full uncropped one
    (the cache used crop_to_object=False for the VLM), DINO the alpha-cropped one. Getting
    this backwards is silent — both produce the right token counts.
    """
    path = _view_path(renders_dir, view)
    if path is None:
        cand = sorted(f for f in os.listdir(renders_dir) if f.endswith((".webp", ".png", ".jpg")))
        if not cand:
            raise FileNotFoundError(f"no render in {renders_dir}")
        path = os.path.join(renders_dir, cand[0])
        view = int(os.path.basename(path)[:3])

    raw = Image.open(path)
    proc = _worker_proc(vlm_path)
    qi = proc(text=[_TEXT_CACHE[vlm_path]], images=[raw.convert("RGB")], return_tensors="pt")

    # DINOv3's own list branch, done here so only a 512² tensor crosses the queue:
    # resize(LANCZOS) → RGB → /255 → CHW. Normalization stays on the GPU side.
    d = _dino_image(raw).resize((dino_size, dino_size), Image.LANCZOS)
    d = torch.from_numpy(np.array(d.convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)

    return {"input_ids": qi["input_ids"][0],                       # (1054,)
            "pixel_values": qi["pixel_values"].to(torch.bfloat16),  # (4096,1536)
            "image_grid_thw": qi["image_grid_thw"][0],             # (3,)
            "dino_px": d,                                          # (3,512,512) fp32
            "view": torch.tensor(int(view))}


def pick_im_views(rng, n_avail: int, k: int = IM_N_VIEWS) -> List[int]:
    """k distinct views, p proportional to IM_VIEW_WEIGHTS, ascending.

    Same weights and same "sorted, distinct" convention as
    build_vlm_cache_v22.pick_im4_views_weighted — the ONLY difference is that `rng` comes
    from the caller instead of being seeded off the sha, which is what unfreezes the combo.
    """
    n = min(n_avail, len(IM_VIEW_WEIGHTS))
    w = np.asarray(IM_VIEW_WEIGHTS[:n], dtype=np.float64)
    k = min(k, n)
    return sorted(int(v) for v in rng.choice(n, size=k, replace=False, p=w / w.sum()))


def prep_im(renders_dir: str, views: Sequence[int], vlm_path: str = V22_CKPT,
            tok_per_view: int = IM_QWEN_TOK_PER_VIEW,
            dino_size: int = DINO_SIZE_IM) -> Dict[str, torch.Tensor]:
    """CPU half for one IM sample — build_vlm_cache_v22 mode=im4, step for step.

    Qwen sees ONE sequence containing all N images, each downscaled to `tok_per_view`
    vision tokens (cap_image on a pixel budget, not a token count — px_per_tok converts).
    DINO sees the N ALPHA-CROPPED renders at 320. The two towers disagree on framing here
    exactly as they do for I1, and again that is the contract, not a bug.
    """
    from .vlm_collate import cap_image, px_per_tok
    proc = _worker_proc(vlm_path)
    max_px = tok_per_view * px_per_tok(proc)

    raws, kept = [], []
    for v in views:
        path = _view_path(renders_dir, v)
        if path is None:
            continue
        raws.append(Image.open(path))
        kept.append(int(v))
    if not raws:
        raise FileNotFoundError(f"no renders for views {list(views)} in {renders_dir}")

    text = proc.apply_chat_template([{"role": "user", "content": im_prompt(len(raws))}],
                                    tokenize=False, add_generation_prompt=True)
    qi = proc(text=[text], images=[cap_image(r.convert("RGB"), max_px) for r in raws],
              return_tensors="pt")
    d = [_dino_image(r).resize((dino_size, dino_size), Image.LANCZOS) for r in raws]
    d = torch.stack([torch.from_numpy(np.array(x.convert("RGB")).astype(np.float32) / 255)
                     .permute(2, 0, 1) for x in d])                    # (N,3,320,320)
    return {"input_ids": qi["input_ids"][0],
            "pixel_values": qi["pixel_values"].to(torch.bfloat16),
            "image_grid_thw": qi["image_grid_thw"],                    # (N,3) — N rows here
            "dino_px": d,
            "views": torch.tensor(kept, dtype=torch.long),
            "modality": "im"}


def prep_t(sha: str, cap_idx: int, caption: str,
           vlm_path: str = V22_CKPT) -> Dict[str, torch.Tensor]:
    """CPU half for one T sample. No image, and — per the cache contract — NO DINO AT ALL:
    flow_heads takes its non-fusion branch for text, so a dino segment here would be a
    silent extra condition the cached run never had."""
    proc = _worker_proc(vlm_path)
    text = proc.apply_chat_template(
        [{"role": "user", "content": txt_prompt(sha, cap_idx, caption)}],
        tokenize=False, add_generation_prompt=True)
    ti = proc(text=[text], return_tensors="pt")
    return {"input_ids": ti["input_ids"][0], "modality": "t"}


# ───────────────────────────── checkpoint staging (/dev/shm) ─────────────────────────────
def stage_to_shm(src: str, dst: Optional[str] = None) -> str:
    """Copy the ~5.6 GB of weights `from_pretrained` actually reads to node-local RAM.

    The checkpoint dir is 32 GB but 26 of that is `global_step1000/` optimizer shards and
    `rng_state_*.pth`; only the safetensors + configs are needed. Loading them off Lustre
    takes ~10 min for ONE process — with 8 local ranks all reading the same 5 GB it is much
    worse, and it happens before the first step of every run. The v2.2 cache builder hit
    this too and solved it the same way (its `_meta.json` records vlm=/dev/shm/v22ckpt).

    Local-rank 0 stages; the others wait on the `.complete` marker. Set LIVE_COND_SHM=0 to
    skip (e.g. a node whose /dev/shm is already full).
    """
    import shutil, time as _t
    if os.environ.get("LIVE_COND_SHM", "1") != "1":
        return src
    if dst is None:
        # The historical fixed dst is kept ONLY for the v2.2 default (v9-era nodes already
        # hold it staged). Any other checkpoint gets its own dir keyed by the source path —
        # a fixed dst would silently hand a swapped-VLM run the previously staged v2.2
        # weights (`.complete` marker present, no freshness check against src).
        from .live_cond import _V22_DEFAULT
        if os.path.abspath(src) == os.path.abspath(_V22_DEFAULT):
            dst = "/dev/shm/v22ckpt_live"
        else:
            import hashlib
            dst = f"/dev/shm/condvlm_{hashlib.sha1(os.path.abspath(src).encode()).hexdigest()[:12]}"
    marker = os.path.join(dst, ".complete")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0 and not os.path.exists(marker):
        tmp = f"{dst}.tmp.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        skip = ("rng_state_", "optimizer", "scheduler", "trainer_state")
        for f in sorted(os.listdir(src)):
            fp = os.path.join(src, f)
            if not os.path.isfile(fp) or f.startswith(skip):
                continue
            shutil.copy2(fp, os.path.join(tmp, f))
        open(os.path.join(tmp, ".complete"), "w").close()
        shutil.rmtree(dst, ignore_errors=True)
        os.rename(tmp, dst)
    for _ in range(1800):                    # 30 min ceiling; a 5.6 GB copy takes ~1
        if os.path.exists(marker):
            return dst
        _t.sleep(1)
    print(f"[live_cond] /dev/shm staging timed out — falling back to {src}", flush=True)
    return src


# ────────────────────────────── GPU half (main process) ──────────────────────────────
class TrainCondEncoder:
    """Resident v2.2 VLM + DINOv3 that turns a list of `prep_i1` dicts into cache-shaped
    conds. ~6.4 GB of the 141 GB card; measured 55.6 ms/image at bs8 (both towers)."""

    def __init__(self, vlm_path: str = V22_CKPT, device: str = "cuda",
                 dino_size: int = DINO_SIZE_I1):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.device, self.dino_size = device, dino_size
        # NOTE the workers keep the ORIGINAL path (they only need the processor, which is a
        # few KB) — staging is for the multi-GB weights this process loads.
        self.vlm_path = vlm_path
        local = stage_to_shm(vlm_path)
        self.proc = AutoProcessor.from_pretrained(local)
        self.model = AutoModelForImageTextToText.from_pretrained(
            local, torch_dtype=torch.bfloat16, device_map={"": device}).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = self.proc.tokenizer
        # (T,) int64 lookup instead of the per-token python `in` set test the cache builder
        # used — that loop alone was 3 ms/sample and it does not vectorize as written.
        self.boiler = torch.tensor(sorted(boiler_ids(self.tok, include_system=False)),
                                   dtype=torch.long, device=device)
        self.dino = DinoV3FeatureExtractor(TRELLIS_DINOV3_NAME, image_size=dino_size)
        self.dino.model.eval().to(device)
        for p in self.dino.model.parameters():
            p.requires_grad_(False)
        self._dino_mean = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
        self._dino_std = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
        self.image_token_id = self._resolve_image_token_id()
        pad = self.tok.pad_token_id
        # Value is irrelevant (those positions are masked out and, being trailing on a
        # causal model, unreachable) but it must be a VALID id or the embedding lookup
        # indexes out of range.
        self.pad_id = int(pad if pad is not None else self.tok.eos_token_id)

    def prep(self, renders_dir: str, view: int) -> Dict[str, torch.Tensor]:
        return prep_i1(renders_dir, view, self.vlm_path, self.dino_size)

    @torch.no_grad()
    def warmup(self, batch_size: int = 8) -> None:
        """Pay the first-call cost here instead of inside training step 1.

        Measured: step 1 of a live run spent 16.0 s inside encode and every step after it
        380-450 ms. That is cuBLAS/flash-attn kernel selection plus the allocator growing to
        hold the 1.28 GB of stacked hidden states — one-time, but it lands in the middle of
        the first optimizer step and looks like a hang.

        Warms through `encode()` on a synthetic black render rather than hand-built tensors.
        Hand-building input_ids is the obvious shortcut and it does not work: the ids have to
        carry exactly 1024 image-pad tokens or Qwen's get_placeholder_mask raises
        "tokens: 0, features: 1024". Let the processor lay them out.
        """
        blank = Image.new("RGB", (I1_QWEN_CANVAS, I1_QWEN_CANVAS))
        qi = self.proc(text=[self._chat_text()], images=[blank], return_tensors="pt")
        prep = {"input_ids": qi["input_ids"][0],
                "pixel_values": qi["pixel_values"].to(torch.bfloat16),
                "image_grid_thw": qi["image_grid_thw"][0],
                "dino_px": torch.zeros(3, self.dino_size, self.dino_size),
                "view": torch.tensor(0)}
        for _ in range(2):
            self.encode([prep] * batch_size)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _chat_text(self) -> str:
        return self.proc.apply_chat_template([{"role": "user", "content": PROMPT_I1}],
                                             tokenize=False, add_generation_prompt=True)

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

    def _qwen_img_pos(self, ids_row: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Per-token patch ordinal within its own view (-1 = text/structural).

        Derived from the real image_pad positions and each image's grid, for the
        same reason _qwen_view_ids is: any hardcoded span is only valid at one
        tokens-per-view setting and breaks the moment GEOTEX_IM_TOK_PER_VIEW
        moves. i1 gives 0..1023 over one view; IM at 64 tok/view gives 0..63 per
        view, four times.
        """
        qp = torch.full((ids_row.shape[0],), -1, dtype=torch.long, device=ids_row.device)
        pos = (ids_row == self.image_token_id).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            return qp
        ms = getattr(self.proc.image_processor, "merge_size", 2)
        o = 0
        for g in grid:
            c = int(g[0] * g[1] * g[2]) // (ms * ms)
            qp[pos[o:o + c]] = torch.arange(c, device=ids_row.device)
            o += c
        return qp

    def _qwen_view_ids(self, ids_row: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Per-token view ordinal over the QWEN segment (-1 = text/structural).

        Derived from the actual image_pad positions and each image's grid, not from the
        cache's hardcoded 292-token layout (10/76/142/208, stride 66) — that layout is only
        valid at 64 tok/view and breaks the moment GEOTEX_IM_TOK_PER_VIEW moves.
        """
        qv = torch.full((ids_row.shape[0],), -1, dtype=torch.long, device=ids_row.device)
        pos = (ids_row == self.image_token_id).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            return qv
        ms = getattr(self.proc.image_processor, "merge_size", 2)
        o = 0
        for v, g in enumerate(grid):
            c = int(g[0] * g[1] * g[2]) // (ms * ms)
            qv[pos[o:o + c]] = v
            o += c
        return qv

    @torch.no_grad()
    def encode(self, preps: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """ONE Qwen forward (+ at most one DINOv3 forward) over the batch → per-sample
        cache-shaped dicts. Handles all three modalities; a batch is homogeneous because
        the mixture samples at batch granularity.

        RIGHT padding, never left. Real tokens keep positions 0..L-1 (identical RoPE to the
        unpadded sequence) and a causal model cannot let trailing pads influence earlier
        positions, so each row's [:L] hiddens match a bs1 forward up to bf16 batched-
        reduction noise. LEFT padding would shift positions and silently corrupt every row.
        """
        B = len(preps)
        dev = self.device
        lens = [int(p["input_ids"].shape[0]) for p in preps]
        L = max(lens)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        am = torch.zeros((B, L), dtype=torch.long)
        for i, p in enumerate(preps):
            ids[i, :lens[i]] = p["input_ids"]
            am[i, :lens[i]] = 1
        ids, am = ids.to(dev, non_blocking=True), am.to(dev, non_blocking=True)

        kw: Dict[str, torch.Tensor] = {}
        grids: List[Optional[torch.Tensor]] = [None] * B
        if "pixel_values" in preps[0]:
            kw["pixel_values"] = torch.cat([p["pixel_values"] for p in preps]).to(dev)
            gs = []
            for i, p in enumerate(preps):
                g = p["image_grid_thw"]
                g = g[None] if g.ndim == 1 else g          # I1 ships (3,), IM ships (N,3)
                grids[i] = g
                gs.append(g)
            kw["image_grid_thw"] = torch.cat(gs).to(dev)

        hid = self.model(input_ids=ids, attention_mask=am, output_hidden_states=True,
                         use_cache=False, **kw).hidden_states[-1]
        keep_all = am.bool() & ~torch.isin(ids, self.boiler)

        dfeat = None
        if "dino_px" in preps[0]:
            px = [p["dino_px"] for p in preps]
            px = [x[None] if x.ndim == 3 else x for x in px]   # I1 (3,S,S) -> (1,3,S,S)
            nviews = [int(x.shape[0]) for x in px]
            d = torch.cat(px).to(dev, non_blocking=True)
            d = (d - self._dino_mean) / self._dino_std
            dfeat = self.dino.extract_features(d)               # (sum K, N_d, 1024)

        out, o = [], 0
        for i, p in enumerate(preps):
            mod = p.get("modality", "i1")
            n = lens[i]
            h, k = hid[i, :n], keep_all[i, :n]
            if mod == "t":
                # The captions cache was built with keep_only=True (build_vlm_cache_v22:
                # keep_only defaults to mode=="captions"), so its stored T entries are
                # ALREADY filtered. Matching that keeps the token counts identical.
                h, k = h[k], k[k]
            rec: Dict[str, torch.Tensor] = {"cond_hidden": h.to(torch.float16),
                                            "cond_keep_mask": k}
            if mod == "im":
                rec["qwen_view_ids"] = self._qwen_view_ids(ids[i, :n], grids[i])
            # Per-token patch index WITHIN its own view, -1 for text/structural.
            # Emitted for every image mode, not just IM, and kept SEPARATE from
            # qwen_view_ids on purpose: the presence of qwen_view_ids is what
            # switches on the qwen-side VIEW embedding, so reusing it here would
            # silently change i1's conditioning and break warm-start parity.
            if mod in ("i1", "im"):
                rec["qwen_img_pos"] = self._qwen_img_pos(ids[i, :n], grids[i])
            if dfeat is not None:
                kv = nviews[i]
                f = dfeat[o:o + kv]                            # (K, N_d, 1024)
                o += kv
                nd = f.shape[1]
                rec["dino_hidden"] = f.reshape(-1, f.shape[-1]).to(torch.float16)
                rec["dino_keep_mask"] = torch.ones(kv * nd, dtype=torch.bool, device=dev)
                rec["dino_view_ids"] = torch.arange(kv, dtype=torch.long,
                                                    device=dev).repeat_interleave(nd)
            if "views" in p:
                rec["views"] = p["views"]
            if "view" in p:
                rec["view"] = int(p["view"])
            out.append(rec)
        return out

    @torch.no_grad()
    def encode_paths(self, items: Sequence[tuple]) -> List[Dict[str, torch.Tensor]]:
        """Convenience: [(renders_dir, view), ...] → conds, doing the CPU half inline.
        Only for tests/benchmarks — in training the CPU half belongs in the workers."""
        return self.encode([self.prep(d, v) for d, v in items])
