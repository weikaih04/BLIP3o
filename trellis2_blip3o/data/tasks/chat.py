"""Chat / LM-loss tasks: VQA, grounding, and pure-text SFT.

All three share:
  - JSONL schema: one conversation per line, optional image path
  - chat-template tokenization via the native VLM processor
  - cross-entropy loss on assistant tokens only (user/system masked to -100)

They differ only in WHICH MANIFEST they read — the loss path is identical.
That's why one class is registered under three task names. Pure NLP just omits
the `image` field; everything else is the same.

JSONL schema (flexible; accepts two common formats):

    # canonical ("role/content")
    {"conversations": [
        {"role": "user",      "content": "What's in the picture?"},
        {"role": "assistant", "content": "A cat sitting on a mat."}
     ],
     "image": "/abs/path/to/cat.jpg"}    # optional

    # LLaVA-style ("from/value") — auto-converted
    {"image": "/abs/.../cat.jpg",
     "conversations": [
        {"from": "human", "value": "<image>\\nWhat's in the picture?"},
        {"from": "gpt",   "value": "A cat sitting on a mat."}
     ]}

Multi-turn supported: loss is on the LAST assistant turn. Multi-turn loss on
ALL assistant turns is a small extension (mask per-turn instead of once at the
end); left as a flag-gated future change.

Important: for the LM gradient to flow through the VLM, train_native must run
with `--freeze_vlm False`. With freeze_vlm=True the loss decreases only if
something downstream (e.g. an LM-head adapter) is trainable; we surface a
warning at startup.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..registry import register_task
# Image-budget helpers: SHARED with the 3D collate path (was a third drifted copy —
# single source of truth lives in trellis2_blip3o/vlm_collate.py).
from ...vlm_collate import cap_image as _cap_image
from ...vlm_collate import px_per_tok as _px_per_tok


IGNORE_INDEX = -100


# ────────────────────────────────────────────────────────────────────────────
# Helpers: schema normalization + image cap
# ────────────────────────────────────────────────────────────────────────────
def _normalize_conversation(conv: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """LLaVA `{from, value}` → canonical `{role, content}`; otherwise pass-through.
    Strips inline `<image>` tokens — image attachment is via the separate `image`
    field, and the processor inserts image markers from the chat template."""
    out: List[Dict[str, str]] = []
    for msg in conv:
        if "role" in msg and "content" in msg:
            role = msg["role"]
            content = msg["content"]
        elif "from" in msg and "value" in msg:
            mp = {"human": "user", "gpt": "assistant", "system": "system"}
            role = mp.get(msg["from"], msg["from"])
            content = msg["value"]
        else:
            raise ValueError(f"unrecognized conversation turn: {msg}")
        if isinstance(content, str):
            content = content.replace("<image>", "").strip()
        out.append({"role": role, "content": content})
    return out


# ────────────────────────────────────────────────────────────────────────────
# Shared dataset
# ────────────────────────────────────────────────────────────────────────────
class _ChatTaskBase(Dataset):
    """JSONL-driven chat dataset; emits raw conversation + optional PIL image.
    All tokenization + label masking happens in `collate_fn`.

    Init args (from yaml `args:`):
      manifest        : path to chat JSONL.
      image_root      : optional prefix prepended to relative `image` paths.
      max_image_px    : cap (W·H) per image; default ≈ 4096 vision tokens
                        (4096 · px_per_tok). Set to 0 to disable.
      require_image   : drop rows without an image field (e.g. for VQA-only mix).
      drop_no_assistant : drop rows whose last turn isn't an assistant message
                        (no learnable target).
    """

    task_name: str = ""

    def __init__(
        self,
        manifest: str,
        image_root: Optional[str] = None,
        max_image_tokens: int = 4096,
        require_image: bool = False,
        drop_no_assistant: bool = True,
    ):
        super().__init__()
        self.manifest_path = manifest
        self.image_root = image_root
        self.max_image_tokens = int(max_image_tokens)
        self.require_image = bool(require_image)

        records: List[Dict[str, Any]] = []
        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        before = len(records)
        kept = []
        for r in records:
            conv = r.get("conversations") or r.get("messages") or []
            if not conv:
                continue
            conv = _normalize_conversation(conv)
            if drop_no_assistant and (not conv or conv[-1]["role"] != "assistant"):
                continue
            r["_conv"] = conv
            if self.require_image and not r.get("image"):
                continue
            kept.append(r)
        self.records = kept
        print(f"[{self.task_name or self.__class__.__name__}] {manifest}: "
              f"{before} → {len(self.records)} after filters "
              f"(require_image={self.require_image})")

    def __len__(self):
        return len(self.records)

    def _resolve_image(self, p: str) -> str:
        if not p:
            return p
        if self.image_root and not p.startswith("/"):
            import os
            return os.path.join(self.image_root, p)
        return p

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.records[i]
        image = None
        img_path = rec.get("image")
        if img_path:
            image = Image.open(self._resolve_image(img_path)).convert("RGB")
        return {
            "_task": self.task_name,
            "conv": rec["_conv"],
            "image": image,         # PIL.Image | None
            "id": rec.get("id", f"idx_{i}"),
        }

    # ------------------------------------------------------------------
    # Collator: chat template → input_ids + labels (CE on assistant tokens).
    #
    # Algorithm (scan-and-mask; matches LLaMA-Factory's default `qwen3` SFT):
    #   1. Build full chat via processor.apply_chat_template(add_generation_prompt=False);
    #      attach the image (if any) to the FIRST user turn (LLaVA convention).
    #   2. processor(text=full_text, images=images) → input_ids w/ <|image_pad|>
    #      expanded to grid_thw tokens.
    #   3. Scan each row for occurrences of `<|im_start|>assistant\n`. For each,
    #      find the next `<|im_end|>` and unmask THAT span only (i.e. the
    #      assistant's CONTENT plus the closing `<|im_end|>`).
    #   → Multi-turn: ALL assistant turns are supervised.
    #   → Image_pad tokens are inside user turns → masked (not supervised).
    #   → Padding gets masked at the end.
    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, Any]], processor) -> Dict[str, Any]:
        px_per_tok = _px_per_tok(processor)
        max_image_tokens = 4096  # default per-image cap; raise as needed

        full_texts: List[str] = []
        flat_images: List[Image.Image] = []

        for inst in batch:
            conv = inst["conv"]
            img = inst.get("image")

            # Attach image to the FIRST user turn (LLaVA-Instruct convention).
            # Multi-turn convs about a single image work; multi-image is out of scope.
            first_user = _first_user_index(conv)
            msgs: List[Dict[str, Any]] = []
            for j, msg in enumerate(conv):
                if msg["role"] == "user" and img is not None and j == first_user:
                    msgs.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": msg["content"]},
                        ],
                    })
                else:
                    msgs.append(msg)

            full_texts.append(processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
            if img is not None:
                if max_image_tokens > 0:
                    img = _cap_image(img, max_image_tokens * px_per_tok)
                flat_images.append(img)

        proc_kwargs = dict(text=full_texts, padding=True, return_tensors="pt")
        if flat_images:
            proc_kwargs["images"] = flat_images
        enc = processor(**proc_kwargs)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # ── Scan-and-mask ──
        labels = _build_supervised_labels(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
        )

        # Sanity: at least one supervised token per row (otherwise CE → NaN).
        nz = (labels != IGNORE_INDEX).any(dim=1)
        if not bool(nz.all()):
            import warnings
            warnings.warn(
                f"[chat collator] {(~nz).sum().item()} row(s) had ZERO supervised "
                f"tokens (truncation? bad chat format?). Those rows won't update."
            )

        out: Dict[str, Any] = {
            "_task": batch[0]["_task"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in enc:
            out["pixel_values"] = enc["pixel_values"]
            out["image_grid_thw"] = enc.get("image_grid_thw")
        return out


def _first_user_index(conv: Sequence[Dict[str, str]]) -> int:
    for i, m in enumerate(conv):
        if m["role"] == "user":
            return i
    return -1


# Cache role-marker token sequences per tokenizer to avoid re-encoding every batch.
_ROLE_MARKER_CACHE: Dict[int, Dict[str, Any]] = {}


def _get_role_markers(tokenizer):
    """Tokenize the role-header strings ONCE per tokenizer:
      assistant_header = encode("<|im_start|>assistant\n")
      im_start_id      = id of "<|im_start|>" (any role's start marker)
    Scan stops at the next <|im_start|> — that span includes the assistant's
    content + closing <|im_end|> + trailing '\n' (exactly what LF supervises
    via `format_assistant = "{content}<|im_end|>\n"`).
    """
    key = id(tokenizer)
    if key in _ROLE_MARKER_CACHE:
        return _ROLE_MARKER_CACHE[key]
    ass_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if im_start_id is None or im_start_id < 0:
        raise RuntimeError(
            "tokenizer has no <|im_start|> token; this collator targets the ChatML "
            "format (Qwen/LLaMA-3-Instruct/etc.). Pass a different collator for "
            "non-ChatML chat templates."
        )
    out = {"assistant_header": ass_header, "im_start_id": int(im_start_id)}
    _ROLE_MARKER_CACHE[key] = out
    return out


def _find_subsequences(haystack: List[int], needle: List[int]) -> List[int]:
    """All starting indices in `haystack` where `needle` appears (non-overlapping)."""
    out = []
    n = len(needle)
    if n == 0 or n > len(haystack):
        return out
    i = 0
    while i <= len(haystack) - n:
        if haystack[i:i + n] == needle:
            out.append(i)
            i += n
        else:
            i += 1
    return out


def _build_supervised_labels(input_ids, attention_mask, tokenizer):
    """Per-row scan: find every `<|im_start|>assistant\n` header, unmask the
    span from right-after-the-header to right-before-the-next-`<|im_start|>`
    (or to the end of the attention region). That span is the assistant's
    CONTENT + closing `<|im_end|>` + trailing `\n` — exactly what LF's
    `format_assistant = "{content}<|im_end|>\n"` supervises.

    Padding / user / system / image_pad-inside-user stays IGNORE_INDEX.
    Supervises ALL assistant turns (matches LLaMA-Factory `qwen3`
    default with `mask_history=False`).
    """
    import torch
    markers = _get_role_markers(tokenizer)
    ass_header: List[int] = markers["assistant_header"]
    im_start_id: int = markers["im_start_id"]
    H = len(ass_header)

    B, T = input_ids.shape
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    for b in range(B):
        row = input_ids[b].tolist()
        # Honor the attention region: never count padding as part of a turn.
        attn_end = int(attention_mask[b].sum())
        for start in _find_subsequences(row[:attn_end], ass_header):
            content_start = start + H
            # Unmask up to (but not including) the NEXT <|im_start|>, or end of attn.
            end = content_start
            while end < attn_end and row[end] != im_start_id:
                end += 1
            for j in range(content_start, end):
                labels[b, j] = input_ids[b, j]
        # Padding stays masked.
        labels[b, attention_mask[b] == 0] = IGNORE_INDEX

    return labels


# ────────────────────────────────────────────────────────────────────────────
# Three registered task classes (semantics identical; names differ for config)
# ────────────────────────────────────────────────────────────────────────────
@register_task("vqa")
class VQADataset(_ChatTaskBase):
    """Image+question → text. JSONL must include `image` field (require_image=True
    by default)."""
    def __init__(self, manifest, image_root=None, max_image_tokens=4096,
                 require_image=True, drop_no_assistant=True):
        super().__init__(manifest, image_root, max_image_tokens, require_image, drop_no_assistant)


@register_task("grounding")
class GroundingDataset(_ChatTaskBase):
    """Image+phrase → bbox text. Same loss as VQA, different data."""
    def __init__(self, manifest, image_root=None, max_image_tokens=4096,
                 require_image=True, drop_no_assistant=True):
        super().__init__(manifest, image_root, max_image_tokens, require_image, drop_no_assistant)


@register_task("text_sft")
class TextSFTDataset(_ChatTaskBase):
    """Pure-text instruction following. No image; preserves VLM LM ability when
    fine-tuning with vision tasks (anti-catastrophic-forget regularizer)."""
    def __init__(self, manifest, image_root=None, max_image_tokens=4096,
                 require_image=False, drop_no_assistant=True):
        super().__init__(manifest, image_root, max_image_tokens, require_image, drop_no_assistant)
