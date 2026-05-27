"""trellis2_blip3o dataset for unified 3D generation.

Replaces BLIP3o's `LazySupervisedMixDataset` (webdataset-tar) with a JSON-manifest
based dataset that loads our TRELLIS-500K cached latents + caption + (optional)
multi-view codebook IDs for Setup B/C.

JSON schema (one line per object, e.g. data/index.jsonl):

    {
      "id": "<sha256>",
      "type": "I_2_3D",
      "image": "data/objxl_sketchfab/renders_cond/<sha>/000.png",
      "txt": "an oak chair, four legs",
      "target_ss_latent": "data/objxl_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16/<sha>.npz",
      "multi_view_renders": ["renders_cond/<sha>/000.png", ..., "renders_cond/<sha>/003.png"],
      "siglip_codebook_ids": "data/.../siglip_codebook_ids/<sha>.npz"   # only Setup B/C
    }

Three setups share the same dataset class; the `use_codebook` flag (read from
data_args) controls whether we inject `<I*>` tokens in the assistant section.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from . import _paths  # noqa: F401

# Import BLIP3o helpers — we keep their tokenize logic for Qwen3 conversation format
from blip3o.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # type: ignore
from blip3o.data.dataset import (  # type: ignore
    preprocess_multimodal,
    preprocess_qwen,
    target_transform,
)

# TRELLIS.2 — directly use their SparseTensor + collate_fn (zero-copy correctness)
from trellis2.modules.sparse import SparseTensor  # type: ignore
from trellis2.datasets.structured_latent import SLat  # type: ignore  (base class w/ shared collate_fn for Shape)
from trellis2.datasets.structured_latent_svpbr import SLatPbr  # type: ignore  (Tex w/ concat_cond stacking)

from .tr2_modules import (
    load_norm_stats,
    SS_FLOW_CONFIG_PATH,
    SHAPE_SLAT_CONFIG_PATH,
    TEX_SLAT_CONFIG_PATH,
)


class TR2BLIP3oDataset(Dataset):
    """JSON-manifest dataset for unified text-image → 3D generation.

    Args:
        tokenizer: HF tokenizer (Qwen3-VL based).
        data_path: path to JSONL manifest.
        data_args: HF DataArguments-style namespace. Should expose:
            - is_multimodal: bool
            - mm_use_im_start_end: bool
            - image_processor: vision tower image processor
            - use_codebook: bool   (NEW — our Setup B/C flag)
            - num_views: int       (NEW — N multi-view for codebook injection)
            - num_image_tokens: int (codebook size, for sanity check)
    """

    def __init__(self, tokenizer, data_path: str, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.use_codebook = getattr(data_args, "use_codebook", False)
        self.num_views = getattr(data_args, "num_views", 4)
        self.modality = torch.tensor(1)  # 1 = gen task

        # Load JSONL manifest
        self.records: List[Dict[str, Any]] = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
        print(f"[TR2BLIP3oDataset] loaded {len(self.records)} samples from {data_path}")

        # ── TRELLIS.2 per-stage normalization stats ──
        # Read directly from TRELLIS.2 stage config JSONs so we apply the
        # EXACT same (x - mean) / std as upstream training, ensuring the
        # pretrained checkpoints see the in-distribution target.
        # SS Flow has no normalization (upstream config omits the field).
        self.ss_norm        = load_norm_stats(SS_FLOW_CONFIG_PATH, "normalization")
        self.shape_norm     = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
        self.tex_pbr_norm   = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
        self.tex_shape_norm = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")

    def __len__(self):
        return len(self.records)

    def process_image(self, image: Image.Image):
        """Same signature as BLIP3o: returns (tensor, size, modality)."""
        proc = self.data_args.image_processor
        image_size = image.size
        image = proc.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, self.modality

    # ------------------------------------------------------------------
    # 3D target loaders — each mirrors TRELLIS.2's own get_instance EXACTLY
    # so preprocessing is identical to upstream training pipelines.
    # ------------------------------------------------------------------
    def process_target_ss_latent(self, path: str) -> torch.Tensor:
        """SS latent loader. Mirrors SparseStructureLatent.get_instance.

        Returns dense (8, 16, 16, 16) float tensor. Compat: TRELLIS.2 standard
        npz key is 'z'; our legacy data uses 'latent'. We accept either.
        """
        latent = np.load(path)
        npz_key = "z" if "z" in latent.files else "latent"
        z = torch.tensor(latent[npz_key]).float()
        if self.ss_norm is not None:
            # NOTE: SS Flow upstream config has no normalization, so this branch
            # is currently dead. Kept for forward-compat.
            z = (z - self.ss_norm["mean"]) / self.ss_norm["std"]
        return z

    def process_target_shape_slat(self, path: str) -> Dict[str, torch.Tensor]:
        """Shape SLAT loader. 1:1 mirror of SLat.get_instance (base class).

        Returns raw {coords, feats} (NOT a SparseTensor) — SLat.collate_fn will
        prepend the batch index column and wrap it into a SparseTensor.
        """
        data = np.load(path)
        coords = torch.tensor(data["coords"]).int()
        feats = torch.tensor(data["feats"]).float()
        if self.shape_norm is not None:
            feats = (feats - self.shape_norm["mean"]) / self.shape_norm["std"]
        return {"coords": coords, "feats": feats}

    def process_target_tex_slat(
        self,
        tex_path: str,
        shape_path: str,
    ) -> Dict[str, "SparseTensor"]:
        """Tex SLAT loader. 1:1 mirror of SLatPbr.get_instance.

        Loads BOTH PBR latent (the Tex SLAT target) AND Shape latent (used as
        `concat_cond` during Tex SLAT training — teacher forcing). Asserts
        coords match between the two (TRELLIS.2's own invariant).

        Returns {'x_0': pbr_st, 'concat_cond': shape_st} — both already wrapped
        as SparseTensor with a placeholder-zero batch column prepended (the
        actual batch index is filled in by SLatPbr.collate_fn via sparse_cat).
        """
        # ── PBR latent (the actual Tex SLAT target) ──
        data = np.load(tex_path)
        coords = torch.tensor(data["coords"]).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        feats = torch.tensor(data["feats"]).float()
        if self.tex_pbr_norm is not None:
            feats = (feats - self.tex_pbr_norm["mean"]) / self.tex_pbr_norm["std"]
        pbr_z = SparseTensor(feats, coords)

        # ── Shape latent (used as Tex's concat_cond, teacher-forced) ──
        data = np.load(shape_path)
        coords = torch.tensor(data["coords"]).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        feats = torch.tensor(data["feats"]).float()
        if self.tex_shape_norm is not None:
            feats = (feats - self.tex_shape_norm["mean"]) / self.tex_shape_norm["std"]
        shape_z = SparseTensor(feats, coords)

        assert torch.equal(shape_z.coords, pbr_z.coords), (
            f"Shape and PBR sparse coords differ: "
            f"{shape_z.coords.shape} vs {pbr_z.coords.shape}. Upstream encoder "
            f"is supposed to guarantee they share the same voxel layout."
        )
        return {"x_0": pbr_z, "concat_cond": shape_z}

    def _build_assistant_text(self, rec: Dict[str, Any]) -> str:
        """Setup A: just caption. Setup B/C: caption + <im_start><S0><V*><I*>...<im_end>."""
        caption = rec.get("txt", "")
        if not self.use_codebook:
            return caption

        # Setup B/C: inject codebook tokens
        codebook_path = rec.get("siglip_codebook_ids")
        if codebook_path is None or not os.path.exists(codebook_path):
            # Fallback for development: skip injection
            return caption

        codebook = np.load(codebook_path)["codebook"]  # shape (N_views, 729)
        N, K = codebook.shape
        assert N == self.num_views, f"codebook has {N} views but data_args.num_views={self.num_views}"

        parts: List[str] = [caption, "<im_start>", "<S0>"]
        for v in range(N):
            parts.append(f"<V{v}>")
            parts.extend(f"<I{int(tid)}>" for tid in codebook[v])
        parts.append("<im_end>")
        return "".join(parts)

    def __getitem__(self, i):
        rec = self.records[i]
        # Build the conversation
        rtype = rec.get("type", "I_2_3D")
        if rtype == "I_2_3D":
            user_text = f"{DEFAULT_IMAGE_TOKEN}\nPlease generate the 3D model. {rec.get('txt', '')}"
        elif rtype == "T_2_3D":
            user_text = f"Please generate the 3D model based on the caption: {rec.get('txt', '')}"
        else:
            raise ValueError(f"Unknown rec type: {rtype}")

        sources = [
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": self._build_assistant_text(rec)},
        ]

        # Preprocess multimodal (wraps gpt's <image> placeholder with <im_start><im_end> if enabled)
        sources = preprocess_multimodal(copy.deepcopy([sources]), self.data_args)

        has_image = "image" in rec
        tok_out = preprocess_qwen(sources, self.tokenizer, has_image=has_image)
        data: Dict[str, Any] = dict(input_ids=tok_out["input_ids"][0], labels=tok_out["labels"][0])

        # Image (input image, fed to vision_tower in user section)
        if has_image:
            img = Image.open(rec["image"]).convert("RGB")
            data["image"] = [self.process_image(img)]
            # BLIP3o expects a "target_image" tensor; we don't use it (no in-line VAE encode)
            # but we keep a dummy zero tensor to not break the collator's instances[0] check.
            data["target_image"] = [torch.zeros(3, 1, 1)]

        # ── 3D targets ──
        # SS Flow target (always present; required for any 3D training).
        data["target_ss_latent"] = self.process_target_ss_latent(rec["target_ss_latent"])

        # Shape SLAT target (optional in manifest; if missing the forward will
        # skip the Shape SLAT flow loss).
        if "target_shape_slat_512" in rec:
            data["target_shape_slat_512_item"] = self.process_target_shape_slat(
                rec["target_shape_slat_512"]
            )

        # Tex SLAT target (optional). Requires Shape latent path on the same
        # record since Tex's `concat_cond` is the GT Shape SLAT (TRELLIS.2
        # teacher-forces this during Tex SLAT training).
        if "target_tex_slat_512" in rec and "target_shape_slat_512" in rec:
            data["target_tex_slat_512_item"] = self.process_target_tex_slat(
                tex_path=rec["target_tex_slat_512"],
                shape_path=rec["target_shape_slat_512"],
            )

        data["id"] = rec.get("id", f"idx_{i}")
        return data


# ----------------------------------------------------------------------------
# Collator: extends BLIP3o's collator to also stack `target_ss_latent`
# ----------------------------------------------------------------------------
@dataclass
class TR2DataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(t, [0]) for t in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids + labels
        input_ids = [inst["input_ids"][: self.tokenizer.model_max_length] for inst in instances]
        labels = [inst["labels"][: self.tokenizer.model_max_length] for inst in instances]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch: Dict[str, Any] = dict(
            input_ids=input_ids,
            labels=labels.long() if labels.dtype == torch.int32 else labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # Images (vision tower input)
        if "image" in instances[0]:
            images = [inst["image"] for inst in instances]
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            batch["images"] = [im[0] for im_list in images for im in im_list]
            # NOTE: do NOT emit a dummy "target_images" — the port forward() has
            # no such kwarg (it uses target_ss/shape/tex latents instead) and would
            # raise TypeError on the unexpected keyword.

        # ── 3D targets ──
        # SS Flow target: dense (B, 8, 16, 16, 16). Plain torch.stack.
        batch["target_ss_latent"] = torch.stack(
            [inst["target_ss_latent"] for inst in instances], dim=0
        )

        # Shape SLAT target: sparse. Use TRELLIS.2's static collate_fn directly
        # — it prepends per-sample batch_idx column, cats coords/feats, wraps
        # into a SparseTensor, and registers layout cache. All identical to
        # what TRELLIS.2 uses in its own training.
        if "target_shape_slat_512_item" in instances[0]:
            shape_items = [inst["target_shape_slat_512_item"] for inst in instances]
            shape_pack = SLat.collate_fn(shape_items)
            batch["target_shape_slat_512"] = shape_pack["x_0"]

        # Tex SLAT target: sparse, two tensors (x_0 = PBR, concat_cond = Shape).
        # SLatPbr.collate_fn dispatches by value type — for SparseTensor it
        # calls sparse_cat which knows how to fix up the batch index column.
        if "target_tex_slat_512_item" in instances[0]:
            tex_items = [inst["target_tex_slat_512_item"] for inst in instances]
            tex_pack = SLatPbr.collate_fn(tex_items)
            batch["target_tex_slat_512"] = tex_pack["x_0"]            # PBR
            batch["tex_concat_cond"]     = tex_pack["concat_cond"]    # Shape (teacher-forced)

        return batch


def make_supervised_data_module(tokenizer, data_args):
    """Drop-in replacement for blip3o.data.dataset.make_supervised_data_module.

    Reads JSONL manifest at data_args.data_path. Three setups share this dataset;
    only differ in data_args.use_codebook / num_image_tokens.
    """
    train_dataset = TR2BLIP3oDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = TR2DataCollator(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# NOTE: registry wiring is done via direct edits in `blip3o/data/dataset.py`
# (`get_dataset_cls` / `make_supervised_data_module`). No monkey-patch needed.
