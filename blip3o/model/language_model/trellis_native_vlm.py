"""TrellisNativeVLM — native-VLM encoder → TRELLIS.2 cascade (continuous, no-discrete).

See QWEN35_VLM_DESIGN.md. Pipeline:

    text (+ optional image / multi-image / video)
        → native VLM (Qwen3-VL-2B or Qwen3.5-2B; vision baked in)  [encode_cond]
        → hidden_states[-1]  (2048-d, full sequence)
        → TRELLIS2Connector (2048 → 1024, UNCHANGED; DINOv3 dist-match)
        → cross-attn into TRELLIS SS / Shape-SLAT / Tex-SLAT flows  (weights reused)
        → SC-VAE decode  (inference only)
    Loss = 3-stage flow MSE (NO CE, NO discrete <I*> codebook).

Design choices (vs blip3oQwenForCausalLM):
  * Composition (HAS-A VLM), NOT inheritance from blip3oMeta (tangled w/ TA-Tok/codebook).
  * VLM is a frozen-or-trained *encoder* — no AR codebook generation, no CE.
  * Backbone-agnostic: `config.vlm_model` selects Qwen3-VL-2B vs Qwen3.5-2B
    (both LLM hidden = 2048, so the connector dim is unchanged for either).
  * Option α (keep TRELLIS cross-attn) — NOT a Qwen-Image-Edit MMDiT clone.

STATUS: SKELETON, UNTESTED. Pending Phase-0 env (qwen3_5 needs new transformers).
Inline TODO(phase0)/TODO(phaseN) mark the parts still to wire (save/load, processor
lives in the Phase-3 collator, video kwargs, AutoModel registration).
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import trellis2_blip3o._paths  # noqa: F401  — sets sys.path so trellis2 imports
from trellis2_blip3o.connector import TRELLIS2Connector
from trellis2_blip3o import flow_heads
from blip3o.model.multimodal_decoder.builder import (
    build_ss_flow,
    build_shape_slat_512,
    build_tex_slat_512,
    build_shape_slat_1024,
    build_tex_slat_1024,
    build_trellis_decoders,
)
from blip3o.utils import rank0_print

# Cross-attention dim of the TRELLIS flow DiTs (cond_channels=1024 across 4B stages).
TRELLIS_COND_DIM = 1024


class TrellisNativeVLMConfig(PretrainedConfig):
    model_type = "trellis_native_vlm"

    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen3.5-2B",
        vlm_hidden_size: int = 2048,          # Qwen3-VL-2B & Qwen3.5-2B both = 2048
        freeze_vlm: bool = True,              # v1 smoke-test: frozen (matches Qwen-Image-Edit)
        build_slat: bool = True,              # False ⇒ SS-only (skip shape/tex flows + decoders)
        slat_resolution: int = 512,           # 512 (paired w/ shape/tex_512 latents) or 1024 (HR cascade)
        detach_cond: bool = False,            # frozen ⇒ detach moot; kept for parity
        cond_max_length: int = 8192,
        mask_drop_prob: float = 0.1,
        cond_fusion: str = "none",            # "none" (hidden[-1]) | "penultimate" ([-2]) | "depthwise"
        fusion_layers: int = 0,               # depthwise: # of VLM layer outputs to fuse (0 = all)
        flow_weight: float = 1.0,
        flow_stage_weights: str = "ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0",
        logitnorm_mean: float = 1.0,
        logitnorm_std: float = 1.0,
        flow_sigma_min: float = 1e-5,
        # TRELLIS ckpt overrides (None → multimodal_decoder.builder defaults)
        trellis_ss_flow_ckpt: Optional[str] = None,
        trellis_shape_slat_ckpt: Optional[str] = None,
        trellis_tex_slat_ckpt: Optional[str] = None,
        trellis_sc_vae_ckpt: Optional[str] = None,
        **kwargs,
    ):
        self.vlm_model = vlm_model
        self.vlm_hidden_size = vlm_hidden_size
        self.freeze_vlm = freeze_vlm
        self.build_slat = build_slat
        self.slat_resolution = slat_resolution
        self.detach_cond = detach_cond
        self.cond_max_length = cond_max_length
        self.mask_drop_prob = mask_drop_prob
        self.cond_fusion = cond_fusion
        self.fusion_layers = fusion_layers
        self.flow_weight = flow_weight
        self.flow_stage_weights = flow_stage_weights
        self.logitnorm_mean = logitnorm_mean
        self.logitnorm_std = logitnorm_std
        self.flow_sigma_min = flow_sigma_min
        self.trellis_ss_flow_ckpt = trellis_ss_flow_ckpt
        self.trellis_shape_slat_ckpt = trellis_shape_slat_ckpt
        self.trellis_tex_slat_ckpt = trellis_tex_slat_ckpt
        self.trellis_sc_vae_ckpt = trellis_sc_vae_ckpt
        super().__init__(**kwargs)


class TrellisNativeVLMForConditionalGeneration(PreTrainedModel):
    config_class = TrellisNativeVLMConfig
    # The VLM submodule has its own _supports_* flags; we don't gate on them here.
    supports_gradient_checkpointing = True

    def __init__(self, config: TrellisNativeVLMConfig):
        super().__init__(config)

        # --- native VLM encoder (vision baked in) ---
        # TODO(phase0): verify AutoModelForImageTextToText resolves model_type
        #   'qwen3_5' (and 'qwen3_vl'); if the auto-mapping misses qwen3_5, import
        #   the concrete class (Qwen3_5ForConditionalGeneration) instead.
        # TODO(save/load): loading pretrained weights in __init__ is a skeleton
        #   shortcut. For clean save_pretrained/from_pretrained of the WHOLE
        #   composite, switch to building the VLM from a nested vlm_config and
        #   loading the backbone separately (or override save/load to exclude the
        #   frozen VLM). Fine for v1 training (we only save connector+flows anyway).
        from transformers import AutoModelForImageTextToText  # local import: new-ish auto class
        # `dtype=` (not the deprecated `torch_dtype=`) — works on transformers 4.57
        # (Qwen3-VL) AND 5.2+ (Qwen3.5, where torch_dtype is removed).
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.vlm_model, dtype=torch.bfloat16
        )
        if config.freeze_vlm:
            self.vlm.requires_grad_(False)
            self.vlm.eval()

        # Derive the LLM hidden dim from the loaded VLM (robust across backbones:
        # Qwen3-VL-2B/Qwen3.5-2B = 2048, Qwen3-VL-4B = 2560, etc.) rather than
        # trusting config.vlm_hidden_size. Falls back to the config value.
        try:
            text_cfg = self.vlm.config.get_text_config()
        except Exception:
            text_cfg = getattr(self.vlm.config, "text_config", self.vlm.config)
        vlm_hidden = getattr(text_cfg, "hidden_size", None) or config.vlm_hidden_size
        config.vlm_hidden_size = int(vlm_hidden)  # keep config in sync for save/reload

        # --- TRELLIS cascade (these builders LOAD the pretrained TRELLIS ckpts) ---
        self.ss_flow = build_ss_flow(config)            # trainable 1.3B SS Flow
        if config.build_slat:
            # Attribute names keep the "_512" suffix for cross-file compatibility; the
            # underlying flow is the 1024 variant when slat_resolution=1024 (manifest
            # must point target_shape_slat_512/tex_slat_512 to the 1024 latent paths).
            _res = int(getattr(config, "slat_resolution", 512))
            if _res == 1024:
                self.shape_slat_512 = build_shape_slat_1024(config)
                self.tex_slat_512 = build_tex_slat_1024(config)
            else:
                self.shape_slat_512 = build_shape_slat_512(config)
                self.tex_slat_512 = build_tex_slat_512(config)
            rank0_print(f"[slat] resolution={_res} → shape/tex flows = "
                        + ("1024 variant" if _res == 1024 else "512 variant"))
            self.trellis_decoders = build_trellis_decoders(config)  # frozen, inference-only
        else:
            # SS-only training: skip the SLAT flows + decoders (lighter, faster).
            # forward()/compute_cascade_flow_loss already guard these as None.
            self.shape_slat_512 = None
            self.tex_slat_512 = None
            self.trellis_decoders = None

        # --- connector (UNCHANGED MLP; input dim auto-matched to the VLM) ---
        self.diffusion_connector = TRELLIS2Connector(
            vlm_hidden_dim=config.vlm_hidden_size,
            trellis_cond_dim=TRELLIS_COND_DIM,
        )

        # --- per-stage flow loss fns (shared helper) ---
        self._loss_fn_ss, self._loss_fn_slat = flow_heads.build_flow_loss_fns(config)

        # --- depth-wise Semantic Routing (optional): per-block fusion over VLM layers ---
        # One router PER flow (each is a separate DiT); the connector is SHARED. Each flow
        # block is wrapped so block d uses router._cond_per_block[d] (cond is args[2]).
        self.cond_fusion = getattr(config, "cond_fusion", "none")
        self._fusion_layers = getattr(config, "fusion_layers", 0)  # 0 = all VLM layer outputs
        if self.cond_fusion == "depthwise":
            from trellis2_blip3o.depth_fusion import DepthFusionRouter, install_depth_routing
            try:
                tcfg = self.vlm.config.get_text_config()
            except Exception:
                tcfg = getattr(self.vlm.config, "text_config", self.vlm.config)
            n_vlm = int(getattr(tcfg, "num_hidden_layers", 24))
            L = n_vlm if self._fusion_layers in (0, None) else min(self._fusion_layers, n_vlm)
            self._fuse_L = L
            self.ss_router = DepthFusionRouter(len(self.ss_flow.blocks), L, self.diffusion_connector)
            install_depth_routing(self.ss_flow, self.ss_router)
            if config.build_slat:
                self.shape_router = DepthFusionRouter(len(self.shape_slat_512.blocks), L, self.diffusion_connector)
                self.tex_router = DepthFusionRouter(len(self.tex_slat_512.blocks), L, self.diffusion_connector)
                install_depth_routing(self.shape_slat_512, self.shape_router)
                install_depth_routing(self.tex_slat_512, self.tex_router)
            rank0_print(f"[depthwise] routing on: fuse last {L} VLM layers → per-block "
                        f"(SS {len(self.ss_flow.blocks)} blocks"
                        + (f", Shape/Tex too" if config.build_slat else "") + ")")

    # HF's `--gradient_checkpointing True` only enables GC on the VLM. The TRELLIS flow
    # blocks have their OWN `use_checkpoint` (default False) — without this propagation,
    # the flow activations dominate memory and OOM at higher latent resolution (1024).
    # Overriding enable/disable so the flag actually flows through.
    def _set_flow_checkpointing(self, enable: bool):
        for name in ("ss_flow", "shape_slat_512", "tex_slat_512"):
            flow = getattr(self, name, None)
            if flow is None:
                continue
            for b in getattr(flow, "blocks", []):
                # depthwise wraps each block in _CondInjectBlock(.block=orig); unwrap.
                target = getattr(b, "block", b)
                if hasattr(target, "use_checkpoint"):
                    target.use_checkpoint = bool(enable)

    def gradient_checkpointing_enable(self, **kwargs):
        super().gradient_checkpointing_enable(**kwargs)
        self._set_flow_checkpointing(True)

    def gradient_checkpointing_disable(self):
        super().gradient_checkpointing_disable()
        self._set_flow_checkpointing(False)

    # ------------------------------------------------------------------
    # Core irreducible piece: run the VLM → full last-layer hidden + mask.
    # The SAME method a future offline caching script would call (caching is a
    # deferred, purely-additive optimization — see QWEN35_VLM_DESIGN.md §9).
    # ------------------------------------------------------------------
    def encode_cond(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ):
        """[text (+ image/multi-image/video)] → (cond_hidden (B,T,H), cond_key_mask (B,T)).
        If return_layers: returns (list of selected VLM layer hiddens, mask) for depth-wise fusion.

        Inputs are exactly what the native `AutoProcessor` emits (Phase-3 collator).
        Uses `hidden_states[-1]` (last layer) — matches Qwen-Image-Edit. Drops nothing
        in v1 (keep full hidden; template-prefix `drop_idx` is deferred polish).
        """
        # Only build the kwargs the VLM actually expects (avoid passing None video
        # tensors that some processors/models dislike).
        vlm_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if pixel_values is not None:
            vlm_inputs["pixel_values"] = pixel_values
            vlm_inputs["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            vlm_inputs["pixel_values_videos"] = pixel_values_videos
            vlm_inputs["video_grid_thw"] = video_grid_thw

        ctx = torch.no_grad() if self.config.freeze_vlm else contextlib.nullcontext()
        with ctx:
            out = self.vlm(**vlm_inputs)

        hs = out.hidden_states  # tuple: (embedding, layer_1, ..., layer_N)
        ref = hs[-1]
        if attention_mask is not None:
            cond_key_mask = attention_mask.bool().to(ref.device)
        else:
            cond_key_mask = torch.ones(ref.shape[:2], dtype=torch.bool, device=ref.device)

        if return_layers:
            # depth-wise fusion: the last L layer OUTPUTS (exclude the embedding hs[0]).
            layers = list(hs[1:])
            L = getattr(self, "_fuse_L", len(layers))
            if 0 < L < len(layers):
                layers = layers[-L:]
            return layers, cond_key_mask

        # single-layer cond: "penultimate" → hs[-2], else last layer hs[-1] (Qwen-Image default).
        idx = -2 if getattr(self.config, "cond_fusion", "none") == "penultimate" else -1
        return hs[idx], cond_key_mask

    def cond_and_null(self, cond_hidden: torch.Tensor):
        """(cond, neg_cond) for CFG sampling. cond = connector(hidden);
        neg_cond = connector(0) = the training-time unconditional (see
        flow_heads.null_cond_like — NOT zeros in cond space). Any generate()/
        sampling path MUST use this for the CFG negative branch."""
        cond = self.diffusion_connector(cond_hidden)
        neg_cond = flow_heads.null_cond_like(self.diffusion_connector, cond_hidden)
        return cond, neg_cond

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        # 3D flow targets (from the existing prep; collator attaches these)
        target_ss_latent: Optional[torch.FloatTensor] = None,
        target_shape_slat_512: Optional[Any] = None,
        target_tex_slat_512: Optional[Any] = None,
        tex_concat_cond: Optional[Any] = None,
        # cached-cond fast path (deferred feature; v1 leaves these None → run VLM)
        cond_hidden: Optional[torch.Tensor] = None,
        cond_key_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if target_ss_latent is None:
            raise ValueError("TrellisNativeVLM.forward requires target_ss_latent (training).")

        # 1. cond: either run the VLM (v1) or use precomputed cached cond (deferred).
        depthwise = (getattr(self, "cond_fusion", "none") == "depthwise") and (cond_hidden is None)
        _mask_drop = self.config.mask_drop_prob
        if depthwise:
            # Depth-wise Semantic Routing: get all VLM layer hiddens, set each flow's per-block
            # cond (the wrappers inject it; the cond passed below is a dummy, overridden). CFG
            # dropout is off on this path in v1 (handled at the layer-stack level later).
            layer_hiddens, cond_key_mask = self.encode_cond(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw,
                return_layers=True,
            )
            self.ss_router.set_cond(layer_hiddens)
            if self.config.build_slat:
                self.shape_router.set_cond(layer_hiddens)
                self.tex_router.set_cond(layer_hiddens)
            cond_hidden = layer_hiddens[-1]   # dummy for the shared helper (per-block overrides it)
            _mask_drop = 0.0
        elif cond_hidden is None:
            cond_hidden, cond_key_mask = self.encode_cond(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        # 2. cond → connector → 3-stage TRELLIS cascade flow loss (shared helper).
        loss, logs = flow_heads.compute_cascade_flow_loss(
            connector=self.diffusion_connector,
            ss_flow=self.ss_flow,
            shape_slat=self.shape_slat_512,
            tex_slat=self.tex_slat_512,
            loss_fn_ss=self._loss_fn_ss,
            loss_fn_slat=self._loss_fn_slat,
            cond_hidden=cond_hidden,
            cond_key_mask=cond_key_mask,
            target_ss_latent=target_ss_latent,
            target_shape_slat_512=target_shape_slat_512,
            target_tex_slat_512=target_tex_slat_512,
            tex_concat_cond=tex_concat_cond,
            cond_max_length=self.config.cond_max_length,
            detach_cond=self.config.detach_cond,
            mask_drop_prob=_mask_drop,
            flow_stage_weights=self.config.flow_stage_weights,
            flow_weight=self.config.flow_weight,
        )

        if depthwise:   # release stashed per-block conds
            self.ss_router.clear()
            if self.config.build_slat:
                self.shape_router.clear(); self.tex_router.clear()

        stage_str = "  ".join(f"{k}={v:.4f}" for k, v in logs["stages"].items())
        rank0_print(f"[loss] total={loss.detach().float().item():.4f}  {stage_str}")

        # No logits (no LM head / no CE) — loss-only output for HF Trainer.
        return CausalLMOutputWithPast(loss=loss, logits=None)


# TODO(phaseN): register with Auto classes once save/load story is finalized.
# from transformers import AutoConfig, AutoModel
# AutoConfig.register("trellis_native_vlm", TrellisNativeVLMConfig)
# AutoModel.register(TrellisNativeVLMConfig, TrellisNativeVLMForConditionalGeneration)
