"""blip3oQwenForCausalLM — forked from BLIP3o-NEXT.

trellis2_blip3o changes:
  * The diffusion-side loss is now TRELLIS.2 SS Flow (3D sparse-structure),
    not Sana DiT (2D image).
  * The flow loss formula is imported from `trellis2.trainers.flow_matching`
    via `trellis2_blip3o.loss.TRELLIS2FlowMatchingLoss`, so any upstream
    update to TRELLIS.2 sigma path / t schedule flows through automatically.
  * Forward consumes a precomputed `target_ss_latent` tensor (the cached
    SS Flow target, shape (B, 8, 16, 16, 16)) instead of `target_images`.
  * Cond slicing is config-switchable via `config.cond_slice`:
      - "full"        — feed entire hidden_states as cond (Setup A / C)
      - "image_block" — slice <im_start>+1 : <im_end> per sample (Setup B)
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from blip3o.model.blip3o_arch import blip3oMetaForCausalLM, blip3oMetaModel
from blip3o.utils import rank0_print

from trellis2_blip3o import flow_heads


class blip3oQwenConfig(Qwen3Config):
    model_type = "blip3o_qwen"


class blip3oQwenModel(blip3oMetaModel, Qwen3Model):
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen3Config):
        super(blip3oQwenModel, self).__init__(config)


def _slice_cond_image_block(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    image_start_tag_id: int,
    image_end_tag_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample slice hidden[start+1:end] where start/end are <im_start>/<im_end>.

    Right-pads to the max slice length across the batch. If a sample lacks the
    tags (no codebook injected), falls back to its last 730 tokens (BLIP3o-NEXT
    convention: 1 scale token + 729 codebook tokens).

    Returns:
        out: (B, max_len, H) padded cond hidden.
        mask: (B, max_len) bool, True = real cond token, False = padding.
    """
    B = hidden_states.size(0)
    selected = []
    for b in range(B):
        lab = labels[b]
        sm = (lab == image_start_tag_id).nonzero(as_tuple=False)
        em = (lab == image_end_tag_id).nonzero(as_tuple=False)
        if sm.numel() > 0 and em.numel() > 0:
            s, e = sm[0].item() + 1, em[0].item()
            h_slice = hidden_states[b, s:e, :]
        else:
            h_slice = hidden_states[b, -730:, :]
        selected.append(h_slice)
    max_len = max(h.size(0) for h in selected)
    H = hidden_states.size(-1)
    out = hidden_states.new_zeros(B, max_len, H)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=hidden_states.device)
    for b, h in enumerate(selected):
        n = h.size(0)
        out[b, :n, :] = h
        mask[b, :n] = True
    return out, mask


class blip3oQwenForCausalLM(Qwen3ForCausalLM, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "blip3o_qwen"
        config.rope_scaling = None

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Per-stage flow loss fns (SS=logitNormal, SLAT=uniform), built lazily via
        # the shared flow_heads helper — SINGLE SOURCE OF TRUTH also used by
        # trellis_native_vlm. (Verified numerically identical to the previous
        # inline loop: tests/test_flow_heads_equiv.py.)
        self._flow_fns = None

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        target_ss_latent: Optional[torch.FloatTensor] = None,
        target_shape_slat_512: Optional[Any] = None,   # SparseTensor or None
        target_tex_slat_512: Optional[Any] = None,     # SparseTensor or None (PBR target)
        tex_concat_cond: Optional[Any] = None,         # SparseTensor — GT shape SLAT, teacher-forced into Tex SLAT
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # ---- 1. multimodal prep + LM forward (unchanged from upstream) ----
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels,
                images, modalities, image_sizes,
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # ---- 2. CE loss on shifted labels (text + optional <I*> codebook tokens) ----
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

        # ---- 3. SS Flow loss (only when 3D target supplied) ----
        if target_ss_latent is not None:
            cond_slice = getattr(self.config, "cond_slice", "full")
            if cond_slice == "image_block":
                # Fixed-length block per sample, padded mask only marks the per-sample
                # valid prefix (in case some samples lack the codebook).
                # Defensive: image_block slicing needs `labels` to locate the
                # <im_start> / <im_end> tags. If we ever wire this into an
                # inference path without labels, error loudly instead of
                # tripping over `labels[b]` on a NoneType.
                if labels is None:
                    raise ValueError(
                        "cond_slice='image_block' requires `labels` to locate "
                        "<im_start>/<im_end> tags, but labels=None. Either pass "
                        "labels (training mode) or switch to cond_slice='full'."
                    )
                cond_hidden, cond_key_mask = _slice_cond_image_block(
                    hidden_states,
                    labels,
                    image_start_tag_id=self.config.image_start_tag_id,
                    image_end_tag_id=self.config.image_end_tag_id,
                )
            elif cond_slice == "full":
                # Full VLM hidden; mask comes from attention_mask returned by
                # prepare_inputs_labels_for_multimodal (1 = real token, 0 = pad).
                cond_hidden = hidden_states
                if attention_mask is not None:
                    cond_key_mask = attention_mask.bool().to(hidden_states.device)
                else:
                    cond_key_mask = torch.ones(
                        hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
                    )
            else:
                raise ValueError(f"Unknown cond_slice: {cond_slice}")

            # 3-stage cascade flow loss via the SHARED helper (single source of
            # truth, also used by trellis_native_vlm). Does truncate → detach →
            # connector(mask_drop) → sdpa_mask → SS/Shape/Tex stages → weighted
            # combine internally. Numerically identical to the previous inline
            # loop (tests/test_flow_heads_equiv.py: diff=0). cond_slice above
            # already produced (cond_hidden, cond_key_mask).
            if self._flow_fns is None:
                self._flow_fns = flow_heads.build_flow_loss_fns(self.config)
            loss_fn_ss, loss_fn_slat = self._flow_fns
            flow_loss, flow_logs = flow_heads.compute_cascade_flow_loss(
                connector=self.model.diffusion_connector,
                ss_flow=self.model.get_ss_flow(),
                shape_slat=(self.model.get_shape_slat_512()
                            if target_shape_slat_512 is not None else None),
                tex_slat=(self.model.get_tex_slat_512()
                          if (target_tex_slat_512 is not None and tex_concat_cond is not None) else None),
                loss_fn_ss=loss_fn_ss, loss_fn_slat=loss_fn_slat,
                cond_hidden=cond_hidden, cond_key_mask=cond_key_mask,
                target_ss_latent=target_ss_latent,
                target_shape_slat_512=target_shape_slat_512,
                target_tex_slat_512=target_tex_slat_512,
                tex_concat_cond=tex_concat_cond,
                cond_max_length=getattr(self.config, "cond_max_length", 8192),
                detach_cond=getattr(self.config, "detach_cond", False),
                mask_drop_prob=0.1,
                flow_stage_weights=getattr(self.config, "flow_stage_weights",
                                           "ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0"),
                flow_weight=getattr(self.config, "flow_weight", 1.0),
            )
            loss = flow_loss if loss is None else loss + flow_loss

            if torch.is_tensor(loss):
                stage_str = "  ".join(f"{k}={v:.4f}" for k, v in flow_logs["stages"].items())
                rank0_print(
                    f"[loss] total={loss.detach().float().item():.4f}  "
                    f"flow_combined={flow_logs['flow_combined']:.4f}  {stage_str}"
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("blip3o_qwen", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForCausalLM)
