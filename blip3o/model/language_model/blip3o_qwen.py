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

from trellis2_blip3o.loss import TRELLIS2FlowMatchingLoss


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


def _mask_drop(latents: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """Classifier-free-guidance dropout on cond (per-sample). Matches BLIP3o."""
    if drop_prob <= 0:
        return latents
    mask = torch.bernoulli(
        torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob
    )
    while len(mask.shape) < len(latents.shape):
        mask = mask.unsqueeze(-1)
    return latents * (1 - mask)


def _parse_flow_stage_weights(spec: str) -> Dict[str, float]:
    """Parse 'ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0' → {'ss':1.0, ...}.

    Used by the joint cascade loss formula:
        L_total = L_ce + λ_flow · Σ ŵ_i · L_flow_i
    where ŵ_i = w_i / Σ w_j (normalized weights, sum to 1).
    """
    out: Dict[str, float] = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"flow_stage_weights entry must look like 'name=value', got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


class blip3oQwenForCausalLM(Qwen3ForCausalLM, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "blip3o_qwen"
        config.rope_scaling = None

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Per-stage flow loss helpers (built lazily after config flags are set).
        # TRELLIS.2 trains each stage with a different t_schedule:
        #   - SS Flow    → logitNormal(μ=1, σ=1)   (more late-noise emphasis)
        #   - Shape SLAT → uniform                 (refines an already-decoded structure)
        #   - Tex SLAT   → uniform                 (same as Shape SLAT)
        # Using the wrong schedule per stage means the model samples timesteps
        # from a distribution different from what the pretrained checkpoint
        # saw during its original training → OOD finetune signal.
        self._flow_loss_fn_ss: Optional[TRELLIS2FlowMatchingLoss] = None
        self._flow_loss_fn_slat: Optional[TRELLIS2FlowMatchingLoss] = None

        self.post_init()

    def get_model(self):
        return self.model

    def _get_flow_loss_fn_ss(self) -> TRELLIS2FlowMatchingLoss:
        """Loss fn for SS Flow — TRELLIS.2 schedule: logitNormal(mean, std)."""
        if self._flow_loss_fn_ss is None:
            self._flow_loss_fn_ss = TRELLIS2FlowMatchingLoss(
                t_schedule="logitNormal",
                t_mean=getattr(self.config, "logitnorm_mean", 1.0),
                t_std=getattr(self.config, "logitnorm_std", 1.0),
                sigma_min=getattr(self.config, "flow_sigma_min", 1e-5),
            )
        return self._flow_loss_fn_ss

    def _get_flow_loss_fn_slat(self) -> TRELLIS2FlowMatchingLoss:
        """Loss fn for SLAT (Shape + Tex) — TRELLIS.2 schedule: uniform."""
        if self._flow_loss_fn_slat is None:
            self._flow_loss_fn_slat = TRELLIS2FlowMatchingLoss(
                t_schedule="uniform",
                sigma_min=getattr(self.config, "flow_sigma_min", 1e-5),
            )
        return self._flow_loss_fn_slat

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

            # Truncate to cond_max_length cap (sized for ~8-frame short video).
            cond_max_length = getattr(self.config, "cond_max_length", 8192)
            if cond_hidden.size(1) > cond_max_length:
                cond_hidden = cond_hidden[:, :cond_max_length, :]
                cond_key_mask = cond_key_mask[:, :cond_max_length]

            # Detach blocks flow_loss gradient from flowing back into the VLM
            # (Setup A / MolmoAct2 style: VLM only learns from text CE).
            # Setup B/C keep joint grad (BLIP3o-NEXT style).
            if getattr(self.config, "detach_cond", False):
                cond_hidden = cond_hidden.detach()

            cond = self.model.diffusion_connector(_mask_drop(cond_hidden))

            # Reshape mask to sdpa-friendly (B, 1, 1, L_kv). True = attend.
            sdpa_mask = cond_key_mask[:, None, None, :]

            # ============================================================
            # Joint 3-stage flow loss — shared cond_VLM across all stages.
            # Per-stage flow_loss math is identical to TRELLIS.2 upstream
            # (sample_t, diffuse, get_v, F.mse_loss — all imported from
            # trellis2.trainers.flow_matching). What's new is the joint
            # combination: one VLM forward → one cond → fed to 3 DiTs.
            #
            # IMPORTANT: Each stage uses its OWN t_schedule matching upstream:
            #   SS Flow → logitNormal(1,1); SLAT stages → uniform.
            # Using one shared schedule would put SLAT training off-distribution.
            # ============================================================
            loss_fn_ss   = self._get_flow_loss_fn_ss()
            loss_fn_slat = self._get_flow_loss_fn_slat()
            flow_losses: Dict[str, torch.Tensor] = {}
            flow_logs_per_stage: Dict[str, Dict] = {}

            # ----------------------------------------------------------------
            # DTYPE NOTE:
            # TRELLIS.2 datasets emit targets as float32 (their SparseStructure
            # Latent / SLat / SLatPbr all use `.float()`), but our pretrained
            # DiTs are bf16. We explicitly cast every target to `cond.dtype`
            # (bf16) so the math works WITHOUT relying on autocast — this is
            # symmetric with the SS Flow path which already did this.
            #
            # AUTOCAST IS STILL REQUIRED, though, because TRELLIS.2 internals
            # (specifically `TimestepEmbedder.timestep_embedding` in
            # sparse_structure_flow.py) hardcode `torch.arange(dtype=float32)`,
            # producing an fp32 t_freq that the bf16 t-embedder MLP can only
            # consume under amp. HF Trainer with `bf16=True` enables autocast
            # automatically. If you call this forward from custom code, wrap
            # the call site with `torch.autocast(device_type="cuda",
            # dtype=torch.bfloat16)` or expect a Float/BFloat16 mismatch.
            # ----------------------------------------------------------------

            # ── Stage 1: SS Flow (dense, logitNormal t-schedule) ──
            ss_flow = self.model.get_ss_flow()
            ss_target = target_ss_latent.to(cond.dtype)
            L_ss, log_ss = loss_fn_ss(ss_flow, ss_target, cond, cond_mask=sdpa_mask)
            flow_losses["ss"] = L_ss
            flow_logs_per_stage["ss"] = log_ss

            # ── Stage 2: Shape SLAT (sparse, uniform t-schedule) ──
            if target_shape_slat_512 is not None:
                shape_slat = self.model.get_shape_slat_512()
                # SparseTensor uses .replace() to swap feats while keeping coords.
                shape_target = target_shape_slat_512.replace(
                    target_shape_slat_512.feats.to(cond.dtype)
                )
                L_shape, log_shape = loss_fn_slat(
                    shape_slat, shape_target, cond, cond_mask=sdpa_mask
                )
                flow_losses["shape_slat_512"] = L_shape
                flow_logs_per_stage["shape_slat_512"] = log_shape

            # ── Stage 3: Tex SLAT (sparse, uniform t-schedule, with GT shape
            #     SLAT teacher-forced as concat_cond — TRELLIS.2 Tex SLAT
            #     input is channel-concat of (noisy_tex, shape_slat), 32+32=64). ──
            if target_tex_slat_512 is not None and tex_concat_cond is not None:
                tex_slat = self.model.get_tex_slat_512()
                tex_target = target_tex_slat_512.replace(
                    target_tex_slat_512.feats.to(cond.dtype)
                )
                tex_cc = tex_concat_cond.replace(
                    tex_concat_cond.feats.to(cond.dtype)
                )
                L_tex, log_tex = loss_fn_slat(
                    tex_slat, tex_target, cond,
                    cond_mask=sdpa_mask,
                    concat_cond=tex_cc,   # teacher-forced GT shape SLAT, bf16
                )
                flow_losses["tex_slat_512"] = L_tex
                flow_logs_per_stage["tex_slat_512"] = log_tex

            # ── Weighted-mean combination ──
            # L_total = L_ce + λ_flow · Σ ŵ_i · L_flow_i
            # Defaults: λ_flow=1.0, all w_i=1.0 → flow_combined = mean(L_flow_i),
            # matching BLIP3o-NEXT precedent of 1:1 ratio between CE and flow.
            stage_w_spec = getattr(
                self.config, "flow_stage_weights",
                "ss=1.0,shape_slat_512=1.0,tex_slat_512=1.0",
            )
            stage_weights_all = _parse_flow_stage_weights(stage_w_spec)
            active_w = {k: stage_weights_all.get(k, 1.0) for k in flow_losses}
            total_w = sum(active_w.values()) or 1.0
            normalized_w = {k: w / total_w for k, w in active_w.items()}
            flow_combined = sum(normalized_w[k] * v for k, v in flow_losses.items())

            flow_weight = getattr(self.config, "flow_weight", 1.0)
            if loss is None:
                loss = flow_weight * flow_combined
            else:
                loss = loss + flow_weight * flow_combined

            if torch.is_tensor(loss):
                stage_str = "  ".join(
                    f"{k}={flow_logs_per_stage[k]['flow_mse']:.4f}" for k in flow_logs_per_stage
                )
                rank0_print(
                    f"[loss] total={loss.detach().float().item():.4f}  flow_combined={flow_combined.detach().float().item():.4f}  {stage_str}"
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
