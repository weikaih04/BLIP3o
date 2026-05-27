# Active 3D model — USE THIS. TRELLIS.2 cascade (SS + Shape + Tex flows).
from blip3o.model.language_model.blip3o_qwen import blip3oQwenConfig, blip3oQwenForCausalLM
# ⚠️ REFERENCE ONLY — original BLIP3o 2D *image* gen/RL (Sana DiT). NOT used by
# the 3D train/inference path. Do not instantiate for 3D work. See the file headers.
from blip3o.model.language_model.blip3o_qwen_inference import blip3oQwenForInferenceLM
from blip3o.model.language_model.blip3o_qwen_grpo import blip3oQwenForGRPOLM
