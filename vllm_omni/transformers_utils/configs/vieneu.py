"""VieNeu-TTS config registration with transformers AutoConfig."""

from transformers import AutoConfig

from vllm_omni.model_executor.models.vieneu.config import (
    VieNeuCodecConfig,
    VieNeuConfig,
    VieNeuTalkerConfig,
)

AutoConfig.register("vieneu", VieNeuConfig)
AutoConfig.register("vieneu_talker", VieNeuTalkerConfig)
AutoConfig.register("vieneu_codec", VieNeuCodecConfig)

__all__ = ["VieNeuConfig", "VieNeuTalkerConfig", "VieNeuCodecConfig"]
