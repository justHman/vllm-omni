"""VieNeu-TTS-v2 architecture registration (TASK 6).

Existing vllm-omni families register all their architecture->class mappings
directly inline in the single central dict
``vllm_omni.model_executor.models.registry._OMNI_MODELS``
(docs/Architecture.md Part A.2) -- there is no per-model ``registry.py``
convention among fish_speech/cosyvoice3/qwen3_tts. This module exists
anyway, per TASK 4/6's requested layout, as the **source of truth** for
VieNeu's entries: the central registry imports ``VIENEU_MODELS`` from here
and merges it in (see the edit to ``model_executor/models/registry.py``),
rather than duplicating the mapping in two places. No monkey-patching is
involved -- this is a plain dict merged at import time, exactly like every
other family's entries.
"""

from __future__ import annotations

# (mod_folder, mod_relname, cls_name) tuples, same shape as
# vllm_omni.model_executor.models.registry._OMNI_MODELS entries.
VIENEU_MODELS: dict[str, tuple[str, str, str]] = {
    "VieNeuTalkerForConditionalGeneration": (
        "vieneu",
        "generation",
        "VieNeuTalkerForConditionalGeneration",
    ),
    "VieNeuCodecDecoder": (
        "vieneu",
        "codec",
        "VieNeuCodecDecoder",
    ),
}

__all__ = ["VIENEU_MODELS"]
