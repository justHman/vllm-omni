"""VieNeu-TTS-v2 talker (stage-0 generation) model (TASK 9).

Per docs/Architecture.md Part C, VieNeu's NeuCodec is a **single FSQ
codebook** -- there is no residual quantizer layer to predict, so unlike
fish_speech's Slow/Fast AR split or qwen3_tts's talker + nested code
predictor, VieNeu's talker needs no ``talker_mtp`` hook. It is a thin
wrapper around vLLM's stock ``Qwen3Model`` (reused directly, not
reimplemented -- see docs/Architecture.md Part A.9 "Reusable abstractions"),
with exactly two VieNeu-specific additions:

  1. EOS override to ``<|SPEECH_GENERATION_END|>`` (id 381 by default)
     instead of the tokenizer's generic ``<|endoftext|>`` (id 375) --
     wired via ``stop_token_ids`` in the stage YAML's
     ``default_sampling_params`` (see ``pipeline.yaml``), not in this class.
  2. ``compute_logits()`` masking to the valid speech-token id range plus
     the stop id, matching the fish_speech/qwen3_tts pattern of
     restricting the sampled vocabulary to codec-valid ids
     (docs/Architecture.md Part A.5).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .config import VieNeuTalkerConfig

logger = init_logger(__name__)


class VieNeuTalkerForConditionalGeneration(nn.Module):
    """vLLM-AR talker for VieNeu-TTS-v2: single-codebook speech-token decode.

    Registered under a distinct architecture name (see ``registry.py``) so
    the stage pipeline can select it via ``hf_overrides`` without touching
    the upstream checkpoint's own ``config.json`` (docs/Architecture.md
    Part C, point 4).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: VieNeuTalkerConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Constant logit mask: only the NeuCodec speech-token range plus the
        # stop token are valid outputs for this stage (docs/Architecture.md
        # Part A.5 -- same masking approach as fish_speech/qwen3_tts).
        vocab = int(self.config.vocab_size)
        allowed_mask = torch.zeros((vocab,), dtype=torch.bool)
        lo = max(0, self.config.speech_token_id_start)
        hi = min(vocab, self.config.speech_token_id_end)
        if hi > lo:
            allowed_mask[lo:hi] = True
        eos_id = self.config.speech_generation_end_id
        if 0 <= eos_id < vocab:
            allowed_mask[eos_id] = True
        self.register_buffer("_speech_allowed_mask", allowed_mask, persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        masked = logits.masked_fill(~self._speech_allowed_mask, float("-inf"))

        # [DEBUG-VIENEU] Per-step probe of stop-token 381 logits vs the top-k=50
        # cutoff. HF generate stops at ~252 tokens but vLLM v0.22.0 runs to
        # max_tokens without ever emitting 381. This log reveals whether 381
        # ever enters the top-50 sampleable set, or is always crowded out by
        # the 65536 speech tokens. Gated to the first 20 steps + every 50th
        # after, to bound log volume. Runs in the WORKER subprocess.
        if not getattr(self, "_dbg_step", 0):
            self._dbg_step = 0
        step = self._dbg_step
        self._dbg_step += 1
        if step < 20 or step % 50 == 0:
            try:
                last = masked[-1]  # [vocab]
                stop_logit = float(last[self.config.speech_generation_end_id].item())
                # top-50 cutoff among allowed (non -inf) tokens
                allowed_vals = last[self._speech_allowed_mask]
                if allowed_vals.numel() > 0:
                    k = min(50, allowed_vals.numel())
                    topk_vals, _ = allowed_vals.topk(k)
                    cutoff = float(topk_vals[-1].item())
                    rank_of_stop = int((allowed_vals > stop_logit).sum().item())
                    logger.warning(
                        "[DEBUG-VIENEU] step=%d stop381_logit=%.4f top50_cutoff=%.4f "
                        "stop_rank_in_allowed=%d (0=top) stop_in_top50=%s",
                        step, stop_logit, cutoff, rank_of_stop,
                        bool(stop_logit >= cutoff),
                    )
            except Exception:
                pass

        return masked

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Wrap raw hidden states into the shared ``OmniOutput`` contract.

        Unlike fish_speech/qwen3_tts, VieNeu's talker has no per-step
        multimodal side-channel. The generated token ids themselves are the
        NeuCodec speech-token sequence, extracted downstream by the stage
        input processor directly from ``output.token_ids``.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        logger.info("Loaded %d weights for VieNeuTalkerForConditionalGeneration", len(loaded))
        return loaded
