"""Tokenizer helpers for VieNeu-TTS-v2.

Per the TASK 3 compatibility analysis (docs/Architecture.md Part C): the
checkpoint's tokenizer is a standard ``Qwen2Tokenizer`` (AutoTokenizer-
compatible) with an extended vocabulary of added tokens for speech/control
codes (Part B.3). **No custom tokenizer class is needed** -- a
``VieNeuTokenizer(PreTrainedTokenizer)`` subclass would just be
reimplementing what ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)``
already does correctly.

What genuinely needs VieNeu-specific handling is not tokenization itself but
(a) extracting the generated speech-token id sequence back out of raw
decoded/produced ids, and (b) resolving the handful of named control tokens
by id. Both live here as plain functions, used by ``processor.py`` (prompt
building) and ``codec.py``/the talker stage (speech-token extraction) --
this mirrors how fish_speech/qwen3_tts keep tokenizer *usage* inline in
their model/processor code rather than behind a custom tokenizer subclass.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Matches literal `<|speech_<id>|>` tokens in decoded text, per
# docs/Architecture.md Part B.3/B.7 (the reference SDK extracts speech
# tokens this way rather than relying on tokenizer-level special-token
# filtering, since only <|endoftext|> is marked "special": true).
_SPEECH_TOKEN_PATTERN = re.compile(r"<\|speech_(\d+)\|>")

# Control-token literal strings, resolved to ids via
# ``tokenizer.convert_tokens_to_ids`` at load time (see ``get_control_token_ids``).
CONTROL_TOKENS = (
    "<|TEXT_PROMPT_START|>",
    "<|TEXT_PROMPT_END|>",
    "<|SPEECH_GENERATION_START|>",
    "<|SPEECH_GENERATION_END|>",
)


def load_tokenizer(model_path: str, *, trust_remote_code: bool = True) -> "PreTrainedTokenizerBase":
    """Load the checkpoint's tokenizer via the standard AutoTokenizer path.

    Kept as a thin wrapper (rather than importing AutoTokenizer directly in
    every call site) so a future checkpoint variant that genuinely needs a
    custom tokenizer class only requires a change here.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def get_control_token_ids(tokenizer: "PreTrainedTokenizerBase") -> dict[str, int]:
    """Resolve VieNeu's control tokens to ids for the loaded tokenizer.

    Raises if a control token is missing, since a checkpoint without these
    tokens is not a VieNeu-TTS checkpoint (see the pipeline auto-detect
    heuristic in ``vllm_omni/config/stage_config.py::_looks_like_vieneu_tts``,
    which checks for exactly ``<|SPEECH_GENERATION_START|>``).
    """
    ids: dict[str, int] = {}
    for token in CONTROL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer is missing expected VieNeu control token: {token!r}")
        ids[token] = token_id
    return ids


def extract_speech_token_ids(text: str) -> list[int]:
    """Pull the integer NeuCodec speech-token id sequence out of decoded text.

    Non-speech control tokens (e.g. ``<|SPEECH_GENERATION_END|>``) simply
    don't match this pattern and are implicitly discarded, matching the
    reference SDK's ``extract_speech_ids`` behavior
    (docs/Architecture.md Part B.7, step 8).
    """
    return [int(m) for m in _SPEECH_TOKEN_PATTERN.findall(text)]


def speech_ids_to_tokens(speech_ids: list[int]) -> str:
    """Render a sequence of NeuCodec code indices as literal `<|speech_i|>` text.

    Used when building a voice-cloning prompt from reference-audio codes
    (docs/Architecture.md Part B.6): the reference codes are placed as
    literal vocabulary tokens immediately after
    ``<|SPEECH_GENERATION_START|>``.
    """
    return "".join(f"<|speech_{i}|>" for i in speech_ids)
