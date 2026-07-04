"""Unit tests for VieNeu-TTS-v2 prompt/processor logic (pure-Python, no vLLM).

These tests exercise the lightweight helpers in
``vllm_omni/model_executor/models/vieneu/processor.py`` and ``tokenizer.py``
that do NOT depend on the ``sea-g2p`` G2P package or on vLLM/torch:

  - ``_split_into_chunks``  -- sentence-boundary / minor-punctuation /
    whitespace / hard-cut fallback chunking.
  - ``build_prompt``        -- prompt layout (text region + speech region).
  - ``speech_ids_to_tokens`` / ``extract_speech_token_ids`` -- tokenizer
    helpers from ``tokenizer.py``.
  - ``load_presets``        -- ``voices.json`` parsing incl. ``meta`` license.

The modules under test import only stdlib + (``TYPE_CHECKING``-guarded)
``transformers`` typing, so they are imported directly -- no ``sys.modules``
stubbing is required. ``normalize_and_chunk`` and ``phonemize`` are
intentionally NOT tested here because they require ``sea-g2p``.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Hermetic import setup.
#
# ``vllm_omni/__init__.py`` imports ``.patch`` (needs ``aenum``) and several
# other submodules that pull in vLLM/torch. To import just the pure-Python
# ``vieneu.processor``/``vieneu.tokenizer`` modules without triggering that
# heavy package init, we register stub *package* modules for the
# ``vllm_omni`` parent chain in ``sys.modules`` with real ``__path__`` values.
# Python then resolves relative imports (``from .tokenizer import ...``)
# against the real on-disk files without ever executing the real
# ``__init__.py`` files. The stubs are inert (no attributes), which is fine
# because the modules under test only use intra-package relative imports.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VLLM_OMNI = _REPO_ROOT / "vllm_omni"


def _ensure_stub_pkg(dotted: str, path: Path) -> None:
    """Register ``dotted`` as a stub package in sys.modules with ``__path__``."""
    if dotted in sys.modules and getattr(sys.modules[dotted], "__path__", None):
        return
    mod = types.ModuleType(dotted)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[dotted] = mod


_ensure_stub_pkg("vllm_omni", _VLLM_OMNI)
_ensure_stub_pkg("vllm_omni.model_executor", _VLLM_OMNI / "model_executor")
_ensure_stub_pkg("vllm_omni.model_executor.models", _VLLM_OMNI / "model_executor" / "models")
_ensure_stub_pkg("vllm_omni.model_executor.models.vieneu", _VLLM_OMNI / "model_executor" / "models" / "vieneu")

# Now the real module files import cleanly via their stubbed parent packages.
from vllm_omni.model_executor.models.vieneu.processor import (  # noqa: E402
    ReferenceVoice,
    _split_into_chunks,
    build_prompt,
    load_presets,
)
from vllm_omni.model_executor.models.vieneu.tokenizer import (  # noqa: E402
    extract_speech_token_ids,
    speech_ids_to_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    def test_short_text_returns_single_chunk(self):
        text = "Hello world."
        chunks = _split_into_chunks(text, max_chars=256)
        assert chunks == ["Hello world."]

    def test_cuts_at_sentence_boundary_within_window(self):
        # Two sentences; the first ends inside the max_chars window.
        text = "This is sentence one. This is sentence two and it is longer."
        chunks = _split_into_chunks(text, max_chars=30)
        # The first '.' sits at index 20 (< 30) so the cut lands after it.
        assert chunks[0] == "This is sentence one."
        assert len(chunks) >= 2
        # No chunk should exceed max_chars (sentence-boundary cut keeps it in-window;
        # the trailing sentence is shorter than max_chars).
        for c in chunks:
            assert len(c) <= 30

    def test_falls_back_to_minor_punctuation(self):
        # No sentence-end char inside the window, but a comma exists.
        text = "alpha beta gamma delta epsilon zeta eta theta, then more text follows here"
        chunks = _split_into_chunks(text, max_chars=40)
        # The first chunk should end right after the comma (cut == comma_index+1).
        assert "," in chunks[0]
        assert chunks[0].endswith(",")

    def test_falls_back_to_whitespace(self):
        # No sentence-end, no minor punctuation, but a space exists in window.
        text = "aaaaaaaaaa bbbbbbbbbbb cccccccccccc dddddddddddd eeeeeeeeeeee"
        chunks = _split_into_chunks(text, max_chars=20)
        assert len(chunks) >= 2
        # Each chunk must be <= max_chars; whitespace-cut never exceeds the window.
        for c in chunks:
            assert len(c) <= 20

    def test_hard_cut_when_no_break_in_window(self):
        # A single long run with no punctuation and no spaces at all.
        text = "x" * 100
        chunks = _split_into_chunks(text, max_chars=10)
        assert chunks == ["x" * 10] * 10


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_layout_and_chunk_metadata(self):
        ref = ReferenceVoice(ref_codes=[0, 1, 2], ref_text="hi")
        prompt = build_prompt(
            input_text_phonemes="X",
            reference=ref,
            ref_text_phonemes="H",
            chunk_index=2,
            chunk_count=5,
        )

        # Starts with the text-prompt-start control token.
        assert prompt.text.startswith("<|TEXT_PROMPT_START|>")
        # Ends with the reference speech tokens (the speech region has no
        # trailing end token -- generation continues until SPEECH_GENERATION_END).
        assert prompt.text.endswith("<|speech_0|><|speech_1|><|speech_2|>")
        # Phoneme text sits in the middle, between the ref transcript and the
        # text-prompt-end marker.
        assert "H X" in prompt.text
        assert "<|TEXT_PROMPT_END|>" in prompt.text
        assert "<|SPEECH_GENERATION_START|>" in prompt.text
        # The speech region follows the text region.
        assert prompt.text.index("<|TEXT_PROMPT_END|>") < prompt.text.index("<|SPEECH_GENERATION_START|>")
        # Chunk metadata is echoed back.
        assert prompt.chunk_index == 2
        assert prompt.chunk_count == 5


# ---------------------------------------------------------------------------
# tokenizer helpers
# ---------------------------------------------------------------------------


class TestTokenizerHelpers:
    def test_speech_ids_to_tokens(self):
        assert speech_ids_to_tokens([0, 1, 65535]) == "<|speech_0|><|speech_1|><|speech_65535|>"

    def test_speech_ids_to_tokens_empty(self):
        assert speech_ids_to_tokens([]) == ""

    def test_extract_speech_token_ids_filters_control_tokens(self):
        text = "<|TEXT_PROMPT_START|><|speech_10|><|speech_20|><|SPEECH_GENERATION_END|><|speech_30|>"
        ids = extract_speech_token_ids(text)
        # Only the <|speech_N|> tokens contribute; control tokens are ignored.
        assert ids == [10, 20, 30]

    def test_extract_speech_token_ids_none_when_no_speech_tokens(self):
        assert extract_speech_token_ids("<|TEXT_PROMPT_START|>plain text<|SPEECH_GENERATION_END|>") == []


# ---------------------------------------------------------------------------
# load_presets
# ---------------------------------------------------------------------------


class TestLoadPresets:
    def test_loads_voices_and_ignores_meta(self, tmp_path: Path):
        voices = {
            "meta": {"license": "CC BY-NC 4.0", "author": "VieNeu"},
            "Binh": {"codes": [0, 1, 2, 3], "text": "xin chao"},
            "Tuyen": {"codes": [4, 5, 6], "text": "tam biet"},
        }
        path = tmp_path / "voices.json"
        path.write_text(json.dumps(voices), encoding="utf-8")

        presets = load_presets(path)

        # Both real voice entries are returned with correct fields.
        assert set(presets.keys()) == {"Binh", "Tuyen"}
        assert presets["Binh"].ref_codes == [0, 1, 2, 3]
        assert presets["Binh"].ref_text == "xin chao"
        assert presets["Binh"].name == "Binh"
        # The license is propagated from meta.
        assert presets["Binh"].license_note == "CC BY-NC 4.0"
        assert presets["Tuyen"].ref_codes == [4, 5, 6]
        assert presets["Tuyen"].ref_text == "tam biet"
        # The meta key is NOT treated as a voice.
        assert "meta" not in presets
