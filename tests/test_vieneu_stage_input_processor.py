"""Unit tests for the VieNeu talker->codec stage input processor.

Tests the pure-Python logic in
``vllm_omni/model_executor/stage_input_processors/vieneu.py``:

  - ``_to_codec_code_ids``      -- speech-token-id -> raw codec code mapping,
                                    with stop-id truncation.
  - ``talker2codec_async_chunk`` -- streaming chunk emission, using minimal
    fakes for the transfer_manager / request / connector.

HERMETIC IMPORT NOTE
--------------------
The module under test imports two names that pull in the full vLLM graph:

    from vllm.logger import init_logger
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

Neither is used by ``_to_codec_code_ids`` or ``talker2codec_async_chunk``
(``_validate_stage_inputs`` is only used by the non-async ``talker2codec``).
To keep this test hermetic (no vLLM / aenum / torch / neucodec), we inject
stub modules into ``sys.modules`` *before* importing the target module.

Only stdlib + pytest are required to run this file.
"""

from __future__ import annotations

import logging
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy imports BEFORE importing the module under test.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VLLM_OMNI = _REPO_ROOT / "vllm_omni"


def _ensure_stub_pkg(dotted: str, path: Path) -> None:
    """Register ``dotted`` as a stub package in sys.modules with ``__path__``.

    Using a real ``__path__`` lets Python resolve the target module's relative
    imports against on-disk files while skipping the heavy real ``__init__.py``
    of ``vllm_omni`` (which needs aenum + vLLM + torch).
    """
    if dotted in sys.modules and getattr(sys.modules[dotted], "__path__", None):
        return
    mod = types.ModuleType(dotted)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[dotted] = mod


# ``vllm`` is not installed in the test environment; provide a minimal stub
# whose ``logger.init_logger`` returns a plain stdlib logger.
if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    vllm_logger_stub = types.ModuleType("vllm.logger")

    def _init_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

    vllm_logger_stub.init_logger = _init_logger  # type: ignore[attr-defined]
    vllm_stub.logger = vllm_logger_stub  # type: ignore[attr-defined]
    sys.modules["vllm"] = vllm_stub
    sys.modules["vllm.logger"] = vllm_logger_stub

# Register the vllm_omni parent chain as inert stub packages with real paths so
# the target module (and its relative imports) load without triggering the real
# vllm_omni/__init__.py (which imports aenum/vLLM).
_ensure_stub_pkg("vllm_omni", _VLLM_OMNI)
_ensure_stub_pkg("vllm_omni.model_executor", _VLLM_OMNI / "model_executor")
_ensure_stub_pkg("vllm_omni.model_executor.stage_input_processors", _VLLM_OMNI / "model_executor" / "stage_input_processors")

# ``vllm_omni.model_executor.stage_input_processors.qwen3_omni`` is imported at
# the top of the module under test but is NOT used by the functions under test.
# Provide a no-op ``_validate_stage_inputs`` so the import succeeds.
qwen3_omni_path = "vllm_omni.model_executor.stage_input_processors.qwen3_omni"
if qwen3_omni_path not in sys.modules:
    qwen3_omni_stub = types.ModuleType(qwen3_omni_path)

    def _validate_stage_inputs(stage_list, engine_input_source):  # noqa: D401
        """No-op stand-in for the real validator (unused by these tests)."""
        return stage_list

    qwen3_omni_stub._validate_stage_inputs = _validate_stage_inputs  # type: ignore[attr-defined]
    sys.modules[qwen3_omni_path] = qwen3_omni_stub

# Now import the module under test -- its top-level imports resolve to the stubs.
from vllm_omni.model_executor.stage_input_processors.vieneu import (  # noqa: E402
    _to_codec_code_ids,
    talker2codec_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# _to_codec_code_ids
# ---------------------------------------------------------------------------


class TestToCodecCodeIds:
    def test_maps_in_range_ids_with_offset(self):
        # 382 -> 0, 383 -> 1, 65917 -> 65535 (65917 < 65918, in range).
        out = _to_codec_code_ids([382, 383, 65917])
        assert out == [0, 1, 65535]

    def test_stop_id_truncates_rest(self):
        # 381 is the SPEECH_GENERATION_END id; everything after it is dropped.
        out = _to_codec_code_ids([382, 383, 381, 400, 500])
        assert out == [0, 1]

    def test_out_of_range_ids_are_skipped(self):
        # Below start, at end-bound (65918 is exclusive -> out of range), and far above.
        out = _to_codec_code_ids([0, 381 - 1, 382, 65918, 99999])
        # Only 382 maps to 0; 381-1=380 is not the stop id and not in range -> skipped.
        assert out == [0]

    def test_empty_input(self):
        assert _to_codec_code_ids([]) == []

    def test_stop_id_alone(self):
        assert _to_codec_code_ids([381]) == []


# ---------------------------------------------------------------------------
# talker2codec_async_chunk -- fakes + scenarios
# ---------------------------------------------------------------------------


class FakeConnector:
    """Minimal connector carrying an ``extra`` config dict."""

    def __init__(self, *, codec_chunk_frames: int = 25, codec_left_context_frames: int = 25):
        self.config = {
            "extra": {
                "codec_chunk_frames": codec_chunk_frames,
                "codec_left_context_frames": codec_left_context_frames,
            }
        }


class FakeTM:
    """Minimal transfer_manager used by talker2codec_async_chunk."""

    def __init__(self, connector: FakeConnector | None = None):
        self.request_payload: dict[str, int] = {}
        self.code_prompt_token_ids: dict[str, list[int]] = defaultdict(list)
        self.connector = connector or FakeConnector()


class FakeRequest:
    """Minimal request object exposing the attributes the processor reads."""

    def __init__(self, req_id: str, token_ids: list[int], finished: bool = False):
        self.external_req_id = req_id
        self.output_token_ids = list(token_ids)
        self._finished = finished

    def is_finished(self) -> bool:
        return self._finished


# Helper: speech-token ids that map 1:1 to raw codec codes [0..N) via _to_codec_code_ids.
def _speech_ids(n: int) -> list[int]:
    """Return n speech-token ids [382, 382+n) which map to codec codes [0, n)."""
    return [382 + i for i in range(n)]


class TestTalker2codecAsyncChunk:
    def test_first_call_below_chunk_size_waits(self):
        tm = FakeTM()
        req = FakeRequest("r1", token_ids=_speech_ids(10), finished=False)

        # 10 new codes < chunk_size(25) and not finished -> should return None.
        result = talker2codec_async_chunk(tm, pooling_output=None, request=req, is_finished=False)
        assert result is None

    def test_emits_chunk_when_full(self):
        tm = FakeTM()
        # 25 speech-token ids -> exactly one full chunk (length % chunk_size == 0).
        req = FakeRequest("r2", token_ids=_speech_ids(25), finished=False)

        result = talker2codec_async_chunk(tm, pooling_output=None, request=req, is_finished=False)
        assert result is not None
        assert isinstance(result["code_predictor_codes"], list)
        # Codes are in raw codec space (offset applied).
        assert all(isinstance(c, int) for c in result["code_predictor_codes"])
        # 25 codes in the chunk + up to 25 left-context frames preceding -> <= 50.
        assert len(result["code_predictor_codes"]) <= 50
        assert result["finished"] is False
        assert isinstance(result["left_context_size"], int)

    def test_finished_with_leftover_emits_final_chunk(self):
        tm = FakeTM()
        # 10 codes (< chunk_size) but finished -> final partial chunk emitted.
        req = FakeRequest("r3", token_ids=_speech_ids(10), finished=True)

        result = talker2codec_async_chunk(tm, pooling_output=None, request=req, is_finished=False)
        assert result is not None
        assert result["finished"] is True
        assert result["code_predictor_codes"]  # non-empty leftover
        assert all(isinstance(c, int) for c in result["code_predictor_codes"])

    def test_finished_with_no_codes_returns_empty(self):
        tm = FakeTM()
        # No generated speech-token ids at all and finished.
        req = FakeRequest("r4", token_ids=[], finished=True)

        result = talker2codec_async_chunk(tm, pooling_output=None, request=req, is_finished=False)
        assert result == {"code_predictor_codes": [], "finished": True}

    def test_uses_is_finished_flag_when_request_not_finished(self):
        """The explicit ``is_finished`` arg overrides request.is_finished()=False."""
        tm = FakeTM()
        req = FakeRequest("r5", token_ids=_speech_ids(10), finished=False)

        # is_finished=True passed explicitly even though request says False.
        result = talker2codec_async_chunk(tm, pooling_output=None, request=req, is_finished=True)
        assert result is not None
        assert result["finished"] is True
