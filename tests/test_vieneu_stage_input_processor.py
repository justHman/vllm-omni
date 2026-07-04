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

# ``vllm_omni.data_entry_keys`` is imported by the module under test for the
# OmniPayloadStruct / CodesStruct / MetaStruct payload schema. The real module
# needs msgspec + torch; stub it with tiny dataclass-like stand-ins so the test
# stays hermetic (no torch/msgspec required). The stub exposes the same attribute
# access shape (``.codes.audio``, ``.meta.left_context_size``) the processor uses.
if "vllm_omni.data_entry_keys" not in sys.modules:
    dek_stub = types.ModuleType("vllm_omni.data_entry_keys")

    class _Simple:
        """Tiny mutable attribute bag mirroring a msgspec.Struct field set."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class CodesStruct(_Simple):
        pass

    class MetaStruct(_Simple):
        pass

    class OmniPayloadStruct(_Simple):
        pass

    def to_struct(payload):  # noqa: D401
        return payload

    dek_stub.CodesStruct = CodesStruct  # type: ignore[attr-defined]
    dek_stub.MetaStruct = MetaStruct  # type: ignore[attr-defined]
    dek_stub.OmniPayloadStruct = OmniPayloadStruct  # type: ignore[attr-defined]
    dek_stub.to_struct = to_struct  # type: ignore[attr-defined]
    sys.modules["vllm_omni.data_entry_keys"] = dek_stub

# ``torch`` is imported by the module under test (for torch.tensor / torch.empty
# when building the codes.audio tensor). Provide a minimal stub that supports
# the two calls used: tensor(list, dtype=...) and empty(n, dtype=...).
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _DType:
        def __init__(self, name):
            self.name = name

    class _Tensor:
        def __init__(self, data, dtype=None):
            self.data = list(data) if hasattr(data, "__iter__") else data
            self.dtype = dtype

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

        def numel(self):
            return len(self.data)

    torch_stub.long = _DType("long")
    torch_stub.float32 = _DType("float32")

    def _tensor(data, dtype=None):
        return _Tensor(data, dtype=dtype)

    def _empty(n, dtype=None):
        return _Tensor([], dtype=dtype)

    torch_stub.tensor = _tensor  # type: ignore[attr-defined]
    torch_stub.empty = _empty  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub

# Now import the module under test -- its top-level imports resolve to the stubs.
from vllm_omni.model_executor.stage_input_processors.vieneu import (  # noqa: E402
    _to_codec_code_ids,
    talker2codec_async_chunk,
)
from vllm_omni.data_entry_keys import OmniPayloadStruct  # noqa: E402

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
        # v0.22.0 returns an OmniPayloadStruct (NOT a plain dict) — the adapter
        # reads result.codes.audio / result.meta.left_context_size as attributes.
        assert isinstance(result, OmniPayloadStruct)
        audio = result.codes.audio
        # Codes are in raw codec space (offset applied), carried as a tensor/list.
        codes = list(audio.data) if hasattr(audio, "data") else list(audio)
        assert all(isinstance(c, int) for c in codes)
        # 25 codes in the chunk + up to 25 left-context frames preceding -> <= 50.
        assert len(codes) <= 50
        # The adapter overwrites meta.finished after this returns, so the
        # processor intentionally does NOT set it; left_context_size is set.
        assert hasattr(result.meta, "left_context_size")
        assert isinstance(result.meta.left_context_size, int)

    def test_finished_with_leftover_emits_final_chunk(self):
        tm = FakeTM()
        # 10 codes (< chunk_size) but finished -> final partial chunk emitted.
        req = FakeRequest("r3", token_ids=_speech_ids(10), finished=True)

        result = talker2codec_async_chunk(tm, pooling_output=None, request=req, is_finished=False)
        assert result is not None
        assert isinstance(result, OmniPayloadStruct)
        audio = result.codes.audio
        codes = list(audio.data) if hasattr(audio, "data") else list(audio)
        assert codes  # non-empty leftover
        assert all(isinstance(c, int) for c in codes)

    def test_finished_with_no_codes_returns_empty_sentinel(self):
        tm = FakeTM()
        # No generated speech-token ids at all and finished -> finish-only
        # sentinel struct (empty codes, left_context_size=0). The adapter fills
        # meta.finished/is_segment_finished from is_finished afterwards.
        req = FakeRequest("r4", token_ids=[], finished=True)

        result = talker2codec_async_chunk(tm, pooling_output=None, request=req, is_finished=False)
        assert isinstance(result, OmniPayloadStruct)
        audio = result.codes.audio
        codes = list(audio.data) if hasattr(audio, "data") else list(audio)
        assert codes == []
        assert result.meta.left_context_size == 0

    def test_uses_is_finished_flag_when_request_not_finished(self):
        """The explicit ``is_finished`` arg overrides request.is_finished()=False.

        The processor itself does not write meta.finished (the adapter does that
        after the call), so this test only checks that a chunk is emitted rather
        than None — the finished flag is applied downstream by the adapter.
        """
        tm = FakeTM()
        req = FakeRequest("r5", token_ids=_speech_ids(10), finished=False)

        # is_finished=True passed explicitly even though request says False.
        result = talker2codec_async_chunk(tm, pooling_output=None, request=req, is_finished=True)
        assert result is not None
        assert isinstance(result, OmniPayloadStruct)
