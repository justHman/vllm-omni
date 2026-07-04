"""Stage input processor for VieNeu-TTS: talker -> NeuCodec decoder."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]) -> list[Any]:
    """Resolve the upstream stage's finished engine outputs for this stage."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")
    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")
    return stage.engine_outputs


# Checkpoint tokenizer ids (docs/Architecture.md Part B.3).
_SPEECH_TOKEN_ID_START = 382
_SPEECH_TOKEN_ID_END = 65918  # exclusive
_SPEECH_GENERATION_END_ID = 381


def _to_codec_code_ids(token_ids: list[int]) -> list[int]:
    """Convert generated speech-token ids to raw NeuCodec code ids.

    VieNeu generates vocabulary token ids in the range
    ``[speech_token_id_start, speech_token_id_end)``. NeuCodec itself expects
    raw code indices ``[0, 65535]``. The stage-1 codec therefore receives the
    speech-token ids offset back into raw codec space.
    """
    codes: list[int] = []
    for token_id in token_ids:
        if token_id == _SPEECH_GENERATION_END_ID:
            break
        if _SPEECH_TOKEN_ID_START <= token_id < _SPEECH_TOKEN_ID_END:
            codes.append(token_id - _SPEECH_TOKEN_ID_START)
    return codes


def talker2codec(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async processor: wait for talker finish, then decode all codes."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    codec_inputs: list[OmniTokensPrompt] = []

    for talker_output in talker_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        codec_codes = _to_codec_code_ids(list(output.token_ids))
        codec_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )

    return codec_inputs


def talker2codec_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async processor: stream generated speech-token ids to the codec stage.

    Unlike qwen3_tts/fish_speech there is no multimodal side-channel carrying
    frame tensors. VieNeu's generated token ids themselves are the codec stream,
    so we slice request.output_token_ids / request.all_token_ids directly,
    convert only the newly arrived speech-token ids into raw NeuCodec codes,
    then emit overlapped frame windows for stage-1 decode.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    generated_token_ids = list(getattr(request, "output_token_ids", []) or [])
    generated_codec_codes = _to_codec_code_ids(generated_token_ids)

    cached_generated_len = int(transfer_manager.request_payload.get(request_id, 0) or 0)
    current_generated_len = len(generated_codec_codes)
    new_frame_count = max(0, current_generated_len - cached_generated_len)
    transfer_manager.request_payload[request_id] = current_generated_len

    if new_frame_count > 0:
        new_codes = generated_codec_codes[-new_frame_count:]
        transfer_manager.code_prompt_token_ids[request_id].extend(new_codes)

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    if length <= 0:
        if finished:
            return {
                "code_predictor_codes": [],
                "finished": True,
            }
        return None

    if chunk_size <= 0 or left_context_size_config < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}"
        )

    chunk_length = length % chunk_size
    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size
    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, int(end_index - context_length))
    window_codes = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    return {
        "code_predictor_codes": list(window_codes),
        "left_context_size": left_context_size,
        "finished": finished,
    }
