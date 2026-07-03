"""VieNeu-TTS-v2 codec stage: NeuCodec decode (TASK 10).

Consumes frame-aligned NeuCodec speech-token ids from ``input_ids`` and
decodes them to a waveform via NeuCodec's ``decode_code()``. Structurally
mirrors ``FishSpeechDACDecoder``/``Qwen3TTSCode2Wav`` (same duck-typed flags,
same ``_split_request_ids`` per-request batching pattern -- see
docs/Architecture.md Part A.6/A.9), but simpler: NeuCodec is a **single
FSQ codebook** (docs/Architecture.md Part B.1), so there is no
codebook-major reshape like fish_speech's DAC decoder needs for its 10
codebooks -- codes are just ``[num_frames]`` per request.

NeuCodec itself is not bundled in the VieNeu-TTS-v2 checkpoint; it's a
separate HF repo/package (``neuphonic/neucodec`` or the distilled
``neuphonic/distill-neucodec``) loaded independently, matching how the
reference SDK does it (docs/Architecture.md Part B.1/B.4).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .config import VieNeuCodecConfig

logger = init_logger(__name__)

# Module-level singleton codec for encode_ref_audio (lazy-loaded). Kept
# separate from VieNeuCodecDecoder._codec (the decode-side instance) so the
# serving path can encode reference audio without instantiating the heavier
# decoder module, and so repeated encode calls don't reload the codec.
_encode_codec: nn.Module | None = None
_encode_codec_repo_id: str | None = None


def encode_ref_audio(
    wav_samples: list[float] | "torch.Tensor",
    sr: int,
    *,
    codec_repo_id: str = "neuphonic/neucodec",
    target_sr: int = 16000,
    device: str = "cpu",
) -> list[int]:
    """Encode reference audio to flat NeuCodec FSQ code list (voice-cloning codes).

    Loads NeuCodec the same way ``VieNeuCodecDecoder._ensure_codec_loaded`` does
    (``from neucodec import NeuCodec`` -> ``NeuCodec.from_pretrained(repo_id)``),
    then calls the codec's ``encode_code`` method, which accepts a float tensor
    of shape ``[B, 1, T_16]`` (mono audio at 16 kHz) and returns a long tensor
    of shape ``[B, 1, num_frames]`` (50 Hz frame rate, single FSQ codebook with
    65536 codes). Verified against the neuphonic/neucodec README and the
    reference SDK's ``vieneu/base.py`` ``encode_reference`` call site
    (``self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)``).

    Args:
        wav_samples: 1D float waveform (list or 1D tensor) at ``sr`` Hz.
        sr: Sample rate of ``wav_samples``. Resampled to ``target_sr`` (16 kHz)
            if different, matching the NeuCodec input requirement.
        codec_repo_id: HF repo id for NeuCodec weights.
        target_sr: NeuCodec input sample rate (16 kHz).
        device: Device to run encode on ("cpu" keeps the serving path off the
            GPU; encode is cheap relative to decode).

    Returns:
        Flat Python list of int codes in ``[0, 65535]``, length ``num_frames``.
    """
    global _encode_codec, _encode_codec_repo_id

    try:
        from neucodec import NeuCodec
    except ImportError as e:
        raise ImportError(
            "VieNeu-TTS voice cloning requires the `neucodec` package to encode "
            "reference audio. Install it with `pip install neucodec`."
        ) from e

    if _encode_codec is None or _encode_codec_repo_id != codec_repo_id:
        codec = NeuCodec.from_pretrained(codec_repo_id)
        codec.eval()
        codec.to(device=device)
        _encode_codec = codec
        _encode_codec_repo_id = codec_repo_id
        logger.info("Loaded NeuCodec encoder %s on %s for ref_audio", codec_repo_id, device)

    wav = torch.as_tensor(wav_samples, dtype=torch.float32)
    if wav.ndim > 1:
        wav = wav.reshape(-1)
    wav = wav.squeeze()

    if sr != target_sr:
        import torchaudio

        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

    wav = wav.to(device=device)
    # NeuCodec.encode_code expects [B, 1, T_16] mono float tensor.
    wav_1_1_t = wav.unsqueeze(0).unsqueeze(0)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        codes = _encode_codec.encode_code(wav_1_1_t)
    codes = codes.reshape(-1).to(dtype=torch.long).cpu()
    return [int(c) for c in codes.tolist()]


class VieNeuCodecDecoder(nn.Module):
    """Stage-1 NeuCodec decoder for VieNeu-TTS-v2 (GenerationModelRunner).

    Consumes flat NeuCodec speech-token ids from ``input_ids`` (already
    offset back into raw codec-code space by the stage input processor --
    see ``processor.py``/tokenizer id ranges in ``config.py``) and decodes
    waveform via NeuCodec's decoder.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.codec_config: VieNeuCodecConfig = getattr(
            vllm_config.model_config.hf_config, "codec_config", None
        ) or VieNeuCodecConfig()

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._codec: nn.Module | None = None
        self._output_sample_rate: int = self.codec_config.sample_rate
        self._num_codebooks: int = self.codec_config.num_codebooks
        self._logged_codec_stats = False

    def _ensure_codec_loaded(self) -> None:
        """Lazily load NeuCodec from its own HF repo (not this checkpoint)."""
        if self._codec is not None:
            return
        try:
            from neucodec import NeuCodec
        except ImportError as e:
            raise ImportError(
                "VieNeu-TTS-v2 codec decode requires the `neucodec` package. "
                "Install it with `pip install neucodec`."
            ) from e

        device = self.vllm_config.device_config.device
        codec = NeuCodec.from_pretrained(self.codec_config.codec_repo_id)
        codec.eval()
        codec.to(device=device)
        self._codec = codec
        logger.info(
            "Loaded NeuCodec decoder %s on %s (sample_rate=%d)",
            self.codec_config.codec_repo_id,
            device,
            self._output_sample_rate,
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings -- codes are decoded directly
        # in forward(), not routed through an embedding table.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The decode-side NeuCodec weights live in the external
        # ``neuphonic/neucodec`` repo and are lazy-loaded by _ensure_codec_loaded().
        # There are intentionally no weights to load from the VieNeu-TTS-v2
        # checkpoint for this vLLM generation-stage wrapper.
        return set()

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments.

        Same pattern as ``FishSpeechDACDecoder._split_request_ids`` /
        ``Qwen3TTSCode2Wav._split_request_ids`` (docs/Architecture.md
        Part A.9) -- kept identical rather than sharing a base class,
        since none exists in vllm-omni for codec stages.
        """
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode NeuCodec codes into an audio waveform.

        input_ids layout per request: flat codes ``[num_frames]`` (single
        codebook -- no codebook-major interleave, unlike fish_speech's 10
        DAC codebooks or qwen3_tts's residual-VQ layers).
        """
        self._ensure_codec_loaded()
        assert self._codec is not None

        sr_tensor = torch.tensor(self._output_sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        num_req = len(request_ids_list)

        left_context_size = [0] * num_req
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                if "left_context_size" in info:
                    left_context_size[i] = info["left_context_size"]

        valid_codes: list[torch.Tensor] = []
        valid_indices: list[int] = []
        parsed_ctx_frames = [0] * num_req
        parsed_total_frames = [0] * num_req

        for i, req_ids in enumerate(request_ids_list):
            n = req_ids.numel()
            if n < 1:
                continue
            parsed_ctx_frames[i] = left_context_size[i]
            parsed_total_frames[i] = n
            valid_codes.append(req_ids)
            valid_indices.append(i)

        if not valid_codes:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty] * num_req, "sr": [sr_tensor] * num_req},
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes[0]
                logger.info(
                    "NeuCodec decoder: frames=%d uniq=%d range=[%d,%d] batch=%d",
                    c.numel(),
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    len(valid_codes),
                )
            except Exception:
                pass

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            codes = valid_codes[j]
            ctx_frames = parsed_ctx_frames[idx]
            total_frames = parsed_total_frames[idx]

            # NeuCodec.decode_code expects [1, 1, num_frames] integer codes.
            codes_1_1_f = codes.reshape(1, 1, -1)
            with torch.amp.autocast("cuda", enabled=False):
                wav = self._codec.decode_code(codes_1_1_f)
            wav = wav.reshape(-1)

            if ctx_frames > 0 and total_frames > 0:
                # Trim left-context frames proportionally, same as fish_speech's
                # DAC decoder (docs/Architecture.md Part A.6) -- avoids
                # re-emitting audio from the overlap window on chunked/streaming
                # decode (see stages.py's async_chunk wiring, TASK 12).
                samples_per_frame = wav.numel() / max(total_frames, 1)
                trim_samples = int(round(ctx_frames * samples_per_frame))
                wav = wav[trim_samples:]

            audios[idx] = wav.to(dtype=torch.float32, device="cpu")

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audios, "sr": srs},
        )
