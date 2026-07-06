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

Streaming boundary splice: OVERLAP-ADD CROSSFADE.
NeuCodec is a CAUSAL conv decoder, so the suffix of a ``[ctx + new]``
decode does NOT reproduce the standalone decode of ``new`` (the conv's
receptive field + causal padding make the suffix position-dependent).
The previous "trim the first ctx_frames" approach (commit 73700459)
therefore produced "bụm" clicks at each chunk boundary: the trimmed
boundary did not match the previous chunk's tail.

The fix here is overlap-add crossfade: decode each chunk WITH left-context
(so its head has no edge effect), then crossfade the re-decoded overlap
with the previous chunk's buffered tail IN THE SAMPLE DOMAIN, instead of
trimming it. ``codec_chunk_frames = C`` and ``codec_left_context_frames =
O`` (the overlap) are configured in pipeline.yaml / stage_configs/vieneu.yaml.
O must be >= the NeuCodec conv receptive field for a seamless splice
(default O=25 == 0.5s @ 50Hz frame rate, matching fish_speech).

The crossfade state (``prev_tail`` samples per request_id) lives on the
decoder module itself (``self._prev_tail``). The codec stage runs in a
single mp-worker subprocess and processes one batch per step, so per-req
state keyed by ``req_id`` (read from ``runtime_additional_information[i]
["req_id"]``) is safe without locking. State is cleared lazily when a
``ctx_frames == 0`` chunk is seen for a request (one-shot / new request).
"""

from __future__ import annotations

import math
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


def _crossfade_window(n: int) -> torch.Tensor:
    """Hann-shaped crossfade window of length ``n`` (linear blend, smooth ends).

    ``fade_in  = sin^2(pi * t)``  (0 -> 1), with ``t = (1..n) / (2n)``
    ``fade_out = 1 - fade_in`` = ``cos^2(pi * t)``

    Two properties matter for the splice:
      1. ``fade_in + fade_out == 1`` everywhere -- so when the two overlap
         signals are identical (O >= receptive field), the blend is the
         IDENTITY (``tail*fade_out + head*fade_in = head``), giving a
         sample-exact seamless splice.
      2. The derivative of sin^2 is zero at both endpoints (Hann shape), so
         there is no slope discontinuity at the boundaries of the overlap
         region -- this is what makes Hann audibly cleaner than a raw linear
         ramp (which has a slope jump at both ends).

    The trade-off vs. an equal-power (sin/cos) window is a ~3dB level dip in
    the middle of the overlap (power = sin^4 + cos^4 dips to 0.5), which is
    acceptable for a short 0.5s splice and is the standard behavior of a
    Hann-shaped linear crossfade. An equal-power window would keep constant
    power but would NOT be sample-exact for identical overlaps
    (``sin + cos = sqrt(2)`` at the midpoint), so we pick blend-linearity
    over equal-power to guarantee seamless splices when O >= receptive field.
    When the two overlaps differ slightly (O < RF), the splice is a smooth
    blend instead of a click.
    """
    if n <= 0:
        return torch.empty(0, dtype=torch.float32)
    t = torch.arange(1, n + 1, dtype=torch.float32) / (2 * n)
    fade_in = torch.sin(math.pi * t) ** 2
    return fade_in

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
        # Streaming chunk config from the connector (codec_chunk_frames=C,
        # codec_left_context_frames=O). Needed so chunk 0 (ctx_frames==0) knows
        # O to seed the prev_tail buffer for chunk 1's crossfade. Same read
        # pattern as glm_tts_dit_wrapper._connector_chunk_config.
        cc = getattr(vllm_config.model_config, "stage_connector_config", None)
        extra = (cc or {}).get("extra") if isinstance(cc, dict) else getattr(cc, "extra", None)
        if not isinstance(extra, dict):
            extra = {}
        cf = extra.get("codec_chunk_frames", 25)
        self._ola_chunk_frames = int(cf[0]) if isinstance(cf, list) else int(cf)
        self._ola_overlap_frames = int(extra.get("codec_left_context_frames", 25))
        # Streaming overlap-add crossfade state. Maps req_id -> (tail_samples,
        # tail_overlap_samples). ``tail_samples`` is the last O frames worth of
        # decoded samples from the previous chunk (the overlap that will be
        # re-decoded with this chunk's left context). ``tail_overlap_samples``
        # is the sample-count of that overlap (= O * samples_per_frame for the
        # PREVIOUS chunk, kept so we can size fade_out correctly even if the
        # codec's samples-per-frame varies slightly per decode).
        self._prev_tail: dict[str, tuple[torch.Tensor, int]] = {}
        # Bounded LRU-ish cleanup: never grow this dict beyond this many live
        # requests. The codec stage is single-process and batch sizes are small
        # (max_num_seqs=4 by default), so 64 slots is plenty of headroom; the
        # cap just guards against a pathological req_id churn leak.
        self._prev_tail_max = 64

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
        req_ids_per_idx: list[str | None] = [None] * num_req
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                # The connector / scheduling-coordinator delivers left_context_size
                # nested under "meta" (see omni_scheduling_coordinator.update_request_metadata:
                # runtime_seed = {"meta": {"left_context_size": ...}} and fish_speech's
                # DAC decoder reads the same path). Reading the flat top-level key was a
                # bug: it was always absent, so ctx_frames stayed 0 and the trim path
                # never ran -- with codec_left_context_frames>0 each chunk then emitted
                # its full ctx+chunk window, doubling the audio (the echo observed in
                # commit f2cd6d8a). Fixing this unlocks streaming overlap-trim decode.
                meta = info.get("meta", {}) if isinstance(info, dict) else {}
                if "left_context_size" in meta:
                    left_context_size[i] = int(meta["left_context_size"])
                elif "left_context_size" in info:
                    left_context_size[i] = int(info["left_context_size"])
                # req_id is set by GPUModelRunner._build_req_infos (gpu_model_runner.py
                # ~line 1417: ``req_infos["req_id"] = req_id``) and reaches us here
                # via runtime_additional_information. Used to key the per-request
                # overlap-add crossfade buffer ``self._prev_tail``. Fall back to the
                # batch position string when absent (e.g. dummy runtime info during
                # capacity estimation -- gpu_generation_model_runner.py:773).
                rid = meta.get("req_id") if isinstance(meta, dict) else None
                if rid is None and isinstance(info, dict):
                    rid = info.get("req_id") or info.get("request_id")
                if isinstance(rid, (list, tuple)) and rid:
                    rid = rid[0]
                if rid is None:
                    rid = f"__batch_pos_{i}"
                req_ids_per_idx[i] = str(rid)

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

        # Track which req_ids appeared in this batch so we can lazily evict
        # stale ``_prev_tail`` entries (defensive cap against memory growth
        # if request_ids are reused or the finish signal never arrives).
        seen_req_ids: set[str] = set()

        if not valid_codes:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty] * num_req, "sr": [sr_tensor] * num_req},
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes[0]
                logger.warning(
                    "[DEBUG-VIENEU] codec first-batch: batch=%d per-request frames=%d uniq=%d "
                    "range=[%d,%d] (decoded_samples will be frames*480 @24kHz)",
                    len(valid_codes),
                    c.numel(),
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                )
            except Exception:
                pass

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            codes = valid_codes[j]
            ctx_frames = parsed_ctx_frames[idx]
            total_frames = parsed_total_frames[idx]
            rid = req_ids_per_idx[idx] or f"__batch_pos_{idx}"
            seen_req_ids.add(rid)

            # NeuCodec.decode_code expects [1, 1, num_frames] integer codes.
            codes_1_1_f = codes.reshape(1, 1, -1)
            with torch.amp.autocast("cuda", enabled=False):
                wav = self._codec.decode_code(codes_1_1_f)
            wav = wav.reshape(-1).to(dtype=torch.float32)

            if ctx_frames <= 0 or total_frames <= 0:
                # ONE-SHOT (codec_chunk_frames huge, ctx disabled) OR the first
                # streaming chunk (chunk 0 has no left context from the
                # connector). We emit the full decode (C samples) unchanged.
                # For chunk 0 of a streaming request we ALSO seed prev_tail with
                # the last ``min(O, total_frames-1)`` frames so chunk 1 can
                # crossfade against it -- O is read from the connector config
                # (codec_left_context_frames) stashed on the module at init.
                # For true one-shot (huge total_frames) we clear stale tail.
                o_cfg = int(getattr(self, "_ola_overlap_frames", 0))
                if o_cfg > 0 and 0 < total_frames <= 4 * o_cfg:
                    # chunk 0 of streaming: emit ONLY the non-overlap part
                    # (C - O samples); the last O samples are buffered as
                    # prev_tail so chunk 1 can crossfade against them. This
                    # prevents the overlap from being emitted twice (echo).
                    seed_o = min(o_cfg, total_frames)
                    spf = wav.numel() / max(total_frames, 1)
                    seed_samples = max(0, min(int(round(seed_o * spf)), int(wav.numel())))
                    self._prev_tail[rid] = (
                        wav[-seed_samples:].clone().cpu() if seed_samples > 0 else wav[:0].clone().cpu(),
                        int(seed_samples),
                    )
                    emit_wav = wav[: int(wav.numel()) - seed_samples] if seed_samples > 0 else wav
                    audios[idx] = emit_wav.to(dtype=torch.float32, device="cpu")
                else:
                    # One-shot: clear stale tail, emit full decode unchanged.
                    self._prev_tail.pop(rid, None)
                    audios[idx] = wav.to(dtype=torch.float32, device="cpu")
                continue

            # STREAMING overlap-add crossfade (no echo).
            # ``wav`` is the decode of ``[ctx(=O) + new(=C)]`` frames (O+C
            # samples). Its first ``O * samples_per_frame`` samples (head_overlap)
            # re-decode the previous chunk's tail and should match it exactly
            # when O >= conv receptive field. Crossfade head_overlap with the
            # buffered ``prev_tail`` in the sample domain, then emit ONLY:
            #   [crossfaded_overlap (O samples)] + [middle non-overlap (C-O samples)]
            # i.e. ``C`` samples total (O + (C - O)). The previous chunk emitted
            # its own ``C - O`` non-overlap samples and buffered its last ``O``
            # as prev_tail -- it did NOT emit that tail. So the timeline is:
            #   chunk 0   emit: [wav minus its last O]            (C - O samples)
            #   chunk k   emit: [crossfaded overlap (O)] + [middle (C - O)]  = C
            # Each chunk contributes exactly C NEW samples (the O overlap is
            # re-decoded only for the splice, never emitted twice), so total
            # audio = ~N*C, NOT 2*N*C. The previous code emitted the full
            # [ctx+new] every chunk -> ~2x audio -> echo (WAV accounting showed
            # codec_frames=455 for codec_codes_emitted=230, ~2x).
            samples_per_frame = wav.numel() / max(total_frames, 1)
            overlap_samples = int(round(ctx_frames * samples_per_frame))
            overlap_samples = max(0, min(overlap_samples, int(wav.numel()) - 1))

            prev_entry = self._prev_tail.get(rid)
            if prev_entry is None or overlap_samples <= 0:
                # No prev_tail yet: emit the non-overlap part only (no splice)
                # and buffer the tail O for the next chunk.
                self._prev_tail[rid] = (
                    wav[-overlap_samples:].clone().cpu() if overlap_samples > 0 else wav[:0].clone().cpu(),
                    int(overlap_samples),
                )
                audios[idx] = wav[overlap_samples:].to(dtype=torch.float32, device="cpu")
                continue

            prev_tail, prev_overlap = prev_entry
            # Size the crossfade to the SHORTER of the two overlaps (the codec
            # hop length can deviate slightly per decode; prev_overlap was the
            # overlap size of the previous chunk, overlap_samples is this
            # chunk's head overlap).
            n_cross = min(int(prev_overlap), int(overlap_samples))
            n_cross = max(0, min(n_cross, prev_tail.numel(), int(wav.numel())))

            head_overlap = wav[:n_cross]
            # Align prev_tail tail to ``n_cross`` samples (it may be longer if
            # the previous chunk buffered more than we need now).
            tail_use = prev_tail[-n_cross:] if n_cross > 0 else prev_tail[:0]

            # Emit [crossfaded overlap (O)] + [middle non-overlap (C-O)].
            # ``middle`` excludes both the head overlap AND the tail overlap
            # (the tail overlap is buffered for chunk k+1, NOT emitted here).
            if overlap_samples > 0:
                middle = wav[n_cross : int(wav.numel()) - overlap_samples]
            else:
                middle = wav[n_cross:]
            if n_cross > 0:
                fade_in = _crossfade_window(n_cross).to(head_overlap.device, head_overlap.dtype)
                fade_out = 1.0 - fade_in
                tail_use_dev = tail_use.to(head_overlap.device, head_overlap.dtype)
                crossfaded = tail_use_dev * fade_out + head_overlap * fade_in
                out = torch.cat([crossfaded, middle], dim=0)
            else:
                out = middle

            # Buffer this chunk's tail overlap for the NEXT crossfade. Keep it
            # on CPU so the buffer doesn't pin GPU memory across chunks.
            new_tail = wav[-overlap_samples:].clone().cpu() if overlap_samples > 0 else wav[:0].clone().cpu()
            self._prev_tail[rid] = (new_tail, int(overlap_samples))

            audios[idx] = out.to(dtype=torch.float32, device="cpu")

        # Lazy eviction of stale per-request crossfade state. Keep only entries
        # for req_ids seen in this batch, plus a hard cap on the dict size.
        if len(self._prev_tail) > self._prev_tail_max or any(
            rid not in seen_req_ids for rid in list(self._prev_tail.keys())
        ):
            for rid in list(self._prev_tail.keys()):
                if rid not in seen_req_ids:
                    self._prev_tail.pop(rid, None)
            # Hard cap: drop oldest by insertion order if still too large.
            while len(self._prev_tail) > self._prev_tail_max:
                self._prev_tail.pop(next(iter(self._prev_tail)))

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audios, "sr": srs},
        )
