"""Preprocessing pipeline for VieNeu-TTS-v2 (TASK 8).

Pipeline (docs/Architecture.md Part B.7):

    text -> normalization -> G2P (phonemization) -> prompt construction
    -> tokenization -> LLM input

Text normalization and phonemization are delegated to the external
``sea-g2p`` package, exactly as the reference SDK does
(docs/Architecture.md Part B.7 step 2) -- reimplementing Vietnamese/English
G2P here would duplicate a maintained bilingual normalizer for no benefit.
This module owns everything *around* that call: chunking, prompt assembly
(text region + speech region per Part B.5), reference-voice resolution
(explicit ref_audio+ref_text, a named preset, or raw ref_codes), and preset
loading from ``voices.json``.

Voice cloning has no learned speaker-embedding path (Part B.6) -- identity
is carried purely by placing the reference audio's encoded speech tokens as
a literal prefix of the generated speech-token stream, paired with a
transcript of that reference audio. A transcript is mandatory; this module
raises rather than silently guessing one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tokenizer import CONTROL_TOKENS, speech_ids_to_tokens

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Reference SDK default (docs/Architecture.md Part B.7 step 2). Sentence-
# boundary-aware chunking keeps a single LM generation call bounded so it
# fits under VieNeuTalkerConfig.max_context_length together with the
# reference-voice prefix.
DEFAULT_MAX_CHARS = 256

# v2 has no chat-template wrapper (Part B.5: `use_chat_format` is only true
# for the plain v1 repo, never for `-v2`), so no branch for it exists here.


class VieNeuPresetError(ValueError):
    """Raised for missing/invalid preset voice lookups."""


@dataclass(frozen=True)
class ReferenceVoice:
    """A resolved reference voice: either preset codes or raw encoded ref audio."""

    ref_codes: list[int]
    ref_text: str
    name: str | None = None
    license_note: str | None = None


@dataclass(frozen=True)
class VieNeuPrompt:
    """A fully-assembled VieNeu prompt, ready for tokenization."""

    text: str
    """The literal prompt string (text region + speech region), per Part B.5."""

    chunk_index: int
    chunk_count: int


def load_presets(voices_json_path: str | Path) -> dict[str, ReferenceVoice]:
    """Load named preset voices from a checkpoint's ``voices.json``.

    Per docs/Architecture.md Part B.6: v2 presets store precomputed
    ``{codes, text, description}`` per voice, not audio files or embedding
    vectors. The v2 checkpoint ships exactly 7 presets (Binh, Tuyen, Vinh,
    Doan, Ly, Sơn, Ngoc) under a CC BY-NC 4.0 license distinct from the
    model weights' Apache-2.0 license -- callers surfacing preset voices in
    a UI should propagate ``license_note``.
    """
    path = Path(voices_json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    meta = data.get("meta") or {}
    license_note = meta.get("license")

    presets: dict[str, ReferenceVoice] = {}
    for name, entry in data.items():
        if name == "meta" or not isinstance(entry, dict):
            continue
        codes = entry.get("codes")
        text = entry.get("text")
        if not isinstance(codes, list) or not isinstance(text, str):
            continue
        presets[name] = ReferenceVoice(
            ref_codes=[int(c) for c in codes],
            ref_text=text,
            name=name,
            license_note=license_note,
        )
    return presets


def resolve_reference_voice(
    *,
    voice: str | None = None,
    ref_codes: Sequence[int] | None = None,
    ref_text: str | None = None,
    presets: dict[str, ReferenceVoice] | None = None,
    default_voice: str | None = None,
    encode_ref_audio: Any = None,
) -> ReferenceVoice:
    """Resolve a caller's voice request into a concrete ``ReferenceVoice``.

    Exactly one of these paths is taken, checked in order:
      1. ``ref_codes`` + ``ref_text`` given explicitly (already-encoded reference).
      2. ``voice`` names a known preset.
      3. ``encode_ref_audio`` is callable and the caller wants ad-hoc voice
         cloning from raw audio -- ``encode_ref_audio(audio) -> list[int]``
         performs the NeuCodec-encode step (owned by ``codec.py``, not this
         module, to keep the codec dependency out of prompt construction).
      4. Fall back to ``default_voice`` in ``presets`` (matches the SDK's
         silent-default behavior, Part B.6 point 5).

    A transcript is mandatory for every path -- voice cloning is not
    zero-shot from audio alone (Part B.6 point 3).
    """
    if ref_codes is not None:
        if not ref_text:
            raise ValueError(
                "Voice cloning from ref_codes requires a matching `ref_text` transcript "
                "(VieNeu voice cloning is not zero-shot from audio/codes alone)."
            )
        return ReferenceVoice(ref_codes=list(ref_codes), ref_text=ref_text)

    if voice is not None:
        if not presets or voice not in presets:
            available = sorted(presets) if presets else []
            raise VieNeuPresetError(f"Unknown voice preset {voice!r}. Available presets: {available}")
        return presets[voice]

    if encode_ref_audio is not None:
        if not ref_text:
            raise ValueError("Voice cloning from ref_audio requires a matching `ref_text` transcript.")
        encoded = encode_ref_audio()
        return ReferenceVoice(ref_codes=list(encoded), ref_text=ref_text)

    if default_voice is not None and presets and default_voice in presets:
        return presets[default_voice]

    raise ValueError(
        "No reference voice resolved: provide `voice` (preset name), "
        "`ref_codes`+`ref_text`, or a ref_audio path with a transcript."
    )


def normalize_and_chunk(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[str]:
    """Normalize and sentence-chunk input text via the external ``sea-g2p`` package.

    Mirrors the reference SDK's ``normalize_to_chunks``
    (docs/Architecture.md Part B.7 step 2): split on paragraph boundaries,
    normalize each paragraph (numbers/dates/currency/punctuation), then
    split into ``max_chars``-bounded chunks that prefer sentence
    boundaries, falling back to minor punctuation/whitespace, and only to a
    hard word-boundary cut if a single sentence exceeds ``max_chars``.

    Requires the ``sea-g2p`` package (a runtime dependency introduced by
    this integration, not previously used elsewhere in vllm-omni --
    see docs/Architecture.md Part C).
    """
    try:
        from sea_g2p import Normalizer
    except ImportError as e:
        raise ImportError(
            "VieNeu-TTS text normalization requires the `sea-g2p` package. "
            "Install it with `pip install sea-g2p`."
        ) from e

    normalizer = Normalizer()
    paragraphs = [p for p in text.split("\n") if p.strip()]
    normalized = [normalizer.normalize(p, punc_norm=True) for p in paragraphs]
    joined = "\n".join(normalized)
    return _split_into_chunks(joined, max_chars=max_chars)


def _split_into_chunks(text: str, *, max_chars: int) -> list[str]:
    """Sentence-boundary-aware chunking (Part B.7 step 2)."""
    sentence_end_chars = ".!?…"
    minor_break_chars = ",;:"

    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        window = remaining[: max_chars + 1]
        cut = -1
        for i in range(len(window) - 1, -1, -1):
            if window[i] in sentence_end_chars:
                cut = i + 1
                break
        if cut == -1:
            for i in range(len(window) - 1, -1, -1):
                if window[i] in minor_break_chars:
                    cut = i + 1
                    break
        if cut == -1:
            for i in range(len(window) - 1, -1, -1):
                if window[i].isspace():
                    cut = i + 1
                    break
        if cut == -1:
            cut = max_chars

        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    return [c for c in chunks if c]


def phonemize(text_chunks: list[str], *, language: str = "auto") -> list[str]:
    """Phonemize normalized text chunks via ``sea-g2p`` (Part B.7 step 3)."""
    try:
        from sea_g2p import G2P
    except ImportError as e:
        raise ImportError(
            "VieNeu-TTS phonemization requires the `sea-g2p` package. Install it with `pip install sea-g2p`."
        ) from e

    g2p = G2P()
    return g2p.phonemize_batch(text_chunks, punc_norm=True)


def build_prompt(
    *,
    input_text_phonemes: str,
    reference: ReferenceVoice,
    ref_text_phonemes: str,
    chunk_index: int = 0,
    chunk_count: int = 1,
    emotion_tag: str | None = None,
) -> VieNeuPrompt:
    """Assemble the literal prompt string per the layout in Part B.5.

        <|TEXT_PROMPT_START|>{emotion_tag}{ref_text_phonemes} {input_text_phonemes}<|TEXT_PROMPT_END|>
        <|SPEECH_GENERATION_START|>{ref_speech_tokens}

    Generation continues after the reference speech tokens until the model
    emits ``<|SPEECH_GENERATION_END|>`` -- that suffix is not part of the
    prompt; it's what generation is expected to produce.
    """
    text_prompt_start, text_prompt_end, speech_generation_start, _ = CONTROL_TOKENS

    emotion_prefix = emotion_tag or ""
    text_region = f"{emotion_prefix}{ref_text_phonemes} {input_text_phonemes}"
    ref_speech_tokens = speech_ids_to_tokens(reference.ref_codes)

    prompt = f"{text_prompt_start}{text_region}{text_prompt_end}{speech_generation_start}{ref_speech_tokens}"
    return VieNeuPrompt(text=prompt, chunk_index=chunk_index, chunk_count=chunk_count)


def build_prompts_for_text(
    text: str,
    *,
    reference: ReferenceVoice,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    language: str = "auto",
    emotion_tag: str | None = None,
) -> list[VieNeuPrompt]:
    """End-to-end: normalize/chunk/phonemize input text and reference transcript,
    then build one ``VieNeuPrompt`` per chunk (multi-chunk texts are joined
    downstream after independent generation+decode, per Part B.7 step 11 --
    that joining is the codec/stage-input-processor's responsibility, not this
    module's).
    """
    chunks = normalize_and_chunk(text, max_chars=max_chars)
    phonemized_chunks = phonemize(chunks, language=language)
    ref_text_phonemes = phonemize([reference.ref_text], language=language)[0]

    return [
        build_prompt(
            input_text_phonemes=chunk_phonemes,
            reference=reference,
            ref_text_phonemes=ref_text_phonemes,
            chunk_index=i,
            chunk_count=len(phonemized_chunks),
            emotion_tag=emotion_tag,
        )
        for i, chunk_phonemes in enumerate(phonemized_chunks)
    ]
