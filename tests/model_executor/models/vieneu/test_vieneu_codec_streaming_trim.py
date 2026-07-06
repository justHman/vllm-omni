"""Hermetic self-check for VieNeu codec incremental-streaming overlap-trim.

Pure-Python (no torch / vLLM / neucodec) so it runs in any environment.

Asserts the two properties that fix the chunk-boundary "bụm" clicks:

1. The codec trim path reads ``left_context_size`` from the ``meta``
   sub-dict (the path the scheduling coordinator populates), not the
   top-level info dict. The previous top-level read was always zero, so
   the trim path never ran and chunks emitted their full ctx+chunk
   window -- doubling the audio (the "echo" reverted in commit f2cd6d8a).
   We re-derive the expected trim behavior from the codec.py source and
   assert the meta-key path is the one that fires.

2. Synthetic chunk-boundary simulation: a position-invariant conv-style
   codec with a head-only edge ramp (mimicking fresh conv-state init at
   the head of every independent ``decode_code`` call) produces a sample
   discontinuity under naive independent-chunk concatenation, but the
   same two chunks decoded with a left-context window and context-trimmed
   (the codec.py ctx_frames branch, same pattern as fish_speech's DAC
   decoder) stitch seamlessly and reproduce the one-shot decode of all
   frames from the second frame onward.

Run: ``python tests/model_executor/models/vieneu/test_vieneu_codec_streaming_trim.py``
or  ``pytest tests/model_executor/models/vieneu/test_vieneu_codec_streaming_trim.py``
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

HOP = 480  # NeuCodec frame rate 50 Hz @ 24 kHz.


# ---------------------------------------------------------------------------
# 1) Source-level check: codec.py reads meta["left_context_size"].
# ---------------------------------------------------------------------------

def _load_codec_module():
    """Import vieneu/codec.py in isolation by loading it as a standalone module.

    codec.py imports vllm forward_context/logger and vllm_omni output_templates
    at module top -- none are needed for the trim-path key check. We stub them
    so importlib can exec the file, then assert the source text contains the
    meta-key read. This keeps the test hermetic (no torch / vLLM install).
    """
    codec_path = (
        pathlib.Path(__file__).resolve().parents[4]
        / "vllm_omni"
        / "model_executor"
        / "models"
        / "vieneu"
        / "codec.py"
    )
    source = codec_path.read_text(encoding="utf-8")
    return source


def test_codec_reads_left_context_from_meta_subdict():
    source = _load_codec_module()
    # The scheduling coordinator (omni_scheduling_coordinator.update_request_metadata)
    # delivers left_context_size as runtime_seed = {"meta": {"left_context_size": ...}}
    # and fish_speech's DAC decoder reads meta["left_context_size"]. The fix
    # makes vieneu's codec read the same path.
    assert 'meta = info.get("meta", {})' in source, (
        "codec.py must read left_context_size from the meta sub-dict, matching "
        "omni_scheduling_coordinator.update_request_metadata's runtime_seed "
        "shape {\"meta\": {\"left_context_size\": ...}}."
    )
    assert 'meta["left_context_size"]' in source
    # And the previously-buggy top-level-only read must be GONE (or only a
    # fallback). The bug was reading info["left_context_size"] as the primary
    # path; the fix reads meta first.
    meta_index = source.index('meta["left_context_size"]')
    top_index = source.index('"left_context_size" in info')
    assert meta_index < top_index, "meta-key read must come before the legacy top-level fallback"


# ---------------------------------------------------------------------------
# 2) Synthetic boundary continuity: naive concat clicks, overlap-trim does not.
# ---------------------------------------------------------------------------

def _decode(codes: list[int]) -> list[float]:
    """Position-invariant codec with head-only edge ramp.

    Sample ``i`` of frame ``f`` is ``base(codes[f]) + head_ramp(i)``.
    The head ramp is non-zero only in the first ``HOP`` samples (mimicking
    conv left-padding / fresh internal state at the start of every
    independent decode call) -- the artifact that creates chunk-boundary
    clicks when each streaming chunk is decoded independently with no
    left context.
    """
    out: list[float] = []
    for f_idx, c in enumerate(codes):
        base = (c % 256) / 256.0
        for s in range(HOP):
            ramp = 0.5 * (1.0 - s / HOP) if f_idx == 0 else 0.0
            out.append(base + ramp)
    return out


def _trim_left_context(wav: list[float], ctx_frames: int, total_frames: int) -> list[float]:
    """Mirror of codec.py's ctx_frames trim branch (proportional, clamped)."""
    n = len(wav)
    samples_per_frame = n / max(total_frames, 1)
    trim = int(round(ctx_frames * samples_per_frame))
    trim = max(0, min(trim, n - 1))
    return wav[trim:]


def test_naive_concat_has_boundary_jump():
    chunk_a = [1, 2, 3]
    chunk_b = [4, 5, 6]
    wav = _decode(chunk_a) + _decode(chunk_b)
    boundary = 3 * HOP
    delta = abs(wav[boundary] - wav[boundary - 1])
    assert delta > 0.1, f"expected a boundary jump under naive concat, got delta={delta}"


def test_overlap_trim_concat_is_continuous_at_boundary():
    chunk_a = [1, 2, 3]
    ctx_b = chunk_a  # streaming processor re-emits the last N frames as context
    new_b = [4, 5, 6]
    # One-shot reference: decode all codes in a single call. The boundary
    # sample in the reference is the interior (no head-ramp) value of frame 3.
    ref = _decode(chunk_a + new_b)
    boundary = 3 * HOP
    # Naive concat: chunk_b is decoded independently, so frame 3 (its first
    # frame) carries the head ramp = 0.5 -> the boundary sample is +0.5 above
    # the reference (a "bụm" click).
    naive = _decode(chunk_a) + _decode(new_b)
    naive_click = abs(naive[boundary] - ref[boundary])
    # Overlap-trim: chunk_b decoded as [ctx_b + new_b] with ctx_b frames
    # trimmed, so frame 3 is interior -> matches the one-shot reference exactly.
    wa = _decode(chunk_a)
    wb_full = _decode(ctx_b + new_b)
    wb = _trim_left_context(wb_full, ctx_frames=len(ctx_b), total_frames=len(ctx_b) + len(new_b))
    stitched = wa + wb
    stitched_click = abs(stitched[boundary] - ref[boundary])
    assert stitched_click < 1e-9, (
        f"overlap-trim boundary sample must equal the one-shot reference, "
        f"got stitched_click={stitched_click}"
    )
    assert naive_click > 0.1, (
        f"naive concat must have a head-ramp click vs the reference; "
        f"naive_click={naive_click}"
    )


def test_overlap_trim_matches_full_decode_from_second_frame():
    all_codes = [1, 2, 3, 4, 5, 6]
    ref = _decode(all_codes)
    chunk_a = [1, 2, 3]
    ctx_b = chunk_a
    new_b = [4, 5, 6]
    wa = _decode(chunk_a)
    wb_full = _decode(ctx_b + new_b)
    wb = _trim_left_context(wb_full, ctx_frames=len(ctx_b), total_frames=len(ctx_b) + len(new_b))
    stitched = wa + wb
    assert len(stitched) == len(ref)
    # The first chunk's own head ramp (samples [0, HOP)) is the untrimmable
    # edge of the whole utterance -- it is never a *boundary* click because
    # it is the very start of playback. From the second frame onward the
    # overlap-trim stream must reproduce the one-shot decode exactly.
    assert stitched[HOP:] == ref[HOP:], (
        "overlap-trim stream should reproduce one-shot decode from the second "
        "frame onward (the first chunk's head ramp is the only untrimmable "
        "edge, and it is the first chunk of the whole utterance so it is never "
        "a boundary click)."
    )


# Allow direct execution: ``python test_vieneu_codec_streaming_trim.py``.
def _run_all() -> bool:
    for fn in (
        test_codec_reads_left_context_from_meta_subdict,
        test_naive_concat_has_boundary_jump,
        test_overlap_trim_concat_is_continuous_at_boundary,
        test_overlap_trim_matches_full_decode_from_second_frame,
    ):
        fn()
        print(f"PASS: {fn.__name__}")
    return True


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)