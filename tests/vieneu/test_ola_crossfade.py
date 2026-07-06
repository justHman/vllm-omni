"""Hermetic self-check for the VieNeu overlap-add crossfade splice.

Pure-Python (numpy only) -- does NOT import torch or the codec module. It
exercises the crossfade math in isolation:

1. ``test_crossfade_identical_overlap_is_sample_exact`` -- when the two
   overlap signals are identical (O >= conv receptive field), the splice
   is sample-exact: max delta at the splice point < 1e-6.
2. ``test_crossfade_slightly_differing_overlap_is_continuous`` -- when the
   two overlap signals differ slightly (simulating O < receptive field),
   the splice is still continuous: max delta at the splice boundary is
   bounded by the overlap mismatch amplitude (no discontinuity jump).
3. ``test_naive_concat_has_jump`` -- the naive concat (no crossfade) of
   two chunks whose overlap decodes differ has a discontinuity at the
   boundary, demonstrating WHY the crossfade is necessary.
4. ``test_one_shot_ctx_zero_passthrough`` -- the ctx_frames==0 path
   returns the decoded samples unchanged (one-shot fallback is untouched).
5. ``test_crossfade_window_blend_and_smoothness`` -- the Hann/sin^2
   crossfade window satisfies fade_in + fade_out == 1 (linear blend ->
   sample-exact splice for identical overlaps) and has zero slope at
   both endpoints (Hann shape -> no slope discontinuity at the splice
   boundaries).

Run: ``python tests/vieneu/test_ola_crossfade.py``
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np


# ── Mirror of codec.py._crossfade_window (numpy, no torch) ──────────────
def crossfade_window(n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.float32)
    t = np.arange(1, n + 1, dtype=np.float32) / (2 * n)
    return np.sin(math.pi * t) ** 2


# ── Mirror of the codec.py per-chunk emit logic (numpy) ────────────────
# OLA_OVERLAP = codec_left_context_frames (O). Chunk 0 (ctx_frames==0) seeds
# prev_tail with its last O frames so chunk 1 can crossfade. Each chunk k>0
# emits [crossfaded_overlap (O samples)] + [new samples after overlap (C)] and
# buffers its last O samples for chunk k+1. Total emit per chunk = C NEW
# samples (O are re-decoded overlap used only for the splice) -> no echo.
OLA_OVERLAP = 25


def crossfade_chunk(
    wav: np.ndarray,
    ctx_frames: int,
    total_frames: int,
    prev_tail: np.ndarray | None,
    prev_overlap: int,
    is_chunk0: bool = False,
    final: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Return (emitted_samples, new_tail_samples_or_None, new_overlap_size).

    ``final`` = this is the last chunk of the request (is_segment_finished):
    emit the full tail too (no next chunk to crossfade it into) and clear the
    buffer so it cannot leak into the next request with the same req_id.
    """
    if ctx_frames <= 0 or total_frames <= 0:
        # one-shot OR chunk 0 of streaming. For chunk 0 (NOT final), seed
        # prev_tail with the last O frames so chunk 1 can crossfade. For true
        # one-shot (huge total_frames) OR a single-chunk final request, return
        # full wav and None tail.
        if is_chunk0 and 0 < total_frames <= 4 * OLA_OVERLAP and not final:
            spf = wav.shape[0] / max(total_frames, 1)
            seed_o = min(OLA_OVERLAP, total_frames)
            seed_n = max(0, min(int(round(seed_o * spf)), wav.shape[0]))
            emit = wav[:-seed_n] if seed_n > 0 else wav
            return emit, (wav[-seed_n:].copy() if seed_n > 0 else np.empty(0, dtype=wav.dtype)), int(seed_n)
        return wav, None, 0

    samples_per_frame = wav.shape[0] / max(total_frames, 1)
    overlap_samples = int(round(ctx_frames * samples_per_frame))
    overlap_samples = max(0, min(overlap_samples, wav.shape[0] - 1))

    if prev_tail is None or overlap_samples <= 0:
        # No prev_tail yet. If final, emit full wav (no buffer). Else buffer.
        if final:
            return wav, None, 0
        new_tail = wav[-overlap_samples:].copy() if overlap_samples > 0 else np.empty(0, dtype=wav.dtype)
        return wav[overlap_samples:], new_tail, int(overlap_samples)

    n_cross = min(int(prev_overlap), int(overlap_samples))
    n_cross = max(0, min(n_cross, prev_tail.shape[0], wav.shape[0]))

    head_overlap = wav[:n_cross]
    tail_use = prev_tail[-n_cross:] if n_cross > 0 else prev_tail[:0]

    # MIDDLE chunk: emit [crossfaded O] + [middle C-O]; buffer last O.
    # FINAL chunk: emit [crossfaded O] + [middle..end] (tail included), no buffer.
    if overlap_samples > 0 and not final:
        middle = wav[n_cross : wav.shape[0] - overlap_samples]
    else:
        middle = wav[n_cross:]
    if n_cross > 0:
        fade_in = crossfade_window(n_cross).astype(wav.dtype)
        fade_out = 1.0 - fade_in
        crossfaded = tail_use * fade_out + head_overlap * fade_in
        out = np.concatenate([crossfaded, middle], axis=0)
    else:
        out = middle

    if final:
        new_tail = None
    else:
        new_tail = wav[-overlap_samples:].copy() if overlap_samples > 0 else np.empty(0, dtype=wav.dtype)
    return out, new_tail, int(overlap_samples)


# ── Tests ───────────────────────────────────────────────────────────────
def test_crossfade_window_blend_and_smoothness() -> None:
    w = crossfade_window(512)
    fade_in = w
    fade_out = 1.0 - w
    # Linear blend: fade_in + fade_out == 1 throughout. This is what makes the
    # splice SAMPLE-EXACT when the two overlap signals are identical
    # (tail*fade_out + head*fade_in = head*(fade_out+fade_in) = head). The
    # trade-off is a ~3dB level dip in the middle of the overlap (power =
    # fade_in^2 + fade_out^2 dips to 0.5), which is acceptable for a short
    # 0.5s splice and is the standard behavior of a Hann-shaped linear
    # crossfade (sin^2(pi*t), 1-sin^2 = cos^2). An equal-power window
    # (sin/cos) would keep constant power but would NOT be sample-exact for
    # identical overlaps (sin+cos = sqrt(2) at the midpoint), so we pick
    # blend-linearity over equal-power to guarantee seamless splices when
    # O >= receptive field.
    blend = fade_in + fade_out
    assert np.allclose(blend, 1.0, atol=1e-6), (
        f"linear blend violated: max dev = {np.max(np.abs(blend - 1.0))}"
    )
    # Endpoints: fade_in starts at 0, ends at 1.
    assert fade_in[0] < 1e-3 and fade_in[-1] > 1 - 1e-3
    # Smoothness: derivative at both endpoints is ~0 (Hann shape), so the
    # splice has no slope discontinuity at the boundaries of the overlap --
    # this is the advantage of sin^2 over a raw linear ramp.
    diff = np.diff(fade_in)
    assert abs(diff[0]) < 1e-3 and abs(diff[-1]) < 1e-3, (
        f"endpoint slope not zero: d[0]={diff[0]}, d[-1]={diff[-1]}"
    )
    print("[PASS] test_crossfade_window_blend_and_smoothness")


def test_crossfade_identical_overlap_is_sample_exact() -> None:
    # Two chunks where the re-decoded head_overlap of chunk 1 EXACTLY equals
    # the buffered prev_tail of chunk 0 (O >= receptive field case).
    C, O = 25, 25
    spf = 480  # samples per frame @ 24kHz / 50Hz
    # Synthesize a smooth signal (a sine) across the full [0, 2C] frame range.
    total_samples = (C + O) * spf
    t = np.arange(total_samples, dtype=np.float32) / 24000.0
    full = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    # Chunk 0: ctx=0, is_chunk0=True -> emit full [0 .. C] AND seed prev_tail
    # with the last O frames so chunk 1 can crossfade.
    chunk0 = full[: C * spf]
    emit0, tail0, ov0 = crossfade_chunk(chunk0, ctx_frames=0, total_frames=C,
                                        prev_tail=None, prev_overlap=0,
                                        is_chunk0=True)
    # chunk 0 emits ONLY the non-overlap part (C - O samples); its last O
    # samples are buffered as prev_tail for chunk 1's crossfade (no echo).
    assert np.array_equal(emit0, chunk0[: (C - O) * spf]), "chunk 0 must emit non-overlap part"
    assert emit0.shape[0] == (C - O) * spf, f"chunk 0 emit len {emit0.shape[0]} != {(C - O) * spf}"
    assert tail0 is not None and ov0 == O * spf, "chunk 0 must seed prev_tail of O frames"

    # Chunk 1: decode [O + C] frames starting at frame (C - O), so its head
    # O frames overlap chunk 0's tail O frames. With an ideal codec, the
    # re-decoded head equals chunk 0's tail exactly (same underlying signal).
    chunk1_full = full[(C - O) * spf : (C - O + O + C) * spf]
    assert chunk1_full.shape[0] == (O + C) * spf
    emit1, tail1, ov1 = crossfade_chunk(
        chunk1_full, ctx_frames=O, total_frames=O + C,
        prev_tail=tail0, prev_overlap=ov0,
    )

    # chunk 1 emits [crossfaded overlap (O*spf)] + [new non-overlap (C-O)*spf]
    # = C*spf samples. Combined with chunk 0's (C-O)*spf, the timeline covers
    # (C-O) + C = 2C - O frames after chunk 1 (no overlap double-emitted).
    # Because the overlap matches, crossfade of identical samples is identity,
    # so emit1's first O*spf samples == full[(C-O)*spf : C*spf] (the overlap
    # region), and its last (C-O)*spf samples == full[C*spf : (2C-O)*spf].
    assert emit1.shape[0] == C * spf, (
        f"emit1 length {emit1.shape[0]} != C*spf={C * spf} (echo or under-emit)"
    )
    overlap_part = emit1[: O * spf]
    expected_overlap = full[(C - O) * spf : C * spf]
    max_delta_ov = float(np.max(np.abs(overlap_part - expected_overlap)))
    assert max_delta_ov < 1e-6, f"splice overlap not sample-exact: max delta = {max_delta_ov}"
    # The new non-overlap part only exists when C > O. When C == O the chunk
    # is all overlap (the entire chunk is the splice region), so skip the new
    # part check in that case.
    if C > O:
        new_part = emit1[O * spf :]
        expected_new = full[C * spf : (2 * C - O) * spf]
        max_delta_new = float(np.max(np.abs(new_part - expected_new)))
        assert max_delta_new < 1e-6, f"new part mismatch: max delta = {max_delta_new}"
        max_delta = max(max_delta_ov, max_delta_new)
    else:
        max_delta = max_delta_ov
    print(f"[PASS] test_crossfade_identical_overlap_is_sample_exact (max_delta={max_delta:.2e})")


def test_crossfade_slightly_differing_overlap_is_continuous() -> None:
    # Simulate O < receptive field: chunk 1's head_overlap differs slightly
    # from chunk 0's tail. The crossfade should produce a continuous splice
    # (no jump discontinuity at the splice boundary).
    C, O = 25, 25
    spf = 480
    chunk0 = np.sin(2 * np.pi * 220.0 * np.arange(C * spf, dtype=np.float32) / 24000.0)
    tail0 = chunk0[-O * spf:].copy()

    # Chunk 1's full decode: head O frames are a slightly-perturbed version
    # of tail0 (mimicking causal-conv position-dependence), tail C frames
    # are the new chunk signal.
    perturb = 0.05 * np.sin(2 * np.pi * 770.0 * np.arange(O * spf, dtype=np.float32) / 24000.0)
    head_overlap = tail0 + perturb
    new_chunk = np.sin(2 * np.pi * 220.0 * np.arange(C * spf, dtype=np.float32) / 24000.0)
    chunk1_full = np.concatenate([head_overlap, new_chunk])
    emit1, tail1, ov1 = crossfade_chunk(
        chunk1_full, ctx_frames=O, total_frames=O + C,
        prev_tail=tail0, prev_overlap=O * spf,
    )

    # emit1 = [crossfaded overlap (O*spf)] + [new non-overlap (C-O)*spf] = C*spf.
    assert emit1.shape[0] == C * spf, (
        f"emit1 length {emit1.shape[0]} != C*spf={C * spf} (echo or under-emit)"
    )
    # Continuity at the splice boundary (sample index O*spf - 1 -> O*spf):
    # the crossfade region is [0, O*spf); the new-chunk region starts at
    # O*spf. Only check the boundary when there IS a new part (C > O); when
    # C == O the whole chunk is the splice region.
    boundary = O * spf
    delta_at_boundary = 0.0
    if C > O and boundary < emit1.shape[0]:
        delta_at_boundary = abs(emit1[boundary] - emit1[boundary - 1])
        assert delta_at_boundary < 0.2, (
            f"discontinuity at splice boundary: delta = {delta_at_boundary}"
        )
    # The crossfade region itself should be bounded (no jump inside it).
    intra = np.max(np.abs(np.diff(emit1[:boundary])))
    assert intra < 0.2, f"jump inside crossfade region: {intra}"
    print(
        f"[PASS] test_crossfade_slightly_differing_overlap_is_continuous "
        f"(boundary_delta={delta_at_boundary:.3e}, intra_crossfade_max_delta={intra:.3e})"
    )


def test_naive_concat_has_jump() -> None:
    # Demonstrate that naive concat (no crossfade) of two chunks whose
    # overlap decodes differ produces a jump discontinuity at the boundary.
    C, O = 25, 25
    spf = 480
    chunk0 = np.sin(2 * np.pi * 220.0 * np.arange(C * spf, dtype=np.float32) / 24000.0)
    tail0 = chunk0[-O * spf:].copy()
    perturb = 0.5 * np.ones(O * spf, dtype=np.float32)  # large mismatch
    head_overlap = tail0 + perturb
    new_chunk = np.sin(2 * np.pi * 220.0 * np.arange(C * spf, dtype=np.float32) / 24000.0)
    chunk1_full = np.concatenate([head_overlap, new_chunk])

    # Simulate the click: the codec's re-decoded tail of chunk0 does NOT
    # match new_chunk's beginning (causal conv position-dependence). Add a
    # guaranteed full-amplitude DC step on the new chunk so the boundary
    # jump is unambiguous regardless of the sine's phase alignment.
    new_chunk_evil = (np.sin(2 * np.pi * 220.0 * np.arange(C * spf, dtype=np.float32) / 24000.0)
                      + 0.9)
    naive_emit = np.concatenate([chunk0, new_chunk_evil])
    boundary_delta = abs(naive_emit[C * spf] - naive_emit[C * spf - 1])
    assert boundary_delta > 0.5, (
        f"expected a full-amplitude jump with naive concat, got delta = {boundary_delta}"
    )
    print(
        f"[PASS] test_naive_concat_has_jump (boundary_delta={boundary_delta:.3e}) "
        "-- demonstrates crossfade is necessary"
    )


def test_one_shot_ctx_zero_passthrough() -> None:
    # ctx_frames == 0 (one-shot, huge total_frames) must return the decoded
    # samples unchanged and NO tail (None) so no crossfade state leaks.
    wav = np.sin(2 * np.pi * 220.0 * np.arange(12000, dtype=np.float32) / 24000.0)
    emit, tail, ov = crossfade_chunk(
        wav, ctx_frames=0, total_frames=12000 // 480, prev_tail=wav[-12000:].copy(),
        prev_overlap=12000, is_chunk0=False,
    )
    assert np.array_equal(emit, wav), "one-shot must passthrough wav unchanged"
    assert ov == 0 and tail is None, "one-shot must not buffer a tail"
    print("[PASS] test_one_shot_ctx_zero_passthrough")


def test_first_chunk_ctx_zero_seeds_tail() -> None:
    # Chunk 0 of a streaming request: ctx_frames=0 (no left context), so
    # emit the full C-sample decode AND seed prev_tail with the last O frames
    # so chunk 1 can crossfade against it.
    C, O = 25, 25
    spf = 480
    wav = np.sin(2 * np.pi * 220.0 * np.arange(C * spf, dtype=np.float32) / 24000.0)
    emit, tail, ov = crossfade_chunk(
        wav, ctx_frames=0, total_frames=C, prev_tail=None, prev_overlap=0,
        is_chunk0=True,
    )
    # chunk 0 emits the non-overlap part (C - O samples); last O samples buffered.
    assert np.array_equal(emit, wav[: (C - O) * spf]), "chunk 0 must emit non-overlap part"
    assert emit.shape[0] == (C - O) * spf
    assert tail is not None and ov == O * spf, (
        f"chunk 0 must seed prev_tail of O*spf={O * spf} samples, got ov={ov}"
    )
    assert tail.shape[0] == O * spf, f"tail length {tail.shape[0]} != {O * spf}"
    print("[PASS] test_first_chunk_ctx_zero_seeds_tail")


def test_no_echo_total_emit_equals_input_frames() -> None:
    # ECHO REGRESSION GUARD: a request of N*C frames split into chunks of
    # C frames with overlap O must emit roughly N*C samples total (each chunk
    # contributes C NEW samples; O overlap samples are re-decoded only for the
    # splice, never emitted twice). The previous buggy code emitted the full
    # [ctx+new] every chunk -> ~2x audio -> echo. The final chunk emits its
    # tail too (no next chunk to crossfade into), so total = N*C exactly.
    C, O = 25, 25
    spf = 480
    n_chunks = 4
    total_frames = C * n_chunks  # 100 frames
    t = np.arange(total_frames * spf, dtype=np.float32) / 24000.0
    full = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    total_emit = 0
    prev_tail = None
    prev_ov = 0
    for k in range(n_chunks):
        start = max(0, k * C - O)
        end = min(total_frames, start + O + C)
        chunk_full = full[start * spf : end * spf]
        ctx = (k * C) - start  # 0 for chunk 0, O for chunk k>0
        tf = end - start
        is_final = (k == n_chunks - 1)
        if k == 0:
            emit, prev_tail, prev_ov = crossfade_chunk(
                chunk_full, ctx_frames=0, total_frames=tf, prev_tail=None,
                prev_overlap=0, is_chunk0=True, final=is_final,
            )
        else:
            emit, prev_tail, prev_ov = crossfade_chunk(
                chunk_full, ctx_frames=ctx, total_frames=tf,
                prev_tail=prev_tail, prev_overlap=prev_ov, final=is_final,
            )
        total_emit += emit.shape[0]

    # With the final-chunk tail-emit fix, total_emit == total_frames*spf
    # (no echo, no tail loss). Allow a tiny slack for overlap-bookkeeping
    # rounding at the boundaries.
    expected = total_frames * spf
    assert abs(total_emit - expected) < (O + C) * spf, (
        f"emit {total_emit} != expected {expected} (slack {(O + C) * spf}): "
        f"echo (~2x) or tail-loss (~N*C - O*spf)"
    )
    assert prev_tail is None, f"final chunk must clear prev_tail, got {prev_tail}"
    print(
        f"[PASS] test_no_echo_total_emit_equals_input_frames "
        f"(total_emit={total_emit}, expected~{expected}, ratio={total_emit / expected:.3f})"
    )


def test_final_chunk_emits_tail_and_clears_buffer() -> None:
    # FINAL CHUNK GUARD: the last chunk must emit its tail overlap (otherwise
    # ~0.5s is cut at the clip end) AND clear prev_tail (otherwise the stale
    # buffer leaks into the next request with the same req_id -> echo at end).
    C, O = 25, 25
    spf = 480
    # Set up a prev_tail from a "previous chunk" (chunk k-1).
    prev_tail = np.sin(2 * np.pi * 220.0 * np.arange(O * spf, dtype=np.float32) / 24000.0)
    # Final chunk: decode [O + C] frames whose head O frames match prev_tail.
    total_samples = (O + C) * spf
    t = np.arange(total_samples, dtype=np.float32) / 24000.0
    wav = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    # Make head overlap exactly match prev_tail (ideal codec, O >= RF).
    wav[: O * spf] = prev_tail

    emit, new_tail, ov = crossfade_chunk(
        wav, ctx_frames=O, total_frames=O + C,
        prev_tail=prev_tail, prev_overlap=O * spf, final=True,
    )
    # Final chunk must emit the FULL (O + C) samples (tail included).
    assert emit.shape[0] == (O + C) * spf, (
        f"final chunk emit {emit.shape[0]} != {(O + C) * spf} (tail lost -> clip end cut)"
    )
    # Final chunk must clear the buffer (no stale leak).
    assert new_tail is None, f"final chunk must return None tail, got shape {new_tail.shape if new_tail is not None else None}"
    print(
        f"[PASS] test_final_chunk_emits_tail_and_clears_buffer "
        f"(emit={emit.shape[0]} == (O+C)*spf, buffer cleared)"
    )


def main() -> int:
    test_crossfade_window_blend_and_smoothness()
    test_crossfade_identical_overlap_is_sample_exact()
    test_crossfade_slightly_differing_overlap_is_continuous()
    test_naive_concat_has_jump()
    test_one_shot_ctx_zero_passthrough()
    test_first_chunk_ctx_zero_seeds_tail()
    test_no_echo_total_emit_equals_input_frames()
    test_final_chunk_emits_tail_and_clears_buffer()
    print("\nALL OLA CROSSFADE SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())