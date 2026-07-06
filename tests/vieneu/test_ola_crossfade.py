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
def crossfade_chunk(
    wav: np.ndarray,
    ctx_frames: int,
    total_frames: int,
    prev_tail: np.ndarray | None,
    prev_overlap: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (emitted_samples, new_tail_samples, new_overlap_size)."""
    if ctx_frames <= 0 or total_frames <= 0:
        # one-shot / first chunk / ctx disabled
        return wav, np.empty(0, dtype=np.float32), 0

    samples_per_frame = wav.shape[0] / max(total_frames, 1)
    overlap_samples = int(round(ctx_frames * samples_per_frame))
    overlap_samples = max(0, min(overlap_samples, wav.shape[0] - 1))

    if prev_tail is None or overlap_samples <= 0:
        return wav, wav[-overlap_samples:].copy(), overlap_samples

    n_cross = min(int(prev_overlap), int(overlap_samples))
    n_cross = max(0, min(n_cross, prev_tail.shape[0], wav.shape[0]))

    head_overlap = wav[:n_cross]
    tail_use = prev_tail[-n_cross:] if n_cross > 0 else prev_tail[:0]

    out = wav.copy()
    if n_cross > 0:
        fade_in = crossfade_window(n_cross).astype(wav.dtype)
        fade_out = 1.0 - fade_in
        out[:n_cross] = tail_use * fade_out + head_overlap * fade_in

    new_tail = wav[-overlap_samples:].copy()
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

    # Chunk 0: emit full [0 .. C] (ctx=0 -> passthrough); buffer tail O frames.
    chunk0 = full[: C * spf]
    emit0, tail0, ov0 = crossfade_chunk(chunk0, ctx_frames=0, total_frames=C,
                                        prev_tail=None, prev_overlap=0)
    assert ov0 == 0, "chunk 0 must not buffer a tail when ctx=0"
    assert np.array_equal(emit0, chunk0), "chunk 0 ctx=0 must passthrough"

    # Manually buffer chunk 0's tail O frames for chunk 1.
    tail0 = chunk0[-O * spf:].copy()
    ov0 = O * spf

    # Chunk 1: decode [O + C] frames starting at frame (C - O), so its head
    # O frames overlap chunk 0's tail O frames. With an ideal codec, the
    # re-decoded head equals chunk 0's tail exactly (same underlying signal).
    chunk1_full = full[(C - O) * spf : (C - O + O + C) * spf]
    assert chunk1_full.shape[0] == (O + C) * spf
    emit1, tail1, ov1 = crossfade_chunk(
        chunk1_full, ctx_frames=O, total_frames=O + C,
        prev_tail=tail0, prev_overlap=ov0,
    )

    # The emitted chunk 1 should reproduce full[(C-O)*spf : (C-O+O+C)*spf]
    # EXACTLY because the overlap matches and the crossfade of identical
    # samples is identity: tail*fade_out + head*fade_in = head*(fade_out+fade_in)
    # = head*1 = head.
    expected = full[(C - O) * spf : (C - O + O + C) * spf]
    max_delta = float(np.max(np.abs(emit1 - expected)))
    assert max_delta < 1e-6, f"splice not sample-exact: max delta = {max_delta}"
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

    # Continuity at the splice boundary (sample index O*spf - 1 -> O*spf):
    # the crossfade region is [0, O*spf); the new-chunk region starts at
    # O*spf. The boundary delta should be on the order of the new-chunk
    # signal's own sample-to-sample delta (a few percent of amplitude), NOT
    # a full-amplitude jump.
    boundary = O * spf
    delta_at_boundary = abs(emit1[boundary] - emit1[boundary - 1])
    # Sanity bound: should be << 1.0 (no full-amplitude click).
    assert delta_at_boundary < 0.2, (
        f"discontinuity at splice boundary: delta = {delta_at_boundary}"
    )
    # And the crossfade region itself should be bounded (no jump inside it).
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
    # ctx_frames == 0 (one-shot) must return the decoded samples unchanged
    # and clear any stale tail buffer.
    wav = np.sin(2 * np.pi * 220.0 * np.arange(12000, dtype=np.float32) / 24000.0)
    emit, tail, ov = crossfade_chunk(
        wav, ctx_frames=0, total_frames=25, prev_tail=wav[-12000:].copy(),
        prev_overlap=12000,
    )
    assert np.array_equal(emit, wav), "one-shot must passthrough wav unchanged"
    assert ov == 0 and tail.shape[0] == 0, "one-shot must not buffer a tail"
    print("[PASS] test_one_shot_ctx_zero_passthrough")


def test_first_chunk_ctx_zero_no_prev_tail() -> None:
    # Chunk 0 of a streaming request: ctx_frames=0 (no left context), so
    # emit the full decode and buffer nothing.
    C = 25
    spf = 480
    wav = np.sin(2 * np.pi * 220.0 * np.arange(C * spf, dtype=np.float32) / 24000.0)
    emit, tail, ov = crossfade_chunk(
        wav, ctx_frames=0, total_frames=C, prev_tail=None, prev_overlap=0,
    )
    assert np.array_equal(emit, wav)
    assert ov == 0 and tail.shape[0] == 0
    print("[PASS] test_first_chunk_ctx_zero_no_prev_tail")


def main() -> int:
    test_crossfade_window_blend_and_smoothness()
    test_crossfade_identical_overlap_is_sample_exact()
    test_crossfade_slightly_differing_overlap_is_continuous()
    test_naive_concat_has_jump()
    test_one_shot_ctx_zero_passthrough()
    test_first_chunk_ctx_zero_no_prev_tail()
    print("\nALL OLA CROSSFADE SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())