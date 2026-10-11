"""Decoded-frame checks and a CLIP prompt-alignment proxy for generated video (numpy only; no model imports).

Generated video has no ground truth here, so these are sanity checks with stated thresholds, not quality metrics:

* ``frame_checks`` — are the decoded frames finite, not blank, moving, and free of abrupt cuts?
* ``first_frame_fidelity`` — for image-to-video, how close is the first frame to the conditioning image?
* ``alignment_summary`` — given CLIP cosine similarities of every frame to the clip's own prompt and to the other
  capability's prompt (the swapped-prompt baseline), does the clip sit closer to its own prompt?

CLIP similarities are another frozen model's uncalibrated opinion (UNC2); they are not probabilities and not a
measure of video quality.
"""
# ruff: noqa: E501  -- contract strings and refusal messages are kept on one line so they read as one rule

from __future__ import annotations

from typing import Any

import numpy as np

BLANK_STD = 0.01  # per-frame pixel standard deviation (0..1 scale) below which a frame counts as blank
FROZEN_DIFF = 0.002  # mean absolute change between consecutive frames below which the pair counts as frozen
CUT_DIFF = 0.25  # mean absolute change above which a consecutive pair counts as an abrupt cut
MIN_MOTION = 0.002  # mean change over the clip below which the clip counts as static


def to_unit(frames: Any) -> np.ndarray:
    """Frames as float32 in 0..1, shape (F, H, W, 3); uint8 input is divided by 255."""
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected frames shaped (F, H, W, 3), got {array.shape}")
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    return array.astype(np.float32)


def to_uint8(frames: Any) -> np.ndarray:
    """Frames as uint8 (F, H, W, 3), clipping a float 0..1 input; non-finite values become 0."""
    unit = np.nan_to_num(to_unit(frames), nan=0.0, posinf=1.0, neginf=0.0)
    return (unit.clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def frame_checks(frames: Any, *, expected_frames: int, height: int, width: int) -> dict[str, Any]:
    """Structural and temporal sanity checks on decoded frames; ``passed`` is True only if every check holds."""
    unit = to_unit(frames)
    finite = np.isfinite(unit)
    finite_fraction = float(finite.mean())
    safe = np.where(finite, unit, 0.0)
    per_frame_std = safe.reshape(len(safe), -1).std(axis=1)
    diffs = np.abs(np.diff(safe, axis=0)).reshape(len(safe) - 1, -1).mean(axis=1) if len(safe) > 1 else np.zeros(0)
    report = {
        "shape": list(unit.shape),
        "expected_shape": [expected_frames, height, width, 3],
        "finite_fraction": round(finite_fraction, 6),
        "blank_frames": int((per_frame_std < BLANK_STD).sum()),
        "frame_std": {"min": round(float(per_frame_std.min()), 4), "mean": round(float(per_frame_std.mean()), 4)},
        "consecutive_change": {
            "mean": round(float(diffs.mean()), 5) if diffs.size else 0.0,
            "min": round(float(diffs.min()), 5) if diffs.size else 0.0,
            "max": round(float(diffs.max()), 5) if diffs.size else 0.0,
        },
        "frozen_pairs": int((diffs < FROZEN_DIFF).sum()),
        "abrupt_cut_pairs": int((diffs > CUT_DIFF).sum()),
        "thresholds": {"blank_std": BLANK_STD, "frozen_diff": FROZEN_DIFF, "cut_diff": CUT_DIFF, "min_motion": MIN_MOTION},
    }
    checks = {
        "shape_as_requested": report["shape"] == report["expected_shape"],
        "all_values_finite": finite_fraction == 1.0,
        "no_blank_frames": report["blank_frames"] == 0,
        "clip_has_motion": report["consecutive_change"]["mean"] >= MIN_MOTION,
        "no_abrupt_cuts": report["abrupt_cut_pairs"] == 0,
    }
    return {**report, "checks": checks, "passed": all(checks.values())}


def first_frame_fidelity(first_frame: Any, conditioning: Any) -> dict[str, Any]:
    """Mean absolute difference and PSNR (dB, 0..1 scale) between the first generated frame and the conditioning image."""
    a = to_unit(np.asarray(first_frame)[None])[0]
    b = to_unit(np.asarray(conditioning)[None])[0]
    if a.shape != b.shape:
        raise ValueError(f"first frame {a.shape} and conditioning image {b.shape} differ in shape")
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
    return {"mean_abs_diff": round(float(np.mean(np.abs(a - b))), 5), "psnr_db": round(psnr, 2)}


def alignment_summary(own: Any, other: Any) -> dict[str, Any]:
    """Summarise per-frame CLIP cosine similarity to the clip's own prompt versus the swapped (other) prompt."""
    own_arr = np.asarray(own, dtype=np.float64).ravel()
    other_arr = np.asarray(other, dtype=np.float64).ravel()
    if own_arr.shape != other_arr.shape or own_arr.size == 0:
        raise ValueError("own and other similarities must be non-empty and the same length")
    margin = own_arr - other_arr
    return {
        "own_prompt_similarity": round(float(own_arr.mean()), 4),
        "swapped_prompt_similarity": round(float(other_arr.mean()), 4),
        "margin": round(float(margin.mean()), 4),
        "frames_closer_to_own_prompt": round(float((margin > 0).mean()), 4),
        "frames_scored": int(own_arr.size),
        "reading": "uncalibrated CLIP ViT-L/14 cosine similarity; a positive margin means the frames sit closer to their own prompt than to the other capability's prompt",
    }
