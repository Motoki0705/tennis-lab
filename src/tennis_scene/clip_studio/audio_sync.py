"""Audio-based sync offset estimation across cameras.

Cameras filming the same match record the same soundscape (ball hits, crowd,
umpire calls), so cross-correlating loudness envelopes yields the relative
time lag between recordings. Lags are converted to the project sync
convention ``local = global + offset_sec`` relative to a reference camera.

Precision is bounded by the envelope rate (default 100 Hz -> 10 ms), which is
finer than one frame at typical 25-60 fps; fine-tuning by frame nudge in the
GUI remains available on top.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.utils.video.audio import audio_envelope, read_audio_mono

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioSyncResult:
    """Estimated sync offsets for all cameras.

    Attributes:
        offsets_sec: Per-camera ``offset_sec`` (project convention); the
            reference camera keeps its provided offset.
        confidences: Normalized correlation peak per camera in [0, 1]
            (1.0 for the reference itself).
        reference_index: Index of the reference camera.
    """

    offsets_sec: list[float]
    confidences: list[float]
    reference_index: int


def estimate_lag_seconds(
    signal: NDArray[np.float32],
    reference: NDArray[np.float32],
    *,
    rate: float,
) -> tuple[float, float]:
    """Estimate how much later ``signal`` runs relative to ``reference``.

    A positive lag means an event appears ``lag`` seconds later in ``signal``
    than in ``reference`` (``signal(t) ~ reference(t - lag)``).

    Returns:
        ``(lag_seconds, confidence)`` where confidence is the normalized
        correlation peak in [0, 1].
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if signal.ndim != 1 or reference.ndim != 1:
        raise ValueError("signal and reference must be 1-D arrays")
    if signal.size < 2 or reference.size < 2:
        raise ValueError("signal and reference must contain at least 2 samples")

    a = signal.astype(np.float64) - float(np.mean(signal))
    b = reference.astype(np.float64) - float(np.mean(reference))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        raise ValueError("cannot correlate constant signals")

    full_length = a.size + b.size - 1
    nfft = 1 << (full_length - 1).bit_length()
    spectrum = np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft))
    correlation = np.fft.irfft(spectrum, nfft)
    # Rearrange circular correlation into lags -(len(b)-1) .. len(a)-1.
    negative_lags = correlation[nfft - (b.size - 1) :]
    positive_lags = correlation[: a.size]
    full = np.concatenate([negative_lags, positive_lags])
    peak_index = int(np.argmax(full))
    lag_samples = peak_index - (b.size - 1)
    confidence = float(np.clip(full[peak_index] / norm, 0.0, 1.0))
    return lag_samples / rate, confidence


def estimate_audio_offsets(
    video_paths: Sequence[str | Path],
    *,
    reference_index: int = 0,
    reference_offset_sec: float = 0.0,
    sample_rate: int = 8000,
    envelope_rate: float = 100.0,
    max_seconds: float | None = None,
) -> AudioSyncResult:
    """Estimate per-camera sync offsets from the videos' audio tracks.

    Args:
        video_paths: One video per camera.
        reference_index: Camera whose offset is kept fixed.
        reference_offset_sec: Offset assigned to the reference camera.
        sample_rate: Audio decode rate.
        envelope_rate: Envelope rate (sets lag resolution, 100 Hz -> 10 ms).
        max_seconds: Optional cap on decoded audio duration per video.

    Raises:
        ValueError: If a video has no usable audio or arguments are invalid.
    """
    paths = [Path(path) for path in video_paths]
    if not paths:
        raise ValueError("video_paths must contain at least one video")
    if not 0 <= reference_index < len(paths):
        raise ValueError(
            f"reference_index {reference_index} out of range [0, {len(paths)})"
        )

    envelopes: list[NDArray[np.float32]] = []
    for path in paths:
        LOGGER.info(f"Decoding audio envelope: {path}")
        samples = read_audio_mono(
            path, sample_rate=sample_rate, max_seconds=max_seconds
        )
        envelopes.append(
            audio_envelope(
                samples, sample_rate=sample_rate, envelope_rate=envelope_rate
            )
        )

    reference_envelope = envelopes[reference_index]
    offsets: list[float] = []
    confidences: list[float] = []
    for index, envelope in enumerate(envelopes):
        if index == reference_index:
            offsets.append(float(reference_offset_sec))
            confidences.append(1.0)
            continue
        lag_sec, confidence = estimate_lag_seconds(
            envelope, reference_envelope, rate=envelope_rate
        )
        # Event at local t_ref in the reference appears at t_ref + lag here,
        # so with local = global + offset: offset_i = offset_ref + lag.
        offsets.append(float(reference_offset_sec) + lag_sec)
        confidences.append(confidence)
        LOGGER.info(
            f"Audio sync {paths[index].name} vs {paths[reference_index].name}: "
            f"lag={lag_sec:+.3f}s confidence={confidence:.3f}"
        )
    return AudioSyncResult(
        offsets_sec=offsets,
        confidences=confidences,
        reference_index=reference_index,
    )


__all__ = ["AudioSyncResult", "estimate_audio_offsets", "estimate_lag_seconds"]
