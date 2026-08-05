"""Audio decoding helpers for media files (PyAV-backed).

Complements the OpenCV-based frame readers with access to the audio track,
which video-frame APIs cannot reach. The primary consumer is audio-based
multi-camera synchronization, but the helpers are task-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray


def read_audio_mono(
    media_path: str | Path,
    *,
    sample_rate: int = 8000,
    max_seconds: float | None = None,
) -> NDArray[np.float32]:
    """Decode the first audio stream of ``media_path`` to mono float32 samples.

    Args:
        media_path: Any container PyAV can open (mp4, mkv, wav, ...).
        sample_rate: Output sample rate; the stream is resampled to it.
        max_seconds: Optional cap on the decoded duration (leading samples).

    Returns:
        1-D float32 array of mono samples at ``sample_rate``.

    Raises:
        ValueError: If the file has no audio stream, decodes to zero samples,
            or an argument is out of range.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError(f"max_seconds must be positive, got {max_seconds}")

    path = Path(media_path)
    max_samples = None if max_seconds is None else int(round(max_seconds * sample_rate))

    chunks: list[NDArray[np.float32]] = []
    total_samples = 0
    with av.open(str(path)) as container:
        audio_streams = [
            stream for stream in container.streams if stream.type == "audio"
        ]
        if not audio_streams:
            raise ValueError(f"No audio stream found in: {path}")
        stream = audio_streams[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)

        def append_frames(frames: list[av.AudioFrame]) -> None:
            nonlocal total_samples
            for out_frame in frames:
                samples = np.asarray(out_frame.to_ndarray(), dtype=np.float32).reshape(
                    -1
                )
                if samples.size == 0:
                    continue
                chunks.append(samples)
                total_samples += samples.size

        for frame in container.decode(stream):
            if not isinstance(frame, av.AudioFrame):
                raise TypeError(
                    "Decoding the selected audio stream returned a non-audio frame."
                )
            append_frames(resampler.resample(frame))
            if max_samples is not None and total_samples >= max_samples:
                break
        else:
            append_frames(resampler.resample(None))

    if total_samples == 0:
        raise ValueError(f"Audio stream decoded to zero samples: {path}")
    mono = np.concatenate(chunks)
    if max_samples is not None:
        mono = mono[:max_samples]
    return np.ascontiguousarray(mono, dtype=np.float32)


def audio_envelope(
    samples: NDArray[np.float32],
    *,
    sample_rate: int,
    envelope_rate: float = 100.0,
) -> NDArray[np.float32]:
    """Reduce mono samples to a windowed-RMS loudness envelope.

    Args:
        samples: 1-D mono samples.
        sample_rate: Sample rate of ``samples``.
        envelope_rate: Output envelope rate in Hz; ``sample_rate`` must be an
            integer multiple of it so windows tile the signal exactly.

    Returns:
        1-D float32 RMS envelope at ``envelope_rate``.

    Raises:
        ValueError: If rates are inconsistent or the signal is shorter than
            one envelope window.
    """
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1-D, got shape {samples.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if envelope_rate <= 0:
        raise ValueError(f"envelope_rate must be positive, got {envelope_rate}")

    hop = int(round(sample_rate / envelope_rate))
    if hop <= 0 or abs(hop * envelope_rate - sample_rate) > 1e-6:
        raise ValueError(
            f"sample_rate={sample_rate} must be an integer multiple of "
            f"envelope_rate={envelope_rate}"
        )

    num_windows = samples.size // hop
    if num_windows == 0:
        raise ValueError(
            f"signal too short for one envelope window: {samples.size} samples, hop={hop}"
        )
    trimmed = samples[: num_windows * hop].astype(np.float64).reshape(num_windows, hop)
    rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
    return np.asarray(rms, dtype=np.float32)


__all__ = ["audio_envelope", "read_audio_mono"]
