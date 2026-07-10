"""Tests for src/tennis_scene/clip_studio/audio_sync.py."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.clip_studio.audio_sync import (
    estimate_audio_offsets,
    estimate_lag_seconds,
)


def make_impulse_train(
    length: int, positions: list[int], rng: np.random.Generator
) -> np.ndarray:
    signal: np.ndarray = rng.normal(0.0, 0.01, size=length).astype(np.float32)
    for position in positions:
        signal[position : position + 20] += 1.0
    return signal


class TestEstimateLagSeconds:
    def test_positive_lag(self, rng: np.random.Generator) -> None:
        rate = 100.0
        reference = make_impulse_train(1000, [100, 380, 730], rng)
        delay = 57  # samples -> events appear later in `signal`
        signal = np.concatenate(
            [np.zeros(delay, dtype=np.float32), reference]
        )[:1000]
        lag, confidence = estimate_lag_seconds(signal, reference, rate=rate)
        assert lag == pytest.approx(delay / rate, abs=1.0 / rate)
        assert confidence > 0.5

    def test_negative_lag(self, rng: np.random.Generator) -> None:
        rate = 100.0
        reference = make_impulse_train(1000, [200, 500, 800], rng)
        advance = 40
        signal = np.concatenate(
            [reference[advance:], np.zeros(advance, dtype=np.float32)]
        )
        lag, _ = estimate_lag_seconds(signal, reference, rate=rate)
        assert lag == pytest.approx(-advance / rate, abs=1.0 / rate)

    def test_different_lengths(self, rng: np.random.Generator) -> None:
        rate = 100.0
        reference = make_impulse_train(1200, [150, 600, 900], rng)
        signal = np.concatenate(
            [np.zeros(30, dtype=np.float32), reference[:700]]
        )
        lag, _ = estimate_lag_seconds(signal, reference, rate=rate)
        assert lag == pytest.approx(0.3, abs=1.0 / rate)

    def test_constant_signal_raises(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            estimate_lag_seconds(
                np.ones(100, dtype=np.float32),
                np.ones(100, dtype=np.float32),
                rate=100.0,
            )

    def test_invalid_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate"):
            estimate_lag_seconds(
                np.zeros(10, dtype=np.float32),
                np.zeros(10, dtype=np.float32),
                rate=0.0,
            )


class TestEstimateAudioOffsets:
    def test_offsets_from_wav_files(
        self,
        tmp_path: Path,
        rng: np.random.Generator,
        wav_writer: Callable[..., None],
    ) -> None:
        sample_rate = 8000
        duration = 4 * sample_rate
        base = make_impulse_train(
            duration, [sample_rate, 2 * sample_rate, 3 * sample_rate], rng
        ) * 0.5

        # cam1 recording started 0.5s earlier: its events occur 0.5s later
        # in local time -> expected offset_sec = +0.5 (local = global + offset).
        delay_samples = sample_rate // 2
        delayed = np.concatenate(
            [np.zeros(delay_samples, dtype=np.float32), base]
        )

        ref_path = tmp_path / "cam0.wav"
        delayed_path = tmp_path / "cam1.wav"
        wav_writer(ref_path, base, sample_rate)
        wav_writer(delayed_path, delayed, sample_rate)

        result = estimate_audio_offsets(
            [ref_path, delayed_path],
            reference_index=0,
            sample_rate=sample_rate,
            envelope_rate=100.0,
        )
        assert result.offsets_sec[0] == 0.0
        assert result.offsets_sec[1] == pytest.approx(0.5, abs=0.02)
        assert result.confidences[0] == 1.0
        assert result.confidences[1] > 0.5

    def test_reference_offset_is_added(
        self,
        tmp_path: Path,
        rng: np.random.Generator,
        wav_writer: Callable[..., None],
    ) -> None:
        sample_rate = 8000
        base = make_impulse_train(2 * sample_rate, [sample_rate], rng) * 0.5
        path = tmp_path / "cam.wav"
        wav_writer(path, base, sample_rate)
        result = estimate_audio_offsets(
            [path, path],
            reference_index=0,
            reference_offset_sec=1.25,
            sample_rate=sample_rate,
        )
        assert result.offsets_sec[0] == 1.25
        assert result.offsets_sec[1] == pytest.approx(1.25, abs=0.02)

    def test_invalid_reference_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="reference_index"):
            estimate_audio_offsets([tmp_path / "a.wav"], reference_index=1)

    def test_empty_paths_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            estimate_audio_offsets([])
