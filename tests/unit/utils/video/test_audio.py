"""Tests for src/utils/video/audio.py."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.utils.video.audio import audio_envelope, read_audio_mono
from src.utils.video.writer import save_video_rgb


class TestReadAudioMono:
    def test_round_trip_sine(
        self, tmp_path: Path, wav_writer: Callable[..., None]
    ) -> None:
        sample_rate = 8000
        t = np.arange(sample_rate * 2) / sample_rate
        sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav = tmp_path / "sine.wav"
        wav_writer(wav, sine, sample_rate)

        decoded = read_audio_mono(wav, sample_rate=sample_rate)
        assert decoded.dtype == np.float32
        assert abs(decoded.size - sine.size) < sample_rate * 0.01
        overlap = min(decoded.size, sine.size)
        rms_error = float(np.sqrt(np.mean((decoded[:overlap] - sine[:overlap]) ** 2)))
        assert rms_error < 0.01

    def test_max_seconds_caps_length(
        self, tmp_path: Path, wav_writer: Callable[..., None]
    ) -> None:
        sample_rate = 8000
        wav = tmp_path / "long.wav"
        wav_writer(wav, np.ones(sample_rate * 3, dtype=np.float32) * 0.1, sample_rate)
        decoded = read_audio_mono(wav, sample_rate=sample_rate, max_seconds=1.0)
        assert decoded.size == sample_rate

    def test_resampling_changes_rate(
        self, tmp_path: Path, wav_writer: Callable[..., None]
    ) -> None:
        wav = tmp_path / "rate.wav"
        wav_writer(wav, np.ones(16000, dtype=np.float32) * 0.1, 16000)
        decoded = read_audio_mono(wav, sample_rate=4000)
        assert abs(decoded.size - 4000) < 100

    def test_no_audio_stream_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "silent.mp4"
        save_video_rgb(np.zeros((4, 32, 48, 3), dtype=np.uint8), video, fps=10.0)
        with pytest.raises(ValueError, match="No audio stream"):
            read_audio_mono(video)

    def test_invalid_args_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            read_audio_mono(tmp_path / "x.wav", sample_rate=0)
        with pytest.raises(ValueError, match="max_seconds"):
            read_audio_mono(tmp_path / "x.wav", max_seconds=0)


class TestAudioEnvelope:
    def test_constant_amplitude_sine(self) -> None:
        sample_rate = 8000
        t = np.arange(sample_rate) / sample_rate
        sine = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        envelope = audio_envelope(sine, sample_rate=sample_rate, envelope_rate=100.0)
        assert envelope.shape == (100,)
        assert np.allclose(envelope, 0.5 / np.sqrt(2), atol=0.01)

    def test_burst_position(self) -> None:
        sample_rate = 1000
        samples: np.ndarray = np.zeros(2000, dtype=np.float32)
        samples[1500:1550] = 1.0  # burst at 1.5s
        envelope = audio_envelope(samples, sample_rate=sample_rate, envelope_rate=100.0)
        assert int(np.argmax(envelope)) == 150

    def test_rate_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="integer multiple"):
            audio_envelope(
                np.zeros(1000, dtype=np.float32), sample_rate=1000, envelope_rate=333.0
            )

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            audio_envelope(
                np.zeros(5, dtype=np.float32), sample_rate=1000, envelope_rate=100.0
            )

    def test_non_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            audio_envelope(
                np.zeros((2, 100), dtype=np.float32),
                sample_rate=1000,
                envelope_rate=100.0,
            )
