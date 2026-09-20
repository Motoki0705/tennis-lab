"""Alignment consumes the common raw LINE head and its loaded-model identity."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.alignment.evidence_source import (
    ProductionCourtLineDetector,
)
from src.synthetic_data_generation.alignment.line_inference_cache import (
    court_line_inference_identity,
)
from src.synthetic_data_generation.alignment.settings import CourtLineModelSettings
from src.tasks.court_detection.inference.predictor import CourtLinePredictor


def _settings(path: Path) -> CourtLineModelSettings:
    path.write_bytes(b"new-multi-head-checkpoint")
    return CourtLineModelSettings(
        checkpoint_path=path,
        device="cpu",
        probability_threshold=0.5,
        maximum_selected_pixels_per_camera=100,
    )


def _identity(settings: CourtLineModelSettings) -> dict[str, object]:
    return {
        "checkpoint_sha256": hashlib.sha256(
            settings.checkpoint_path.read_bytes()
        ).hexdigest(),
        "backbone_sha256": "backbone",
        "architecture": {"name": "residual"},
        "target_bundle": {"kp": 14, "seg": 7, "line": 1},
        "short_side": 256,
    }


@pytest.mark.parametrize("channels", [("court_line",), ("other_line",)])
def test_detector_loads_common_predictor_once_and_preserves_probability_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: tuple[str, ...]
) -> None:
    settings = _settings(tmp_path / "court.ckpt")
    resolver = object()
    probability = torch.linspace(0.0, 1.0, 24).reshape(4, 6)
    image: np.ndarray = np.zeros((16, 24, 3), dtype=np.uint8)
    load_calls: list[object] = []

    def predict(actual: np.ndarray) -> SimpleNamespace:
        assert actual is image
        return SimpleNamespace(probability=probability)

    predictor = SimpleNamespace(
        adapter=SimpleNamespace(
            spec=SimpleNamespace(
                target_bundle=SimpleNamespace(
                    targets={
                        "line": SimpleNamespace(
                            output_channels=1, channel_names=channels
                        ),
                    }
                )
            )
        ),
        model=SimpleNamespace(training=False),
        checkpoint_identity=_identity(settings),
        predict=predict,
    )

    def load(path: Path, **kwargs: object) -> Any:
        assert path == settings.checkpoint_path
        assert kwargs == {"resolver": resolver, "device": "cpu"}
        load_calls.append(path)
        return predictor

    monkeypatch.setattr(CourtLinePredictor, "load_from_checkpoint", load)
    # Production controls these global flags; keep this unit test isolated.
    monkeypatch.setattr(torch, "manual_seed", lambda *_: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *_, **__: None)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
    )
    for backend, name in (
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cuda.matmul, "allow_tf32"),
    ):
        monkeypatch.setattr(backend, name, getattr(backend, name))
    detector = ProductionCourtLineDetector(settings, cast(Any, resolver), seed=42)
    if channels != ("court_line",):
        with pytest.raises(ValueError, match="binary court_line"):
            detector.preflight()
        assert detector._predictor is None
        return
    first = detector.predict_probability(image)
    second = detector.predict_probability(image)
    np.testing.assert_array_equal(first, probability.numpy())
    np.testing.assert_array_equal(second, first)
    assert len(load_calls) == 1
    assert detector.inference_cache_identity()["target_bundle"] == {
        "kp": 14,
        "seg": 7,
        "line": 1,
    }


def test_cache_separates_model_and_preprocessing_without_fit_settings(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "court.ckpt")
    identity = _identity(settings)
    first = court_line_inference_identity(
        settings, seed=42, checkpoint_identity=identity
    )
    assert first["schema"] == "court_line_inference_identity_v3"
    changed_fit = replace(
        settings, probability_threshold=0.7, maximum_selected_pixels_per_camera=50
    )
    assert (
        court_line_inference_identity(
            changed_fit, seed=42, checkpoint_identity=identity
        )
        == first
    )
    for field, value in (
        ("short_side", 512),
        ("target_bundle", {"line": 1}),
        ("backbone_sha256", "other"),
    ):
        assert (
            court_line_inference_identity(
                settings, seed=42, checkpoint_identity={**identity, field: value}
            )
            != first
        )
    settings.checkpoint_path.write_bytes(b"different-checkpoint")
    with pytest.raises(ValueError, match="differs from the cache source"):
        court_line_inference_identity(settings, seed=42, checkpoint_identity=identity)
    assert (
        court_line_inference_identity(
            settings, seed=42, checkpoint_identity=_identity(settings)
        )
        != first
    )
