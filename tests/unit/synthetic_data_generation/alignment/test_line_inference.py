"""Tests for verified line-model loading and output-pixel coordinates."""

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from src.synthetic_data_generation.alignment import line_inference


def test_load_detector_accepts_checkpoint_dictconfig(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    embedded = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "repository_path": "/content/dinov3",
                    "checkpoint_path": "backbone.pth",
                }
            }
        }
    )
    checkpoint = tmp_path / "line.ckpt"
    backbone_repository = tmp_path / "dinov3"
    backbone = backbone_repository / "backbone.pth"
    checkpoint.write_bytes(b"line")
    backbone_repository.mkdir()
    backbone.write_bytes(b"backbone")
    monkeypatch.setattr(
        line_inference.torch,
        "load",
        lambda *_args, **_kwargs: {
            "hyper_parameters": {"config": embedded},
        },
    )
    monkeypatch.setattr(
        line_inference,
        "sha256_file",
        lambda path: "line-hash" if path == checkpoint else "backbone-hash",
    )
    captured: dict[str, Any] = {}

    class FakePredictor:
        short_side = 256

        class Device:
            type = "cpu"

            def __str__(self) -> str:
                return "cpu"

        device = Device()

    def fake_load_from_checkpoint(
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> FakePredictor:
        captured["checkpoint"] = checkpoint_path
        captured.update(kwargs)
        return FakePredictor()

    monkeypatch.setattr(
        line_inference.CourtLinePredictor,
        "load_from_checkpoint",
        fake_load_from_checkpoint,
    )

    detector = line_inference.load_verified_line_detector(
        checkpoint,
        checkpoint_sha256="line-hash",
        backbone_repository=backbone_repository,
        backbone_checkpoint=backbone,
        backbone_checkpoint_sha256="backbone-hash",
        device="cpu",
        expected_short_side=256,
    )

    assert detector.embedded_backbone_path == "backbone.pth"
    config = captured["config"]
    assert config.model.encoder.repository_path == str(backbone_repository)
    assert config.model.encoder.checkpoint_path == str(backbone)


def test_line_pixels_map_output_corners_to_original_corners() -> None:
    probability = np.asarray(
        [
            [0.9, 0.0, 0.8],
            [0.0, 0.0, 0.0],
            [0.7, 0.0, 0.6],
        ],
        dtype=np.float32,
    )

    pixels, scores = line_inference.line_pixels_in_original_image(
        probability,
        original_width=959,
        original_height=539,
        probability_threshold=0.5,
    )

    np.testing.assert_allclose(
        pixels,
        np.asarray(
            [
                [0.0, 0.0],
                [958.0, 0.0],
                [0.0, 538.0],
                [958.0, 538.0],
            ]
        ),
    )
    np.testing.assert_allclose(scores, (0.9, 0.8, 0.7, 0.6))
