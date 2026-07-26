"""Verified court-line inference adapter for alignment entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.ground_line_map import (
    GroundLineMapSettings,
    ProjectedLinePixels,
    project_line_pixels_to_ground,
)
from src.synthetic_data_generation.alignment.ground_plane import GroundPlaneEstimate
from src.synthetic_data_generation.provider.bundle import sha256_file
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.court_detection.inference import CourtLinePredictor


@dataclass(frozen=True)
class VerifiedLineDetector:
    """Loaded predictor plus immutable checkpoint/backbone identity."""

    predictor: CourtLinePredictor
    checkpoint_sha256: str
    backbone_checkpoint_sha256: str
    embedded_backbone_path: str


@dataclass(frozen=True)
class LineProjectionObservation:
    """One line prediction transformed into original pixels and ground points."""

    projection: ProjectedLinePixels
    output_width: int
    output_height: int
    selected_line_pixel_count: int


def load_verified_line_detector(
    checkpoint: Path,
    *,
    checkpoint_sha256: str,
    backbone_repository: Path,
    backbone_checkpoint: Path,
    backbone_checkpoint_sha256: str,
    device: str,
    expected_short_side: int,
) -> VerifiedLineDetector:
    """Verify artifacts and load the line model with an explicit local backbone."""
    _require_sha256(checkpoint, expected=checkpoint_sha256, name="line checkpoint")
    _require_sha256(
        backbone_checkpoint,
        expected=backbone_checkpoint_sha256,
        name="DINOv3 backbone",
    )
    if not backbone_repository.is_dir():
        raise FileNotFoundError(
            f"DINOv3 repository directory not found: {backbone_repository}"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested but CUDA is unavailable."
        )
    raw: Any = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Line checkpoint payload must be a mapping.")
    hyper_parameters = raw.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise ValueError("Line checkpoint has no hyper_parameters mapping.")
    embedded = hyper_parameters.get("config")
    if not isinstance(embedded, (dict, DictConfig)):
        raise ValueError("Line checkpoint has no embedded config mapping.")
    config = OmegaConf.create(embedded)
    embedded_path = str(config.model.encoder.checkpoint_path)
    if Path(embedded_path).name != backbone_checkpoint.name:
        raise ValueError(
            "Configured DINOv3 backbone file does not match checkpoint config: "
            f"{backbone_checkpoint.name!r} vs {Path(embedded_path).name!r}."
        )
    config.model.encoder.repository_path = str(backbone_repository)
    config.model.encoder.checkpoint_path = str(backbone_checkpoint)
    predictor = CourtLinePredictor.load_from_checkpoint(
        checkpoint,
        device=device,
        weights_only=False,
        config=config,
    )
    if predictor.short_side != expected_short_side:
        raise ValueError(
            "Line-checkpoint preprocessing mismatch: "
            f"expected {expected_short_side}, loaded {predictor.short_side}."
        )
    if device.startswith("cuda") and predictor.device.type != "cuda":
        raise RuntimeError(
            f"Predictor silently resolved requested {device!r} to {predictor.device}."
        )
    return VerifiedLineDetector(
        predictor=predictor,
        checkpoint_sha256=checkpoint_sha256,
        backbone_checkpoint_sha256=backbone_checkpoint_sha256,
        embedded_backbone_path=embedded_path,
    )


def infer_line_projection(
    image_rgb: NDArray[np.uint8],
    camera: SceneCamera,
    *,
    detector: VerifiedLineDetector,
    plane: GroundPlaneEstimate,
    bounds: tuple[float, float, float, float],
    settings: GroundLineMapSettings,
) -> LineProjectionObservation:
    """Infer line probabilities and project exact original-image pixel centres."""
    if image_rgb.shape != (camera.height, camera.width, 3):
        raise ValueError(
            f"Provider image shape mismatch for {camera.camera_id}: {image_rgb.shape}."
        )
    probability = detector.predictor.predict(image_rgb)["line_prob"].numpy()
    pixels_xy, selected_probability = line_pixels_in_original_image(
        probability,
        original_width=camera.width,
        original_height=camera.height,
        probability_threshold=settings.probability_threshold,
    )
    projection = project_line_pixels_to_ground(
        camera,
        pixels_xy,
        selected_probability,
        plane=plane,
        bounds=bounds,
        settings=settings,
    )
    output_height, output_width = probability.shape
    return LineProjectionObservation(
        projection=projection,
        output_width=output_width,
        output_height=output_height,
        selected_line_pixel_count=len(pixels_xy),
    )


def line_pixels_in_original_image(
    probability: NDArray[np.floating[Any]],
    *,
    original_width: int,
    original_height: int,
    probability_threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Map selected output-grid pixel centres to the provider image coordinates."""
    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("Line probability must be a 2D grid of at least 2x2.")
    if not np.isfinite(array).all():
        raise ValueError("Line probability must contain only finite values.")
    if original_width < 2 or original_height < 2:
        raise ValueError("Original image dimensions must both be at least two.")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must lie in [0, 1].")
    selected_y, selected_x = np.nonzero(array >= probability_threshold)
    output_height, output_width = array.shape
    pixels_xy = np.column_stack(
        (
            selected_x.astype(np.float64) * (original_width - 1) / (output_width - 1),
            selected_y.astype(np.float64) * (original_height - 1) / (output_height - 1),
        )
    )
    return pixels_xy, array[selected_y, selected_x]


def _require_sha256(path: Path, *, expected: str, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    actual = str(sha256_file(path))
    if actual != expected:
        raise ValueError(
            f"{name} SHA-256 mismatch: declared {expected}, computed {actual}."
        )
