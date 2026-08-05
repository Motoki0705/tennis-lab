"""Court-line inference adapter for alignment entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.components.evidence.ground_line_raster import (
    GroundLineMapSettings,
)
from src.synthetic_data_generation.alignment.components.ground.plane import (
    GroundPlaneEstimate,
)
from src.synthetic_data_generation.alignment.components.ground.projection import (
    ProjectedLinePixels,
    project_line_pixels_to_ground,
)
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.court_detection.inference import CourtLinePredictor
from src.utils.configuration import PathResolver


@dataclass(frozen=True)
class LineDetector:
    """Loaded predictor plus its configured backbone path."""

    predictor: CourtLinePredictor
    embedded_backbone_path: str


@dataclass(frozen=True)
class LineProjectionObservation:
    """One line prediction transformed into original pixels and ground points."""

    projection: ProjectedLinePixels
    output_width: int
    output_height: int
    selected_line_pixel_count: int


def load_line_detector(
    checkpoint: Path,
    *,
    backbone_repository: Path,
    backbone_checkpoint: Path,
    device: str,
    expected_short_side: int,
    resolver: PathResolver,
) -> LineDetector:
    """Load the configured line model and explicit local backbone paths."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Line checkpoint does not exist: {checkpoint}")
    if not backbone_checkpoint.is_file():
        raise FileNotFoundError(
            f"DINOv3 backbone does not exist: {backbone_checkpoint}"
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
    if "hyper_parameters" not in raw:
        raise ValueError("Line checkpoint has no hyper_parameters mapping.")
    hyper_parameters = raw["hyper_parameters"]
    if not isinstance(hyper_parameters, dict):
        raise ValueError("Line checkpoint has no hyper_parameters mapping.")
    if "config" not in hyper_parameters:
        raise ValueError("Line checkpoint has no embedded config mapping.")
    embedded = hyper_parameters["config"]
    if not isinstance(embedded, Mapping):
        raise ValueError("Line checkpoint has no embedded config mapping.")
    config = _plain_mapping(embedded)
    model = _required_mapping(config, "model")
    encoder = _required_mapping(model, "encoder")
    if "checkpoint_path" not in encoder:
        raise ValueError("Line checkpoint config has no encoder checkpoint_path.")
    embedded_value = encoder["checkpoint_path"]
    if not isinstance(embedded_value, str) or not embedded_value:
        raise ValueError("Line checkpoint encoder checkpoint_path must be a string.")
    embedded_path = embedded_value
    if Path(embedded_path).name != backbone_checkpoint.name:
        raise ValueError(
            "Configured DINOv3 backbone file does not match checkpoint config: "
            f"{backbone_checkpoint.name!r} vs {Path(embedded_path).name!r}."
        )
    encoder["repository_path"] = str(backbone_repository)
    encoder["checkpoint_path"] = str(backbone_checkpoint)
    predictor = CourtLinePredictor.load_from_checkpoint(
        checkpoint,
        resolver=resolver,
        device=device,
        allow_device_fallback=False,
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
    return LineDetector(
        predictor=predictor,
        embedded_backbone_path=embedded_path,
    )


def infer_line_projection(
    image_rgb: NDArray[np.uint8],
    camera: SceneCamera,
    *,
    detector: LineDetector,
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


def _plain_mapping(value: Mapping[object, object]) -> dict[str, Any]:
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_value(item) for item in value]
    return value


def _required_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key not in value:
        raise ValueError(f"Line checkpoint config has no {key} mapping.")
    nested = value[key]
    if not isinstance(nested, dict):
        raise ValueError(f"Line checkpoint config has no {key} mapping.")
    return nested
