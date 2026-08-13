"""Resolved qualitative render adapters for court model outputs."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from src.tasks.court_detection.model_io.adapters import (
    CourtKeypointModelIO,
    CourtLineModelIO,
    CourtModelIOAdapter,
    CourtSegmentationModelIO,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtModelIOError,
)
from src.tasks.court_detection.visualization.io.frames import (
    CourtFrame,
    KpFramePrediction,
)
from src.tasks.court_detection.visualization.rendering import (
    CourtRenderStyle,
    render_kp_frames,
    render_line_frames,
    render_seg_frames,
)
from src.tasks.court_detection.visualization.rendering.common import (
    denormalize_tensor_to_rgb,
)


class CourtQualitativeRenderer(Protocol):
    """Render one batch sample under a once-selected court task contract."""

    def render(
        self,
        *,
        batch: dict[str, Any],
        logits: torch.Tensor,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        """Decode and render one sample."""
        ...


class _KeypointQualitativeRenderer:
    def __init__(self, adapter: CourtKeypointModelIO) -> None:
        self.adapter = adapter

    def render(
        self,
        *,
        batch: dict[str, Any],
        logits: torch.Tensor,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        frame = batch_to_court_frame(batch)
        prediction = self.adapter.decode_prediction(
            logits[:1],
            original_size_hw=frame.rgb.shape[:2],
            subpixel_refine=False,
        )
        rendered_prediction = KpFramePrediction(
            keypoints_px=prediction.keypoints[prediction.valid].numpy(),
            mean_heatmap=torch.sigmoid(prediction.heatmaps).amax(0).numpy(),
        )
        rendered: list[np.ndarray] = render_kp_frames(
            frames=[frame],
            predictions=[rendered_prediction],
            style=style,
            clip_label=clip_label,
        )
        return rendered


class _SegmentationQualitativeRenderer:
    def __init__(self, adapter: CourtSegmentationModelIO) -> None:
        self.adapter = adapter

    def render(
        self,
        *,
        batch: dict[str, Any],
        logits: torch.Tensor,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        frame = batch_to_court_frame(batch)
        prediction = self.adapter.decode_prediction(
            logits[:1],
            original_size_hw=frame.rgb.shape[:2],
            subpixel_refine=False,
        )
        rendered: list[np.ndarray] = render_seg_frames(
            frames=[frame],
            masks=[prediction.mask.numpy().astype(np.int32)],
            style=style,
            clip_label=clip_label,
        )
        return rendered


class _LineQualitativeRenderer:
    def __init__(self, adapter: CourtLineModelIO) -> None:
        self.adapter = adapter

    def render(
        self,
        *,
        batch: dict[str, Any],
        logits: torch.Tensor,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        frame = batch_to_court_frame(batch)
        prediction = self.adapter.decode_prediction(
            logits[:1],
            original_size_hw=frame.rgb.shape[:2],
            subpixel_refine=False,
        )
        rendered: list[np.ndarray] = render_line_frames(
            frames=[frame],
            probs=[prediction.probability.numpy()],
            style=style,
            clip_label=clip_label,
        )
        return rendered


def build_court_qualitative_renderer(
    adapter: CourtModelIOAdapter,
) -> CourtQualitativeRenderer:
    """Select the matching qualitative decode/render adapter once."""
    if isinstance(adapter, CourtKeypointModelIO):
        return _KeypointQualitativeRenderer(adapter)
    if isinstance(adapter, CourtSegmentationModelIO):
        return _SegmentationQualitativeRenderer(adapter)
    if isinstance(adapter, CourtLineModelIO):
        return _LineQualitativeRenderer(adapter)
    raise CourtModelIOError(
        f"No qualitative renderer for adapter {type(adapter).__name__}."
    )


def batch_to_court_frame(
    batch: dict[str, Any],
    *,
    sample_idx: int = 0,
) -> CourtFrame:
    """Extract one validated normalized image and convert it to RGB."""
    if "image" not in batch:
        raise CourtModelIOError("Qualitative batch is missing required field 'image'.")
    image = batch["image"]
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise CourtModelIOError("Qualitative batch image must be a rank-4 Tensor.")
    if sample_idx < 0 or sample_idx >= image.shape[0]:
        raise CourtModelIOError("Qualitative sample_idx is outside the batch.")
    rgb = denormalize_tensor_to_rgb(image[sample_idx])
    return CourtFrame(name=f"sample_{sample_idx:02d}", rgb=rgb)


__all__ = [
    "CourtQualitativeRenderer",
    "batch_to_court_frame",
    "build_court_qualitative_renderer",
]
