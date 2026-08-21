"""Qualitative rendering for one selected head from a Court target bundle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtModelIOError,
    CourtSegmentationPrediction,
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
    def render(
        self,
        *,
        batch: dict[str, Any],
        logits: Tensor,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]: ...


class _BundleHeadQualitativeRenderer:
    def __init__(
        self,
        adapter: CourtModelIOAdapter,
        *,
        kind: CourtTargetKind,
    ) -> None:
        if kind not in adapter.spec.target_bundle.targets:
            raise CourtModelIOError(
                f"Qualitative target {kind!r} is not in the Court target bundle."
            )
        self.adapter = adapter
        self.kind = kind

    def render(
        self,
        *,
        batch: dict[str, Any],
        logits: Tensor,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        frame = batch_to_court_frame(batch)
        height, width = frame.rgb.shape[:2]
        cropped_logits = logits[:1, :, :height, :width]
        max_peaks = self._max_peaks(batch)
        prediction = self.adapter.decode_prediction(
            self.kind,
            cropped_logits,
            original_size_hw=(height, width),
            subpixel_refine=False,
            max_peaks=max_peaks,
        )
        if self.kind == "kp":
            if not isinstance(prediction, CourtKeypointPrediction):
                raise CourtModelIOError("Court KP qualitative decode type changed.")
            rendered_prediction = KpFramePrediction(
                keypoints_px=prediction.keypoints[prediction.valid].numpy(),
                mean_heatmap=torch.sigmoid(prediction.heatmaps).amax(0).numpy(),
            )
            rendered_kp: list[np.ndarray] = render_kp_frames(
                frames=[frame],
                predictions=[rendered_prediction],
                style=style,
                clip_label=clip_label,
            )
            return rendered_kp
        if self.kind == "seg":
            if not isinstance(prediction, CourtSegmentationPrediction):
                raise CourtModelIOError(
                    "Court segmentation qualitative decode type changed."
                )
            rendered_seg: list[np.ndarray] = render_seg_frames(
                frames=[frame],
                masks=[prediction.mask.numpy().astype(np.int32)],
                style=style,
                clip_label=clip_label,
            )
            return rendered_seg
        if not isinstance(prediction, CourtLinePrediction):
            raise CourtModelIOError("Court line qualitative decode type changed.")
        rendered_line: list[np.ndarray] = render_line_frames(
            frames=[frame],
            probs=[prediction.probability.numpy()],
            style=style,
            clip_label=clip_label,
        )
        return rendered_line

    def _max_peaks(self, batch: Mapping[str, object]) -> int:
        if self.kind != "kp":
            return 1
        targets = batch.get("targets")
        if not isinstance(targets, Mapping):
            return 1
        payload = targets.get("kp")
        if not isinstance(payload, Mapping):
            return 1
        visible = payload.get("point_visible")
        if not isinstance(visible, Tensor) or visible.ndim != 3:
            return 1
        return int(visible.shape[2])


def build_court_qualitative_renderer(
    adapter: CourtModelIOAdapter,
    *,
    kind: CourtTargetKind,
) -> CourtQualitativeRenderer:
    """Bind qualitative rendering to one explicit head at composition time."""
    return _BundleHeadQualitativeRenderer(adapter, kind=kind)


def batch_to_court_frame(
    batch: dict[str, Any],
    *,
    sample_idx: int = 0,
) -> CourtFrame:
    """Extract one unpadded normalized image and convert it to RGB."""
    image = batch.get("image")
    if not isinstance(image, Tensor) or image.ndim != 4:
        raise CourtModelIOError("Qualitative batch image must be a rank-4 Tensor.")
    if sample_idx < 0 or sample_idx >= image.shape[0]:
        raise CourtModelIOError("Qualitative sample_idx is outside the batch.")
    image_size = batch.get("image_size")
    if not isinstance(image_size, Tensor) or image_size.shape != (image.shape[0], 2):
        raise CourtModelIOError(
            "Qualitative batch image_size must have shape (B,2)."
        )
    height, width = (int(value) for value in image_size[sample_idx].tolist())
    rgb = denormalize_tensor_to_rgb(image[sample_idx, :, :height, :width])
    raw_ids = batch.get("sample_id")
    name = (
        cast(list[str], raw_ids)[sample_idx]
        if isinstance(raw_ids, list) and sample_idx < len(raw_ids)
        else f"sample_{sample_idx:02d}"
    )
    return CourtFrame(name=name, rgb=rgb)


__all__ = [
    "CourtQualitativeRenderer",
    "batch_to_court_frame",
    "build_court_qualitative_renderer",
]
