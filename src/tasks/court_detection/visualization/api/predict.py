"""Once-selected prediction and rendering pipelines for court visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
import torch

from src.tasks.court_detection.inference import (
    CourtKeypointPredictor,
    CourtLinePredictor,
    CourtSegPredictor,
)
from src.tasks.court_detection.model_io.contracts import CourtModelIOError
from src.tasks.court_detection.visualization.adapters.predict_inputs import (
    to_predictor_input,
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
from src.utils.configuration import PathResolver

CourtTargetHead: TypeAlias = Literal["kp", "seg", "line"]

logger = logging.getLogger(__name__)


class CourtVisualizationPipeline(Protocol):
    """Predict and render frames under one checkpoint-selected target head."""

    def render(
        self,
        frames: list[CourtFrame],
        *,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        """Predict and render every frame without per-frame head dispatch."""
        ...


class _KeypointVisualizationPipeline:
    def __init__(self, predictor: CourtKeypointPredictor) -> None:
        self.predictor = predictor

    def render(
        self,
        frames: list[CourtFrame],
        *,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        predictions: list[KpFramePrediction] = []
        for index, frame in enumerate(frames):
            output = self.predictor.predict(to_predictor_input(frame))
            predictions.append(
                KpFramePrediction(
                    keypoints_px=output.keypoints[output.valid].numpy(),
                    mean_heatmap=torch.sigmoid(output.heatmaps).amax(0).numpy(),
                )
            )
            _log_progress(index, len(frames))
        rendered: list[np.ndarray] = render_kp_frames(
            frames=frames,
            predictions=predictions,
            style=style,
            clip_label=clip_label,
        )
        return rendered


class _SegmentationVisualizationPipeline:
    def __init__(self, predictor: CourtSegPredictor) -> None:
        self.predictor = predictor

    def render(
        self,
        frames: list[CourtFrame],
        *,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            output = self.predictor.predict(to_predictor_input(frame))
            masks.append(output.mask.numpy().astype(np.uint8))
            _log_progress(index, len(frames))
        rendered: list[np.ndarray] = render_seg_frames(
            frames=frames,
            masks=masks,
            style=style,
            clip_label=clip_label,
        )
        return rendered


class _LineVisualizationPipeline:
    def __init__(self, predictor: CourtLinePredictor) -> None:
        self.predictor = predictor

    def render(
        self,
        frames: list[CourtFrame],
        *,
        style: CourtRenderStyle,
        clip_label: str,
    ) -> list[np.ndarray]:
        probabilities: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            output = self.predictor.predict(to_predictor_input(frame))
            probabilities.append(output.probability.numpy())
            _log_progress(index, len(frames))
        rendered: list[np.ndarray] = render_line_frames(
            frames=frames,
            probs=probabilities,
            style=style,
            clip_label=clip_label,
        )
        return rendered


def build_court_visualization_pipeline(
    task: CourtTargetHead,
    *,
    checkpoint_path: str | Path,
    device: str,
    resolver: PathResolver,
) -> CourtVisualizationPipeline:
    """Select and load the exact target-head pipeline before the frame loop."""
    if task == "kp":
        return _KeypointVisualizationPipeline(
            CourtKeypointPredictor.load_from_checkpoint(
                checkpoint_path,
                device=device,
                resolver=resolver,
                subpixel_refine=True,
            )
        )
    if task == "seg":
        return _SegmentationVisualizationPipeline(
            CourtSegPredictor.load_from_checkpoint(
                checkpoint_path,
                device=device,
                resolver=resolver,
            )
        )
    if task == "line":
        return _LineVisualizationPipeline(
            CourtLinePredictor.load_from_checkpoint(
                checkpoint_path,
                device=device,
                resolver=resolver,
            )
        )
    raise CourtModelIOError(f"Unsupported Court visualization head {task!r}.")


def _log_progress(index: int, total: int) -> None:
    if (index + 1) % 10 == 0 or index == 0 or index == total - 1:
        logger.info("  [Inference] Processing frame %d/%d...", index + 1, total)


__all__ = ["CourtVisualizationPipeline", "build_court_visualization_pipeline"]
