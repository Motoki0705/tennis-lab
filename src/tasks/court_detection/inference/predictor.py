"""Typed inference predictor for Court keypoint heads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, TypeAlias, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import BoundModelIO
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.inference.checkpoint import load_court_pair
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedOutput,
    CourtKeypointPrediction,
    CourtLogits,
    CourtModelIOError,
    CourtModelOutput,
)
from src.tasks.court_detection.model_io.images import prepare_court_input
from src.tasks.court_detection.model_io.keypoint_decoder import (
    CourtKeypointDecoderConfig,
    decode_court_keypoint_logits,
)
from src.utils.configuration import PathResolver
from src.utils.device import resolve_device

CourtBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object],
    CourtLogits | CourtModelOutput,
    CourtLogits | CourtDecodedOutput,
]


class CourtKeypointPredictor(BasePredictor[CourtKeypointPrediction]):
    """Predict one KP head from a single- or multi-target checkpoint.

    The default peak contract is one candidate per semantic channel, matching
    the ordered single-court KP14 supervision. Multi-court callers must opt in
    to additional candidates with an explicit ``max_peaks``.
    """

    def __init__(
        self,
        model_io: CourtBoundModelIO,
        device: torch.device,
        *,
        subpixel_refine: bool,
        peak_threshold: float = 0.05,
        nms_kernel: int = 7,
        max_peaks: int = 1,
    ) -> None:
        if not isinstance(model_io.adapter, CourtModelIOAdapter):
            raise CourtModelIOError(
                "CourtKeypointPredictor requires CourtModelIOAdapter."
            )
        if "kp" not in model_io.adapter.spec.target_bundle.targets:
            raise CourtModelIOError(
                "CourtKeypointPredictor requires a checkpoint with a KP head."
            )
        self.model_io = model_io
        self.model = model_io.model
        self.adapter = model_io.adapter
        self.device = device
        self.subpixel_refine = subpixel_refine
        self._decoder_config = CourtKeypointDecoderConfig(
            threshold=peak_threshold,
            nms_kernel=nms_kernel,
            max_peaks=max_peaks,
        )

        self.adapter.validate_model_pair(self.model)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        subpixel_refine: bool,
        peak_threshold: float = 0.05,
        nms_kernel: int = 7,
        max_peaks: int = 1,
        **kwargs: Any,
    ) -> Self:
        """Load one checkpoint and preserve its serialized target bundle."""
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError("Court inference requires exactly one checkpoint")
        pair = load_court_pair(checkpoints[0], resolver=resolver, **kwargs)
        resolved_device = resolve_device(device)
        return cls(
            pair,
            resolved_device,
            subpixel_refine=subpixel_refine,
            peak_threshold=peak_threshold,
            nms_kernel=nms_kernel,
            max_peaks=max_peaks,
        )

    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
    ) -> CourtKeypointPrediction:
        """Return multi-peak KP channels, scores, validity, and heatmaps."""
        source_from_model_xy = (1.0, 1.0)
        if isinstance(image, Tensor):
            if image.ndim not in {3, 4}:
                raise CourtModelIOError(
                    "Court predictor tensors must have shape "
                    "(C,H,W) or (1,C,H,W)."
                )
            original_size_hw = (image.shape[-2], image.shape[-1])
            images = image.unsqueeze(0) if image.ndim == 3 else image
            if images.shape[0] != 1:
                raise CourtModelIOError(
                    "Court predictors accept exactly one image."
                )
            images = images.to(self.device)
        else:
            prepared = prepare_court_input(image, spec=self.adapter.spec, device=self.device)
            images = prepared.images
            original_size_hw = prepared.original_size_hw
            source_from_model_xy = prepared.source_from_model_xy

        with torch.no_grad():
            call = self.adapter.prepare_images(images)
            output = cast(CourtLogits | CourtModelOutput, self.model(*call.model_args))
            self.adapter.validate_logits(output, call)
            logits = (
                output.dense_logits if isinstance(output, CourtModelOutput) else output
            )
        decoded = decode_court_keypoint_logits(
            logits["kp"],
            original_size_hw=(images.shape[-2], images.shape[-1]),
            subpixel_refine=self.subpixel_refine,
            config=self.decoder_config,
        )

        points = decoded.keypoints * decoded.keypoints.new_tensor(source_from_model_xy)
        height, width = original_size_hw
        inside = (points[..., 0] >= 0) & (points[..., 0] < width) & (points[..., 1] >= 0) & (points[..., 1] < height)
        valid = decoded.valid & inside
        return CourtKeypointPrediction(points, decoded.scores * valid, valid, decoded.heatmaps)

    @property
    def task(self) -> CourtTargetKind:
        return "kp"

    @property
    def decoder_config(self) -> CourtKeypointDecoderConfig:
        """Validated peak-extraction contract used by :meth:`predict`."""
        return self._decoder_config

    @property
    def peak_threshold(self) -> float:
        """Extraction threshold applied to each channel's peak scores."""
        return float(self.decoder_config.threshold)

    @property
    def nms_kernel(self) -> int:
        """Odd max-pooling kernel used for peak non-maximum suppression."""
        return int(self.decoder_config.nms_kernel)

    @property
    def max_peaks(self) -> int:
        """Candidate budget per semantic channel (1 for the singleton contract)."""
        return int(self.decoder_config.max_peaks)

    @property
    def short_side(self) -> int:
        return int(self.adapter.spec.short_side)


__all__ = ["CourtKeypointPredictor"]
