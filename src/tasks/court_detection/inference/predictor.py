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
from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.inference.checkpoint_compat import (
    load_court_inference_lightning_module,
)
from src.tasks.court_detection.inference.keypoint_decoder import (
    CourtKeypointDecoderConfig,
    decode_court_keypoint_logits,
)
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLogits,
    CourtModelIOError,
    CourtModelOutput,
)
from src.tasks.court_detection.model_io.images import prepare_court_image
from src.utils.configuration import PathResolver
from src.utils.device import resolve_device

CourtBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object],
    CourtLogits,
    CourtLogits,
]


class CourtKeypointPredictor(BasePredictor[CourtKeypointPrediction]):
    """Predict the KP head from a dense-only or pose-enabled checkpoint.

    Ordered single-court inference is the default: one candidate per semantic
    channel at probability 0.5 or higher. Multi-court users must explicitly
    opt in to a larger ``max_peaks`` value and, when appropriate, a different
    threshold.
    """

    def __init__(
        self,
        model_io: CourtBoundModelIO,
        device: torch.device,
        *,
        subpixel_refine: bool,
        peak_threshold: float = 0.5,
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
        self.decoder_config = CourtKeypointDecoderConfig(
            threshold=peak_threshold,
            nms_kernel=nms_kernel,
            max_peaks=max_peaks,
        )
        self.peak_threshold = self.decoder_config.threshold
        self.nms_kernel = self.decoder_config.nms_kernel
        self.max_peaks = self.decoder_config.max_peaks

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
        peak_threshold: float = 0.5,
        nms_kernel: int = 7,
        max_peaks: int = 1,
        **kwargs: Any,
    ) -> Self:
        """Load one checkpoint and preserve its serialized target bundle.

        The loader supports current checkpoints and the deployed residual-head
        pose checkpoint while keeping the training configuration parser strict.
        Only ``config``, ``strict``, and ``weights_only=False`` are accepted as
        checkpoint-loading options.
        """
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )

        config_override = kwargs.pop("config", None)
        strict = kwargs.pop("strict", True)
        weights_only = kwargs.pop("weights_only", False)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Court checkpoint options: {unknown}.")
        if not isinstance(strict, bool):
            raise TypeError("Court checkpoint strict option must be boolean.")
        if weights_only is not False:
            raise ValueError(
                "Court inference requires weights_only=False so serialized "
                "configuration and target contracts can be restored."
            )

        resolved_device = resolve_device(device)
        lightning_module = load_court_inference_lightning_module(
            checkpoints[0],
            config_override=config_override,
            runtime_path_roots=resolver.roots.as_mapping(),
            strict=strict,
        )
        adapter = lightning_module.model_io
        adapter.validate_model_pair(lightning_module.model)
        return cls(
            cast(
                CourtBoundModelIO,
                bind_model_io(lightning_module.model, adapter),
            ),
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
        """Return configured KP candidates, scores, validity, and heatmaps."""
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
            images, original_height, original_width = prepare_court_image(
                image,
                short_side=self.adapter.spec.short_side,
                device=self.device,
            )
            original_size_hw = (original_height, original_width)

        with torch.no_grad():
            call = self.adapter.prepare_images(images)
            output = cast(CourtLogits | CourtModelOutput, self.model(*call.model_args))
            self.adapter.validate_logits(output, call)
            logits = (
                output.dense_logits if isinstance(output, CourtModelOutput) else output
            )
        return decode_court_keypoint_logits(
            logits["kp"],
            original_size_hw=original_size_hw,
            subpixel_refine=self.subpixel_refine,
            config=self.decoder_config,
        )

    @property
    def task(self) -> CourtTargetKind:
        return "kp"

    @property
    def short_side(self) -> int:
        return int(self.adapter.spec.short_side)


__all__ = ["CourtKeypointPredictor"]
