"""One checkpoint, one forward, raw Court heads and optional hybrid geometry."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.homography import (
    COURT_HOMOGRAPHY_EDGES,
    court_template_xy,
)
from src.tasks.court_detection.geometry.hybrid_homography import (
    DEFAULT_HYBRID_CONFIG,
    HybridHomographyConfig,
    estimate_hybrid_homography,
)
from src.tasks.court_detection.inference.checkpoint import load_court_checkpoint
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedPrediction,
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtLogits,
    CourtModelIOError,
    CourtModelOutput,
    CourtSegmentationPrediction,
)
from src.tasks.court_detection.model_io.factory import CourtDetectionBoundModelIO
from src.tasks.court_detection.model_io.images import prepare_court_image
from src.tasks.court_detection.model_io.keypoint_decoder import (
    CourtKeypointDecoderConfig,
    decode_court_keypoint_logits,
)
from src.utils.configuration import PathResolver
from src.utils.device import resolve_device
from src.utils.schema.court import GROUND_COURT_KP_NAMES

CourtImage: TypeAlias = np.ndarray | Image.Image | Tensor
CourtBoundModelIO: TypeAlias = CourtDetectionBoundModelIO


class CourtPredictor(BasePredictor[CourtPrediction]):
    """Shared execution path for multi-head and explicitly head-only consumers."""

    def __init__(
        self,
        model_io: CourtBoundModelIO,
        device: torch.device,
        *,
        subpixel_refine: bool = True,
        peak_threshold: float = 0.05,
        nms_kernel: int = 7,
        max_peaks: int = 1,
        hybrid_config: HybridHomographyConfig = DEFAULT_HYBRID_CONFIG,
        checkpoint_identity: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(model_io.adapter, CourtModelIOAdapter):
            raise CourtModelIOError("CourtPredictor requires CourtModelIOAdapter")
        self.model_io = model_io
        self.model = model_io.model
        self.adapter = model_io.adapter
        self.device = device
        self.subpixel_refine = subpixel_refine
        self._decoder_config = CourtKeypointDecoderConfig(
            threshold=peak_threshold, nms_kernel=nms_kernel, max_peaks=max_peaks
        )
        self.hybrid_config = hybrid_config
        self.checkpoint_identity = checkpoint_identity or {
            "schema": "provided_court_model_io"
        }
        self.adapter.validate_model_pair(self.model)
        self.model.to(device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        device: str | torch.device,
        resolver: PathResolver | None = None,
        subpixel_refine: bool = True,
        peak_threshold: float = 0.05,
        nms_kernel: int = 7,
        max_peaks: int = 1,
        hybrid_config: HybridHomographyConfig = DEFAULT_HYBRID_CONFIG,
    ) -> Self:
        if resolver is not None:
            paths = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        else:
            paths = (
                [Path(checkpoint_path)]
                if isinstance(checkpoint_path, (str, Path))
                else [Path(p) for p in checkpoint_path]
            )
            if any(not p.is_absolute() for p in paths):
                raise CourtModelIOError(
                    "Relative checkpoints require an explicit PathResolver"
                )
        if len(paths) != 1:
            raise CourtModelIOError("CourtPredictor requires exactly one checkpoint")
        loaded = load_court_checkpoint(paths[0], resolver=resolver)
        return cls(
            loaded.model_io,
            resolve_device(device),
            subpixel_refine=subpixel_refine,
            peak_threshold=peak_threshold,
            nms_kernel=nms_kernel,
            max_peaks=max_peaks,
            hybrid_config=hybrid_config,
            checkpoint_identity=loaded.identity,
        )

    def predict(
        self,
        image: CourtImage,
        *,
        postprocess: Literal["hybrid", "none"] = "hybrid",
        heads: Iterable[CourtTargetKind] | None = None,
    ) -> CourtPrediction:
        if postprocess not in {"hybrid", "none"}:
            raise CourtModelIOError("postprocess must be 'hybrid' or 'none'")
        requested = (
            self.adapter.spec.target_bundle.kinds if heads is None else tuple(heads)
        )
        if not requested or len(set(requested)) != len(requested):
            raise CourtModelIOError("Requested Court heads must be nonempty and unique")
        if set(requested) - set(self.adapter.spec.target_bundle.kinds):
            raise CourtModelIOError(
                f"Checkpoint has no requested head(s): {set(requested) - set(self.adapter.spec.target_bundle.kinds)}"
            )
        if postprocess == "hybrid":
            if not {"kp", "line"}.issubset(requested):
                raise CourtModelIOError(
                    "Hybrid postprocess requires both KP and LINE heads"
                )
            spec = self.adapter.spec.target_bundle.targets["kp"]
            if (
                tuple(spec.channel_names) != GROUND_COURT_KP_NAMES
                or self.max_peaks != 1
            ):
                raise CourtModelIOError(
                    "Hybrid postprocess requires ordered KP14 and max_peaks=1"
                )
        images, original_size = self._prepare_image(image)
        with torch.no_grad():
            call = self.adapter.prepare_images(images)
            output = cast(CourtLogits | CourtModelOutput, self.model(*call.model_args))
            self.adapter.validate_logits(output, call)
            logits = (
                output.dense_logits if isinstance(output, CourtModelOutput) else output
            )
            decoded: dict[CourtTargetKind, CourtDecodedPrediction] = {}
            for kind in requested:
                if kind == "kp":
                    decoded[kind] = decode_court_keypoint_logits(
                        logits[kind],
                        original_size_hw=original_size,
                        subpixel_refine=self.subpixel_refine,
                        config=self.decoder_config,
                    )
                else:
                    decoded[kind] = self.adapter.decode_prediction(
                        kind,
                        logits[kind],
                        original_size_hw=original_size,
                        subpixel_refine=False,
                    )
        geometry = None
        if postprocess == "hybrid":
            kp, line = decoded["kp"], decoded["line"]
            if not isinstance(kp, CourtKeypointPrediction) or not isinstance(
                line, CourtLinePrediction
            ):
                raise CourtModelIOError("Unexpected decoded KP/LINE types")
            observed = kp.keypoints[:, 0].numpy().astype(np.float64)
            observed[~kp.valid[:, 0].numpy()] = np.nan
            geometry = estimate_hybrid_homography(
                court_template_xy(14),
                observed,
                kp.scores[:, 0].numpy(),
                line.probability.numpy(),
                edges=np.asarray(COURT_HOMOGRAPHY_EDGES),
                image_size_hw=original_size,
                config=self.hybrid_config,
            )
        return CourtPrediction(
            decoded,
            original_size,
            tuple(next(iter(logits.values())).shape[-2:]),
            geometry,
            self.adapter.spec.target_bundle.targets["kp"].schema
            if "kp" in decoded
            else None,
        )

    def _prepare_image(self, image: CourtImage) -> tuple[Tensor, tuple[int, int]]:
        if isinstance(image, Tensor):
            if image.ndim not in {3, 4}:
                raise CourtModelIOError(
                    "Court predictor tensors must have shape (C,H,W) or (1,C,H,W)"
                )
            images = image.unsqueeze(0) if image.ndim == 3 else image
            if images.shape[0] != 1:
                raise CourtModelIOError("Court predictors accept exactly one image")
            return images.to(self.device), (image.shape[-2], image.shape[-1])
        images, height, width = prepare_court_image(
            image, short_side=self.short_side, device=self.device
        )
        return images, (height, width)

    @property
    def decoder_config(self) -> CourtKeypointDecoderConfig:
        return self._decoder_config

    @property
    def peak_threshold(self) -> float:
        return float(self.decoder_config.threshold)

    @property
    def nms_kernel(self) -> int:
        return int(self.decoder_config.nms_kernel)

    @property
    def max_peaks(self) -> int:
        return int(self.decoder_config.max_peaks)

    @property
    def short_side(self) -> int:
        return int(self.adapter.spec.short_side)


class _CourtHeadPredictor:
    """Compatibility view for explicitly raw head-only inference."""

    target_kind: CourtTargetKind

    def __init__(
        self, model_io: CourtBoundModelIO, device: torch.device, **kwargs: Any
    ) -> None:
        if (
            not isinstance(model_io.adapter, CourtModelIOAdapter)
            or self.target_kind not in model_io.adapter.spec.target_bundle.targets
        ):
            raise CourtModelIOError(
                f"Court predictor requires a checkpoint with a {self.target_kind} head"
            )
        self.predictor = CourtPredictor(model_io, device, **kwargs)
        self.model_io = self.predictor.model_io
        self.model = self.predictor.model
        self.adapter = self.predictor.adapter
        self.device = self.predictor.device

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str | Path | Iterable[str | Path], **kwargs: Any
    ) -> Self:
        predictor = CourtPredictor.load_from_checkpoint(checkpoint_path, **kwargs)
        return cls(
            predictor.model_io,
            predictor.device,
            subpixel_refine=predictor.subpixel_refine,
            peak_threshold=predictor.peak_threshold,
            nms_kernel=predictor.nms_kernel,
            max_peaks=predictor.max_peaks,
            checkpoint_identity=predictor.checkpoint_identity,
        )

    @property
    def checkpoint_identity(self) -> dict[str, Any]:
        return self.predictor.checkpoint_identity

    @property
    def task(self) -> CourtTargetKind:
        return self.target_kind

    def _predict_head(self, image: CourtImage) -> CourtDecodedPrediction:
        return self.predictor.predict(
            image, postprocess="none", heads=(self.target_kind,)
        ).raw_heads[self.target_kind]


class CourtKeypointPredictor(
    _CourtHeadPredictor, BasePredictor[CourtKeypointPrediction]
):
    """Raw KP view; downstream geometry consumers use CourtPredictor instead."""

    target_kind: CourtTargetKind = "kp"

    def predict(self, image: CourtImage) -> CourtKeypointPrediction:
        return cast(CourtKeypointPrediction, self._predict_head(image))

    @property
    def short_side(self) -> int:
        return self.predictor.short_side

    @property
    def decoder_config(self) -> CourtKeypointDecoderConfig:
        return self.predictor.decoder_config

    @property
    def peak_threshold(self) -> float:
        return self.predictor.peak_threshold

    @property
    def nms_kernel(self) -> int:
        return self.predictor.nms_kernel

    @property
    def max_peaks(self) -> int:
        return self.predictor.max_peaks

    @property
    def subpixel_refine(self) -> bool:
        return self.predictor.subpixel_refine


class CourtLinePredictor(_CourtHeadPredictor, BasePredictor[CourtLinePrediction]):
    target_kind: CourtTargetKind = "line"

    def predict(self, image: CourtImage) -> CourtLinePrediction:
        return cast(CourtLinePrediction, self._predict_head(image))


class CourtSegPredictor(
    _CourtHeadPredictor, BasePredictor[CourtSegmentationPrediction]
):
    target_kind: CourtTargetKind = "seg"

    def predict(self, image: CourtImage) -> CourtSegmentationPrediction:
        return cast(CourtSegmentationPrediction, self._predict_head(image))


class CourtSemanticLinePredictor(CourtSegPredictor):
    target_kind: CourtTargetKind = "semantic_line"


__all__ = [
    "CourtPredictor",
    "CourtPrediction",
    "CourtKeypointPredictor",
    "CourtLinePredictor",
    "CourtSegPredictor",
    "CourtSemanticLinePredictor",
]
