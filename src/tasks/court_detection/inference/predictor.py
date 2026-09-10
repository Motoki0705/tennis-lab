"""Typed inference predictor for Court keypoint heads."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, Self, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.pose import CourtDecodedPose
from src.tasks.court_detection.model_io.adapters import (
    CourtModelIOAdapter,
    CourtPoseModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedOutput,
    CourtKeypointPrediction,
    CourtLogits,
    CourtModelIOError,
    CourtModelOutput,
    CourtPosePrediction,
)
from src.tasks.court_detection.model_io.images import (
    PreparedCourtPoseImage,
    prepare_court_image,
    prepare_court_pose_image,
)
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.configuration import PathResolver


class CourtPredictorModelIO(Protocol):
    """Structural bound pair used by both dense and pose Court predictors."""

    model: torch.nn.Module
    adapter: CourtModelIOAdapter


class CourtKeypointPredictor(BasePredictor[CourtKeypointPrediction]):
    """Predict one KP head from a single- or multi-target checkpoint."""

    def __init__(
        self,
        model_io: CourtPredictorModelIO,
        device: torch.device,
        *,
        subpixel_refine: bool,
        max_peaks: int = 4,
    ) -> None:
        if isinstance(model_io.adapter, CourtPoseModelIOAdapter):
            raise CourtModelIOError(
                "Pose-enabled checkpoints require CourtPosePredictor so pose-safe "
                "image geometry is preserved."
            )
        if not isinstance(model_io.adapter, CourtModelIOAdapter):
            raise CourtModelIOError(
                "CourtKeypointPredictor requires CourtModelIOAdapter."
            )
        if "kp" not in model_io.adapter.spec.target_bundle.targets:
            raise CourtModelIOError(
                "CourtKeypointPredictor requires a checkpoint with a KP head."
            )
        if max_peaks <= 0:
            raise ValueError("Court keypoint max_peaks must be positive.")
        self.model_io = model_io
        self.model = model_io.model
        self.adapter: CourtModelIOAdapter = model_io.adapter
        self.device = device
        self.subpixel_refine = subpixel_refine
        self.max_peaks = max_peaks

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
        max_peaks: int = 4,
        **kwargs: Any,
    ) -> Self:
        """Load one checkpoint and preserve its serialized target bundle."""
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            CourtDetectionLightningModule,
            resolver=resolver,
            device=device,
            weights_only=False,
            **_checkpoint_load_kwargs(kwargs),
        )
        adapter = lightning_module.model_io
        adapter.validate_model_pair(lightning_module.model)
        return cls(
            cast(
                CourtPredictorModelIO,
                bind_model_io(lightning_module.model, adapter),
            ),
            resolved_device,
            subpixel_refine=subpixel_refine,
            max_peaks=max_peaks,
        )

    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
    ) -> CourtKeypointPrediction:
        """Return multi-peak KP channels, scores, validity, and heatmaps."""
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
            output = self.model(*call.model_args)
            logits = _dense_logits(output)
        return cast(
            CourtKeypointPrediction,
            self.adapter.decode_prediction(
                "kp",
                logits["kp"],
                original_size_hw=original_size_hw,
                subpixel_refine=self.subpixel_refine,
                max_peaks=self.max_peaks,
            ),
        )

    @property
    def task(self) -> CourtTargetKind:
        return "kp"

    @property
    def short_side(self) -> int:
        return int(self.adapter.spec.short_side)


class CourtPosePredictor(BasePredictor[CourtPosePrediction]):
    """Predict camera pose and every dense head from a pose-enabled checkpoint."""

    def __init__(
        self,
        model_io: CourtPredictorModelIO,
        device: torch.device,
        *,
        patch_size: int,
        subpixel_refine: bool = True,
        max_peaks: int = 4,
    ) -> None:
        if not isinstance(model_io.adapter, CourtPoseModelIOAdapter):
            raise CourtModelIOError(
                "CourtPosePredictor requires a pose-enabled checkpoint adapter."
            )
        if patch_size <= 0:
            raise ValueError("Court pose predictor patch_size must be positive.")
        if max_peaks <= 0:
            raise ValueError("Court pose predictor max_peaks must be positive.")
        self.model_io = model_io
        self.model = model_io.model
        self.adapter: CourtPoseModelIOAdapter = model_io.adapter
        self.device = device
        self.patch_size = patch_size
        self.subpixel_refine = subpixel_refine
        self.max_peaks = max_peaks

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
        subpixel_refine: bool = True,
        max_peaks: int = 4,
        **kwargs: Any,
    ) -> Self:
        """Load exactly one serialized pose-enabled Court checkpoint."""

        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            CourtDetectionLightningModule,
            resolver=resolver,
            device=device,
            weights_only=False,
            **_checkpoint_load_kwargs(kwargs),
        )
        runtime = CourtTrainingConfig.from_config(lightning_module.config)
        adapter = lightning_module.model_io
        if not isinstance(adapter, CourtPoseModelIOAdapter):
            raise CourtModelIOError(
                "CourtPosePredictor requires a checkpoint trained with pose enabled."
            )
        adapter.validate_model_pair(lightning_module.model)
        return cls(
            cast(
                CourtPredictorModelIO,
                bind_model_io(lightning_module.model, adapter),
            ),
            resolved_device,
            patch_size=runtime.data.augmentation.patch_size,
            subpixel_refine=subpixel_refine,
            max_peaks=max_peaks,
        )

    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
    ) -> CourtPosePrediction:
        """Return camera pose and decoded dense predictions in source-image pixels."""

        prepared = self._prepare_image(image)
        image_size = torch.tensor(
            [prepared.model_size_hw],
            dtype=torch.long,
            device=self.device,
        )
        content_size = torch.tensor(
            [prepared.content_size_hw],
            dtype=torch.long,
            device=self.device,
        )
        model_call = self.adapter.build_call(
            {
                "image": prepared.images,
                "image_size": image_size,
                "content_size_hw": content_size,
            }
        )
        with torch.no_grad():
            output = self.model(*model_call.args, **dict(model_call.kwargs))
            decoded = self.adapter.decode_output(output)

        dense = {
            kind: self.adapter.decode_prediction(
                kind,
                logits,
                original_size_hw=prepared.original_size_hw,
                subpixel_refine=self.subpixel_refine if kind == "kp" else False,
                max_peaks=self.max_peaks,
            )
            for kind, logits in decoded.dense_logits.items()
        }
        return CourtPosePrediction(
            pose=_pose_in_source_pixels(
                decoded,
                source_to_model_scale=prepared.source_to_model_scale,
            ),
            dense=MappingProxyType(dense),
        )

    def _prepare_image(
        self,
        image: np.ndarray | Image.Image | Tensor,
    ) -> PreparedCourtPoseImage:
        if isinstance(image, Tensor):
            if image.ndim not in {3, 4}:
                raise CourtModelIOError(
                    "Court pose predictor tensors must have shape "
                    "(C,H,W) or (1,C,H,W)."
                )
            images = image.unsqueeze(0) if image.ndim == 3 else image
            if images.shape[0] != 1:
                raise CourtModelIOError(
                    "Court pose predictor accepts exactly one image."
                )
            height, width = images.shape[-2:]
            return PreparedCourtPoseImage(
                images=images.to(self.device),
                original_size_hw=(height, width),
                content_size_hw=(height, width),
                model_size_hw=(height, width),
                source_to_model_scale=1.0,
            )
        return prepare_court_pose_image(
            image,
            long_side=self.adapter.spec.short_side,
            patch_size=self.patch_size,
            device=self.device,
        )

    @property
    def short_side(self) -> int:
        """Serialized validation size (the long side for pose-safe checkpoints)."""

        return int(self.adapter.spec.short_side)


def _dense_logits(output: object) -> CourtLogits:
    if isinstance(output, CourtModelOutput):
        return output.dense_logits
    if not isinstance(output, Mapping):
        raise CourtModelIOError(
            "Court dense predictor requires mapping or CourtModelOutput."
        )
    return cast(CourtLogits, output)


def _checkpoint_load_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Remove run-only mixed-source data configuration before model restore.

    Mixed-source training builds and serializes the ordinary Court model from the
    standard config. Its saved run YAML additionally contains a top-level
    ``mixed`` data-loader section, which is deliberately outside that model
    configuration contract.
    """

    result = dict(kwargs)
    config = result.get("config")
    if not isinstance(config, DictConfig) or "mixed" not in config:
        return result
    unresolved = OmegaConf.to_container(config, resolve=False)
    if not isinstance(unresolved, dict):
        raise TypeError("Court checkpoint config must resolve to a mapping.")
    standard = dict(unresolved)
    standard.pop("mixed")
    result["config"] = OmegaConf.create(standard)
    return result


def _pose_in_source_pixels(
    output: CourtDecodedOutput,
    *,
    source_to_model_scale: float,
) -> CourtDecodedPose:
    if not math.isfinite(source_to_model_scale) or source_to_model_scale <= 0.0:
        raise CourtModelIOError("Court pose preprocessing scale must be positive.")
    pose = output.pose
    focal_px = pose.focal_px / source_to_model_scale
    return CourtDecodedPose(
        translation_m=pose.translation_m.detach().cpu(),
        rotation=pose.rotation.detach().cpu(),
        focal_px=focal_px.detach().cpu(),
        log_focal=(pose.log_focal - math.log(source_to_model_scale)).detach().cpu(),
    )


__all__ = ["CourtKeypointPredictor", "CourtPosePredictor"]
