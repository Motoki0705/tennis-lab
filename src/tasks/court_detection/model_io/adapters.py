"""Task-specific court input, loss, and output adapters."""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol, TypeAlias, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.tasks.base.data.court_peaks import COURT_SEMANTIC_CLASS_NAMES
from src.tasks.base.model_io import ModelCall
from src.tasks.base.training.losses import (
    FocalBCEWithLogitsLoss,
    validate_focal_bce_inputs,
)
from src.tasks.court_detection.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtModelCall,
    CourtModelIOError,
    CourtModelSpec,
    CourtSegmentationPrediction,
    CourtTrainingCall,
    CourtTrainingResult,
)
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.tasks.court_detection.training.losses import BinaryDiceLoss, DiceLoss
from src.utils.data.heatmaps import (
    heatmaps_to_peaks,
    heatmaps_to_pixel_coords,
    refine_peaks_log_parabolic,
)

CourtDecodedPrediction: TypeAlias = (
    CourtKeypointPrediction | CourtSegmentationPrediction | CourtLinePrediction
)

_NORMALIZED_IMAGE_MIN = tuple(
    -mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)
)
_NORMALIZED_IMAGE_MAX = tuple(
    (1.0 - mean) / std
    for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)
)


class CourtModelExecutionBoundary(Protocol):
    """Prepare encoder-specific arguments outside the model forward path."""

    def bind_model(self, model: CourtHierarchicalModel) -> None:
        """Bind the construction-validated court model."""
        ...

    def prepare(self, call: CourtModelCall) -> CourtModelCall:
        """Return a complete validated model invocation."""
        ...


def _run_dinov3_intermediate_layers(
    encoder: CourtDINOv3Encoder,
    images: Tensor,
) -> object:
    return cast(Any, encoder)._get_intermediate_layers(
        images,
        n=encoder.out_indices,
        reshape=False,
        return_class_token=False,
        norm=True,
    )


def _run_frozen_dinov3_intermediate_layers(
    encoder: CourtDINOv3Encoder,
    images: Tensor,
) -> object:
    with torch.no_grad():
        return _run_dinov3_intermediate_layers(encoder, images)


class CourtDINOv3ExecutionBoundary:
    """Own DINO execution and four-level response validation/decoding."""

    def __init__(self, *, frozen_backbone: bool) -> None:
        self._model_ref: weakref.ReferenceType[CourtHierarchicalModel] | None = None
        self._backbone_executor = (
            _run_frozen_dinov3_intermediate_layers
            if frozen_backbone
            else _run_dinov3_intermediate_layers
        )

    def bind_model(self, model: CourtHierarchicalModel) -> None:
        if not isinstance(model.encoder, CourtDINOv3Encoder):
            raise CourtModelIOError(
                "Court DINOv3 execution boundary requires CourtDINOv3Encoder."
            )
        self._model_ref = weakref.ref(model)

    def prepare(self, call: CourtModelCall) -> CourtModelCall:
        if self._model_ref is None or (model := self._model_ref()) is None:
            raise CourtModelIOError(
                "Court DINOv3 execution boundary is not bound to its model."
            )
        encoder = cast(CourtDINOv3Encoder, model.encoder)
        patch_size = encoder.patch_size
        pad_h = (-call.height) % patch_size
        pad_w = (-call.width) % patch_size
        padded_images = (
            call.images
            if pad_h == 0 and pad_w == 0
            else F.pad(call.images, (0, pad_w, 0, pad_h), mode="replicate")
        )
        patch_height = padded_images.shape[-2] // patch_size
        patch_width = padded_images.shape[-1] // patch_size
        expected_shape = (
            call.batch_size,
            patch_height * patch_width,
            encoder.backbone.embed_dim,
        )
        raw_output = self._backbone_executor(encoder, padded_images)
        if not isinstance(raw_output, tuple) or len(raw_output) != 4:
            raise CourtModelIOError(
                "Court DINOv3 get_intermediate_layers must return four tensors."
            )

        tokens: list[Tensor] = []
        output_dtype: torch.dtype | None = None
        for level, value in enumerate(raw_output):
            if not isinstance(value, Tensor):
                raise CourtModelIOError(
                    f"Court DINOv3 level {level} output must be a Tensor."
                )
            if value.shape != expected_shape or not value.is_floating_point():
                raise CourtModelIOError(
                    f"Court DINOv3 level {level} must have floating shape "
                    f"{expected_shape}, got {tuple(value.shape)} and {value.dtype}."
                )
            if value.device != padded_images.device:
                raise CourtModelIOError(
                    "Court DINOv3 outputs must remain on the input device."
                )
            _require_finite(value, name=f"Court DINOv3 level {level}")
            if output_dtype is None:
                output_dtype = value.dtype
            elif value.dtype != output_dtype:
                raise CourtModelIOError(
                    "Court DINOv3 outputs must use one consistent dtype."
                )
            tokens.append(value)

        feature_maps = tuple(
            level_tokens.transpose(1, 2).reshape(
                call.batch_size,
                encoder.backbone.embed_dim,
                patch_height,
                patch_width,
            )
            for level_tokens in tokens
        )
        return replace(
            call,
            model_args=(call.images, *feature_maps),
        )


class CourtModelIOAdapter(nn.Module, ABC):
    """Base contract shared by the three explicitly selected court tasks."""

    def __init__(
        self,
        spec: CourtModelSpec,
        *,
        execution_boundary: CourtModelExecutionBoundary | None = None,
    ) -> None:
        super().__init__()
        if spec.in_channels != 3:
            raise CourtModelIOError("Court model input must use exactly three RGB channels.")
        if spec.output_channels <= 0:
            raise CourtModelIOError("Court model output channels must be positive.")
        if spec.short_side <= 0:
            raise CourtModelIOError("Court preprocessing short_side must be positive.")
        self.spec = spec
        self.execution_boundary = execution_boundary
        self._prepare_execution = (
            self._prepare_direct_execution
            if execution_boundary is None
            else execution_boundary.prepare
        )

    @property
    def model_type(self) -> type[nn.Module]:
        """Return the sole court model class supported by the adapter."""
        return cast("type[nn.Module]", CourtHierarchicalModel)

    def validate_model_pair(self, model: nn.Module) -> None:
        """Reject model/adapter mismatches at composition time."""
        if not isinstance(model, CourtHierarchicalModel):
            raise CourtModelIOError(
                "Court model-I/O adapters require CourtHierarchicalModel, got "
                f"{type(model).__name__}."
            )
        if (
            model.in_channels != self.spec.in_channels
            or model.num_classes != self.spec.output_channels
        ):
            raise CourtModelIOError(
                "Court model channels do not match the selected task adapter."
            )
        if self.execution_boundary is not None:
            self.execution_boundary.bind_model(model)

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        """Implement the shared lifecycle for a validated training batch."""
        prepared = self.prepare_training_batch(batch)
        return ModelCall(args=prepared.model_call.model_args)

    def decode_output(self, output: Tensor) -> Tensor:
        """Implement the shared lifecycle's raw-logit validation."""
        self.validate_logits(output)
        return output

    def prepare_images(self, images: Tensor) -> CourtModelCall:
        """Validate a normalized ``(B,3,H,W)`` tensor before forward."""
        if images.ndim != 4:
            raise CourtModelIOError(
                "Court images must be a rank-4 tensor (B, C, H, W)."
            )
        if images.dtype != torch.float32:
            raise CourtModelIOError(
                f"Court images must use torch.float32, got {images.dtype}."
            )
        batch_size, channels, height, width = images.shape
        if batch_size <= 0 or height <= 0 or width <= 0:
            raise CourtModelIOError("Court images must have positive dimensions.")
        if channels != self.spec.in_channels:
            raise CourtModelIOError(
                f"Court images require {self.spec.in_channels} channels, got {channels}."
            )
        _require_finite(images, name="Court images")
        lower = images.new_tensor(_NORMALIZED_IMAGE_MIN).view(1, 3, 1, 1)
        upper = images.new_tensor(_NORMALIZED_IMAGE_MAX).view(1, 3, 1, 1)
        if bool(torch.any((images < lower) | (images > upper))):
            raise CourtModelIOError(
                "Court images must be ImageNet-normalized values derived from "
                "RGB samples in [0, 1]."
            )
        call = CourtModelCall(
            images=images,
            model_args=(images,),
            batch_size=batch_size,
            height=height,
            width=width,
        )
        return self._prepare_execution(call)

    @staticmethod
    def _prepare_direct_execution(call: CourtModelCall) -> CourtModelCall:
        return call

    def validate_logits(
        self,
        logits: Tensor,
        call: CourtModelCall | None = None,
    ) -> None:
        """Validate court output rank, dtype, channels, and spatial semantics."""
        if logits.ndim != 4 or not logits.is_floating_point():
            raise CourtModelIOError("Court logits must be a rank-4 floating tensor.")
        _require_finite(logits, name="Court logits")
        if logits.shape[1] != self.spec.output_channels:
            raise CourtModelIOError(
                f"Court logits require {self.spec.output_channels} channels, "
                f"got {logits.shape[1]}."
            )
        if call is not None and logits.shape != (
            call.batch_size,
            self.spec.output_channels,
            call.height,
            call.width,
        ):
            raise CourtModelIOError(
                "Court logits must preserve the validated batch and spatial size; "
                f"got {tuple(logits.shape)}."
            )

    @abstractmethod
    def prepare_training_batch(
        self,
        batch: Mapping[str, object],
    ) -> CourtTrainingCall:
        """Validate the task's exact required batch fields."""

    @abstractmethod
    def training_result(
        self,
        logits: Tensor,
        call: CourtTrainingCall,
    ) -> CourtTrainingResult:
        """Validate output and compute the selected task loss."""

    @abstractmethod
    def test_payload(
        self,
        batch: Mapping[str, object],
        logits: Tensor,
    ) -> dict[str, object]:
        """Decode a task-specific test persistence payload."""

    @abstractmethod
    def decode_prediction(
        self,
        logits: Tensor,
        *,
        original_size_hw: tuple[int, int],
        subpixel_refine: bool,
    ) -> CourtDecodedPrediction:
        """Decode one inference result into a typed task result."""


class CourtKeypointModelIO(CourtModelIOAdapter):
    """Keypoint heatmap input, focal loss, and decode contract."""

    def __init__(
        self,
        spec: CourtModelSpec,
        *,
        focal_gamma: float,
        execution_boundary: CourtModelExecutionBoundary | None = None,
    ) -> None:
        if spec.task != "kp":
            raise CourtModelIOError("CourtKeypointModelIO requires task='kp'.")
        super().__init__(spec, execution_boundary=execution_boundary)
        self.loss_fn = FocalBCEWithLogitsLoss(gamma=focal_gamma)

    def prepare_training_batch(self, batch: Mapping[str, object]) -> CourtTrainingCall:
        images = _tensor(batch, "image")
        target = _tensor(batch, "heatmap")
        keypoints = _tensor(batch, "keypoints")
        visible = _tensor(batch, "kp_visible")
        call = self.prepare_images(images)
        expected_heatmap = (
            call.batch_size,
            self.spec.output_channels,
            call.height,
            call.width,
        )
        if target.shape != expected_heatmap or not target.is_floating_point():
            raise CourtModelIOError(
                f"Keypoint heatmap must have shape {expected_heatmap} and float dtype."
            )
        _require_finite(target, name="Keypoint heatmap")
        if bool(torch.any((target < 0.0) | (target > 1.0))):
            raise CourtModelIOError("Keypoint heatmap values must be in [0, 1].")
        if keypoints.shape != (call.batch_size, self.spec.output_channels, 2):
            raise CourtModelIOError("keypoints must have shape (B, K, 2).")
        if not keypoints.is_floating_point():
            raise CourtModelIOError("keypoints must use a floating dtype.")
        _require_finite(keypoints, name="keypoints")
        if visible.shape != (call.batch_size, self.spec.output_channels):
            raise CourtModelIOError("kp_visible must have shape (B, K).")
        if visible.dtype != torch.bool and not visible.is_floating_point():
            raise CourtModelIOError("kp_visible must use a boolean or floating dtype.")
        if visible.is_floating_point():
            _require_finite(visible, name="kp_visible")
            if bool(torch.any((visible < 0.0) | (visible > 1.0))):
                raise CourtModelIOError("kp_visible values must be in [0, 1].")
        return CourtTrainingCall(call, target, dict(batch))

    def training_result(
        self, logits: Tensor, call: CourtTrainingCall
    ) -> CourtTrainingResult:
        self.validate_logits(logits, call.model_call)
        validate_focal_bce_inputs(logits, call.target)
        return CourtTrainingResult(self.loss_fn(logits, call.target), logits)

    def test_payload(
        self, batch: Mapping[str, object], logits: Tensor
    ) -> dict[str, object]:
        self.validate_logits(logits)
        return {
            "image_id": _required(batch, "image_id"),
            "image_size": _required(batch, "image_size"),
            "pred_keypoints": heatmaps_to_pixel_coords(logits),
            "target_keypoints": _tensor(batch, "keypoints"),
        }

    def decode_prediction(
        self,
        logits: Tensor,
        *,
        original_size_hw: tuple[int, int],
        subpixel_refine: bool,
        max_peaks: int = 4,
    ) -> CourtKeypointPrediction:
        self.validate_logits(logits)
        probabilities = torch.sigmoid(logits)
        coords, scores, valid = heatmaps_to_peaks(
            probabilities,
            threshold=0.05,
            nms_kernel=7,
            max_peaks=max_peaks,
        )
        if subpixel_refine:
            coords = refine_peaks_log_parabolic(probabilities, coords)
        original_height, original_width = original_size_hw
        scale = coords.new_tensor(
            [float(max(original_width - 1, 0)), float(max(original_height - 1, 0))]
        )
        covariance_scale = coords.new_tensor(
            [float(max(original_width - 1, 1)), float(max(original_height - 1, 1))]
        )
        covariance = _local_peak_covariance(probabilities, coords)
        covariance = (
            covariance
            * covariance_scale.view(1, 1, 2, 1)
            * covariance_scale.view(1, 1, 1, 2)
        )
        keypoints = (coords[0] * scale).masked_fill(~valid[0].unsqueeze(-1), 0.0)
        covariance = covariance[0].masked_fill(
            ~valid[0].unsqueeze(-1).unsqueeze(-1), 0.0
        )
        return CourtKeypointPrediction(
            keypoints=keypoints.cpu(),
            scores=scores[0].cpu(),
            valid=valid[0].cpu(),
            covariance=covariance.cpu(),
            heatmaps=logits[0].cpu(),
            semantic_class_names=(
                COURT_SEMANTIC_CLASS_NAMES
                if self.spec.output_channels == len(COURT_SEMANTIC_CLASS_NAMES)
                else None
            ),
            image_size_hw=original_size_hw,
        )


class CourtSegmentationModelIO(CourtModelIOAdapter):
    """Multi-class segmentation input, loss, and decode contract."""

    def __init__(
        self,
        spec: CourtModelSpec,
        *,
        ce_weight: float,
        dice_weight: float,
        execution_boundary: CourtModelExecutionBoundary | None = None,
    ) -> None:
        if spec.task != "seg":
            raise CourtModelIOError("CourtSegmentationModelIO requires task='seg'.")
        super().__init__(spec, execution_boundary=execution_boundary)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.dice_loss_fn = DiceLoss(num_classes=spec.output_channels)

    def prepare_training_batch(self, batch: Mapping[str, object]) -> CourtTrainingCall:
        images = _tensor(batch, "image")
        target = _tensor(batch, "mask")
        call = self.prepare_images(images)
        if target.shape != (call.batch_size, call.height, call.width):
            raise CourtModelIOError("Segmentation mask must have shape (B, H, W).")
        if target.dtype != torch.long:
            raise CourtModelIOError("Segmentation mask must use torch.int64 labels.")
        if bool(torch.any((target < 0) | (target >= self.spec.output_channels))):
            raise CourtModelIOError(
                "Segmentation mask contains an invalid class index."
            )
        return CourtTrainingCall(call, target, dict(batch))

    def training_result(
        self, logits: Tensor, call: CourtTrainingCall
    ) -> CourtTrainingResult:
        self.validate_logits(logits, call.model_call)
        loss = self.ce_weight * self.ce_loss_fn(logits, call.target)
        loss = loss + self.dice_weight * self.dice_loss_fn(logits, call.target)
        return CourtTrainingResult(loss, logits)

    def test_payload(
        self, batch: Mapping[str, object], logits: Tensor
    ) -> dict[str, object]:
        self.validate_logits(logits)
        target = _tensor(batch, "mask")
        batch_size = logits.shape[0]
        return {
            "image_id": _required(batch, "image_id"),
            "image_size": _required(batch, "image_size"),
            "pred_mask_flat": logits.argmax(dim=1).reshape(batch_size, -1),
            "target_mask_flat": target.reshape(batch_size, -1),
            "padded_size": _padded_size(logits),
        }

    def decode_prediction(
        self,
        logits: Tensor,
        *,
        original_size_hw: tuple[int, int],
        subpixel_refine: bool,
    ) -> CourtSegmentationPrediction:
        _ = (original_size_hw, subpixel_refine)
        self.validate_logits(logits)
        return CourtSegmentationPrediction(
            mask=logits[0].argmax(0).to(torch.long).cpu(),
            logits=logits[0].cpu(),
        )


class CourtLineModelIO(CourtModelIOAdapter):
    """Binary court-line input, weighted BCE/Dice loss, and decode contract."""

    def __init__(
        self,
        spec: CourtModelSpec,
        *,
        bce_weight: float,
        dice_weight: float,
        pos_weight: float,
        execution_boundary: CourtModelExecutionBoundary | None = None,
    ) -> None:
        if spec.task != "line":
            raise CourtModelIOError("CourtLineModelIO requires task='line'.")
        super().__init__(spec, execution_boundary=execution_boundary)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.dice_loss_fn = BinaryDiceLoss()

    def prepare_training_batch(self, batch: Mapping[str, object]) -> CourtTrainingCall:
        images = _tensor(batch, "image")
        target = _tensor(batch, "mask")
        call = self.prepare_images(images)
        if target.shape != (call.batch_size, 1, call.height, call.width):
            raise CourtModelIOError("Line mask must have shape (B, 1, H, W).")
        if not target.is_floating_point():
            raise CourtModelIOError("Line mask must use a floating dtype.")
        _require_finite(target, name="Line mask")
        if bool(torch.any((target < 0.0) | (target > 1.0))):
            raise CourtModelIOError("Line mask values must be in [0, 1].")
        return CourtTrainingCall(call, target, dict(batch))

    def training_result(
        self, logits: Tensor, call: CourtTrainingCall
    ) -> CourtTrainingResult:
        self.validate_logits(logits, call.model_call)
        loss = self.bce_weight * F.binary_cross_entropy_with_logits(
            logits,
            call.target,
            pos_weight=self.get_buffer("pos_weight"),
        )
        loss = loss + self.dice_weight * self.dice_loss_fn(logits, call.target)
        return CourtTrainingResult(loss, logits)

    def test_payload(
        self, batch: Mapping[str, object], logits: Tensor
    ) -> dict[str, object]:
        self.validate_logits(logits)
        target = _tensor(batch, "mask")
        batch_size = logits.shape[0]
        return {
            "image_id": _required(batch, "image_id"),
            "image_size": _required(batch, "image_size"),
            "pred_line_prob_flat": torch.sigmoid(logits).reshape(batch_size, -1),
            "target_line_mask_flat": target.reshape(batch_size, -1),
            "padded_size": _padded_size(logits),
        }

    def decode_prediction(
        self,
        logits: Tensor,
        *,
        original_size_hw: tuple[int, int],
        subpixel_refine: bool,
    ) -> CourtLinePrediction:
        _ = (original_size_hw, subpixel_refine)
        self.validate_logits(logits)
        return CourtLinePrediction(
            probability=torch.sigmoid(logits[0, 0]).cpu(),
            logits=logits[0].cpu(),
        )


def _required(batch: Mapping[str, object], key: str) -> object:
    if key not in batch:
        raise CourtModelIOError(f"Court batch is missing required field {key!r}.")
    return batch[key]


def _tensor(batch: Mapping[str, object], key: str) -> Tensor:
    value = _required(batch, key)
    if not isinstance(value, Tensor):
        raise CourtModelIOError(f"Court batch field {key!r} must be a Tensor.")
    return value


def _padded_size(logits: Tensor) -> Tensor:
    return logits.new_tensor(
        [logits.shape[-2], logits.shape[-1]], dtype=torch.int64
    ).repeat(logits.shape[0], 1)


def _local_peak_covariance(probability: Tensor, coords: Tensor) -> Tensor:
    """Estimate normalized covariance from 5x5 heatmap local moments."""
    if probability.ndim != 4 or coords.ndim != 4 or coords.shape[-1] != 2:
        raise CourtModelIOError(
            "local covariance requires (B,C,H,W) and (B,C,P,2)."
        )
    batch_size, channels, height, width = probability.shape
    if coords.shape[:2] != (batch_size, channels):
        raise CourtModelIOError("peak coordinates must share heatmap B/C axes.")
    center_x = (coords[..., 0] * max(width - 1, 1)).round().long()
    center_y = (coords[..., 1] * max(height - 1, 1)).round().long()
    batch_index = torch.arange(batch_size, device=probability.device).view(
        batch_size, 1, 1
    )
    channel_index = torch.arange(channels, device=probability.device).view(
        1, channels, 1
    )
    covariance = probability.new_zeros(*coords.shape[:-1], 2, 2)
    normalizer = probability.new_zeros(coords.shape[:-1])
    for y_offset in range(-2, 3):
        for x_offset in range(-2, 3):
            sample_x = (center_x + x_offset).clamp(0, width - 1)
            sample_y = (center_y + y_offset).clamp(0, height - 1)
            weight = probability[
                batch_index, channel_index, sample_y, sample_x
            ]
            offset = probability.new_tensor(
                (
                    x_offset / float(max(width - 1, 1)),
                    y_offset / float(max(height - 1, 1)),
                )
            )
            covariance += weight[..., None, None] * (
                offset[:, None] * offset[None, :]
            )
            normalizer += weight
    return covariance / normalizer.clamp_min(
        torch.finfo(probability.dtype).eps
    )[..., None, None]


def _require_finite(tensor: Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise CourtModelIOError(f"{name} must contain only finite values.")


__all__ = [
    "CourtDecodedPrediction",
    "CourtDINOv3ExecutionBoundary",
    "CourtKeypointModelIO",
    "CourtLineModelIO",
    "CourtModelIOAdapter",
    "CourtSegmentationModelIO",
]
