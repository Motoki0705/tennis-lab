"""Bundle-aware Court input, loss, and output adapter."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.tasks.base.model_io import ModelCall
from src.tasks.base.training.losses import FocalBCEWithLogitsLoss
from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedPrediction,
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtLogits,
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
    refine_peaks_log_parabolic,
)

_NORMALIZED_IMAGE_MIN = tuple(
    -mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)
)
_NORMALIZED_IMAGE_MAX = tuple(
    (1.0 - mean) / std
    for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)
)


class CourtModelExecutionBoundary(Protocol):
    def bind_model(self, model: CourtHierarchicalModel) -> None: ...

    def prepare(self, call: CourtModelCall) -> CourtModelCall: ...


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
    """Own DINO execution and validate all four intermediate feature levels."""

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
        padded = (
            call.images
            if pad_h == 0 and pad_w == 0
            else F.pad(call.images, (0, pad_w, 0, pad_h), mode="replicate")
        )
        patch_height = padded.shape[-2] // patch_size
        patch_width = padded.shape[-1] // patch_size
        expected = (
            call.batch_size,
            patch_height * patch_width,
            encoder.backbone.embed_dim,
        )
        raw_output = self._backbone_executor(encoder, padded)
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
            if value.shape != expected or not value.is_floating_point():
                raise CourtModelIOError(
                    f"Court DINOv3 level {level} must have floating shape {expected}, "
                    f"got {tuple(value.shape)} and {value.dtype}."
                )
            if value.device != padded.device:
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
        features = tuple(
            value.transpose(1, 2).reshape(
                call.batch_size,
                encoder.backbone.embed_dim,
                patch_height,
                patch_width,
            )
            for value in tokens
        )
        return replace(call, model_args=(call.images, *features))


class CourtModelIOAdapter(nn.Module):
    """Validate one target bundle and compute one loss per selected head."""

    def __init__(
        self,
        spec: CourtModelSpec,
        *,
        loss_config: CourtLossConfig,
        execution_boundary: CourtModelExecutionBoundary | None = None,
    ) -> None:
        super().__init__()
        if spec.in_channels != 3:
            raise CourtModelIOError("Court model input must use exactly three RGB channels.")
        if spec.short_side <= 0:
            raise CourtModelIOError("Court preprocessing short_side must be positive.")
        self.spec = spec
        self.loss_config = loss_config
        self.execution_boundary = execution_boundary
        self._prepare_execution = (
            self._prepare_direct_execution
            if execution_boundary is None
            else execution_boundary.prepare
        )
        self.kp_loss = FocalBCEWithLogitsLoss(gamma=loss_config.kp_focal_gamma)
        self.seg_dice = DiceLoss(
            num_classes=(
                spec.target_bundle.targets["seg"].output_channels
                if "seg" in spec.target_bundle.targets
                else 1
            )
        )
        self.line_dice = BinaryDiceLoss()

    @property
    def model_type(self) -> type[nn.Module]:
        return cast("type[nn.Module]", CourtHierarchicalModel)

    def validate_model_pair(self, model: nn.Module) -> None:
        if not isinstance(model, CourtHierarchicalModel):
            raise CourtModelIOError(
                "Court model-I/O requires CourtHierarchicalModel, got "
                f"{type(model).__name__}."
            )
        if model.in_channels != self.spec.in_channels:
            raise CourtModelIOError("Court model input channels disagree with adapter.")
        if model.target_bundle_spec != self.spec.target_bundle:
            raise CourtModelIOError("Court model heads disagree with target bundle.")
        if self.execution_boundary is not None:
            self.execution_boundary.bind_model(model)

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        call = self.prepare_images(_tensor(batch, "image"))
        return ModelCall(args=call.model_args)

    def decode_output(self, output: CourtLogits) -> CourtLogits:
        self.validate_logits(output)
        return MappingProxyType(dict(output))

    def prepare_images(self, images: Tensor) -> CourtModelCall:
        if images.ndim != 4 or images.dtype != torch.float32:
            raise CourtModelIOError(
                "Court images must be float32 with shape (B,3,H,W)."
            )
        batch_size, channels, height, width = images.shape
        if (
            batch_size <= 0
            or height <= 0
            or width <= 0
            or channels != self.spec.in_channels
        ):
            raise CourtModelIOError("Court image dimensions/channels are invalid.")
        _require_finite(images, name="Court images")
        lower = images.new_tensor(_NORMALIZED_IMAGE_MIN).view(1, 3, 1, 1)
        upper = images.new_tensor(_NORMALIZED_IMAGE_MAX).view(1, 3, 1, 1)
        if bool(torch.any((images < lower) | (images > upper))):
            raise CourtModelIOError(
                "Court images must be ImageNet-normalized RGB values from [0,1]."
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

    def prepare_training_batch(
        self,
        batch: Mapping[str, object],
    ) -> CourtTrainingCall:
        call = self.prepare_images(_tensor(batch, "image"))
        raw_targets = batch.get("targets")
        if not isinstance(raw_targets, Mapping):
            raise CourtModelIOError("Court training batch requires a targets mapping.")
        if set(raw_targets) != set(self.spec.target_bundle.kinds):
            raise CourtModelIOError(
                "Court batch target keys must exactly match the resolved bundle."
            )
        targets: dict[CourtTargetKind, object] = {}
        for kind in self.spec.target_bundle.kinds:
            value = raw_targets[kind]
            if kind == "kp":
                targets[kind] = self._validate_kp_target(value, call=call)
            elif kind == "seg":
                targets[kind] = self._validate_seg_target(value, call=call)
            elif kind == "line":
                targets[kind] = self._validate_line_target(value, call=call)
        return CourtTrainingCall(
            model_call=call,
            targets=MappingProxyType(targets),
            batch=MappingProxyType(dict(batch)),
        )

    def validate_logits(
        self,
        logits: CourtLogits,
        call: CourtModelCall | None = None,
    ) -> None:
        if not isinstance(logits, Mapping):
            raise CourtModelIOError("Court model output must be a target mapping.")
        if set(logits) != set(self.spec.target_bundle.kinds):
            raise CourtModelIOError(
                "Court model output keys must exactly match the target bundle."
            )
        for kind, spec in self.spec.target_bundle.targets.items():
            value = logits[kind]
            if not isinstance(value, Tensor) or value.ndim != 4:
                raise CourtModelIOError(f"Court {kind} logits must be rank-4 Tensor.")
            if not value.is_floating_point():
                raise CourtModelIOError(f"Court {kind} logits must be floating.")
            _require_finite(value, name=f"Court {kind} logits")
            if value.shape[1] != spec.output_channels:
                raise CourtModelIOError(
                    f"Court {kind} logits require {spec.output_channels} channels."
                )
            if call is not None and value.shape != (
                call.batch_size,
                spec.output_channels,
                call.height,
                call.width,
            ):
                raise CourtModelIOError(
                    f"Court {kind} logits must preserve batch/spatial shape."
                )

    def training_result(
        self,
        logits: CourtLogits,
        call: CourtTrainingCall,
    ) -> CourtTrainingResult:
        self.validate_logits(logits, call.model_call)
        losses: dict[CourtTargetKind, Tensor] = {}
        for kind in self.spec.target_bundle.kinds:
            value = logits[kind]
            target = call.targets[kind]
            if kind == "kp":
                heatmap = cast(Mapping[str, Tensor], target)["heatmap"]
                losses[kind] = self.kp_loss(value, heatmap)
            elif kind == "seg":
                labels = cast(Tensor, target)
                losses[kind] = (
                    self.loss_config.seg_ce_weight
                    * F.cross_entropy(value, labels)
                    + self.loss_config.seg_dice_weight
                    * self.seg_dice(value, labels)
                )
            elif kind == "line":
                binary = cast(Tensor, target)
                pos_weight = value.new_tensor([self.loss_config.line_pos_weight])
                losses[kind] = (
                    self.loss_config.line_bce_weight
                    * F.binary_cross_entropy_with_logits(
                        value,
                        binary,
                        pos_weight=pos_weight,
                    )
                    + self.loss_config.line_dice_weight
                    * self.line_dice(value, binary)
                )
        total = torch.stack(tuple(losses.values())).sum()
        return CourtTrainingResult(
            loss=total,
            losses=MappingProxyType(losses),
            logits=MappingProxyType(dict(logits)),
        )

    def test_payload(
        self,
        batch: Mapping[str, object],
        logits: CourtLogits,
    ) -> dict[str, object]:
        self.validate_logits(logits)
        predictions: dict[str, object] = {}
        for kind, value in logits.items():
            if kind == "kp":
                probability = torch.sigmoid(value)
                coords, scores, valid = heatmaps_to_peaks(
                    probability,
                    threshold=0.05,
                    nms_kernel=7,
                    max_peaks=4,
                )
                predictions[kind] = {
                    "keypoints_normalized": coords,
                    "scores": scores,
                    "valid": valid,
                    "heatmaps": value,
                }
            elif kind == "seg":
                predictions[kind] = {"mask": value.argmax(dim=1), "logits": value}
            else:
                predictions[kind] = {
                    "probability": torch.sigmoid(value),
                    "logits": value,
                }
        return {
            "sample_id": batch.get("sample_id"),
            "image_size": batch.get("image_size"),
            "predictions": predictions,
        }

    def decode_prediction(
        self,
        kind: CourtTargetKind,
        logits: Tensor,
        *,
        original_size_hw: tuple[int, int],
        subpixel_refine: bool,
        max_peaks: int = 4,
    ) -> CourtDecodedPrediction:
        spec = self.spec.target_bundle.targets.get(kind)
        if spec is None:
            raise CourtModelIOError(f"Court bundle has no {kind!r} head.")
        self._validate_one_logits(kind, logits, spec.output_channels)
        if kind == "kp":
            probability = torch.sigmoid(logits)
            coords, scores, valid = heatmaps_to_peaks(
                probability,
                threshold=0.05,
                nms_kernel=7,
                max_peaks=max_peaks,
            )
            if subpixel_refine:
                coords = refine_peaks_log_parabolic(probability, coords)
            height, width = original_size_hw
            scale = coords.new_tensor(
                [float(max(width - 1, 0)), float(max(height - 1, 0))]
            )
            return CourtKeypointPrediction(
                keypoints=(coords[0] * scale).cpu(),
                scores=scores[0].cpu(),
                valid=valid[0].cpu(),
                heatmaps=logits[0].cpu(),
            )
        if kind == "seg":
            return CourtSegmentationPrediction(
                mask=logits.argmax(dim=1)[0].cpu(),
                logits=logits[0].cpu(),
            )
        return CourtLinePrediction(
            probability=torch.sigmoid(logits)[0, 0].cpu(),
            logits=logits[0, 0].cpu(),
        )

    def _validate_kp_target(
        self,
        value: object,
        *,
        call: CourtModelCall,
    ) -> Mapping[str, Tensor]:
        if not isinstance(value, Mapping) or set(value) != {
            "heatmap",
            "points_xy",
            "point_visible",
            "physical_indices",
        }:
            raise CourtModelIOError("Court KP target payload fields changed.")
        heatmap = _mapping_tensor(value, "heatmap")
        points = _mapping_tensor(value, "points_xy")
        visible = _mapping_tensor(value, "point_visible")
        physical = _mapping_tensor(value, "physical_indices")
        channels = self.spec.target_bundle.targets["kp"].output_channels
        if heatmap.shape != (call.batch_size, channels, call.height, call.width):
            raise CourtModelIOError("Court KP heatmap shape is invalid.")
        if not heatmap.is_floating_point():
            raise CourtModelIOError("Court KP heatmap must be floating.")
        _require_unit_interval(heatmap, name="Court KP heatmap")
        if (
            points.ndim != 4
            or points.shape[:2] != (call.batch_size, channels)
            or points.shape[-1] != 2
        ):
            raise CourtModelIOError("Court KP points must have shape (B,C,P,2).")
        if visible.shape != points.shape[:-1] or visible.dtype != torch.bool:
            raise CourtModelIOError("Court KP visibility must be bool (B,C,P).")
        if physical.shape != visible.shape or physical.dtype != torch.long:
            raise CourtModelIOError("Court KP physical IDs must be int64 (B,C,P).")
        if not points.is_floating_point():
            raise CourtModelIOError("Court KP points must be floating.")
        _require_finite(points, name="Court KP points")
        return {
            "heatmap": heatmap,
            "points_xy": points,
            "point_visible": visible,
            "physical_indices": physical,
        }

    def _validate_seg_target(
        self,
        value: object,
        *,
        call: CourtModelCall,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise CourtModelIOError("Court segmentation target must be a Tensor.")
        channels = self.spec.target_bundle.targets["seg"].output_channels
        if (
            value.shape != (call.batch_size, call.height, call.width)
            or value.dtype != torch.long
        ):
            raise CourtModelIOError(
                "Court segmentation target must be int64 (B,H,W)."
            )
        if bool(torch.any((value < 0) | (value >= channels))):
            raise CourtModelIOError("Court segmentation labels are out of range.")
        return value

    def _validate_line_target(
        self,
        value: object,
        *,
        call: CourtModelCall,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise CourtModelIOError("Court line target must be a Tensor.")
        if value.shape != (call.batch_size, 1, call.height, call.width):
            raise CourtModelIOError(
                "Court line target must have shape (B,1,H,W)."
            )
        if not value.is_floating_point():
            raise CourtModelIOError("Court line target must be floating.")
        _require_unit_interval(value, name="Court line target")
        return value

    @staticmethod
    def _validate_one_logits(
        kind: CourtTargetKind,
        value: Tensor,
        channels: int,
    ) -> None:
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != channels:
            raise CourtModelIOError(
                f"Court {kind} inference logits must have shape (1,{channels},H,W)."
            )
        if not value.is_floating_point():
            raise CourtModelIOError(f"Court {kind} inference logits must be floating.")
        _require_finite(value, name=f"Court {kind} inference logits")


def _tensor(mapping: Mapping[str, object], key: str) -> Tensor:
    value = mapping.get(key)
    if not isinstance(value, Tensor):
        raise CourtModelIOError(f"Court batch field {key!r} must be a Tensor.")
    return value


def _mapping_tensor(mapping: Mapping[object, object], key: str) -> Tensor:
    value = mapping[key]
    if not isinstance(value, Tensor):
        raise CourtModelIOError(f"Court target field {key!r} must be a Tensor.")
    return value


def _require_finite(value: Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise CourtModelIOError(f"{name} must contain only finite values.")


def _require_unit_interval(value: Tensor, *, name: str) -> None:
    _require_finite(value, name=name)
    if bool(torch.any((value < 0.0) | (value > 1.0))):
        raise CourtModelIOError(f"{name} values must be in [0,1].")


__all__ = [
    "CourtDINOv3ExecutionBoundary",
    "CourtModelExecutionBoundary",
    "CourtModelIOAdapter",
]
