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
from src.tasks.court_detection.configuration import (
    CourtLossConfig,
)
from src.tasks.court_detection.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.pose import (
    canonical_semantic_court_points_batched,
    decode_pose10d_strict,
    project_predicted_canonical_points,
    validate_proper_rotation,
    validate_square_intrinsics,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtConsistencyResult,
    CourtDecodedOutput,
    CourtDecodedPrediction,
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtLogits,
    CourtModelCall,
    CourtModelIOError,
    CourtModelOutput,
    CourtModelSpec,
    CourtPoseLossKind,
    CourtPosePrediction,
    CourtPoseTargetBatch,
    CourtPoseTrainingResult,
    CourtRawPoseOutput,
    CourtSegmentationPrediction,
    CourtTrainingCall,
    CourtTrainingResult,
    CourtTrainingTargetKind,
)
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.tasks.court_detection.training.losses import (
    BinaryDiceLoss,
    DiceLoss,
    consistency_effective_weight,
    keypoint_pose_consistency_loss,
    pose_losses,
)
from src.utils.data.heatmaps import (
    heatmaps_to_peaks,
    heatmaps_to_soft_argmax,
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

    @property
    def patch_valid_mask_required(self) -> bool:
        """Whether the bound hierarchical model consumes a patch keep-mask."""

        if self._model_ref is None or (model := self._model_ref()) is None:
            raise CourtModelIOError(
                "Court DINOv3 execution boundary is not bound to its model."
            )
        return bool(model.transformer_enabled)

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

    def attach_patch_valid_mask(
        self,
        call: CourtModelCall,
        content_size_hw: Tensor,
    ) -> CourtModelCall:
        """Append the per-sample content-to-patch keep mask to a prepared call."""

        if self._model_ref is None or (model := self._model_ref()) is None:
            raise CourtModelIOError(
                "Court DINOv3 execution boundary is not bound to its model."
            )
        if len(call.model_args) != 5:
            raise CourtModelIOError(
                "Court DINOv3 patch masking requires four prepared feature maps."
            )
        if (
            content_size_hw.shape != (call.batch_size, 2)
            or content_size_hw.dtype != torch.long
        ):
            raise CourtModelIOError("Court content_size_hw must be int64 (B,2).")
        if content_size_hw.device != call.images.device:
            raise CourtModelIOError(
                "Court content_size_hw and images must share device."
            )
        encoder = cast(CourtDINOv3Encoder, model.encoder)
        deepest = call.model_args[-1]
        patch_height, patch_width = deepest.shape[-2:]
        valid_height = torch.div(
            content_size_hw[:, 0] + encoder.patch_size - 1,
            encoder.patch_size,
            rounding_mode="floor",
        )
        valid_width = torch.div(
            content_size_hw[:, 1] + encoder.patch_size - 1,
            encoder.patch_size,
            rounding_mode="floor",
        )
        if bool(torch.any(valid_height > patch_height)) or bool(
            torch.any(valid_width > patch_width)
        ):
            raise CourtModelIOError(
                "Court content_size_hw exceeds the prepared DINO patch grid."
            )
        rows = torch.arange(patch_height, device=call.images.device).view(1, -1, 1)
        columns = torch.arange(patch_width, device=call.images.device).view(1, 1, -1)
        patch_valid_mask = (rows < valid_height[:, None, None]) & (
            columns < valid_width[:, None, None]
        )
        if bool(torch.any(~patch_valid_mask.flatten(1).any(dim=1))):
            raise CourtModelIOError(
                "Court content_size_hw must keep at least one DINO patch per sample."
            )
        return replace(call, model_args=(*call.model_args, patch_valid_mask))


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
        call = self._prepare_batch_model_call(batch)
        return ModelCall(args=call.model_args)

    def decode_output(
        self, output: CourtLogits | CourtModelOutput
    ) -> CourtLogits | CourtDecodedOutput:
        logits = output.dense_logits if isinstance(output, CourtModelOutput) else output
        self.validate_logits(logits)
        return MappingProxyType(dict(logits))

    def prepare_images(self, images: Tensor) -> CourtModelCall:
        return self._prepare_execution(
            _prepare_image_call(images, in_channels=self.spec.in_channels)
        )

    def _prepare_batch_model_call(
        self,
        batch: Mapping[str, object],
    ) -> CourtModelCall:
        call = self.prepare_images(_tensor(batch, "image"))
        boundary = self.execution_boundary
        if (
            isinstance(boundary, CourtDINOv3ExecutionBoundary)
            and boundary.patch_valid_mask_required
        ):
            image_size = self._validate_image_size(
                batch.get("image_size"),
                call=call,
            )
            content_size = self._validate_content_size(
                batch.get("content_size_hw"),
                image_size=image_size,
                call=call,
            )
            call = boundary.attach_patch_valid_mask(call, content_size)
        return call

    @staticmethod
    def _prepare_direct_execution(call: CourtModelCall) -> CourtModelCall:
        return call

    def prepare_training_batch(
        self,
        batch: Mapping[str, object],
    ) -> CourtTrainingCall:
        call = self._prepare_batch_model_call(batch)
        raw_targets = batch.get("targets")
        if not isinstance(raw_targets, Mapping):
            raise CourtModelIOError("Court training batch requires a targets mapping.")
        if set(raw_targets) != set(self.spec.target_bundle.kinds):
            raise CourtModelIOError(
                "Court batch target keys must exactly match the resolved bundle."
            )
        targets: dict[CourtTrainingTargetKind, object] = {}
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
        logits: CourtLogits | CourtModelOutput,
        call: CourtModelCall | None = None,
    ) -> None:
        if isinstance(logits, CourtModelOutput):
            logits = logits.dense_logits
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
        logits: CourtLogits | CourtModelOutput,
        call: CourtTrainingCall,
        *,
        progress_fraction: float | None = None,
    ) -> CourtTrainingResult | CourtPoseTrainingResult:
        _ = progress_fraction
        dense_logits = logits.dense_logits if isinstance(logits, CourtModelOutput) else logits
        self.validate_logits(dense_logits, call.model_call)
        raw_losses: dict[CourtTargetKind, Tensor] = {}
        configured_weights: dict[CourtTargetKind, Tensor] = {}
        effective_weights: dict[CourtTargetKind, Tensor] = {}
        weighted_losses: dict[CourtTargetKind, Tensor] = {}
        for kind in self.spec.target_bundle.kinds:
            value = dense_logits[kind]
            target = call.targets[kind]
            dense_weight = float(
                getattr(self.loss_config, "dense_weights", {}).get(kind, 1.0)
            )
            if kind == "kp":
                heatmap = cast(Mapping[str, Tensor], target)["heatmap"]
                raw_loss = self.kp_loss(value, heatmap)
            elif kind == "seg":
                labels = cast(Tensor, target)
                raw_loss = (
                    self.loss_config.seg_ce_weight * F.cross_entropy(value, labels)
                    + self.loss_config.seg_dice_weight * self.seg_dice(value, labels)
                )
            elif kind == "line":
                binary = cast(Tensor, target)
                pos_weight = value.new_tensor([self.loss_config.line_pos_weight])
                raw_loss = (
                    self.loss_config.line_bce_weight
                    * F.binary_cross_entropy_with_logits(
                        value,
                        binary,
                        pos_weight=pos_weight,
                    )
                    + self.loss_config.line_dice_weight
                    * self.line_dice(value, binary)
                )
            weight = raw_loss.new_tensor(dense_weight)
            raw_losses[kind] = raw_loss
            configured_weights[kind] = weight
            effective_weights[kind] = weight
            weighted_losses[kind] = raw_loss * weight
        raw_total = torch.stack(tuple(raw_losses.values())).sum()
        total = torch.stack(tuple(weighted_losses.values())).sum()
        frozen_raw_losses = MappingProxyType(raw_losses)
        frozen_weights = MappingProxyType(configured_weights)
        frozen_effective_weights = MappingProxyType(effective_weights)
        frozen_weighted_losses = MappingProxyType(weighted_losses)
        return CourtTrainingResult(
            loss=total,
            losses=frozen_weighted_losses,
            logits=MappingProxyType(dict(dense_logits)),
            raw_loss=raw_total,
            raw_losses=frozen_raw_losses,
            configured_weights=frozen_weights,
            effective_weights=frozen_effective_weights,
            weighted_losses=frozen_weighted_losses,
        )

    def test_payload(
        self,
        batch: Mapping[str, object],
        logits: CourtLogits,
    ) -> dict[str, object] | CourtPosePrediction:
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
    def _validate_image_size(value: object, *, call: CourtModelCall) -> Tensor:
        if (
            not isinstance(value, Tensor)
            or value.shape != (call.batch_size, 2)
            or value.dtype != torch.long
        ):
            raise CourtModelIOError("Court image_size must be int64 (B,2).")
        if value.device != call.images.device or bool(torch.any(value <= 0)) or bool(
            torch.any(value > value.new_tensor([call.height, call.width]))
        ):
            raise CourtModelIOError(
                "Court image_size is outside padded image bounds."
            )
        return value

    @staticmethod
    def _validate_content_size(
        value: object,
        *,
        image_size: Tensor,
        call: CourtModelCall,
    ) -> Tensor:
        if (
            not isinstance(value, Tensor)
            or value.shape != (call.batch_size, 2)
            or value.dtype != torch.long
        ):
            raise CourtModelIOError("Court content_size_hw must be int64 (B,2).")
        if value.device != call.images.device:
            raise CourtModelIOError(
                "Court content_size_hw and images must share device."
            )
        if bool(torch.any(value <= 0)) or bool(torch.any(value > image_size)):
            raise CourtModelIOError(
                "Court content_size_hw is outside image_size bounds."
            )
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


class CourtPoseModelIOAdapter(CourtModelIOAdapter):
    """Model-I/O seam for the optional typed raw pose branch.

    Dense-only callers continue using :class:`CourtModelIOAdapter`; this
    subclass adds pose target validation, direct pose terms, and the optional
    KP/pose consistency objective.  It intentionally accepts a structural
    output (``dense_logits`` plus ``pose.values``), so the adapter is agnostic
    to how the hierarchical model obtains its global feature.
    """

    def __init__(
        self,
        spec: CourtModelSpec,
        *,
        loss_config: object,
        execution_boundary: CourtModelExecutionBoundary | None = None,
    ) -> None:
        dense_config = cast(CourtLossConfig, getattr(loss_config, "dense", loss_config))
        super().__init__(spec, loss_config=dense_config, execution_boundary=execution_boundary)
        self.pose_loss_config: CourtLossConfig = cast(CourtLossConfig, loss_config)
        consistency = self.pose_loss_config.consistency
        self.consistency_instrumented = bool(
            consistency is not None and getattr(consistency, "enabled", False)
        )

    @property
    def model_type(self) -> type[nn.Module]:
        return cast("type[nn.Module]", CourtHierarchicalModel)

    def validate_output(
        self,
        output: object,
        *,
        call: CourtModelCall | None = None,
    ) -> CourtModelOutput:
        if not isinstance(output, (CourtModelOutput, Mapping)):
            dense = getattr(output, "dense_logits", None)
            if dense is None:
                dense = getattr(output, "dense_outputs", None)
            pose = getattr(output, "pose", None)
            if pose is None:
                pose_values = getattr(output, "pose_raw", None)
                if pose_values is None:
                    pose_values = getattr(output, "raw_pose", None)
                if isinstance(pose_values, CourtRawPoseOutput):
                    pose = pose_values
                    pose_values = None
                if isinstance(pose_values, Tensor):
                    pose = CourtRawPoseOutput(pose_values)
            if isinstance(dense, Mapping):
                output = CourtModelOutput(dense_logits=dense, pose=pose)
        if isinstance(output, Mapping):
            output = CourtModelOutput(dense_logits=output)
        if not isinstance(output, CourtModelOutput):
            raise CourtModelIOError("Court model output must be CourtModelOutput.")
        self.validate_logits(output.dense_logits, call)
        if output.pose is None:
            raise CourtModelIOError("Pose-enabled adapter requires raw pose output.")
        if call is not None and output.pose.values.shape[0] != call.batch_size:
            raise CourtModelIOError("Court pose output batch must match model call.")
        return output

    def decode_output(self, output: object) -> CourtDecodedOutput:
        checked = self.validate_output(output)
        assert checked.pose is not None
        return CourtDecodedOutput(
            pose=decode_pose10d_strict(checked.pose.values),
            dense_logits=checked.dense_logits,
        )

    def prepare_training_batch(self, batch: Mapping[str, object]) -> CourtTrainingCall:
        call = super().prepare_training_batch(batch)
        pose_enabled = bool(getattr(getattr(self.pose_loss_config, "pose", None), "enabled", True))
        pose_value = batch.get("pose_target")
        if pose_enabled or pose_value is not None:
            pose_target = self._validate_pose_target(
                pose_value, batch_size=call.model_call.batch_size
            )
            kp_target = call.targets.get("kp")
            consistency = getattr(self.pose_loss_config, "consistency", None)
            consistency_enabled = bool(getattr(consistency, "enabled", False))
            image_size: Tensor | None = None
            if consistency_enabled or isinstance(batch.get("image_size"), Tensor):
                image_size = self._validate_image_size(
                    batch.get("image_size"), call=call.model_call
                )
            raw_content_size = batch.get("content_size_hw")
            if consistency_enabled and raw_content_size is None:
                raise CourtModelIOError(
                    "Consistency requires a typed content_size_hw target."
                )
            if raw_content_size is not None:
                if image_size is None:
                    raise CourtModelIOError(
                        "Court content_size_hw requires a typed image_size target."
                    )
                self._validate_content_size(
                    raw_content_size,
                    image_size=image_size,
                    call=call.model_call,
                )
            if consistency_enabled and not isinstance(kp_target, Mapping):
                raise CourtModelIOError("Consistency requires a KP target.")
            if isinstance(kp_target, Mapping) and "physical_indices" in kp_target:
                physical = cast(Tensor, kp_target["physical_indices"])
                if consistency_enabled and physical.shape[1:] != (14, 1):
                    raise CourtModelIOError("Pose KP target must be singleton (B,14,1).")
                if consistency_enabled and not torch.equal(
                    physical[:, :, 0], pose_target.semantic_to_physical
                ):
                    raise CourtModelIOError("KP physical order disagrees with pose authority.")
            enriched_targets: dict[CourtTrainingTargetKind, object] = {
                **call.targets,
                "pose": pose_target,
            }
            if image_size is not None:
                enriched_targets["image_size"] = image_size
            return CourtTrainingCall(
                model_call=call.model_call,
                targets=MappingProxyType(enriched_targets),
                batch=call.batch,
            )
        return call

    def training_result(
        self,
        output: object,
        call: CourtTrainingCall,
        *,
        progress_fraction: float | None = None,
    ) -> CourtPoseTrainingResult:
        checked = self.validate_output(output, call=call.model_call)
        pose_target = call.targets.get("pose")
        image_size = call.targets.get("image_size")
        if not isinstance(pose_target, CourtPoseTargetBatch):
            raise CourtModelIOError("Pose training call lacks a typed pose target.")
        dense_targets: dict[CourtTrainingTargetKind, object] = {
            kind: call.targets[kind] for kind in self.spec.target_bundle.kinds
        }
        dense_call = CourtTrainingCall(call.model_call, MappingProxyType(dense_targets), call.batch)
        dense_result = cast(
            CourtTrainingResult,
            super().training_result(checked.dense_logits, dense_call),
        )
        direct_dense = dense_result.loss
        assert checked.pose is not None
        decoded_pose = decode_pose10d_strict(checked.pose.values)
        pose_config = self.pose_loss_config.pose
        pose_enabled = pose_config.enabled
        raw_pose_losses = pose_losses(decoded_pose, pose_target)
        pose_loss_map = MappingProxyType(raw_pose_losses if pose_enabled else {})
        weighted_pose_losses: dict[CourtPoseLossKind, Tensor] = {}
        pose_configured_weights: dict[CourtPoseLossKind, Tensor] = {}
        pose_effective_weights: dict[CourtPoseLossKind, Tensor] = {}
        direct_pose = direct_dense.new_zeros(())
        if pose_enabled:
            configured_pose_weight_values: dict[CourtPoseLossKind, float] = {
                "pose_translation": float(pose_config.translation_weight),
                "pose_rotation": float(pose_config.rotation_weight),
                "pose_focal": float(pose_config.focal_weight),
            }
            pose_configured_weights = {
                name: loss.new_tensor(configured_pose_weight_values[name])
                for name, loss in raw_pose_losses.items()
            }
            pose_effective_weights = {
                name: loss.new_tensor(configured_pose_weight_values[name])
                for name, loss in raw_pose_losses.items()
            }
            weighted_pose_losses = {
                name: loss * pose_effective_weights[name]
                for name, loss in raw_pose_losses.items()
            }
            direct_pose = torch.stack(tuple(weighted_pose_losses.values())).sum()
        consistency_result: CourtConsistencyResult | None = None
        weighted_auxiliary = direct_dense.new_zeros(())
        consistency = self.pose_loss_config.consistency
        if consistency.enabled:
            if not isinstance(image_size, Tensor):
                raise CourtModelIOError("Consistency requires a typed image_size target.")
            if progress_fraction is None:
                raise CourtModelIOError("Enabled consistency requires progress_fraction.")
            kp_target = cast(Mapping[str, Tensor], dense_targets.get("kp"))
            kp_logits = checked.dense_logits.get("kp")
            if kp_logits is None or kp_target["points_xy"].shape[2:] != (1, 2):
                raise CourtModelIOError("Consistency requires singleton KP14 supervision.")
            raw_content_size = call.batch.get("content_size_hw")
            if raw_content_size is None:
                raise CourtModelIOError(
                    "Consistency requires a typed content_size_hw target."
                )
            content_size = self._validate_content_size(
                raw_content_size,
                image_size=image_size,
                call=call.model_call,
            )
            valid_mask = self._valid_region_mask(
                image_size=image_size,
                content_size_hw=content_size,
                logits=kp_logits,
            )
            normalized_dense = heatmaps_to_soft_argmax(
                kp_logits,
                temperature=float(consistency.temperature),
                valid_mask=valid_mask,
            )
            padded_height, padded_width = kp_logits.shape[-2:]
            dense_points_xy = normalized_dense * kp_logits.new_tensor(
                [float(padded_width - 1), float(padded_height - 1)]
            )
            canonical_points = canonical_semantic_court_points_batched(
                pose_target.semantic_to_physical,
                dtype=decoded_pose.translation_m.dtype,
                device=decoded_pose.translation_m.device,
            )
            projection = project_predicted_canonical_points(
                decoded_pose, canonical_points, pose_target.intrinsics[:, :2, 2]
            )
            consistency_loss = keypoint_pose_consistency_loss(
                dense_points_xy,
                projection.points_xy,
                projection.depth_m,
                content_size,
                kp_target["point_visible"].squeeze(-1),
                huber_delta=float(consistency.huber_delta),
                min_depth_m=float(consistency.min_depth_m),
                depth_scale_m=float(consistency.depth_scale_m),
                cheirality_weight=float(consistency.cheirality_weight),
                gradient_flow=consistency.gradient_flow,
            )
            effective = consistency_effective_weight(
                weight=float(consistency.weight),
                warmup_fraction=float(consistency.warmup_fraction),
                progress=float(progress_fraction),
            )
            configured_tensor = consistency_loss.auxiliary.new_tensor(
                float(consistency.weight)
            )
            effective_tensor = consistency_loss.auxiliary.new_tensor(effective)
            weighted_auxiliary = consistency_loss.auxiliary * effective_tensor
            consistency_result = CourtConsistencyResult(
                coordinate_loss=consistency_loss.coordinate,
                cheirality_loss=consistency_loss.cheirality,
                auxiliary_loss=consistency_loss.auxiliary,
                weighted_auxiliary_loss=weighted_auxiliary,
                configured_weight=configured_tensor,
                effective_weight=effective_tensor,
                visible_point_count=consistency_loss.visible_point_count,
                mean_distance_px=consistency_loss.mean_distance_px,
                invalid_depth_rate=consistency_loss.invalid_depth_fraction,
                dense_points_xy=dense_points_xy,
                pose_points_xy=projection.points_xy,
                pose_depth_m=projection.depth_m,
            )
        return CourtPoseTrainingResult(
            loss=direct_dense + direct_pose + weighted_auxiliary,
            raw_dense_loss=dense_result.raw_loss,
            direct_dense_loss=direct_dense,
            direct_pose_loss=direct_pose,
            raw_dense_losses=dense_result.raw_losses,
            dense_losses=dense_result.losses,
            dense_configured_weights=dense_result.configured_weights,
            dense_effective_weights=dense_result.effective_weights,
            weighted_dense_losses=dense_result.weighted_losses,
            pose_losses=pose_loss_map,
            weighted_pose_losses=MappingProxyType(weighted_pose_losses),
            pose_configured_weights=MappingProxyType(pose_configured_weights),
            pose_effective_weights=MappingProxyType(pose_effective_weights),
            consistency=consistency_result,
            output=checked,
            decoded_pose=decoded_pose,
        )

    @staticmethod
    def _valid_region_mask(
        *,
        image_size: Tensor,
        logits: Tensor,
        content_size_hw: Tensor | None = None,
    ) -> Tensor:
        batch_size, channels, height, width = logits.shape
        if image_size.shape != (batch_size, 2) or image_size.dtype != torch.long:
            raise CourtModelIOError("Court image_size must be int64 (B,2).")
        if image_size.device != logits.device:
            raise CourtModelIOError("Court image_size and logits must share device.")
        if bool(torch.any(image_size <= 0)) or bool(torch.any(image_size > image_size.new_tensor([height, width]))):
            raise CourtModelIOError("Court image_size is outside logits bounds.")
        valid_size = image_size
        if content_size_hw is not None:
            if (
                content_size_hw.shape != (batch_size, 2)
                or content_size_hw.dtype != torch.long
            ):
                raise CourtModelIOError("Court content_size_hw must be int64 (B,2).")
            if content_size_hw.device != logits.device:
                raise CourtModelIOError(
                    "Court content_size_hw and logits must share device."
                )
            if bool(torch.any(content_size_hw <= 0)) or bool(
                torch.any(content_size_hw > image_size)
            ):
                raise CourtModelIOError(
                    "Court content_size_hw is outside image_size bounds."
                )
            valid_size = content_size_hw
        valid_y = torch.arange(height, device=logits.device)[None, :] < valid_size[:, 0:1]
        valid_x = torch.arange(width, device=logits.device)[None, :] < valid_size[:, 1:2]
        return (valid_y[:, :, None] & valid_x[:, None, :])[:, None].expand(batch_size, channels, height, width)

    def test_payload(self, batch: Mapping[str, object], output: object) -> CourtPosePrediction:
        checked = self.validate_output(output)
        assert checked.pose is not None
        dense: dict[CourtTargetKind, object] = {}
        for kind, value in checked.dense_logits.items():
            if kind == "kp":
                flat = value.flatten(2)
                index = flat.argmax(dim=-1)
                height, width = value.shape[-2:]
                dense[kind] = {
                    "keypoints_normalized": torch.stack(((index % width).to(value.dtype) / float(max(width - 1, 1)), torch.div(index, width, rounding_mode="floor").to(value.dtype) / float(max(height - 1, 1))), dim=-1).unsqueeze(2),
                    "scores": torch.sigmoid(flat.amax(dim=-1)).unsqueeze(2),
                    "valid": torch.ones((*index.shape, 1), dtype=torch.bool, device=value.device),
                    "heatmaps": value,
                }
            elif kind == "seg":
                dense[kind] = {"mask": value.argmax(dim=1), "logits": value}
            else:
                dense[kind] = {"probability": torch.sigmoid(value), "logits": value}
        _ = batch
        return CourtPosePrediction(pose=decode_pose10d_strict(checked.pose.values), dense=MappingProxyType(dense))

    @staticmethod
    def _validate_pose_target(value: object, *, batch_size: int) -> CourtPoseTargetBatch:
        expected = {"translation_m", "rotation", "log_focal", "intrinsics", "semantic_to_physical", "raw_pose10d"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CourtModelIOError("Court pose target fields changed.")
        target = CourtPoseTargetBatch(**{name: _mapping_tensor(value, name) for name in expected})
        if target.translation_m.shape != (batch_size, 3) or target.rotation.shape != (batch_size, 3, 3) or target.log_focal.shape != (batch_size,) or target.intrinsics.shape != (batch_size, 3, 3):
            raise CourtModelIOError("Court pose target batch shapes are invalid.")
        if target.semantic_to_physical.shape != (batch_size, 14) or target.semantic_to_physical.dtype != torch.long:
            raise CourtModelIOError("Court pose semantic order must be int64 (B,14).")
        expected_physical = torch.arange(14, device=target.semantic_to_physical.device).expand(batch_size, 14)
        if not torch.equal(torch.sort(target.semantic_to_physical, dim=1).values, expected_physical):
            raise CourtModelIOError("Court pose semantic order must be a 0..13 bijection.")
        if target.raw_pose10d.shape != (batch_size, 10):
            raise CourtModelIOError("Court raw pose target must be (B,10).")
        for name, tensor in (("translation", target.translation_m), ("rotation", target.rotation), ("log-focal", target.log_focal), ("intrinsics", target.intrinsics), ("raw pose", target.raw_pose10d)):
            _require_finite(tensor, name=f"Court pose target {name}")
        validate_proper_rotation(target.rotation)
        for intrinsics in target.intrinsics:
            validate_square_intrinsics(intrinsics)
        reconstructed = torch.cat((target.translation_m, target.rotation[:, :2].reshape(batch_size, 6), target.log_focal.unsqueeze(-1)), dim=-1)
        if not bool(torch.allclose(target.raw_pose10d, reconstructed, atol=1.0e-6, rtol=0.0)):
            raise CourtModelIOError("Court raw pose target order/content changed.")
        return target


def _prepare_image_call(images: Tensor, *, in_channels: int) -> CourtModelCall:
    if images.ndim != 4 or images.dtype != torch.float32:
        raise CourtModelIOError(
            "Court images must be float32 with shape (B,3,H,W)."
        )
    batch_size, channels, height, width = images.shape
    if (
        batch_size <= 0
        or height <= 0
        or width <= 0
        or channels != in_channels
    ):
        raise CourtModelIOError("Court image dimensions/channels are invalid.")
    _require_finite(images, name="Court images")
    lower = images.new_tensor(_NORMALIZED_IMAGE_MIN).view(1, 3, 1, 1)
    upper = images.new_tensor(_NORMALIZED_IMAGE_MAX).view(1, 3, 1, 1)
    if bool(torch.any((images < lower) | (images > upper))):
        raise CourtModelIOError(
            "Court images must be ImageNet-normalized RGB values from [0,1]."
        )
    return CourtModelCall(
        images=images,
        model_args=(images,),
        batch_size=batch_size,
        height=height,
        width=width,
    )


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
    "CourtPoseModelIOAdapter",
]
