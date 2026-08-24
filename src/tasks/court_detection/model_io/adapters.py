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
    CourtQueryLossConfig,
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
    CourtDecodedPrediction,
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtLogits,
    CourtModelCall,
    CourtModelIOError,
    CourtModelSpec,
    CourtPoseTargetBatch,
    CourtQueryConsistencyResult,
    CourtQueryDecodedOutput,
    CourtQueryModelCall,
    CourtQueryModelSpec,
    CourtQueryPrediction,
    CourtQueryRawOutput,
    CourtQueryTrainingCall,
    CourtQueryTrainingResult,
    CourtSegmentationPrediction,
    CourtTrainingCall,
    CourtTrainingResult,
)
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.tasks.court_detection.models.query_encoder.backbone import (
    CourtQueryDINOv3Backbone,
)
from src.tasks.court_detection.models.query_encoder.contracts import PatchTokenBatch
from src.tasks.court_detection.models.query_encoder.model import CourtQueryEncoderModel
from src.tasks.court_detection.training.losses import (
    BinaryDiceLoss,
    DiceLoss,
    consistency_effective_weight,
    query_keypoint_pose_consistency_loss,
    query_pose_losses,
)
from src.utils.data.heatmaps import (
    heatmaps_to_peaks,
    heatmaps_to_soft_argmax,
    refine_peaks_log_parabolic,
)
from src.utils.models.loading import require_dinov3_patch_tokens

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


class CourtQueryModelExecutionBoundary(Protocol):
    def bind_model(self, model: CourtQueryEncoderModel) -> None: ...

    def prepare(self, call: CourtModelCall) -> CourtQueryModelCall: ...


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


def _run_query_dinov3(
    backbone: CourtQueryDINOv3Backbone,
    images: Tensor,
) -> object:
    return backbone.execute_patch_features(images)


def _run_frozen_query_dinov3(
    backbone: CourtQueryDINOv3Backbone,
    images: Tensor,
) -> object:
    with torch.no_grad():
        return backbone.execute_patch_features(images)


class CourtQueryDINOv3ExecutionBoundary:
    """Extract and validate patch-only DINO output before query-model forward."""

    def __init__(self, *, frozen_backbone: bool) -> None:
        self._model_ref: weakref.ReferenceType[CourtQueryEncoderModel] | None = None
        self._backbone_executor = (
            _run_frozen_query_dinov3 if frozen_backbone else _run_query_dinov3
        )

    def bind_model(self, model: CourtQueryEncoderModel) -> None:
        if not isinstance(model.backbone, CourtQueryDINOv3Backbone):
            raise CourtModelIOError(
                "Court query DINO boundary requires CourtQueryDINOv3Backbone."
            )
        if model.backbone.frozen_execution != (
            self._backbone_executor is _run_frozen_query_dinov3
        ):
            raise CourtModelIOError(
                "Court query DINO boundary frozen mode disagrees with its backbone."
            )
        self._model_ref = weakref.ref(model)

    def prepare(self, call: CourtModelCall) -> CourtQueryModelCall:
        if self._model_ref is None or (model := self._model_ref()) is None:
            raise CourtModelIOError(
                "Court query DINO execution boundary is not bound to its model."
            )
        backbone = model.backbone
        patch_size = backbone.patch_size
        pad_h = (-call.height) % patch_size
        pad_w = (-call.width) % patch_size
        padded = (
            call.images
            if pad_h == 0 and pad_w == 0
            else F.pad(call.images, (0, pad_w, 0, pad_h), mode="replicate")
        )
        padded_hw = (int(padded.shape[-2]), int(padded.shape[-1]))
        grid_hw = (padded_hw[0] // patch_size, padded_hw[1] // patch_size)
        expected_tokens = grid_hw[0] * grid_hw[1]
        raw_output = self._backbone_executor(backbone, padded)
        try:
            tokens = require_dinov3_patch_tokens(
                raw_output,
                expected_batch_size=call.batch_size,
                expected_embed_dim=backbone.embed_dim,
                expected_num_tokens=expected_tokens,
                context="Court query DINOv3 forward_features",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CourtModelIOError(str(error)) from error
        if not tokens.is_floating_point():
            raise CourtModelIOError("Court query DINO patch tokens must be floating.")
        if tokens.device != padded.device:
            raise CourtModelIOError(
                "Court query DINO patch tokens must remain on the image device."
            )
        _require_finite(tokens, name="Court query DINO patch tokens")
        patch_batch = PatchTokenBatch(
            tokens=tokens,
            original_hw=(call.height, call.width),
            padded_hw=padded_hw,
            padding_hw=(pad_h, pad_w),
            grid_hw=grid_hw,
            patch_size=patch_size,
        )
        grid_tensor = torch.tensor(grid_hw, device=tokens.device, dtype=torch.long)
        padded_tensor = torch.tensor(
            padded_hw,
            device=tokens.device,
            dtype=torch.long,
        )
        return CourtQueryModelCall(
            images=call.images,
            patch_batch=patch_batch,
            model_args=(call.images, tokens, grid_tensor, padded_tensor),
            batch_size=call.batch_size,
            height=call.height,
            width=call.width,
        )


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
        return self._prepare_execution(
            _prepare_image_call(images, in_channels=self.spec.in_channels)
        )

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


class CourtQueryModelIOAdapter(nn.Module):
    """Typed query pose/dense adapter with singleton KP and weighted losses."""

    def __init__(
        self,
        spec: CourtQueryModelSpec,
        *,
        execution_boundary: CourtQueryModelExecutionBoundary,
        loss_config: CourtQueryLossConfig,
    ) -> None:
        super().__init__()
        if spec.in_channels != 3:
            raise CourtModelIOError(
                "Court query model input must use exactly three RGB channels."
            )
        if spec.short_side <= 0:
            raise CourtModelIOError(
                "Court query preprocessing short_side must be positive."
            )
        self.spec = spec
        self.execution_boundary = execution_boundary
        self.loss_config = loss_config
        kp_spec = spec.target_bundle.targets.get("kp")
        if (
            kp_spec is None
            or kp_spec.output_channels != 14
            or kp_spec.schema
            != "synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1"
        ):
            raise CourtModelIOError(
                "Court query model requires the V3 target-court singleton KP14 bundle."
            )
        self.dense_adapter = CourtModelIOAdapter(
            CourtModelSpec(
                target_bundle=spec.target_bundle,
                in_channels=spec.in_channels,
                short_side=spec.short_side,
            ),
            loss_config=loss_config.dense,
        )
        self._consistency_instrumented: bool = bool(
            loss_config.consistency.enabled
            or loss_config.name == "query_direct_all_v1"
        )

    @property
    def consistency_instrumented(self) -> bool:
        """Whether this explicit route emits cross-branch loss/metrics."""
        return self._consistency_instrumented

    @property
    def model_type(self) -> type[nn.Module]:
        return cast("type[nn.Module]", CourtQueryEncoderModel)

    def validate_model_pair(self, model: nn.Module) -> None:
        if not isinstance(model, CourtQueryEncoderModel):
            raise CourtModelIOError(
                "Court query model-I/O requires CourtQueryEncoderModel, got "
                f"{type(model).__name__}."
            )
        if model.in_channels != self.spec.in_channels:
            raise CourtModelIOError(
                "Court query model input channels disagree with adapter."
            )
        if model.target_bundle_spec != self.spec.target_bundle:
            raise CourtModelIOError(
                "Court query model heads disagree with target bundle."
            )
        self.execution_boundary.bind_model(model)

    def prepare_images(self, images: Tensor) -> CourtQueryModelCall:
        call = _prepare_image_call(images, in_channels=self.spec.in_channels)
        return self.execution_boundary.prepare(call)

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        call = self.prepare_images(_tensor(batch, "image"))
        return ModelCall(args=call.model_args)

    def validate_output(
        self,
        output: CourtQueryRawOutput,
        *,
        call: CourtQueryModelCall | None = None,
    ) -> None:
        if not isinstance(output, CourtQueryRawOutput):
            raise CourtModelIOError(
                "Court query model output must be CourtQueryRawOutput."
            )
        logits = output.dense_logits
        if set(logits) != set(self.spec.target_bundle.kinds):
            raise CourtModelIOError(
                "Court query dense output keys must exactly match the target bundle."
            )
        if call is not None and output.pose.values.shape[0] != call.batch_size:
            raise CourtModelIOError(
                "Court query pose output batch size must match its model call."
            )
        for kind, target_spec in self.spec.target_bundle.targets.items():
            value = logits[kind]
            if value.ndim != 4 or value.shape[1] != target_spec.output_channels:
                raise CourtModelIOError(
                    f"Court query {kind} logits have invalid rank/channels."
                )
            if call is not None and value.shape != (
                call.batch_size,
                target_spec.output_channels,
                call.height,
                call.width,
            ):
                raise CourtModelIOError(
                    f"Court query {kind} logits must preserve input batch/H/W."
                )
            _require_finite(value, name=f"Court query {kind} logits")

    def decode_output(self, output: CourtQueryRawOutput) -> CourtQueryDecodedOutput:
        self.validate_output(output)
        return CourtQueryDecodedOutput(
            pose=decode_pose10d_strict(output.pose.values),
            dense_logits=output.dense_logits,
        )

    def prepare_training_batch(
        self,
        batch: Mapping[str, object],
    ) -> CourtQueryTrainingCall:
        query_call = self.prepare_images(_tensor(batch, "image"))
        dense_call = self.dense_adapter.prepare_training_batch(batch)
        kp_target = cast(Mapping[str, Tensor], dense_call.targets["kp"])
        if kp_target["points_xy"].shape != (
            query_call.batch_size,
            14,
            1,
            2,
        ):
            raise CourtModelIOError(
                "Court query KP target must be singleton with shape (B,14,1,2)."
            )
        pose_target = self._validate_pose_target(
            batch.get("pose_target"),
            batch_size=query_call.batch_size,
        )
        image_size = self._validate_image_size(
            batch.get("image_size"),
            call=query_call,
        )
        if not torch.equal(
            kp_target["physical_indices"][:, :, 0],
            pose_target.semantic_to_physical,
        ):
            raise CourtModelIOError(
                "Court query KP14 physical order disagrees with pose authority."
            )
        return CourtQueryTrainingCall(
            model_call=query_call,
            dense_targets=dense_call.targets,
            pose_target=pose_target,
            image_size=image_size,
            batch=MappingProxyType(dict(batch)),
        )

    def training_result(
        self,
        output: CourtQueryRawOutput,
        call: CourtQueryTrainingCall,
        *,
        progress_fraction: float | None = None,
    ) -> CourtQueryTrainingResult:
        self.validate_output(output, call=call.model_call)
        dense_call = CourtTrainingCall(
            model_call=CourtModelCall(
                images=call.model_call.images,
                model_args=(call.model_call.images,),
                batch_size=call.model_call.batch_size,
                height=call.model_call.height,
                width=call.model_call.width,
            ),
            targets=call.dense_targets,
            batch=call.batch,
        )
        dense_result = self.dense_adapter.training_result(
            output.dense_logits,
            dense_call,
        )
        decoded_pose = decode_pose10d_strict(output.pose.values)
        pose_losses = query_pose_losses(decoded_pose, call.pose_target)
        weighted_dense_terms = [
            self.loss_config.dense_weights[kind] * loss
            for kind, loss in dense_result.losses.items()
        ]
        direct_dense_loss = torch.stack(weighted_dense_terms).sum()
        direct_pose_loss = direct_dense_loss.new_zeros(())
        if self.loss_config.pose.enabled:
            direct_pose_loss = torch.stack(
                (
                    self.loss_config.pose.translation_weight
                    * pose_losses["pose_translation"],
                    self.loss_config.pose.rotation_weight
                    * pose_losses["pose_rotation"],
                    self.loss_config.pose.focal_weight * pose_losses["pose_focal"],
                )
            ).sum()
        consistency_result: CourtQueryConsistencyResult | None = None
        weighted_auxiliary = direct_dense_loss.new_zeros(())
        if self.consistency_instrumented:
            if progress_fraction is None:
                raise CourtModelIOError(
                    "Instrumented Court query loss requires an explicit progress fraction."
                )
            kp_logits = output.dense_logits["kp"]
            valid_mask = self._valid_region_mask(
                image_size=call.image_size,
                logits=kp_logits,
            )
            normalized_dense = heatmaps_to_soft_argmax(
                kp_logits,
                temperature=self.loss_config.consistency.temperature,
                valid_mask=valid_mask,
            )
            padded_height, padded_width = kp_logits.shape[-2:]
            pixel_scale = kp_logits.new_tensor(
                [float(padded_width - 1), float(padded_height - 1)]
            )
            dense_points_xy = normalized_dense * pixel_scale
            canonical_points = canonical_semantic_court_points_batched(
                call.pose_target.semantic_to_physical,
                dtype=decoded_pose.translation_m.dtype,
                device=decoded_pose.translation_m.device,
            )
            projection = project_predicted_canonical_points(
                decoded_pose,
                canonical_points,
                call.pose_target.intrinsics[:, :2, 2],
            )
            kp_target = cast(Mapping[str, Tensor], call.dense_targets["kp"])
            consistency = query_keypoint_pose_consistency_loss(
                dense_points_xy,
                projection.points_xy,
                projection.depth_m,
                call.image_size,
                kp_target["point_visible"].squeeze(-1),
                huber_delta=self.loss_config.consistency.huber_delta,
                min_depth_m=self.loss_config.consistency.min_depth_m,
                depth_scale_m=self.loss_config.consistency.depth_scale_m,
                cheirality_weight=self.loss_config.consistency.cheirality_weight,
                gradient_flow=self.loss_config.consistency.gradient_flow,
            )
            effective_weight = consistency_effective_weight(
                weight=self.loss_config.consistency.weight,
                warmup_fraction=self.loss_config.consistency.warmup_fraction,
                progress=progress_fraction,
            )
            effective_weight_tensor = consistency.auxiliary.new_tensor(
                effective_weight
            )
            weighted_auxiliary = consistency.auxiliary * effective_weight_tensor
            consistency_result = CourtQueryConsistencyResult(
                coordinate_loss=consistency.coordinate,
                cheirality_loss=consistency.cheirality,
                auxiliary_loss=consistency.auxiliary,
                weighted_auxiliary_loss=weighted_auxiliary,
                effective_weight=effective_weight_tensor,
                visible_point_count=consistency.visible_point_count,
                mean_distance_px=consistency.mean_distance_px,
                invalid_depth_rate=consistency.invalid_depth_fraction,
                dense_points_xy=dense_points_xy,
                pose_points_xy=projection.points_xy,
                pose_depth_m=projection.depth_m,
            )
        total = direct_dense_loss + direct_pose_loss + weighted_auxiliary
        return CourtQueryTrainingResult(
            loss=total,
            direct_dense_loss=direct_dense_loss,
            direct_pose_loss=direct_pose_loss,
            dense_losses=dense_result.losses,
            pose_losses=MappingProxyType(
                pose_losses if self.loss_config.pose.enabled else {}
            ),
            consistency=consistency_result,
            output=output,
            decoded_pose=decoded_pose,
        )

    @staticmethod
    def _valid_region_mask(*, image_size: Tensor, logits: Tensor) -> Tensor:
        """Build the exact per-channel mask from strict unpadded ``(H,W)``."""
        batch_size, channels, height, width = logits.shape
        if image_size.shape != (batch_size, 2) or image_size.dtype != torch.long:
            raise CourtModelIOError(
                "Court image_size must be int64 with shape (B,2) in (H,W) order."
            )
        if image_size.device != logits.device:
            raise CourtModelIOError(
                "Court image_size and query logits must be on the same device."
            )
        if bool(torch.any(image_size <= 0)):
            raise CourtModelIOError("Court image_size values must be positive.")
        bounds = image_size.new_tensor([height, width])
        if bool(torch.any(image_size > bounds)):
            raise CourtModelIOError(
                "Court image_size cannot exceed the query logits spatial bounds."
            )
        valid_y = torch.arange(height, device=logits.device)[None, :] < image_size[
            :, 0:1
        ]
        valid_x = torch.arange(width, device=logits.device)[None, :] < image_size[
            :, 1:2
        ]
        spatial = valid_y[:, :, None] & valid_x[:, None, :]
        return spatial[:, None].expand(batch_size, channels, height, width)

    @staticmethod
    def _validate_image_size(
        value: object,
        *,
        call: CourtQueryModelCall,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise CourtModelIOError("Court batch image_size must be a Tensor.")
        if value.shape != (call.batch_size, 2) or value.dtype != torch.long:
            raise CourtModelIOError(
                "Court batch image_size must be int64 (B,2) in (H,W) order."
            )
        if value.device != call.images.device:
            raise CourtModelIOError(
                "Court batch image_size and images must be on the same device."
            )
        if bool(torch.any(value <= 0)):
            raise CourtModelIOError("Court batch image_size values must be positive.")
        bounds = value.new_tensor([call.height, call.width])
        if bool(torch.any(value > bounds)):
            raise CourtModelIOError(
                "Court batch image_size cannot exceed the padded image bounds."
            )
        return value

    def test_payload(
        self,
        batch: Mapping[str, object],
        output: CourtQueryRawOutput,
    ) -> CourtQueryPrediction:
        self.validate_output(output)
        dense: dict[CourtTargetKind, object] = {}
        for kind, value in output.dense_logits.items():
            if kind == "kp":
                flat = value.flatten(2)
                index = flat.argmax(dim=-1)
                height, width = value.shape[-2:]
                coordinates = torch.stack(
                    (
                        (index % width).to(dtype=value.dtype)
                        / float(max(width - 1, 1)),
                        torch.div(index, width, rounding_mode="floor").to(
                            dtype=value.dtype
                        )
                        / float(max(height - 1, 1)),
                    ),
                    dim=-1,
                ).unsqueeze(2)
                scores = torch.sigmoid(flat.amax(dim=-1)).unsqueeze(2)
                dense[kind] = {
                    "keypoints_normalized": coordinates,
                    "scores": scores,
                    "valid": torch.ones_like(scores, dtype=torch.bool),
                    "heatmaps": value,
                }
            elif kind == "seg":
                dense[kind] = {"mask": value.argmax(dim=1), "logits": value}
            else:
                dense[kind] = {
                    "probability": torch.sigmoid(value),
                    "logits": value,
                }
        _ = batch
        return CourtQueryPrediction(
            pose=decode_pose10d_strict(output.pose.values),
            dense=MappingProxyType(dense),
        )

    @staticmethod
    def _validate_pose_target(
        value: object,
        *,
        batch_size: int,
    ) -> CourtPoseTargetBatch:
        expected = {
            "translation_m",
            "rotation",
            "log_focal",
            "intrinsics",
            "semantic_to_physical",
            "raw_pose10d",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CourtModelIOError("Court query pose target fields changed.")
        target = CourtPoseTargetBatch(
            translation_m=_mapping_tensor(value, "translation_m"),
            rotation=_mapping_tensor(value, "rotation"),
            log_focal=_mapping_tensor(value, "log_focal"),
            intrinsics=_mapping_tensor(value, "intrinsics"),
            semantic_to_physical=_mapping_tensor(value, "semantic_to_physical"),
            raw_pose10d=_mapping_tensor(value, "raw_pose10d"),
        )
        if target.translation_m.shape != (batch_size, 3):
            raise CourtModelIOError("Court pose translation target must be (B,3).")
        if target.rotation.shape != (batch_size, 3, 3):
            raise CourtModelIOError("Court pose rotation target must be (B,3,3).")
        if target.log_focal.shape != (batch_size,):
            raise CourtModelIOError("Court pose log-focal target must be (B,).")
        if target.intrinsics.shape != (batch_size, 3, 3):
            raise CourtModelIOError("Court pose intrinsics target must be (B,3,3).")
        if (
            target.semantic_to_physical.shape != (batch_size, 14)
            or target.semantic_to_physical.dtype != torch.long
        ):
            raise CourtModelIOError(
                "Court pose semantic-to-physical target must be int64 (B,14)."
            )
        expected_physical = torch.arange(
            14,
            device=target.semantic_to_physical.device,
        ).expand(batch_size, 14)
        if not torch.equal(
            torch.sort(target.semantic_to_physical, dim=1).values,
            expected_physical,
        ):
            raise CourtModelIOError(
                "Court pose semantic-to-physical target must be a 0..13 bijection."
            )
        if target.raw_pose10d.shape != (batch_size, 10):
            raise CourtModelIOError("Court raw pose target must be (B,10).")
        for name, tensor in (
            ("translation", target.translation_m),
            ("rotation", target.rotation),
            ("log-focal", target.log_focal),
            ("intrinsics", target.intrinsics),
            ("raw pose", target.raw_pose10d),
        ):
            _require_finite(tensor, name=f"Court pose target {name}")
        validate_proper_rotation(target.rotation)
        for intrinsics in target.intrinsics:
            validate_square_intrinsics(intrinsics)
        reconstructed = torch.cat(
            (
                target.translation_m,
                target.rotation[:, :2].reshape(batch_size, 6),
                target.log_focal.unsqueeze(-1),
            ),
            dim=-1,
        )
        if not bool(
            torch.allclose(
                target.raw_pose10d,
                reconstructed,
                atol=1.0e-6,
                rtol=0.0,
            )
        ):
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
    "CourtQueryDINOv3ExecutionBoundary",
    "CourtQueryModelExecutionBoundary",
    "CourtQueryModelIOAdapter",
]
