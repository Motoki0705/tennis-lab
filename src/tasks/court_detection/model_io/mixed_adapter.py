"""Pose adapter that masks auxiliary supervision to synthetic samples."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import torch
from torch import Tensor

from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.pose import CourtDecodedPose
from src.tasks.court_detection.model_io.adapters import (
    CourtModelIOAdapter,
    CourtPoseModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtModelCall,
    CourtModelIOError,
    CourtModelOutput,
    CourtPoseTargetBatch,
    CourtPoseTrainingResult,
    CourtTrainingCall,
    CourtTrainingResult,
    CourtTrainingTargetKind,
)
from src.tasks.court_detection.models.pose_head import CourtRawPoseOutput

POSE_SUPERVISION_MASK = "pose_supervision_mask"


class MixedCourtPoseModelIOAdapter(CourtPoseModelIOAdapter):
    """Keep dense losses on the full batch and pose terms on masked samples."""

    @staticmethod
    def _validate_pose_supervision_mask(
        value: object,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if value is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if (
            not isinstance(value, Tensor)
            or value.shape != (batch_size,)
            or value.dtype != torch.bool
        ):
            raise CourtModelIOError(
                "Court pose_supervision_mask must be bool with shape (B,)."
            )
        if value.device != device:
            raise CourtModelIOError(
                "Court pose_supervision_mask and images must share device."
            )
        return value

    def pose_supervision_mask(self, call: CourtTrainingCall) -> Tensor:
        """Return the validated full-batch synthetic supervision mask."""
        return self._validate_pose_supervision_mask(
            call.batch.get(POSE_SUPERVISION_MASK),
            batch_size=call.model_call.batch_size,
            device=call.model_call.images.device,
        )

    def prepare_training_batch(self, batch: Mapping[str, object]) -> CourtTrainingCall:
        dense_call = CourtModelIOAdapter.prepare_training_batch(self, batch)
        mask = self._validate_pose_supervision_mask(
            batch.get(POSE_SUPERVISION_MASK),
            batch_size=dense_call.model_call.batch_size,
            device=dense_call.model_call.images.device,
        )
        supervised_count = int(mask.sum().item())
        pose_value = batch.get("pose_target")
        if supervised_count == 0:
            if pose_value is not None:
                raise CourtModelIOError(
                    "Court pose_target must be absent when no sample is supervised."
                )
            targets: dict[CourtTrainingTargetKind, object] = dict(
                dense_call.targets
            )
            image_size_value = batch.get("image_size")
            if image_size_value is not None:
                targets["image_size"] = self._validate_image_size(
                    image_size_value,
                    call=dense_call.model_call,
                )
            return CourtTrainingCall(
                model_call=dense_call.model_call,
                targets=MappingProxyType(targets),
                batch=dense_call.batch,
            )

        pose_target = self._validate_pose_target(
            pose_value,
            batch_size=supervised_count,
        )
        kp_target = dense_call.targets.get("kp")
        consistency = self.pose_loss_config.consistency
        consistency_enabled = consistency.enabled
        image_size: Tensor | None = None
        if consistency_enabled or isinstance(batch.get("image_size"), Tensor):
            image_size = self._validate_image_size(
                batch.get("image_size"),
                call=dense_call.model_call,
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
                call=dense_call.model_call,
            )
        if consistency_enabled and not isinstance(kp_target, Mapping):
            raise CourtModelIOError("Consistency requires a KP target.")
        if isinstance(kp_target, Mapping) and "physical_indices" in kp_target:
            physical = cast(Tensor, kp_target["physical_indices"])
            if consistency_enabled and physical.shape[1:] != (14, 1):
                raise CourtModelIOError(
                    "Pose KP target must be singleton (B,14,1)."
                )
            if consistency_enabled and not torch.equal(
                physical[mask, :, 0],
                pose_target.semantic_to_physical,
            ):
                raise CourtModelIOError(
                    "Supervised KP physical order disagrees with pose authority."
                )

        enriched: dict[CourtTrainingTargetKind, object] = {
            **dense_call.targets,
            "pose": pose_target,
        }
        if image_size is not None:
            enriched["image_size"] = image_size
        return CourtTrainingCall(
            model_call=dense_call.model_call,
            targets=MappingProxyType(enriched),
            batch=dense_call.batch,
        )

    @staticmethod
    def _slice_tensor(value: Tensor, mask: Tensor, *, batch_size: int) -> Tensor:
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[mask]
        return value

    @classmethod
    def _slice_target(
        cls,
        value: object,
        mask: Tensor,
        *,
        batch_size: int,
    ) -> object:
        if isinstance(value, Tensor):
            return cls._slice_tensor(value, mask, batch_size=batch_size)
        if isinstance(value, Mapping):
            return MappingProxyType(
                {
                    key: cls._slice_target(item, mask, batch_size=batch_size)
                    for key, item in value.items()
                }
            )
        return value

    @staticmethod
    def _empty_decoded_pose(values: Tensor) -> CourtDecodedPose:
        return CourtDecodedPose(
            translation_m=values.new_empty((0, 3)),
            rotation=values.new_empty((0, 3, 3)),
            focal_px=values.new_empty((0,)),
            log_focal=values.new_empty((0,)),
        )

    def _supervised_call(
        self,
        call: CourtTrainingCall,
        *,
        mask: Tensor,
    ) -> CourtTrainingCall:
        batch_size = call.model_call.batch_size
        model_call = CourtModelCall(
            images=call.model_call.images[mask],
            model_args=tuple(
                self._slice_tensor(value, mask, batch_size=batch_size)
                for value in call.model_call.model_args
            ),
            batch_size=int(mask.sum().item()),
            height=call.model_call.height,
            width=call.model_call.width,
        )
        targets: dict[CourtTrainingTargetKind, object] = {}
        for kind in self.spec.target_bundle.kinds:
            targets[kind] = self._slice_target(
                call.targets[kind],
                mask,
                batch_size=batch_size,
            )
        pose_target = call.targets.get("pose")
        if not isinstance(pose_target, CourtPoseTargetBatch):
            raise CourtModelIOError(
                "Supervised mixed Court call lacks a typed pose target."
            )
        targets["pose"] = pose_target
        image_size = call.targets.get("image_size")
        if isinstance(image_size, Tensor):
            targets["image_size"] = image_size[mask]

        sub_batch: dict[str, object] = {}
        content_size = call.batch.get("content_size_hw")
        if isinstance(content_size, Tensor):
            sub_batch["content_size_hw"] = content_size[mask]
        return CourtTrainingCall(
            model_call=model_call,
            targets=MappingProxyType(targets),
            batch=MappingProxyType(sub_batch),
        )

    def training_result(
        self,
        output: object,
        call: CourtTrainingCall,
        *,
        progress_fraction: float | None = None,
    ) -> CourtPoseTrainingResult:
        checked = self.validate_output(output, call=call.model_call)
        dense_targets: dict[CourtTrainingTargetKind, object] = {
            kind: call.targets[kind] for kind in self.spec.target_bundle.kinds
        }
        dense_call = CourtTrainingCall(
            model_call=call.model_call,
            targets=MappingProxyType(dense_targets),
            batch=call.batch,
        )
        dense_result = cast(
            CourtTrainingResult,
            CourtModelIOAdapter.training_result(
                self,
                checked.dense_logits,
                dense_call,
            ),
        )
        mask = self.pose_supervision_mask(call)
        supervised_count = int(mask.sum().item())
        assert checked.pose is not None
        if supervised_count == 0:
            zero = dense_result.loss.new_zeros(())
            empty = MappingProxyType({})
            return CourtPoseTrainingResult(
                loss=dense_result.loss,
                raw_dense_loss=dense_result.raw_loss,
                direct_dense_loss=dense_result.loss,
                direct_pose_loss=zero,
                raw_dense_losses=dense_result.raw_losses,
                dense_losses=dense_result.losses,
                dense_configured_weights=dense_result.configured_weights,
                dense_effective_weights=dense_result.effective_weights,
                weighted_dense_losses=dense_result.weighted_losses,
                pose_losses=empty,
                weighted_pose_losses=empty,
                pose_configured_weights=empty,
                pose_effective_weights=empty,
                consistency=None,
                output=checked,
                decoded_pose=self._empty_decoded_pose(checked.pose.values),
            )

        supervised_call = self._supervised_call(call, mask=mask)
        supervised_output = CourtModelOutput(
            dense_logits=MappingProxyType(
                {
                    cast(CourtTargetKind, kind): value[mask]
                    for kind, value in checked.dense_logits.items()
                }
            ),
            pose=CourtRawPoseOutput(checked.pose.values[mask]),
        )
        supervised_result = CourtPoseModelIOAdapter.training_result(
            self,
            supervised_output,
            supervised_call,
            progress_fraction=progress_fraction,
        )
        auxiliary = (
            supervised_result.consistency.weighted_auxiliary_loss
            if supervised_result.consistency is not None
            else dense_result.loss.new_zeros(())
        )
        return CourtPoseTrainingResult(
            loss=(
                dense_result.loss
                + supervised_result.direct_pose_loss
                + auxiliary
            ),
            raw_dense_loss=dense_result.raw_loss,
            direct_dense_loss=dense_result.loss,
            direct_pose_loss=supervised_result.direct_pose_loss,
            raw_dense_losses=dense_result.raw_losses,
            dense_losses=dense_result.losses,
            dense_configured_weights=dense_result.configured_weights,
            dense_effective_weights=dense_result.effective_weights,
            weighted_dense_losses=dense_result.weighted_losses,
            pose_losses=supervised_result.pose_losses,
            weighted_pose_losses=supervised_result.weighted_pose_losses,
            pose_configured_weights=supervised_result.pose_configured_weights,
            pose_effective_weights=supervised_result.pose_effective_weights,
            consistency=supervised_result.consistency,
            output=checked,
            decoded_pose=supervised_result.decoded_pose,
        )


__all__ = ["MixedCourtPoseModelIOAdapter", "POSE_SUPERVISION_MASK"]
