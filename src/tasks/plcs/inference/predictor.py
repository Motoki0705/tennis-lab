"""Inference boundary for a once-bound standard PLCS model and I/O adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import torch
from torch import Tensor, nn

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    build_physical_court_provenance,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import ModelCall, ModelInputContractError
from src.tasks.plcs.court_keypoint_contract import (
    headings_target_to_physical,
    normalized_points_target_to_physical,
)
from src.tasks.plcs.model_io import (
    PLCSDecodedPrediction,
    PLCSInputProfile,
    PLCSModelIOAdapter,
    PLCSPhysicalPrediction,
    PLCSPreparedBatch,
    PLCSReferenceMetadata,
    PLCSStandardBoundModelIO,
    bind_plcs_model_io,
    prepare_plcs_checkpoint_court_keypoint_config,
)
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.utils.configuration import PathResolver
from src.utils.schema.court_normalization import load_and_validate_checkpoint


class PLCSPredictor(BasePredictor):
    """Run standard PLCS inference through its construction-bound adapter."""

    def __init__(
        self,
        *,
        model: nn.Module,
        adapter: PLCSModelIOAdapter,
        device: torch.device,
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> None:
        bound = bind_plcs_model_io(model, adapter)
        self.model_io: PLCSStandardBoundModelIO = bound
        self.model = self.model_io.model.to(device).eval()
        self.io_adapter = adapter
        self.device = device
        self.court_keypoint_contract = (
            court_keypoint_contract or adapter.court_keypoint_contract
        )
        if self.court_keypoint_contract != adapter.court_keypoint_contract:
            raise ModelInputContractError(
                "PLCS predictor and adapter CourtKP20 contracts do not match."
            )

    @property
    def input_profile(self) -> PLCSInputProfile:
        """Return the profile fixed by the checkpoint composition."""
        return PLCSInputProfile(self.io_adapter.profile)

    def require_input_profile(self, profile: PLCSInputProfile | str) -> None:
        """Fail before assembly when a consumer needs another input profile."""
        self.io_adapter.require_profile(profile)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        court_keypoint_contract: CourtKeypointContract | None = None,
        **kwargs: Any,
    ) -> Self:
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )
        if "config" in kwargs:
            raise TypeError(
                "PLCSPredictor restores checkpoint config internally; do not pass config."
            )
        checkpoint = load_and_validate_checkpoint(checkpoints[0])
        checkpoint_config, keypoint_contract = (
            prepare_plcs_checkpoint_court_keypoint_config(
                checkpoint,
                court_keypoint_contract,
                location=str(checkpoints[0]),
            )
        )
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoints[0],
            PLCSLightningModule,
            resolver=resolver,
            device=device,
            config=checkpoint_config,
            strict=bool(kwargs.pop("strict", True)),
            weights_only=bool(kwargs.pop("weights_only", False)),
            **kwargs,
        )
        adapter = lightning_module.io_adapter
        if not isinstance(adapter, PLCSModelIOAdapter):
            raise ModelInputContractError(
                "Loaded checkpoint does not contain a standard PLCS I/O adapter."
            )
        return cls(
            model=lightning_module.model,
            adapter=adapter,
            device=resolved_device,
            court_keypoint_contract=keypoint_contract,
        )

    def _move_call(self, call: ModelCall) -> ModelCall:
        return ModelCall(
            args=tuple(
                value.to(self.device) if isinstance(value, Tensor) else None
                for value in call.args
            ),
            kwargs={
                key: value.to(self.device) if isinstance(value, Tensor) else None
                for key, value in call.kwargs.items()
            },
        )

    def _run_prepared(self, prepared: PLCSPreparedBatch) -> PLCSDecodedPrediction:
        moved_call = self._move_call(prepared.call)
        raw_output = self.model_io.execute_call(moved_call)
        decoded = self.io_adapter.decode_prepared_output(raw_output, prepared)
        return PLCSDecodedPrediction(
            position=decoded.position.detach().cpu(),
            rotation=decoded.rotation.detach().cpu(),
            canonical_pose=(
                decoded.canonical_pose.detach().cpu()
                if decoded.canonical_pose is not None
                else None
            ),
            auxiliary_position=(
                decoded.auxiliary_position.detach().cpu()
                if decoded.auxiliary_position is not None
                else None
            ),
            court_reference_provenance=decoded.court_reference_provenance,
            reference_metadata=(
                prepared.reference_metadata.cpu()
                if prepared.reference_metadata is not None
                else None
            ),
        )

    def _physical_outputs(
        self,
        position: Tensor,
        rotation: Tensor,
        provenance: tuple[CourtReferenceFrameProvenance, ...] | None,
    ) -> tuple[Tensor, Tensor]:
        if provenance is None:
            raise ModelInputContractError(
                "PLCS physical prediction requires Court reference provenance."
            )
        if len(provenance) == 1:
            return (
                normalized_points_target_to_physical(
                    position,
                    provenance[0],
                ),
                headings_target_to_physical(rotation, provenance[0]),
            )
        if position.shape[0] != len(provenance):
            raise ModelInputContractError(
                "PLCS prediction batch and provenance cardinality do not match."
            )
        return (
            torch.stack(
                [
                    normalized_points_target_to_physical(
                        position[index],
                        item,
                    )
                    for index, item in enumerate(provenance)
                ]
            ),
            torch.stack(
                [
                    headings_target_to_physical(rotation[index], item)
                    for index, item in enumerate(provenance)
                ]
            ),
        )

    def _resolve_reference_provenance(
        self,
        provenance: CourtReferenceFrameProvenance
        | Sequence[CourtReferenceFrameProvenance]
        | None,
        reference_metadata: PLCSReferenceMetadata | None,
    ) -> (
        CourtReferenceFrameProvenance
        | tuple[CourtReferenceFrameProvenance, ...]
        | None
    ):
        if reference_metadata is None:
            if provenance is None or isinstance(
                provenance,
                CourtReferenceFrameProvenance,
            ):
                return provenance
            return tuple(provenance)
        if self.court_keypoint_contract.selector != CAMERA_VIEW_V2_SELECTOR:
            raise ModelInputContractError(
                "physical_v1 PLCS inference cannot consume reference metadata."
            )
        metadata_provenance = tuple(
            selection.provenance for selection in reference_metadata.selections
        )
        if provenance is not None:
            explicit = (
                (provenance,)
                if isinstance(provenance, CourtReferenceFrameProvenance)
                else tuple(provenance)
            )
            if explicit != metadata_provenance:
                raise ModelInputContractError(
                    "PLCS provenance and typed reference metadata do not match."
                )
        return metadata_provenance

    def _reference_metadata_for_scene(
        self,
        scene: object,
        cameras: Sequence[int],
        reference_camera_id: str | None,
    ) -> PLCSReferenceMetadata | None:
        if self.court_keypoint_contract.selector != CAMERA_VIEW_V2_SELECTOR:
            return None
        if reference_camera_id is None:
            raise ModelInputContractError(
                "camera_view_v2 scene prediction requires reference_camera_id."
            )
        scene_cameras = getattr(scene, "cameras", None)
        if not isinstance(scene_cameras, Sequence):
            raise ModelInputContractError("PLCS scene must expose a cameras sequence.")
        selected_indices = (
            tuple(cameras)
            if self.input_profile is PLCSInputProfile.MULTIVIEW
            else (cameras[0],)
        )
        complete_views = tuple(
            getattr(camera, "court_view", None) for camera in scene_cameras
        )
        if any(not isinstance(view, CourtViewRecord) for view in complete_views):
            raise ModelInputContractError(
                "camera_view_v2 scene cameras require typed CourtKP20 metadata."
            )
        typed_complete = cast("tuple[CourtViewRecord, ...]", complete_views)
        selected_views = tuple(typed_complete[index] for index in selected_indices)
        try:
            table = StableCameraIdTable.from_complete_scene_camera_ids(
                tuple(view.camera_id for view in typed_complete)
            )
            selection = ReferenceViewSelection.create(
                stable_camera_id_table=table,
                selected_views=selected_views,
                reference_camera_id=reference_camera_id,
            )
        except ValueError as error:
            raise ModelInputContractError(str(error)) from error
        fields = selection.to_tensor_fields(dtype=torch.float32)
        forward = fields["reference_from_physical"].unsqueeze(0)
        return PLCSReferenceMetadata(
            selections=(selection,),
            stable_camera_id_tables=(table,),
            reference_view_index=fields["reference_view_index"].unsqueeze(0),
            view_camera_ids=fields["view_camera_ids"].unsqueeze(0),
            reference_camera_id=fields["reference_camera_id"].unsqueeze(0),
            reference_from_physical=forward,
            physical_from_reference=forward.transpose(-1, -2),
        )

    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor,
        padding_mask: Tensor,
        court_vis: Tensor,
        *,
        denormalize: bool,
        court_reference_provenance: CourtReferenceFrameProvenance
        | Sequence[CourtReferenceFrameProvenance]
        | None = None,
        court_keypoint_metadata: dict[str, object] | None = None,
        reference_metadata: PLCSReferenceMetadata | None = None,
    ) -> dict[str, Tensor]:
        """Validate, invoke, and decode caller-provided model-ready tensors."""
        with torch.no_grad():
            effective_provenance = self._resolve_reference_provenance(
                court_reference_provenance,
                reference_metadata,
            )
            if (
                effective_provenance is None
                and self.court_keypoint_contract.selector == PHYSICAL_V1_SELECTOR
            ):
                effective_provenance = build_physical_court_provenance()
            prepared = PLCSPreparedBatch(
                call=self.io_adapter.build_call(
                    {
                        "human_kp": human_kp,
                        "court_kp": court_kp,
                        "human_vis": human_vis,
                        "padding_mask": padding_mask,
                        "court_vis": court_vis,
                        "court_keypoint_metadata": (
                            court_keypoint_metadata
                        ),
                        "court_reference_provenance": effective_provenance,
                    }
                ),
                court_reference_provenance=(
                    (effective_provenance,)
                    if isinstance(
                        effective_provenance,
                        CourtReferenceFrameProvenance,
                    )
                    else effective_provenance
                ),
                reference_metadata=reference_metadata,
            )
            decoded = self._run_prepared(prepared)
            result = {
                "position": decoded.position,
                "rotation": decoded.rotation,
            }
            if decoded.canonical_pose is not None:
                result["canonical_pose"] = decoded.canonical_pose
            if decoded.auxiliary_position is not None:
                result["auxiliary_position"] = decoded.auxiliary_position
            if denormalize:
                physical_position, physical_heading = self._physical_outputs(
                    decoded.position,
                    decoded.rotation,
                    decoded.court_reference_provenance,
                )
                result["position_meters"] = physical_position
                result["yaw_radians"] = torch.atan2(
                    physical_heading[..., 1], physical_heading[..., 0]
                )
            return result

    def predict_scene(
        self,
        scene: object,
        cameras: Sequence[int],
        *,
        reference_camera_id: str | None = None,
    ) -> PLCSDecodedPrediction:
        """Assemble and predict one loaded PLCS scene through the adapter."""
        with torch.no_grad():
            prepared = self.io_adapter.prepare_scene(
                scene,
                cameras,
                reference_camera_id=reference_camera_id,
            )
            reference_metadata = self._reference_metadata_for_scene(
                scene,
                cameras,
                reference_camera_id,
            )
            return self._run_prepared(
                replace(prepared, reference_metadata=reference_metadata)
            )

    def predict_multiview_observations(
        self,
        *,
        human_kp: np.ndarray,
        court_kp: np.ndarray,
        human_vis: np.ndarray,
        padding_mask: np.ndarray,
        court_vis: np.ndarray,
        court_reference_provenance: CourtReferenceFrameProvenance
        | Sequence[CourtReferenceFrameProvenance]
        | None = None,
        court_keypoint_metadata: dict[str, object] | None = None,
        reference_metadata: PLCSReferenceMetadata | None = None,
    ) -> PLCSPhysicalPrediction:
        """Decode explicit NumPy ``(B,V,T,...)`` observations to physical units."""
        with torch.no_grad():
            effective_provenance = self._resolve_reference_provenance(
                court_reference_provenance,
                reference_metadata,
            )
            prepare_arguments: dict[str, Any] = {
                "human_kp": human_kp,
                "court_kp": court_kp,
                "human_vis": human_vis,
                "padding_mask": padding_mask,
                "court_vis": court_vis,
                "court_keypoint_metadata": court_keypoint_metadata,
                "court_reference_provenance": effective_provenance,
            }
            prepared = self.io_adapter.prepare_multiview_observations(
                **prepare_arguments,
            )
            if reference_metadata is not None:
                metadata_provenance = tuple(
                    selection.provenance
                    for selection in reference_metadata.selections
                )
                if prepared.court_reference_provenance != metadata_provenance:
                    raise ModelInputContractError(
                        "PLCS multiview provenance and typed reference metadata "
                        "do not match."
                    )
                prepared = replace(
                    prepared,
                    reference_metadata=reference_metadata,
                )
            decoded = self._run_prepared(prepared)
            position_physical, heading_physical = self._physical_outputs(
                decoded.position,
                decoded.rotation,
                decoded.court_reference_provenance,
            )
            position_meters = position_physical.numpy()
            yaw_radians = torch.atan2(
                heading_physical[..., 1], heading_physical[..., 0]
            ).numpy()
            canonical_pose = (
                decoded.canonical_pose.numpy()
                if decoded.canonical_pose is not None
                else None
            )
            return PLCSPhysicalPrediction(
                position_meters=position_meters.astype(np.float32, copy=False),
                yaw_radians=yaw_radians.astype(np.float32, copy=False),
                canonical_pose=(
                    canonical_pose.astype(np.float32, copy=False)
                    if canonical_pose is not None
                    else None
                ),
                court_reference_provenance=decoded.court_reference_provenance,
                reference_metadata=decoded.reference_metadata,
            )


__all__ = ["PLCSPredictor"]
