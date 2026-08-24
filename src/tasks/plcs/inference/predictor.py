"""Inference boundary for a once-bound standard PLCS model and I/O adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor, nn

from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
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
        return self.io_adapter.profile

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

    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor,
        padding_mask: Tensor,
        court_vis: Tensor,
        *,
        denormalize: bool,
        court_reference_provenance: CourtReferenceFrameProvenance | None = None,
        court_keypoint_metadata: dict[str, object] | None = None,
    ) -> dict[str, Tensor]:
        """Validate, invoke, and decode caller-provided model-ready tensors."""
        with torch.no_grad():
            effective_provenance = court_reference_provenance
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
                    if effective_provenance is not None
                    else None
                ),
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
            return self._run_prepared(
                self.io_adapter.prepare_scene(
                    scene,
                    cameras,
                    reference_camera_id=reference_camera_id,
                )
            )

    def predict_multiview_observations(
        self,
        *,
        human_kp: np.ndarray,
        court_kp: np.ndarray,
        human_vis: np.ndarray,
        padding_mask: np.ndarray,
        court_vis: np.ndarray,
        court_reference_provenance: CourtReferenceFrameProvenance | None = None,
        court_keypoint_metadata: dict[str, object] | None = None,
    ) -> PLCSPhysicalPrediction:
        """Decode explicit NumPy ``(B,V,T,...)`` observations to physical units."""
        with torch.no_grad():
            prepared = self.io_adapter.prepare_multiview_observations(
                human_kp=human_kp,
                court_kp=court_kp,
                human_vis=human_vis,
                padding_mask=padding_mask,
                court_vis=court_vis,
                court_keypoint_metadata=(
                    court_keypoint_metadata
                ),
                court_reference_provenance=court_reference_provenance,
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
            )


__all__ = ["PLCSPredictor"]
