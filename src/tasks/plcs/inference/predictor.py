"""Inference boundary for a once-bound standard PLCS model and I/O adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor, nn

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import ModelCall, ModelInputContractError
from src.tasks.plcs.model_io import (
    PLCSDecodedPrediction,
    PLCSInputProfile,
    PLCSModelIOAdapter,
    PLCSPhysicalPrediction,
    PLCSPreparedBatch,
    PLCSStandardBoundModelIO,
    bind_plcs_model_io,
)
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.court_normalization import load_and_validate_checkpoint


class PLCSPredictor(BasePredictor):
    """Run standard PLCS inference through its construction-bound adapter."""

    def __init__(
        self,
        *,
        model: nn.Module,
        adapter: PLCSModelIOAdapter,
        device: torch.device,
    ) -> None:
        bound = bind_plcs_model_io(model, adapter)
        self.model_io: PLCSStandardBoundModelIO = bound
        self.model = self.model_io.model.to(device).eval()
        self.io_adapter = adapter
        self.device = device
        self._norm_scale_xyz = COURT_COORD_SCALE_XYZ

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
        **kwargs: Any,
    ) -> Self:
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )
        load_and_validate_checkpoint(checkpoints[0])
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoints[0],
            PLCSLightningModule,
            resolver=resolver,
            device=device,
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
    ) -> dict[str, Tensor]:
        """Validate, invoke, and decode caller-provided model-ready tensors."""
        with torch.no_grad():
            prepared = PLCSPreparedBatch(
                call=self.io_adapter.build_call(
                    {
                        "human_kp": human_kp,
                        "court_kp": court_kp,
                        "human_vis": human_vis,
                        "padding_mask": padding_mask,
                        "court_vis": court_vis,
                    }
                )
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
                result["position_meters"] = self._denormalize_coords(
                    decoded.position, self._norm_scale_xyz
                )
                result["yaw_radians"] = torch.atan2(
                    decoded.rotation[..., 1], decoded.rotation[..., 0]
                )
            return result

    def predict_scene(
        self,
        scene: object,
        cameras: Sequence[int],
    ) -> PLCSDecodedPrediction:
        """Assemble and predict one loaded PLCS scene through the adapter."""
        with torch.no_grad():
            return self._run_prepared(self.io_adapter.prepare_scene(scene, cameras))

    def predict_multiview_observations(
        self,
        *,
        human_kp: np.ndarray,
        court_kp: np.ndarray,
        human_vis: np.ndarray,
        padding_mask: np.ndarray,
        court_vis: np.ndarray,
    ) -> PLCSPhysicalPrediction:
        """Decode explicit NumPy ``(B,V,T,...)`` observations to physical units."""
        with torch.no_grad():
            prepared = self.io_adapter.prepare_multiview_observations(
                human_kp=human_kp,
                court_kp=court_kp,
                human_vis=human_vis,
                padding_mask=padding_mask,
                court_vis=court_vis,
            )
            decoded = self._run_prepared(prepared)
            position_meters = self._denormalize_coords(
                decoded.position, self._norm_scale_xyz
            ).numpy()
            yaw_radians = torch.atan2(
                decoded.rotation[..., 1], decoded.rotation[..., 0]
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
            )


__all__ = ["PLCSPredictor"]
