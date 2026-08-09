"""Unified predictor class for BLCS inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.blcs.model_io import (
    BLCSTrajectoryPrediction,
    TrajectoryBoundModelIO,
    TrajectoryModelIOAdapter,
    compose_blcs_trajectory_model_io,
)
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_config
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class BLCSPredictor(BasePredictor[BLCSTrajectoryPrediction]):
    """Unified BLCS model inference predictor.

    Supports:
    - `blcs` (single-view)
    - `blcs_multiview` (multi-view)

    Attributes:
        model: The BLCS model.
        device: The inference device.

    Example:
        >>> predictor = BLCSPredictor.load_from_checkpoint("model.ckpt", device="cuda")
        >>> result = predictor.predict(ball_uv, court_kp, ball_vis, ball_mask, court_vis)
        >>> print(result.position.shape)  # (B, T, 3)

    """

    def __init__(
        self,
        model_io: TrajectoryBoundModelIO,
        device: torch.device,
        norm_scale_xyz: tuple[float, float, float] = COURT_COORD_SCALE_XYZ,
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized BLCS model.
            device: Inference device.

        """
        self.model_io = model_io
        self.model = model_io.model.to(device)
        self.io_adapter = cast("TrajectoryModelIOAdapter", model_io.adapter)
        self.device = device
        self.norm_scale_xyz = norm_scale_xyz
        self.model.eval()

    @property
    def input_profile(self) -> str:
        """Return the adapter-declared input profile without model class checks."""
        return str(self.io_adapter.input_profile)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        **kwargs: Any,
    ) -> Self:
        """Create a BLCSPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Forwarded to `BLCSLightningModule.load_from_checkpoint`.

        Returns:
            Initialized BLCSPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects exactly one checkpoint, got {len(checkpoints)}."
            )
        binding = compose_blcs_trajectory_model_io(
            load_checkpoint_config(checkpoints[0])
        )
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoints[0],
            BLCSLightningModule,
            resolver=resolver,
            device=device,
            model_io=binding,
            strict=True,
            weights_only=False,
            **kwargs,
        )
        return cls(model_io=lightning_module.model_io, device=resolved_device)

    def predict_batch(
        self,
        batch: Mapping[str, object],
        *,
        denormalize: bool,
    ) -> BLCSTrajectoryPrediction:
        """Validate, execute, and decode one typed trajectory batch."""
        moved = {
            key: value.to(self.device) if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            prediction = self.model_io.run(moved)
        position = prediction.position
        velocity = prediction.velocity
        if denormalize:
            position = self._denormalize_coords(position, self.norm_scale_xyz)
            if velocity is not None:
                velocity = self._denormalize_coords(velocity, self.norm_scale_xyz)
        return BLCSTrajectoryPrediction(
            position=position.detach().cpu(),
            velocity=None if velocity is None else velocity.detach().cpu(),
        )

    def predict_multiview_arrays(
        self,
        *,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_],
        court_vis: NDArray[np.bool_] | NDArray[np.float32],
        denormalize: bool,
    ) -> BLCSTrajectoryPrediction:
        """Build and predict one explicit multiview scene-array window."""
        if self.input_profile != "multiview":
            raise ValueError(
                "predict_multiview_arrays requires a multiview BLCS checkpoint."
            )
        batch = self.io_adapter.build_inference_batch_from_arrays(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_vis=ball_vis,
            court_vis=court_vis,
        )
        return self.predict_batch(batch, denormalize=denormalize)

    def predict_scene(
        self,
        scene: Mapping[str, object],
        cameras: list[int],
        *,
        denormalize: bool,
    ) -> BLCSTrajectoryPrediction:
        """Build the selected profile from a scene and return a typed decode."""
        batch = self.io_adapter.build_inference_batch_from_scene(scene, cameras)
        return self.predict_batch(batch, denormalize=denormalize)

    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor,
        ball_mask: Tensor,
        court_vis: Tensor,
        denormalize: bool = True,
    ) -> BLCSTrajectoryPrediction:
        """Predict and return the adapter's typed trajectory decode.

        Args:
            ball_uv: Ball 2D trajectory tensor accepted by the loaded model.
            court_kp: Court keypoint tensor accepted by the loaded model.
            ball_vis: Ball visibility tensor.
            ball_mask: Ball validity/padding mask tensor.
            court_vis: Court keypoint visibility tensor.
            denormalize: If True, convert positions to meters.

        """
        return self.predict_batch(
            {
                "ball_uv": ball_uv,
                "court_kp": court_kp,
                "ball_vis": ball_vis,
                "ball_mask": ball_mask,
                "court_vis": court_vis,
            },
            denormalize=denormalize,
        )
