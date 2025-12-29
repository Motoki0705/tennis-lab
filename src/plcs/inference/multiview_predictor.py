"""PLCS multi-view inference predictor for tennis analysis."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, Self, TypeVar, cast

import torch
from torch import Tensor

from src.base.api.predictor import BasePredictor
from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.plcs.training.multiview_lightning_module import PLCSMultiViewLightningModule
from src.utils.geometry.constants import COURT_COORD_SCALE_XYZ

P = ParamSpec("P")
R = TypeVar("R")


def _no_grad(func: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], torch.no_grad()(func))


class PLCSMultiViewPredictor(BasePredictor):
    """PLCS multi-view model inference predictor.

    Predicts 3D position and orientation from multiple camera views.

    Attributes:
        model: The PLCS multi-view model.
        device: The inference device.

    Example:
        >>> predictor = PLCSMultiViewPredictor.load_from_checkpoint(
        ...     "model.ckpt", device="cuda"
        ... )
        >>> results = predictor.predict(human_kp, court_kp)
        >>> print(results["position"].shape)  # (B, 3)

    """

    def __init__(
        self,
        model: PLCSMultiViewModel,
        device: torch.device,
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized PLCS multi-view model.
            device: Inference device.

        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        self._norm_scale_xyz = COURT_COORD_SCALE_XYZ

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Create a PLCSMultiViewPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Unused (for compatibility).

        Returns:
            Initialized PLCSMultiViewPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(device)
        lightning_module = PLCSMultiViewLightningModule.load_from_checkpoint(
            strict=False,
            checkpoint_path=checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        return cls(model=lightning_module.model, device=device)

    @_no_grad
    def predict(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_kp_mask: Tensor | None = None,
        court_kp_mask: Tensor | None = None,
        view_mask: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict player 3D position and orientation from multiple views.

        Args:
            human_kp: Human keypoints. Shape (B, N, 17, 2) or (N, 17, 2).
            court_kp: Court keypoints. Shape (B, N, 20, 2) or (N, 20, 2).
            human_kp_mask: Human keypoint visibility mask. Shape (B, N, 17).
            court_kp_mask: Court keypoint visibility mask. Shape (B, N, 20).
            view_mask: Valid view mask. Shape (B, N).
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary:
                - position: Normalized position (B, 3)
                - position_meters: Position in meters (B, 3) (if denormalize=True)
                - rotation: (sin, cos) (B, 2)
                - yaw_radians: Yaw angle in radians (B,) (if denormalize=True)

        """
        # Add batch dimension if needed
        if human_kp.dim() == 3:
            human_kp = human_kp.unsqueeze(0)
        if court_kp.dim() == 3:
            court_kp = court_kp.unsqueeze(0)
        if human_kp_mask is not None and human_kp_mask.dim() == 2:
            human_kp_mask = human_kp_mask.unsqueeze(0)
        if court_kp_mask is not None and court_kp_mask.dim() == 2:
            court_kp_mask = court_kp_mask.unsqueeze(0)
        if view_mask is not None and view_mask.dim() == 1:
            view_mask = view_mask.unsqueeze(0)

        # Move to device
        human_kp = human_kp.to(self.device)
        court_kp = court_kp.to(self.device)
        if human_kp_mask is not None:
            human_kp_mask = human_kp_mask.to(self.device)
        if court_kp_mask is not None:
            court_kp_mask = court_kp_mask.to(self.device)
        if view_mask is not None:
            view_mask = view_mask.to(self.device)

        # Forward pass
        outputs = self.model(
            human_kp=human_kp,
            court_kp=court_kp,
            human_kp_mask=human_kp_mask,
            court_kp_mask=court_kp_mask,
            view_mask=view_mask,
        )

        position = outputs["position"].cpu()
        rotation = outputs["rotation"].cpu()

        result: dict[str, Tensor] = {
            "position": position,
            "rotation": rotation,
        }

        if denormalize:
            scale = torch.tensor(
                list(self._norm_scale_xyz),
                dtype=position.dtype,
            )
            result["position_meters"] = position * scale
            result["yaw_radians"] = torch.atan2(rotation[:, 0], rotation[:, 1])

        return result

    @_no_grad
    def predict_sequence(
        self,
        human_kp_seq: Tensor,
        court_kp_seq: Tensor,
        human_kp_mask_seq: Tensor | None = None,
        court_kp_mask_seq: Tensor | None = None,
        view_mask: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        """Predict player 3D position and orientation for a sequence.

        Args:
            human_kp_seq: Human keypoints sequence. Shape (T, N, 17, 2).
            court_kp_seq: Court keypoints sequence. Shape (T, N, 20, 2).
            human_kp_mask_seq: Visibility mask. Shape (T, N, 17).
            court_kp_mask_seq: Visibility mask. Shape (T, N, 20).
            view_mask: Valid view mask. Shape (N,) - shared across frames.
            denormalize: If True, convert positions to meters.

        Returns:
            Inference results dictionary:
                - position: Normalized position (T, 3)
                - position_meters: Position in meters (T, 3) (if denormalize=True)
                - rotation: (sin, cos) (T, 2)
                - yaw_radians: Yaw angle in radians (T,) (if denormalize=True)

        """
        T = human_kp_seq.shape[0]

        positions = []
        rotations = []

        for t in range(T):
            human_kp = human_kp_seq[t : t + 1]  # (1, N, 17, 2)
            court_kp = court_kp_seq[t : t + 1]  # (1, N, 20, 2)
            hm = human_kp_mask_seq[t : t + 1] if human_kp_mask_seq is not None else None
            cm = court_kp_mask_seq[t : t + 1] if court_kp_mask_seq is not None else None
            vm = view_mask.unsqueeze(0) if view_mask is not None else None

            out = self.predict(
                human_kp=human_kp,
                court_kp=court_kp,
                human_kp_mask=hm,
                court_kp_mask=cm,
                view_mask=vm,
                denormalize=False,
            )
            positions.append(out["position"])
            rotations.append(out["rotation"])

        position = torch.cat(positions, dim=0)  # (T, 3)
        rotation = torch.cat(rotations, dim=0)  # (T, 2)

        result: dict[str, Tensor] = {
            "position": position,
            "rotation": rotation,
        }

        if denormalize:
            scale = torch.tensor(
                list(self._norm_scale_xyz),
                dtype=position.dtype,
            )
            result["position_meters"] = position * scale
            result["yaw_radians"] = torch.atan2(rotation[:, 0], rotation[:, 1])

        return result
