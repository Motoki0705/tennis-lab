"""PLCS module for 3D player localization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.tennis_scene.pipeline.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class PLCSResult:
    """Result of PLCS inference.

    Attributes:
        position: Player 3D position in court coords (T, 3), meters.
        yaw: Player yaw angle (T,), radians.

    """

    position: NDArray[np.float32]
    yaw: NDArray[np.float32]


class PLCSModule(BasePipelineModule):
    """PLCS module for 3D player localization.

    Predicts player 3D position and yaw from 2D human keypoints
    and court keypoints.

    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        """Initialize the module.

        Args:
            checkpoint_path: Path to PLCS model checkpoint.
            device: Inference device.

        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._predictor = None

    def load(self) -> None:
        """Load the PLCS predictor."""
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading PLCS model from {self.checkpoint_path}")

        from src.plcs.inference.predictor import PLCSPredictor

        self._predictor = PLCSPredictor.load_from_checkpoint(
            self.checkpoint_path, device=self.device
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._predictor is not None

    def process(
        self,
        human_kp_2d: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        human_kp_vis: NDArray[np.float32] | None = None,
        court_vis: NDArray[np.float32] | None = None,
    ) -> PLCSResult:
        """Run PLCS inference.

        Args:
            human_kp_2d: Human 2D keypoints (T, 17, 2), normalized [0, 1].
            court_kp: Court keypoints (20, 2), normalized [0, 1].
            human_kp_vis: Human keypoint visibility (T, 17).
            court_vis: Court keypoint visibility (20,).

        Returns:
            PLCSResult with 3D position and yaw.

        """
        if not self.is_loaded:
            self.load()

        LOGGER.info("Running PLCS player localization...")

        T = len(human_kp_2d)
        positions = []
        yaws = []

        court_kp_t = torch.from_numpy(court_kp).float()
        court_vis_t = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float()

        for t in range(T):
            human_kp_t = torch.from_numpy(human_kp_2d[t]).float().unsqueeze(0)
            human_vis_t = None
            if human_kp_vis is not None:
                human_vis_t = torch.from_numpy(human_kp_vis[t]).float().unsqueeze(0)

            pred = self._predictor.predict(
                human_kp=human_kp_t,
                court_kp=court_kp_t.unsqueeze(0),
                human_vis=human_vis_t,
                court_vis=court_vis_t.unsqueeze(0) if court_vis_t is not None else None,
                denormalize=True,
            )

            positions.append(pred["position_meters"].squeeze(0).numpy())
            yaws.append(pred["yaw_radians"].item())

        return PLCSResult(
            position=np.stack(positions, axis=0).astype(np.float32),
            yaw=np.array(yaws, dtype=np.float32),
        )


if __name__ == "__main__":
    print("PLCSModule: 3D player localization module")
    print("Use PLCSModule(checkpoint_path, device) to create")
