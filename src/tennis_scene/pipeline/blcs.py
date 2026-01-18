"""BLCS module for 3D ball localization."""

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
class BLCSResult:
    """Result of BLCS inference.

    Attributes:
        ball_3d: Ball 3D position in court coords (T, 3), meters.

    """

    ball_3d: NDArray[np.float32]


class BLCSModule(BasePipelineModule):
    """BLCS module for 3D ball localization.

    Predicts ball 3D trajectory from 2D ball positions
    and court keypoints.

    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        """Initialize the module.

        Args:
            checkpoint_path: Path to BLCS model checkpoint.
            device: Inference device.

        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._predictor = None

    def load(self) -> None:
        """Load the BLCS predictor."""
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading BLCS model from {self.checkpoint_path}")

        from src.blcs.inference.predictor import BLCSPredictor

        self._predictor = BLCSPredictor.load_from_checkpoint(
            self.checkpoint_path, device=self.device
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._predictor is not None

    def process(
        self,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_] | None = None,
        court_vis: NDArray[np.float32] | None = None,
    ) -> BLCSResult:
        """Run BLCS inference.

        Args:
            ball_uv: Ball 2D positions (T, 2), normalized [0, 1].
            court_kp: Court keypoints (20, 2), normalized [0, 1].
            ball_vis: Ball visibility mask (T,).
            court_vis: Court keypoint visibility (20,).

        Returns:
            BLCSResult with 3D ball trajectory.

        """
        if not self.is_loaded:
            self.load()

        LOGGER.info("Running BLCS ball localization...")

        ball_uv_t = torch.from_numpy(ball_uv).float()
        court_kp_t = torch.from_numpy(court_kp).float()

        ball_mask_t = None
        if ball_vis is not None:
            ball_mask_t = torch.from_numpy(ball_vis.astype(np.float32))

        court_vis_t = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float()

        pred = self._predictor.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_mask=ball_mask_t,
            court_vis=court_vis_t,
            denormalize=True,
        )

        ball_3d = pred["position"].squeeze(0).numpy().astype(np.float32)

        return BLCSResult(ball_3d=ball_3d)


if __name__ == "__main__":
    print("BLCSModule: 3D ball localization module")
    print("Use BLCSModule(checkpoint_path, device) to create")
