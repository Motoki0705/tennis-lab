"""Court keypoint detection module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.tennis_scene.pipeline.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class CourtKPResult:
    """Result of court keypoint detection.

    Attributes:
        keypoints: Court keypoints (20, 2), normalized [0, 1].
        visibility: Keypoint visibility (20,).
        frame_index: Frame index used for detection.

    """

    keypoints: NDArray[np.float32]
    visibility: NDArray[np.float32]
    frame_index: int


class CourtKPModule(BasePipelineModule):
    """Court keypoint detection module.

    Detects 20 court keypoints from a single frame (fixed camera assumption).

    Attributes:
        checkpoint_path: Path to model checkpoint.
        device: Inference device.

    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        """Initialize the module.

        Args:
            checkpoint_path: Path to model checkpoint.
            device: Inference device.

        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._predictor = None

    def load(self) -> None:
        """Load the court keypoint predictor."""
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading Court KP model from {self.checkpoint_path}")
        from src.court_detection.inference.predictor import CourtKeypointPredictor

        self._predictor = CourtKeypointPredictor.from_checkpoint(
            self.checkpoint_path, device=self.device
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._predictor is not None

    def process(
        self,
        frame: NDArray[np.uint8],
        frame_index: int = 0,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> CourtKPResult:
        """Detect court keypoints from a frame.

        Args:
            frame: RGB frame array (H, W, 3).
            frame_index: Frame index (for metadata).
            image_width: Image width for normalization.
            image_height: Image height for normalization.

        Returns:
            CourtKPResult with normalized keypoints.

        """
        if not self.is_loaded:
            self.load()

        result = self._predictor.predict(frame)

        keypoints = result["keypoints"].astype(np.float32)
        visibility = result["visibility"].astype(np.float32)

        if image_width is not None and image_height is not None:
            keypoints[..., 0] /= image_width
            keypoints[..., 1] /= image_height

        return CourtKPResult(
            keypoints=keypoints,
            visibility=visibility,
            frame_index=frame_index,
        )


if __name__ == "__main__":
    print("CourtKPModule: court keypoint detection module")
    print("Use CourtKPModule(checkpoint_path, device) to create")
