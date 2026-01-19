"""WASB module for ball detection."""

from __future__ import annotations

import json
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
class WASBConfig:
    """Configuration for WASB module.

    Attributes:
        checkpoint: Path to WASB model checkpoint.
        batch_size: Batch size for inference.
        device: Inference device.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.

    """

    checkpoint: str | Path
    batch_size: int = 64
    device: str = "cuda"
    save_result: bool = False
    output_path: str | Path | None = None


@dataclass
class WASBResult:
    """Result of WASB ball detection.

    Attributes:
        ball_uv: Ball 2D position (T, 2), normalized [0, 1].
        ball_uv_px: Ball 2D position (T, 2), in pixels.
        visibility: Ball visibility mask (T,).
        score: Detection confidence score (T,).

    """

    ball_uv: NDArray[np.float32]
    ball_uv_px: NDArray[np.float32]
    visibility: NDArray[np.bool_]
    score: NDArray[np.float32]

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "ball_uv": self.ball_uv.tolist(),
            "ball_uv_px": self.ball_uv_px.tolist(),
            "visibility": self.visibility.tolist(),
            "score": self.score.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WASBResult":
        """Create result from dict."""
        return cls(
            ball_uv=np.array(data["ball_uv"], dtype=np.float32),
            ball_uv_px=np.array(data["ball_uv_px"], dtype=np.float32),
            visibility=np.array(data["visibility"], dtype=np.bool_),
            score=np.array(data["score"], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved WASB result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "WASBResult":
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class WASBModule(BasePipelineModule):
    """WASB module for ball detection.

    Detects ball positions in video frames using WASB predictor.

    """

    def __init__(self, config: WASBConfig) -> None:
        """Initialize the module.

        Args:
            config: WASB configuration.

        """
        self.config = config
        self._pipeline = None

    def load(self) -> None:
        """Load the WASB pipeline."""
        if self._pipeline is not None:
            return

        LOGGER.info(f"Loading WASB model from {self.config.checkpoint}")

        from src.wasb.inference import WASBPredictor
        from src.wasb.pipeline import VideoBallLocalizationPipeline

        predictor = WASBPredictor.load_from_checkpoint(
            self.config.checkpoint, device=self.config.device
        )

        self._pipeline = VideoBallLocalizationPipeline(
            predictor, batch_size=self.config.batch_size
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._pipeline is not None

    def process(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> WASBResult:
        """Run ball detection on video.

        Args:
            video_path: Path to input video.
            max_frames: Maximum frames to process.
            image_width: Image width for normalization.
            image_height: Image height for normalization.

        Returns:
            WASBResult with ball positions.

        """
        if not self.is_loaded:
            self.load()

        LOGGER.info("Running WASB ball detection...")
        result = self._pipeline.run(video_path, max_frames=max_frames)

        ball_uv_px = result.ball_xy_px.astype(np.float32)

        if image_width is not None and image_height is not None:
            ball_uv = ball_uv_px.copy()
            ball_uv[..., 0] /= image_width
            ball_uv[..., 1] /= image_height
        else:
            ball_uv = ball_uv_px

        result = WASBResult(
            ball_uv=ball_uv,
            ball_uv_px=ball_uv_px,
            visibility=result.visibility,
            score=result.score.astype(np.float32),
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result


if __name__ == "__main__":
    # Quick smoke test for module instantiation
    print("WASBModule: ball detection module")
    print("Use WASBModule(WASBConfig(...)) to create")

    # Test config creation
    config = WASBConfig(
        checkpoint="test.ckpt",
        device="cpu",
        save_result=True,
        output_path="test_output.json",
    )
    print(f"Config: {config}")
    assert config.device == "cpu"
    print("Smoke test passed.")
