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
        load_path: Path to load pre-computed result from (skips inference).

    """

    checkpoint: str | Path
    batch_size: int = 64
    device: str = "cuda"
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


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

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if self.ball_uv.ndim != 2 or self.ball_uv.shape[1] != 2:
            errors.append(f"ball_uv shape must be (T, 2), got {self.ball_uv.shape}")
        if self.ball_uv_px.ndim != 2 or self.ball_uv_px.shape[1] != 2:
            errors.append(
                f"ball_uv_px shape must be (T, 2), got {self.ball_uv_px.shape}"
            )
        if self.visibility.ndim != 1:
            errors.append(f"visibility shape must be (T,), got {self.visibility.shape}")
        if self.score.ndim != 1:
            errors.append(f"score shape must be (T,), got {self.score.shape}")

        t_uv = self.ball_uv.shape[0]
        if self.ball_uv_px.shape[0] != t_uv:
            errors.append("ball_uv_px length does not match ball_uv length")
        if self.visibility.shape[0] != t_uv:
            errors.append("visibility length does not match ball_uv length")
        if self.score.shape[0] != t_uv:
            errors.append("score length does not match ball_uv length")

        if not np.isfinite(self.ball_uv).all():
            errors.append("ball_uv contains non-finite values")
        if not np.isfinite(self.ball_uv_px).all():
            errors.append("ball_uv_px contains non-finite values")
        if not np.isfinite(self.score).all():
            errors.append("score contains non-finite values")

        if not np.isin(self.visibility, [0, 1, False, True]).all():
            errors.append("visibility must contain only 0 or 1")

        tol = 1e-6
        if np.any(self.ball_uv < -tol) or np.any(self.ball_uv > 1.0 + tol):
            errors.append("ball_uv must be normalized to [0, 1]")
        if np.any(self.ball_uv_px < -tol):
            errors.append("ball_uv_px must be non-negative")
        if np.any(self.score < -tol):
            errors.append("score must be non-negative")

        if self.visibility.size:
            invalid = ~self.visibility.astype(bool)
            if invalid.any():
                if np.any(np.abs(self.ball_uv[invalid]) > tol):
                    errors.append("ball_uv must be zero for invalid frames")
                if np.any(np.abs(self.ball_uv_px[invalid]) > tol):
                    errors.append("ball_uv_px must be zero for invalid frames")
                if np.any(np.abs(self.score[invalid]) > tol):
                    errors.append("score must be zero for invalid frames")

        return len(errors) == 0, errors

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
        # Check if we should load from pre-computed result
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(f"Loading WASB result from {load_path} (skipping inference)")
                return WASBResult.load(load_path)
            else:
                LOGGER.warning(f"load_path specified but not found: {load_path}, running inference")

        if not self.is_loaded:
            self.load()

        LOGGER.info("Running WASB ball detection...")
        result = self._pipeline.run(video_path, max_frames=max_frames)

        ball_uv_px = result.ball_xy_px.astype(np.float32)
        visibility = result.visibility

        score = result.score.astype(np.float32)
        finite_uv = np.isfinite(ball_uv_px).all(axis=-1)
        finite_score = np.isfinite(score)
        valid_mask = visibility & finite_uv & finite_score

        # Replace invalid values with zeros to keep JSON strictly numeric
        ball_uv_px[~valid_mask] = 0.0
        score[~valid_mask] = 0.0

        if image_width is not None and image_height is not None:
            ball_uv = ball_uv_px.copy()
            ball_uv[..., 0] /= image_width
            ball_uv[..., 1] /= image_height
        else:
            ball_uv = ball_uv_px.copy()

        wasb_result = WASBResult(
            ball_uv=ball_uv,
            ball_uv_px=ball_uv_px,
            visibility=valid_mask.astype(np.bool_),
            score=score,
        )

        if self.config.save_result and self.config.output_path is not None:
            wasb_result.save(self.config.output_path)

        return wasb_result


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
