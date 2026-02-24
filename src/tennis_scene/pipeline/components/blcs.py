"""BLCS module for 3D ball localization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.tennis_scene.pipeline.components.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class BLCSConfig:
    """Configuration for BLCS module.

    Attributes:
        checkpoint_path: Path to BLCS model checkpoint.
        device: Inference device.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).

    """

    checkpoint_path: str | Path
    device: str = "cuda"
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class BLCSResult:
    """Result of BLCS inference.

    Attributes:
        ball_3d: Ball 3D position in court coords (T, 3), meters.
        visibility: Ball visibility mask (T,).

    """

    ball_3d: NDArray[np.float32]
    visibility: NDArray[np.bool_] | None = None

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        data = {"ball_3d": self.ball_3d.tolist()}
        if self.visibility is not None:
            data["visibility"] = self.visibility.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "BLCSResult":
        """Create result from dict."""
        return cls(
            ball_3d=np.array(data["ball_3d"], dtype=np.float32),
            visibility=np.array(data["visibility"], dtype=np.bool_)
            if "visibility" in data
            else None,
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved BLCS result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.
        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if self.ball_3d.ndim != 2 or self.ball_3d.shape[1] != 3:
            errors.append(f"ball_3d shape must be (T, 3), got {self.ball_3d.shape}")
        if not np.isfinite(self.ball_3d).all():
            errors.append("ball_3d contains non-finite values")
        if self.visibility is not None:
            if self.visibility.ndim != 1:
                errors.append(
                    f"visibility shape must be (T,), got {self.visibility.shape}"
                )
            if self.visibility.shape[0] != self.ball_3d.shape[0]:
                errors.append("visibility length does not match ball_3d length")
            if not np.isin(self.visibility, [0, 1, False, True]).all():
                errors.append("visibility must contain only 0 or 1")
            invalid = ~self.visibility.astype(bool)
            if invalid.any():
                tol = 1e-6
                if np.any(np.abs(self.ball_3d[invalid]) > tol):
                    errors.append("ball_3d must be zero for invalid frames")
        return len(errors) == 0, errors

    @classmethod
    def load(cls, path: str | Path) -> "BLCSResult":
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class BLCSModule(BasePipelineModule):
    """BLCS module for 3D ball localization.

    Predicts ball 3D trajectory from 2D ball positions
    and court keypoints.

    """

    def __init__(
        self,
        config: BLCSConfig,
    ) -> None:
        """Initialize the module.

        Args:
            config: BLCS configuration.

        """
        self.config = config
        self.checkpoint_path = Path(self.config.checkpoint_path)
        self.device = self.config.device
        self._predictor = None

    def load(self) -> None:
        """Load the BLCS predictor."""
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading BLCS model from {self.checkpoint_path}")

        from src.tasks.blcs.inference.predictor import BLCSPredictor

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
        # Check if we should load from pre-computed result
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(f"Loading BLCS result from {load_path} (skipping inference)")
                return BLCSResult.load(load_path)
            else:
                LOGGER.warning(f"load_path specified but not found: {load_path}, running inference")

        if not self.is_loaded:
            self.load()

        LOGGER.info("Running BLCS ball localization...")

        if ball_vis is not None:
            effective_vis = ball_vis.astype(np.bool_)
        else:
            effective_vis = np.ones(len(ball_uv), dtype=bool)

        # BLCS models expect batched inputs: (B, T, 2), (B, 20, 2), (B, T), (B, 20).
        ball_uv_t = torch.from_numpy(ball_uv).float().unsqueeze(0)
        court_kp_t = torch.from_numpy(court_kp).float().unsqueeze(0)

        ball_vis_t = torch.from_numpy(effective_vis.astype(np.float32)).unsqueeze(0)

        court_vis_t = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float().unsqueeze(0)

        pred = self._predictor.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_vis=ball_vis_t,
            court_vis=court_vis_t,
            denormalize=True,
        )

        ball_3d = pred["position"].squeeze(0).cpu().numpy().astype(np.float32)

        # Mask out invalid frames with zeros to keep JSON strictly numeric
        ball_3d[~effective_vis] = 0.0

        result = BLCSResult(ball_3d=ball_3d, visibility=effective_vis)

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result


if __name__ == "__main__":
    # Quick smoke test for module instantiation
    print("BLCSModule: 3D ball localization module")
    print("Use BLCSModule(BLCSConfig(...)) to create")

    # Test config creation
    config = BLCSConfig(
        checkpoint_path="test.ckpt",
        device="cpu",
        save_result=True,
        output_path="test_output.json",
    )
    print(f"Config: {config}")
    assert config.device == "cpu"
    print("Smoke test passed.")
