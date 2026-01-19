"""BLCS module for 3D ball localization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.tennis_scene.pipeline.base import BasePipelineModule

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

    """

    checkpoint_path: str | Path
    device: str = "cuda"
    save_result: bool = False
    output_path: str | Path | None = None


@dataclass
class BLCSResult:
    """Result of BLCS inference.

    Attributes:
        ball_3d: Ball 3D position in court coords (T, 3), meters.

    """

    ball_3d: NDArray[np.float32]

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "ball_3d": self.ball_3d.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BLCSResult":
        """Create result from dict."""
        return cls(
            ball_3d=np.array(data["ball_3d"], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved BLCS result to {path}")

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
        config: BLCSConfig | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        save_result: bool = False,
        output_path: str | Path | None = None,
    ) -> None:
        """Initialize the module.

        Args:
            config: BLCS configuration (preferred).
            checkpoint_path: Path to BLCS model checkpoint (legacy).
            device: Inference device (legacy).
            save_result: Whether to save result (legacy).
            output_path: Path to save result (legacy).

        """
        if config is not None:
            self.config = config
        else:
            if checkpoint_path is None:
                raise ValueError("Either config or checkpoint_path must be provided")
            self.config = BLCSConfig(
                checkpoint_path=checkpoint_path,
                device=device,
                save_result=save_result,
                output_path=output_path,
            )
        self.checkpoint_path = Path(self.config.checkpoint_path)
        self.device = self.config.device
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

        ball_3d = pred["position"].squeeze(0).cpu().numpy().astype(np.float32)

        result = BLCSResult(ball_3d=ball_3d)

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
