"""BLCS module for 3D ball localization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.tennis_scene.pipeline.components.base import BasePipelineModule
from src.utils.io import load_json, save_json

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tasks.blcs.inference.predictor import BLCSPredictor

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
    def from_dict(cls, data: dict) -> BLCSResult:
        """Create result from dict."""
        return cls(
            ball_3d=np.array(data["ball_3d"], dtype=np.float32),
            visibility=np.array(data["visibility"], dtype=np.bool_)
            if "visibility" in data
            else None,
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        save_json(self.to_dict(), path)
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
    def load(cls, path: str | Path) -> BLCSResult:
        """Load result from JSON file."""
        return cls.from_dict(load_json(path))


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
        self._predictor: BLCSPredictor | None = None

    def load(self) -> None:
        """Load the BLCS predictor."""
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading BLCS model from {self.checkpoint_path}")

        from src.tasks.blcs.inference.predictor import BLCSPredictor

        self._predictor = BLCSPredictor.load_from_checkpoint(
            self.checkpoint_path, device=self.device
        )
        self._validate_pipeline_checkpoint_profile()

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._predictor is not None

    def _validate_pipeline_checkpoint_profile(self) -> None:
        """Reject single-view BLCS checkpoints before pipeline tensor assembly."""
        if self._predictor is None:
            raise RuntimeError("BLCS predictor is not loaded")

        from src.tasks.blcs.models import BLCSMultiViewAxialModel, BLCSMultiViewModel

        supported = (BLCSMultiViewModel, BLCSMultiViewAxialModel)
        model = self._predictor.model
        if not isinstance(model, supported):
            raise ValueError(
                "tennis_scene BLCS pipeline requires a multiview BLCS checkpoint "
                "(model.io.input_profile=multiview) because it passes tensors as "
                "(B, N, T, 2). "
                f"Loaded model class {model.__class__.__name__!r} is not supported; "
                "single-view checkpoints must not be used here."
            )

    def process(
        self,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_] | None = None,
        court_vis: NDArray[np.float32] | None = None,
    ) -> BLCSResult:
        """Run BLCS inference.

        Args:
            ball_uv: Ball 2D positions (N, T, 2), normalized [0, 1].
            court_kp: Court keypoints, shape (N, T, K, 2), normalized [0, 1].
            ball_vis: Ball visibility mask (N, T).
            court_vis: Court keypoint visibility, shape (N, T, K).

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
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("BLCS predictor is not loaded")

        LOGGER.info("Running BLCS ball localization...")

        if ball_uv.ndim != 3 or ball_uv.shape[-1] != 2:
            raise ValueError(f"ball_uv must have shape (N, T, 2), got {ball_uv.shape}")
        num_cameras, num_frames = ball_uv.shape[:2]

        if ball_vis is not None:
            if ball_vis.shape != (num_cameras, num_frames):
                raise ValueError(
                    "ball_vis must have shape (N, T), "
                    f"got {ball_vis.shape} for {(num_cameras, num_frames)}"
                )
            effective_vis = ball_vis.astype(np.bool_)
        else:
            effective_vis = np.ones((num_cameras, num_frames), dtype=bool)

        if court_kp.ndim != 4 or court_kp.shape[-1] != 2:
            raise ValueError(
                f"court_kp must have shape (N, T, K, 2), got {court_kp.shape}"
            )
        if court_kp.shape[:2] != (num_cameras, num_frames):
            raise ValueError(
                "court_kp leading shape must match ball_uv (N, T), "
                f"got {court_kp.shape[:2]} and {(num_cameras, num_frames)}"
            )
        if court_vis is not None:
            if court_vis.ndim != 3:
                raise ValueError(
                    f"court_vis must have shape (N, T, K), got {court_vis.shape}"
                )
            if court_vis.shape[:2] != (num_cameras, num_frames):
                raise ValueError(
                    "court_vis leading shape must match ball_uv (N, T), "
                    f"got {court_vis.shape[:2]} and {(num_cameras, num_frames)}"
                )

        # BLCS models expect batched inputs:
        # (B, N, T, 2), (B, N, T, K, 2), (B, N, T), and optional court_vis.
        ball_uv_t = torch.from_numpy(ball_uv).float().unsqueeze(0)
        court_kp_t = torch.from_numpy(court_kp).float().unsqueeze(0)

        ball_vis_t = torch.from_numpy(effective_vis.astype(np.float32)).unsqueeze(0)
        ball_mask_t = torch.ones_like(ball_vis_t)

        court_vis_t = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float().unsqueeze(0)

        pred = predictor.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_vis=ball_vis_t,
            ball_mask=ball_mask_t,
            court_vis=court_vis_t,
            denormalize=True,
        )

        ball_3d = pred["position"].squeeze(0).cpu().numpy().astype(np.float32)
        if ball_3d.shape != (num_frames, 3):
            raise ValueError(
                f"BLCS predictor position must have shape (T, 3), got {ball_3d.shape}"
            )
        output_visibility = effective_vis.any(axis=0)

        # Mask out invalid frames with zeros to keep JSON strictly numeric
        ball_3d[~output_visibility] = 0.0

        result = BLCSResult(ball_3d=ball_3d, visibility=output_visibility)

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
