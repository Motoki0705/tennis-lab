"""PLCS module for 3D player localization."""

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

    from src.tasks.plcs.inference.predictor import PLCSPredictor

LOGGER = logging.getLogger(__name__)


@dataclass
class PLCSConfig:
    """Configuration for PLCS module."""

    checkpoint_path: str | Path
    device: str = "cuda"
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class PLCSResult:
    """Result of PLCS inference.

    Attributes:
        position: Player 3D positions in court coords, shape (P, T, 3), meters.
        yaw: Player yaw angles, shape (P, T), radians.
        track_ids: Track IDs aligned to player axis, shape (P,).
    """

    position: NDArray[np.float32]
    yaw: NDArray[np.float32]
    track_ids: NDArray[np.int32] | None = None

    def to_dict(self) -> dict:
        result = {
            "position": self.position.tolist(),
            "yaw": self.yaw.tolist(),
        }
        if self.track_ids is not None:
            result["track_ids"] = self.track_ids.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> PLCSResult:
        position = np.array(data["position"], dtype=np.float32)
        yaw = np.array(data["yaw"], dtype=np.float32)

        track_ids = data.get("track_ids")
        if track_ids is not None:
            track_ids = np.array(track_ids, dtype=np.int32)

        return cls(position=position, yaw=yaw, track_ids=track_ids)

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)
        LOGGER.info(f"Saved PLCS result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        errors: list[str] = []

        if self.position.ndim != 3 or self.position.shape[-1] != 3:
            errors.append(
                f"position shape must be (P, T, 3), got {self.position.shape}"
            )
        if self.yaw.ndim != 2:
            errors.append(f"yaw shape must be (P, T), got {self.yaw.shape}")

        if (
            self.position.ndim == 3
            and self.yaw.ndim == 2
            and self.position.shape[:2] != self.yaw.shape
        ):
            errors.append("position and yaw shapes are inconsistent on (P, T)")

        if self.track_ids is not None:
            if self.track_ids.ndim != 1:
                errors.append(
                    f"track_ids shape must be (P,), got {self.track_ids.shape}"
                )
            elif self.track_ids.shape[0] != self.position.shape[0]:
                errors.append("track_ids length does not match player count")

        if not np.isfinite(self.position).all():
            errors.append("position contains non-finite values")
        if not np.isfinite(self.yaw).all():
            errors.append("yaw contains non-finite values")

        return len(errors) == 0, errors

    @classmethod
    def load(cls, path: str | Path) -> PLCSResult:
        return cls.from_dict(load_json(path))


class PLCSModule(BasePipelineModule):
    """PLCS module for 3D player localization."""

    def __init__(self, config: PLCSConfig) -> None:
        self.config = config
        self.checkpoint_path = Path(self.config.checkpoint_path)
        self.device = self.config.device
        self._predictor: PLCSPredictor | None = None

    def load(self) -> None:
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading PLCS model from {self.checkpoint_path}")
        from src.tasks.plcs.inference.predictor import PLCSPredictor

        self._predictor = PLCSPredictor.load_from_checkpoint(
            self.checkpoint_path, device=self.device
        )
        self._validate_pipeline_checkpoint_profile()

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def _validate_pipeline_checkpoint_profile(self) -> None:
        """Reject single-view PLCS checkpoints before pipeline tensor assembly."""
        if self._predictor is None:
            raise RuntimeError("PLCS predictor is not loaded")

        from src.tasks.plcs.models import (
            PLCSMultiViewAxialCamTokenModel,
            PLCSMultiViewAxialModel,
            PLCSMultiViewAxialSplitModel,
            PLCSMultiViewModel,
        )

        supported = (
            PLCSMultiViewModel,
            PLCSMultiViewAxialModel,
            PLCSMultiViewAxialSplitModel,
            PLCSMultiViewAxialCamTokenModel,
        )
        model = self._predictor.model
        if not isinstance(model, supported):
            raise ValueError(
                "tennis_scene PLCS pipeline requires a multiview PLCS checkpoint "
                "(model.io.input_profile=multiview) because it passes tensors as "
                "(P, N, T, 17, 2). "
                f"Loaded model class {model.__class__.__name__!r} is not supported; "
                "frame/sequence single-view checkpoints must not be used here."
            )

    def process(
        self,
        human_kp_2d: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        human_kp_vis: NDArray[np.float32] | None = None,
        court_vis: NDArray[np.float32] | None = None,
        track_ids: NDArray[np.int32] | None = None,
    ) -> PLCSResult:
        """Run PLCS inference.

        Args:
            human_kp_2d: Human 2D keypoints, shape (P, N, T, 17, 2), normalized [0, 1].
            court_kp: Court keypoints, shape (N, T, K, 2), normalized [0, 1].
            human_kp_vis: Human keypoint visibility, shape (P, N, T, 17).
            court_vis: Court keypoint visibility, shape (N, T, K).
            track_ids: Optional track IDs aligned with P.

        Returns:
            PLCSResult with position/yaw in (P, T, ...).
        """
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(f"Loading PLCS result from {load_path} (skipping inference)")
                return PLCSResult.load(load_path)
            LOGGER.warning(
                f"load_path specified but not found: {load_path}, running inference"
            )

        if not self.is_loaded:
            self.load()
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("PLCS predictor is not loaded")

        if human_kp_2d.ndim != 5 or human_kp_2d.shape[3:] != (17, 2):
            raise ValueError(
                "human_kp_2d shape must be (P, N, T, 17, 2), "
                f"got {human_kp_2d.shape}"
            )

        num_players, num_cameras, num_frames = human_kp_2d.shape[:3]
        if human_kp_vis is not None and human_kp_vis.shape != (
            num_players,
            num_cameras,
            num_frames,
            17,
        ):
            raise ValueError(
                "human_kp_vis shape must match (P, N, T, 17), "
                f"got {human_kp_vis.shape}"
            )

        if track_ids is None:
            track_ids = np.arange(num_players, dtype=np.int32)

        LOGGER.info(
            "Running PLCS player localization for "
            f"{num_players} players and {num_cameras} cameras..."
        )

        if court_kp.ndim != 4 or court_kp.shape[-1] != 2:
            raise ValueError(
                f"court_kp must have shape (N, T, K, 2), got {court_kp.shape}"
            )
        if court_kp.shape[:2] != (num_cameras, num_frames):
            raise ValueError(
                "court_kp leading shape must match human_kp_2d (N, T), "
                f"got {court_kp.shape[:2]} and {(num_cameras, num_frames)}"
            )
        court_kp_t = (
            torch.from_numpy(court_kp)
            .float()
            .unsqueeze(0)
            .expand(num_players, *court_kp.shape)
        )

        court_vis_batch = None
        if court_vis is not None:
            if court_vis.ndim != 3:
                raise ValueError(
                    f"court_vis must have shape (N, T, K), got {court_vis.shape}"
                )
            if court_vis.shape[:2] != (num_cameras, num_frames):
                raise ValueError(
                    "court_vis leading shape must match human_kp_2d (N, T), "
                    f"got {court_vis.shape[:2]} and {(num_cameras, num_frames)}"
                )
            court_vis_batch = (
                torch.from_numpy(court_vis)
                .float()
                .unsqueeze(0)
                .expand(num_players, *court_vis.shape)
            )

        human_kp_t = torch.from_numpy(human_kp_2d).float()
        human_vis_t = None
        if human_kp_vis is not None:
            human_vis_t = torch.from_numpy(human_kp_vis).float()
        human_mask_t = torch.ones(
            (num_players, num_cameras, num_frames),
            dtype=torch.float32,
        )

        pred = predictor.predict(
            human_kp=human_kp_t,
            court_kp=court_kp_t,
            human_vis=human_vis_t,
            human_mask=human_mask_t,
            court_vis=court_vis_batch,
            denormalize=True,
        )

        positions = pred["position_meters"].numpy().astype(np.float32)
        yaws = pred["yaw_radians"].numpy().astype(np.float32)
        if positions.shape != (num_players, num_frames, 3):
            raise ValueError(
                "PLCS predictor position_meters must have shape (P, T, 3), "
                f"got {positions.shape}"
            )
        if yaws.shape != (num_players, num_frames):
            raise ValueError(
                "PLCS predictor yaw_radians must have shape (P, T), "
                f"got {yaws.shape}"
            )

        result = PLCSResult(
            position=positions,  # (P, T, 3)
            yaw=yaws,  # (P, T)
            track_ids=track_ids.astype(np.int32),
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result


if __name__ == "__main__":
    print("PLCSModule: 3D player localization module")
