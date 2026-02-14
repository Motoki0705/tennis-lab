"""Trajectory completion module for ball UV sequences."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.tennis_scene.pipeline.components.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory completion module."""

    checkpoint_path: str | Path
    device: str = "cuda"
    merge_observed: bool = True
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class TrajectoryResult:
    """Result of trajectory completion inference."""

    ball_uv_pred: NDArray[np.float32]
    ball_uv_completed: NDArray[np.float32]
    ball_obs_mask: NDArray[np.bool_]

    def to_dict(self) -> dict:
        return {
            "ball_uv_pred": self.ball_uv_pred.tolist(),
            "ball_uv_completed": self.ball_uv_completed.tolist(),
            "ball_obs_mask": self.ball_obs_mask.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrajectoryResult":
        return cls(
            ball_uv_pred=np.asarray(data["ball_uv_pred"], dtype=np.float32),
            ball_uv_completed=np.asarray(data["ball_uv_completed"], dtype=np.float32),
            ball_obs_mask=np.asarray(data["ball_obs_mask"], dtype=np.bool_),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved trajectory result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryResult":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class TrajectoryModule(BasePipelineModule):
    """Trajectory completion module."""

    def __init__(self, config: TrajectoryConfig) -> None:
        self.config = config
        self._predictor = None

    def load(self) -> None:
        """Load trajectory completion predictor."""
        if self._predictor is not None:
            return

        from src.trajectory_completion.inference import UVTrajectoryCompletionPredictor

        LOGGER.info(f"Loading trajectory model from {self.config.checkpoint_path}")
        self._predictor = UVTrajectoryCompletionPredictor.load_from_checkpoint(
            self.config.checkpoint_path,
            device=self.config.device,
        )

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def process(
        self,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_] | None = None,
        court_vis: NDArray[np.float32] | None = None,
    ) -> TrajectoryResult:
        """Run UV trajectory completion.

        Args:
            ball_uv: Ball UV trajectory with shape (T, 2).
            court_kp: Court keypoints with shape (20, 2).
            ball_vis: Optional visibility mask with shape (T,).
            court_vis: Optional court visibility mask with shape (20,).

        Returns:
            Completed trajectory outputs.
        """
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(
                    f"Loading trajectory result from {load_path} (skipping inference)"
                )
                return TrajectoryResult.load(load_path)
            LOGGER.warning(
                f"load_path specified but not found: {load_path}, running inference"
            )

        if not self.is_loaded:
            self.load()

        obs_mask = (
            ball_vis.astype(np.bool_)
            if ball_vis is not None
            else np.ones((ball_uv.shape[0],), dtype=np.bool_)
        )
        ball_uv_t = torch.from_numpy(ball_uv).float()
        ball_vis_t = torch.from_numpy(obs_mask.astype(np.float32))
        ball_mask_t = torch.ones_like(ball_vis_t)
        court_kp_t = torch.from_numpy(court_kp).float()
        court_vis_t = torch.from_numpy(court_vis).float() if court_vis is not None else None

        pred = self._predictor.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_vis=ball_vis_t,
            ball_mask=ball_mask_t,
            court_vis=court_vis_t,
            merge_observed=self.config.merge_observed,
        )

        ball_uv_pred = pred["ball_uv_pred"].squeeze(0).numpy().astype(np.float32)
        if "ball_uv_completed" in pred:
            ball_uv_completed = pred["ball_uv_completed"].squeeze(0).numpy().astype(np.float32)
        else:
            ball_uv_completed = ball_uv_pred.copy()

        result = TrajectoryResult(
            ball_uv_pred=ball_uv_pred,
            ball_uv_completed=ball_uv_completed,
            ball_obs_mask=obs_mask,
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result
