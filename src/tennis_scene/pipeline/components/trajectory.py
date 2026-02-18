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
    in_frame_threshold: float = 0.5
    cut_out_of_frame: bool = False
    use_in_frame_pred_for_visibility: bool = True
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class TrajectoryResult:
    """Result of trajectory completion inference."""

    ball_uv_pred: NDArray[np.float32]
    ball_uv_completed: NDArray[np.float32] | None
    in_frame_pred: NDArray[np.bool_] | None

    def to_dict(self) -> dict:
        data = {
            "ball_uv_pred": self.ball_uv_pred.tolist(),
        }
        data["ball_uv_completed"] = (
            self.ball_uv_completed.tolist()
            if self.ball_uv_completed is not None
            else None
        )
        data["in_frame_pred"] = (
            self.in_frame_pred.tolist() if self.in_frame_pred is not None else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrajectoryResult":
        in_frame_raw = data.get("in_frame_pred")
        if in_frame_raw is None:
            in_frame_raw = data.get("ball_obs_mask")
        in_frame = (
            np.asarray(in_frame_raw, dtype=np.bool_)
            if in_frame_raw is not None
            else None
        )
        return cls(
            ball_uv_pred=np.asarray(data["ball_uv_pred"], dtype=np.float32),
            ball_uv_completed=(
                np.asarray(data["ball_uv_completed"], dtype=np.float32)
                if data.get("ball_uv_completed") is not None
                else None
            ),
            in_frame_pred=in_frame,
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

        ball_uv_t = torch.from_numpy(ball_uv).float()
        ball_vis_t = (
            torch.from_numpy(ball_vis.astype(np.float32))
            if ball_vis is not None
            else None
        )
        ball_mask_t = torch.ones((ball_uv_t.shape[0],), dtype=torch.float32)
        court_kp_t = torch.from_numpy(court_kp).float()
        court_vis_t = torch.from_numpy(court_vis).float() if court_vis is not None else None

        pred = self._predictor.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_vis=ball_vis_t,
            ball_mask=ball_mask_t,
            court_vis=court_vis_t,
            merge_observed=self.config.merge_observed,
            in_frame_threshold=float(self.config.in_frame_threshold),
            cut_out_of_frame=bool(self.config.cut_out_of_frame),
        )

        ball_uv_pred = pred["ball_uv_pred"].squeeze(0).numpy().astype(np.float32)
        if "ball_uv_completed" in pred:
            ball_uv_completed = pred["ball_uv_completed"].squeeze(0).numpy().astype(np.float32)
        else:
            ball_uv_completed = None
        if "in_frame_pred" in pred:
            in_frame_pred = pred["in_frame_pred"].squeeze(0).numpy().astype(np.bool_)
        else:
            in_frame_pred = None

        result = TrajectoryResult(
            ball_uv_pred=ball_uv_pred,
            ball_uv_completed=ball_uv_completed,
            in_frame_pred=in_frame_pred,
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result
