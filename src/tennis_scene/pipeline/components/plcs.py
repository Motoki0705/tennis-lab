"""PLCS module for 3D player localization."""

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
    def from_dict(cls, data: dict) -> "PLCSResult":
        position = np.array(data["position"], dtype=np.float32)
        yaw = np.array(data["yaw"], dtype=np.float32)

        track_ids = data.get("track_ids")
        if track_ids is not None:
            track_ids = np.array(track_ids, dtype=np.int32)

        return cls(position=position, yaw=yaw, track_ids=track_ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved PLCS result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        errors: list[str] = []

        if self.position.ndim != 3 or self.position.shape[-1] != 3:
            errors.append(
                f"position shape must be (P, T, 3), got {self.position.shape}"
            )
        if self.yaw.ndim != 2:
            errors.append(f"yaw shape must be (P, T), got {self.yaw.shape}")

        if self.position.ndim == 3 and self.yaw.ndim == 2:
            if self.position.shape[:2] != self.yaw.shape:
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
    def load(cls, path: str | Path) -> "PLCSResult":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class PLCSModule(BasePipelineModule):
    """PLCS module for 3D player localization."""

    def __init__(
        self,
        config: PLCSConfig | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        save_result: bool = False,
        output_path: str | Path | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            if checkpoint_path is None:
                raise ValueError("Either config or checkpoint_path must be provided")
            self.config = PLCSConfig(
                checkpoint_path=checkpoint_path,
                device=device,
                save_result=save_result,
                output_path=output_path,
            )
        self.checkpoint_path = Path(self.config.checkpoint_path)
        self.device = self.config.device
        self._predictor = None

    def load(self) -> None:
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading PLCS model from {self.checkpoint_path}")
        from src.tasks.plcs.inference.predictor import PLCSPredictor

        self._predictor = PLCSPredictor.load_from_checkpoint(
            self.checkpoint_path, device=self.device
        )

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

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
            human_kp_2d: Human 2D keypoints, shape (P, T, 17, 2), normalized [0, 1].
            court_kp: Court keypoints (20, 2), normalized [0, 1].
            human_kp_vis: Human keypoint visibility, shape (P, T, 17).
            court_vis: Court keypoint visibility (20,).
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

        if human_kp_2d.ndim != 4 or human_kp_2d.shape[2:] != (17, 2):
            raise ValueError(
                "human_kp_2d shape must be (P, T, 17, 2), "
                f"got {human_kp_2d.shape}"
            )

        num_players, num_frames = human_kp_2d.shape[:2]
        if human_kp_vis is not None:
            if human_kp_vis.shape != (num_players, num_frames, 17):
                raise ValueError(
                    "human_kp_vis shape must match (P, T, 17), "
                    f"got {human_kp_vis.shape}"
                )

        if track_ids is None:
            track_ids = np.arange(num_players, dtype=np.int32)

        LOGGER.info(f"Running PLCS player localization for {num_players} players...")

        batch_size = num_players * num_frames

        court_kp_t = torch.from_numpy(court_kp).float()
        court_kp_batch = court_kp_t.unsqueeze(0).repeat(batch_size, 1, 1)

        court_vis_batch = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float()
            court_vis_batch = court_vis_t.unsqueeze(0).repeat(batch_size, 1)

        human_kp_t = torch.from_numpy(human_kp_2d).float().reshape(
            batch_size, 17, 2
        )  # (P*T, 17, 2)
        human_vis_t = None
        if human_kp_vis is not None:
            human_vis_t = torch.from_numpy(human_kp_vis).float().reshape(
                batch_size, 17
            )  # (P*T, 17)
        human_mask_t = torch.ones((batch_size,), dtype=torch.float32)

        pred = self._predictor.predict(
            human_kp=human_kp_t,
            court_kp=court_kp_batch,
            human_vis=human_vis_t,
            human_mask=human_mask_t,
            court_vis=court_vis_batch,
            denormalize=True,
        )

        positions = pred["position_meters"].numpy().astype(np.float32).reshape(
            num_players, num_frames, 3
        )  # (P, T, 3)
        yaws = pred["yaw_radians"].numpy().astype(np.float32).reshape(
            num_players, num_frames
        )  # (P, T)

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
