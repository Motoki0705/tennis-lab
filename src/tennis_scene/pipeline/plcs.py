"""PLCS module for 3D player localization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.tennis_scene.pipeline.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class PLCSConfig:
    """Configuration for PLCS module.

    Attributes:
        checkpoint_path: Path to PLCS model checkpoint.
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
class PLCSResult:
    """Result of PLCS inference for a single player.

    Attributes:
        position: Player 3D position in court coords (T, 3), meters.
        yaw: Player yaw angle (T,), radians.
        track_id: Track ID for this player (optional).

    """

    position: NDArray[np.float32]
    yaw: NDArray[np.float32]
    track_id: int | None = None

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        result = {
            "position": self.position.tolist(),
            "yaw": self.yaw.tolist(),
        }
        if self.track_id is not None:
            result["track_id"] = self.track_id
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PLCSResult":
        """Create result from dict."""
        return cls(
            position=np.array(data["position"], dtype=np.float32),
            yaw=np.array(data["yaw"], dtype=np.float32),
            track_id=data.get("track_id"),
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved PLCS result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PLCSResult":
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Check if this is a multi-player result
            if "players" in data:
                multi = PLCSMultiResult.from_dict(data)
                if multi.players:
                    first_id = next(iter(multi.players))
                    return multi.players[first_id]
            return cls.from_dict(data)


@dataclass
class PLCSMultiResult:
    """Result of PLCS inference for multiple players.

    Attributes:
        players: Dict mapping track_id to PLCSResult.

    """

    players: dict[int, PLCSResult]

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "players": {str(k): v.to_dict() for k, v in self.players.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PLCSMultiResult":
        """Create result from dict."""
        players = {}
        for k, v in data.get("players", {}).items():
            track_id = int(k)
            result = PLCSResult.from_dict(v)
            result.track_id = track_id
            players[track_id] = result
        return cls(players=players)

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved PLCS multi-player result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PLCSMultiResult":
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def get_first(self) -> PLCSResult | None:
        """Get the first player result (for single-player compatibility)."""
        if self.players:
            first_id = next(iter(self.players))
            return self.players[first_id]
        return None


class PLCSModule(BasePipelineModule):
    """PLCS module for 3D player localization.

    Predicts player 3D position and yaw from 2D human keypoints
    and court keypoints.

    """

    def __init__(
        self,
        config: PLCSConfig | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        save_result: bool = False,
        output_path: str | Path | None = None,
    ) -> None:
        """Initialize the module.

        Args:
            config: PLCS configuration (preferred).
            checkpoint_path: Path to PLCS model checkpoint (legacy).
            device: Inference device (legacy).
            save_result: Whether to save result (legacy).
            output_path: Path to save result (legacy).

        """
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
        """Load the PLCS predictor."""
        if self._predictor is not None:
            return

        LOGGER.info(f"Loading PLCS model from {self.checkpoint_path}")

        from src.plcs.inference.predictor import PLCSPredictor

        self._predictor = PLCSPredictor.load_from_checkpoint(
            self.checkpoint_path, device=self.device
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._predictor is not None

    def process(
        self,
        human_kp_2d: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        human_kp_vis: NDArray[np.float32] | None = None,
        court_vis: NDArray[np.float32] | None = None,
    ) -> PLCSResult:
        """Run PLCS inference.

        Args:
            human_kp_2d: Human 2D keypoints (T, 17, 2), normalized [0, 1].
            court_kp: Court keypoints (20, 2), normalized [0, 1].
            human_kp_vis: Human keypoint visibility (T, 17).
            court_vis: Court keypoint visibility (20,).

        Returns:
            PLCSResult with 3D position and yaw.

        """
        # Check if we should load from pre-computed result
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(f"Loading PLCS result from {load_path} (skipping inference)")
                return PLCSResult.load(load_path)
            else:
                LOGGER.warning(f"load_path specified but not found: {load_path}, running inference")

        if not self.is_loaded:
            self.load()

        LOGGER.info("Running PLCS player localization...")

        T = len(human_kp_2d)
        positions = []
        yaws = []

        court_kp_t = torch.from_numpy(court_kp).float()
        court_vis_t = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float()

        for t in range(T):
            human_kp_t = torch.from_numpy(human_kp_2d[t]).float().unsqueeze(0)
            human_vis_t = None
            if human_kp_vis is not None:
                human_vis_t = torch.from_numpy(human_kp_vis[t]).float().unsqueeze(0)

            pred = self._predictor.predict(
                human_kp=human_kp_t,
                court_kp=court_kp_t.unsqueeze(0),
                human_vis=human_vis_t,
                court_vis=court_vis_t.unsqueeze(0) if court_vis_t is not None else None,
                denormalize=True,
            )

            positions.append(pred["position_meters"].squeeze(0).numpy())
            yaws.append(pred["yaw_radians"].item())

        result = PLCSResult(
            position=np.stack(positions, axis=0).astype(np.float32),
            yaw=np.array(yaws, dtype=np.float32),
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result


if __name__ == "__main__":
    # Quick smoke test for module instantiation
    print("PLCSModule: 3D player localization module")
    print("Use PLCSModule(PLCSConfig(...)) to create")

    # Test config creation
    config = PLCSConfig(
        checkpoint_path="test.ckpt",
        device="cpu",
        save_result=True,
        output_path="test_output.json",
    )
    print(f"Config: {config}")
    assert config.device == "cpu"
    print("Smoke test passed.")
