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

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if self.position.ndim != 2 or self.position.shape[1] != 3:
            errors.append(f"position shape must be (T, 3), got {self.position.shape}")
        if self.yaw.ndim != 1:
            errors.append(f"yaw shape must be (T,), got {self.yaw.shape}")
        if self.position.shape[0] != self.yaw.shape[0]:
            errors.append("position length does not match yaw length")
        if not np.isfinite(self.position).all():
            errors.append("position contains non-finite values")
        if not np.isfinite(self.yaw).all():
            errors.append("yaw contains non-finite values")
        if self.track_id is not None and self.track_id < 0:
            errors.append(f"track_id must be non-negative, got {self.track_id}")
        return len(errors) == 0, errors

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

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if not self.players:
            errors.append("players must not be empty")
            return False, errors
        for track_id, result in self.players.items():
            ok, result_errors = result.validate()
            if not ok:
                errors.extend([f"player {track_id}: {msg}" for msg in result_errors])
            if result.track_id is not None and result.track_id != track_id:
                errors.append(
                    f"player {track_id}: track_id mismatch ({result.track_id})"
                )
        return len(errors) == 0, errors

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

        multi = self.process_multi(
            human_kp_2d=human_kp_2d[None, ...],
            court_kp=court_kp,
            human_kp_vis=human_kp_vis[None, ...] if human_kp_vis is not None else None,
            court_vis=court_vis,
            track_ids=[0],
        )
        first = multi.get_first()
        if first is None:
            raise RuntimeError("PLCS returned no player result")
        result = first

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result

    def process_multi(
        self,
        human_kp_2d: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        human_kp_vis: NDArray[np.float32] | None = None,
        court_vis: NDArray[np.float32] | None = None,
        track_ids: list[int] | None = None,
    ) -> PLCSMultiResult:
        """Run batched PLCS inference for multiple players.

        Args:
            human_kp_2d: Human 2D keypoints (P, T, 17, 2), normalized [0, 1].
            court_kp: Court keypoints (20, 2), normalized [0, 1].
            human_kp_vis: Human keypoint visibility (P, T, 17).
            court_vis: Court keypoint visibility (20,).
            track_ids: Track IDs for players. If None, uses [0..P-1].

        Returns:
            PLCSMultiResult indexed by track_id.
        """
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(f"Loading PLCS result from {load_path} (skipping inference)")
                with load_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if "players" in data:
                    return PLCSMultiResult.from_dict(data)
                single = PLCSResult.from_dict(data)
                track_id = 0 if single.track_id is None else int(single.track_id)
                single.track_id = track_id
                return PLCSMultiResult(players={track_id: single})
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
        if track_ids is None:
            track_ids = list(range(num_players))
        if len(track_ids) != num_players:
            raise ValueError(
                f"track_ids length ({len(track_ids)}) must match num_players ({num_players})"
            )

        LOGGER.info(f"Running PLCS multi-player localization for {num_players} players...")

        court_kp_t = torch.from_numpy(court_kp).float()
        court_kp_batch = court_kp_t.unsqueeze(0).repeat(num_players, 1, 1)
        court_vis_batch = None
        if court_vis is not None:
            court_vis_t = torch.from_numpy(court_vis).float()
            court_vis_batch = court_vis_t.unsqueeze(0).repeat(num_players, 1)

        positions_per_frame: list[np.ndarray] = []
        yaws_per_frame: list[np.ndarray] = []

        for t in range(num_frames):
            human_kp_t = torch.from_numpy(human_kp_2d[:, t]).float()  # (P, 17, 2)
            human_vis_t = None
            if human_kp_vis is not None:
                human_vis_t = torch.from_numpy(human_kp_vis[:, t]).float()  # (P, 17)

            pred = self._predictor.predict(
                human_kp=human_kp_t,
                court_kp=court_kp_batch,
                human_vis=human_vis_t,
                court_vis=court_vis_batch,
                denormalize=True,
            )
            positions_per_frame.append(pred["position_meters"].numpy())  # (P, 3)
            yaws_per_frame.append(pred["yaw_radians"].numpy())  # (P,)

        positions = np.stack(positions_per_frame, axis=0).astype(np.float32)  # (T, P, 3)
        yaws = np.stack(yaws_per_frame, axis=0).astype(np.float32)  # (T, P)

        players: dict[int, PLCSResult] = {}
        for idx, track_id in enumerate(track_ids):
            players[int(track_id)] = PLCSResult(
                position=positions[:, idx, :],
                yaw=yaws[:, idx],
                track_id=int(track_id),
            )

        result = PLCSMultiResult(players=players)
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
