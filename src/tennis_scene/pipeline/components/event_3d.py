"""3D trajectory event detection module."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.tennis_scene.pipeline.components.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class Event3DConfig:
    """Configuration for 3D trajectory event detection module."""

    checkpoint_path: str | Path
    device: str = "cuda"
    threshold: float = 0.5
    min_distance: int = 1
    top_k: int | None = None
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class Event3DResult:
    """Result of 3D trajectory event detection."""

    event_probs: NDArray[np.float32]
    event_peak_mask: NDArray[np.bool_]
    event_names: list[str]
    event_peaks: list[list[list[int]]]
    event_peak_scores: list[list[list[float]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_probs": self.event_probs.tolist(),
            "event_peak_mask": self.event_peak_mask.tolist(),
            "event_names": self.event_names,
            "event_peaks": self.event_peaks,
            "event_peak_scores": self.event_peak_scores,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event3DResult":
        return cls(
            event_probs=np.asarray(data["event_probs"], dtype=np.float32),
            event_peak_mask=np.asarray(data["event_peak_mask"], dtype=np.bool_),
            event_names=[str(x) for x in data["event_names"]],
            event_peaks=[
                [[int(v) for v in per_event] for per_event in per_batch]
                for per_batch in data["event_peaks"]
            ],
            event_peak_scores=[
                [[float(v) for v in per_event] for per_event in per_batch]
                for per_batch in data["event_peak_scores"]
            ],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved 3D event result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Event3DResult":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class Event3DModule(BasePipelineModule):
    """3D trajectory event detection module."""

    def __init__(self, config: Event3DConfig) -> None:
        self.config = config
        self._predictor = None

    def load(self) -> None:
        """Load 3D event detection predictor."""
        if self._predictor is not None:
            return

        from src.event_detection.inference import Traj3DEventPredictor

        LOGGER.info(f"Loading 3D event model from {self.config.checkpoint_path}")
        self._predictor = Traj3DEventPredictor.load_from_checkpoint(
            self.config.checkpoint_path,
            device=self.config.device,
        )

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def _peaks_to_mask(
        self,
        peaks: list[list[list[int]]],
        batch_size: int,
        num_frames: int,
        num_events: int,
    ) -> NDArray[np.bool_]:
        mask = np.zeros((batch_size, num_frames, num_events), dtype=np.bool_)
        for b in range(batch_size):
            for e in range(num_events):
                for t in peaks[b][e]:
                    if 0 <= t < num_frames:
                        mask[b, t, e] = True
        return mask

    def process(
        self,
        ball_3d: NDArray[np.float32],
    ) -> Event3DResult:
        """Run 3D trajectory event detection.

        Args:
            ball_3d: Ball 3D trajectory with shape (T, 3).

        Returns:
            Event probabilities, peak mask, and metadata.
        """
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(
                    f"Loading 3D event result from {load_path} (skipping inference)"
                )
                return Event3DResult.load(load_path)
            LOGGER.warning(
                f"load_path specified but not found: {load_path}, running inference"
            )

        if not self.is_loaded:
            self.load()

        ball_3d_t = torch.from_numpy(ball_3d).float()
        seq_len_t = torch.tensor([ball_3d.shape[0]], dtype=torch.long)

        pred = self._predictor.predict(
            ball_pos_world=ball_3d_t,
            seq_len=seq_len_t,
            threshold=self.config.threshold,
            min_distance=self.config.min_distance,
            top_k=self.config.top_k,
        )

        event_probs = pred["event_probs"].numpy().astype(np.float32)
        event_peaks = pred["event_peaks"]
        event_peak_scores = pred["event_peak_scores"]
        event_names = [str(x) for x in pred["event_names"]]
        peak_mask = self._peaks_to_mask(
            peaks=event_peaks,
            batch_size=event_probs.shape[0],
            num_frames=event_probs.shape[1],
            num_events=event_probs.shape[2],
        )

        result = Event3DResult(
            event_probs=event_probs,
            event_peak_mask=peak_mask,
            event_names=event_names,
            event_peaks=event_peaks,
            event_peak_scores=event_peak_scores,
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result
