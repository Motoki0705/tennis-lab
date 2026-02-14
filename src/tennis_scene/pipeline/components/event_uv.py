"""UV event detection module."""

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
class EventUVConfig:
    """Configuration for UV event detection module."""

    checkpoint_path: str | Path
    device: str = "cuda"
    threshold: float = 0.5
    min_distance: int = 1
    top_k: int | None = None
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class EventUVResult:
    """Result of UV event detection."""

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
    def from_dict(cls, data: dict[str, Any]) -> "EventUVResult":
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
        LOGGER.info(f"Saved UV event result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "EventUVResult":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class EventUVModule(BasePipelineModule):
    """UV event detection module."""

    def __init__(self, config: EventUVConfig) -> None:
        self.config = config
        self._predictor = None

    def load(self) -> None:
        """Load UV event detection predictor."""
        if self._predictor is not None:
            return

        from src.event_detection.inference import UVEventPredictor

        LOGGER.info(f"Loading UV event model from {self.config.checkpoint_path}")
        self._predictor = UVEventPredictor.load_from_checkpoint(
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
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_] | None = None,
        court_vis: NDArray[np.float32] | None = None,
    ) -> EventUVResult:
        """Run UV event detection.

        Args:
            ball_uv: Ball UV trajectory with shape (T, 2).
            court_kp: Court keypoints with shape (20, 2).
            ball_vis: Optional visibility mask with shape (T,).
            court_vis: Optional court visibility mask with shape (20,).

        Returns:
            Event probabilities, peak mask, and metadata.
        """
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(
                    f"Loading UV event result from {load_path} (skipping inference)"
                )
                return EventUVResult.load(load_path)
            LOGGER.warning(
                f"load_path specified but not found: {load_path}, running inference"
            )

        if not self.is_loaded:
            self.load()

        ball_uv_t = torch.from_numpy(ball_uv).float()
        court_kp_t = torch.from_numpy(court_kp).float()
        ball_vis_t = torch.from_numpy(ball_vis.astype(np.float32)) if ball_vis is not None else None
        court_vis_t = torch.from_numpy(court_vis).float() if court_vis is not None else None
        seq_len_t = torch.tensor([ball_uv.shape[0]], dtype=torch.long)

        pred = self._predictor.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_vis=ball_vis_t,
            court_vis=court_vis_t,
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

        result = EventUVResult(
            event_probs=event_probs,
            event_peak_mask=peak_mask,
            event_names=event_names,
            event_peaks=event_peaks,
            event_peak_scores=event_peak_scores,
        )

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result
