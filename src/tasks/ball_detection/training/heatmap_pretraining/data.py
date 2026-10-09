from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.data.coordinate_dataset import CoordinateWindowDataset
from src.tasks.ball_detection.data.supervision import (
    OBSERVED_ONLY,
    resolve_frame_supervision,
)


class HeatmapWindowDataset(CoordinateWindowDataset):
    """Same frozen play windows; add known-absence supervision, never fake unknown negatives."""
    def __init__(self, manifest: Path, *, split: str, jpeg_decoder: str = "opencv") -> None:
        super().__init__(manifest, split=split, requires_pose=False, jpeg_decoder=jpeg_decoder)
        self._heatmap_supervision = resolve_frame_supervision(self.store, OBSERVED_ONLY).supervised

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample: dict[str, Any] = super().__getitem__(index)
        clip = self.store.clip_by_id(sample["clip_id"])
        states = self._states[clip.clip_id]
        indices = sample["frame_indices"].numpy()
        rows = self.store.clip_rows(clip)[indices]
        # Preserve the single-ball contract of the frozen coordinate dataset.
        # Ambiguous multi-instance frames are ignored, never made blank negatives.
        valid = self._heatmap_supervision[rows] & states.target[indices] & (states.count[indices] <= 1)
        sample["heatmap_valid"] = torch.from_numpy(valid.copy())
        return sample
