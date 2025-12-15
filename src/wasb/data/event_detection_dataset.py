"""Trajectory event-detection dataset for WASB.

This dataset provides fixed-length windows of ball trajectories (x, y) together
with the per-frame `status` label from `Label.csv`.

Status convention:
    - 0: none
    - 1: shot
    - 2: bounce
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from src.wasb.data.trajectory_dataset import TrajectoryWindow, build_trajectory_windows


class TrajectoryEventWindowDataset(Dataset):
    """Windowed trajectory dataset that yields per-frame event labels."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        matches: Sequence[str],
        sequence_length: int,
        step: int = 1,
        image_ext: str = ".jpg",
        csv_filename: str = "Label.csv",
        min_visible_per_window: int = 1,
        xy_scale: tuple[float, float] = (1920.0, 1080.0),
        ignore_invisible_targets: bool = True,
        ignore_index: int = -100,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if step <= 0:
            raise ValueError("step must be positive")

        self.root_dir = Path(root_dir)
        self.matches = list(matches)
        self.sequence_length = int(sequence_length)
        self.step = int(step)
        self.image_ext = image_ext
        self.csv_filename = csv_filename
        self.min_visible_per_window = int(min_visible_per_window)
        self.xy_scale = (float(xy_scale[0]), float(xy_scale[1]))
        self.ignore_invisible_targets = bool(ignore_invisible_targets)
        self.ignore_index = int(ignore_index)

        self.windows = build_trajectory_windows(
            root_dir=self.root_dir,
            matches=self.matches,
            sequence_length=self.sequence_length,
            step=self.step,
            image_ext=self.image_ext,
            csv_filename=self.csv_filename,
            min_visible_per_window=self.min_visible_per_window,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def iter_all_targets(self) -> Iterable[torch.Tensor]:
        """Iterate per-window target status tensors (useful for statistics)."""
        for w in self.windows:
            yield torch.tensor([r.status for r in w.labels], dtype=torch.int64)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        window: TrajectoryWindow = self.windows[index]
        labels = window.labels
        L = len(labels)

        xy = torch.tensor([[r.x, r.y] for r in labels], dtype=torch.float32)
        visibility = torch.tensor([r.visibility for r in labels], dtype=torch.int64)
        target_status = torch.tensor([r.status for r in labels], dtype=torch.int64)

        scale = torch.tensor(self.xy_scale, dtype=torch.float32)
        xy_norm = xy / scale

        if self.ignore_invisible_targets:
            target_status = target_status.clone()
            target_status[visibility <= 0] = self.ignore_index

        return {
            "xy_norm": xy_norm,  # (L, 2)
            "visibility": visibility,  # (L,)
            "target_status": target_status,  # (L,)
            "match": window.match,
            "clip": window.clip,
        }

