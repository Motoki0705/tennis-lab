from __future__ import annotations

from pathlib import Path

import torch

from src.wasb.data.event_detection_dataset import TrajectoryEventWindowDataset
from src.wasb.data.trajectory_dataset import TrajectoryWindowDataset
from src.wasb.tennis_format import TennisLabelRow, save_label_csv


def _write_dummy_clip(root: Path) -> None:
    clip_dir = root / "game1" / "clip1"
    clip_dir.mkdir(parents=True, exist_ok=True)

    frame_names = [f"{i:04d}.jpg" for i in range(1, 9)]
    for name in frame_names:
        (clip_dir / name).write_bytes(b"")

    rows: list[TennisLabelRow] = []
    for i, name in enumerate(frame_names):
        # Create a sparse pattern of events with a visibility drop.
        status = 0
        if i == 2:
            status = 1
        if i == 5:
            status = 2
        visibility = 1 if i != 4 else 0
        rows.append(TennisLabelRow(file_name=name, visibility=visibility, x=100.0, y=200.0, status=status, score=0.9))
    save_label_csv(clip_dir / "Label.csv", rows)


def test_trajectory_window_dataset_exposes_status(tmp_path: Path) -> None:
    _write_dummy_clip(tmp_path)
    ds = TrajectoryWindowDataset(
        root_dir=tmp_path,
        matches=["game1"],
        sequence_length=4,
        step=1,
        min_visible_per_window=1,
        block_mask_min_len=0,
        block_mask_max_len=0,
        sparse_mask_prob=0.0,
        noise_prob=0.0,
    )
    sample = ds[0]
    assert "status" in sample
    status = sample["status"]
    assert isinstance(status, torch.Tensor)
    assert status.shape == (4,)


def test_event_detection_dataset_ignores_invisible_targets(tmp_path: Path) -> None:
    _write_dummy_clip(tmp_path)
    ds = TrajectoryEventWindowDataset(
        root_dir=tmp_path,
        matches=["game1"],
        sequence_length=8,
        step=8,
        min_visible_per_window=1,
        ignore_invisible_targets=True,
        ignore_index=-100,
    )
    sample = ds[0]
    target = sample["target_status"]
    visibility = sample["visibility"]
    assert isinstance(target, torch.Tensor)
    assert isinstance(visibility, torch.Tensor)
    assert (target[visibility <= 0] == -100).all()

