"""Canonical compact BLCS task-consumer tests."""

from pathlib import Path

import numpy as np

from src.synthetic_data_generation.dataset.blcs.assembler import (
    BLCSAllViewTrajectory,
    BLCSTrackIndex,
    BLCSTrajectoryIndex,
)
from src.tasks.blcs.data import dataset as dataset_module


class _Reader:
    def __init__(self, _directory: Path) -> None:
        track = BLCSTrackIndex("trajectory", "train", "ball", 0, 1, 4)
        self.index = BLCSTrajectoryIndex(
            "trajectory", "train", 4, ("camera-a", "camera-b"), (track,)
        )

    def split_tracks(self, _split: str) -> tuple[BLCSTrackIndex, ...]:
        return self.index.tracks

    def materialize_all_views(self, _trajectory_id: str) -> BLCSAllViewTrajectory:
        return BLCSAllViewTrajectory(
            index=self.index,
            ball_uv=np.full((2, 4, 1, 2), 0.5, dtype=np.float32),
            ball_visible=np.ones((2, 4, 1), dtype=np.bool_),
            court_kp=np.full((2, 20, 2), 0.5, dtype=np.float32),
            court_visible=np.ones((2, 20), dtype=np.bool_),
            positions_court_m=np.ones((4, 1, 3), dtype=np.float32),
            velocities_court_mps=np.ones((4, 1, 3), dtype=np.float32),
            present=np.ones((4, 1), dtype=np.bool_),
            camera_R=np.stack((np.eye(3), np.eye(3))).astype(np.float32),
            camera_C=np.zeros((2, 3), dtype=np.float32),
            camera_f=np.ones(2, dtype=np.float32),
            camera_cx=np.ones(2, dtype=np.float32),
            camera_cy=np.ones(2, dtype=np.float32),
            camera_w=np.full(2, 32, dtype=np.float32),
            camera_h=np.full(2, 24, dtype=np.float32),
        )


def test_standard_dataset_retains_all_manifest_views(monkeypatch) -> None:
    monkeypatch.setattr(dataset_module, "BLCSCompactDatasetReader", _Reader)
    monkeypatch.setattr(
        dataset_module, "BLCSBallObservationAugmentation", lambda _config: object()
    )
    dataset = dataset_module.BallTrajectoryDataset(
        dataset_dir="unused",
        split="train",
        config={
            "data": {
                "seq_len_range": [3, 3],
                "num_court_kp": 20,
                "augmentation": {"enabled": False},
            }
        },
        augment=False,
    )

    sample = dataset[0]

    assert sample["ball_uv"].shape == (2, 3, 2)
    assert sample["court_kp"].shape == (2, 3, 20, 2)
    assert sample["seq_len"].item() == 3
