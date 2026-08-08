"""Canonical compact PLCS task-consumer tests."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.plcs.assembler import PLCSSupervisionArrays
from src.synthetic_data_generation.dataset.plcs.validation import (
    PLCSAllViewScene,
    PLCSSceneIndex,
    PLCSTrackIndex,
)
from src.tasks.plcs.data import dataset as dataset_module


class _Reader:
    def __init__(self, _directory: Path) -> None:
        track = PLCSTrackIndex("scene", "train", "player", 0, 1, 4)
        self.index = PLCSSceneIndex(
            "scene", "train", 4, ("camera-a", "camera-b"), ("player",), (track,)
        )

    def split_tracks(self, _split: str) -> tuple[PLCSTrackIndex, ...]:
        return self.index.tracks

    def materialize_all_views(self, _scene_id: str) -> PLCSAllViewScene:
        present: NDArray[np.bool_] = np.ones((4, 1), dtype=np.bool_)
        return PLCSAllViewScene(
            index=self.index,
            supervision=PLCSSupervisionArrays(
                human_kp=np.full((4, 2, 1, 17, 2), 0.5, dtype=np.float32),
                human_vis=np.ones((4, 2, 1, 17), dtype=np.bool_),
                court_kp=np.full((4, 2, 20, 2), 0.5, dtype=np.float32),
                court_vis=np.ones((4, 2, 20), dtype=np.bool_),
                human_mask=np.ones((4, 2, 1), dtype=np.bool_),
                position=np.zeros((4, 1, 3), dtype=np.float32),
                position_court_m=np.zeros((4, 1, 3), dtype=np.float32),
                rotation=np.broadcast_to(
                    np.asarray((1.0, 0.0), dtype=np.float32), (4, 1, 2)
                ).copy(),
                present=present,
                human_kp_3d=np.zeros((4, 1, 17, 3), dtype=np.float32),
                canonical_pose_3d=np.zeros((4, 1, 52, 3), dtype=np.float32),
            ),
        )


def test_standard_dataset_retains_all_manifest_views(monkeypatch) -> None:
    monkeypatch.setattr(dataset_module, "PLCSCompactDatasetReader", _Reader)
    monkeypatch.setattr(
        dataset_module, "PLCSObservationAugmentation", lambda _config: object()
    )
    dataset = dataset_module.SceneDataset(
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

    assert sample["human_kp"].shape == (2, 3, 17, 2)
    assert sample["court_kp"].shape == (2, 3, 20, 2)
    assert sample["human_mask"].all()
