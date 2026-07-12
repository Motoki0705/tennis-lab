"""Tests for SLCS recording-level split policies."""

from pathlib import Path

from src.tasks.slcs.data.splits import generate_overfit_splits
from src.tasks.slcs.data.synthetic import (
    SyntheticDatasetConfig,
    build_synthetic_dataset,
)


def test_overfit_split_assigns_every_recording_to_train(tmp_path: Path) -> None:
    index = build_synthetic_dataset(
        tmp_path / "dataset",
        SyntheticDatasetConfig(recordings=("rec-a", "rec-b")),
    )

    assert generate_overfit_splits(index) == {"rec-a": "train", "rec-b": "train"}
