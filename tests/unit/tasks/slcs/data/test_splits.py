"""Tests for SLCS recording-level split policies."""

from pathlib import Path

from src.tasks.slcs.data.splits import generate_overfit_splits
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


def test_overfit_split_assigns_every_recording_to_train(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset",
        SLCSFixtureDatasetConfig(recordings=("rec-a", "rec-b")),
    )

    assert generate_overfit_splits(index) == {"rec-a": "train", "rec-b": "train"}
