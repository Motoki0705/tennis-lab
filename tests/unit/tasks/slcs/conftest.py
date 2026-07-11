"""Shared fixtures for SLCS unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.slcs.data.contract import DatasetIndex
from src.tasks.slcs.data.dataset import SLCSDataConfig
from src.tasks.slcs.data.dino_tokens import DinoTokenSpec
from src.tasks.slcs.data.splits import generate_recording_splits, save_split_file
from src.tasks.slcs.data.synthetic import (
    DEFAULT_TEST_DINO_SPEC,
    SyntheticDatasetConfig,
    build_synthetic_dataset,
)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> DatasetIndex:
    """A small contract-conformant dataset, built once per test session."""
    root = tmp_path_factory.mktemp("slcs_dataset")
    return build_synthetic_dataset(root, SyntheticDatasetConfig())


@pytest.fixture(scope="session")
def synthetic_split_file(synthetic_dataset: DatasetIndex) -> Path:
    """Split file covering the synthetic dataset (1 recording per split)."""
    assignments = generate_recording_splits(
        synthetic_dataset, val_ratio=0.34, test_ratio=0.33, seed=0
    )
    path = synthetic_dataset.root / "splits.json"
    save_split_file(path, assignments, seed=0, val_ratio=0.34, test_ratio=0.33)
    return Path(path)


@pytest.fixture
def dino_spec() -> DinoTokenSpec:
    return DEFAULT_TEST_DINO_SPEC


@pytest.fixture
def data_config(dino_spec: DinoTokenSpec) -> SLCSDataConfig:
    """Data config matching the synthetic dataset dimensions."""
    return SLCSDataConfig(
        window_size=16,
        train_stride=8,
        eval_stride=16,
        num_players=2,
        num_court_kp=14,
        dino_spec=dino_spec,
    )
