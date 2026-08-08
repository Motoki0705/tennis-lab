"""Tests for strict canonical dataset sequence-window configuration."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.tasks.base.data.canonical_dataset import CanonicalDataset


def test_hydra_list_config_is_a_valid_sequence_length_range() -> None:
    config = OmegaConf.create({"data": {"seq_len_range": [2, 4]}})

    dataset = CanonicalDataset[object](config=config, augment=False)

    assert dataset.seq_len_range == (2, 4)


@pytest.mark.parametrize(
    "invalid_range",
    [
        "12",
        {"lower": 1, "upper": 2},
        [1],
        [1, 2, 3],
        [True, 2],
        [1.0, 2],
    ],
)
def test_sequence_length_range_rejects_non_integer_pairs(
    invalid_range: object,
) -> None:
    with pytest.raises(ValueError, match="must contain two integers"):
        CanonicalDataset[object](
            config={"data": {"seq_len_range": invalid_range}},
            augment=False,
        )
