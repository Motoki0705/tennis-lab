"""Tests for strict Court task configuration contracts."""

from __future__ import annotations

import pytest

from src.tasks.court_detection.configuration import CourtLossConfig
from src.utils.configuration import (
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

pytestmark = pytest.mark.unit


def _loss_mapping(*, positive_weight: float = 1.0) -> dict[str, object]:
    return {
        "seg": {"ce_weight": 1.0, "dice_weight": 1.0},
        "kp": {"focal_gamma": 2.0, "positive_weight": positive_weight},
        "line": {"bce_weight": 1.0, "dice_weight": 1.0, "pos_weight": 8.0},
    }


def test_kp_positive_weight_is_required_and_exactly_named() -> None:
    missing = _loss_mapping()
    kp_missing = missing["kp"]
    assert isinstance(kp_missing, dict)
    del kp_missing["positive_weight"]

    with pytest.raises(MissingConfigurationKeyError, match="positive_weight"):
        CourtLossConfig.from_mapping(missing)

    unknown = _loss_mapping()
    kp_unknown = unknown["kp"]
    assert isinstance(kp_unknown, dict)
    kp_unknown["pos_weight"] = 2.0
    with pytest.raises(UnknownConfigurationKeyError, match="loss.kp.pos_weight"):
        CourtLossConfig.from_mapping(unknown)


@pytest.mark.parametrize("positive_weight", [0.0, -1.0, float("nan"), float("inf")])
def test_kp_positive_weight_must_be_finite_and_positive(
    positive_weight: float,
) -> None:
    with pytest.raises(SemanticConfigurationError, match="positive_weight|finite"):
        CourtLossConfig.from_mapping(
            _loss_mapping(positive_weight=positive_weight)
        )


def test_kp_positive_weight_is_retained_by_typed_contract() -> None:
    config = CourtLossConfig.from_mapping(_loss_mapping(positive_weight=6.0))

    assert config.kp_positive_weight == 6.0
