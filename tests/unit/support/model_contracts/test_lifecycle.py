"""The comparison engine must inspect nested fields, schemas and NaN masks."""

from dataclasses import dataclass, replace

import pytest
import torch

from tests.support.model_contracts.lifecycle import assert_tree_equal


@dataclass(frozen=True)
class _Prediction:
    value: torch.Tensor
    unit: str
    threshold: float
    valid: torch.Tensor


@pytest.mark.parametrize(
    "fault", ["head", "dtype", "shape", "unit", "threshold", "mask", "nan"]
)
def test_every_nested_output_field_is_compared(fault):
    expected = _Prediction(
        torch.tensor([1.0, float("nan")]), "m/s", 0.63, torch.tensor([True, False])
    )
    replacement = {
        "head": {"value": torch.tensor([2.0, float("nan")])},
        "dtype": {"value": expected.value.double()},
        "shape": {"value": expected.value[None]},
        "unit": {"unit": "normalized"},
        "threshold": {"threshold": 0.5},
        "mask": {"valid": ~expected.valid},
        "nan": {"value": torch.tensor([float("nan"), 1.0])},
    }[fault]
    assert_tree_equal({"prediction": expected}, {"prediction": expected})
    with pytest.raises(AssertionError, match="output.prediction"):
        assert_tree_equal(
            {"prediction": expected}, {"prediction": replace(expected, **replacement)}
        )
