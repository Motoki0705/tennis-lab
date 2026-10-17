from typing import Any

import pytest
import torch

from src.utils.models.state import state_dict_sha256


def test_hash_is_independent_of_mapping_order_storage_and_strides() -> None:
    tensor = torch.arange(6.).reshape(2, 3).T
    first = dict(weight=tensor, bias=torch.tensor(2., requires_grad=True))
    second = dict(bias=torch.tensor(2.), weight=tensor.contiguous().clone())
    assert state_dict_sha256(first) == state_dict_sha256(second)
    assert len(state_dict_sha256(first)) == 64


@pytest.mark.parametrize("changed", [
    dict(other=torch.zeros(2, 3)), dict(weight=torch.ones(2, 3)),
    dict(weight=torch.zeros(3, 2)), dict(weight=torch.zeros(2, 3, dtype=torch.float64)),
])
def test_hash_covers_names_values_shape_and_dtype(changed: dict[str, torch.Tensor]) -> None:
    assert state_dict_sha256(dict(weight=torch.zeros(2, 3))) != state_dict_sha256(changed)


def test_bfloat16_scalar_empty_and_complex_views_are_supported() -> None:
    state = dict(bf16=torch.tensor([1., 2.], dtype=torch.bfloat16), scalar=torch.tensor(8), empty=torch.empty(0))
    assert state_dict_sha256(state) == state_dict_sha256({k: v.clone() for k, v in state.items()})
    value = torch.tensor([2 + 3j]).conj()
    assert state_dict_sha256(dict(value=value)) == state_dict_sha256(dict(value=value.resolve_conj()))


@pytest.mark.parametrize("state", [
    {1: torch.zeros(1), "x": torch.zeros(1)}, {"x": None},
    {"x": torch.empty(1, device="meta")}, {"x": torch.eye(2).to_sparse()},
    {"x": torch.quantize_per_tensor(torch.zeros(1), .1, 0, torch.quint8)},
])
def test_unsupported_states_are_rejected(state: Any) -> None:
    with pytest.raises(ValueError):
        state_dict_sha256(state)
