"""Tests for the shared model I/O lifecycle contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from src.tasks.base.model_io import (
    ModelAdapterMismatchError,
    ModelCall,
    ModelInputContractError,
    ModelIOContractError,
    ModelOutputContractError,
    TensorSpec,
    bind_model_io,
    require_tensor,
)


class _ScaleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, values: Tensor) -> Tensor:
        self.calls += 1
        return values * 2.0


class _OtherModel(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return values


@dataclass(frozen=True)
class _ScaleAdapter:
    input_spec = TensorSpec(shape=(None, 2), dtypes=frozenset({torch.float32}))

    @property
    def model_type(self) -> type[nn.Module]:
        return _ScaleModel

    def build_call(self, batch: dict[str, object]) -> ModelCall:
        values = require_tensor(batch, "values", spec=self.input_spec)
        return ModelCall(kwargs={"values": values})

    def decode_output(self, output: Tensor) -> Tensor:
        if output.shape[-1] != 2:
            raise ModelOutputContractError("scaled output must preserve feature width")
        return output.cpu()


def test_binding_runs_validated_model_call_and_decode() -> None:
    model = _ScaleModel()
    binding = bind_model_io(model, _ScaleAdapter())
    values = torch.tensor([[1.0, 3.0]], dtype=torch.float32)

    result = binding.run({"values": values})

    torch.testing.assert_close(result, values * 2.0)
    assert model.calls == 1


def test_binding_rejects_model_adapter_mismatch_at_composition() -> None:
    with pytest.raises(ModelAdapterMismatchError, match="requires"):
        bind_model_io(_OtherModel(), _ScaleAdapter())


def test_output_contract_rejects_invalid_decoded_shape() -> None:
    binding = bind_model_io(_ScaleModel(), _ScaleAdapter())

    with pytest.raises(ModelOutputContractError, match="feature width"):
        binding.decode_output(torch.ones(1, 3))


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ({}, "missing"),
        ({"values": [[1.0, 2.0]]}, "torch.Tensor"),
        ({"values": torch.ones(2)}, "rank 2"),
        ({"values": torch.ones(1, 3)}, "axis 1"),
        ({"values": torch.ones(1, 2, dtype=torch.float64)}, "torch.float32"),
    ],
)
def test_invalid_batch_fails_before_model_forward(
    batch: dict[str, object], message: str
) -> None:
    model = _ScaleModel()
    binding = bind_model_io(model, _ScaleAdapter())

    with pytest.raises(ModelInputContractError, match=message):
        binding.run(batch)

    assert model.calls == 0


def test_model_call_copies_keyword_mapping() -> None:
    values = torch.ones(1, 2)
    kwargs = {"values": values}

    call = ModelCall(kwargs=kwargs)
    kwargs.clear()

    assert call.kwargs == {"values": values}


def test_model_call_rejects_non_tensor_arguments_at_boundary() -> None:
    with pytest.raises(ModelInputContractError, match="tensor or None"):
        ModelCall(args=("not-a-tensor",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "spec_kwargs",
    [
        {"shape": (None, -1)},
        {"dtypes": frozenset()},
    ],
)
def test_tensor_spec_rejects_invalid_static_contract(
    spec_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ModelIOContractError):
        TensorSpec(**spec_kwargs)  # type: ignore[arg-type]
