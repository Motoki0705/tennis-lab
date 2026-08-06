"""Integration smoke test for a bound model I/O training lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.base.model_io import ModelCall, TensorSpec, bind_model_io, require_tensor


class _Regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.linear(features))


@dataclass(frozen=True)
class _RegressionAdapter:
    spec = TensorSpec(shape=(None, 2), dtypes=frozenset({torch.float32}))

    @property
    def model_type(self) -> type[nn.Module]:
        return _Regressor

    def build_call(self, batch: dict[str, object]) -> ModelCall:
        return ModelCall(
            kwargs={"features": require_tensor(batch, "features", spec=self.spec)}
        )

    def decode_output(self, output: Tensor) -> Tensor:
        return output


def test_bound_lifecycle_preserves_gradient_flow_for_one_training_step() -> None:
    model = _Regressor()
    binding = bind_model_io(model, _RegressionAdapter())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    features = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    target = torch.tensor([[0.5], [-1.0]])
    before = model.linear.weight.detach().clone()

    prediction = binding.run({"features": features})
    loss = torch.nn.functional.mse_loss(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert model.linear.weight.grad is not None
    assert not torch.equal(model.linear.weight.detach(), before)
