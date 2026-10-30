"""Observe optimizer boundaries in task-owned, non-Lightning training loops."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    assert_gradients,
    assert_optimizer_coverage,
    assert_update,
    snapshot,
)


class StepAudit:
    """Check the declared phase, gradients and weights around every real step.

    The phase is selected by the completed update count, never inferred from
    requires_grad or observed gradients. Optimizer implementations are unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        phase: Callable[[int], tuple[ParameterContract, OptimizerContract]],
    ) -> None:
        self.model = model
        self.phase = phase
        self.updates = 0
        self.losses: list[float] = []
        self._handles: list[RemovableHandle] = []
        self._before: Any = None
        self._update: OptimizerContract | None = None

    def record_loss(self, loss: Tensor) -> None:
        assert loss.ndim == 0 and torch.isfinite(loss), "nonfinite/scalar training loss"
        self.losses.append(float(loss.detach()))

    def attach(self, optimizer: Optimizer) -> None:
        assert not self._handles, "optimizer audit already attached"
        self._handles = [
            optimizer.register_step_pre_hook(self._pre),
            optimizer.register_step_post_hook(self._post),
        ]

    def _pre(self, optimizer: Optimizer, args: Any, kwargs: Any) -> None:
        parameters, contract = self.phase(self.updates)
        assert_optimizer_coverage(self.model, parameters, [optimizer], [contract])
        assert_gradients(
            self.model,
            active=parameters.active,
            absent=parameters.frozen | parameters.unused,
        )
        self._before = snapshot(self.model)
        # Frozen/unused tensors owned by a staged optimizer must stay unchanged.
        self._update = replace(
            contract, parameters=contract.parameters & parameters.active
        )

    def _post(self, optimizer: Optimizer, args: Any, kwargs: Any) -> None:
        assert self._update is not None
        assert_update(self.model, self._before, self._update)
        self.updates += 1

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
