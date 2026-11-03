"""Compose/collate/real-loss/backward plus the real task Trainer update."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tests.support.model_contracts.lightning import fit_contract
from tests.support.model_contracts.parameters import assert_gradients
from tests.support.model_contracts.registry import ModelFamily


def exercise_family(
    family: ModelFamily,
    root: Path,
    *,
    gan: bool = False,
    mutate: Any = None,
) -> Any:
    config = family.compose(root, gan)
    module, datamodule = family.build(config)
    parameters, optimizers = family.parameters(module)
    parameters.validate(module)
    batch = family.batch(config)
    family.assert_batch(batch)
    if mutate is not None:
        mutate(module)
    result = family.supervised_result(module, batch)
    loss = result["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss), "nonfinite/scalar supervised loss"
    family.assert_prediction(result["outputs"])
    generator = optimizers[0].parameters
    assert loss.requires_grad, (
        f"missing gradients (None): detached supervised loss for {sorted(parameters.active & generator)}"
    )
    loss.backward()
    assert_gradients(
        module,
        active=parameters.active & generator,
        absent=parameters.frozen | parameters.unused | (parameters.active - generator),
    )
    module.zero_grad(set_to_none=True)
    fit = family.fit or fit_contract
    return fit(module, datamodule, parameters, optimizers, root / "fit", gan=gan)
