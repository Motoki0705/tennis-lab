"""The assertion engine must distinguish missing/zero/frozen/unused by name."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.optim import SGD

from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    assert_gradients,
    assert_optimizer_coverage,
    assert_update,
    names_under,
    snapshot,
)
from tests.support.model_contracts.registry import assert_selector_coverage


def _parameters() -> tuple[nn.ParameterDict, ParameterContract, OptimizerContract]:
    model = nn.ParameterDict(
        {
            "head": nn.Parameter(torch.ones(2)),
            "unused": nn.Parameter(torch.ones(2)),
            "frozen": nn.Parameter(torch.ones(2), requires_grad=False),
        }
    )
    contract = ParameterContract(
        frozenset({"head"}), frozenset({"frozen"}), frozenset({"unused"})
    )
    optimizer = OptimizerContract(
        "generator", frozenset({"head", "unused"}), {"head": frozenset({"head"})}
    )
    return model, contract, optimizer


def test_zero_gradient_is_connected_but_none_is_missing() -> None:
    model, contract, _ = _parameters()
    contract.validate(model)
    with pytest.raises(AssertionError, match="missing gradients.*head"):
        assert_gradients(
            model, active=contract.active, absent=contract.frozen | contract.unused
        )
    (model["head"].sum() * 0).backward()
    assert_gradients(
        model, active=contract.active, absent=contract.frozen | contract.unused
    )
    model["head"].grad[0] = torch.nan
    with pytest.raises(AssertionError, match="nonfinite gradients.*head"):
        assert_gradients(
            model, active=contract.active, absent=contract.frozen | contract.unused
        )


def test_partition_checks_flags_and_rejects_unknown_unused_exceptions() -> None:
    model, contract, _ = _parameters()
    model["head"].requires_grad_(False)
    with pytest.raises(AssertionError, match="requires_grad.*head"):
        contract.validate(model)
    with pytest.raises(AssertionError, match="parameter partition.*unknown"):
        replace(contract, unused=frozenset({"unused", "head.*"})).validate(model)


def test_unused_and_frozen_parameters_cannot_acquire_gradients() -> None:
    model, contract, _ = _parameters()
    model["head"].sum().backward()
    model["unused"].sum().backward()
    with pytest.raises(AssertionError, match="unexpected gradients.*unused"):
        assert_gradients(
            model, active=contract.active, absent=contract.frozen | contract.unused
        )


def test_optimizer_requires_complete_ownership_without_duplicates() -> None:
    model, parameters, contract = _parameters()
    optimizer = SGD([model["head"], model["unused"]], lr=0.1)
    assert_optimizer_coverage(model, parameters, [optimizer], [contract])
    optimizer.param_groups[0]["params"].pop()
    with pytest.raises(AssertionError, match="optimizer missing.*unused"):
        assert_optimizer_coverage(model, parameters, [optimizer], [contract])
    optimizer.param_groups[0]["params"].append(model["head"])
    with pytest.raises(AssertionError, match="duplicate optimizer names.*head"):
        assert_optimizer_coverage(model, parameters, [optimizer], [contract])


def test_parameters_cannot_belong_to_two_optimizers() -> None:
    model, parameters, contract = _parameters()
    optimizer = SGD([model["head"], model["unused"]], lr=0.1)
    with pytest.raises(AssertionError, match="shared between optimizers.*head"):
        assert_optimizer_coverage(
            model,
            parameters,
            [optimizer, optimizer],
            [contract, replace(contract, name="other")],
        )


def test_update_requires_head_movement_and_preserves_unowned_weights() -> None:
    model, _, contract = _parameters()
    optimizer = SGD([model["head"], model["unused"]], lr=0.1)
    before = snapshot(model)
    with pytest.raises(AssertionError, match="required group did not update.*head"):
        assert_update(model, before, contract)
    model["head"].sum().backward()
    optimizer.step()
    assert_update(model, before, contract)
    with torch.no_grad():
        model["frozen"].add_(1)
    with pytest.raises(AssertionError, match="updated unowned parameters.*frozen"):
        assert_update(model, before, contract)


def test_shared_parameters_use_canonical_full_names() -> None:
    model = nn.Module()
    model.embedding = nn.Embedding(2, 3)
    model.alias = model.embedding
    assert names_under(model, "alias") == frozenset({"embedding.weight"})


def test_new_model_selector_cannot_be_silently_unregistered() -> None:
    registered = {"axial_small": "axial", "axial_large": "axial"}
    discovered = {**registered, "future_model": "new_architecture"}
    with pytest.raises(AssertionError, match="unregistered=.*future_model"):
        assert_selector_coverage(discovered, registered)
