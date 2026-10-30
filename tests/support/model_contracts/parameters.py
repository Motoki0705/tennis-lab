"""Named parameter contracts shared by real task training tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class ParameterContract:
    """Exhaustive names, independent of the model's current requires_grad flags.

    Unused parameters remain trainable and owned by an optimizer, but must have
    no gradient in this loss/phase. Exceptions must be exact, existing names.
    A finite zero gradient satisfies connectivity; None does not.
    """

    active: frozenset[str]
    frozen: frozenset[str]
    unused: frozenset[str]

    def validate(self, model: nn.Module) -> None:
        named = dict(model.named_parameters())
        partitions = (self.active, self.frozen, self.unused)
        declared = set().union(*partitions)
        assert declared == set(named), (
            f"parameter partition: missing={sorted(set(named) - declared)}, "
            f"unknown={sorted(declared - set(named))}"
        )
        assert sum(map(len, partitions)) == len(declared), "overlapping partitions"
        assert not any("*" in name or "?" in name for name in declared), (
            "parameter names must be exact, without wildcards"
        )
        expected = self.active | self.unused
        observed = {name for name, param in named.items() if param.requires_grad}
        assert observed == expected, (
            f"requires_grad: unexpectedly frozen={sorted(expected - observed)}, "
            f"unexpectedly trainable={sorted(observed - expected)}"
        )


@dataclass(frozen=True)
class OptimizerContract:
    """One optimizer's ownership and groups that must move on each update."""

    name: str
    parameters: frozenset[str]
    required_updates: Mapping[str, frozenset[str]]


def names_under(model: nn.Module, *paths: str) -> frozenset[str]:
    """Resolve required modules to full names; a misspelled/empty group fails."""
    canonical = {id(param): name for name, param in model.named_parameters()}
    names: set[str] = set()
    for path in paths:
        child = model.get_submodule(path)
        # Shared embeddings may be registered at more than one module path.
        group = {canonical[id(param)] for param in child.parameters()}
        assert group, f"empty parameter group: {path}"
        names.update(group)
    return frozenset(names)


def assert_optimizer_coverage(
    model: nn.Module,
    parameters: ParameterContract,
    optimizers: Sequence[Optimizer],
    contracts: Sequence[OptimizerContract],
) -> None:
    parameters.validate(model)
    assert len(optimizers) == len(contracts), "optimizer count differs"
    named = dict(model.named_parameters())
    by_id = {id(param): name for name, param in named.items()}
    all_names: list[str] = []
    for optimizer, contract in zip(optimizers, contracts, strict=True):
        observed: list[str] = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                assert id(param) in by_id, (
                    f"{contract.name}: foreign optimizer parameter"
                )
                observed.append(by_id[id(param)])
        duplicates = sorted(name for name, n in Counter(observed).items() if n > 1)
        assert not duplicates, (
            f"{contract.name}: duplicate optimizer names={duplicates}"
        )
        assert set(observed) == contract.parameters, (
            f"{contract.name}: optimizer missing={sorted(contract.parameters - set(observed))}, "
            f"unexpected={sorted(set(observed) - contract.parameters)}"
        )
        assert contract.required_updates, f"{contract.name}: no required update groups"
        for label, group_names in contract.required_updates.items():
            assert group_names and group_names <= (
                contract.parameters & parameters.active
            ), f"{contract.name}/{label}: invalid required update group"
        all_names.extend(observed)
    duplicates = sorted(name for name, n in Counter(all_names).items() if n > 1)
    assert not duplicates, f"parameters shared between optimizers: {duplicates}"
    expected = parameters.active | parameters.unused
    assert set(all_names) == expected, (
        f"optimizer union missing={sorted(expected - set(all_names))}, "
        f"unexpected={sorted(set(all_names) - expected)}"
    )


def assert_gradients(
    model: nn.Module, *, active: frozenset[str], absent: frozenset[str]
) -> None:
    named = dict(model.named_parameters())
    assert active | absent <= named.keys(), "unknown gradient contract names"
    observed = {name for name, param in named.items() if param.grad is not None}
    missing = sorted(active - observed)
    assert not missing, f"missing gradients (None): {missing}"
    unexpected = sorted(absent & observed)
    assert not unexpected, f"unexpected gradients: {unexpected}"
    nonfinite = [
        name for name in sorted(active) if not torch.isfinite(named[name].grad).all()
    ]
    assert not nonfinite, f"nonfinite gradients: {nonfinite}"


def snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def assert_update(
    model: nn.Module, before: Mapping[str, Tensor], contract: OptimizerContract
) -> None:
    named = dict(model.named_parameters())
    nonfinite = sorted(
        name for name, param in named.items() if not torch.isfinite(param).all()
    )
    assert not nonfinite, f"nonfinite weights: {nonfinite}"
    changed = {
        name for name, param in named.items() if not torch.equal(before[name], param)
    }
    unexpected = changed - contract.parameters
    assert not unexpected, (
        f"{contract.name}: updated unowned parameters={sorted(unexpected)}"
    )
    for label, names in contract.required_updates.items():
        assert changed & names, (
            f"{contract.name}: required group did not update: {label}"
        )
