"""Staged optimizers retain frozen tensors while connectivity changes by phase."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    assert_optimizer_coverage,
    snapshot,
)
from tests.support.model_contracts.steps import StepAudit


def test_frozen_optimizer_ownership_must_be_explicit():
    model = nn.ParameterDict(
        {
            "encoder": nn.Parameter(torch.ones(2), requires_grad=False),
            "head": nn.Parameter(torch.ones(2)),
        }
    )
    parameters = ParameterContract(
        frozenset({"head"}), frozenset({"encoder"}), frozenset()
    )
    contract = OptimizerContract(
        "generator", frozenset({"encoder", "head"}), {"head": frozenset({"head"})}
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(AssertionError, match="optimizer union.*unexpected=.*encoder"):
        assert_optimizer_coverage(model, parameters, [optimizer], [contract])
    assert_optimizer_coverage(
        model,
        replace(parameters, optimizer_frozen=parameters.frozen),
        [optimizer],
        [contract],
    )
    with pytest.raises(AssertionError, match="optimizer_frozen must be frozen"):
        replace(parameters, optimizer_frozen=frozenset({"head"})).validate(model)


@pytest.mark.parametrize(
    "fault", [None, "detach", "optimizer", "thaw", "zero_loss", "frozen_update"]
)
def test_two_phase_updates_and_failures_are_detected_by_name(fault):
    model = nn.ParameterDict(
        {"encoder": nn.Parameter(torch.ones(2)), "head": nn.Parameter(torch.ones(2))}
    )
    all_names = frozenset({"encoder", "head"})

    def phase(step):
        frozen = frozenset({"encoder"}) if step == 0 else frozenset()
        active = all_names - frozen
        return (
            ParameterContract(active, frozen, frozenset(), optimizer_frozen=frozen),
            OptimizerContract(
                "generator", all_names, {name: frozenset({name}) for name in active}
            ),
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    audit = StepAudit(model, phase)
    audit.attach(optimizer)
    messages = {
        "detach": "missing gradients.*encoder",
        "optimizer": "optimizer missing.*encoder",
        "thaw": "requires_grad.*encoder",
        "zero_loss": "required group did not update.*encoder",
        "frozen_update": "updated unowned parameters.*encoder",
    }

    def train():
        for step in range(2):
            model["encoder"].requires_grad_(step == 1 and fault != "thaw")
            optimizer.zero_grad(set_to_none=True)
            if step == 1 and fault == "optimizer":
                optimizer.param_groups[0]["params"] = [model["head"]]
            encoder = (
                model["encoder"].detach()
                if step == 1 and fault == "detach"
                else model["encoder"]
            )
            scale = 0.0 if step == 1 and fault == "zero_loss" else 1.0
            (encoder.sum() * scale + model["head"].sum()).backward()
            before = snapshot(model)
            if step == 0 and fault == "frozen_update":
                # Faulty optimizer implementation mutates a frozen tensor after
                # the audit snapshot, independently of the missing-gradient test.
                def corrupt(*_args):
                    with torch.no_grad():
                        model["encoder"].add_(1)

                handle = optimizer.register_step_pre_hook(corrupt)
                try:
                    optimizer.step()
                finally:
                    handle.remove()
            else:
                optimizer.step()
            if step == 0:
                torch.testing.assert_close(
                    model["encoder"], before["encoder"], rtol=0, atol=0
                )

    try:
        if fault is None:
            train()
            assert audit.updates == 2
        else:
            with pytest.raises(AssertionError, match=messages[fault]):
                train()
    finally:
        audit.close()
