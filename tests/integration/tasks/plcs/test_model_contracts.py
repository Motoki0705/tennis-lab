"""PLCS direct/temporal canonical training and representative fault injections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.integration.tasks.plcs.model_contracts import (
    AXIAL,
    CONFIG_DIR,
    SELECTORS,
    TEMPORAL,
)
from tests.support.model_contracts.exercise import exercise_family
from tests.support.model_contracts.lifecycle import (
    exercise_checkpoint,
    exercise_resume,
    omit_head,
    resume_fault,
)
from tests.support.model_contracts.parameters import names_under
from tests.support.model_contracts.registry import (
    assert_selector_coverage,
    configured_selectors,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("gan", [False, True], ids=["supervised", "gan"])
def test_axial_real_loss_and_training_update(tmp_path: Path, gan: bool) -> None:
    exercise_family(AXIAL, tmp_path, gan=gan)


def test_temporal_canonical_head_real_loss_and_training_update(tmp_path: Path) -> None:
    exercise_family(TEMPORAL, tmp_path)


def test_every_public_model_selector_is_registered() -> None:
    assert_selector_coverage(configured_selectors(CONFIG_DIR), SELECTORS)


@pytest.mark.parametrize("fault", ["detach", "optimizer", "freeze", "omit_loss"])
def test_contract_detects_disconnected_canonical_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    def mutate(module: Any) -> None:
        head = module.model.canonical_pose_head
        if fault == "detach":
            head.register_forward_hook(lambda mod, inputs, output: output.detach())
        elif fault == "optimizer":
            excluded = names_under(module, "model.canonical_pose_head")
            monkeypatch.setattr(
                module,
                "optimizer_param_groups",
                lambda: [
                    {
                        "params": [
                            param
                            for name, param in module.named_parameters()
                            if name not in excluded
                        ],
                    }
                ],
            )
        elif fault == "freeze":
            head.requires_grad_(False)
        else:
            module.loss_fn.loss_weights["canonical_pose"] = 0.0

    error = (
        "optimizer missing=.*canonical_pose"
        if fault == "optimizer"
        else "missing gradients.*canonical_pose"
    )
    with pytest.raises(AssertionError, match=error):
        exercise_family(AXIAL, tmp_path, mutate=mutate)


@pytest.mark.parametrize("family", [AXIAL, TEMPORAL], ids=lambda family: family.name)
def test_checkpoint_all_public_outputs(tmp_path, family):
    exercise_checkpoint(family, tmp_path)


def test_checkpoint_missing_head_is_detected(tmp_path):
    with pytest.raises(RuntimeError, match="Missing key"):
        exercise_checkpoint(
            AXIAL, tmp_path, mutate=omit_head("model.canonical_pose_head")
        )


@pytest.mark.parametrize("gan", [False, True], ids=["supervised", "gan"])
def test_interrupted_training_matches_continuous(tmp_path, gan):
    exercise_resume(AXIAL, tmp_path, gan=gan)


@pytest.mark.parametrize("fault", ["optimizer", "rng", "data"])
def test_checkpoint_resume_state_omission_is_detected(tmp_path, fault):
    with pytest.raises(AssertionError, match="resume"):
        exercise_resume(AXIAL, tmp_path, mutate=resume_fault(fault))


def test_gan_resume_after_transition(tmp_path):
    exercise_resume(AXIAL, tmp_path, gan=True, interrupt_epoch=2)
