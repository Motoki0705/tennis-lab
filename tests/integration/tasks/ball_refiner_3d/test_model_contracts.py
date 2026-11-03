"""Actual regression/Flow/GAN/physics losses and named optimizer updates."""

import pytest

from tests.integration.tasks.ball_refiner_3d.model_contracts import (
    CONFIG_DIR,
    FLOW,
    PHYSICS,
    REGRESSION,
    SELECTORS,
)
from tests.support.model_contracts.exercise import exercise_family
from tests.support.model_contracts.faults import head_fault
from tests.support.model_contracts.lifecycle import (
    exercise_checkpoint,
    exercise_resume,
    omit_head,
    resume_fault,
)
from tests.support.model_contracts.registry import (
    assert_selector_coverage,
    configured_selectors,
)

pytestmark = pytest.mark.integration


def test_selector_coverage():
    assert_selector_coverage(
        configured_selectors(CONFIG_DIR, identity="architecture"), SELECTORS
    )


@pytest.mark.parametrize(
    "family,gan",
    [(REGRESSION, False), (REGRESSION, True), (FLOW, False), (PHYSICS, False)],
    ids=["regression", "GAN", "Flow", "physics"],
)
def test_real_training_contract(tmp_path, family, gan):
    exercise_family(family, tmp_path, gan=gan)


@pytest.mark.parametrize("family", [REGRESSION, FLOW], ids=lambda family: family.name)
@pytest.mark.parametrize(
    "fault,expected",
    [
        ("detach", "missing gradients.*event_head"),
        ("optimizer", "optimizer missing.*event_head"),
    ],
)
def test_head_fault_is_detected(tmp_path, family, fault, expected):
    with pytest.raises(AssertionError, match=expected):
        exercise_family(family, tmp_path, mutate=head_fault("model.event_head", fault))


@pytest.mark.parametrize(
    "family", [REGRESSION, FLOW, PHYSICS], ids=lambda family: family.name
)
def test_checkpoint_all_public_outputs(tmp_path, family):
    exercise_checkpoint(family, tmp_path)


@pytest.mark.parametrize(
    "family,head",
    [
        (REGRESSION, "model.event_head"),
        (FLOW, "model.event_head"),
        (PHYSICS, "model.field_head"),
    ],
    ids=["regression", "flow", "physics"],
)
def test_checkpoint_missing_head_is_detected(tmp_path, family, head):
    with pytest.raises(RuntimeError, match="Missing key"):
        exercise_checkpoint(family, tmp_path, mutate=omit_head(head))


@pytest.mark.parametrize("gan", [False, True], ids=["supervised", "gan"])
def test_interrupted_training_matches_continuous(tmp_path, gan):
    exercise_resume(REGRESSION, tmp_path, gan=gan)


@pytest.mark.parametrize("family", [FLOW, PHYSICS], ids=lambda family: family.name)
def test_flow_and_physics_resume(tmp_path, family):
    exercise_resume(family, tmp_path)


@pytest.mark.parametrize("fault", ["optimizer", "rng", "data"])
def test_checkpoint_resume_state_omission_is_detected(tmp_path, fault):
    with pytest.raises(AssertionError, match="resume"):
        exercise_resume(FLOW, tmp_path, mutate=resume_fault(fault))
