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
