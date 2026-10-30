"""Names, real updates and mutation sensitivity for SLCS."""

import pytest

from tests.integration.tasks.slcs.model_contracts import (
    CONFIG_DIR,
    CONTEXTS,
    SELECTORS,
    SHARED,
)
from tests.support.model_contracts.exercise import exercise_family
from tests.support.model_contracts.faults import head_fault
from tests.support.model_contracts.registry import (
    assert_selector_coverage,
    configured_selectors,
)

pytestmark = pytest.mark.integration


def test_selector_coverage():
    assert_selector_coverage(configured_selectors(CONFIG_DIR), SELECTORS)


@pytest.mark.parametrize("family", [SHARED, CONTEXTS], ids=lambda family: family.name)
def test_real_training_contract(tmp_path, family):
    exercise_family(family, tmp_path)


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("detach", "missing gradients.*ball_position_head"),
        ("optimizer", "optimizer missing.*ball_position_head"),
    ],
)
def test_head_fault_is_detected(tmp_path, fault, expected):
    with pytest.raises(AssertionError, match=expected):
        exercise_family(
            SHARED, tmp_path, mutate=head_fault("model.ball_position_head", fault)
        )
