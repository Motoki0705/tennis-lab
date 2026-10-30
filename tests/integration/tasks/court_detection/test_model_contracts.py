"""Court task graph and mutation tests, with only external DINOv3 faked."""

import pytest

from tests.integration.tasks.court_detection.model_contracts import (
    CONFIG_DIR,
    FROZEN,
    FULL,
    LORA,
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
    assert_selector_coverage(configured_selectors(CONFIG_DIR), SELECTORS)


@pytest.mark.parametrize("family", [FROZEN, LORA, FULL], ids=lambda family: family.name)
def test_real_task_training_contract(tmp_path, family):
    exercise_family(family, tmp_path)


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("detach", "missing gradients.*heads.line"),
        ("optimizer", "optimizer missing.*heads.line"),
    ],
)
def test_head_fault_is_detected(tmp_path, fault, expected):
    with pytest.raises(AssertionError, match=expected):
        exercise_family(FROZEN, tmp_path, mutate=head_fault("model.heads.line", fault))
