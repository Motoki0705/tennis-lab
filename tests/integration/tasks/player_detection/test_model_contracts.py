"""Player wrapper contracts; the external CUDA DINO graph is explicitly fake."""

import pytest

from tests.integration.tasks.player_detection.model_contracts import (
    CONFIG_DIR,
    DINO,
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


def test_cpu_boundary_training_contract(tmp_path):
    exercise_family(DINO, tmp_path)


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("detach", "missing gradients.*bbox_embed"),
        ("optimizer", "optimizer missing.*bbox_embed"),
    ],
)
def test_head_fault_is_detected(tmp_path, fault, expected):
    with pytest.raises(AssertionError, match=expected):
        exercise_family(
            DINO, tmp_path, mutate=head_fault("model.dino.bbox_embed", fault)
        )
