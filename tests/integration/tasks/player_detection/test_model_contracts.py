"""Player wrapper contracts; the external CUDA DINO graph is explicitly fake."""

import pytest

from tests.integration.tasks.player_detection.model_contracts import (
    CONFIG_DIR,
    DINO,
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


@pytest.mark.parametrize("family", [DINO], ids=lambda family: family.name)
def test_checkpoint_all_public_outputs(tmp_path, family):
    exercise_checkpoint(family, tmp_path)


def test_checkpoint_missing_head_is_detected(tmp_path):
    with pytest.raises(RuntimeError, match="Missing key"):
        exercise_checkpoint(DINO, tmp_path, mutate=omit_head("model.dino.bbox_embed"))


def test_interrupted_training_matches_continuous(tmp_path):
    exercise_resume(DINO, tmp_path)


@pytest.mark.parametrize("fault", ["optimizer", "rng", "data"])
def test_checkpoint_resume_state_omission_is_detected(tmp_path, fault):
    with pytest.raises(AssertionError, match="resume"):
        exercise_resume(DINO, tmp_path, mutate=resume_fault(fault))
