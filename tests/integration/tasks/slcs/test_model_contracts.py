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


@pytest.mark.parametrize("family", [SHARED, CONTEXTS], ids=lambda family: family.name)
def test_checkpoint_all_public_outputs(tmp_path, family):
    exercise_checkpoint(family, tmp_path)


def test_checkpoint_missing_head_is_detected(tmp_path):
    with pytest.raises(RuntimeError, match="Missing key"):
        exercise_checkpoint(
            SHARED, tmp_path, mutate=omit_head("model.ball_position_scale_head")
        )


@pytest.mark.parametrize("family", [SHARED, CONTEXTS], ids=lambda family: family.name)
def test_interrupted_training_matches_continuous(tmp_path, family):
    exercise_resume(family, tmp_path)


@pytest.mark.parametrize("fault", ["optimizer", "rng", "data"])
def test_checkpoint_resume_state_omission_is_detected(tmp_path, fault):
    with pytest.raises(AssertionError, match="resume"):
        exercise_resume(CONTEXTS, tmp_path, mutate=resume_fault(fault))


def test_uncertainty_config_change_without_shape_change_is_detected(tmp_path):
    def mutate(payload):
        payload["hyper_parameters"]["config"]["model"]["log_b_max"] = 1.75

    with pytest.raises(AssertionError, match="config.*log_b_max"):
        exercise_checkpoint(CONTEXTS, tmp_path, mutate=mutate)
