"""All valid Ball selectors, actual task loops and named-head faults."""

import pytest

from tests.integration.tasks.ball_detection.model_contracts import (
    CONFIG_DIR,
    DEEP_QUERY,
    FAMILIES,
    POSE_VARIANTS,
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
    assert set(FAMILIES) == set(SELECTORS)


@pytest.mark.parametrize(
    "family",
    [*FAMILIES.values(), DEEP_QUERY, *POSE_VARIANTS],
    ids=lambda family: family.name,
)
def test_real_training_contract(tmp_path, family):
    exercise_family(family, tmp_path)


@pytest.mark.parametrize(
    "family", [DEEP_QUERY, FAMILIES["mdd_hybrid"]], ids=lambda family: family.name
)
def test_failure_to_thaw_encoder_is_detected(tmp_path, family):
    def inject(model):
        freeze = model.freeze_encoder
        model.freeze_encoder = lambda frozen: freeze(True)

    with pytest.raises(AssertionError, match="requires_grad.*encoder"):
        exercise_family(family, tmp_path, mutate=inject)


FAULTS = [
    (FAMILIES["conv_next_unet"], "model.final_conv"),
    (FAMILIES["mdd_pose"], "head"),
    (FAMILIES["mdd_query"], "head"),
    (FAMILIES["mdd_dpt_pretrain"], "head"),
    (DEEP_QUERY, "head"),
    (FAMILIES["mdd_hybrid"], "presence_head"),
    (FAMILIES["mdd_dpt_heatmap"], "head"),
    (FAMILIES["mdd_heatmap_hybrid"], "heatmap_head"),
]


@pytest.mark.parametrize(
    "family,path", FAULTS, ids=[family.name for family, _ in FAULTS]
)
@pytest.mark.parametrize("fault", ["detach", "optimizer"])
def test_head_fault_is_detected(tmp_path, family, path, fault):
    def inject(model):
        if path.startswith("model."):
            head_fault(path, fault)(model)
        elif fault == "detach":
            model.get_submodule(path).register_forward_hook(
                lambda _module, _args, output: output.detach()
            )
        else:
            model._contract_optimizer_exclusions = frozenset(
                id(p) for p in model.get_submodule(path).parameters()
            )

    expected = "missing gradients" if fault == "detach" else "optimizer missing"
    with pytest.raises(AssertionError, match=f"{expected}.*{path}"):
        exercise_family(family, tmp_path, mutate=inject)
