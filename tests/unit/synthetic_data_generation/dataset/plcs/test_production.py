"""Typed PLCS production-mode contract tests."""

import pytest

from src.synthetic_data_generation.dataset.plcs.production import (
    PLCSProductionMode,
    validate_plcs_production_contract,
)


@pytest.mark.parametrize("category", ["running", "walking", "general"])
def test_single_object_mode_accepts_one_real_category_at_frame_zero(
    category: str,
) -> None:
    validate_plcs_production_contract(
        mode=PLCSProductionMode.SINGLE_OBJECT,
        configured_motion_categories=(category,),
        object_motion_categories=(category,),
        object_start_frames=(0,),
    )


def test_multi_object_mode_retains_running_walking_general_contract() -> None:
    validate_plcs_production_contract(
        mode=PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE,
        configured_motion_categories=("running", "walking", "general"),
        object_motion_categories=("running", "walking", "general"),
        object_start_frames=(0, 120, 240),
    )


@pytest.mark.parametrize(
    ("mode", "categories", "starts", "message"),
    [
        (
            PLCSProductionMode.SINGLE_OBJECT,
            ("running", "walking"),
            (0, 0),
            "exactly one object",
        ),
        (
            PLCSProductionMode.SINGLE_OBJECT,
            ("running",),
            (1,),
            "start_frame=0",
        ),
        (
            PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE,
            ("running",),
            (0,),
            "at least two objects",
        ),
        (
            PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE,
            ("running", "walking"),
            (0, 1),
            "requires running, walking, and general",
        ),
    ],
)
def test_mode_cardinality_and_inventory_mismatches_fail_closed(
    mode: PLCSProductionMode,
    categories: tuple[str, ...],
    starts: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_plcs_production_contract(
            mode=mode,
            configured_motion_categories=categories,
            object_motion_categories=categories,
            object_start_frames=starts,
        )


def test_unknown_or_boolean_mode_is_not_a_production_mode() -> None:
    with pytest.raises(ValueError):
        PLCSProductionMode("single")
    with pytest.raises(TypeError, match="PLCSProductionMode"):
        validate_plcs_production_contract(
            mode=True,  # type: ignore[arg-type]
            configured_motion_categories=("running",),
            object_motion_categories=("running",),
            object_start_frames=(0,),
        )
