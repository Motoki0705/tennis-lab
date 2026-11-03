"""Read published saved-train authority without restoring the retired importer."""

import copy
import json
from pathlib import Path

import pytest

from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    digest,
    stable_id,
)
from src.synthetic_data_generation.dataset.court.scale.curves import ArcMap, Curve
from src.synthetic_data_generation.dataset.court.scale.planning import point_record
from src.synthetic_data_generation.dataset.court.scale.strategies import (
    strategy_definition,
)
from src.synthetic_data_generation.dataset.court.scale.validation import (
    validate_plan_geometry,
)


def common_sample_geometry(plan, layout):
    """Complete the fixed authority fixture using the common point contract.

    The source, expected curves/views and trajectory IDs were frozen with the
    historical importer. They are never obtained from the restored validator.
    Rebuilding common samples after mutations ensures an authority comparison,
    rather than a stale point/hash, is what rejects the altered saved geometry.
    """
    recipe = GeometryRecipe.from_dict(plan["recipe"])
    plan["recipe_sha256"] = digest(plan["recipe"])
    for group in plan["trajectory_groups"]:
        curve = Curve.from_dict(group["curve"])
        arc = ArcMap.freeze(curve, recipe)
        group["arc_map"] = arc.to_dict()
        group["samples"] = []
        for q in arc.fractions(plan["density_level"]):
            record = point_record(
                curve=curve,
                arc=arc,
                q=q,
                trajectory_id=group["trajectory_id"],
                view=group["view"],
                seed=plan["generation_seed"],
                recipe=recipe,
                layout=layout,
            )
            record["scale"]["comparison_pair_id"] = stable_id(
                "comparison_pair",
                plan["scene_id"],
                plan["scene_geometry_id"],
                plan["method"],
                plan["generation_seed"],
                group["provenance"],
                [q.numerator, q.denominator],
            )
            group["samples"].append(record)


@pytest.fixture(scope="module")
def saved_plan_authority():
    # Keep only the independent source/geometry oracle on disk, without large
    # arc tables, derived sparse projections, images or a generating API.
    path = Path(__file__).with_name("fixtures") / "saved_train_authority.json"
    value = json.loads(path.read_text())
    # Numeric-runtime rejection is already covered by test_scale_publication;
    # this fixture pins the historical geometry, not the CI installation.
    value["strategy"] = strategy_definition()
    return value


def test_saved_train_plan_preserves_original_train_geometry(
    saved_plan_authority, multi_court_layout
):
    plan = copy.deepcopy(saved_plan_authority)
    common_sample_geometry(plan, multi_court_layout)
    records = validate_plan_geometry(plan, multi_court_layout)
    assert records
    assert {record["split"] for record in records.values()} == {"train"}
    assert {
        g["provenance"]["saved_group"]["trajectory"]["trajectory_group_id"]
        for g in plan["trajectory_groups"]
    } == {"group-0", "group-2"}
    for group in plan["trajectory_groups"]:
        assert group["view"]["hfov_degrees"] == 45.0
        assert group["curve"]["parameters"]["orientation"] == 0.7
        assert group["curve"]["parameters"]["phase"] == 1.1


@pytest.mark.parametrize(
    "mutation",
    [
        "split",
        "curve",
        "center",
        "view",
        "recipe",
        "id",
        "ordinal",
        "source_split",
        "duplicate_source_group",
        "replay_rule",
    ],
)
def test_saved_train_rejects_changed_authority(
    mutation, saved_plan_authority, multi_court_layout
):
    plan = copy.deepcopy(saved_plan_authority)
    group = plan["trajectory_groups"][0]
    if mutation == "split":
        group["provenance"]["saved_group"]["split"] = "test"
    elif mutation == "curve":
        group["curve"]["parameters"]["phase"] = 0.0
    elif mutation == "center":
        group["curve"]["scene_from_local"][3] = 0.5
    elif mutation == "view":
        group["view"]["hfov_degrees"] = 75.0
    elif mutation == "recipe":
        plan["recipe"]["width"] = 81
    elif mutation == "id":
        group["trajectory_id"] = "0" * 64
    elif mutation == "ordinal":
        group["ordinal"] = True
    elif mutation == "source_split":
        plan["saved_source"]["trajectory_groups"][0]["split"] = "validation"
    elif mutation == "duplicate_source_group":
        plan["saved_source"]["trajectory_groups"][1]["trajectory"][
            "trajectory_group_id"
        ] = "group-0"
    elif mutation == "replay_rule":
        plan["saved_replay_rule"] = "unknown"
    else:
        raise AssertionError(mutation)
    common_sample_geometry(plan, multi_court_layout)
    with pytest.raises(ValueError, match="Saved (train|source)"):
        validate_plan_geometry(plan, multi_court_layout)
