"""Strict read-only validation of published saved-train Court trajectory plans.

Saved source metadata owns the original split, curve, centre and selected view.
Only the pure reconstruction needed to validate existing publications is retained.
"""

from __future__ import annotations

from typing import Any

from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitTrajectorySpec,
)

from .core import GeometryRecipe, digest, stable_id
from .curves import Curve
from .planning import view_definition
from .strategies import _orbit_curve

SOURCE_SCHEMA = "court_saved_train_source_v1"
REPLAY_RULE = "published_train_group_geometry_v1"


def saved_train_recipe(source: dict[str, Any]) -> GeometryRecipe:
    """Arc/resolution authority is saved; each group's view remains independent.

    GeometryRecipe's HFOV/height layer lists describe procedural generators and
    are unused by saved-train plans. The exact saved view is checked independently.
    """
    policy = source["sampling_policy"]
    if policy["mode"] != "uniform_arc_length":
        raise ValueError("Only the saved uniform-arc-length policy is supported.")
    jitters = {
        v["look_at_jitter_radius_m"]
        for g in source["trajectory_groups"]
        for v in g["views"]
    }
    if len(jitters) != 1:
        raise ValueError("Saved source requires a common explicit jitter radius.")
    return GeometryRecipe(
        base_step_m=policy["max_arc_step_m"],
        minimum_points=policy["minimum_sample_count"],
        count_multiple=policy["sample_count_multiple"],
        arc_intervals=4096,
        jitter_radius_m=jitters.pop(),
        width=source["width"],
        height=source["height"],
    )


def _train_groups(source: dict[str, Any]) -> list[dict[str, Any]]:
    if source["schema"] != SOURCE_SCHEMA:
        raise ValueError("Unknown saved train source schema.")
    all_groups = source["trajectory_groups"]
    ids = [g["trajectory"]["trajectory_group_id"] for g in all_groups]
    if len(ids) != len(set(ids)) or any(
        g["split"] not in {"train", "validation", "test"} for g in all_groups
    ):
        raise ValueError("Saved source has ambiguous group identities/splits.")
    groups = sorted(
        (g for g in all_groups if g["split"] == "train"),
        key=lambda g: g["trajectory"]["trajectory_group_id"],
    )
    if not groups:
        raise ValueError("Saved source has no train trajectories.")
    return groups


def _definition(
    raw: dict[str, Any], recipe: GeometryRecipe
) -> tuple[Curve, dict[str, Any]]:
    spec = OrbitTrajectorySpec.from_mapping(raw["trajectory"])
    center = OrbitCenter.from_mapping(raw["center"])
    if center.key() != (spec.center_kind, spec.center_court_instance_id):
        raise ValueError("Saved trajectory and centre disagree.")
    if len(raw["views"]) != 1:
        raise ValueError(
            "Saved train validation requires one exact selected view per group."
        )
    saved = raw["views"][0]
    if (
        saved["target_mode"] != "court_center"
        or saved["target_kind"] != "court"
        or saved["look_at_jitter_radius_m"] != recipe.jitter_radius_m
    ):
        raise ValueError("Saved view has an unsupported target or jitter rule.")
    expected_policy = {
        "mode": "trajectory_center_court"
        if spec.center_court_instance_id is not None
        else "nearest_camera",
        "centre_court_instance_id": spec.center_court_instance_id,
    }
    if raw["target_court_policy"] != expected_policy:
        raise ValueError("Saved target-court policy disagrees with the curve centre.")
    return _orbit_curve(spec, center), view_definition(
        recipe, saved["hfov_degrees"], saved["look_at_height_m"]
    )


def _trajectory_id(plan: dict[str, Any], raw: dict[str, Any]) -> str:
    return stable_id(
        "saved_train_trajectory",
        REPLAY_RULE,
        plan["scene_id"],
        plan["scene_geometry_id"],
        digest(plan["saved_source"]),
        raw,
        plan["recipe_sha256"],
        plan["generation_seed"],
    )


def validate_saved_train_authority(plan: dict[str, Any]) -> None:
    """Validate original split and exact geometry before the common sample checks."""
    source = plan["saved_source"]
    groups = _train_groups(source)
    if (
        plan["saved_replay_rule"] != REPLAY_RULE
        or plan["scene_id"] != source["scene_id"]
        or plan["trajectory_count"] != len(groups)
        or plan["view_authority"] != "saved_source.trajectory_groups[].views"
    ):
        raise ValueError(
            "Saved train plan changed its source identity/count/authority."
        )
    recipe = saved_train_recipe(source)
    if recipe.to_dict() != plan["recipe"]:
        raise ValueError("Saved train arc/resolution/jitter recipe changed.")
    for group in plan["trajectory_groups"]:
        ordinal = group["ordinal"]
        if type(ordinal) is not int or not 0 <= ordinal < len(groups):
            raise ValueError("Saved train group has an unknown source ordinal.")
        raw = groups[ordinal]
        curve, view = _definition(raw, recipe)
        provenance = {
            "replay_rule_version": REPLAY_RULE,
            "source_index_sha256": source["index_sha256"],
            "saved_group": raw,
        }
        if (
            group["trajectory_id"] != _trajectory_id(plan, raw)
            or group["curve"] != curve.to_dict()
            or group["view"] != view
            or group["provenance"] != provenance
        ):
            raise ValueError(
                "Saved train group changed its original split, curve, centre or view."
            )
