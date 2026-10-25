"""Densify the exact train groups of an explicitly pinned published Court store.

The source metadata, including its original split, centre and selected view, is
the authority. This importer never re-runs budget selection or split assignment.
New-point jitter uses scale's stable-point disk RNG, preserving the legacy
uniform-area distribution without reusing its sample-index RNG stream.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitSamplingPolicy,
    OrbitTrajectorySpec,
)
from src.synthetic_data_generation.dataset.court.sample_store import open_court_store
from src.synthetic_data_generation.scene_contract import MultiCourtLayout, SceneCamera
from src.utils.data.image_record_store import file_sha256

from .core import GeometryRecipe, TrajectoryMethod, digest, require_integer, stable_id
from .curves import ArcMap, Curve
from .planning import PLAN_SCHEMA, point_record, scene_geometry_id, view_definition
from .strategies import _orbit_curve, strategy_definition

SOURCE_SCHEMA = "court_saved_train_source_v1"
REPLAY_RULE = "published_train_group_geometry_v1"


def pin_saved_source(
    store_root: Path, *, dataset_sha256: str, index_sha256: str
) -> dict[str, Any]:
    """Require caller-supplied hashes; a nearby or newer dataset is never used."""
    if (
        file_sha256(store_root / "dataset.json") != dataset_sha256
        or file_sha256(store_root / "samples/index.npz") != index_sha256
    ):
        raise ValueError("Pinned source Court dataset/index hash mismatch.")
    store = open_court_store(store_root)
    metadata = store.metadata
    if metadata["schema"] != "canonical_court_dataset_v3":
        raise ValueError("Saved train import requires the explicit v3 source schema.")
    dimensions = {
        (store.record(i)["width"], store.record(i)["height"]) for i in range(len(store))
    }
    if len(dimensions) != 1:
        raise ValueError("Saved train source requires one explicit image resolution.")
    width, height = dimensions.pop()
    return {
        "schema": SOURCE_SCHEMA,
        "scene_id": metadata["scene_id"],
        "dataset_sha256": dataset_sha256,
        "index_sha256": index_sha256,
        "source_seed": metadata["seed"],
        "sampling_policy": metadata["sampling_policy"],
        "width": width,
        "height": height,
        "trajectory_groups": metadata["trajectory_groups"],
    }


def saved_train_recipe(source: dict[str, Any]) -> GeometryRecipe:
    """Arc/resolution authority is saved; each group's view remains independent.

    GeometryRecipe's HFOV/height layer lists describe procedural generators and
    are unused by this importer. The exact saved view is checked independently.
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
            "Saved train import requires one exact selected view per group."
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


def saved_train_header(
    *,
    source: dict[str, Any],
    seed: int,
    density_level: int,
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
) -> dict[str, Any]:
    require_integer(seed, "generation_seed", minimum=0)
    require_integer(density_level, "density_level", minimum=0)
    if density_level > 20:
        raise ValueError("Saved density_level exceeds 20.")
    recipe = saved_train_recipe(source)
    return {
        "schema": PLAN_SCHEMA,
        "scene_id": source["scene_id"],
        "method": TrajectoryMethod.SAVED_TRAIN_TRAJECTORIES.value,
        "recipe": recipe.to_dict(),
        "recipe_sha256": digest(recipe.to_dict()),
        "generation_seed": seed,
        "trajectory_count": len(_train_groups(source)),
        "density_level": density_level,
        "split": "train",
        "strategy": strategy_definition(),
        "scene_geometry_id": scene_geometry_id(layout, cameras),
        "saved_source": source,
        "saved_replay_rule": REPLAY_RULE,
        "view_authority": "saved_source.trajectory_groups[].views",
    }


def iter_saved_train_groups(
    plan: dict[str, Any], layout: MultiCourtLayout
) -> Iterator[dict[str, Any]]:
    recipe = GeometryRecipe.from_dict(plan["recipe"])
    layout = replace(
        layout, courts=tuple(sorted(layout.courts, key=lambda c: c.court_instance_id))
    )
    physical_curves: dict[str, str] = {}
    for ordinal, raw in enumerate(_train_groups(plan["saved_source"])):
        curve, view = _definition(raw, recipe)
        arc = ArcMap.freeze(curve, recipe)
        if arc.base_count != raw["sample_count"]:
            raise ValueError("Imported base arc count differs from the saved group.")
        legacy = sample_uniform_arc_length(
            OrbitTrajectorySpec.from_mapping(raw["trajectory"]),
            OrbitCenter.from_mapping(raw["center"]),
            OrbitSamplingPolicy.from_mapping(plan["saved_source"]["sampling_policy"]),
        )
        positions = np.stack([arc.at(q) for q in arc.fractions(0)])
        if not np.allclose(positions, legacy.points_scene_m, rtol=0, atol=1e-9):
            raise ValueError(
                "Imported curve does not reproduce the saved base camera positions."
            )
        trajectory_id = _trajectory_id(plan, raw)
        physical = curve.physical_key()
        if len(layout.courts) == 1:
            physical["center_court_id"] = layout.courts[0].court_instance_id
        fingerprint = stable_id("physical_curve_view", physical, view)
        duplicate = physical_curves.get(fingerprint)
        physical_curves.setdefault(fingerprint, trajectory_id)
        provenance = {
            "replay_rule_version": REPLAY_RULE,
            "source_index_sha256": plan["saved_source"]["index_sha256"],
            "saved_group": raw,
        }
        samples = []
        for q in arc.fractions(plan["density_level"]):
            record = point_record(
                curve=curve,
                arc=arc,
                q=q,
                trajectory_id=trajectory_id,
                view=view,
                seed=plan["generation_seed"],
                recipe=recipe,
                layout=layout,
            )
            if duplicate is not None:
                record["reasons"].append("exact_duplicate_curve_view")
            record["scale"]["comparison_pair_id"] = stable_id(
                "comparison_pair",
                plan["scene_id"],
                plan["scene_geometry_id"],
                plan["method"],
                plan["generation_seed"],
                provenance,
                [q.numerator, q.denominator],
            )
            samples.append(record)
        yield {
            "ordinal": ordinal,
            "trajectory_id": trajectory_id,
            "provenance": provenance,
            "view": view,
            "curve": curve.to_dict(),
            "reasons": [] if duplicate is None else ["exact_duplicate_curve_view"],
            "duplicate_of": duplicate,
            "arc_map": arc.to_dict(),
            "samples": samples,
            "planned_points_scene_m": curve.at(np.linspace(0, 1, 257)).tolist(),
        }


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
