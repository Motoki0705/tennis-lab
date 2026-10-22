"""Scale-profile geometry validation; never invoke the legacy index-jitter reader."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

from src.synthetic_data_generation.scene_contract import MultiCourtLayout

from .core import GeometryRecipe, TrajectoryMethod, digest, stable_id
from .curves import ArcMap, Curve
from .planning import point_record
from .strategies import strategy_definition


def validate_plan_geometry(
    plan: dict[str, Any], layout: MultiCourtLayout
) -> dict[str, dict[str, Any]]:
    TrajectoryMethod(plan["method"])
    if plan["strategy"] != strategy_definition():
        raise ValueError("Candidate plan RNG/Sobol/runtime version mismatch.")
    recipe = GeometryRecipe.from_dict(plan["recipe"])
    if digest(recipe.to_dict()) != plan["recipe_sha256"]:
        raise ValueError("Candidate plan recipe hash mismatch.")
    layout = replace(
        layout, courts=tuple(sorted(layout.courts, key=lambda c: c.court_instance_id))
    )
    planned: dict[str, dict[str, Any]] = {}
    for group in plan["trajectory_groups"]:
        if group["curve"] is None:
            if group["samples"] or not group["reasons"]:
                raise ValueError("Rejected curve has samples or no rejection reason.")
            continue
        curve = Curve.from_dict(group["curve"])
        arc = ArcMap.freeze(curve, recipe)
        if arc.to_dict() != group["arc_map"]:
            raise ValueError("Candidate plan frozen arc map changed.")
        fractions = arc.fractions(plan["density_level"])
        if len(fractions) != len(group["samples"]):
            raise ValueError("Candidate plan density/count mismatch.")
        for q, record in zip(fractions, group["samples"], strict=True):
            expected = point_record(
                curve=curve,
                arc=arc,
                q=q,
                trajectory_id=group["trajectory_id"],
                view=group["view"],
                seed=plan["generation_seed"],
                recipe=recipe,
                layout=layout,
            )
            if group["duplicate_of"] is not None:
                expected["reasons"].append("exact_duplicate_curve_view")
            expected["scale"]["comparison_pair_id"] = stable_id(
                "comparison_pair",
                plan["scene_id"],
                plan["scene_geometry_id"],
                plan["method"],
                plan["generation_seed"],
                group["provenance"],
                [q.numerator, q.denominator],
            )
            if expected != record or record["sample_id"] in planned:
                raise ValueError(
                    "Candidate stable ID/camera/target/projection geometry mismatch."
                )
            planned[record["sample_id"]] = record
    return planned


def validate_record_geometry(record: dict[str, Any], planned: dict[str, Any]) -> None:
    for key, value in planned.items():
        if key not in {"reasons", "projection"} and record.get(key) != value:
            raise ValueError(
                f"Stored scale sample {key} disagrees with stable plan geometry."
            )

    def geometric(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: geometric(v)
                for k, v in value.items()
                if k
                not in {
                    "renderer_visible",
                    "visible_point_count",
                    "visible_class_names",
                }
            }
        if isinstance(value, list):
            return [geometric(v) for v in value]
        return value

    if geometric(record["projection"]) != geometric(planned["projection"]):
        raise ValueError("Stored v3 KP projection disagrees with planned geometry.")
    if not record.get("reasons") and record["projection"] is not None:
        flags = [
            p["renderer_visible"]
            for c in record["projection"]["courts"]
            for cls in c["classes"]
            for p in cls["points"]
        ]
        if any(type(v) is not bool for v in flags) or not any(flags):
            raise ValueError(
                "Accepted scale sample lacks renderer visibility evidence."
            )


def arc_fraction(record: dict[str, Any]) -> Fraction:
    return Fraction(*record["scale"]["arc_fraction"])
