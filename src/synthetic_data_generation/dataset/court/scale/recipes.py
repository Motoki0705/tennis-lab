"""Named production authority, separate from freely selected comparison recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import GeometryRecipe, QualityPolicy, TrajectoryMethod, digest

PRODUCTION_RECIPE_NAME = "stratified_counter_production_v1"


@dataclass(frozen=True)
class ProductionRecipe:
    schema: str
    name: str
    method: TrajectoryMethod
    geometry: GeometryRecipe
    quality: QualityPolicy
    jitter: str

    def __post_init__(self) -> None:
        if (
            self.schema != "court_scale_production_recipe_v1"
            or self.name != PRODUCTION_RECIPE_NAME
            or self.method is not TrajectoryMethod.STRATIFIED_COUNTER
        ):
            raise ValueError("Unknown production recipe name/version or method.")
        if (
            self.geometry.hull_expansion_percent != 0.0
            or self.geometry.hull_margin_m != 0.5
            or self.geometry.hfov_degrees != (75.0, 92.5, 110.0)
            or self.geometry.jitter_radius_m != 1.0
            or self.jitter != "stable_point_uniform_area_disk_v1"
            or self.quality.schema != "court_scale_quality_policy_v2"
        ):
            raise ValueError("Production recipe must preserve the agreed D2/D5/D6.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "method": self.method.value,
            "geometry": self.geometry.to_dict(),
            "quality": self.quality.to_dict(),
            "jitter": self.jitter,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProductionRecipe:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Production recipe requires every explicit field.")
        if set(value["quality"]) != set(QualityPolicy.__dataclass_fields__):
            raise ValueError("Production recipe requires the complete D5 policy.")
        return cls(
            **{
                **value,
                "method": TrajectoryMethod(value["method"]),
                "geometry": GeometryRecipe.from_dict(value["geometry"]),
                "quality": QualityPolicy(**value["quality"]),
            }
        )

    def validate_plan(self, plan: dict[str, Any]) -> None:
        if (
            plan["method"] != self.method.value
            or plan["recipe"] != self.geometry.to_dict()
            or plan["recipe_sha256"] != digest(self.geometry.to_dict())
        ):
            raise ValueError("Candidate plan disagrees with the production recipe.")
