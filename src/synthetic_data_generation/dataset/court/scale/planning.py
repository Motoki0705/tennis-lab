"""Common stable cameras/targets and pre-render candidate manifests."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from fractions import Fraction
from typing import Any

import numpy as np

from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    _look_at_opencv,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    resolve_target_court,
)
from src.synthetic_data_generation.dataset.court.components.labels import (
    project_court_semantics_v3,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    TargetCourtPolicyV2,
    TargetCourtResolutionPolicy,
)
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)

from .core import (
    GeometryRecipe,
    TrajectoryMethod,
    digest,
    disk_offset,
    require_integer,
    stable_id,
)
from .curves import ArcMap, Curve
from .strategies import CurveSequence, strategy_definition

PLAN_SCHEMA = "court_scale_candidate_plan_v1"


def scene_geometry_id(layout: MultiCourtLayout, cameras: Sequence[SceneCamera]) -> str:
    value = layout.to_dict()
    value["courts"] = sorted(value["courts"], key=lambda c: c["court_instance_id"])
    return str(
        stable_id(
            "scene_geometry",
            value,
            [c.to_dict() for c in sorted(cameras, key=lambda c: c.camera_id)],
        )
    )


def view_definition(
    recipe: GeometryRecipe, hfov: float, look_height: float
) -> dict[str, Any]:
    return {
        "schema": "court_scale_view_v1",
        "target_policy": "center_court_else_nearest_camera",
        "hfov_degrees": hfov,
        "look_at_height_m": look_height,
        "look_at_jitter_radius_m": recipe.jitter_radius_m,
        "jitter": "stable_point_uniform_area_disk_v1",
        "width": recipe.width,
        "height": recipe.height,
    }


def point_record(
    *,
    curve: Curve,
    arc: ArcMap,
    q: Fraction,
    trajectory_id: str,
    view: dict[str, Any],
    seed: int,
    recipe: GeometryRecipe,
    layout: MultiCourtLayout,
) -> dict[str, Any]:
    view_id = stable_id("view", view)
    point_id = arc.point_id(trajectory_id, view_id, q)
    position = arc.at(q)
    policy = TargetCourtPolicyV2(
        mode=TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT
        if curve.center_court_id is not None
        else TargetCourtResolutionPolicy.NEAREST_CAMERA,
        centre_court_instance_id=curve.center_court_id,
    )
    target = resolve_target_court(
        policy=policy,
        camera_center_scene_m=position,
        layout=layout,
        selection_seed=seed,
    )
    local_target = disk_offset(seed, point_id, recipe.jitter_radius_m) + [
        0,
        0,
        view["look_at_height_m"],
    ]
    look_at = target.binding.scene_from_court.apply(local_target[None])[0]
    focal = recipe.width / (2 * math.tan(math.radians(view["hfov_degrees"]) / 2))
    pose = _look_at_opencv(
        position, look_at, vertical_scene=curve.scene_from_local.matrix()[:3, 2]
    )
    camera = SceneCamera(
        point_id,
        0,
        recipe.width,
        recipe.height,
        (
            focal,
            0.0,
            (recipe.width - 1) / 2,
            0.0,
            focal,
            (recipe.height - 1) / 2,
            0.0,
            0.0,
            1.0,
        ),
        RigidTransform.from_matrix(pose),
        f"generated/court/{point_id}.jpg",
    )
    birth = 0
    while (arc.base_count * 2**birth) % q.denominator:
        birth += 1
    reasons: list[str] = []
    try:
        projection = project_court_semantics_v3(
            camera, layout, near_plane_m=recipe.near_plane_m
        ).to_dict()
    except ValueError as error:
        if "ambiguous_camera_relative_near_far" not in str(error):
            raise
        projection = None
        reasons.append(str(error))
    if projection is not None:
        in_frame = sum(
            p["in_frame"]
            for c in projection["courts"]
            for cls in c["classes"]
            for p in cls["points"]
        )
        if in_frame < 4:
            reasons.append("insufficient_in_frame_semantic_points")
    metadata = {
        "target_court": target.binding.court_instance_id,
        "candidate_id": target.binding.candidate_id,
        "transform": target.binding.scene_from_court.to_list(),
        "camera_profile": "scale_train_v1",
        "camera_parameters": {
            "view_id": view_id,
            "camera_center_scene_m": position.tolist(),
            "intrinsics": list(camera.intrinsics),
            "camera_to_scene": camera.camera_to_scene.to_list(),
        },
        "seed": seed,
    }
    return {
        "sample_id": point_id,
        "sample_index": int(point_id[:12], 16),
        "trajectory_group_id": trajectory_id,
        "trajectory_id": trajectory_id,
        "view_id": view_id,
        "trajectory_frame_index": q.numerator,
        "split": "train",
        "shard_id": f"birth-{birth}",
        "width": recipe.width,
        "height": recipe.height,
        "camera": camera.to_dict(),
        "target_court": target.to_dict(),
        "projection": projection,
        "metadata": metadata,
        "scale": {
            "schema": "court_scale_point_v1",
            "arc_fraction": [q.numerator, q.denominator],
            "birth_level": birth,
            "look_at_scene_m": look_at.tolist(),
        },
        "reasons": reasons,
    }


def iter_trajectory_plans(
    *,
    scene_id: str,
    method: TrajectoryMethod,
    recipe: GeometryRecipe,
    seed: int,
    trajectory_count: int,
    density_level: int,
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    replay_configuration: CourtDatasetConfiguration | None = None,
) -> Iterator[dict[str, Any]]:
    require_integer(trajectory_count, "trajectory_count", minimum=1)
    require_integer(density_level, "density_level", minimum=0)
    if density_level > 20:
        raise ValueError("density_level exceeds the explicit planning limit of 20.")
    geometry_id = scene_geometry_id(layout, cameras)
    sequence = CurveSequence(
        method=method,
        recipe=recipe,
        scene_geometry_id=geometry_id,
        seed=seed,
        cameras=cameras,
        layout=layout,
        replay_configuration=replay_configuration,
    )
    replay = None if replay_configuration is None else asdict(replay_configuration)
    scope = {
        "scene_id": scene_id,
        "geometry_id": geometry_id,
        "method": method.value,
        "recipe": recipe.to_dict(),
        "strategy": strategy_definition(),
        "seed": seed,
        "replay_configuration": replay,
    }
    physical_curves: dict[str, str] = {}
    emitted = 0
    for candidate in itertools.islice(sequence, trajectory_count):
        emitted += 1
        view = view_definition(
            recipe, candidate.hfov_degrees, candidate.look_at_height_m
        )
        curve = candidate.curve
        trajectory_id = stable_id(
            "trajectory",
            scope,
            candidate.provenance,
            None if curve is None else curve.to_dict(),
        )
        entry: dict[str, Any] = {
            "ordinal": candidate.ordinal,
            "trajectory_id": trajectory_id,
            "provenance": candidate.provenance,
            "view": view,
            "curve": None if curve is None else curve.to_dict(),
            "reasons": [] if candidate.reason is None else [candidate.reason],
            "samples": [],
            "arc_map": None,
            "duplicate_of": None,
        }
        if curve is not None:
            physical = stable_id("physical_curve_view", curve.to_dict(), view)
            if physical in physical_curves:
                entry["duplicate_of"] = physical_curves[physical]
                entry["reasons"].append("exact_duplicate_curve_view")
            else:
                physical_curves[physical] = trajectory_id
            arc = ArcMap.freeze(curve, recipe)
            entry["arc_map"] = arc.to_dict()
            entry["planned_points_scene_m"] = curve.at(np.linspace(0, 1, 257)).tolist()
            for q in arc.fractions(density_level):
                record = point_record(
                    curve=curve,
                    arc=arc,
                    q=q,
                    trajectory_id=trajectory_id,
                    view=view,
                    seed=seed,
                    recipe=recipe,
                    layout=layout,
                )
                if entry["duplicate_of"] is not None:
                    record["reasons"].append("exact_duplicate_curve_view")
                entry["samples"].append(record)
        yield entry
    if emitted != trajectory_count:
        raise ValueError(
            f"{method.value} candidate sequence exhausted: requested={trajectory_count}, available={emitted}."
        )


def build_candidate_plan(**kwargs: Any) -> dict[str, Any]:
    """In-memory convenience for previews; persistent planning streams per curve."""
    groups = list(iter_trajectory_plans(**kwargs))
    ids = [s["sample_id"] for group in groups for s in group["samples"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Stable point ID collision.")
    return {
        "schema": PLAN_SCHEMA,
        "scene_id": kwargs["scene_id"],
        "method": kwargs["method"].value,
        "recipe": kwargs["recipe"].to_dict(),
        "recipe_sha256": digest(kwargs["recipe"].to_dict()),
        "generation_seed": kwargs["seed"],
        "trajectory_count": kwargs["trajectory_count"],
        "density_level": kwargs["density_level"],
        "split": "train",
        "strategy": strategy_definition(),
        "scene_geometry_id": scene_geometry_id(kwargs["layout"], kwargs["cameras"]),
        "trajectory_groups": groups,
    }
