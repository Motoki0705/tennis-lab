"""Five explicit camera-path sequences; rendering and sampling are shared."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import qmc

from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    _views_for_group_v2,
    select_budgeted_coverage,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sfm_bounds import (
    bound_trajectory_candidates,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.trajectory import (
    derive_orbit_centers,
    generate_trajectory_candidates,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitCenterKind,
    OrbitCurveMode,
    OrbitSamplingPolicy,
    OrbitShape,
    OrbitTrajectorySpec,
)
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    SceneCamera,
)

from .core import GeometryRecipe, TrajectoryMethod, field_rng, stable_id
from .curves import Curve

STRATEGY_SCHEMA = "court_scale_strategies_v1"
STRATA = tuple(
    itertools.product(
        ("circle", "ellipse", "rectangle", "superellipse"),
        ("complex", "court"),
        ("planar", "sinusoidal_height"),
        range(3),
        range(2),
    )
)


@dataclass(frozen=True)
class CandidateCurve:
    ordinal: int
    provenance: dict[str, Any]
    curve: Curve | None
    hfov_degrees: float
    look_at_height_m: float
    reason: str | None = None


def strategy_definition() -> dict[str, Any]:
    """Pin implementation and numeric distributions along with the recipe."""
    return {
        "schema": STRATEGY_SCHEMA,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "rng": "PCG64_SeedSequence_sha256_le_u32_v1",
        "sobol_bits": 30,
        "sobol_scramble": True,
        "sobol_optimization": None,
        "strata": [list(s) for s in STRATA],
        "scale": [0.65, 0.95],
        "axis_ratio": [0.35, 0.8],
        "orientation": [0.0, math.pi],
        "height_m": [1.5, 3.0],
        "amplitude_m": [0.05, 0.5],
        "phase": [0.0, 2 * math.pi],
        "local_length_m": [0.5, 2.0],
        "bezier_pair_distance_m": [2.0, 8.0],
        "bezier_delta_xy_radius_m": 0.5,
        "bezier_delta_z_m": [-0.25, 0.25],
    }


class CurveSequence:
    def __init__(
        self,
        *,
        method: TrajectoryMethod,
        recipe: GeometryRecipe,
        scene_geometry_id: str,
        seed: int,
        cameras: Sequence[SceneCamera],
        layout: MultiCourtLayout,
        replay_configuration: CourtDatasetConfiguration | None = None,
    ) -> None:
        if not isinstance(method, TrajectoryMethod):
            raise ValueError("An explicit TrajectoryMethod is required.")
        if method is TrajectoryMethod.SAVED_TRAIN_TRAJECTORIES:
            raise ValueError(
                "Saved train trajectories require the explicit saved_train importer."
            )
        if (method is TrajectoryMethod.GRID_BUDGET_REPLAY) != (
            replay_configuration is not None
        ):
            raise ValueError(
                "Saved replay configuration is required exactly for grid_budget_replay."
            )
        self.method, self.recipe, self.seed = method, recipe, seed
        self.cameras = tuple(sorted(cameras, key=lambda c: c.camera_id))
        if len({c.camera_id for c in self.cameras}) != len(self.cameras):
            raise ValueError("Duplicate captured camera IDs.")
        self.layout, self.replay_configuration = layout, replay_configuration
        self.centers = derive_orbit_centers(
            self.cameras,
            layout,
            use_court_centroid=True,
            use_captured_hull_centroid=True,
        )
        self.reference = self.centers[0].scene_from_center
        self.local = self.reference.inverse().apply(
            np.stack([c.camera_to_scene.matrix()[:3, 3] for c in self.cameras])
        )
        hull = ConvexHull(self.local[:, :2])
        self.normals = hull.equations[:, :2]
        origin = self.local[hull.vertices, :2].mean(axis=0)
        distances = -(self.normals @ origin + hull.equations[:, 2])
        self.allowance = (
            -hull.equations[:, 2]
            + recipe.hull_expansion_percent / 100 * distances
            - recipe.hull_margin_m
        )
        self.scope = [STRATEGY_SCHEMA, method.value, scene_geometry_id, seed]

    def inside(self, xy: np.ndarray) -> bool:
        return bool(np.all(np.asarray(xy) @ self.normals.T <= self.allowance + 1e-9))

    def __iter__(self) -> Iterator[CandidateCurve]:
        if self.method is TrajectoryMethod.GRID_BUDGET_REPLAY:
            yield from self._grid()
        elif self.method in {
            TrajectoryMethod.STRATIFIED_COUNTER,
            TrajectoryMethod.SOBOL,
        }:
            yield from self._stratified()
        elif self.method is TrajectoryMethod.CAPTURE_LOCAL:
            yield from self._capture_local()
        elif self.method is TrajectoryMethod.OPEN_BEZIER:
            yield from self._bezier()
        else:
            raise ValueError(f"Unknown trajectory method: {self.method}")

    def _stratified(self) -> Iterator[CandidateCurve]:
        # Counter and Sobol share this exact ordering and all discrete choices.
        scope = [*self.scope[:1], "stratified_common", *self.scope[2:]]
        order = field_rng(self.seed, scope, "stratum_order").permutation(len(STRATA))
        sobols: dict[int, Any] = {}
        for ordinal in itertools.count():
            stratum_id = int(order[ordinal % len(order)])
            counter = ordinal // len(order)
            shape, kind, profile, hfov, height = STRATA[stratum_id]
            identity = [scope, stratum_id, counter]
            if self.method is TrajectoryMethod.SOBOL:
                if stratum_id not in sobols:
                    scramble_seed = int(
                        stable_id("sobol_scramble", scope, stratum_id)[:8], 16
                    )
                    sobols[stratum_id] = qmc.Sobol(
                        d=6,
                        scramble=True,
                        bits=30,
                        optimization=None,
                        seed=scramble_seed,
                    )
                values = sobols[stratum_id].random(1)[0]
            else:
                values = [
                    float(field_rng(self.seed, identity, field).random())
                    for field in (
                        "scale",
                        "ratio",
                        "orientation",
                        "height",
                        "amplitude",
                        "phase",
                    )
                ]
            scale, ratio, angle, z, amplitude, phase = values
            ratio = 1.0 if shape == "circle" else 0.35 + 0.45 * ratio
            angle = 0.0 if shape == "circle" else math.pi * angle
            center = (
                self.centers[0]
                if kind == "complex"
                else self.centers[1 + counter % (len(self.centers) - 1)]
            )
            spec = OrbitTrajectorySpec(
                "pending",
                "pending",
                OrbitShape(shape),
                OrbitCenterKind(kind),
                center.court_instance_id,
                center.base_radius_m,
                0.65 + 0.3 * scale,
                ratio,
                angle,
                1.5 + 1.5 * z,
                0.0 if profile == "planar" else 0.05 + 0.45 * amplitude,
                0 if profile == "planar" else 1,
                0.0 if profile == "planar" else 2 * math.pi * phase,
                OrbitCurveMode(profile),
            )
            provenance = {
                "stratum": list(STRATA[stratum_id]),
                "stratum_id": stratum_id,
                "counter": counter,
            }
            try:
                bounded = bound_trajectory_candidates(
                    (spec,),
                    centers=(center,),
                    cameras=self.cameras,
                    margin_m=self.recipe.hull_margin_m,
                    expansion_percent=self.recipe.hull_expansion_percent,
                )[0]
            except ValueError as error:
                yield CandidateCurve(
                    ordinal,
                    provenance,
                    None,
                    self.recipe.hfov_degrees[hfov],
                    self.recipe.look_at_heights_m[height],
                    f"curve_boundary:{error}",
                )
                continue
            yield CandidateCurve(
                ordinal,
                provenance,
                _orbit_curve(bounded, center),
                self.recipe.hfov_degrees[hfov],
                self.recipe.look_at_heights_m[height],
            )

    def _view(self, ordinal: int) -> tuple[float, float]:
        # Repeating fixed six-layer order, independent of the requested prefix.
        order = field_rng(self.seed, self.scope, "view_order").permutation(6)
        layer = int(order[ordinal % 6])
        return self.recipe.hfov_degrees[layer % 3], self.recipe.look_at_heights_m[
            layer // 3
        ]

    def _capture_local(self) -> Iterator[CandidateCurve]:
        order = sorted(
            range(len(self.cameras)),
            key=lambda i: stable_id("anchor", self.scope, self.cameras[i].camera_id),
        )
        for ordinal in itertools.count():
            anchor = order[ordinal % len(order)]
            camera = self.cameras[anchor]
            provenance = {"anchor_id": camera.camera_id, "cycle": ordinal // len(order)}
            identity = [self.scope, provenance]
            p = self.local[anchor]
            hfov, height = self._view(ordinal)
            if not self.inside(p[:2]):
                yield CandidateCurve(
                    ordinal, provenance, None, hfov, height, "anchor_outside_inset_hull"
                )
                continue
            angle = float(
                field_rng(self.seed, identity, "direction").uniform(0, 2 * math.pi)
            )
            direction = np.array([math.cos(angle), math.sin(angle), 0.0])
            denominators = np.abs(self.normals @ direction[:2])
            distances = self.allowance - self.normals @ p[:2]
            limits = np.divide(
                2 * distances,
                denominators,
                out=np.full_like(distances, np.inf),
                where=denominators > 1e-12,
            )
            upper = min(2.0, float(limits.min()))
            if upper < 0.5:
                yield CandidateCurve(
                    ordinal, provenance, None, hfov, height, "local_segment_too_short"
                )
                continue
            length = float(field_rng(self.seed, identity, "length").uniform(0.5, upper))
            controls = np.stack(
                [p - length / 2 * direction, p + length / 2 * direction]
            )
            yield CandidateCurve(
                ordinal,
                provenance,
                Curve("line", self.reference, {"controls": controls.tolist()}, None),
                hfov,
                height,
            )

    def _bezier(self) -> Iterator[CandidateCurve]:
        valid = [i for i, p in enumerate(self.local) if self.inside(p[:2])]
        points = self.local[valid]
        if len(valid) < 2:
            return
        pairs = [
            (valid[a], valid[b])
            for a, b in cKDTree(points[:, :2]).query_pairs(8.0)
            if np.linalg.norm(points[a, :2] - points[b, :2]) >= 2.0
        ]
        pairs.sort(
            key=lambda ab: stable_id(
                "pair",
                self.scope,
                self.cameras[ab[0]].camera_id,
                self.cameras[ab[1]].camera_id,
            )
        )
        for ordinal, (a, b) in enumerate(pairs):
            provenance = {
                "anchor_ids": [self.cameras[a].camera_id, self.cameras[b].camera_id]
            }
            identity = [self.scope, provenance]
            p0, p3 = self.local[a], self.local[b]
            controls = np.stack([p0, (2 * p0 + p3) / 3, (p0 + 2 * p3) / 3, p3])
            for i in (1, 2):
                radius = 0.5 * math.sqrt(
                    float(field_rng(self.seed, identity, f"control{i}.radius").random())
                )
                angle = float(
                    field_rng(self.seed, identity, f"control{i}.angle").uniform(
                        0, 2 * math.pi
                    )
                )
                z = float(
                    field_rng(self.seed, identity, f"control{i}.z").uniform(-0.25, 0.25)
                )
                controls[i] += [radius * math.cos(angle), radius * math.sin(angle), z]
            hfov, height = self._view(ordinal)
            if not self.inside(controls[:, :2]):
                yield CandidateCurve(
                    ordinal,
                    provenance,
                    None,
                    hfov,
                    height,
                    "bezier_controls_outside_inset_hull",
                )
                continue
            try:
                curve = Curve(
                    "bezier", self.reference, {"controls": controls.tolist()}, None
                )
            except ValueError as error:
                yield CandidateCurve(
                    ordinal, provenance, None, hfov, height, f"bezier_geometry:{error}"
                )
                continue
            yield CandidateCurve(ordinal, provenance, curve, hfov, height)

    def _grid(self) -> Iterator[CandidateCurve]:
        config = self.replay_configuration
        assert config is not None
        if config.schema_version.value != "v3":
            raise ValueError(
                "Grid replay currently requires an explicit saved v3 configuration."
            )
        centers = derive_orbit_centers(
            self.cameras,
            self.layout,
            use_court_centroid=config.trajectory.sfm_boundary_margin_m is not None,
            use_captured_hull_centroid=config.trajectory.sfm_complex_center_on_hull,
        )
        policy = OrbitSamplingPolicy.from_configuration(config.sampling)
        candidates = generate_trajectory_candidates(
            config.trajectory,
            centers,
            seed=policy.seed,
            stable_field_order=policy.stable_field_order,
        )
        if config.trajectory.sfm_boundary_margin_m is not None:
            candidates = bound_trajectory_candidates(
                candidates,
                centers=centers,
                cameras=self.cameras,
                margin_m=config.trajectory.sfm_boundary_margin_m,
                expansion_percent=config.trajectory.sfm_boundary_expansion_percent,
            )
        selected = select_budgeted_coverage(
            candidates,
            centers=centers,
            policy=policy,
            first_group_view_count=1,
            spatial_coverage_cell_m=config.trajectory.spatial_coverage_cell_m,
        )
        for ordinal, item in enumerate(selected):
            views = _views_for_group_v2(
                group_index=ordinal,
                group_id=item.trajectory.trajectory_group_id,
                configuration=config,
            )
            if len(views) != 1:
                raise ValueError(
                    "Grid replay requires exactly one court-centre view per group."
                )
            yield CandidateCurve(
                ordinal,
                {
                    "replay_rule_version": "saved_preset_parameterization_v2",
                    "saved_trajectory": item.trajectory.to_dict(),
                    "saved_view": views[0].to_dict(),
                },
                _orbit_curve(item.trajectory, item.center),
                views[0].hfov_degrees,
                views[0].look_at_height_m,
            )


def _orbit_curve(spec: OrbitTrajectorySpec, center: OrbitCenter) -> Curve:
    # Preserve the saved 3-D parameterization. New samples normalize at construction;
    # duplicate identity is handled separately by Curve.physical_key().
    return Curve(
        "orbit",
        center.scene_from_center,
        {
            "shape": spec.shape.value,
            "radius_x_m": spec.radius_x_m,
            "radius_y_m": spec.radius_y_m,
            "orientation": spec.orientation_radians,
            "height": spec.base_height_m,
            "amplitude": spec.vertical_amplitude_m,
            "phase": spec.vertical_phase_radians,
            "cycles": spec.vertical_cycles,
        },
        spec.center_court_instance_id,
    )
