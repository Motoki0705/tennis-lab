from fractions import Fraction

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    TrajectoryMethod,
    disk_offset,
    field_rng,
)
from src.synthetic_data_generation.dataset.court.scale.curves import ArcMap, Curve
from src.synthetic_data_generation.scene_contract import RigidTransform


@pytest.mark.parametrize("shape", ["circle", "ellipse", "rectangle", "superellipse"])
@pytest.mark.parametrize("amplitude", [0.0, 0.4])
def test_frozen_arc_doubles_without_moving_existing_points(shape, amplitude):
    curve = Curve(
        "orbit",
        RigidTransform.identity(),
        {
            "shape": shape,
            "radius_x_m": 9.1,
            "radius_y_m": 5.2,
            "orientation": 0.3,
            "height": 2.0,
            "amplitude": amplitude,
            "phase": 0.2,
        },
        None,
    )
    recipe = GeometryRecipe(base_step_m=2.3, minimum_points=11, count_multiple=3)
    arc = ArcMap.freeze(curve, recipe)
    assert arc.base_count % 3 == 0
    coarse = arc.fractions(0)
    fine = arc.fractions(2)
    assert coarse == fine[::4]
    assert len(fine) == 4 * len(coarse)
    assert max(coarse) < 1
    for q in coarse:
        assert arc.point_id("trajectory", "view", q) == arc.point_id(
            "trajectory", "view", Fraction(q.numerator * 4, q.denominator * 4)
        )
        assert np.array_equal(arc.at(q), arc.at(fine[fine.index(q)]))
    points = np.stack([arc.at(q) for q in fine])
    assert (
        np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1).max()
        <= recipe.base_step_m / 4 + recipe.arc_tolerance_m
    )


@pytest.mark.parametrize(
    "kind,controls",
    [
        ("line", [[0, 0, 2], [2.7, 0, 2]]),
        ("bezier", [[0, 0, 2], [1, 0.3, 2.1], [2, -0.3, 2.2], [3, 0, 2]]),
    ],
)
def test_open_curve_is_half_open_and_retains_endpoint_in_plan(kind, controls):
    curve = Curve(kind, RigidTransform.identity(), {"controls": controls}, None)
    arc = ArcMap.freeze(curve, GeometryRecipe(minimum_points=9, count_multiple=4))
    assert arc.base_count == 12
    assert not curve.closed
    assert np.array_equal(arc.at(Fraction(1)), controls[-1])
    assert Fraction(1) not in arc.fractions(2)
    assert np.linalg.norm(arc.at(Fraction(1)) - arc.at(arc.fractions(0)[-1])) > 0
    assert Curve.from_dict(curve.to_dict()).to_dict() == curve.to_dict()


def test_field_randomness_is_independent_and_recipe_is_strict():
    original = disk_offset(7, "fixed-point", 1.0)
    field_rng(7, "fixed-point", "new_field").random(100)
    assert np.array_equal(original, disk_offset(7, "fixed-point", 1.0))
    assert not np.array_equal(original, disk_offset(8, "fixed-point", 1.0))
    assert not np.array_equal(original, disk_offset(7, "other-point", 1.0))
    assert np.linalg.norm(original) <= 1
    recipe = GeometryRecipe()
    assert GeometryRecipe.from_dict(recipe.to_dict()) == recipe
    with pytest.raises(ValueError):
        GeometryRecipe.from_dict({**recipe.to_dict(), "worker_count": 4})
    with pytest.raises(ValueError):
        TrajectoryMethod("unknown")
    with pytest.raises(ValueError):
        GeometryRecipe(arc_intervals=17)
    with pytest.raises(ValueError):
        Curve("circle", RigidTransform.identity(), {}, None)
