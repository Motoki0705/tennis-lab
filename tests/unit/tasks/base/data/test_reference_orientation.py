from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.base.data.reference_orientation import (
    REFERENCE_BASELINE_AMBIGUITY_MARGIN_M,
    deterministic_sample_rng,
    orientation_signs_from_camera_centers,
    reflect_court_vectors,
    reflect_heading,
    select_counterfactual_reference_views,
    select_reference_view,
    validate_declared_reference_orientation,
)


def test_deterministic_sample_rng_is_order_independent() -> None:
    first = deterministic_sample_rng(719, "scene-a").integers(0, 10_000, size=4)
    _ = deterministic_sample_rng(719, "scene-b").integers(0, 10_000, size=4)
    repeated = deterministic_sample_rng(719, "scene-a").integers(
        0, 10_000, size=4
    )

    assert np.array_equal(first, repeated)


def test_baseline_distance_orientation_has_declared_signs_and_margin() -> None:
    centers = torch.tensor(
        [[0.0, -20.0, 3.0], [0.0, 20.0, 3.0], [20.0, 0.0, 3.0]]
    )

    signs, valid = orientation_signs_from_camera_centers(centers)

    torch.testing.assert_close(signs, torch.tensor([1.0, -1.0, -1.0]))
    assert torch.equal(valid, torch.tensor([True, True, False]))


def test_baseline_ambiguity_policy_is_strictly_one_meter() -> None:
    assert pytest.approx(1.0) == REFERENCE_BASELINE_AMBIGUITY_MARGIN_M
    centers = torch.tensor(
        [
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.499, 0.0],
            [0.0, 0.499, 0.0],
        ],
        dtype=torch.float64,
    )

    signs, valid = orientation_signs_from_camera_centers(centers)

    torch.testing.assert_close(
        signs,
        torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64),
    )
    assert torch.equal(valid, torch.tensor([True, True, False, False]))


def test_declared_reference_validation_rejects_ambiguous_and_wrong_sign() -> None:
    centers = torch.tensor(
        [[[0.0, -0.499, 0.0], [0.0, 0.5, 0.0]]],
        dtype=torch.float64,
    )
    view_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="orientation-ambiguous"):
        validate_declared_reference_orientation(
            centers,
            view_mask,
            torch.tensor([0]),
            torch.tensor([1.0], dtype=torch.float64),
        )

    with pytest.raises(ValueError, match="does not match"):
        validate_declared_reference_orientation(
            centers,
            view_mask,
            torch.tensor([1]),
            torch.tensor([1.0], dtype=torch.float64),
        )


def test_reference_selection_is_deterministic_and_rejects_side_on_only() -> None:
    centers = torch.tensor([[0.0, -20.0, 3.0], [0.0, 20.0, 3.0]])
    first = select_reference_view(centers, rng=np.random.default_rng(719))
    second = select_reference_view(centers, rng=np.random.default_rng(719))
    assert first == second

    with pytest.raises(ValueError, match="no reference camera"):
        select_reference_view(
            torch.tensor([[20.0, 0.0, 3.0]]),
            rng=np.random.default_rng(719),
        )


def test_position_vector_and_heading_reflect_only_y() -> None:
    vectors = torch.tensor([[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]])
    signs = torch.tensor([-1.0, 1.0])
    reflected = reflect_court_vectors(vectors, signs)
    torch.testing.assert_close(
        reflected,
        torch.tensor([[[[1.0, -2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]]),
    )

    heading = torch.tensor([[[[0.6, 0.8]]], [[[0.0, 2.0]]]])
    reflected_heading = reflect_heading(heading, signs)
    torch.testing.assert_close(
        reflected_heading,
        torch.tensor([[[[0.6, -0.8]]], [[[0.0, 1.0]]]]),
    )


def test_counterfactual_reference_prefers_opposite_side_and_requires_alternate() -> None:
    centers = torch.tensor(
        [[[0.0, -20.0, 3.0], [0.0, -18.0, 3.0], [0.0, 20.0, 3.0]]]
    )
    index, sign = select_counterfactual_reference_views(
        centers,
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([0]),
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(index, torch.tensor([2]))
    torch.testing.assert_close(sign, torch.tensor([-1.0]))

    with pytest.raises(ValueError, match="valid alternate"):
        select_counterfactual_reference_views(
            centers[:, :1],
            torch.ones(1, 1, dtype=torch.bool),
            torch.tensor([0]),
            torch.tensor([1.0]),
        )
