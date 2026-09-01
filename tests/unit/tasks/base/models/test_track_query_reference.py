"""Reference-selector track-query spatial M-RoPE contract tests."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.models.track_query_reference import (
    REFERENCE_SELECTOR_ROPE_CONTRACT,
    ROLE_ROPE_CONTRACT,
    ReferenceContextMaskError,
    ReferenceSelectorMode,
    TrackQueryReferenceModelError,
    TrackQueryRopeDimensionError,
    build_compressed_track_query_spatial_coordinates,
    resolve_reference_selector_mode,
    resolve_track_query_rope_contract,
    validate_reference_context_mask,
    validate_track_query_rope_dimensions,
)
from src.utils.models.components.rope import RotaryFrequencyComputer


def test_compressed_coordinates_keep_width_order_and_reference_identity() -> None:
    reference = torch.tensor([2], dtype=torch.int64)
    compressed = build_compressed_track_query_spatial_coordinates(
        reference,
        num_frames=1,
        num_views=3,
        num_queries=2,
    )

    assert compressed.tolist() == [
        [[0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 2, 1], [0, 3, 0]]
    ]


def test_coordinates_drive_expected_round_robin_selector_frequencies() -> None:
    reference = torch.tensor([0, 1], dtype=torch.int64)
    positions = build_compressed_track_query_spatial_coordinates(
        reference,
        num_frames=1,
        num_views=2,
        num_queries=1,
    )
    computer = RotaryFrequencyComputer(dim=6, base=10000.0, n_axes=3)
    frequencies = computer(positions)

    assert computer.axis_indices.tolist() == [0, 1, 2]
    assert frequencies.shape == (2, 3, 1, 3)
    # Camera 0 is reference only in batch 0. The third rotary pair therefore
    # has angle 0 in batch 0 and a nonzero angle in batch 1.
    assert frequencies[0, 1, 0, 2] == torch.polar(torch.tensor(1.0), torch.tensor(0.0))
    assert frequencies[1, 1, 0, 2] != frequencies[0, 1, 0, 2]


def test_v2_requires_all_axis_coverage_while_v1_dim4_remains_valid() -> None:
    validate_track_query_rope_dimensions(
        contract=ROLE_ROPE_CONTRACT,
        rope_dim=4,
        head_dim=8,
    )
    validate_track_query_rope_dimensions(
        contract=REFERENCE_SELECTOR_ROPE_CONTRACT,
        rope_dim=6,
        head_dim=8,
    )
    with pytest.raises(TrackQueryRopeDimensionError, match="rope_dim >= 6"):
        validate_track_query_rope_dimensions(
            contract=REFERENCE_SELECTOR_ROPE_CONTRACT,
            rope_dim=4,
            head_dim=8,
        )
    with pytest.raises(TrackQueryRopeDimensionError, match="even"):
        validate_track_query_rope_dimensions(
            contract=REFERENCE_SELECTOR_ROPE_CONTRACT,
            rope_dim=7,
            head_dim=8,
        )
    with pytest.raises(TrackQueryRopeDimensionError, match="head_dim"):
        validate_track_query_rope_dimensions(
            contract=REFERENCE_SELECTOR_ROPE_CONTRACT,
            rope_dim=10,
            head_dim=8,
        )


def test_contract_and_selector_resolvers_are_exact_and_non_inferred() -> None:
    assert (
        resolve_track_query_rope_contract("time_camera_reference_selector_v1")
        is REFERENCE_SELECTOR_ROPE_CONTRACT
    )
    assert (
        resolve_reference_selector_mode("reference") is ReferenceSelectorMode.REFERENCE
    )
    with pytest.raises(TrackQueryReferenceModelError, match="Unknown"):
        resolve_track_query_rope_contract("v2")
    with pytest.raises(TrackQueryReferenceModelError, match="Unknown"):
        resolve_reference_selector_mode("role_rope_enabled")


def test_context_invariant_is_visibility_independent_and_skips_unsupervised_padding() -> (
    None
):
    reference = torch.tensor([1], dtype=torch.int64)
    context_valid = torch.tensor(
        [[[True, False, True], [True, False, True]]],
        dtype=torch.bool,
    )
    # The fully padded middle frame is not supervised when no explicit mask is
    # supplied; detection visibility is intentionally not an input.
    validate_reference_context_mask(reference, context_valid)

    supervised = torch.tensor([[True, False, True]], dtype=torch.bool)
    validate_reference_context_mask(
        reference,
        context_valid,
        supervised_time_mask=supervised,
    )


def test_context_invariant_rejects_reference_padding_at_supervised_time() -> None:
    reference = torch.tensor([1], dtype=torch.int64)
    context_valid = torch.tensor(
        [[[True, True], [True, False]]],
        dtype=torch.bool,
    )
    with pytest.raises(ReferenceContextMaskError, match="\(batch,time\)"):
        validate_reference_context_mask(reference, context_valid)


@pytest.mark.parametrize(
    "reference",
    [
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([-1], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
    ],
)
def test_coordinate_builder_rejects_dtype_padding_and_range(
    reference: torch.Tensor,
) -> None:
    with pytest.raises(TrackQueryReferenceModelError):
        build_compressed_track_query_spatial_coordinates(
            reference,
            num_frames=1,
            num_views=2,
            num_queries=1,
        )
