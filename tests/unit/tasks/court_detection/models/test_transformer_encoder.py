"""Shape and positional contracts for the optional court Transformer trunk."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.court_detection.models.transformer_encoder import (
    CourtTransformerEncoder,
    build_patch_positions,
)


@pytest.mark.parametrize("depth", [None, 0])
def test_none_or_zero_depth_is_a_parameter_free_identity(depth: int | None) -> None:
    encoder = CourtTransformerEncoder(dim=16, depth=depth, num_heads=4, rope_dim=4)
    features = torch.randn(2, 16, 2, 3)

    output = encoder(features)

    torch.testing.assert_close(output.spatial, features)
    assert output.pose_query is None
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 0
    with pytest.raises(ValueError, match="identity Transformer"):
        encoder(
            features,
            patch_valid_mask=torch.ones(2, 2, 3, dtype=torch.bool),
        )


def test_depth_and_non_square_grid_preserve_spatial_shape_and_return_query() -> None:
    encoder = CourtTransformerEncoder(
        dim=32,
        depth=2,
        num_heads=4,
        rope_dim=8,
        ffn_dim=64,
    )
    features = torch.randn(2, 32, 2, 3, requires_grad=True)

    output = encoder(features)
    assert output.pose_query is not None
    loss = output.spatial.square().mean() + output.pose_query.square().mean()
    loss.backward()

    assert output.spatial.shape == features.shape
    assert output.pose_query is not None
    assert output.pose_query.shape == (2, 32)
    assert encoder.pose_query.grad is not None
    assert features.grad is not None


@pytest.mark.parametrize("depth", [1, 8])
def test_depth_presets_build_the_requested_number_of_blocks(depth: int) -> None:
    encoder = CourtTransformerEncoder(
        dim=16,
        depth=depth,
        num_heads=4,
        rope_dim=4,
        ffn_dim=32,
    )

    output = encoder(torch.zeros(1, 16, 2, 2))

    assert len(encoder.blocks) == depth
    assert output.pose_query is not None


def test_config_depth_is_resolved_without_changing_token_dimension() -> None:
    encoder = CourtTransformerEncoder(
        dim=32,
        config=SimpleNamespace(
            depth=1,
            num_heads=4,
            rope_dim=8,
            ffn_dim=64,
            rope_theta=10000.0,
            dropout=0.0,
        ),
    )

    assert encoder.depth == 1
    assert encoder(torch.zeros(1, 32, 3, 2)).spatial.shape == (1, 32, 3, 2)


def test_patch_positions_use_zero_query_then_row_major_y_x_coordinates() -> None:
    positions = build_patch_positions((2, 3), device=torch.device("cpu"))

    assert positions.tolist() == [
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
    ]


def test_query_rope_frequency_is_identity_while_patch_frequency_is_not() -> None:
    encoder = CourtTransformerEncoder(
        dim=32,
        depth=1,
        num_heads=4,
        rope_dim=8,
    )
    frequencies = encoder.frequency_computer(
        build_patch_positions((2, 3), device=torch.device("cpu"))
    )

    torch.testing.assert_close(frequencies[0], torch.ones_like(frequencies[0]))
    assert not torch.equal(frequencies[2], frequencies[0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dim": 30, "num_heads": 4}, "divisible"),
        ({"dim": 32, "num_heads": 3}, "divisible"),
        ({"dim": 32, "num_heads": 4, "rope_dim": 6}, "four"),
        ({"dim": 32, "num_heads": 4, "rope_dim": 40}, "head_dim"),
        ({"dim": 32, "num_heads": 4, "depth": -1}, "depth"),
    ],
)
def test_invalid_transformer_dimensions_fail_at_construction(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CourtTransformerEncoder(**kwargs)  # type: ignore[arg-type]


def test_input_channel_shape_dtype_and_device_are_strict() -> None:
    encoder = CourtTransformerEncoder(dim=16, depth=1, num_heads=4, rope_dim=4)

    with pytest.raises(ValueError, match="shape"):
        encoder(torch.zeros(1, 16, 2))
    with pytest.raises(ValueError, match="channels"):
        encoder(torch.zeros(1, 8, 2, 2))
    with pytest.raises(TypeError, match="floating"):
        encoder(torch.zeros(1, 16, 2, 2, dtype=torch.int64))
    with pytest.raises(TypeError, match="dtype"):
        encoder(torch.zeros(1, 16, 2, 2, dtype=torch.float64))


def test_padding_token_perturbation_cannot_reach_pose_query_or_gradients() -> None:
    torch.manual_seed(7)
    encoder = CourtTransformerEncoder(
        dim=16,
        depth=2,
        num_heads=4,
        rope_dim=4,
        ffn_dim=32,
    ).eval()
    features = torch.randn(1, 16, 2, 3, requires_grad=True)
    patch_valid_mask = torch.tensor(
        [[[True, True, False], [True, True, False]]],
        dtype=torch.bool,
    )
    perturbed = features.detach().clone()
    perturbed[:, :, :, -1] += 10_000.0

    output = encoder(features, patch_valid_mask=patch_valid_mask)
    perturbed_output = encoder(
        perturbed,
        patch_valid_mask=patch_valid_mask,
    )

    torch.testing.assert_close(output.pose_query, perturbed_output.pose_query)
    torch.testing.assert_close(output.spatial, perturbed_output.spatial)
    assert output.pose_query is not None
    (output.pose_query.square().sum() + output.spatial.square().sum()).backward()

    assert features.grad is not None
    invalid = ~patch_valid_mask.unsqueeze(1).expand_as(features)
    assert torch.count_nonzero(features.grad[invalid]).item() == 0
    assert torch.count_nonzero(features.grad[~invalid]).item() > 0


def test_patch_valid_mask_is_strict_and_rejects_all_invalid_samples() -> None:
    encoder = CourtTransformerEncoder(dim=16, depth=1, num_heads=4, rope_dim=4)
    features = torch.zeros(1, 16, 2, 3)

    with pytest.raises(ValueError, match="shape"):
        encoder(
            features,
            patch_valid_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        )
    with pytest.raises(TypeError, match="torch.bool"):
        encoder(features, patch_valid_mask=torch.ones(1, 2, 3))
    with pytest.raises(ValueError, match="at least one patch"):
        encoder(
            features,
            patch_valid_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
        )
