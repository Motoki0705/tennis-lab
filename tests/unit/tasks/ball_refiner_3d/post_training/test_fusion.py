"""Check the frozen temporal network and the numerical fusion contracts."""

import pytest
import torch
from torch import nn

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.models.generators.regression import RegressionRefiner
from src.tasks.ball_refiner_3d.post_training.fusion import (
    FUSION_MODES,
    FusionMode,
    MotionRefiner,
    PlayerFusion,
)
from src.tasks.ball_refiner_3d.post_training.motion import PLAYER_FEATURES


@pytest.mark.parametrize("mode", FUSION_MODES)
def test_zero_residual_and_frozen_base_receive_one_token_per_frame(
    mode: FusionMode,
) -> None:
    torch.manual_seed(10)
    base = RegressionRefiner(
        ModelConfig(3, "regression", 16, 1, 2, 0.2, 2, 32, 8, 10000, "swiglu", False)
    ).eval()
    coordinates = torch.randn(2, 5, 3)
    missing = torch.zeros((2, 5), dtype=torch.bool)
    missing[0, 1] = True
    padding = torch.zeros_like(missing)
    padding[1, -1] = missing[1, -1] = True
    players = torch.randn(2, 5, 4, PLAYER_FEATURES)
    valid = torch.ones((2, 5, 4), dtype=torch.bool)
    valid[0, 2] = False

    expected = base(coordinates, missing, padding)
    model = MotionRefiner(base, mode).train()
    seen = []
    hook = base.blocks[0].register_forward_pre_hook(
        lambda _, args: seen.append(args[0].shape)
    )
    result = model(coordinates, missing, padding, players, valid)
    hook.remove()
    assert seen == [(2, 5, 16)]
    assert not base.training
    torch.testing.assert_close(result.coordinates, expected.coordinates, atol=0, rtol=0)
    torch.testing.assert_close(
        result.event_logits, expected.event_logits, atol=0, rtol=0
    )

    optimizer = torch.optim.SGD(model.fusion.parameters(), lr=0.1)
    result.event_logits.square().mean().backward()
    projection = (
        model.fusion.interaction.output
        if model.fusion.interaction is not None
        else model.fusion.encoders[0][-1]
    )
    assert projection.weight.grad.count_nonzero() > 0
    assert not model.fusion.encoders[0][0].weight.grad.count_nonzero()
    optimizer.step()
    optimizer.zero_grad()
    model(
        coordinates, missing, padding, players, valid
    ).event_logits.square().mean().backward()
    assert model.fusion.encoders[0][0].weight.grad.count_nonzero() > 0
    if model.fusion.interaction is not None:
        assert (
            model.fusion.interaction.attention.in_proj_weight.grad.count_nonzero() > 0
        )
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())


@pytest.mark.parametrize("mode", FUSION_MODES)
def test_learned_residual_excludes_missing_players(mode: FusionMode) -> None:
    torch.manual_seed(42)
    fusion = PlayerFusion(mode, 16).eval()
    projections = (
        [fusion.interaction.output]
        if fusion.interaction is not None
        else [encoder[-1] for encoder in fusion.encoders]
    )
    for projection in projections:
        nn.init.normal_(projection.weight)
        nn.init.constant_(projection.bias, 2)
    ball = torch.randn(1, 2, 16)
    players = torch.randn(1, 2, 4, PLAYER_FEATURES)
    valid = torch.tensor([[[True, False, True, False], [False] * 4]])
    actual = fusion(ball, players, valid)
    corrupted = players.clone()
    corrupted[~valid] = torch.nan
    torch.testing.assert_close(actual, fusion(ball, corrupted, valid), atol=0, rtol=0)
    torch.testing.assert_close(actual[:, 1], ball[:, 1], atol=0, rtol=0)
    assert not torch.equal(actual[:, 0], ball[:, 0])

    order = [2, 3, 0, 1]
    swapped = fusion(ball, players[:, :, order], valid[:, :, order])
    if mode == "per-player":
        assert not torch.allclose(swapped[:, 0], actual[:, 0])
    else:
        torch.testing.assert_close(swapped, actual, atol=2e-5, rtol=1e-6)


def test_per_player_encoders_duplicate_the_shared_architecture_without_tying_weights() -> (
    None
):
    a, b, c = (PlayerFusion(mode, 16) for mode in FUSION_MODES)
    assert [len(f.encoders) for f in (a, b, c)] == [1, 4, 1]
    expected = [value.shape for value in a.encoders[0].state_dict().values()]
    for encoder in (*b.encoders, *c.encoders):
        assert [value.shape for value in encoder.state_dict().values()] == expected
    assert len({p.data_ptr() for p in b.encoders.parameters()}) == 4 * len(expected)


def test_set_attention_matches_a_set_without_the_missing_players() -> None:
    torch.manual_seed(31)
    width = 16
    fusion = PlayerFusion("set-attention", width).eval()
    interaction = fusion.interaction
    assert interaction is not None
    nn.init.eye_(interaction.output.weight)
    nn.init.constant_(interaction.output.bias, 0.25)
    ball = torch.randn(2, 3, width)
    players = torch.randn(2, 3, 4, PLAYER_FEATURES)
    present = torch.tensor([True, False, True, False])
    valid = present.expand(2, 3, -1)

    with torch.inference_mode():
        encoded = fusion.encoders[0](players[:, :, present])
        reduced = ball + interaction(ball, encoded, valid[:, :, present])
        full = fusion(ball, players, valid)
    assert not torch.allclose(reduced, ball)
    torch.testing.assert_close(full, reduced, atol=1e-6, rtol=1e-6)
