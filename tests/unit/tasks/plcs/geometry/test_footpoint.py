from __future__ import annotations

import torch

from src.tasks.plcs.geometry.footpoint import footpoint_prior
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d
from src.utils.schema.court_normalization import normalize_court_position


def observations() -> tuple[torch.Tensor, ...]:
    xy = normalize_court_position(court_keypoints_3d(STANDARD_COURT_CONFIG))[:14, :2]
    h = torch.tensor([[0.3, 0.06, 0.5], [0.02, -0.25, 0.5], [0.04, 0.12, 1.0]])

    def project(p: torch.Tensor) -> torch.Tensor:
        q = torch.cat((p, torch.ones_like(p[..., :1])), -1) @ h.T
        return q[..., :2] / q[..., 2:]

    court = project(xy).reshape(1, 1, 1, 14, 2).expand(2, 3, 4, 14, 2).clone()
    human = (
        project(torch.tensor([0.12, -0.45]))
        .reshape(1, 1, 1, 1, 2)
        .expand(2, 3, 4, 17, 2)
        .clone()
    )
    return (
        human,
        court,
        torch.ones(2, 3, 4, 17),
        torch.ones(2, 3, 4, 14),
        torch.zeros(2, 3, 4, dtype=torch.bool),
    )


def test_projective_ground_recovery_and_masked_corruption() -> None:
    human, court, hv, cv, pad = observations()
    court[..., 4, :] = 1000
    cv[..., 4] = 0
    human[..., 15, :] = -1000
    hv[..., 15] = 0
    ground, valid, anchor = footpoint_prior(human, court, hv, cv, pad)
    assert valid.all()
    torch.testing.assert_close(
        ground, torch.tensor([0.12, -0.45]).expand_as(ground), atol=2e-4, rtol=0
    )
    torch.testing.assert_close(anchor[..., 2], torch.zeros_like(anchor[..., 2]))


def test_invalid_geometry_and_padding_are_explicit() -> None:
    human, court, hv, cv, pad = observations()
    cv[:, 0] = 0
    court[:, 1] = 0.5  # degenerate, many visible but no geometric information
    pad[:, 2] = True
    ground, valid, anchor = footpoint_prior(human, court, hv, cv, pad)
    assert not valid.any()
    assert torch.isfinite(anchor).all()
    assert not ground.any() and not anchor.any()


def test_missing_ankles_excluded_from_multiview_average() -> None:
    human, court, hv, cv, pad = observations()
    hv[:, 0, :, 15:17] = 0
    ground, valid, anchor = footpoint_prior(human, court, hv, cv, pad)
    assert not valid[:, 0].any()
    assert valid[:, 1:].all()
    torch.testing.assert_close(anchor[..., :2], ground[:, 1], atol=1e-5, rtol=0)


def test_residual_model_zero_head_returns_prior_and_learns_correction() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from src.tasks.plcs.configuration import PLCSTrainingConfig
    from src.tasks.plcs.model_io.attention_masks import prepare_axial_attention_masks
    from src.tasks.plcs.model_io.factory import build_plcs_model_io

    with initialize_config_dir(
        config_dir=str(Path("src/tasks/plcs/configs").resolve()), version_base="1.3"
    ):
        config = compose(
            config_name="train",
            overrides=[
                "model=multiview_axial_foot_residual",
                "data.num_court_kp=14",
                "model.hidden_dim=32",
                "model.num_heads=4",
                "model.ffn_dim=64",
                "model.rope_dim=4",
                "model.rot_num_task_layers=1",
                "model.pose_num_task_layers=1",
                "training.compile.enabled=false",
            ],
        )
    bound = build_plcs_model_io(PLCSTrainingConfig.from_config(config))
    model = bound.model
    torch.nn.init.zeros_(model.position_head.mlp[-1].weight)
    torch.nn.init.zeros_(model.position_head.mlp[-1].bias)
    human, court, hv, cv, pad = observations()
    cm, tm = prepare_axial_attention_masks(pad)
    output = model(human, court, hv, pad, cv, cm, tm)
    _, _, anchor = footpoint_prior(human, court, hv, cv, pad)
    torch.testing.assert_close(output["position"], anchor)
    target = anchor + torch.tensor([0.04, -0.02, 0.08])
    loss = (output["position"] - target).square().mean()
    loss.backward()
    assert model.position_head.mlp[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(output["rotation"]).all()
    # Strict checkpoints must not silently restore an absolute-head architecture.
    config.model.name = "plcs_multiview_axial_split"
    original = build_plcs_model_io(PLCSTrainingConfig.from_config(config)).model
    import pytest

    with pytest.raises(RuntimeError, match="Unexpected key"):
        original.load_state_dict(model.state_dict(), strict=True)
