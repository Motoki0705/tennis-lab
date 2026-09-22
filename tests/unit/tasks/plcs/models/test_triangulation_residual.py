"""Core PLCS geometric residual model properties."""

from dataclasses import replace
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.models.triangulation_residual import (
    GeometricResidualModel,
    reconstruct_world,
)


def _model_config():
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        config = compose(config_name="train_triangulation_residual")
    return replace(
        validate_residual_config(config).model,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )


def test_zero_heads_reconstruct_the_geometric_seed() -> None:
    model = GeometricResidualModel(_model_config()).eval()
    features = torch.randn(2, 3, 8, model.input_dim)
    valid = torch.ones((2, 3, 8), dtype=torch.bool)
    time = torch.arange(8, dtype=torch.float32)[None].expand(2, -1)
    output = model(features, valid, time)
    root = torch.randn(2, 8, 3)
    relative = torch.randn(2, 8, 17, 3)
    world, reconstructed_root, reconstructed_relative = reconstruct_world(
        output, root, relative
    )
    torch.testing.assert_close(reconstructed_root, root)
    torch.testing.assert_close(reconstructed_relative, relative)
    torch.testing.assert_close(world, root[:, :, None] + relative)


def test_camera_permutation_and_masked_padding_do_not_change_output() -> None:
    model = GeometricResidualModel(_model_config()).eval()
    for head in (model.root_head, model.relative_head):
        torch.nn.init.normal_(head[-1].weight, std=0.01)
    features = torch.randn(2, 3, 8, model.input_dim)
    valid = torch.ones((2, 3, 8), dtype=torch.bool)
    time = torch.arange(8, dtype=torch.float32)[None].expand(2, -1)
    before = model(features, valid, time)
    permuted = model(features[:, [2, 0, 1]], valid[:, [2, 0, 1]], time)
    padded = model(
        torch.cat((features, torch.randn(2, 1, 8, model.input_dim) * 100), dim=1),
        torch.cat((valid, torch.zeros(2, 1, 8, dtype=torch.bool)), dim=1),
        time,
    )
    for key in before:
        torch.testing.assert_close(before[key], permuted[key], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(before[key], padded[key], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        before["relative_residual"][:, :, [11, 12]].mean(2),
        torch.zeros(2, 8, 3),
        atol=1e-7,
        rtol=0,
    )
    sum(value.square().sum() for value in before.values()).backward()
    assert torch.isfinite(model.root_head[-1].weight.grad).all()
