"""Explicit layout, initialization pairing and multiscale image coordinates."""
from dataclasses import asdict

import pytest
import torch

from src.tasks.ball_detection.models.mdd.multiscale_config import (
    MDDMultiScaleQueryConfig,
)
from src.tasks.ball_detection.models.mdd.multiscale_query import (
    DeepMDDMultiScaleQueryDetector,
    spatial_grid,
)
from tests.support.tasks.ball_detection.temporal import small_mhc_model


def multiscale_config(**overrides):
    values = dict(**asdict(small_mhc_model(frames=3, layers=2, dropout=0., mhc_update_scale_init=.5)),
                  spatial_layers=("M", "C"), msda_levels=(8, 16, 32, 64), msda_points=4,
                  msda_chunk_size=1, msda_im2col_step=64)
    values.update(overrides)
    return MDDMultiScaleQueryConfig(**values)


@pytest.mark.parametrize("changes", [
    {"spatial_layers": ("M",)}, {"spatial_layers": ("M", "X")}, {"msda_levels": (64, 32, 16)},
    {"msda_levels": (8, 16, 64)}, {"msda_chunk_size": 65}, {"msda_chunk_size": True},
    {"msda_points": 0}, {"activation_checkpointing": True},
])
def test_invalid_layout_or_execution_is_rejected(changes) -> None:
    with pytest.raises(ValueError):
        multiscale_config(**changes)


def test_layer_order_pairs_initial_weights_by_kind_and_leaves_queries_distinct() -> None:
    torch.manual_seed(42)
    first = DeepMDDMultiScaleQueryDetector(multiscale_config())
    torch.manual_seed(42)
    second = DeepMDDMultiScaleQueryDetector(multiscale_config(spatial_layers=("C", "M")))
    for a, b in zip(first.blocks, reversed(second.blocks), strict=True):
        assert type(a) is type(b)
        for name, weight in a.state_dict().items():
            torch.testing.assert_close(weight, b.state_dict()[name], rtol=0, atol=0)
    for name in ("ball_query", "level_embedding"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
    assert torch.unique(first.ball_query.detach(), dim=0).shape[0] == 4


def test_asymmetric_grid_uses_xy_order_and_partial_border_centers() -> None:
    grid = spatial_grid(2, 3, stride=8, image_size=(13, 19), like=torch.empty(1))
    expected = torch.tensor([[3.5/18, 3.5/12], [11.5/18, 3.5/12], [17/18, 3.5/12],
                             [3.5/18, 10/12], [11.5/18, 10/12], [17/18, 10/12]])
    torch.testing.assert_close(grid, expected, rtol=0, atol=1e-7)


def test_memory_keeps_level_order_hw_starts_and_complete_frame_validity() -> None:
    torch.set_num_threads(2)
    model = DeepMDDMultiScaleQueryDetector(multiscale_config())
    features = tuple(torch.randn(2, 3, c, h, w) for c, h, w in ((8, 5, 7), (8, 3, 4), (8, 2, 2), (16, 1, 1)))
    patches, memory = model.prepare_memory(features, image_size=(37, 53))
    assert patches.shape == (2, 3, 1, 64)
    assert memory.shapes.tolist() == [[5, 7], [3, 4], [2, 2], [1, 1]]
    assert memory.starts.tolist() == [0, 35, 47, 51]
    assert memory.values.shape == (2, 3, 52, 64)
    assert not bool(memory.mask.any()) and bool((memory.valid_ratios == 1).all())
    memory.values.square().mean().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in model.level_projections.parameters())


def test_shipped_model_configs_roundtrip_through_cli_loader() -> None:
    from pathlib import Path

    from src.tasks.ball_detection.models.mdd.stage_config import MDDStageHeatmapConfig
    from src.tasks.ball_detection.training.temporal.configuration import (
        load_model_config,
    )

    root = Path("src/tasks/ball_detection/configs/temporal_models")
    query = load_model_config(root/"mdd_multiscale_query.yaml", "query")
    assert type(query) is MDDMultiScaleQueryConfig and query.dim == 512 and query.layers == 6
    cnn = load_model_config(root/"mdd_stage_dpt_heatmap.yaml", "dpt")
    assert type(cnn) is MDDStageHeatmapConfig
    assert cnn.stage_blocks == ("ib",)*5
