"""Spatial scale, finite temporal support and true-clip-edge chunk contracts."""

from dataclasses import replace

import pytest
import torch

from src.tasks.ball_detection.models.mdd.cnn_encoder import build_mdd_encoder
from src.tasks.ball_detection.models.mdd.x3d_csn import X3DCSNEncoder
from tests.support.tasks.ball_detection.temporal import small_mhc_model


@pytest.mark.parametrize("height,width", [(17, 25), (65, 97)])
def test_irregular_spatial_features_preserve_all_frames(height: int, width: int) -> None:
    torch.set_num_threads(2)
    config = small_mhc_model(frames=5)
    encoder = build_mdd_encoder(config)
    assert isinstance(encoder, X3DCSNEncoder)
    levels = encoder(torch.randn(2, 5, 5, height, width))
    assert [tuple(level.shape) for level in levels] == [
        (2, 5, channels, (height + scale - 1) // scale, (width + scale - 1) // scale)
        for channels, scale in zip((8, 8, 8, 16), (8, 16, 32, 64), strict=True)
    ]


def test_every_feature_has_exact_full_temporal_support_without_normalization_leak() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(83)
    config = small_mhc_model(frames=25, residual_blocks=(0, 1, 2, 3, 3, 2))
    encoder = build_mdd_encoder(config).double().eval()
    assert encoder.temporal_radius == 11
    x = torch.randn(1, 5, 25, 17, 25, dtype=torch.float64, requires_grad=True)
    for level, radius in zip(encoder(x), (3, 6, 9, 11), strict=True):
        # An asymmetric scalar avoids a normalization-invariant channel sum.
        coefficients = torch.arange(1, level[0, 12].numel() + 1, dtype=torch.float64).reshape_as(level[0, 12])
        grad, = torch.autograd.grad((level[0, 12] * coefficients).sum(), x, retain_graph=True)
        active = grad.abs().sum(dim=(0, 1, 3, 4)).nonzero().flatten().tolist()
        assert active == list(range(12 - radius, 13 + radius)), (radius, active)


@pytest.mark.parametrize("chunk_size,checkpointing", [(1, False), (6, False), (5, True)])
def test_halo_eleven_matches_full_outputs_and_all_gradients_at_true_edges(
    chunk_size: int, checkpointing: bool,
) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(87)
    config = small_mhc_model(frames=27, residual_blocks=(0, 1, 2, 3, 3, 2))
    full = build_mdd_encoder(config).double()
    chunked = build_mdd_encoder(replace(config, frame_chunk_size=chunk_size, activation_checkpointing=checkpointing)).double()
    chunked.load_state_dict(full.state_dict(), strict=True)
    x = torch.randn(1, 5, 27, 17, 25, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = full(x), chunked(y)
    for first, second in zip(expected, actual, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-8, atol=1e-9)
    weights = torch.linspace(.3, 1.7, 27, dtype=torch.float64)[None, :, None, None, None]
    sum((level.square() * weights).mean() for level in expected).backward()
    sum((level.square() * weights).mean() for level in actual).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-9)
    for (name, first), (other, second) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other and first.grad is not None and second.grad is not None
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-7, atol=1e-8, msg=name)


def stage_config(*, kinds=("ib",) * 5, expansion=2.0):
    from dataclasses import asdict

    from src.tasks.ball_detection.models.mdd.stage_config import MDDStageHeatmapConfig
    from tests.support.tasks.ball_detection.temporal import small_model

    return MDDStageHeatmapConfig(
        **asdict(replace(small_model("dpt"), encoder_variant="x3d_csn", temporal_mixing="depthwise3d",
                         residual_blocks=(0, 1, 2, 3, 3, 2))),
        stage_blocks=kinds, expansion_ratios=(expansion,) * 5,
    )


def test_explicit_ib_retains_legacy_state_and_exact_values() -> None:
    from dataclasses import asdict

    from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig

    torch.set_num_threads(2)
    config = stage_config()
    raw = asdict(config)
    raw.pop("stage_blocks")
    raw.pop("expansion_ratios")
    legacy = build_mdd_encoder(MDDHeatmapConfig(**raw)).double()
    explicit = build_mdd_encoder(config).double()
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    data = torch.randn(1, 5, 9, 17, 25, dtype=torch.float64)
    for a, b in zip(legacy(data), explicit(data), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("kinds,expansion,radii,spatial", [
    (("ib",) * 5, 1.5, (3, 6, 9, 11), (55, 167, 391, 711)),
    (("convnext", "convnext", "ffn", "ffn", "ffn"), 2., (3, 3, 3, 3), (55, 71, 103, 167)),
    (("convnext", "convnext", "ib", "ib", "ib"), 2., (3, 6, 9, 11), (55, 167, 391, 711)),
])
def test_stage_variants_keep_downsamples_and_have_expected_temporal_support(kinds, expansion, radii, spatial) -> None:
    import math

    from src.tasks.ball_detection.models.mdd.x3d_csn import X3DCSNBlock

    torch.set_num_threads(2)
    torch.manual_seed(1859)
    config = stage_config(kinds=kinds, expansion=expansion)
    encoder = build_mdd_encoder(config).double()
    assert encoder.feature_temporal_radii == radii
    assert encoder.convolutional_spatial_fields == spatial
    widths = (*config.stem_channels, *config.mixed_channels)
    baseline = build_mdd_encoder(stage_config()).double()
    for index in range(6):
        # The first six stage modules are the fixed transitions (stem: three).
        count = 3 if index == 0 else 6
        for a, b in zip(list(encoder.stages[index])[:count], list(baseline.stages[index])[:count], strict=True):
            if isinstance(a, torch.nn.Conv3d):
                assert (a.weight.shape, a.kernel_size, a.stride, a.groups) == (b.weight.shape, b.kernel_size, b.stride, b.groups)
        for block in list(encoder.stages[index])[count:]:
            assert isinstance(block, X3DCSNBlock)
            assert block.hidden_channels == math.ceil(widths[index] * expansion)
    x = torch.randn(1, 5, 25, 17, 25, dtype=torch.float64, requires_grad=True)
    levels = encoder(x)
    for level, radius, width, scale in zip(levels, radii, widths[2:], (8, 16, 32, 64), strict=True):
        assert tuple(level.shape) == (1, 25, width, math.ceil(17/scale), math.ceil(25/scale))
        coefficient = torch.arange(1, level[0, 12].numel() + 1, dtype=torch.float64).reshape_as(level[0, 12])
        grad, = torch.autograd.grad((level[0, 12] * coefficient).sum(), x, retain_graph=True)
        active = grad.abs().sum((0, 1, 3, 4)).nonzero().flatten().tolist()
        assert active == list(range(12-radius, 13+radius))
    sum(level.square().mean() for level in levels).backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.abs().sum() > 0
               for p in encoder.parameters())


def test_stage_artifact_restores_explicit_topology_and_rejects_legacy_relabel() -> None:
    from src.tasks.ball_detection.model_io.mdd_heatmap import (
        heatmap_model_payload,
        restore_heatmap_model,
    )
    from src.tasks.ball_detection.model_io.mdd_temporal import build_temporal_model

    torch.set_num_threads(2)
    model = build_temporal_model(stage_config(kinds=("convnext", "convnext", "ffn", "ffn", "ffn"))).eval()
    payload = heatmap_model_payload(model)
    restored = restore_heatmap_model(payload).eval()
    rgb = torch.randint(256, (1, 32, 3, 17, 25), dtype=torch.uint8)
    with torch.no_grad():
        torch.testing.assert_close(model(rgb), restored(rgb), rtol=0, atol=0)
    payload["model_kind"] = "mdd_dpt_heatmap"
    with pytest.raises(ValueError, match="field"):
        restore_heatmap_model(payload)


@pytest.mark.parametrize("changes", [
    {"stage_blocks": ("ib",) * 4},
    {"stage_blocks": ("ib", "ib", "convnext_v2", "ib", "ib")},
    {"expansion_ratios": (2.,) * 4},
    {"expansion_ratios": (2., 2., float("nan"), 2., 2.)},
    {"expansion_ratios": (2., 2., True, 2., 2.)},
    {"encoder_variant": "convnext_v2", "temporal_mixing": "factorized"},
])
def test_invalid_stage_choices_are_rejected(changes) -> None:
    with pytest.raises(ValueError):
        replace(stage_config(), **changes)
