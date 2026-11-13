"""Explicit layout, initialization pairing and multiscale image coordinates."""
from dataclasses import asdict, replace

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
    {"msda_points": 0}, {"checkpoint_scope": "unknown"},
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


@pytest.mark.parametrize("scope", ["msda", "encoder_msda"])
def test_checkpointing_preserves_all_gradients_rng_and_state_keys_on_cpu(monkeypatch, scope) -> None:
    from src.tasks.ball_detection.model_io.mdd_temporal import (
        restore_temporal_model,
        temporal_model_payload,
    )
    from src.utils.models.components.ops.msda.api import OfficialMSDeformAttn

    def test_spatial_read(self, query, reference, values, shapes, starts, padding_mask=None):
        # Declared differentiable CPU double at the CUDA boundary only. Every
        # official trainable projection participates; real interpolation is
        # independently covered by integration's native CUDA tests.
        op = self.op
        value = op.value_proj(values).mean(1, keepdim=True)
        offsets = op.sampling_offsets(query).mean(-1, keepdim=True)
        weights = op.attention_weights(query).mean(-1, keepdim=True)
        return op.output_proj(value + offsets + weights + reference.mean((-1, -2), keepdim=False)[..., None])

    monkeypatch.setattr(OfficialMSDeformAttn, "forward", test_spatial_read)
    torch.set_num_threads(2)
    config = multiscale_config(dropout=.2, msda_chunk_size=2)
    ordinary = DeepMDDMultiScaleQueryDetector(config).train()
    replay = DeepMDDMultiScaleQueryDetector(replace(config, activation_checkpointing=True, checkpoint_scope=scope)).train()
    replay.load_state_dict(ordinary.state_dict(), strict=True)
    assert ordinary.state_dict().keys() == replay.state_dict().keys()
    rgb = torch.randint(256, (1, 3, 3, 33, 49), dtype=torch.uint8)
    times = torch.tensor([[.1, .14, .22]])
    probe = torch.tensor([[[.2, -.7], [.9, .3], [-1.1, .4]]])
    records = []
    for model in (ordinary, replay):
        calls = dict(spatial=0, encoder=0, ca=0)
        handles = []
        for key, module in (("spatial", model.blocks[0].cross.op.value_proj),
                            ("encoder", model.encoder.stages[0]), ("ca", model.blocks[1])):
            def count(_module, _args, key=key, calls=calls):
                calls[key] += 1
            handles.append(module.register_forward_pre_hook(count))
        torch.manual_seed(187)
        prediction = model(rgb, times)
        (prediction * probe).sum().backward()
        for handle in handles:
            handle.remove()
        records.append((prediction.detach(), torch.get_rng_state(), calls))
    torch.testing.assert_close(records[0][0], records[1][0], rtol=0, atol=0)
    assert torch.equal(records[0][1], records[1][1])
    # Replacing checkpoint_call with a no-op must fail: numerical equivalence
    # alone would not detect a disabled memory policy.
    assert records[1][2]["spatial"] == 2 * records[0][2]["spatial"]
    assert records[1][2]["encoder"] == (1 if scope == "msda" else 2)
    assert records[1][2]["ca"] == 1
    for (name, a), (other, b) in zip(ordinary.named_parameters(), replay.named_parameters(), strict=True):
        assert name == other and a.grad is not None and b.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7, msg=name)
    # Original artifacts have no scope field. Restoration must not mutate them
    # or invent parameter/buffer keys, and must retain their disabled policy.
    saved = temporal_model_payload(ordinary)
    saved["model_config"].pop("checkpoint_scope")
    restored = restore_temporal_model(saved).eval()
    assert "checkpoint_scope" not in saved["model_config"]
    assert not restored.config.activation_checkpointing
    assert restored.state_dict().keys() == ordinary.state_dict().keys()
    ordinary.eval()
    torch.testing.assert_close(restored(rgb, times), ordinary(rgb, times), rtol=0, atol=0)
