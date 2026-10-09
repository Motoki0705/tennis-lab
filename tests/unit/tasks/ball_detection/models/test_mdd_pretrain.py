from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.models.mdd_pretrain import (
    DeepMDDQueryDetector,
    MDDDPTDetector,
    MDDPretrainConfig,
)
from src.tasks.ball_detection.models.mdd_pretrain.encoder import DeepMDDEncoder
from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import (
    SCHEMA,
    transfer_encoder,
)
from src.tasks.ball_detection.training.heatmap_pretraining.objective import (
    heatmap_objective,
)
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic
from src.utils.models.components.ffn_layers import SwiGLU


def small_config() -> MDDPretrainConfig:
    return MDDPretrainConfig(stem_channels=(4, 8, 8, 8), mixed_channels=(8, 16),
        residual_blocks=(0, 1, 1, 1, 1, 1), decoder_channels=8, dim=16, heads=2, layers=1, ffn_dim=64, dropout=0., rope_base=10000., activation_checkpointing=False)


def test_dpt_trains_all_spatial_stages_and_both_3d_layers() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = MDDDPTDetector(small_config())
    rgb = torch.randint(256, (1, 32, 3, 64, 96), dtype=torch.uint8)
    out = model(rgb)
    assert out.shape == (1, 32, 16, 24)
    batch = dict(uv=torch.full((1, 32, 2), .45), position_valid=torch.ones(1, 32, dtype=torch.bool),
                 heatmap_valid=torch.ones(1, 32, dtype=torch.bool))
    loss, _ = heatmap_objective(out, batch)
    loss.backward()
    for stage in model.encoder.stages:
        assert next(stage.parameters()).grad.abs().sum() > 0
    for layer in model.encoder.temporal:
        assert layer.conv.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_encoder_temporal_receptive_field_does_not_leak_through_normalization() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = DeepMDDEncoder(small_config()).eval()
    x = torch.randn(1, 2, 12, 32, 48)
    y = x.clone()
    y[:, :, 6] += torch.randn_like(y[:, :, 6]) * 4
    with torch.no_grad():
        a, b = model(x), model(y)
    for i, radius in enumerate((0, 0, 1, 2)):
        unchanged = [t for t in range(12) if abs(t - 6) > radius]
        torch.testing.assert_close(a[i][:, unchanged], b[i][:, unchanged], rtol=0, atol=0)
        assert not torch.allclose(a[i][:, 6], b[i][:, 6])


def test_uniform_clip_does_not_explode_normalization_gradients() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = MDDDPTDetector(small_config())
    logits = model(torch.zeros(1, 32, 3, 32, 32, dtype=torch.uint8))
    batch = dict(uv=torch.full((1, 32, 2), .4), position_valid=torch.ones(1, 32, dtype=torch.bool),
                 heatmap_valid=torch.ones(1, 32, dtype=torch.bool))
    loss, _ = heatmap_objective(logits, batch)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    assert float(grad) < 100
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_unknown_frames_have_zero_gradient_and_known_absence_is_supervised() -> None:
    logits = torch.zeros(1, 3, 18, 32, requires_grad=True)
    batch = dict(uv=torch.tensor([[[.3, .7], [float("nan"), float("nan")], [0., 0.]]]),
                 position_valid=torch.tensor([[True, False, False]]), heatmap_valid=torch.tensor([[True, False, True]]))
    loss, _ = heatmap_objective(logits, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[0, 2].min() > 0


def test_transfer_is_encoder_only_and_supports_frozen_then_finetuned_stages(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    source = MDDDPTDetector(small_config())
    path = tmp_path / "pretrain.pt"
    torch.save(dict(schema=SCHEMA, stage="heatmap_pretraining", model_config=asdict(source.config),
        state_dict=source.state_dict(), recipe=dict(input_contract=source.mdd.input_contract(), image_decode={}), global_step=10), path)
    target = DeepMDDQueryDetector(small_config())
    before = target.ball_query.detach().clone()
    transfer_encoder(path, target)
    assert not any(p.requires_grad for p in target.encoder.parameters())
    for name, value in source.encoder.state_dict().items():
        torch.testing.assert_close(value, target.encoder.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(before, target.ball_query, rtol=0, atol=0)
    assert isinstance(target.blocks[0].ffn[0], SwiGLU)
    rgb = torch.randint(256, (1, 32, 3, 32, 48), dtype=torch.uint8)
    target.train()
    assert not target.encoder.training
    target(rgb, torch.arange(32)[None].float()).sum().backward()
    assert target.ball_query.grad is not None
    assert all(p.grad is None for p in target.encoder.parameters())
    target.freeze_encoder(False)
    target(rgb, torch.arange(32)[None].float()).sum().backward()
    assert next(target.encoder.parameters()).grad is not None
    bad = DeepMDDQueryDetector(replace(small_config(), residual_blocks=(0, 0, 0, 0, 0, 0)))
    with pytest.raises(ValueError, match="topology mismatch"):
        transfer_encoder(path, bad)


def test_shipped_config_has_requested_depth_and_swiglu_width() -> None:
    path = Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs/model/mdd_dpt_pretrain.yaml"
    config = MDDPretrainConfig.load(path)
    model = MDDDPTDetector(config)
    conv2 = [m for m in model.encoder.modules() if isinstance(m, torch.nn.Conv2d)]
    conv3 = [m for m in model.encoder.modules() if isinstance(m, torch.nn.Conv3d)]
    assert len(conv2) == 22 and len(conv3) == 2
    assert config.ffn_dim == 704 and config.dim == 256
    with pytest.raises(ValueError, match="SwiGLU"):
        replace(config, ffn_dim=1024)


def test_quarter_grid_subpixel_coordinates_use_endpoint_convention() -> None:
    from src.utils.data.heatmaps import generate_gaussian_heatmaps
    uv = torch.tensor([[[.333, .621]]])
    target = generate_gaussian_heatmaps((180, 320), uv, .012)
    peak, _ = heatmaps_to_argmax(target)
    refined = refine_peaks_log_parabolic(target, peak)
    torch.testing.assert_close(refined, uv, atol=1e-6, rtol=0)
