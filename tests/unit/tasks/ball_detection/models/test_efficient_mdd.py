from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.models.mdd_pretrain import (
    DeepMDDQueryDetector,
    MDDDPTDetector,
)
from src.tasks.ball_detection.models.mdd_pretrain.efficient_blocks import (
    GlobalResponseNorm,
)
from src.tasks.ball_detection.models.mdd_pretrain.encoder import DeepMDDEncoder
from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import (
    pretraining_config,
    transfer_encoder,
)
from tests.unit.tasks.ball_detection.models.test_mdd_pretrain import small_config


def test_grn_blank_bf16_input_has_finite_gradients_without_promoting_activations() -> None:
    layer = GlobalResponseNorm(8)
    value = torch.zeros(2, 4, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    output = layer(value)
    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(layer.gamma.grad).all()


@pytest.mark.parametrize("variant", ["convnext_v2", "fasternet"])
def test_efficient_encoders_preserve_pyramid_gradients_and_time_receptive_field(variant: str) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = replace(small_config(), encoder_variant=variant, temporal_mixing="factorized")
    encoder = DeepMDDEncoder(cfg)
    x = torch.randn(1, 2, 12, 32, 48)
    changed = x.clone()
    changed[:, :, 6] += 3 * torch.randn_like(x[:, :, 6])
    a, b = encoder(x), encoder(changed)
    for i, radius in enumerate((0, 0, 1, 2)):
        indices = [t for t in range(12) if abs(t - 6) > radius]
        torch.testing.assert_close(a[i][:, indices], b[i][:, indices], atol=0, rtol=0)
        assert not torch.allclose(a[i][:, 6], b[i][:, 6])
    sum(f.square().mean() for f in a).backward()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
    for block in encoder.temporal:
        assert block.conv.weight.grad.abs().sum() > 0
    model = MDDDPTDetector(cfg)
    logits = model(torch.zeros(1, 32, 3, 32, 48, dtype=torch.uint8))
    assert logits.shape == (1, 32, 8, 12)
    logits.square().mean().backward()
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))


def test_historical_baseline_checkpoint_is_explicitly_mapped_and_cross_arch_transfer_rejected(tmp_path: Path) -> None:
    cfg = small_config()
    model = MDDDPTDetector(cfg)
    old = asdict(cfg)
    old.pop("encoder_variant")
    old.pop("temporal_mixing")
    saved = dict(schema="mdd_dpt_pretraining.v1", stage="heatmap_pretraining", model_config=old,
        state_dict=model.state_dict(), recipe=dict(input_contract=model.mdd.input_contract(), image_decode={}), global_step=12)
    restored = pretraining_config(saved)
    assert restored.encoder_variant == "residual" and restored.temporal_mixing == "dense3d"
    path = tmp_path / "v1.pt"
    torch.save(saved, path)
    same = DeepMDDQueryDetector(cfg)
    transfer_encoder(path, same)
    wrong = DeepMDDQueryDetector(replace(cfg, encoder_variant="convnext_v2"))
    with pytest.raises(ValueError, match="encoder_variant"):
        transfer_encoder(path, wrong)
    saved["model_config"]["encoder_variant"] = "fasternet"
    with pytest.raises(ValueError, match="historical"):
        pretraining_config(saved)
