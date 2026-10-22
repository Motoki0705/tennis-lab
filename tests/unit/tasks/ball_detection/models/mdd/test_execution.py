from dataclasses import asdict, replace

import pytest
import torch

from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig
from src.tasks.ball_detection.models.mdd.cnn_encoder import DeepMDDEncoder
from tests.unit.tasks.ball_detection.models.mdd.test_cnn import small_config


@pytest.mark.parametrize("variant", ["residual", "convnext_v2", "fasternet"])
@pytest.mark.parametrize("mixing", ["dense3d", "factorized"])
@pytest.mark.parametrize("chunk_size", [1, 3, 8, 20])
def test_chunk_halos_preserve_every_feature_and_input_parameter_gradients(variant: str, mixing: str, chunk_size: int) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(43)
    config = MDDHeatmapConfig(**dict(asdict(small_config()), input_channels=5, encoder_variant=variant, temporal_mixing=mixing), frames=9)
    full = DeepMDDEncoder(config)
    chunked = DeepMDDEncoder(replace(config, frame_chunk_size=chunk_size))
    chunked.load_state_dict(full.state_dict(), strict=True)
    x = torch.randn(2, 5, 9, 17, 25, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = full(x), chunked(y)
    for first, second in zip(expected, actual, strict=True):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6)
    sum(level.square().mean() for level in expected).backward()
    sum(level.square().mean() for level in actual).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=2e-6)
    for (name, first), (other, second) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other and first.grad is not None and second.grad is not None
        torch.testing.assert_close(first.grad, second.grad, rtol=2e-4, atol=2e-5, msg=name)
