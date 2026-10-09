import torch

from src.utils.models.dpt import DPTDecoder


def test_native_multiscale_dpt_handles_odd_shapes_and_backpropagates_each_scale() -> None:
    torch.set_num_threads(2)
    channels = (4, 8, 8, 16)
    features = [torch.randn(2, c, h, w, requires_grad=True)
                for c, h, w in zip(channels, (9, 5, 3, 2), (15, 8, 4, 2), strict=True)]
    model = DPTDecoder(encoder_channels=channels, decoder_channels=8, reassemble_factors=(1.,) * 4)
    output = model(features)
    assert output.shape == (2, 8, 9, 15)
    output.square().mean().backward()
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in features)
