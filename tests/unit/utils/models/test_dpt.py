import pytest
import torch

from src.utils.models.dpt import DPTDecoder


@pytest.mark.parametrize("block_style", ["wise", "residual"])
def test_native_multiscale_dpt_handles_odd_shapes_and_backpropagates_each_scale(block_style: str) -> None:
    torch.set_num_threads(2)
    channels = (4, 8, 8, 16)
    features = [torch.randn(2, c, h, w, requires_grad=True)
                for c, h, w in zip(channels, (9, 5, 3, 2), (15, 8, 4, 2), strict=True)]
    model = DPTDecoder(encoder_channels=channels, decoder_channels=8, reassemble_factors=(1.,) * 4,
                       block_style=block_style)
    output = model(features)
    assert output.shape == (2, 8, 9, 15)
    output.square().mean().backward()
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in features)
    assert all(p.grad is None for p in model.fusion_blocks[-1].skip_block.parameters())
    assert all(p.grad is not None for block in model.fusion_blocks[:-1] for p in block.skip_block.parameters())
    saved = model.state_dict()
    assert any(key.startswith("fusion_blocks.3.skip_block.") for key in saved)
    restored = DPTDecoder(encoder_channels=channels, decoder_channels=8, reassemble_factors=(1.,) * 4,
                          block_style=block_style)
    restored.load_state_dict(saved, strict=True)
    torch.testing.assert_close(restored(features), output, rtol=0, atol=0)


def test_decoder_requires_explicit_block_style() -> None:
    with pytest.raises(TypeError, match="block_style"):
        DPTDecoder(encoder_channels=(4, 8, 8, 16), decoder_channels=8,
                   reassemble_factors=(1.,) * 4)  # type: ignore[call-arg]
