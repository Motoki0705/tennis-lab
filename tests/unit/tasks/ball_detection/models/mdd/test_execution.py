from dataclasses import asdict, replace

import pytest
import torch

from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig
from src.tasks.ball_detection.models.mdd.cnn_encoder import DeepMDDEncoder
from tests.unit.tasks.ball_detection.models.mdd.test_cnn import small_config


@pytest.mark.parametrize("variant", ["residual", "convnext_v2", "fasternet"])
@pytest.mark.parametrize("mixing", ["dense3d", "factorized"])
@pytest.mark.parametrize("chunk_size", [1, 3, 8, 20])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_chunk_halos_preserve_every_feature_and_input_parameter_gradients(variant: str, mixing: str, chunk_size: int, checkpointing: bool) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(43)
    config = MDDHeatmapConfig(**dict(asdict(small_config()), input_channels=5, encoder_variant=variant, temporal_mixing=mixing), frames=9, frame_chunk_size=None)
    full = DeepMDDEncoder(config)
    chunked = DeepMDDEncoder(replace(config, frame_chunk_size=chunk_size, activation_checkpointing=checkpointing))
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


def test_checkpointing_recomputes_spatial_stages_in_reverse_order() -> None:
    torch.set_num_threads(1)
    # Exercise the legacy configuration, which has no frame splitting.
    encoder = DeepMDDEncoder(replace(small_config(), activation_checkpointing=True))
    recomputed: list[int] = []
    handles = [stage.register_forward_pre_hook(lambda _stage, _inputs, index=index: recomputed.append(index))
               for index, stage in enumerate(encoder.stages)]
    try:
        features = encoder(torch.randn(1, 2, 9, 33, 49, requires_grad=True))
        loss = sum(level.square().mean() for level in features)
        recomputed.clear()
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    # A whole-encoder checkpoint instead rebuilds all stages in forward order.
    assert recomputed == list(reversed(range(len(encoder.stages))))


@pytest.mark.parametrize("variant", ["residual", "convnext_v2", "fasternet"])
@pytest.mark.parametrize("mixing", ["dense3d", "factorized"])
@pytest.mark.parametrize("chunk_size", [None, 3])
def test_encoder_checkpointing_reduces_forward_saved_tensor_bytes(variant: str, mixing: str, chunk_size: int | None) -> None:
    torch.set_num_threads(1)
    config = MDDHeatmapConfig(**dict(asdict(small_config()), input_channels=5,
        encoder_variant=variant, temporal_mixing=mixing), frames=9, frame_chunk_size=chunk_size)
    x = torch.randn(1, 5, 9, 33, 49, requires_grad=True)
    saved_bytes: list[int] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_bytes[-1] += tensor.numel() * tensor.element_size()
        return tensor

    for enabled in (False, True):
        encoder = DeepMDDEncoder(replace(config, activation_checkpointing=enabled))
        saved_bytes.append(0)
        # Count logical saved bytes during forward, not allocator peak memory.
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            encoder(x)
    assert 0 < saved_bytes[1] < saved_bytes[0], saved_bytes
