"""Four independent image queries, one residual-free span and real-time RoPE."""

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd import DeepMDDMHCQueryDetector
from src.tasks.ball_detection.models.mdd.mhc_query import MHCQueryBlock
from src.tasks.ball_detection.training.coordinates.objective import coordinate_loss
from tests.support.tasks.ball_detection.temporal import small_mhc_model


def test_distinct_query_initialization_mean_readout_and_all_branch_gradients() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(91)
    model = DeepMDDMHCQueryDetector(small_mhc_model(frames=5, dropout=0.0, mhc_update_scale_init=.5))
    assert model.ball_query.shape == (4, 64)
    assert .015 < float(model.ball_query.detach().std()) < .025
    assert abs(float(model.ball_query.detach().mean())) < .005
    assert torch.unique(model.ball_query.detach(), dim=0).shape[0] == 4
    assert not hasattr(model, "decoder") and not hasattr(model, "presence_head")
    captures: dict[str, Tensor] = {}

    def streams(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
        captures["streams"] = output.detach()

    def readout(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        captures["readout"] = args[0].detach()

    handles = [model.blocks[-1].register_forward_hook(streams), model.head.register_forward_pre_hook(readout)]
    try:
        rgb = torch.randint(256, (1, 5, 3, 65, 129), dtype=torch.uint8)
        output = model(rgb, torch.tensor([[0., .03, .07, .12, .18]]))
        assert output.shape == (1, 5, 2) and output.dtype == torch.float32
        assert bool(((output > 0) & (output < 1)).all())
        torch.testing.assert_close(captures["readout"], captures["streams"].mean(dim=-2), rtol=0, atol=0)
        coordinate_loss(output, torch.rand_like(output), torch.ones(1, 5, dtype=torch.bool)).backward()
    finally:
        for handle in handles:
            handle.remove()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    assert all(bool(torch.isfinite(grad).all()) for grad in gradients.values() if grad is not None)
    for prefix in ("encoder.stages.0.", "encoder.stages.5.", "projection.", "spatial_position.",
                   "ball_query", "blocks.0.cross.", "blocks.0.mhc.pre_head.", "blocks.0.mhc.update_scale",
                   "blocks.0.temporal.", "blocks.0.ffn.", "head."):
        assert sum(float(grad.abs().sum()) for name, grad in gradients.items() if name.startswith(prefix) and grad is not None) > 0, prefix


def test_zero_scale_blocks_first_temporal_gradients_then_learns_an_update() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(92)
    block = MHCQueryBlock(small_mhc_model(frames=5, dropout=0.0))
    queries, patches = torch.randn(1, 5, 4, 64), torch.randn(1, 5, 3, 64)
    times = torch.arange(5)[None].float() / 30
    probe = torch.randn_like(queries)
    optimizer = torch.optim.SGD(block.parameters(), lr=.1)
    (block(queries, patches, times) * probe).mean().backward()
    assert block.mhc.update_scale.grad is not None and float(block.mhc.update_scale.grad.abs()) > 0
    for module in (block.temporal, block.ffn):
        assert all(parameter.grad is not None and parameter.grad.count_nonzero() == 0 for parameter in module.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (block(queries, patches, times) * probe).mean().backward()
    for module in (block.temporal, block.ffn):
        assert sum(float(parameter.grad.abs().sum()) for parameter in module.parameters() if parameter.grad is not None) > 0


def test_single_mhc_span_writes_zero_when_both_sublayer_updates_are_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    block = MHCQueryBlock(small_mhc_model(frames=3, dropout=0.0, mhc_update_scale_init=.5)).eval()
    with torch.no_grad():
        block.temporal.output.weight.zero_()
        block.temporal.output.bias.zero_()
        block.ffn[0].w2.weight.zero_()
    calls: list[str] = []
    captured: dict[str, Tensor] = {}
    pre, post = block.mhc.pre, block.mhc.post

    def observe_pre(residual, mask):
        calls.append("pre")
        captured["residual"] = residual.clone()
        return pre(residual, mask)

    def observe_post(update, residual, state):
        calls.append("post")
        captured["update"] = update.clone()
        return post(update, residual, state)

    monkeypatch.setattr(block.mhc, "pre", observe_pre)
    monkeypatch.setattr(block.mhc, "post", observe_post)
    result = block(torch.randn(1, 3, 4, 64), torch.randn(1, 3, 5, 64), torch.arange(3)[None].float())
    assert calls == ["pre", "post"]
    assert captured["update"].count_nonzero() == 0
    torch.testing.assert_close(result, captured["residual"], rtol=0, atol=0)


def test_cross_attention_is_frame_local_and_temporal_attention_uses_real_time() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(93)
    config = replace(small_mhc_model(), dropout=0.0, mhc_update_scale_init=.5)
    block = MHCQueryBlock(config).eval()
    queries, patches = torch.randn(1, 5, 4, 64), torch.randn(1, 5, 3, 64)
    changed = patches.clone()
    changed[:, 0, :, 0] += 5
    original = patches.clone()
    times = torch.tensor([[0., .03, .07, .12, .18]])
    inputs: list[Tensor] = []
    handle = block.temporal.register_forward_pre_hook(lambda _module, args: inputs.append(args[0].detach().clone()))
    try:
        first = block(queries, patches, times)
        second = block(queries, changed, times)
        slower = block(queries, patches, times * 7)
        translated = block(queries, patches, times + 13)
    finally:
        handle.remove()
    torch.testing.assert_close(patches, original, rtol=0, atol=0)
    torch.testing.assert_close(inputs[0][:, 1:], inputs[1][:, 1:], rtol=0, atol=0)
    assert not torch.allclose(inputs[0][:, 0], inputs[1][:, 0])
    assert not torch.allclose(first[:, 1:], second[:, 1:])
    assert not torch.allclose(first, slower)
    torch.testing.assert_close(first, translated, rtol=2e-5, atol=2e-6)


def test_explicit_chunk_checkpoint_flags_preserve_training_outputs_and_gradients() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(94)
    config = small_mhc_model(frames=17, layers=2, mhc_update_scale_init=.5)
    full = DeepMDDMHCQueryDetector(config)
    chunked = DeepMDDMHCQueryDetector(replace(config, frame_chunk_size=6, activation_checkpointing=True))
    chunked.load_state_dict(full.state_dict(), strict=True)
    rgb = torch.randint(256, (1, 17, 3, 17, 25), dtype=torch.uint8)
    times = torch.arange(17)[None].float() / 30
    rng = torch.get_rng_state()
    first = full(rgb, times)
    first.square().mean().backward()
    torch.set_rng_state(rng)
    second = chunked(rgb, times)
    second.square().mean().backward()
    torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6)
    for (name, left), (other, right) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other and left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6, msg=name)
