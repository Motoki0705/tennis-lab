"""Observed-only temporal anchors and legacy compatibility."""

from typing import cast

import pytest
import torch

from src.tasks.slcs.data.augmentation import (
    ObservationAugmentationConfig,
    augment_observations,
)
from src.tasks.slcs.data.types import SLCSSample
from src.tasks.slcs.data.windows import plan_windows
from src.tasks.slcs.evaluation.conditions import condition_inputs
from src.tasks.slcs.models.components.missing_ball_temporal_context import (
    MissingBallTemporalContext,
)
from tests.unit.tasks.slcs.models.test_slcs_model import _inputs, _model


def test_exact_anchors_edges_padding_and_independent_windows() -> None:
    module = MissingBallTemporalContext(dim=1)
    with torch.no_grad():
        module.weight.fill_(1)
    tokens = torch.tensor(
        [
            [
                [99.0],
                [2.0],
                [float("nan")],
                [8.0],
                [99.0],
                [99.0],
                [20.0],
                [float("inf")],
            ],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        ],
        requires_grad=True,
    )
    valid = torch.tensor(
        [[False, True, False, True, False, False, True, True], [False] * 8]
    )
    padding = torch.zeros(2, 8, dtype=torch.bool)
    padding[:, -1] = True
    actual = module(tokens, valid, padding)
    torch.testing.assert_close(
        actual[0, :, 0], torch.tensor([0.0, 0.0, 5.0, 0.0, 12.0, 16.0, 0.0, 0.0])
    )
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.equal(actual[:1], module(tokens[:1], valid[:1], padding[:1]))
    actual.sum().backward()
    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad[~(valid & ~padding)]) == 0
    assert (tokens.grad[0, [1, 3, 6]] > 0).all()
    assert module.weight.grad is not None and module.weight.grad.item() == 33
    changed = tokens.detach().clone()
    changed[~(valid & ~padding)] = -123456
    assert torch.equal(actual, module(changed, valid, padding))


@pytest.mark.parametrize("observed", [[], [2], list(range(8))])
def test_no_bracket_means_zero(observed: list[int]) -> None:
    module = MissingBallTemporalContext(dim=3)
    with torch.no_grad():
        module.weight.fill_(1)
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[:, observed] = True
    assert (
        torch.count_nonzero(
            module(torch.randn(1, 8, 3), valid, torch.zeros_like(valid))
        )
        == 0
    )


@pytest.mark.parametrize("dim", [0, -1])
def test_invalid_projection_dimension_fails_at_construction(dim: int) -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        MissingBallTemporalContext(dim=dim)


@pytest.mark.parametrize("key", ["ball_vis", "padding_mask"])
@pytest.mark.parametrize("problem", ["shape", "dtype"])
def test_temporal_masks_are_validated_before_model_forward(key: str, problem: str) -> None:
    from src.tasks.base.model_io import ModelInputContractError, bind_model_io
    from tests.unit.tasks.slcs.model_io.test_adapter import _adapter

    model = _model(num_shared_layers=1, missing_ball_temporal_context=True)
    calls: list[object] = []
    model.register_forward_pre_hook(lambda *_: calls.append(object()))
    inputs = _inputs()
    inputs[key] = inputs[key][:, :1] if problem == "shape" else inputs[key].float()
    with pytest.raises(ModelInputContractError, match=key):
        bind_model_io(model, _adapter()).run(inputs)
    assert not calls


def test_empty_window_is_validated_before_model_forward() -> None:
    from src.tasks.base.model_io import ModelInputContractError, bind_model_io
    from tests.unit.tasks.slcs.model_io.test_adapter import _adapter, _batch

    model = _model(num_shared_layers=1, missing_ball_temporal_context=True)
    calls: list[object] = []
    model.register_forward_pre_hook(lambda *_: calls.append(object()))
    with pytest.raises(ModelInputContractError, match="0<T"):
        bind_model_io(model, _adapter()).run(_batch(frames=0))
    assert not calls


@pytest.mark.parametrize("court", [False, True])
def test_zero_init_preserves_rng_shared_state_and_forward(court: bool) -> None:
    torch.manual_seed(42)
    legacy = _model(num_shared_layers=1, missing_ball_court_context=court).eval()
    state = torch.get_rng_state()
    torch.manual_seed(42)
    enabled = _model(
        num_shared_layers=1,
        missing_ball_court_context=court,
        missing_ball_temporal_context=True,
    ).eval()
    assert torch.equal(state, torch.get_rng_state())
    assert set(enabled.state_dict()) - set(legacy.state_dict()) == {
        "missing_ball_temporal_context.weight"
    }
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[name])
    restored = _model(num_shared_layers=1, missing_ball_court_context=court)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    inputs = _inputs()
    inputs["ball_vis"][:, 2:5] = False
    inputs["padding_mask"][:, -1] = True
    expected, actual = legacy(**inputs), enabled(**inputs)
    for key in expected:
        assert torch.equal(expected[key], actual[key])
    actual["ball_position"].square().sum().backward()
    assert enabled.missing_ball_temporal_context is not None
    grad = enabled.missing_ball_temporal_context.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        enabled.missing_ball_temporal_context.weight.copy_(torch.eye(32))
        learned = enabled(**inputs)
    assert not torch.equal(
        expected["ball_position"][:, 2:5], learned["ball_position"][:, 2:5]
    )


def test_production_windows_are_uniform_contiguous_and_local() -> None:
    for length in (3, 19):
        for window in plan_windows(length, window_size=8, stride=5):
            indices = torch.from_numpy(window.frame_indices()[: window.length])
            assert torch.equal(
                indices, torch.arange(window.start, window.start + window.length)
            )
    module = MissingBallTemporalContext(dim=1)
    with torch.no_grad():
        module.weight.fill_(1)
    # A left anchor in one sample and right anchor in another never combine.
    valid = torch.tensor([[True, False, False], [False, False, True]])
    assert (
        torch.count_nonzero(module(torch.ones(2, 3, 1), valid, torch.zeros_like(valid)))
        == 0
    )


@pytest.mark.parametrize("mode", ["augmentation", "detector_gap", "rgb_only"])
@pytest.mark.parametrize("court", [False, True])
def test_actual_forward_uses_post_mask_pre_residual_observations(
    mode: str, court: bool
) -> None:
    model = _model(
        num_shared_layers=1,
        missing_ball_court_context=court,
        missing_ball_temporal_context=True,
    )
    module = model.missing_ball_temporal_context
    assert module is not None
    with torch.no_grad():
        module.weight.copy_(torch.eye(32))
        if model.missing_ball_context is not None:
            model.missing_ball_context.weight.fill_(2)
    original = _inputs()
    original["padding_mask"][:, -1] = True
    if mode == "augmentation":
        torch.manual_seed(2)
        config = ObservationAugmentationConfig(True, 0.0, 0.0, 0.0, 1.0, 3, 0.0, 0.0)
        samples = [
            augment_observations(
                cast(SLCSSample, {k: v[i] for k, v in original.items()}), config
            )
            for i in range(2)
        ]
        inputs = {
            key: torch.stack(
                [cast(dict[str, torch.Tensor], sample)[key] for sample in samples]
            )
            for key in original
        }
    else:
        model.eval()
        inputs = condition_inputs(original, mode)
    captured: list[tuple[torch.Tensor, ...]] = []
    embedded: list[torch.Tensor] = []

    def capture(
        _module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured.append((*args, output))

    def capture_embed(
        _module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        embedded.append(output)

    hook = module.register_forward_hook(capture)
    embed_hook = model.ball_embed.register_forward_hook(capture_embed)
    output = model(**inputs)
    hook.remove()
    embed_hook.remove()
    tokens, valid, padding, residual = captured[0]
    assert torch.equal(tokens, embedded[0].reshape(2, 8, 32))
    assert torch.equal(valid, (inputs["ball_vis"] > 0) & ~inputs["padding_mask"])
    assert torch.equal(padding, inputs["padding_mask"])
    assert torch.count_nonzero(residual[valid | padding]) == 0
    if mode == "rgb_only":
        assert torch.count_nonzero(residual) == 0
    else:
        assert torch.count_nonzero(residual) > 0
    changed = {k: v.clone() for k, v in inputs.items()}
    changed["ball_uv"][~valid] = 999
    actual = model(**changed)
    for key in output:
        assert torch.equal(output[key], actual[key])
    output["ball_position"].square().sum().backward()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
