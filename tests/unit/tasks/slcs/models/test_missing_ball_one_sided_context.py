"""One-sided observed context, masking, and initialization contracts."""

import math

import pytest
import torch

from src.tasks.slcs.models.components.missing_ball_one_sided_context import (
    MissingBallOneSidedContext,
)
from tests.unit.tasks.slcs.models.test_slcs_model import _inputs, _model


@pytest.mark.parametrize("observed", [[], [2], [1, 4], list(range(6))])
def test_exact_reference_and_source_gradients(observed: list[int]) -> None:
    module = MissingBallOneSidedContext(dim=2)
    with torch.no_grad():
        module.weight.copy_(
            torch.tensor([[1.0, 2.0, 3.0, 5.0, 7.0], [2.0, 1.0, 5.0, 3.0, 11.0]])
        )
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[:, observed] = True
    valid[:, 7] = True  # A padded "observation" never anchors a gap.
    padding = torch.tensor([[False] * 6 + [True] * 2])
    source = valid & ~padding
    tokens[~source] = torch.tensor([float("nan"), float("inf")])
    tokens.requires_grad_()
    expected = torch.zeros_like(tokens)
    used: set[int] = set()
    for t in range(6):
        left = [a for a in observed if a < t]
        right = [a for a in observed if a > t]
        if t in observed or bool(left) == bool(right):
            continue
        a = max(left) if left else min(right)
        q = torch.cat(
            [
                tokens.detach()[0, a],
                torch.tensor(
                    [float(bool(left)), float(bool(right)), math.log1p(abs(t - a))]
                ),
            ]
        )
        expected[0, t] = module.weight.detach() @ q
        used.add(a)
    actual = module(tokens, valid, padding)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad[~source]) == 0
    for a in used:
        assert (tokens.grad[0, a] > 0).all()
    changed = tokens.detach().clone()
    changed[~source] = -9876
    assert torch.equal(actual, module(changed, valid, padding))
    # Appending padded time never rescales distances or changes anchors.
    extended = module(
        torch.cat([tokens.detach(), torch.full((1, 3, 2), float("nan"))], dim=1),
        torch.cat([valid, torch.ones(1, 3, dtype=torch.bool)], dim=1),
        torch.cat([padding, torch.ones(1, 3, dtype=torch.bool)], dim=1),
    )
    assert torch.equal(actual, extended[:, :8])
    assert torch.count_nonzero(extended[:, 8:]) == 0


def test_direction_flags_distance_and_time_reversal() -> None:
    module = MissingBallOneSidedContext(dim=3)
    with torch.no_grad():
        module.weight[:, 3:] = torch.eye(3)
    tokens = torch.zeros(1, 7, 3)
    valid = torch.tensor([[False, False, True, False, True, False, False]])
    padding = torch.zeros_like(valid)
    actual = module(tokens, valid, padding)
    expected = torch.tensor(
        [
            [
                [0, 1, math.log(3)],
                [0, 1, math.log(2)],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, math.log(2)],
                [1, 0, math.log(3)],
            ]
        ]
    )
    torch.testing.assert_close(actual, expected)
    reversed_result = module(tokens.flip([1]), valid.flip([1]), padding.flip([1])).flip(
        [1]
    )
    assert torch.equal(actual, reversed_result[..., [1, 0, 2]])


@pytest.mark.parametrize("dim", [0, -1])
def test_invalid_dimension(dim: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        MissingBallOneSidedContext(dim=dim)


def test_parameter_count_and_constructor_dependency() -> None:
    assert MissingBallOneSidedContext(dim=128).weight.numel() == 16768
    with pytest.raises(ValueError, match="requires missing_ball_temporal_context"):
        _model(num_shared_layers=1, missing_ball_one_sided_context=True)


def test_independent_windows_and_adapter_forward_backward() -> None:
    from src.tasks.base.model_io import ModelInputContractError, bind_model_io
    from tests.unit.tasks.slcs.model_io.test_adapter import _adapter

    module = MissingBallOneSidedContext(dim=1)
    with torch.no_grad():
        module.weight[0, 0] = 1
    tokens = torch.tensor([[[2.0], [99.0], [99.0]], [[99.0], [99.0], [7.0]]])
    valid = torch.tensor([[True, False, False], [False, False, True]])
    actual = module(tokens, valid, torch.zeros_like(valid))
    assert torch.equal(actual[..., 0], torch.tensor([[0.0, 2.0, 2.0], [7.0, 7.0, 0.0]]))

    model = _model(
        num_shared_layers=1,
        missing_ball_temporal_context=True,
        missing_ball_one_sided_context=True,
    )
    bound = bind_model_io(model, _adapter())
    inputs = _inputs()
    inputs["ball_vis"][:, :4] = False
    output = bound.run(inputs)
    output.ball_position.square().sum().backward()
    context = model.missing_ball_one_sided_context
    assert context is not None and context.weight.grad is not None
    assert context.weight.grad.abs().sum() > 0
    calls: list[object] = []
    hook = model.register_forward_pre_hook(lambda *_: calls.append(object()))
    inputs["padding_mask"] = inputs["padding_mask"].float()
    with pytest.raises(ModelInputContractError, match="padding_mask"):
        bound.run(inputs)
    hook.remove()
    assert not calls


@pytest.mark.parametrize("court", [False, True])
def test_zero_init_rng_shared_parameters_strict_state_and_backward(court: bool) -> None:
    torch.manual_seed(42)
    control = _model(
        num_shared_layers=1,
        missing_ball_temporal_context=True,
        missing_ball_court_context=court,
    ).eval()
    state = torch.get_rng_state()
    torch.manual_seed(42)
    enabled = _model(
        num_shared_layers=1,
        missing_ball_temporal_context=True,
        missing_ball_court_context=court,
        missing_ball_one_sided_context=True,
    ).eval()
    assert torch.equal(state, torch.get_rng_state())
    assert set(enabled.state_dict()) - set(control.state_dict()) == {
        "missing_ball_one_sided_context.weight"
    }
    for name, value in control.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[name])
    restored = _model(
        num_shared_layers=1,
        missing_ball_temporal_context=True,
        missing_ball_court_context=court,
        missing_ball_one_sided_context=False,
    )
    restored.load_state_dict(control.state_dict(), strict=True)
    inputs = _inputs()
    inputs["ball_vis"][:] = False
    inputs["ball_vis"][:, 3] = True
    inputs["padding_mask"][:, -1] = True
    expected, actual = control(**inputs), enabled(**inputs)
    for key in expected:
        assert torch.equal(expected[key], actual[key])
    actual["ball_position"].square().sum().backward()
    module = enabled.missing_ball_one_sided_context
    assert module is not None
    grad = module.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("rgb_only", [False, True])
def test_model_source_is_original_embedding_before_other_residuals(
    rgb_only: bool,
) -> None:
    model = _model(
        num_shared_layers=1,
        missing_ball_temporal_context=True,
        missing_ball_court_context=True,
        missing_ball_one_sided_context=True,
    ).eval()
    module = model.missing_ball_one_sided_context
    assert module is not None
    assert model.missing_ball_context is not None
    assert model.missing_ball_temporal_context is not None
    with torch.no_grad():
        module.weight.fill_(1)
        model.missing_ball_context.weight.fill_(2)
        model.missing_ball_temporal_context.weight.fill_(3)
    inputs = _inputs()
    inputs["ball_vis"][:] = False
    if not rgb_only:
        inputs["ball_vis"][:, [2, 4]] = True
    inputs["padding_mask"][:, -1] = True
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
    assert torch.equal(valid, inputs["ball_vis"] & ~padding)
    assert torch.count_nonzero(residual[valid | padding]) == 0
    if rgb_only:
        assert torch.count_nonzero(residual) == 0
    else:
        assert torch.count_nonzero(residual[:, 3]) == 0
        assert torch.count_nonzero(residual[:, :2]) > 0
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["ball_uv"][~valid] = 999
    actual = model(**changed)
    for key in output:
        assert torch.equal(output[key], actual[key])
    output["ball_position"].square().sum().backward()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
