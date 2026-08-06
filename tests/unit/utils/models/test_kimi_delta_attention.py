"""Unit tests for :mod:`src.utils.models.kimi_delta_attention`."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from src.utils.models import kimi_delta_attention as exported_kda
from src.utils.models.kimi_delta_attention import kimi_delta_attention


def _random_inputs(
    *,
    batch_size: int = 2,
    sequence_length: int = 5,
    num_heads: int = 2,
    key_dim: int = 3,
    value_dim: int = 4,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(
        batch_size,
        sequence_length,
        num_heads,
        key_dim,
        generator=generator,
        requires_grad=requires_grad,
    )
    key = torch.randn(
        batch_size,
        sequence_length,
        num_heads,
        key_dim,
        generator=generator,
        requires_grad=requires_grad,
    )
    value = torch.randn(
        batch_size,
        sequence_length,
        num_heads,
        value_dim,
        generator=generator,
        requires_grad=requires_grad,
    )
    log_decay = (
        -torch.rand(
            batch_size,
            sequence_length,
            num_heads,
            key_dim,
            generator=generator,
        )
    ).requires_grad_(requires_grad)
    beta = torch.rand(
        batch_size,
        sequence_length,
        num_heads,
        generator=generator,
        requires_grad=requires_grad,
    )
    return query, key, value, log_decay, beta


def _independent_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.tolist()
    outputs: list[list[list[list[float]]]] = []
    batch_size, sequence_length, num_heads, key_dim = query.shape
    value_dim = value.shape[-1]

    for batch_index in range(batch_size):
        outputs.append([])
        for _ in range(sequence_length):
            outputs[batch_index].append(
                [[0.0 for _ in range(value_dim)] for _ in range(num_heads)]
            )

    for time_index in range(sequence_length):
        for batch_index in range(batch_size):
            for head_index in range(num_heads):
                decayed = [
                    [
                        state[batch_index][head_index][key_index][value_index]
                        * float(
                            log_decay[
                                batch_index, time_index, head_index, key_index
                            ].exp()
                        )
                        for value_index in range(value_dim)
                    ]
                    for key_index in range(key_dim)
                ]
                residual = [
                    float(value[batch_index, time_index, head_index, value_index])
                    - sum(
                        float(key[batch_index, time_index, head_index, key_index])
                        * decayed[key_index][value_index]
                        for key_index in range(key_dim)
                    )
                    for value_index in range(value_dim)
                ]
                for key_index in range(key_dim):
                    for value_index in range(value_dim):
                        state[batch_index][head_index][key_index][value_index] = (
                            decayed[key_index][value_index]
                            + float(beta[batch_index, time_index, head_index])
                            * float(
                                key[
                                    batch_index,
                                    time_index,
                                    head_index,
                                    key_index,
                                ]
                            )
                            * residual[value_index]
                        )
                for value_index in range(value_dim):
                    outputs[batch_index][time_index][head_index][value_index] = sum(
                        float(
                            query[
                                batch_index,
                                time_index,
                                head_index,
                                key_index,
                            ]
                        )
                        * state[batch_index][head_index][key_index][value_index]
                        for key_index in range(key_dim)
                    )

    return torch.tensor(outputs), torch.tensor(state)


def test_matches_hand_calculated_inclusive_scalar_recurrence() -> None:
    query = torch.tensor([[[[1.0]], [[2.0]]]])
    key = torch.ones_like(query)
    value = torch.tensor([[[[2.0]], [[4.0]]]])
    log_decay = torch.full_like(query, math.log(0.5))
    beta = torch.tensor([[[1.0], [0.5]]])

    output, state = kimi_delta_attention(query, key, value, log_decay, beta)

    torch.testing.assert_close(output, torch.tensor([[[[2.0]], [[5.0]]]]))
    torch.testing.assert_close(state, torch.tensor([[[[2.5]]]]))
    assert output[0, 1, 0, 0] > value.max()  # not a softmax-weighted value average


def test_matches_independent_small_loop_reference() -> None:
    inputs = _random_inputs(
        batch_size=1,
        sequence_length=3,
        num_heads=1,
        key_dim=2,
        value_dim=2,
    )
    initial_state = torch.tensor([[[[0.2, -0.3], [0.4, 0.1]]]])

    output, state = kimi_delta_attention(*inputs, initial_state=initial_state)
    expected_output, expected_state = _independent_reference(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        initial_state,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(state, expected_state)


def test_full_sequence_matches_split_calls_and_preserves_initial_state() -> None:
    inputs = _random_inputs()
    initial_state = torch.randn(2, 2, 3, 4)
    initial_snapshot = initial_state.clone()

    full_output, full_state = kimi_delta_attention(
        *inputs,
        initial_state=initial_state,
    )
    first_output, first_state = kimi_delta_attention(
        *(tensor[:, :2] for tensor in inputs),
        initial_state=initial_state,
    )
    second_output, split_state = kimi_delta_attention(
        *(tensor[:, 2:] for tensor in inputs),
        initial_state=first_state,
    )

    torch.testing.assert_close(
        torch.cat((first_output, second_output), dim=1), full_output
    )
    torch.testing.assert_close(split_state, full_state)
    torch.testing.assert_close(initial_state, initial_snapshot)
    assert full_output.shape == (2, 5, 2, 4)
    assert full_state.shape == (2, 2, 3, 4)


def test_valid_mask_skips_state_transition_and_zeroes_output() -> None:
    inputs = _random_inputs(
        batch_size=1,
        sequence_length=3,
        num_heads=1,
        key_dim=2,
        value_dim=2,
    )
    valid_mask = torch.tensor([[True, False, True]])

    output, state = kimi_delta_attention(*inputs, valid_mask=valid_mask)
    compressed_output, compressed_state = kimi_delta_attention(
        *(tensor[:, [0, 2]] for tensor in inputs)
    )

    torch.testing.assert_close(output[:, [0, 2]], compressed_output)
    torch.testing.assert_close(output[:, 1], torch.zeros_like(output[:, 1]))
    torch.testing.assert_close(state, compressed_state)


def test_output_uses_value_dtype_while_state_stays_float32() -> None:
    query, key, value, log_decay, beta = _random_inputs()

    output, state = kimi_delta_attention(
        query.bfloat16(),
        key.bfloat16(),
        value.bfloat16(),
        log_decay.bfloat16(),
        beta.bfloat16(),
    )

    assert output.dtype is torch.bfloat16
    assert state.dtype is torch.float32
    assert output.device == value.device
    assert state.device == value.device


def test_gradients_are_finite_for_all_recurrence_inputs() -> None:
    inputs = _random_inputs(requires_grad=True)

    output, state = kimi_delta_attention(*inputs)
    (output.float().square().mean() + state.square().mean()).backward()

    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_public_export_is_canonical_function() -> None:
    assert exported_kda is kimi_delta_attention


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda xs: (xs[0][..., 0], *xs[1:]), "query must have shape"),
        (
            lambda xs: (xs[0], xs[1][..., :2], *xs[2:]),
            "key must have the same",
        ),
        (
            lambda xs: (*xs[:2], xs[2][:, :, :1], *xs[3:]),
            "value must have shape",
        ),
        (
            lambda xs: (*xs[:3], xs[3][..., :2], xs[4]),
            "log_decay must have the same",
        ),
        (lambda xs: (*xs[:4], xs[4][:, :, :1]), "beta must have shape"),
        (
            lambda xs: (xs[0].to(torch.int64), *xs[1:]),
            "query must be a floating-point",
        ),
    ],
)
def test_rejects_invalid_sequence_inputs(
    mutate: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    match: str,
) -> None:
    inputs = _random_inputs()

    with pytest.raises(ValueError, match=match):
        kimi_delta_attention(*mutate(inputs))


@pytest.mark.parametrize(
    ("valid_mask", "match"),
    [
        (torch.ones(2, 5, 1, dtype=torch.bool), "shape"),
        (torch.ones(2, 5), "dtype bool"),
        (torch.ones(2, 4, dtype=torch.bool), "shape"),
    ],
)
def test_rejects_invalid_valid_mask(
    valid_mask: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        kimi_delta_attention(*_random_inputs(), valid_mask=valid_mask)


def test_rejects_invalid_initial_state_shape_and_dtype() -> None:
    inputs = _random_inputs()
    with pytest.raises(ValueError, match="shape"):
        kimi_delta_attention(*inputs, initial_state=torch.zeros(2, 2, 3, 3))
    with pytest.raises(ValueError, match="dtype float32"):
        kimi_delta_attention(
            *inputs,
            initial_state=torch.zeros(2, 2, 3, 4, dtype=torch.float64),
        )


def test_rejects_mixed_devices_before_computation() -> None:
    inputs = _random_inputs()
    meta_key = torch.empty_like(inputs[1], device="meta")

    with pytest.raises(ValueError, match="same|all sequence inputs"):
        kimi_delta_attention(inputs[0], meta_key, *inputs[2:])
