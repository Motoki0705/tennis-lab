"""Pure PyTorch CPU reference for Kimi Delta Attention (KDA).

KDA is the recurrent linear-attention operator from Kimi Linear, not softmax
attention.  For each token, this implementation evaluates the official naive
recurrence in float32:

``S_bar = Diag(exp(log_decay_t)) S_prev``
``S_t = S_bar + beta_t k_t (v_t - S_bar.T k_t).T``
``o_t = S_t.T q_t``

The output therefore includes the current token's update and never reads a
future token.  See Kimi Linear Eq. 1 and the official FLA ``naive_recurrent_kda``
reference:

- https://arxiv.org/abs/2510.26692
- https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
"""

from __future__ import annotations

import torch

__all__ = ["kimi_delta_attention"]


def kimi_delta_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the KDA recurrence for a complete sequence or one chunk.

    Args:
        query: Query tensor with shape ``[B, T, H, K]``.
        key: Key tensor with shape ``[B, T, H, K]``.
        value: Value tensor with shape ``[B, T, H, V]``.
        log_decay: Per-channel log decay with shape ``[B, T, H, K]``.
            The function applies ``exp`` directly; callers must supply the
            desired log-space gate.
        beta: Delta update coefficient with shape ``[B, T, H]``.  The function
            uses it directly and does not apply a sigmoid.
        valid_mask: Optional boolean validity mask with shape ``[B, T]``.
            ``True`` marks a valid token.  A ``False`` token neither decays nor
            updates the state and emits an exactly zero output.
        initial_state: Optional float32 state with shape ``[B, H, K, V]``.
            It is read without mutation.  When omitted, an all-zero state is
            created on the input device.

    Returns:
        A pair ``(output, final_state)``.  ``output`` has shape
        ``[B, T, H, V]`` and the same dtype as ``value``.  ``final_state`` has
        shape ``[B, H, K, V]`` and dtype float32.  Passing that state as the
        next call's ``initial_state`` is equivalent to processing the joined
        sequence in one call.

    Raises:
        ValueError: If a shape, dtype, or device violates the documented
            contract.  All five sequence inputs must be floating-point tensors
            on one device.  Recurrence inputs are explicitly converted to
            float32, while ``initial_state`` must already be float32.
    """
    _validate_inputs(
        query,
        key,
        value,
        log_decay,
        beta,
        valid_mask=valid_mask,
        initial_state=initial_state,
    )

    batch_size, sequence_length, num_heads, key_dim = query.shape
    value_dim = value.shape[-1]
    query_f, key_f, value_f, log_decay_f, beta_f = (
        tensor.float() for tensor in (query, key, value, log_decay, beta)
    )

    if initial_state is None:
        state = torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        state = initial_state.clone()

    output_steps: list[torch.Tensor] = []
    for time_index in range(sequence_length):
        query_t = query_f[:, time_index]
        key_t = key_f[:, time_index]
        value_t = value_f[:, time_index]
        log_decay_t = log_decay_f[:, time_index]
        beta_t = beta_f[:, time_index]

        if valid_mask is not None:
            valid_vector = valid_mask[:, time_index, None]
            valid_matrix = valid_vector[..., None]
            query_t = torch.where(valid_matrix, query_t, 0.0)
            key_t = torch.where(valid_matrix, key_t, 0.0)
            value_t = torch.where(valid_matrix, value_t, 0.0)
            log_decay_t = torch.where(valid_matrix, log_decay_t, 0.0)
            beta_t = torch.where(valid_vector, beta_t, 0.0)

        decayed_state = state * log_decay_t.exp().unsqueeze(-1)
        state_read = torch.einsum("bhk,bhkv->bhv", key_t, decayed_state)
        residual = value_t - state_read
        state = decayed_state + torch.einsum(
            "bhk,bhv->bhkv",
            beta_t.unsqueeze(-1) * key_t,
            residual,
        )
        output_steps.append(torch.einsum("bhk,bhkv->bhv", query_t, state))

    if output_steps:
        output = torch.stack(output_steps, dim=1)
    else:
        output = value_f.new_empty(batch_size, 0, num_heads, value_dim)
    return output.to(dtype=value.dtype), state


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    initial_state: torch.Tensor | None,
) -> None:
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": log_decay,
        "beta": beta,
    }
    for name, tensor in tensors.items():
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")

    if query.ndim != 4:
        raise ValueError(
            f"query must have shape [B, T, H, K], got {tuple(query.shape)}"
        )
    if key.shape != query.shape:
        raise ValueError(
            "key must have the same [B, T, H, K] shape as query, got "
            f"{tuple(key.shape)} and {tuple(query.shape)}"
        )
    if log_decay.shape != query.shape:
        raise ValueError(
            "log_decay must have the same [B, T, H, K] shape as query, got "
            f"{tuple(log_decay.shape)} and {tuple(query.shape)}"
        )

    batch_size, sequence_length, num_heads, key_dim = query.shape
    if value.ndim != 4 or value.shape[:3] != (
        batch_size,
        sequence_length,
        num_heads,
    ):
        raise ValueError(
            "value must have shape [B, T, H, V] matching query's first three "
            f"dimensions, got {tuple(value.shape)}"
        )
    if beta.shape != (batch_size, sequence_length, num_heads):
        raise ValueError(f"beta must have shape [B, T, H], got {tuple(beta.shape)}")

    device = query.device
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"all sequence inputs must be on {device}; {name} is on {tensor.device}"
            )

    if valid_mask is not None:
        if valid_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"valid_mask must have shape [B, T], got {tuple(valid_mask.shape)}"
            )
        if valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must have dtype bool, got {valid_mask.dtype}")
        if valid_mask.device != device:
            raise ValueError(f"valid_mask must be on {device}, got {valid_mask.device}")

    if initial_state is not None:
        expected_state_shape = (
            batch_size,
            num_heads,
            key_dim,
            value.shape[-1],
        )
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                "initial_state must have shape [B, H, K, V], got "
                f"{tuple(initial_state.shape)}"
            )
        if initial_state.dtype is not torch.float32:
            raise ValueError(
                f"initial_state must have dtype float32, got {initial_state.dtype}"
            )
        if initial_state.device != device:
            raise ValueError(
                f"initial_state must be on {device}, got {initial_state.device}"
            )
