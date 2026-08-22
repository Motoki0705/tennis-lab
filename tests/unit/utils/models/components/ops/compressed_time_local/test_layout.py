"""Tests for compressed sliding-window index layouts."""

from __future__ import annotations

import pytest
import torch

from src.utils.models.components.ops.compressed_time_local.layout import (
    build_compressed_sliding_window_layout,
)


def _loop_layout(
    query_len: int,
    compression_ratio: int,
    window_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_len = (query_len + compression_ratio - 1) // compression_ratio
    indices: list[list[int]] = []
    valid: list[list[bool]] = []
    for query_index in range(query_len):
        center = query_index // compression_ratio
        index_row: list[int] = []
        valid_row: list[bool] = []
        for offset in range(-window_radius, window_radius + 1):
            raw_index = center + offset
            index_row.append(min(max(raw_index, 0), key_len - 1))
            valid_row.append(0 <= raw_index < key_len)
        indices.append(index_row)
        valid.append(valid_row)
    return torch.tensor(indices), torch.tensor(valid)


@pytest.mark.parametrize(
    ("query_len", "compression_ratio", "window_radius"),
    [
        (2, 4, 1),  # T < m
        (8, 4, 0),  # exact multiple and radius zero
        (9, 4, 1),  # partial tail
        (5, 2, 5),  # window wider than Tc
    ],
)
def test_layout_matches_loop_oracle(
    query_len: int,
    compression_ratio: int,
    window_radius: int,
) -> None:
    key_len = (query_len + compression_ratio - 1) // compression_ratio

    indices, index_valid = build_compressed_sliding_window_layout(
        query_len=query_len,
        key_len=key_len,
        compression_ratio=compression_ratio,
        window_radius=window_radius,
        device=torch.device("cpu"),
    )
    expected_indices, expected_valid = _loop_layout(
        query_len, compression_ratio, window_radius
    )

    assert indices.shape == (query_len, 2 * window_radius + 1)
    assert indices.dtype == torch.long
    assert index_valid.dtype == torch.bool
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(index_valid, expected_valid)


def test_first_middle_last_windows_keep_boundary_validity_separate() -> None:
    indices, index_valid = build_compressed_sliding_window_layout(
        query_len=9,
        key_len=3,
        compression_ratio=4,
        window_radius=1,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(indices[0], torch.tensor([0, 0, 1]))
    torch.testing.assert_close(index_valid[0], torch.tensor([False, True, True]))
    torch.testing.assert_close(indices[4], torch.tensor([0, 1, 2]))
    assert index_valid[4].all()
    torch.testing.assert_close(indices[-1], torch.tensor([1, 2, 2]))
    torch.testing.assert_close(index_valid[-1], torch.tensor([True, True, False]))


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"query_len": 0}, ValueError, "query_len must be a positive int"),
        ({"key_len": 0}, ValueError, "key_len must be a positive int"),
        ({"compression_ratio": 1}, ValueError, "at least 2"),
        ({"compression_ratio": True}, ValueError, "at least 2"),
        ({"window_radius": -1}, ValueError, "non-negative int"),
        ({"window_radius": 1.0}, ValueError, "non-negative int"),
        ({"key_len": 3}, ValueError, "must equal ceil"),
    ],
)
def test_layout_rejects_invalid_contracts(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "query_len": 8,
        "key_len": 2,
        "compression_ratio": 4,
        "window_radius": 1,
        "device": torch.device("cpu"),
    }
    kwargs.update(overrides)
    with pytest.raises(error_type, match=message):
        build_compressed_sliding_window_layout(**kwargs)  # type: ignore[arg-type]


def test_layout_requires_explicit_torch_device() -> None:
    with pytest.raises(TypeError, match="device must be torch.device"):
        build_compressed_sliding_window_layout(
            query_len=4,
            key_len=2,
            compression_ratio=2,
            window_radius=1,
            device="cpu",  # type: ignore[arg-type]
        )
