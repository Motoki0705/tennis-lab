"""Layout tests for previous/current token-compressor sources."""

from __future__ import annotations

import pytest
import torch

from src.utils.models.components.ops.token_compressor import (
    build_token_compressor_layout,
)


def test_layout_preserves_previous_current_mapping_and_partial_tail() -> None:
    layout = build_token_compressor_layout(5, 2, torch.device("cpu"))

    torch.testing.assert_close(
        layout.source_indices,
        torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3], [2, 3, 4, 4]]),
    )
    torch.testing.assert_close(
        layout.source_branches,
        torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]),
    )
    torch.testing.assert_close(
        layout.boundary_valid,
        torch.tensor(
            [
                [False, False, True, True],
                [True, True, True, True],
                [True, True, True, False],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("sequence_length", "ratio", "message"),
    [(0, 4, "positive int"), (4, 1, "at least 2"), (True, 4, "positive int")],
)
def test_layout_rejects_invalid_static_contract(
    sequence_length: int, ratio: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_token_compressor_layout(sequence_length, ratio, torch.device("cpu"))
