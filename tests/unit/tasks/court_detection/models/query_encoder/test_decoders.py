"""Tests for the common variant-local dense decoder protocol."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_detection.configuration import (
    CourtQueryDPTDecoderConfig,
    CourtQueryLinearDecoderConfig,
    CourtQueryProgressiveDecoderConfig,
)
from src.tasks.court_detection.models.query_encoder.contracts import CourtEncoderTap
from src.tasks.court_detection.models.query_encoder.decoders import (
    build_query_dense_decoder,
)


def _taps() -> tuple[CourtEncoderTap, ...]:
    return tuple(
        CourtEncoderTap(
            layer_index=index,
            patch_tokens=torch.randn(2, 6, 16, requires_grad=True),
            grid_hw=(2, 3),
        )
        for index in range(4)
    )


@pytest.mark.parametrize(
    "config",
    [
        CourtQueryLinearDecoderConfig(
            family="linear",
            width=8,
            tap_indices=(3,),
        ),
        CourtQueryProgressiveDecoderConfig(
            family="progressive",
            width=10,
            tap_indices=(3,),
            stage_count=2,
        ),
        CourtQueryDPTDecoderConfig(
            family="dpt",
            width=12,
            tap_indices=(0, 1, 2, 3),
            fusion_levels=4,
            reassemble_factors=(4.0, 2.0, 1.0, 0.5),
        ),
    ],
)
def test_all_decoder_families_return_input_hw_and_backpropagate(config: object) -> None:
    assert isinstance(
        config,
        (
            CourtQueryLinearDecoderConfig,
            CourtQueryProgressiveDecoderConfig,
            CourtQueryDPTDecoderConfig,
        ),
    )
    taps = _taps()
    decoder = build_query_dense_decoder(hidden_dim=16, config=config)

    output = decoder(taps, output_hw=(17, 19))
    output.square().mean().backward()

    assert output.shape == (2, config.width, 17, 19)
    assert all(
        tap.patch_tokens.grad is not None
        for tap in taps
        if tap.layer_index in config.tap_indices
    )


def test_decoder_missing_declared_tap_fails_without_fallback() -> None:
    decoder = build_query_dense_decoder(
        hidden_dim=16,
        config=CourtQueryLinearDecoderConfig(
            family="linear",
            width=8,
            tap_indices=(3,),
        ),
    )

    with pytest.raises(ValueError, match="missing declared tap"):
        decoder(_taps()[:3], output_hw=(17, 19))
