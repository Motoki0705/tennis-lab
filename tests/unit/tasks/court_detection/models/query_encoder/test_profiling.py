"""Capacity and schema tests for deterministic query-model profiling."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_detection.configuration import (
    CourtQueryDPTDecoderConfig,
)
from src.tasks.court_detection.models.query_encoder.contracts import CourtEncoderTap
from src.tasks.court_detection.models.query_encoder.decoders import (
    build_query_dense_decoder,
)
from src.tasks.court_detection.models.query_encoder.profiling import (
    DECODER_MAC_DEFINITION,
    PROFILE_SCHEMA,
    count_decoder_macs,
    count_parameters,
    validate_profile_record,
)


def _taps() -> tuple[CourtEncoderTap, ...]:
    return tuple(
        CourtEncoderTap(
            layer_index=index,
            patch_tokens=torch.zeros(1, 16, 32),
            grid_hw=(4, 4),
        )
        for index in range(4)
    )


def test_decoder_size_profiles_have_monotonic_parameters_and_macs() -> None:
    taps = _taps()
    configs = (
        CourtQueryDPTDecoderConfig(
            family="dpt", width=32, tap_indices=(0, 1), fusion_levels=2,
            reassemble_factors=(2.0, 1.0)
        ),
        CourtQueryDPTDecoderConfig(
            family="dpt", width=64, tap_indices=(0, 1, 2, 3), fusion_levels=4,
            reassemble_factors=(4.0, 2.0, 1.0, 0.5)
        ),
        CourtQueryDPTDecoderConfig(
            family="dpt", width=128, tap_indices=(0, 1, 2, 3), fusion_levels=4,
            reassemble_factors=(4.0, 2.0, 1.0, 0.5)
        ),
        CourtQueryDPTDecoderConfig(
            family="dpt", width=256, tap_indices=(0, 1, 2, 3), fusion_levels=4,
            reassemble_factors=(4.0, 2.0, 1.0, 0.5)
        ),
    )
    decoders = [
        build_query_dense_decoder(hidden_dim=32, config=config) for config in configs
    ]
    parameters = [count_parameters(decoder) for decoder in decoders]
    macs = [
        count_decoder_macs(decoder, taps, output_hw=(64, 64)) for decoder in decoders
    ]

    assert parameters == sorted(parameters)
    assert len(set(parameters)) == 4
    assert macs == sorted(macs)
    assert len(set(macs)) == 4


@pytest.mark.parametrize(
    "config",
    [
        CourtQueryDPTDecoderConfig(
            family="dpt",
            width=16,
            tap_indices=(0, 1, 2, 3),
            fusion_levels=4,
            reassemble_factors=(4.0, 2.0, 1.0, 0.5),
        ),
    ],
)
def test_mac_counter_executes_every_decoder_family(config: object) -> None:
    assert isinstance(config, CourtQueryDPTDecoderConfig)
    decoder = build_query_dense_decoder(hidden_dim=32, config=config)

    assert count_decoder_macs(decoder, _taps(), output_hw=(64, 64)) > 0


def test_cpu_diagnostic_schema_cannot_be_used_as_gpu_adoption_evidence() -> None:
    profile = {
        "schema": PROFILE_SCHEMA,
        "candidate": {"family": "dpt", "size": "tiny"},
        "evidence": {
            "kind": "cpu_diagnostic",
            "device_name": "cpu",
            "latency_is_adoption_evidence": False,
        },
        "execution_contract": {
            "model_mode": "eval",
            "autograd_enabled": False,
            "latency_statistic": "arithmetic_mean_and_population_std_ms",
            "peak_scope": "end_to_end_forward",
        },
        "input_contract": {
            "batch_size": 1,
            "channels": 3,
            "height": 256,
            "width": 256,
            "dtype": "float32",
            "device": "cpu",
        },
        "parameters": {"decoder": 10, "trainable": 20, "total": 30},
        "decoder_macs": {"count": 100, "definition": DECODER_MAC_DEFINITION},
        "latency_ms": {
            "warmup": 0,
            "repeats": 1,
            "decoder_mean": 1.0,
            "decoder_std": 0.0,
            "end_to_end_mean": 2.0,
            "end_to_end_std": 0.0,
        },
        "peak_memory": {
            "bytes": None,
            "method": "unavailable_for_cpu_diagnostic",
        },
    }

    validate_profile_record(profile, require_gpu_evidence=False)
    with pytest.raises(ValueError, match="requires GPU runtime"):
        validate_profile_record(profile, require_gpu_evidence=True)
