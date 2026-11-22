"""CPU contracts for TensorRT builder option parsing."""

import pytest
from variant_build import BuildOptions, parse_options


def test_parse_options_defaults_to_the_baseline_build() -> None:
    assert parse_options({}) == BuildOptions()
    assert parse_options({}).tactic_sources is None


def test_parse_options_keeps_explicit_empty_tactic_sources() -> None:
    options = parse_options(
        {
            "builder_optimization_level": 0,
            "tactic_sources": [],
            "max_aux_streams": 0,
            "tiling_optimization_level": "FAST",
            "set_flags": ["DISTRIBUTIVE_INDEPENDENCE"],
        }
    )

    assert options.builder_optimization_level == 0
    assert options.tactic_sources == ()
    assert options.max_aux_streams == 0
    assert options.tiling_optimization_level == "FAST"
    assert options.set_flags == ("DISTRIBUTIVE_INDEPENDENCE",)


@pytest.mark.parametrize(
    "raw",
    [
        {"precision": "fp32"},
        {"builder_optimization_level": 6},
        {"builder_optimization_level": True},
        {"max_aux_streams": -1},
        {"tiling_optimization_level": "fast"},
        {"set_flags": ["FP16", "FP16"]},
        {"set_flags": "FP16"},
    ],
)
def test_parse_options_rejects_unapplied_or_ambiguous_requests(raw) -> None:
    with pytest.raises(ValueError):
        parse_options(raw)
