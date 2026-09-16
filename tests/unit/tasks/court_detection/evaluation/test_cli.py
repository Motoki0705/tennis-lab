"""CLI contract tests: device policy, required model inputs, and file names."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.tasks.court_detection.configuration import (
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.scripts.benchmark_alignment import (
    DEFAULT_CONFIG,
    build_parser,
    resolve_config,
)

_CONFIG = Path(DEFAULT_CONFIG)


def _arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "config": _CONFIG,
        "repo_root": _CONFIG.parents[4],
        "output_dir": Path("/tmp/court-benchmark-cli-test"),
        "models": "ours",
        "datasets": "real_validation",
        "max_samples_per_domain": None,
        "seed": None,
        "ours_checkpoint": Path("/tmp/court-detection-epoch=17.ckpt"),
        "tcd_repo": None,
        "tcd_checkpoint": None,
        "device": "cpu",
        "force": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_a_lightning_checkpoint_filename_with_equals_signs_passes_through() -> None:
    parser = build_parser()

    parsed = parser.parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--ours-checkpoint",
            "/tmp/court-detection-epoch=17-v1=2.ckpt",
        ]
    )

    assert parsed.ours_checkpoint == Path("/tmp/court-detection-epoch=17-v1=2.ckpt")


def test_the_config_defaults_to_the_repository_benchmark_settings() -> None:
    assert DEFAULT_CONFIG.is_file()
    assert DEFAULT_CONFIG.name == "benchmark_alignment.yaml"


def test_the_benchmark_refuses_a_non_cpu_device() -> None:
    with pytest.raises(ValueError, match="CPU-only"):
        resolve_config(_arguments(device="cuda"), command="benchmark_alignment")


def test_running_ours_requires_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="--ours-checkpoint is required"):
        resolve_config(_arguments(ours_checkpoint=None), command="benchmark_alignment")


def test_running_tcd_requires_both_external_paths() -> None:
    with pytest.raises(ValueError, match="--tcd-repo and --tcd-checkpoint"):
        resolve_config(
            _arguments(models="tcd", ours_checkpoint=None, tcd_repo=Path("/tmp/TCD")),
            command="benchmark_alignment",
        )


def test_all_models_require_both_inputs() -> None:
    with pytest.raises(ValueError, match="--tcd-repo and --tcd-checkpoint"):
        resolve_config(_arguments(models="all"), command="benchmark_alignment")


def test_an_absolute_output_directory_is_required() -> None:
    with pytest.raises(ValueError, match="absolute path"):
        resolve_config(
            _arguments(output_dir=Path("relative/out")), command="benchmark_alignment"
        )


def test_sample_caps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        resolve_config(
            _arguments(max_samples_per_domain=0), command="benchmark_alignment"
        )


def test_overrides_replace_the_configured_selection() -> None:
    config = resolve_config(
        _arguments(seed=11, max_samples_per_domain=3, datasets="all", models="ours"),
        command="benchmark_alignment",
    )

    assert config.settings.selection.seed == 11
    assert config.settings.selection.max_samples_per_domain == 3
    assert config.domains == ("real_validation", "synthetic_test")
    assert config.models == ("ours",)


def test_the_real_domain_is_labelled_as_validation_not_test() -> None:
    config = resolve_config(_arguments(), command="benchmark_alignment")
    settings = config.settings.domains["real_validation"]

    assert settings.display_name == "real_validation"
    assert settings.split == "val"


def test_the_real_domain_keeps_the_production_split_mapping() -> None:
    """The benchmark narrows the read, not the source contract."""
    config = resolve_config(_arguments(), command="benchmark_alignment")
    source = config.settings.domains["real_validation"].source

    assert isinstance(source, TennisCourtDetectorSourceConfig)
    assert dict(source.split_mapping) == {
        "train": "train",
        "val": "val",
        "test": None,
    }


def test_a_baseline_only_run_resolves_and_reviews_its_own_model() -> None:
    from src.tasks.court_detection.scripts.benchmark_alignment import (
        _primary_review_model,
    )

    config = resolve_config(
        _arguments(
            models="tcd",
            ours_checkpoint=None,
            tcd_repo=Path("/tmp/TCD"),
            tcd_checkpoint=Path("/tmp/TCD/model_best.pt"),
        ),
        command="benchmark_alignment",
    )

    assert config.models == ("tcd",)
    assert _primary_review_model(config) == "tcd"


def test_an_ours_only_run_still_reviews_ours() -> None:
    from src.tasks.court_detection.scripts.benchmark_alignment import (
        _primary_review_model,
    )

    config = resolve_config(_arguments(), command="benchmark_alignment")

    assert config.models == ("ours",)
    assert _primary_review_model(config) == "ours"


def test_a_combined_run_prefers_ours_as_the_review_primary() -> None:
    from src.tasks.court_detection.scripts.benchmark_alignment import (
        _primary_review_model,
    )

    config = resolve_config(
        _arguments(
            models="all",
            tcd_repo=Path("/tmp/TCD"),
            tcd_checkpoint=Path("/tmp/TCD/model_best.pt"),
        ),
        command="benchmark_alignment",
    )

    assert config.models == ("ours", "tcd")
    assert _primary_review_model(config) == "ours"
