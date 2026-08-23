"""Hydra composition smoke for every ordered query-ablation command."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.experiments.configuration import (
    QueryAblationConfig,
    QueryProfileConfig,
    QuerySummaryConfig,
)
from src.tasks.court_detection.experiments.query_ablation import (
    PHASE_ORDER,
    build_ablation_manifest,
    validate_ablation_manifest,
)

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def test_script_roots_compose_and_default_manifest_exposes_only_phase_one() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        ablation_cfg = compose(config_name="run_query_ablation")
        profile_cfg = compose(config_name="profile_query_model")
        summary_cfg = compose(config_name="summarize_query_ablation")
    runtime = QueryAblationConfig.from_config(ablation_cfg)
    manifest = build_ablation_manifest(runtime)

    assert isinstance(QueryProfileConfig.from_config(profile_cfg), QueryProfileConfig)
    assert isinstance(QuerySummaryConfig.from_config(summary_cfg), QuerySummaryConfig)
    assert manifest["phase_order"] == list(PHASE_ORDER)
    assert len(cast(list[object], manifest["runs"])) == 51
    assert (
        sum(
            bool(cast(dict[str, object], run)["queue_ready"])
            for run in cast(list[object], manifest["runs"])
        )
        == 12
    )


@pytest.mark.parametrize("family", ["linear", "progressive", "dpt"])
@pytest.mark.parametrize("size", ["tiny", "small", "base"])
def test_profiler_composes_every_decoder_family_size(
    family: str,
    size: str,
) -> None:
    overrides = [
        "model.preset=raw",
        f"model/decoder=query_{family}_{size}",
        f"profile.candidate.family={family}",
        f"profile.candidate.size={size}",
    ]
    if family == "dpt" and size == "tiny":
        overrides.append("model.task_encoder.tap_indices=[0,1]")
    elif family == "dpt" and size == "small":
        overrides.append("model.task_encoder.tap_indices=[0,1,2,3]")
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        profile_cfg = compose(
            config_name="profile_query_model",
            overrides=overrides,
        )

    runtime = QueryProfileConfig.from_config(profile_cfg)
    assert runtime.model.decoder.family == family
    assert runtime.candidate_size == size


def test_profiler_rejects_non_rgb_input_contract() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        profile_cfg = compose(
            config_name="profile_query_model",
            overrides=["profile.input.channels=4"],
        )

    with pytest.raises(ValueError, match="requires three RGB channels"):
        QueryProfileConfig.from_config(profile_cfg)


def test_every_resolved_manifest_command_composes_as_strict_training_config() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        manifest_cfg = compose(
            config_name="run_query_ablation",
            overrides=[
                "ablation.selected.encoder_depth=4",
                "ablation.selected.decoder_family=dpt",
                "ablation.selected.decoder_size=base",
            ],
        )
        manifest = build_ablation_manifest(
            QueryAblationConfig.from_config(manifest_cfg)
        )
        validate_ablation_manifest(manifest, require_resolved=True)
        for run in cast(list[dict[str, object]], manifest["runs"]):
            argv = cast(list[str], run["command_argv"])
            training_cfg = compose(config_name="train", overrides=argv[3:])
            runtime = CourtTrainingConfig.from_config(training_cfg)
            assert runtime.shared.run.seed in {42, 43, 44}
            assert runtime.shared.training.trainer.max_epochs == 15
