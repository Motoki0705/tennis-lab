"""Hydra composition smoke for the separate Issue #790 scaling route."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir

from src.tasks.court_detection.configuration import (
    CourtQueryModelConfig,
    CourtTrainingConfig,
    SyntheticCourtSourceConfig,
)
from src.tasks.court_detection.experiments.configuration import (
    QueryAblationConfig,
    QueryProfileConfig,
)
from src.tasks.court_detection.experiments.query_ablation import build_ablation_manifest
from src.tasks.court_detection.experiments.query_consistency import (
    SHARED_DATA_ROOT,
    SHARED_EXTERNAL_ASSET_ROOT,
    V3_DERIVED_TARGET_ROOT,
    V3_WORKSPACE_ROOT,
    QueryConsistencyAblationConfig,
    build_query_consistency_manifest,
)
from src.tasks.court_detection.experiments.query_consistency_summary import (
    QueryConsistencySummaryConfig,
)

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def test_new_roots_compose_without_changing_the_legacy_route() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        old_config = compose(config_name="run_query_ablation")
        new_config = compose(config_name="run_query_consistency_ablation")
        summary_config = compose(
            config_name="summarize_query_consistency_ablation"
        )
    old_manifest = build_ablation_manifest(QueryAblationConfig.from_config(old_config))
    new_manifest = build_query_consistency_manifest(
        QueryConsistencyAblationConfig.from_config(new_config)
    )

    assert len(cast(list[object], old_manifest["runs"])) == 10
    assert len(cast(list[object], new_manifest["runs"])) == 28
    assert old_manifest["schema"] != new_manifest["schema"]
    assert isinstance(
        QueryConsistencySummaryConfig.from_config(summary_config),
        QueryConsistencySummaryConfig,
    )


def test_every_fully_resolved_training_argv_composes_strictly() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="run_query_consistency_ablation",
            overrides=[
                "consistency_ablation.selected.input_long_side=384",
                "consistency_ablation.selected.encoder_depth=8",
                "consistency_ablation.selected.decoder_family=dpt",
                "consistency_ablation.selected.decoder_size=base",
            ],
        )
        manifest = build_query_consistency_manifest(
            QueryConsistencyAblationConfig.from_config(config)
        )
        for run in cast(list[dict[str, object]], manifest["runs"]):
            argv = cast(list[str], run["command_argv"])
            training_config = compose(config_name="train", overrides=argv[3:])
            runtime = CourtTrainingConfig.from_config(training_config)
            data_root = Path(SHARED_DATA_ROOT)
            assert runtime.shared.run.test_after_fit is True
            assert runtime.shared.training.trainer.max_epochs == 15
            assert runtime.shared.run.seed == 42
            assert isinstance(runtime.model, CourtQueryModelConfig)
            assert runtime.model.heads.dense_targets == ("kp", "seg", "line")
            assert isinstance(runtime.data.source, SyntheticCourtSourceConfig)
            assert runtime.data.source.workspace_root == data_root / V3_WORKSPACE_ROOT
            assert runtime.data.processing.derived_target_root == (
                data_root / V3_DERIVED_TARGET_ROOT
            )
            assert runtime.data.source.workspace_root.is_relative_to(data_root)
            assert runtime.data.processing.derived_target_root.is_relative_to(data_root)


def test_every_unique_capacity_profile_argv_composes_strictly() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="run_query_consistency_ablation",
            overrides=[
                "consistency_ablation.selected.input_long_side=384",
                "consistency_ablation.selected.encoder_depth=8",
                "consistency_ablation.selected.decoder_family=dpt",
                "consistency_ablation.selected.decoder_size=base",
            ],
        )
        manifest = build_query_consistency_manifest(
            QueryConsistencyAblationConfig.from_config(config)
        )
        seen: set[tuple[object, ...]] = set()
        for run in cast(list[dict[str, object]], manifest["runs"]):
            architecture = cast(dict[str, object], run["architecture"])
            identity = (
                architecture["encoder_depth"],
                architecture["decoder_family"],
                architecture["decoder_size"],
            )
            if identity in seen:
                continue
            seen.add(identity)
            argv = cast(list[str], run["profile_command_argv"])
            assert not any(
                token.startswith(("paths.data_root=", "data.source.workspace_root="))
                for token in argv
            )
            assert f"paths.external_asset_root={SHARED_EXTERNAL_ASSET_ROOT}" in argv
            profile_config = compose(
                config_name="profile_query_model",
                overrides=argv[3:],
            )
            runtime = QueryProfileConfig.from_config(profile_config)
            assert runtime.device == "cuda"
            assert runtime.model.heads.dense_targets == ("kp", "seg", "line")
