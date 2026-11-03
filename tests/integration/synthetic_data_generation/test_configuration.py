"""Hydra composition coverage for the canonical scene-pipeline hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenterKind,
    OrbitCoverageMode,
    OrbitCoverageObjective,
    OrbitCurveMode,
    OrbitSamplingMode,
    OrbitShape,
    OrbitStableField,
    OrbitTargetMode,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.utils.configuration import (
    PathRole,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.paths import PROJECT_ROOT
from tests.support.synthetic_data_generation.configuration import compose_scene_config

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"
_NHT_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
}


def _compose(tmp_path: Path, *overrides: str) -> ScenePipelineConfiguration:
    config = compose_scene_config(tmp_path, *overrides)
    return ScenePipelineConfiguration.from_config(config)


@pytest.mark.parametrize(
    ("selector", "version", "target_modes"),
    [
        (
            "dataset/court=v1",
            CourtDatasetSchemaVersion.V1,
            set(OrbitTargetMode),
        ),
        (
            "dataset/court=v2",
            CourtDatasetSchemaVersion.V2,
            {OrbitTargetMode.COURT_CENTER},
        ),
    ],
)
def test_court_selectors_compose_and_validate_exact_typed_versions(
    tmp_path: Path,
    selector: str,
    version: CourtDatasetSchemaVersion,
    target_modes: set[OrbitTargetMode],
) -> None:
    court = _compose(tmp_path, selector).court

    assert court.schema_version is version
    assert set(court.trajectory.shapes) == {OrbitShape.CIRCLE, OrbitShape.ELLIPSE}
    assert set(court.trajectory.center_kinds) == set(OrbitCenterKind)
    assert set(court.trajectory.curve_modes) == set(OrbitCurveMode)
    assert set(court.view.target_modes) == target_modes
    assert set(court.view.coverage_modes) == set(OrbitCoverageMode)
    assert court.sampling.mode is OrbitSamplingMode.UNIFORM_ARC_LENGTH
    assert set(court.sampling.stable_field_order) == set(OrbitStableField)
    assert set(court.sampling.coverage_objective) == set(OrbitCoverageObjective)


def test_default_and_compatibility_train_selectors_remain_exact_v1(
    tmp_path: Path,
) -> None:
    default = _compose(tmp_path).court
    compatibility = _compose(tmp_path, "dataset/court=train").court

    assert default.schema_version is CourtDatasetSchemaVersion.V1
    assert compatibility.schema_version is CourtDatasetSchemaVersion.V1
    assert default == compatibility
    assert tuple(default.view.target_modes) == (
        OrbitTargetMode.COURT_CENTER,
        OrbitTargetMode.COMPLEX_CENTER,
        OrbitTargetMode.NEAR_BASELINE,
        OrbitTargetMode.FAR_BASELINE,
    )


@pytest.mark.parametrize(
    "override",
    [
        "dataset.court.schema_version=v3",
        "dataset/court=v1",
        "dataset/court=v2",
    ],
)
def test_version_or_version_specific_target_mismatch_fails_closed(
    tmp_path: Path,
    override: str,
) -> None:
    extra = {
        "dataset/court=v1": "dataset.court.view.target_modes=[court_center]",
        "dataset/court=v2": (
            "dataset.court.view.target_modes=[court_center,complex_center]"
        ),
    }.get(override)
    overrides = (override,) if extra is None else (override, extra)
    with pytest.raises(SemanticConfigurationError):
        _compose(tmp_path, *overrides)


@pytest.mark.parametrize(
    "override",
    [
        "dataset.court.trajectory.shapes=[circle,unknown_shape]",
        "dataset.court.trajectory.center_kinds=[complex,unknown_center]",
        "dataset.court.trajectory.curve_modes=[planar,unknown_curve]",
        "dataset.court.view.target_modes=[court_center,unknown_target]",
        "dataset.court.view.coverage_modes=[full,unknown_coverage]",
        "dataset.court.sampling.mode=unknown_sampling",
        "dataset.court.sampling.stable_field_order=[shape,unknown_field]",
        "dataset.court.sampling.coverage_objective=[coverage_mode,unknown_objective]",
    ],
)
def test_unknown_court_modes_fail_at_configuration_boundary(
    tmp_path: Path, override: str
) -> None:
    with pytest.raises(SemanticConfigurationError, match="unknown value"):
        _compose(tmp_path, override)


def test_unknown_court_key_fails_at_configuration_boundary(
    tmp_path: Path,
) -> None:
    with pytest.raises(UnknownConfigurationKeyError, match="unknown_key"):
        _compose(tmp_path, "+dataset.court.trajectory.unknown_key=true")


def test_public_nht_commands_and_trainer_runtime_are_explicit(
    tmp_path: Path,
) -> None:
    runtime = _compose(tmp_path)

    assert runtime.nht.reconstruct_executable == "nht-reconstruct"
    assert runtime.nht.render_executable == "nht-render"
    assert (
        runtime.nht.pipeline_config.path
        == (
            runtime.resolver.roots.external_asset_root / "nht/configs/production.yaml"
        ).resolve()
    )
    assert runtime.nht.pipeline_config.schema == "nht_pipeline_config_v1"
    assert runtime.nht.training_runtime.python == (
        runtime.resolver.roots.external_asset_root / "nht/.trainer-venv/bin/python"
    )
    assert (
        runtime.nht.training_runtime.trainer
        == (
            runtime.resolver.roots.external_asset_root
            / "nht/gsplat/examples/simple_trainer_nht.py"
        ).resolve()
    )
    assert runtime.nht.environment == _NHT_ENVIRONMENT
    assert runtime.nht.reconstruction_timeout_seconds == 86_400.0
    assert runtime.nht.render_timeout_seconds == 3_600.0
    assert not hasattr(runtime.nht, "sha256")
    assert not hasattr(runtime.nht, "commit")
    assert not hasattr(runtime.nht, "repository_root")
    assert not hasattr(runtime.nht, "reconstruction_config_path")


def test_configured_paths_retain_their_declared_runtime_roles(
    tmp_path: Path,
) -> None:
    runtime = _compose(tmp_path)
    line_model = runtime.alignment.evidence.line_model

    assert (
        runtime.resolver.validate(
            PathRole.EXTERNAL_ASSET,
            runtime.nht.pipeline_config.path,
        )
        == runtime.nht.pipeline_config.path
    )
    assert (
        runtime.resolver.validate(
            PathRole.CHECKPOINT,
            line_model.checkpoint_path,
        )
        == line_model.checkpoint_path
    )
    assert (
        line_model.checkpoint_path
        == (
            runtime.resolver.roots.checkpoint_root
            / "court_detection/multiscale_depth3/b863df1f01f0.ckpt"
        ).resolve()
    )
