"""Hydra composition coverage for the canonical scene-pipeline hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

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

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"
_NHT_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
}

pytestmark = pytest.mark.local_data


def _resource_repository_root() -> Path:
    for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        if (candidate / "data/synthetic_data_generation/raw/B00.mp4").is_file() and (
            candidate / "third_party/nht/configs/production.yaml"
        ).is_file():
            return Path(candidate)
    raise FileNotFoundError("Canonical synthetic-data local resources are unavailable.")


def _compose(*overrides: str) -> ScenePipelineConfiguration:
    resource_root = _resource_repository_root()
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        config = compose(
            config_name="run_scene_pipeline",
            overrides=[
                *overrides,
                f"roots.data_root={(resource_root / 'data').as_posix()}",
                f"roots.checkpoint_root={(resource_root / 'ckpt').as_posix()}",
                (
                    "roots.external_asset_root="
                    f"{(resource_root / 'third_party').as_posix()}"
                ),
            ],
        )
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
    selector: str,
    version: CourtDatasetSchemaVersion,
    target_modes: set[OrbitTargetMode],
) -> None:
    court = _compose(selector).court

    assert court.schema_version is version
    assert set(court.trajectory.shapes) == {OrbitShape.CIRCLE, OrbitShape.ELLIPSE}
    assert set(court.trajectory.center_kinds) == set(OrbitCenterKind)
    assert set(court.trajectory.curve_modes) == set(OrbitCurveMode)
    assert set(court.view.target_modes) == target_modes
    assert set(court.view.coverage_modes) == set(OrbitCoverageMode)
    assert court.sampling.mode is OrbitSamplingMode.UNIFORM_ARC_LENGTH
    assert set(court.sampling.stable_field_order) == set(OrbitStableField)
    assert set(court.sampling.coverage_objective) == set(OrbitCoverageObjective)


def test_default_and_compatibility_train_selectors_remain_exact_v1() -> None:
    default = _compose().court
    compatibility = _compose("dataset/court=train").court

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
        _compose(*overrides)


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
def test_unknown_court_modes_fail_at_configuration_boundary(override: str) -> None:
    with pytest.raises(SemanticConfigurationError, match="unknown value"):
        _compose(override)


def test_unknown_court_key_fails_at_configuration_boundary() -> None:
    with pytest.raises(UnknownConfigurationKeyError, match="unknown_key"):
        _compose("+dataset.court.trajectory.unknown_key=true")


def test_public_nht_commands_and_trainer_runtime_are_explicit() -> None:
    runtime = _compose()

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


def test_configured_paths_retain_their_declared_runtime_roles() -> None:
    runtime = _compose()
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
