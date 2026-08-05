"""Strict semantic and role-aware Synthetic configuration tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.configuration import (
    SYNTHETIC_PATH_ROLE_MAP,
    non_hydra_path_resolver,
    validate_config,
)
from src.utils.configuration import (
    ConfigurationError,
    PathContractError,
    PathRole,
)
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"


def _compose(name: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        return compose(config_name=name)


@pytest.mark.parametrize(
    ("boundary", "config_name"),
    [
        ("synthetic.dataset.pipeline", "dataset/pipeline"),
        ("synthetic.dataset.blcs.feature_fit", "dataset/blcs_feature_fit"),
        (
            "synthetic.alignment.infer_ground_line_map",
            "alignment/infer_ground_line_map",
        ),
        (
            "synthetic.alignment.fit_ground_courts",
            "alignment/fit_ground_courts",
        ),
        (
            "synthetic.alignment.calibrate_court_alignment",
            "alignment/calibrate_court_alignment",
        ),
        (
            "synthetic.alignment.export_scene_provider",
            "alignment/export_scene_provider",
        ),
        (
            "synthetic.alignment.geometry_bridge",
            "alignment/geometry_bridge",
        ),
    ],
)
def test_every_hydra_boundary_prevalidates_declared_paths(
    boundary: str,
    config_name: str,
) -> None:
    runtime = validate_config(boundary, _compose(config_name))

    assert runtime.path_roles == SYNTHETIC_PATH_ROLE_MAP[boundary]
    assert set(runtime.resolved_paths) == set(runtime.path_roles)
    assert all(path.is_absolute() for path in runtime.resolved_paths.values())


def test_runtime_path_rejects_a_different_role() -> None:
    runtime = validate_config(
        "synthetic.dataset.blcs.feature_fit",
        _compose("dataset/blcs_feature_fit"),
    )

    with pytest.raises(PathContractError, match="declared as external_asset"):
        runtime.path(PathRole.DATA, "source")


def test_export_uses_narrow_assets_and_a_separate_system_executable() -> None:
    runtime = validate_config(
        "synthetic.alignment.export_scene_provider",
        _compose("alignment/export_scene_provider"),
    )

    assert runtime.resolver.roots.external_asset_root == Path(
        "/home/kamimura/projects"
    )
    asset_scope = runtime.path(PathRole.EXTERNAL_ASSET, "external_asset_scope")
    assert asset_scope == Path("/home/kamimura/projects/gaussian-splating")
    assert runtime.path(PathRole.EXTERNAL_ASSET, "dataset_root").is_relative_to(
        asset_scope
    )
    executable = runtime.system_executable("geometry_executable")
    assert executable.root == Path("/usr/bin")
    assert executable.path == Path("/usr/bin/python3.12")
    assert "geometry_executable" not in runtime.path_roles


def _temporary_executable_config(
    tmp_path: Path,
    *,
    executable: bool = True,
) -> tuple[dict[str, str], Path]:
    root = (tmp_path / "system-bin").resolve()
    root.mkdir()
    path = root / "geometry-python"
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755 if executable else 0o644)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"root": str(root), "path": path.name, "sha256": digest}, path


def test_export_accepts_a_pinned_executable_beneath_its_system_root(
    tmp_path: Path,
) -> None:
    value, path = _temporary_executable_config(tmp_path)
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(config, "geometry_executable", value, merge=False)

    runtime = validate_config("synthetic.alignment.export_scene_provider", config)

    assert runtime.system_executable("geometry_executable").verify() == path


def test_export_rejects_filesystem_root_as_external_asset_authority(
    tmp_path: Path,
) -> None:
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(config, "roots.external_asset_root", "/", merge=False)
    OmegaConf.update(config, "roots.data_root", str(tmp_path / "data"), merge=False)

    with pytest.raises(PathContractError, match="filesystem root"):
        validate_config("synthetic.alignment.export_scene_provider", config)

    assert not (tmp_path / "data").exists()


def test_export_rejects_filesystem_root_as_system_executable_authority(
    tmp_path: Path,
) -> None:
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(config, "geometry_executable.root", "/", merge=False)
    OmegaConf.update(config, "roots.data_root", str(tmp_path / "data"), merge=False)

    with pytest.raises(PathContractError, match="filesystem root"):
        validate_config("synthetic.alignment.export_scene_provider", config)

    assert not (tmp_path / "data").exists()


def test_export_rejects_an_executable_symlink_that_escapes_its_root(
    tmp_path: Path,
) -> None:
    system_root = (tmp_path / "system-bin").resolve()
    system_root.mkdir()
    outside = tmp_path / "outside-python"
    outside.write_bytes(b"#!/bin/sh\nexit 0\n")
    outside.chmod(0o755)
    (system_root / "geometry-python").symlink_to(outside)
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(
        config,
        "geometry_executable",
        {
            "root": str(system_root),
            "path": "geometry-python",
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        },
        merge=False,
    )

    with pytest.raises(PathContractError, match="outside its declared root"):
        validate_config("synthetic.alignment.export_scene_provider", config)


def test_export_rejects_a_directory_as_the_system_executable(tmp_path: Path) -> None:
    system_root = (tmp_path / "system-bin").resolve()
    system_root.mkdir()
    (system_root / "geometry-python").mkdir()
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(
        config,
        "geometry_executable",
        {
            "root": str(system_root),
            "path": "geometry-python",
            "sha256": "0" * 64,
        },
        merge=False,
    )

    with pytest.raises(PathContractError, match="not a regular file"):
        validate_config("synthetic.alignment.export_scene_provider", config)


def test_export_rejects_a_non_executable_system_file(tmp_path: Path) -> None:
    value, _ = _temporary_executable_config(tmp_path, executable=False)
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(config, "geometry_executable", value, merge=False)

    with pytest.raises(PathContractError, match="not executable"):
        validate_config("synthetic.alignment.export_scene_provider", config)


def test_export_rejects_a_system_executable_digest_mismatch(tmp_path: Path) -> None:
    value, _ = _temporary_executable_config(tmp_path)
    value["sha256"] = "0" * 64
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(config, "geometry_executable", value, merge=False)

    with pytest.raises(PathContractError, match="SHA-256 mismatch"):
        validate_config("synthetic.alignment.export_scene_provider", config)


def test_export_rejects_external_assets_outside_the_declared_root_before_output(
    tmp_path: Path,
) -> None:
    executable, _ = _temporary_executable_config(tmp_path)
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(config, "geometry_executable", executable, merge=False)
    OmegaConf.update(
        config,
        "roots.external_asset_root",
        str((tmp_path / "external").resolve()),
        merge=False,
    )
    OmegaConf.update(config, "dataset_root", "../outside", merge=False)
    OmegaConf.update(config, "roots.data_root", str(tmp_path / "data"), merge=False)

    with pytest.raises(PathContractError, match="escapes"):
        validate_config("synthetic.alignment.export_scene_provider", config)

    assert not (tmp_path / "data").exists()


def test_export_rejects_a_sibling_project_inside_the_role_root(
    tmp_path: Path,
) -> None:
    executable, _ = _temporary_executable_config(tmp_path)
    config = _compose("alignment/export_scene_provider")
    OmegaConf.update(config, "geometry_executable", executable, merge=False)
    OmegaConf.update(config, "dataset_root", "other-project/dataset", merge=False)
    OmegaConf.update(config, "roots.data_root", str(tmp_path / "data"), merge=False)

    with pytest.raises(PathContractError, match="outside the export asset scope"):
        validate_config("synthetic.alignment.export_scene_provider", config)

    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("boundary", "config_name", "key", "invalid"),
    [
        (
            "synthetic.dataset.pipeline",
            "dataset/pipeline",
            "renderer.command",
            ["renderer", "{ouptut}"],
        ),
        (
            "synthetic.dataset.blcs.feature_fit",
            "dataset/blcs_feature_fit",
            "feature_lr",
            math.nan,
        ),
        (
            "synthetic.dataset.blcs.feature_fit",
            "dataset/blcs_feature_fit",
            "target_appearance_space_sha256",
            "ABC",
        ),
        (
            "synthetic.alignment.infer_ground_line_map",
            "alignment/infer_ground_line_map",
            "ground_plane.min_camera_height",
            0.5,
        ),
        (
            "synthetic.alignment.infer_ground_line_map",
            "alignment/infer_ground_line_map",
            "line_projection.probability_threshold",
            1.5,
        ),
        (
            "synthetic.alignment.fit_ground_courts",
            "alignment/fit_ground_courts",
            "fit.proposal_fraction",
            0.5,
        ),
        (
            "synthetic.alignment.fit_ground_courts",
            "alignment/fit_ground_courts",
            "fit.max_scale_scene_per_metre",
            0.01,
        ),
        (
            "synthetic.alignment.calibrate_court_alignment",
            "alignment/calibrate_court_alignment",
            "checkpoint_best_val_dice",
            0.1,
        ),
        (
            "synthetic.alignment.calibrate_court_alignment",
            "alignment/calibrate_court_alignment",
            "local_refit.scale_relative_tolerance",
            1.0,
        ),
        (
            "synthetic.alignment.calibrate_court_alignment",
            "alignment/calibrate_court_alignment",
            "metric_thresholds.minimum_accepted_view_fraction",
            1.1,
        ),
        (
            "synthetic.alignment.export_scene_provider",
            "alignment/export_scene_provider",
            "factor",
            0,
        ),
        (
            "synthetic.alignment.export_scene_provider",
            "alignment/export_scene_provider",
            "expectations.camera_array_sha256",
            "invalid",
        ),
    ],
)
def test_semantically_invalid_values_fail_at_the_hydra_boundary(
    boundary: str,
    config_name: str,
    key: str,
    invalid: object,
) -> None:
    config = _compose(config_name)
    OmegaConf.update(config, key, invalid, merge=False)

    with pytest.raises(ConfigurationError):
        validate_config(boundary, config)


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("paths.dataset_root", "../outside"),
        ("paths.dataset_root", "/tmp/outside"),
        ("paths.dataset_root", "data/duplicated-root"),
        ("renderer.working_directory", "third_party/nht"),
    ],
)
def test_pipeline_rejects_escape_absolute_and_root_prefixed_paths(
    key: str,
    invalid: str,
) -> None:
    config = _compose("dataset/pipeline")
    OmegaConf.update(config, key, invalid, merge=False)

    with pytest.raises(PathContractError):
        validate_config("synthetic.dataset.pipeline", config)


def test_non_hydra_roots_require_all_seven_absolute_values(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    roots = {
        f"{role.value}_root": str(root / role.value) for role in PathRole
    }
    roots["project_root"] = str(root / "project")
    roots["cache_root"] = "relative-cache"

    with pytest.raises(ValueError, match="cache_root must be an absolute"):
        non_hydra_path_resolver(json.dumps(roots))
