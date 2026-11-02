"""Installation must report all faults without selecting substitutes or using CUDA."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tennis_scene import preflight
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.installation_assets import INSTALLATION_ASSETS
from src.tennis_scene.pipeline.definition import enabled_model_assets
from src.tennis_scene.preflight import (
    AssetRequirement,
    check_assets,
    standard_requirements,
)
from src.utils.checksum import FileIntegrityError
from src.utils.configuration import PathContractError
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def installation_config(tmp_path: Path) -> DictConfig:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"):
        return compose(config_name="pipeline", overrides=[f"paths.checkpoint_root={tmp_path / 'ckpt'}",
            f"paths.data_root={tmp_path / 'data'}", f"paths.external_asset_root={tmp_path / 'third_party'}"])


def test_reports_missing_corrupt_and_extension_faults_together(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"wrong checkpoint")

    def incompatible_extension() -> Path:
        raise RuntimeError("PyTorch binary mismatch")

    result = check_assets([
        AssetRequirement("court", tmp_path / "court.ckpt", "0" * 64),
        AssetRequirement("ball", tmp_path / "ball.ckpt", "1" * 64),
        AssetRequirement("encoder", corrupt, "2" * 64),
    ], extension_validator=incompatible_extension)
    assert [(check.name, check.status) for check in result] == [
        ("court", "missing"), ("ball", "missing"), ("encoder", "mismatch"), ("dino_extension", "error")]
    assert result[2].sha256 == hashlib.sha256(corrupt.read_bytes()).hexdigest()
    assert "PyTorch binary mismatch" in result[-1].detail
    assert not (tmp_path / "court.ckpt").exists()


def test_hash_integrity_failure_does_not_hide_other_missing_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    present = tmp_path / "present.pth"
    present.touch()

    def fail(path: Path) -> str:
        raise FileIntegrityError("file changed during read", details={"path": str(path)})

    monkeypatch.setattr(preflight, "dual_sha256", fail)
    result = check_assets([AssetRequirement("unreadable", present, "0" * 64), AssetRequirement("missing", tmp_path / "absent")],
                          extension_validator=lambda: tmp_path / "extension.so")
    assert [check.status for check in result] == ["error", "missing", "ok"]
    assert "file changed" in result[0].detail


def test_shared_checkpoint_is_hashed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared = tmp_path / "encoder.pth"
    shared.write_bytes(b"encoder")
    digest = hashlib.sha256(b"encoder").hexdigest()
    calls = []

    def hash_file(path: Path) -> str:
        calls.append(path)
        return digest

    monkeypatch.setattr(preflight, "dual_sha256", hash_file)
    result = check_assets([AssetRequirement("tracking", shared, digest), AssetRequirement("association", shared, digest)],
                          extension_validator=lambda: tmp_path / "extension.so")
    assert all(check.status == "ok" for check in result)
    assert calls == [shared]


def test_standard_inventory_covers_recipe_source_sample_and_visualization(tmp_path: Path) -> None:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"):
        config = compose(config_name="pipeline", overrides=[f"paths.checkpoint_root={tmp_path / 'ckpt'}",
            f"paths.data_root={tmp_path / 'data'}", f"paths.external_asset_root={tmp_path / 'third_party'}"])
    runtime = PipelineRuntimeConfig.from_config(config)
    requirements = standard_requirements(runtime)
    paths = {item.path for item in requirements}
    assert set(enabled_model_assets(runtime).values()) <= paths
    assert {tmp_path / "ckpt" / asset.filename for asset in INSTALLATION_ASSETS} <= paths
    names = {item.name for item in requirements}
    assert {"dino_models", "dino_util", "dino_ops", "dinov3_hub", "dinov3_backbones",
            "sample_clip.json", "sample_LICENSE", "video_cam0", "video_cam1", "video_cam2"} <= names
    assert all(item.sha256 is not None for item in requirements if item.path.is_relative_to(tmp_path / "ckpt"))


@pytest.mark.parametrize("profile", ["standard", "no_body", "no_people"])
def test_diagnostic_inventory_matches_typed_runtime(installation_config: DictConfig, profile: str) -> None:
    if profile != "standard":
        installation_config.gvhmr.enabled = False
    if profile == "no_people":
        installation_config.person_observations.enabled = False
        installation_config.player_reconstruction.enabled = False
    runtime = PipelineRuntimeConfig.from_config(installation_config)
    diagnostic = preflight.configured_requirements(installation_config)
    assert {item.name: item for item in diagnostic} == {item.name: item for item in standard_requirements(runtime)}


def test_normalized_sample_spelling_keeps_byte_pins_and_companion_files(installation_config: DictConfig) -> None:
    installation_config.video_paths = ["samples/./cam0.mp4", "samples/sub/../cam1.mp4", "samples/cam2.mp4"]
    runtime = PipelineRuntimeConfig.from_config(installation_config)
    diagnostic = preflight.configured_requirements(installation_config)
    assert {item.name: item for item in diagnostic} == {item.name: item for item in standard_requirements(runtime)}
    sample = [item for item in diagnostic if item.name.startswith(("video_", "sample_"))]
    assert len(sample) == 5
    assert all(item.sha256 is not None for item in sample)


def test_unreadable_association_recipe_does_not_hide_independent_assets(
    installation_config: DictConfig, tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    installation_config.paths.project_root = str(project)
    association = project / "src/tasks/player_association/configs/association_i964_r14_lovo_a.yaml"
    association.parent.mkdir(parents=True)
    association.write_text("appearance: [invalid YAML")
    checks = check_assets(preflight.configured_requirements(installation_config),
                          extension_validator=lambda: tmp_path / "extension.so")
    selected = {check.name: check for check in checks}
    assert selected["association_configuration"].status == "error"
    assert selected["association_configuration"].path == str(association)
    assert "Association encoder inventory unavailable" in selected["association_configuration"].detail
    assert "association_encoder" not in selected
    assert selected["court"].status == "missing"
    assert selected["ball"].status == "missing"
    assert selected["video_cam0"].status == "missing"


@pytest.mark.parametrize("asset", ["court", "tracking_encoder", "body_model", "dino_models", "video_cam0"])
def test_rejected_asset_is_not_inspected_and_other_failures_remain(
    installation_config: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asset: str,
) -> None:
    entry = next(item.path for item in preflight.configured_requirements(installation_config) if item.name == asset)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must not be read")
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.symlink_to(outside)

    original_is_file = Path.is_file
    original_open = Path.open
    original_hash = preflight.dual_sha256

    def guarded_is_file(path: Path) -> bool:
        assert path not in (entry, outside), "Rejected asset was inspected"
        return original_is_file(path)

    def guarded_open(path: Path, *args, **kwargs):
        assert path not in (entry, outside), "Rejected asset was opened"
        return original_open(path, *args, **kwargs)

    def guarded_hash(path: Path) -> str:
        assert path not in (entry, outside), "Rejected asset was hashed"
        actual: str = original_hash(path)
        return actual

    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(preflight, "dual_sha256", guarded_hash)
    requirements = preflight.configured_requirements(installation_config)
    checks = check_assets(requirements, extension_validator=lambda: tmp_path / "extension.so")
    selected = {check.name: check for check in checks}
    assert selected[asset].status == "invalid_path"
    assert "escapes" in selected[asset].detail
    assert selected["ball"].status == "missing"
    assert selected["video_cam1"].status == "missing"
    assert selected["dino_extension"].status == "ok"
    if asset == "court":
        with pytest.raises(PathContractError, match="escapes"):
            PipelineRuntimeConfig.from_config(installation_config)
