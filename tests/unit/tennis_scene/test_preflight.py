"""Installation must report all faults without selecting substitutes or using CUDA."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

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
from src.utils.paths import PROJECT_ROOT


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
