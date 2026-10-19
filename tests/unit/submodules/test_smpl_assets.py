"""Acquisition provenance, checked local installation and actionable failures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.submodules import smpl_assets
from src.submodules.models import Hmr2FeatureExtractor
from src.submodules.smpl_assets import SMPL_ASSET_SOURCES, SmplAssetSource
from src.tennis_scene.visualization.rendering.smpl_court import (
    load_smpl_joint_regressor,
)


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Five synthetic files with real hashes; no licensed binaries in tests."""
    upstream = tmp_path / "official-checkout"
    sources = {}
    for name, source in SMPL_ASSET_SOURCES.items():
        data = name.encode()
        path = upstream / source.upstream_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        sources[name] = SmplAssetSource(source.upstream_path, hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(smpl_assets, "SMPL_ASSET_SOURCES", sources)
    return upstream


def test_install_verifies_copies_and_preserves_matching_files(source_tree: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "published-models"
    installed = smpl_assets.install_smpl_assets(gvhmr_root=source_tree, checkpoint_root=checkpoint)
    assert len(installed) == 5
    for name, source in smpl_assets.SMPL_ASSET_SOURCES.items():
        assert (checkpoint / source.checkpoint_path).read_bytes() == name.encode()
    mtimes = [path.stat().st_mtime_ns for path in installed]
    assert smpl_assets.install_smpl_assets(gvhmr_root=source_tree, checkpoint_root=checkpoint) == installed
    assert [path.stat().st_mtime_ns for path in installed] == mtimes


@pytest.mark.parametrize("failure", ["missing", "digest", "conflict", "symlink"])
def test_install_rejects_invalid_input_before_copying_anything(
    source_tree: Path, tmp_path: Path, failure: str,
) -> None:
    checkpoint = tmp_path / "published-models"
    source = list(smpl_assets.SMPL_ASSET_SOURCES.values())[-1]
    original = source_tree / source.upstream_path
    target = checkpoint / source.checkpoint_path
    expected_error: type[Exception]
    if failure == "missing":
        original.unlink()
        expected_error = FileNotFoundError
    elif failure == "digest":
        original.write_bytes(b"wrong version")
        expected_error = ValueError
    elif failure == "conflict":
        target.parent.mkdir(parents=True)
        target.write_bytes(b"keep this existing file")
        expected_error = FileExistsError
    else:
        target.parent.mkdir(parents=True)
        target.symlink_to(original)
        expected_error = ValueError
    with pytest.raises(expected_error):
        smpl_assets.install_smpl_assets(gvhmr_root=source_tree, checkpoint_root=checkpoint)
    for other in list(smpl_assets.SMPL_ASSET_SOURCES.values())[:-1]:
        assert not (checkpoint / other.checkpoint_path).exists()
    if failure == "conflict":
        assert target.read_bytes() == b"keep this existing file"


def test_standalone_hmr2_missing_mean_parameters_shows_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hmr2.ckpt"
    checkpoint.touch()
    model = Hmr2FeatureExtractor(checkpoint, device="cpu", batch_size=1, mean_params_path=tmp_path / "missing.npz")
    with pytest.raises(FileNotFoundError) as error:
        model.load()
    assert SMPL_ASSET_SOURCES["hmr2_mean_params"].source_url in str(error.value)


def test_visualization_missing_regressor_shows_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        load_smpl_joint_regressor(tmp_path / "missing.pt")
    assert SMPL_ASSET_SOURCES["smpl_neutral_joint_regressor"].source_url in str(error.value)


def test_list_cli_exposes_all_sources_without_requiring_a_checkout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from src.submodules.scripts.install_smpl_assets import main

    monkeypatch.setattr("sys.argv", ["install_smpl_assets", "--list"])
    main()
    output = capsys.readouterr().out
    for source in SMPL_ASSET_SOURCES.values():
        assert source.source_url in output
        assert source.sha256 in output
        assert str(Path("ckpt") / source.checkpoint_path) in output
