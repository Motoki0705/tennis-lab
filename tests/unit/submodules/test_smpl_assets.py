"""Acquisition provenance, checked local installation and actionable failures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.submodules import smpl_assets
from src.submodules.models import Hmr2FeatureExtractor
from src.submodules.scripts import install_smpl_assets as install_cli
from src.submodules.smpl_assets import SMPL_ASSET_SOURCES, SmplAssetSource
from src.tennis_scene.visualization.rendering.smpl_court import (
    load_smpl_joint_regressor,
)
from src.utils.configuration import PathContractError


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "published-models"
    monkeypatch.setattr("sys.argv", [
        "install_smpl_assets", "--list", "--checkpoint-root", str(checkpoint),
    ])
    install_cli.main()
    output = capsys.readouterr().out
    for source in SMPL_ASSET_SOURCES.values():
        assert source.source_url in output
        assert source.sha256 in output
        assert str(checkpoint / source.checkpoint_path) in output
    assert not checkpoint.exists()


def test_install_cli_accepts_explicit_absolute_roots(
    source_tree: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "published-models"
    monkeypatch.setattr("sys.argv", [
        "install_smpl_assets", "--gvhmr-root", str(source_tree),
        "--checkpoint-root", str(checkpoint),
    ])

    install_cli.main()

    assert set(capsys.readouterr().out.splitlines()) == {
        str(checkpoint / source.checkpoint_path)
        for source in smpl_assets.SMPL_ASSET_SOURCES.values()
    }
    for name, source in smpl_assets.SMPL_ASSET_SOURCES.items():
        assert (checkpoint / source.checkpoint_path).read_bytes() == name.encode()


@pytest.mark.parametrize("option", ["--gvhmr-root", "--checkpoint-root"])
def test_install_cli_rejects_relative_paths_before_creating_output(
    source_tree: Path, tmp_path: Path, option: str,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "published-models"
    arguments = [
        "install_smpl_assets", "--gvhmr-root", str(source_tree),
        "--checkpoint-root", str(checkpoint),
    ]
    arguments[arguments.index(option) + 1] = "relative-root"
    monkeypatch.setattr("sys.argv", arguments)

    with pytest.raises(SystemExit) as error:
        install_cli.main()

    assert error.value.code == 2
    assert "absolute paths" in capsys.readouterr().err
    assert not checkpoint.exists()


@pytest.mark.parametrize("list_only", [False, True])
@pytest.mark.parametrize("source_kind", ["missing", "file"])
def test_install_cli_validates_supplied_checkout_even_when_listing(
    tmp_path: Path, list_only: bool, source_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = tmp_path / "invalid-checkout"
    if source_kind == "file":
        upstream.write_text("not a directory", encoding="utf-8")
    checkpoint = tmp_path / "published-models"
    arguments = [
        "install_smpl_assets", "--gvhmr-root", str(upstream),
        "--checkpoint-root", str(checkpoint),
    ]
    if list_only:
        arguments.append("--list")
    monkeypatch.setattr("sys.argv", arguments)

    with pytest.raises(PathContractError, match="does not exist|existing directory"):
        install_cli.main()

    assert not checkpoint.exists()


@pytest.mark.parametrize("list_only", [False, True])
def test_install_cli_requires_explicit_checkpoint_root(
    tmp_path: Path, list_only: bool,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = ["install_smpl_assets", "--gvhmr-root", str(tmp_path)]
    if list_only:
        arguments.append("--list")
    monkeypatch.setattr("sys.argv", arguments)

    with pytest.raises(SystemExit) as error:
        install_cli.main()

    assert error.value.code == 2
    assert "required: --checkpoint-root" in capsys.readouterr().err


def test_install_cli_requires_checkout_when_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "published-models"
    monkeypatch.setattr("sys.argv", [
        "install_smpl_assets", "--checkpoint-root", str(checkpoint),
    ])

    with pytest.raises(SystemExit) as error:
        install_cli.main()

    assert error.value.code == 2
    assert "--gvhmr-root is required unless --list" in capsys.readouterr().err
    assert not checkpoint.exists()
