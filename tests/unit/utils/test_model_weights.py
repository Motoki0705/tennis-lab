"""Publication bytes are authenticated before becoming inference assets."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from src.utils import model_weights as weights
from src.utils.configuration import PathContractError
from src.utils.paths import PROJECT_ROOT


def test_release_paths_match_pipeline_and_documented_hashes() -> None:
    pipeline = yaml.safe_load(
        (PROJECT_ROOT / "src/tennis_scene/configs/pipeline.yaml").read_text()
    )
    licenses = (PROJECT_ROOT / "LICENSES.md").read_text()
    configuration_keys = {"court": "court_kp", "ball": "ball_detection"}
    for weight in weights.PUBLISHED_WEIGHTS:
        assert pipeline[configuration_keys[weight.name]]["checkpoint"] == weight.filename
        assert f"`{weight.filename}` | `{weight.sha256}`" in licenses


@pytest.fixture
def published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, bytes], Mock]:
    content = {"court": b"court checkpoint", "ball": b"ball checkpoint"}
    registry = tuple(
        weights.PublishedWeight(
            name, f"{name}_detection/release.ckpt", hashlib.sha256(data).hexdigest()
        )
        for name, data in content.items()
    )
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    for weight in registry:
        (cache / weight.name).write_bytes(content[weight.name])
    download = Mock(
        side_effect=lambda **kwargs: str(
            cache
            / next(
                weight.name
                for weight in registry
                if weight.filename == kwargs["filename"]
            )
        )
    )
    monkeypatch.setattr(weights, "PUBLISHED_WEIGHTS", registry)
    monkeypatch.setattr(weights, "hf_hub_download", download)
    return content, download


def test_download_places_verified_assets_under_pipeline_paths(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
) -> None:
    content, download = published
    root = tmp_path / "ckpt"
    paths = weights.ensure_weights(root, revision="fixed-release")
    assert len(paths) == 2
    for weight, path in zip(weights.PUBLISHED_WEIGHTS, paths, strict=True):
        assert path == root / weight.filename
        assert path.read_bytes() == content[weight.name]
        download.assert_any_call(
            repo_id="Motoki098/tennis-lab-models",
            filename=weight.filename,
            repo_type="model",
            revision="fixed-release",
        )
    assert download.call_count == 2
    assert not list(root.rglob("*.tmp"))


@pytest.mark.parametrize("verify_only", [False, True])
def test_valid_existing_files_work_offline_without_overwrite(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
    verify_only: bool,
) -> None:
    content, download = published
    root = tmp_path / "ckpt"
    identities = []
    for weight in weights.PUBLISHED_WEIGHTS:
        path = root / weight.filename
        path.parent.mkdir(parents=True)
        path.write_bytes(content[weight.name])
        identities.append(path.stat().st_ino)
    download.side_effect = AssertionError("offline")
    paths = weights.ensure_weights(root, verify_only=verify_only)
    assert [path.stat().st_ino for path in paths] == identities
    download.assert_not_called()


def test_conflict_fails_entire_selection_before_download(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
) -> None:
    _, download = published
    root = tmp_path / "ckpt"
    bad = root / weights.PUBLISHED_WEIGHTS[1].filename
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"old training checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        weights.ensure_weights(root)
    assert bad.read_bytes() == b"old training checkpoint"
    download.assert_not_called()
    assert not (root / weights.PUBLISHED_WEIGHTS[0].filename).exists()


def test_missing_file_fails_offline_verification(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
) -> None:
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        weights.ensure_weights(tmp_path / "ckpt", verify_only=True)
    published[1].assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "corrupt", "network"])
def test_bad_download_is_never_installed(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
    failure: str,
) -> None:
    _, download = published
    cached = tmp_path / "bad-cache"
    if failure == "corrupt":
        cached.write_bytes(b"tampered")
    if failure == "network":
        download.side_effect = RuntimeError("Hub request failed")
        expected_error: type[Exception] = RuntimeError
    else:
        download.side_effect = None
        download.return_value = str(cached)
        expected_error = ValueError if failure == "corrupt" else FileNotFoundError
    root = tmp_path / "ckpt"
    with pytest.raises(expected_error):
        weights.ensure_weights(root, model="ball")
    assert not root.exists()
    assert download.call_count == 1


def test_corrupted_copy_is_removed_before_publication(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        weights.shutil,
        "copyfile",
        lambda source, target: Path(target).write_bytes(b"bad copy"),
    )
    root = tmp_path / "ckpt"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        weights.ensure_weights(root, model="ball")
    assert not list(root.rglob("*.ckpt"))
    assert not list(root.rglob("*.tmp"))


def test_concurrent_target_is_not_overwritten(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    link = os.link

    def race(source: Path, target: Path) -> None:
        target.write_bytes(b"concurrent writer")
        link(source, target)

    monkeypatch.setattr(weights.os, "link", race)
    root = tmp_path / "ckpt"
    with pytest.raises(FileExistsError):
        weights.ensure_weights(root, model="ball")
    assert (
        root / weights.PUBLISHED_WEIGHTS[1].filename
    ).read_bytes() == b"concurrent writer"
    assert not list(root.rglob("*.tmp"))


def test_symlink_cannot_redirect_installation_outside_root(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
) -> None:
    root = tmp_path / "ckpt"
    elsewhere = tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (root / "ball_detection").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(PathContractError):
        weights.ensure_weights(root, model="ball")
    assert not list(elsewhere.iterdir())
    published[1].assert_not_called()


def test_target_appearing_during_download_is_not_trusted(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
) -> None:
    _, download = published
    cached_download = download.side_effect
    root = tmp_path / "ckpt"
    destination = root / weights.PUBLISHED_WEIGHTS[1].filename

    def race(**kwargs: str) -> str:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"unverified concurrent file")
        return str(cached_download(**kwargs))

    download.side_effect = race
    with pytest.raises(FileExistsError):
        weights.ensure_weights(root, model="ball")
    assert destination.read_bytes() == b"unverified concurrent file"
    assert not list(root.rglob("*.tmp"))


@pytest.mark.parametrize(("model", "revision"), [("unknown", "main"), ("all", "")])
def test_invalid_selection_fails_before_network(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
    model: str,
    revision: str,
) -> None:
    with pytest.raises(ValueError):
        weights.ensure_weights(tmp_path / "ckpt", model=model, revision=revision)
    published[1].assert_not_called()


def test_relative_root_is_rejected(published: tuple[dict[str, bytes], Mock]) -> None:
    with pytest.raises(ValueError, match="absolute"):
        weights.ensure_weights(Path("ckpt"))
    published[1].assert_not_called()


def test_cli_selects_one_model_and_verifies_offline(
    tmp_path: Path,
    published: tuple[dict[str, bytes], Mock],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ckpt"
    weights.ensure_weights(root, model="ball")
    published[1].reset_mock()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_weights",
            "--checkpoint-root",
            str(root),
            "--model",
            "ball",
            "--verify-only",
        ],
    )
    weights.main()
    assert (
        capsys.readouterr().out.strip()
        == f"Verified {root / weights.PUBLISHED_WEIGHTS[1].filename}"
    )
    published[1].assert_not_called()
