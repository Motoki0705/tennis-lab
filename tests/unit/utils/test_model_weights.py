"""Publication bytes are authenticated before becoming inference assets."""

from __future__ import annotations

import hashlib
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import yaml
from huggingface_hub import constants as hub_constants
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import _http as hub_http

from src.utils import model_weights as weights
from src.utils.configuration import PathContractError
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize("license_document", ["LICENSES.md", "LICENSES.ja.md"])
def test_release_paths_match_pipeline_and_license_documents(
    license_document: str,
) -> None:
    pipeline = yaml.safe_load(
        (PROJECT_ROOT / "src/tennis_scene/configs/pipeline.yaml").read_text()
    )
    licenses = (PROJECT_ROOT / license_document).read_text()
    assert f"https://huggingface.co/{weights.MODEL_REPO}" in licenses
    configuration_keys = {"court": "court_kp", "ball": "ball_detection"}
    for weight in weights.PUBLISHED_WEIGHTS:
        assert (
            pipeline[configuration_keys[weight.name]]["checkpoint"] == weight.filename
        )
        assert f"`{weight.filename}`" in licenses


@pytest.fixture
def published_content(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    content = {"court": b"court checkpoint", "ball": b"ball checkpoint"}
    registry = tuple(
        weights.PublishedWeight(
            name, f"{name}_detection/release.ckpt", hashlib.sha256(data).hexdigest()
        )
        for name, data in content.items()
    )
    monkeypatch.setattr(weights, "PUBLISHED_WEIGHTS", registry)
    return content


@pytest.fixture
def published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_content: dict[str, bytes],
) -> tuple[dict[str, bytes], Mock]:
    content = published_content
    registry = weights.PUBLISHED_WEIGHTS
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
            force_download=True,
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


@pytest.mark.parametrize("revision", ["main", "a" * 40], ids=["branch", "commit"])
@pytest.mark.parametrize("method", ["HEAD", "GET"])
@pytest.mark.parametrize("failure", ["timeout", "503"])
def test_cli_rejects_http_failure_with_valid_hub_cache(
    tmp_path: Path,
    published_content: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    revision: str,
    method: str,
    failure: str,
) -> None:
    cache = tmp_path / "hub-cache"
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hub_constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    monkeypatch.setattr(hub_constants, "HF_TOKEN_PATH", str(tmp_path / "no-token"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    weight = weights.PUBLISHED_WEIGHTS[1]
    content = published_content[weight.name]
    requests: list[str] = []
    fail_requests = False

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        if fail_requests and request.method == method:
            if failure == "timeout":
                raise httpx.ReadTimeout("simulated Hub timeout", request=request)
            return httpx.Response(503)
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "X-Repo-Commit": "a" * 40,
                    "ETag": f'"{weight.sha256}"',
                    "Content-Length": str(len(content)),
                },
            )
        assert request.method == "GET"
        return httpx.Response(200, content=content)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(hub_http, "get_session", lambda: client)
        # Populate a real Hub blob, snapshot and branch ref through the actual
        # downloader. Only its HTTP transport is replaced, including on failure.
        cached = Path(
            weights.hf_hub_download(
                repo_id=weights.MODEL_REPO,
                filename=weight.filename,
                revision=revision,
            )
        )
        assert cached.is_relative_to(cache)
        assert hashlib.sha256(cached.read_bytes()).hexdigest() == weight.sha256
        assert requests == ["HEAD", "GET"]
        requests.clear()
        fail_requests = True
        root = tmp_path / "ckpt"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "model_weights",
                "--checkpoint-root",
                str(root),
                "--model",
                weight.name,
                "--revision",
                revision,
            ],
        )
        transport_error = httpx.ReadTimeout if failure == "timeout" else HfHubHTTPError
        expected_error = ValueError if method == "HEAD" else transport_error
        with pytest.raises(expected_error) as error:
            runpy.run_module("src.utils.model_weights", run_name="__main__")
        # HEAD errors are chained by huggingface_hub; GET errors propagate directly.
        underlying = error.value.__cause__ if method == "HEAD" else error.value
        assert isinstance(underlying, transport_error)
        if isinstance(underlying, HfHubHTTPError):
            assert underlying.response.status_code == 503
        assert requests == (["HEAD"] if method == "HEAD" else ["HEAD", "GET"])
        assert not root.exists()
        assert cached.read_bytes() == content
        assert "Verified" not in capsys.readouterr().out


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

    def race(**kwargs: object) -> str:
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
    runpy.run_module("src.utils.model_weights", run_name="__main__")
    assert (
        capsys.readouterr().out.strip()
        == f"Verified {root / weights.PUBLISHED_WEIGHTS[1].filename}"
    )
    published[1].assert_not_called()
