"""The public sample is authenticated and never replaces a user's inputs."""

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

from src.utils import sample_dataset as sample
from src.utils.configuration import PathContractError
from src.utils.paths import PROJECT_ROOT


def test_release_layout_matches_default_pipeline() -> None:
    pipeline = yaml.safe_load(
        (PROJECT_ROOT / "src/tennis_scene/configs/pipeline.yaml").read_text()
    )
    videos = [p for p in sample.PUBLISHED_FILES if p.filename.endswith(".mp4")]
    assert pipeline["video_paths"] == [f"samples/{p.filename}" for p in videos]
    assert pipeline["camera_ids"] == [Path(p.filename).stem for p in videos]
    assert {p.filename for p in sample.PUBLISHED_FILES} == {
        "cam0.mp4",
        "cam1.mp4",
        "cam2.mp4",
        "clip.json",
        "LICENSE",
    }


@pytest.fixture
def content(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    files = {
        p.filename: f"sample {p.filename}".encode() for p in sample.PUBLISHED_FILES
    }
    monkeypatch.setattr(
        sample,
        "PUBLISHED_FILES",
        tuple(
            sample.PublishedSampleFile(name, hashlib.sha256(data).hexdigest())
            for name, data in files.items()
        ),
    )
    return files


@pytest.fixture
def download(
    tmp_path: Path, content: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> Mock:
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    for name, data in content.items():
        (cache / name).write_bytes(data)
    mock = Mock(side_effect=lambda **kwargs: str(cache / kwargs["filename"]))
    monkeypatch.setattr(sample, "hf_hub_download", mock)
    return mock


def test_install_downloads_only_release_files_and_verifies_offline(
    tmp_path: Path, content: dict[str, bytes], download: Mock
) -> None:
    root = tmp_path / "data"
    result = sample.ensure_sample_dataset(root, revision="fixed-release")
    assert result == tuple(root / "samples" / name for name in content)
    assert download.call_count == len(content)
    for path, call in zip(result, download.call_args_list, strict=True):
        assert path.read_bytes() == content[path.name]
        assert call.kwargs == dict(
            repo_id=sample.DATASET_REPO,
            filename=path.name,
            repo_type="dataset",
            revision="fixed-release",
            force_download=True,
        )
    before = [p.stat().st_mtime_ns for p in result]
    download.reset_mock()
    assert sample.ensure_sample_dataset(root, verify_only=True) == result
    assert sample.ensure_sample_dataset(root) == result
    download.assert_not_called()
    assert [p.stat().st_mtime_ns for p in result] == before
    assert sorted(p.name for p in (root / "samples").iterdir()) == sorted(content)


def test_conflicting_last_file_prevents_all_downloads(
    tmp_path: Path, download: Mock
) -> None:
    root = tmp_path / "data"
    conflict = root / "samples" / sample.PUBLISHED_FILES[-1].filename
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"local input")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        sample.ensure_sample_dataset(root)
    assert conflict.read_bytes() == b"local input"
    assert list(conflict.parent.iterdir()) == [conflict]
    download.assert_not_called()


def test_missing_file_fails_offline_without_creating_root(
    tmp_path: Path, download: Mock
) -> None:
    root = tmp_path / "data"
    with pytest.raises(FileNotFoundError, match="Missing sample file"):
        sample.ensure_sample_dataset(root, verify_only=True)
    assert not root.exists()
    download.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "corrupt", "network"])
def test_bad_download_is_not_installed(
    tmp_path: Path, download: Mock, failure: str
) -> None:
    cached = tmp_path / "invalid"
    if failure == "corrupt":
        cached.write_bytes(b"corrupt")
    download.side_effect = (
        RuntimeError("Hub unavailable") if failure == "network" else None
    )
    download.return_value = str(cached)
    expected = {
        "missing": FileNotFoundError,
        "corrupt": ValueError,
        "network": RuntimeError,
    }
    root = tmp_path / "data"
    with pytest.raises(expected[failure]):
        sample.ensure_sample_dataset(root)
    assert not root.exists()
    assert download.call_count == 1


def test_corrupt_copy_is_removed(
    tmp_path: Path, download: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sample.shutil, "copyfile", lambda src, dst: Path(dst).write_bytes(b"bad copy")
    )
    root = tmp_path / "data"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        sample.ensure_sample_dataset(root)
    assert list((root / "samples").iterdir()) == []


def test_concurrent_file_is_not_overwritten(
    tmp_path: Path, download: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = os.link

    def race(source: Path, target: Path) -> None:
        target.write_bytes(b"concurrent input")
        original_link(source, target)

    monkeypatch.setattr(sample.os, "link", race)
    root = tmp_path / "data"
    with pytest.raises(FileExistsError):
        sample.ensure_sample_dataset(root)
    destination = root / "samples" / sample.PUBLISHED_FILES[0].filename
    assert destination.read_bytes() == b"concurrent input"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("link_target", ["directory", "file"])
def test_symlink_cannot_escape_data_root(
    tmp_path: Path, download: Mock, link_target: str
) -> None:
    root = tmp_path / "data"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    if link_target == "directory":
        (root / "samples").symlink_to(outside, target_is_directory=True)
    else:
        (root / "samples").mkdir()
        (root / "samples" / sample.PUBLISHED_FILES[0].filename).symlink_to(
            outside / "new.mp4"
        )
    with pytest.raises(PathContractError):
        sample.ensure_sample_dataset(root)
    assert list(outside.iterdir()) == []
    download.assert_not_called()


@pytest.mark.parametrize(
    ("root", "revision"), [(Path("data"), "main"), (Path("/tmp/sample-data"), " ")]
)
def test_invalid_arguments_fail_before_network(
    root: Path, revision: str, download: Mock
) -> None:
    with pytest.raises(ValueError):
        sample.ensure_sample_dataset(root, revision=revision)
    download.assert_not_called()


def test_cli_verifies_without_network(
    tmp_path: Path,
    content: dict[str, bytes],
    download: Mock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "data"
    (root / "samples").mkdir(parents=True)
    for name, data in content.items():
        (root / "samples" / name).write_bytes(data)
    monkeypatch.setattr(
        sys, "argv", ["sample_dataset", "--data-root", str(root), "--verify-only"]
    )
    runpy.run_module("src.utils.sample_dataset", run_name="__main__")
    assert capsys.readouterr().out.count("Verified ") == len(content)
    download.assert_not_called()


@pytest.mark.parametrize("revision", ["main", "a" * 40], ids=["branch", "commit"])
@pytest.mark.parametrize("method", ["HEAD", "GET"])
def test_http_failure_does_not_use_valid_hub_cache(
    tmp_path: Path,
    content: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
    method: str,
) -> None:
    cache = tmp_path / "real-hub-cache"
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hub_constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    monkeypatch.setattr(hub_constants, "HF_TOKEN_PATH", str(tmp_path / "no-token"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    published = sample.PUBLISHED_FILES[0]
    data = content[published.filename]
    fail = False
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        if fail and request.method == method:
            return httpx.Response(503)
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "X-Repo-Commit": "a" * 40,
                    "ETag": f'"{published.sha256}"',
                    "Content-Length": str(len(data)),
                },
            )
        return httpx.Response(200, content=data)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(hub_http, "get_session", lambda: client)
        cached = Path(
            sample.hf_hub_download(
                repo_id=sample.DATASET_REPO,
                filename=published.filename,
                repo_type="dataset",
                revision=revision,
            )
        )
        assert cached.is_relative_to(cache)
        assert cached.read_bytes() == data
        fail = True
        requests.clear()
        root = tmp_path / "data"
        expected = ValueError if method == "HEAD" else HfHubHTTPError
        with pytest.raises(expected):
            sample.ensure_sample_dataset(root, revision=revision)
        assert requests == (["HEAD"] if method == "HEAD" else ["HEAD", "GET"])
        assert not root.exists()
        assert cached.read_bytes() == data
