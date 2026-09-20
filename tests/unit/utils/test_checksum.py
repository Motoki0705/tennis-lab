from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.utils import checksum


@pytest.mark.parametrize("data", [b"", b"abc", b"abc" * (1024 * 1024)])
def test_known_digest_and_chunked_input(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "data"
    path.write_bytes(data)
    assert checksum.dual_sha256(path) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_links(tmp_path: Path, link: str) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    target = tmp_path / "link"
    if link == "symbolic":
        target.symlink_to(path)
    else:
        os.link(path, target)
    assert checksum.dual_sha256(target) == hashlib.sha256(b"abc").hexdigest()


@pytest.mark.parametrize("provider", ["primary", "independent"])
def test_disagreement_fails_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    faulty = MagicMock()
    faulty.hexdigest.return_value = "0" * 64
    factory = MagicMock(return_value=faulty)
    if provider == "primary":
        monkeypatch.setattr(checksum.hashlib, "sha256", factory)
    else:
        monkeypatch.setattr(
            checksum.importlib.import_module("_sha256"), "sha256", factory
        )
    with pytest.raises(
        checksum.FileIntegrityError, match="providers disagree"
    ) as caught:
        checksum.dual_sha256(path)
    factory.assert_called_once_with()
    assert caught.value.details["bytes_read"] == 3
    assert caught.value.details["path"] == str(path)
    assert (
        caught.value.details["hashlib_sha256"] != caught.value.details["cpython_sha256"]
    )


def test_unavailable_independent_provider_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(checksum.importlib, "import_module", unavailable)
    with pytest.raises(checksum.FileIntegrityError, match="provider is required"):
        checksum.dual_sha256(tmp_path / "data")


@pytest.mark.parametrize("change", ["overwrite", "replace", "short_read"])
def test_file_changed_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    original_open = Path.open
    handle = original_open(path, "rb")
    wrapper = MagicMock()
    wrapper.__enter__.return_value = wrapper
    wrapper.__exit__.side_effect = lambda *args: handle.close()
    wrapper.fileno.side_effect = handle.fileno
    mutated = False

    def read(size: int) -> bytes:
        nonlocal mutated
        if change == "short_read":
            return b""
        data = handle.read(size)
        if not mutated:
            mutated = True
            if change == "overwrite":
                with original_open(path, "wb") as output:
                    output.write(b"defgh")
            else:
                replacement = tmp_path / "replacement"
                replacement.write_bytes(b"abc")
                replacement.replace(path)
        return data

    wrapper.read.side_effect = read

    def patched_open(self: Path, mode: str = "r") -> object:
        if self == path:
            return wrapper
        return original_open(self, mode)

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(checksum.FileIntegrityError) as caught:
        checksum.dual_sha256(path)
    assert caught.value.details["descriptor_after"] is not None
    assert caught.value.details["path_after"] is not None


def test_providers_receive_same_immutable_chunk_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"x" * (1024 * 1024 + 7))
    primary = MagicMock(wraps=hashlib.sha256())
    independent = MagicMock(wraps=checksum.importlib.import_module("_sha256").sha256())
    monkeypatch.setattr(checksum.hashlib, "sha256", lambda: primary)
    monkeypatch.setattr(
        checksum.importlib.import_module("_sha256"), "sha256", lambda: independent
    )
    checksum.dual_sha256(path)
    assert [len(call.args[0]) for call in primary.update.call_args_list] == [
        1024 * 1024,
        7,
    ]
    for left, right in zip(
        primary.update.call_args_list, independent.update.call_args_list, strict=True
    ):
        assert type(left.args[0]) is bytes
        assert left.args[0] is right.args[0]
