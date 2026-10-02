"""The experiment disk watchdog tolerates removals, but never hides IO failures."""

from __future__ import annotations

import errno
import os
import runpy
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import cast

import pytest

BUNDLE = (
    Path(__file__).resolve().parents[4]
    / "knowledge/runs/run-i935-precision-variants-s42-r21-20260930"
)


@pytest.fixture
def output_size(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], int]:
    monkeypatch.syspath_prepend(str(BUNDLE))
    return cast(Callable[[Path], int], runpy.run_path(str(BUNDLE / "run_variants.py"))["output_size"])


def test_counts_nested_files_without_following_directory_symlinks(
    tmp_path: Path, output_size: Callable[[Path], int]
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/data").write_bytes(b"12345")
    (tmp_path / "file-link").symlink_to(tmp_path / "nested/data")
    (tmp_path / "dir-link").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "broken-link").symlink_to(tmp_path / "missing")
    assert output_size(tmp_path) == 10
    assert output_size(tmp_path / "missing") == 0


def test_directory_removed_during_walk_does_not_hide_surviving_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_size: Callable[[Path], int]
) -> None:
    vanished = tmp_path / "compiler-temporary"
    vanished.mkdir()
    payload = vanished / "partial"
    payload.write_bytes(b"discard")
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep/result").write_bytes(b"keep")
    scandir = os.scandir
    removed: list[Path] = []

    def remove_before_scan(path: Path) -> AbstractContextManager[Iterator[os.DirEntry[str]]]:
        if Path(path) == vanished and not removed:
            payload.unlink()
            vanished.rmdir()
            removed.append(vanished)
        return scandir(path)

    monkeypatch.setattr(os, "scandir", remove_before_scan)
    assert output_size(tmp_path) == 4
    assert removed == [vanished]


def test_file_removed_after_discovery_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_size: Callable[[Path], int]
) -> None:
    vanished = tmp_path / "partial"
    vanished.write_bytes(b"discard")
    (tmp_path / "result").write_bytes(b"keep")
    scandir = os.scandir

    @contextmanager
    def remove_after_scan(path: Path) -> Iterator[Iterator[os.DirEntry[str]]]:
        with scandir(path) as entries:
            discovered = list(entries)
        vanished.unlink(missing_ok=True)
        yield iter(discovered)

    monkeypatch.setattr(os, "scandir", remove_after_scan)
    assert output_size(tmp_path) == 4


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
@pytest.mark.parametrize("during_iteration", [False, True])
def test_other_walk_errors_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_size: Callable[[Path], int],
    error_number: int,
    during_iteration: bool,
) -> None:
    failure = OSError(error_number, "disk monitor must fail closed")

    def broken_iterator() -> Iterator[os.DirEntry[str]]:
        raise failure
        yield  # pragma: no cover

    @contextmanager
    def broken_scan(path: Path) -> Iterator[Iterator[os.DirEntry[str]]]:
        if not during_iteration:
            raise failure
        yield broken_iterator()

    monkeypatch.setattr(os, "scandir", broken_scan)
    with pytest.raises(OSError) as caught:
        output_size(tmp_path)
    assert caught.value is failure
