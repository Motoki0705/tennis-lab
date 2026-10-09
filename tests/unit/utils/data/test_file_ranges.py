from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import pytest

from src.utils.data.file_ranges import FileRangeReader
from src.utils.shared_file_verification import file_version


def test_disjoint_repeated_ranges_keep_exact_order_and_survive_pickle(tmp_path: Path) -> None:
    path = tmp_path / "bytes"
    payload = bytes(range(64))
    path.write_bytes(payload)
    expected = file_version(path)
    reader = FileRangeReader(max_open_files=1)
    ranges = [(19, 5), (0, 3), (19, 5)]
    assert reader.read(path, ranges, expected).tobytes() == payload[19:24] + payload[:3] + payload[19:24]
    restored = pickle.loads(pickle.dumps(reader))
    reader.close()
    assert restored.read(path, ranges, expected).tobytes() == payload[19:24] + payload[:3] + payload[19:24]
    restored.close()


def test_partial_reads_are_completed_and_eof_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bytes"
    path.write_bytes(b"0123456789")
    reader = FileRangeReader()
    expected = file_version(path)
    original = os.preadv

    def partial(fd: int, buffers: Any, offset: int) -> int:
        return original(fd, [buffers[0][:2]], offset)

    monkeypatch.setattr(os, "preadv", partial)
    assert reader.read(path, [(2, 7)], expected).tobytes() == b"2345678"
    monkeypatch.setattr(os, "preadv", lambda *_: 0)
    with pytest.raises(ValueError, match="EOF"):
        reader.read(path, [(0, 4)], expected)
    reader.close()


def test_path_replacement_cannot_reuse_verified_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "bytes"
    path.write_bytes(b"same")
    expected = file_version(path)
    reader = FileRangeReader()
    reader.read(path, [(0, 4)], expected)
    replacement = tmp_path / "new"
    replacement.write_bytes(b"same")
    replacement.replace(path)
    with pytest.raises(ValueError, match="changed"):
        reader.read(path, [(0, 4)], expected)
    reader.close()


def test_mutation_during_read_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bytes"
    path.write_bytes(b"original")
    expected = file_version(path)
    original = os.preadv

    def mutate(fd: int, buffers: Any, offset: int) -> int:
        count = original(fd, buffers, offset)
        path.write_bytes(b"changed and longer")
        return count

    monkeypatch.setattr(os, "preadv", mutate)
    reader = FileRangeReader()
    with pytest.raises(ValueError, match="changed"):
        reader.read(path, [(0, 8)], expected)
    reader.close()


@pytest.mark.parametrize("ranges", [[], [(-1, 1)], [(0, 0)], [(2, 9)]])
def test_invalid_ranges_are_rejected(tmp_path: Path, ranges: list[tuple[int, int]]) -> None:
    path = tmp_path / "bytes"
    path.write_bytes(b"0123")
    reader = FileRangeReader()
    with pytest.raises(ValueError, match="ranges"):
        reader.read(path, ranges, file_version(path))
