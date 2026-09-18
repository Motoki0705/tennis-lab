"""Tiny CPU captures only: never use a real checkpoint in these tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "capture_fixture", Path(__file__).with_name("capture.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_fixture(tmp_path: Path, data: bytes = b"tiny checkpoint") -> tuple[Path, str]:
    source = tmp_path / "live.bin"
    source.write_bytes(data)
    return source, hashlib.sha256(data).hexdigest()


def test_same_read_is_hashed_and_saved_across_chunk_boundary(
    driver: ModuleType, tmp_path: Path
) -> None:
    data = b"a" * (1024 * 1024) + b"last"
    source, expected = live_fixture(tmp_path, data)
    snapshot = tmp_path / "snapshot.bin"
    row = driver.read_stream(source, snapshot, expected)
    driver.verify_snapshot(row)
    assert row["status"] == "passed"
    assert snapshot.read_bytes() == data
    assert [chunk["size"] for chunk in row["chunks"]] == [1024 * 1024, 4]
    assert row["chunks"][1]["offset"] == 1024 * 1024
    assert row["isolated_python"]["hashlib_sha256"] == expected
    assert row["external_sha256sum"]["sha256"] == expected


def test_mutable_changed_buffer_is_rejected(driver: ModuleType, tmp_path: Path) -> None:
    source, expected = live_fixture(tmp_path)
    buffer = bytearray(b"mutable")
    with source.open("rb") as real:
        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.fileno.return_value = real.fileno()
        reader.read.return_value = buffer
        row = driver.read_stream(
            source, tmp_path / "snapshot.bin", expected, opener=lambda _: reader
        )
    buffer[:] = b"changed"
    assert row["status"] == "failed"
    assert row["errors"][0]["type"] == "TypeError"
    assert "immutable bytes" in row["errors"][0]["error"]
    assert row["bytes_written"] == 0


def test_provider_disagreement_preserves_captured_bytes(
    driver: ModuleType, tmp_path: Path
) -> None:
    source, expected = live_fixture(tmp_path)
    provider = MagicMock()
    provider.hexdigest.return_value = "0" * 64
    row = driver.read_stream(
        source, tmp_path / "snapshot.bin", expected, secondary_factory=lambda: provider
    )
    driver.verify_snapshot(row)
    assert row["status"] == "failed"
    assert not row["checks"]["providers_agree"]
    assert Path(row["snapshot"]).read_bytes() == b"tiny checkpoint"
    assert row["isolated_python"]["hashlib_sha256"] == expected


def test_wrong_pin_stops_remaining_passes_and_keeps_failure(
    driver: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = live_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    original = driver.read_stream
    calls = []

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(driver, "read_stream", counted)
    report = driver.run_plan(source, output, 3, "0" * 64)
    assert report["status"] == "failed"
    assert len(calls) == report["completed_passes"] == 1
    assert report["cancelled_passes"] == 2
    assert (output / "pass-001.bin").exists()
    assert not (output / "pass-002.bin").exists()
    assert json.loads((output / "audit.json").read_text())["status"] == "failed"


def test_short_live_read_is_not_retried(driver: ModuleType, tmp_path: Path) -> None:
    source, expected = live_fixture(tmp_path)
    with source.open("rb") as real:
        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.fileno.return_value = real.fileno()
        reader.read.side_effect = [b"tiny", b""]
        row = driver.read_stream(
            source, tmp_path / "snapshot.bin", expected, opener=lambda _: reader
        )
    assert row["status"] == "failed"
    assert not row["checks"]["byte_count_matches"]
    assert row["bytes_read"] == row["bytes_written"] == 4
    assert reader.read.call_count == 2


def test_saved_byte_change_is_detected_without_live_reread(
    driver: ModuleType, tmp_path: Path
) -> None:
    source, expected = live_fixture(tmp_path)
    row = driver.read_stream(source, tmp_path / "snapshot.bin", expected)
    Path(row["snapshot"]).write_bytes(b"different saved bytes")
    source.unlink()
    driver.verify_snapshot(row)
    assert row["status"] == "failed"
    assert not row["snapshot_checks"]["saved_chunks_match_capture"]
    assert not row["snapshot_checks"]["saved_providers_match_capture"]


def test_finite_success_keeps_all_passes_and_pair_comparisons(
    driver: ModuleType, tmp_path: Path
) -> None:
    source, expected = live_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    result = driver.run_plan(source, output, 3, expected)
    assert result["status"] == "passed"
    assert len(result["passes"]) == 3
    assert len(result["comparisons"]) == 3
    assert all(row["different_bytes"] == 0 for row in result["comparisons"])
    assert len(list(output.glob("*.bin"))) == 3


def test_snapshot_difference_offsets_and_length_tail(
    driver: ModuleType, tmp_path: Path
) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.write_bytes(b"abcde")
    right.write_bytes(b"abXdeZ")
    result = driver.compare_snapshots(
        {"snapshot": str(left), "chunks": [{"sha256": "a"}]},
        {"snapshot": str(right), "chunks": [{"sha256": "b"}]},
    )
    assert result["different_bytes"] == 2
    assert result["first_difference_offset_zero_based"] == 2
    assert result["different_chunk_indices"] == [0]


def test_existing_output_refused_before_live_read(
    driver: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("unchanged")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture",
            "--checkpoint",
            str(tmp_path / "nonexistent"),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError):
        driver.main()
    assert marker.read_text() == "unchanged"
    assert set(output.iterdir()) == {marker}


@pytest.mark.parametrize("passes", [0, 4])
def test_pass_limit(driver: ModuleType, tmp_path: Path, passes: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        driver.run_plan(tmp_path / "missing", tmp_path, passes)
