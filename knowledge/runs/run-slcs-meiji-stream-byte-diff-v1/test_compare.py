# mypy: disallow_untyped_decorators=False
"""Saved-file-only tests; no checkpoint/model loading."""

from __future__ import annotations

import hashlib
import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "byte_diff", Path(__file__).with_name("compare.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(
    driver: ModuleType, tmp_path: Path, left: bytes, right: bytes, cap: int = 4096
) -> Any:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(left)
    b.write_bytes(right)
    return driver.scan(
        a,
        b,
        (hashlib.sha256(left).hexdigest(), hashlib.sha256(right).hexdigest()),
        cap=cap,
    )


def test_bit_flips_offsets_and_exact_totals(driver: ModuleType, tmp_path: Path) -> None:
    left, right = bytearray(4100), bytearray(4100)
    right[2], right[4097] = 1, 0x81
    report = compare(driver, tmp_path, bytes(left), bytes(right))
    assert report["gates_passed"] and report["differing_bytes"] == 2
    assert report["flipped_bits_total"] == 3
    assert report["xor_distribution"] == {"0x01": 1, "0x81": 1}
    assert report["bit_flip_counts_lsb0"] == [2, 0, 0, 0, 0, 0, 0, 1]
    assert report["first_difference"] == 2 and report["last_difference"] == 4097
    assert report["details"][1]["page_offset"] == 1
    assert report["difference_ranges_inclusive"] == [[2, 2], [4097, 4097]]


def test_length_tail_and_detail_cap_keep_exact_totals(
    driver: ModuleType, tmp_path: Path
) -> None:
    report = compare(driver, tmp_path, b"abc", b"aXYZmore", cap=1)
    assert report["gates_passed"] and not report["byte_equal"]
    assert report["differing_bytes"] == 7 and report["length_only_bytes"] == 5
    assert report["details_truncated"] and len(report["details"]) == 1
    assert report["difference_ranges_inclusive"] == [[1, 7]]


def test_expected_hash_mismatch_fails_gate(driver: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "saved.bin"
    path.write_bytes(b"saved")
    report = driver.scan(path, path, ("0" * 64, "0" * 64))
    assert report["byte_equal"] and not report["gates_passed"]


def test_equal_streams(driver: ModuleType, tmp_path: Path) -> None:
    report = compare(driver, tmp_path, b"abc", b"abc")
    assert report["gates_passed"] and report["byte_equal"]
    assert report["first_difference"] is None and report["details"] == []


def test_zip_header_only_location(driver: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "saved.bin"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data/0", b"saved tensor bytes")
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()[0]
        start = info.header_offset + 30 + len(info.filename.encode())
    result = driver.archive_entries(path, [[start + 3, start + 3]])
    assert result["entries"][0]["filename"] == "archive/data/0"
    assert result["entries"][0]["differing_payload_bytes"] == 1
    assert result["entries"][0]["pickle_loaded"] is False


def test_requires_disabled_cuda(
    driver: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        driver.run(tmp_path / "good.json", tmp_path / "bad.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()
