"""Resume decisions must follow fresh content evidence, not path existence."""

# mypy: disallow_untyped_decorators=False

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location("transfer_data", Path(__file__).with_name("transfer_data.py"))
assert spec is not None and spec.loader is not None
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
fingerprints, matching = transfer.fingerprints, transfer.matching


def test_resume_skips_only_content_matches() -> None:
    expected = {"shards/a.bin": dict(size_bytes=3, md5="abc"), "shards/b.bin": dict(size_bytes=4, md5="def")}
    observed = dict(entries=[dict(path="shards", type="directory"),
                            dict(path="shards/a.bin", type="file", size_bytes=3, hashes=dict(md5="abc"))])
    assert matching(observed, expected) == {"shards/a.bin"}


@pytest.mark.parametrize("size,digest", [(3, "wrong"), (2, "abc"), (3, None)])
def test_resume_rejects_partial_corrupt_or_unverified_entries(size: int, digest: str | None) -> None:
    expected = {"a": dict(size_bytes=3, md5="abc")}
    observed = dict(entries=[dict(path="a", type="file", size_bytes=size, hashes=dict(md5=digest))])
    with pytest.raises(ValueError, match="differs"):
        matching(observed, expected)


def test_resume_rejects_files_outside_the_pinned_asset() -> None:
    with pytest.raises(ValueError, match="outside"):
        matching(dict(entries=[dict(path="unexpected", type="file")]), {})


def test_fingerprint_receipt_rechecks_changed_files(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    expected = {"source.bin": dict(size_bytes=3, sha256=hashlib.sha256(b"abc").hexdigest())}
    result: dict[str, Any] = fingerprints({"source.bin": source}, expected, tmp_path)
    assert result["source.bin"]["md5"] == hashlib.md5(b"abc").hexdigest()
    assert fingerprints({"source.bin": source}, expected, tmp_path) == result
    source.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="immutable distribution"):
        fingerprints({"source.bin": source}, expected, tmp_path)
