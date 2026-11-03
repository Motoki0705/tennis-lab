"""Resume decisions must follow fresh content evidence, not path existence."""

# mypy: disallow_untyped_decorators=False

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace
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


@pytest.mark.parametrize("duplicate", [False, True])
def test_workers_start_only_after_unique_payload_directory_is_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate: bool,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "metadata.json").write_text(json.dumps({"version": "test-v1"}))
    (stage / "distribution.json").write_text("{}")
    files = {"metadata.json": stage / "metadata.json"}
    for index in range(4):
        source = tmp_path / f"{index}.bin"
        source.write_bytes(f"payload-{index}".encode())
        files[f"shards/{index}.bin"] = source
    manifest = {
        "clips": [{"payload": name} for name in files if name.startswith("shards/")],
        "files": [dict(path=name, size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                  for name, path in files.items()],
    }
    remote = "data/ball_detection/test-v1"
    events: list[str] = []
    entries: list[dict[str, Any]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=10)

    def fake_checked(_receipts: Path, _label: str, arguments: list[str]) -> dict[str, Any]:
        if arguments[0] == "mkdir":
            events.append(arguments[1])
            if arguments[1] == remote + "/shards":
                entries.append(dict(path="shards", type="directory"))
        elif arguments[0] == "inventory":
            events.append("inventory")
            document = dict(entries=list(entries), ambiguous_paths=["shards"] if duplicate and entries else [])
            Path(arguments[arguments.index("--output") + 1]).write_text(json.dumps(document))
        else:
            pytest.fail(f"Unexpected Drive operation: {arguments[0]}")
        return dict(returncode=0)

    def fake_call(_receipts: Path, _label: str, arguments: list[str], **_kwargs: Any) -> dict[str, Any]:
        assert arguments[0] == "upload"
        assert events[-2:] == [remote + "/shards", "inventory"]
        barrier.wait()
        source = Path(arguments[1])
        name = arguments[2].removeprefix(remote + "/")
        with lock:
            entries.append(dict(path=name, type="file", size_bytes=source.stat().st_size,
                                hashes=dict(md5=hashlib.md5(source.read_bytes()).hexdigest())))
        return dict(returncode=0)

    monkeypatch.setenv("RCLONE_CONFIG", "/explicit/test-config")
    monkeypatch.setattr(transfer, "load_distribution", lambda _: manifest)
    monkeypatch.setattr(transfer, "distribution_files", lambda *_args, **_kwargs: dict(files))
    monkeypatch.setattr(transfer, "checked", fake_checked)
    monkeypatch.setattr(transfer, "call", fake_call)
    args = SimpleNamespace(distribution=stage, source_store=tmp_path, drive_path=remote,
                           receipts=tmp_path / "receipts", workers=4, max_files=4, interrupt_largest=False)
    if duplicate:
        with pytest.raises(ValueError, match="Duplicate Drive paths"):
            transfer.transfer(args)
        assert all(entry["type"] != "file" for entry in entries)
    else:
        transfer.transfer(args)
        assert len([entry for entry in entries if entry["type"] == "file"]) == 4
