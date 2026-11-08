"""The CLI adapter must fail closed before declaring the old mount receipt complete."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path("knowledge/runs/run-ball-capacity-scaling-20261103/cli_adapter.py")
spec = importlib.util.spec_from_file_location("capacity_cli_adapter", SCRIPT)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


def dataset(tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    payload = b"independent expected payload\x00\x97"
    (target / "part.bin").write_bytes(payload)
    entries = [dict(path="part.bin", size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest())]
    raw = json.dumps(dict(schema_version="ball_frame_store_distribution.v1",
                         source=dict(version="ball-mix-v2-existing500-720p-20261018"),
                         files=entries)).encode()
    (target / "distribution.json").write_bytes(raw)
    return target, entries, raw


def test_alias_preserves_dedicated_backend_and_rejects_wrong_tree(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "alias"
    source.write_text("[gdrive]\ntype = drive\nclient_id = dedicated-test\ntoken = private-test\n")
    adapter.make_alias(source, destination)
    assert destination.stat().st_mode & 0o777 == 0o600
    assert "remote = gdrive:tennis_lab" in destination.read_text()
    destination.write_text(destination.read_text().replace("tennis_lab", "another_tree"))
    with pytest.raises(ValueError, match="original tree"):
        adapter.make_alias(source, destination)


@pytest.mark.parametrize("mutation", ["hash", "missing", "extra"])
def test_corruption_never_leaves_complete_receipt(tmp_path, mutation):
    target, entries, _ = dataset(tmp_path)
    receipts = tmp_path / "receipts"
    adapter.atomic_json(receipts / "progress.json", dict(complete=True))
    adapter.atomic_json(receipts / "files.json", ["old"])
    path = target / "part.bin"
    if mutation == "hash":
        path.write_bytes(b"!" * path.stat().st_size)
    elif mutation == "missing":
        path.unlink()
    else:
        (target / "unexpected").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="differs"):
        adapter.verify_dataset(target, entries, receipts,
                               dict(expected_files=1, expected_bytes=entries[0]["size_bytes"]),
                               workers=1)
    assert json.loads((receipts / "progress.json").read_text())["complete"] is False
    assert not (receipts / "files.json").exists()


def test_full_verification_has_both_digests_and_waits_for_remote_check(tmp_path):
    target, entries, _ = dataset(tmp_path)
    receipts = tmp_path / "receipts"
    state = adapter.verify_dataset(target, entries, receipts,
                                   dict(expected_files=1, expected_bytes=entries[0]["size_bytes"]),
                                   workers=1)
    assert state["verified_files"] == 1
    assert state["verified_bytes"] == len(b"independent expected payload\x00\x97")
    assert state["complete"] is False
    record = json.loads((receipts / "files.json").read_text())[0]
    assert record["sha256"] == record["independent_sha256"] == entries[0]["sha256"]


def test_distribution_pins_hash_and_rejects_duplicate_membership(tmp_path):
    _, entries, raw = dataset(tmp_path)
    kwargs = dict(expected_files=1, expected_bytes=entries[0]["size_bytes"],
                  expected_sha=hashlib.sha256(raw).hexdigest())
    assert adapter.distribution(raw, **kwargs) == entries
    with pytest.raises(ValueError, match="SHA256"):
        adapter.distribution(raw + b" ", **kwargs)
    value = json.loads(raw)
    value["files"] *= 2
    duplicated = json.dumps(value).encode()
    with pytest.raises(ValueError, match="membership"):
        adapter.distribution(duplicated, expected_files=2,
                             expected_bytes=2 * entries[0]["size_bytes"],
                             expected_sha=hashlib.sha256(duplicated).hexdigest())


def test_remote_inventory_rejects_duplicate_path(monkeypatch):
    response = [dict(Path="part.bin", IsDir=False, Size=1)] * 2
    monkeypatch.setattr(adapter, "rclone", lambda *args: json.dumps(response).encode())
    with pytest.raises(ValueError, match="Duplicate remote"):
        adapter.inventory(None, "mounted_drive:explicit")


def test_incomplete_expected_count_does_not_publish_files(tmp_path):
    target, entries, _ = dataset(tmp_path)
    with pytest.raises(ValueError, match="Incomplete"):
        adapter.verify_dataset(target, entries, tmp_path / "receipts",
                               dict(expected_files=2, expected_bytes=entries[0]["size_bytes"]),
                               workers=1)
    assert not (tmp_path / "receipts/files.json").exists()


@pytest.mark.parametrize("when", ["before", "after"])
def test_remote_discrepancy_cannot_complete_download(tmp_path, monkeypatch, when):
    original, entries, raw = dataset(tmp_path)
    workspace = tmp_path / "workspace"
    target = workspace / adapter.DATA
    target.parent.mkdir(parents=True)
    original.rename(target)
    monkeypatch.setattr(adapter, "distribution", lambda value: entries)
    monkeypatch.setattr(adapter, "EXPECTED_FILES", 1)
    monkeypatch.setattr(adapter, "EXPECTED_BYTES", entries[0]["size_bytes"])
    calls = []

    def remote(*args):
        calls.append(args[1])
        return raw if args[1] == "cat" else b""

    observed = {"part.bin": dict(Size=entries[0]["size_bytes"]),
                "distribution.json": dict(Size=len(raw))}
    inventories = [dict(observed)]
    if when == "before":
        inventories[0]["part.bin"] = dict(Size=0)
    else:
        inventories.append(dict(observed, unexpected=dict(Size=9)))
    monkeypatch.setattr(adapter, "rclone", remote)
    monkeypatch.setattr(adapter, "inventory", lambda *args: inventories.pop(0))
    with pytest.raises(ValueError, match="Remote dataset"):
        adapter.download(None, workspace, workers=1)
    state = json.loads((workspace / adapter.DOWNLOAD / "progress.json").read_text())
    assert state["complete"] is False
    if when == "before":
        assert "copy" not in calls
