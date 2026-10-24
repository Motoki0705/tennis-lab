from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.synthetic_data_generation.dataset.court.scale import durable_storage
from src.synthetic_data_generation.dataset.court.scale.drive_folders import DriveFolders
from src.utils.data.image_record_store import file_sha256


class Session:
    """ID-addressed Drive with duplicate names and a concurrent writer hook."""

    def __init__(self):
        self.entries = {}
        self.calls = []
        self.before_copy = lambda: None
        self.after_copy = lambda: None
        self.add("root", "tennis_lab")

    def add(self, parent, name, content=None):
        identifier = f"entry-{len(self.entries)}"
        self.entries[identifier] = {
            "ID": identifier,
            "Name": name,
            "IsDir": content is None,
            "parent": parent,
            "content": content,
        }
        return identifier

    def call(self, endpoint, **params):
        self.calls.append((endpoint, params))
        if endpoint == "operations/list":
            parent = params["fs"].get("root_folder_id", "root")
            return {
                "list": [
                    {
                        **v,
                        **(
                            {
                                "Size": len(v["content"]),
                                "Hashes": {
                                    "md5": hashlib.md5(v["content"]).hexdigest()
                                },
                            }
                            if not v["IsDir"]
                            else {}
                        ),
                    }
                    for v in self.entries.values()
                    if v["parent"] == parent
                ]
            }
        if endpoint == "operations/mkdir":
            self.add(params["fs"]["root_folder_id"], params["remote"])
            return {}
        assert endpoint == "operations/copyfile"
        assert params["_config"] == {
            "Immutable": True,
            "IgnoreExisting": True,
            "IgnoreChecksum": True,
            "IgnoreSize": True,
        }
        self.before_copy()
        if isinstance(params["srcFs"], dict):
            parent = params["srcFs"]["root_folder_id"]
            value = next(
                v
                for v in self.entries.values()
                if v["parent"] == parent and v["Name"] == params["srcRemote"]
            )
            (Path(params["dstFs"]) / params["dstRemote"]).write_bytes(value["content"])
        else:
            parent = params["dstFs"]["root_folder_id"]
            content = (Path(params["srcFs"]) / params["srcRemote"]).read_bytes()
            existing = [
                v
                for v in self.entries.values()
                if v["parent"] == parent and v["Name"] == params["dstRemote"]
            ]
            if not existing:
                self.add(parent, params["dstRemote"], content)
        self.after_copy()
        return {}

    def close(self):
        pass


@pytest.fixture
def adapter():
    adapter = object.__new__(durable_storage.SkillDriveFiles)
    adapter.drive_path = "data/synthetic_data_generation/court-scale/fixture"
    adapter.session = Session()
    adapter.folders = DriveFolders(adapter.session, "gdrive:tennis_lab")
    yield adapter
    adapter.close()


def test_adapter_verifies_existing_file_without_overwriting(adapter, tmp_path):
    source = tmp_path / "bundle.tar"
    source.write_bytes(b"fixture")
    receipt = adapter.put(source, "objects/fixture.tar")
    assert receipt["id"] and receipt["verified"]
    adapter.session.calls.clear()
    assert adapter.put(source, "objects/fixture.tar") == receipt
    assert all(op != "operations/copyfile" for op, _ in adapter.session.calls)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="conflict"):
        adapter.put(source, "objects/fixture.tar")


@pytest.mark.parametrize("position", ["leaf", "ancestor", "project_root"])
def test_cached_folder_ids_never_hide_concurrent_duplicates(
    adapter, tmp_path, position
):
    source = tmp_path / "bundle"
    source.write_bytes(b"fixture")
    adapter.put(source, "objects/one.tar")
    entries = adapter.session.entries
    name = {"leaf": "one.tar", "ancestor": "fixture", "project_root": "tennis_lab"}[
        position
    ]
    entry = next(v for v in entries.values() if v["Name"] == name)
    adapter.session.add(entry["parent"], name, entry["content"])
    adapter.session.calls.clear()
    with pytest.raises(ValueError, match="Ambiguous"):
        adapter.put(
            source, "objects/two.tar" if position != "leaf" else "objects/one.tar"
        )
    assert all(op != "operations/copyfile" for op, _ in adapter.session.calls)


@pytest.mark.parametrize("change", ["id", "missing", "file"])
def test_cached_parent_replacement_stops_before_transfer(adapter, tmp_path, change):
    source = tmp_path / "bundle"
    source.write_bytes(b"fixture")
    adapter.put(source, "objects/one.tar")
    entries = adapter.session.entries
    entry = next(v for v in entries.values() if v["Name"] == "fixture")
    del entries[entry["ID"]]
    if change != "missing":
        replacement = adapter.session.add(entry["parent"], "fixture")
        entries[replacement]["IsDir"] = change != "file"
        if change == "file":
            entries[replacement]["content"] = b"file"
    adapter.session.calls.clear()
    with pytest.raises(ValueError, match="ID changed|disappeared|not a directory"):
        adapter.put(source, "objects/two.tar")
    assert all(op != "operations/copyfile" for op, _ in adapter.session.calls)


def test_absence_is_never_cached_and_competing_file_is_not_overwritten(
    adapter, tmp_path
):
    source = tmp_path / "bundle"
    source.write_bytes(b"fixture")
    assert not adapter.exists("objects/one.tar")
    adapter.put(source, "objects/two.tar")
    parent = next(
        v["ID"] for v in adapter.session.entries.values() if v["Name"] == "objects"
    )
    competitor = adapter.session.add(parent, "one.tar", b"other")
    with pytest.raises(ValueError, match="conflict"):
        adapter.put(source, "objects/one.tar")
    assert adapter.session.entries[competitor]["content"] == b"other"


def test_duplicate_created_during_upload_is_not_acknowledged(adapter, tmp_path):
    source = tmp_path / "bundle"
    source.write_bytes(b"fixture")

    def duplicate():
        parent = next(
            v["ID"] for v in adapter.session.entries.values() if v["Name"] == "objects"
        )
        adapter.session.add(parent, "one.tar", b"fixture")

    adapter.session.after_copy = duplicate
    with pytest.raises(ValueError, match="Ambiguous"):
        adapter.put(source, "objects/one.tar")


def test_competing_file_created_between_preflight_and_copy_is_preserved(
    adapter, tmp_path
):
    source = tmp_path / "bundle"
    source.write_bytes(b"fixture")

    def competing_writer():
        parent = next(
            v["ID"] for v in adapter.session.entries.values() if v["Name"] == "objects"
        )
        adapter.session.add(parent, "one.tar", b"other")

    adapter.session.before_copy = competing_writer
    with pytest.raises(ValueError, match="conflict"):
        adapter.put(source, "objects/one.tar")
    assert (
        next(v for v in adapter.session.entries.values() if v["Name"] == "one.tar")[
            "content"
        ]
        == b"other"
    )


@pytest.mark.parametrize("failure", ["hash", "missing_hash", "id", "transfer"])
def test_unverified_transfer_cannot_acknowledge_a_cache(adapter, tmp_path, failure):
    source = tmp_path / "bundle"
    source.write_bytes(b"fixture")
    if failure == "transfer":

        def fail():
            raise RuntimeError("transfer failed")

        adapter.session.before_copy = fail
    else:

        def damage():
            entry = next(
                v for v in adapter.session.entries.values() if v["Name"] == "one.tar"
            )
            if failure == "hash":
                entry["content"] = b"corrupt"
            elif failure == "id":
                entry["ID"] = None
            else:
                original = adapter.session.call

                def no_hash(endpoint, **params):
                    result = original(endpoint, **params)
                    for value in result.get("list", []):
                        value.pop("Hashes", None)
                    return result

                adapter.session.call = no_hash

        adapter.session.after_copy = damage
    with pytest.raises(
        (ValueError, RuntimeError), match="conflict|hash|ID|transfer failed"
    ):
        adapter.put(source, "objects/one.tar")


def test_optional_get_distinguishes_missing_and_propagates_errors(adapter, tmp_path):
    assert not adapter.get("objects/missing.tar", tmp_path / "absent", missing_ok=True)
    source = tmp_path / "bundle"
    source.write_bytes(b"fixture")
    adapter.put(source, "objects/one.tar")

    def fail():
        raise RuntimeError("network failure")

    adapter.session.before_copy = fail
    with pytest.raises(RuntimeError, match="network failure"):
        adapter.get("objects/one.tar", tmp_path / "failed", missing_ok=True)


def test_download_detects_remote_change_and_refuses_local_overwrite(adapter, tmp_path):
    source = tmp_path / "bundle"
    source.write_bytes(b"fixture")
    adapter.put(source, "objects/one.tar")
    with pytest.raises(ValueError, match="must be new"):
        adapter.get("objects/one.tar", source)

    def change():
        entry = next(
            v for v in adapter.session.entries.values() if v["Name"] == "one.tar"
        )
        entry["content"] = b"changed"

    adapter.session.after_copy = change
    with pytest.raises(ValueError, match="conflict"):
        adapter.get("objects/one.tar", tmp_path / "download")


def test_new_cache_and_restore_have_bounded_calls_without_extra_processes(
    adapter, tmp_path
):
    (tmp_path / "data").write_bytes(b"fixture")
    store = durable_storage.BundleStore(tmp_path, adapter)
    adapter.folders.lookup(
        adapter._path("receipts/caches/warm.json"), create_parents=True
    )
    adapter.folders.lookup(adapter._path("objects/warm.tar"), create_parents=True)
    adapter.session.calls.clear()
    assert store.receipt("receipts/caches/one.json") is None
    receipt = store.publish(
        "receipts/caches/one.json", ["data"], identity={"kind": "cache"}
    )
    assert len(adapter.session.calls) <= 65
    assert sum(op == "operations/copyfile" for op, _ in adapter.session.calls) == 3
    adapter.session.calls.clear()
    restored = durable_storage.BundleStore(tmp_path / "restored", adapter)
    assert restored.receipt("receipts/caches/one.json") == receipt
    restored.restore(receipt)
    assert len(adapter.session.calls) <= 32
    assert (tmp_path / "restored/data").read_bytes() == b"fixture"


def test_republishing_verified_inputs_does_not_download_archive_again(
    adapter, tmp_path
):
    (tmp_path / "plan").write_bytes(b"frozen input")
    store = durable_storage.BundleStore(tmp_path, adapter)
    receipt = store.publish(
        "receipts/request.json", ["plan"], identity={"kind": "inputs"}
    )
    restored = durable_storage.BundleStore(tmp_path / "restored", adapter)
    restored.restore(receipt)
    adapter.session.calls.clear()
    assert (
        restored.publish("receipts/request.json", ["plan"], identity={"kind": "inputs"})
        == receipt
    )
    transfers = [
        params for op, params in adapter.session.calls if op == "operations/copyfile"
    ]
    assert len(transfers) == 1 and transfers[0]["srcRemote"] == "request.json"
    next(v for v in adapter.session.entries.values() if v["Name"].endswith(".tar"))[
        "content"
    ] = b"bad"
    with pytest.raises(ValueError, match="conflict"):
        restored.publish("receipts/request.json", ["plan"], identity={"kind": "inputs"})


@pytest.mark.parametrize("kind", ["parent", "symlink", "duplicate", "extra"])
def test_restore_rejects_unsafe_or_undeclared_archive_members(tmp_path, kind):
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w") as stream:
        names = ["../escape"] if kind == "parent" else ["expected"]
        if kind == "duplicate":
            names.append("expected")
        if kind == "extra":
            names.append("unexpected")
        for name in names:
            info = tarfile.TarInfo(name)
            if kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, "../escape"
            stream.addfile(info)
    archive = tmp_path / "source.tar"
    archive.write_bytes(content.getvalue())
    remote = SimpleNamespace(
        get=lambda relative, local: local.write_bytes(content.getvalue())
    )
    root = tmp_path / "consumer"
    store = durable_storage.BundleStore(root, remote)
    sha = file_sha256(archive)
    with pytest.raises(ValueError, match="relative POSIX|unexpected files"):
        store.restore(
            {
                "schema": "court_scale_durable_bundle_v1",
                "archive_path": f"objects/{sha}.tar",
                "archive_sha256": sha,
                "archive_bytes": archive.stat().st_size,
                "files": ["../escape"] if kind == "parent" else ["expected"],
            }
        )
    assert not (tmp_path / "escape").exists()
    assert not (root / "expected").exists()
