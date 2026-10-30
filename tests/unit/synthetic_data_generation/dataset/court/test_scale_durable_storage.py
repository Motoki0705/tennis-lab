from __future__ import annotations

import io
import tarfile
from types import SimpleNamespace

import pytest

from src.synthetic_data_generation.dataset.court.scale import durable_storage
from src.utils.data.image_record_store import file_sha256


def test_adapter_verifies_existing_file_without_overwriting(monkeypatch, tmp_path):
    adapter = object.__new__(durable_storage.SkillDriveFiles)
    adapter.drive_path = "data/synthetic_data_generation/court-scale/fixture"
    monkeypatch.setattr(adapter, "exists", lambda relative: True)
    calls = []

    def call(*args):
        calls.append(args)
        return {"matches": True} if args[0] == "verify" else {"id": "drive-file-id"}

    monkeypatch.setattr(adapter, "_call", call)
    source = tmp_path / "bundle.tar"
    source.write_bytes(b"fixture")
    receipt = adapter.put(source, "objects/fixture.tar")
    assert receipt["id"] == "drive-file-id" and receipt["verified"]
    assert [args[0] for args in calls] == ["verify", "inspect"]
    assert "--download" in calls[0]
    monkeypatch.setattr(adapter, "_call", lambda *args: {"matches": False})
    with pytest.raises(ValueError, match="conflict"):
        adapter.put(source, "objects/fixture.tar")


@pytest.mark.parametrize(
    "result",
    [
        {"status": "planned", "verified": False},
        {"status": "completed", "verified": False},
    ],
)
def test_unverified_transfer_cannot_acknowledge_a_cache(monkeypatch, tmp_path, result):
    adapter = object.__new__(durable_storage.SkillDriveFiles)
    adapter.drive_path = "data/synthetic_data_generation/court-scale/fixture"
    monkeypatch.setattr(adapter, "exists", lambda relative: False)
    monkeypatch.setattr(adapter, "_call", lambda *args: result)
    with pytest.raises(ValueError, match="not verified"):
        adapter.put(tmp_path / "source", "objects/fixture.tar")
    with pytest.raises(ValueError, match="not verified"):
        adapter.get("objects/fixture.tar", tmp_path / "destination")


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
