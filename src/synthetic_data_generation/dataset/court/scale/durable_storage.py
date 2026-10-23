"""Verified immutable bundles through the tennis-drive skill's session library."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from src.utils.data.image_record_store import file_sha256
from src.utils.paths import PROJECT_ROOT

from .core import canonical_bytes
from .drive_folders import DriveFolders
from .publication import read_json


class RemoteFiles(Protocol):
    """Transfer boundary used by CPU resume tests and the Drive adapter."""

    def exists(self, relative: str) -> bool: ...

    def put(self, local: Path, relative: str) -> dict[str, Any]: ...

    def get(self, relative: str, local: Path, *, missing_ok: bool = False) -> bool: ...

    def verify(self, local: Path, relative: str) -> dict[str, Any]: ...


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(p in {"", ".", ".."} for p in value.split("/"))
        or any(c in value for c in ("\\", ":", "\n", "\r"))
    ):
        raise ValueError("Expected a normalized relative POSIX path.")
    return value


class SkillDriveFiles:
    """Immutable files over one skill-library session, scoped to one attempt."""

    def __init__(self, drive_path: str) -> None:
        self.drive_path = relative_path(drive_path)
        if not self.drive_path.startswith(
            "data/synthetic_data_generation/court-scale/"
        ):
            raise ValueError(
                "Choose a new generated dataset path under data/synthetic_data_generation/court-scale/."
            )
        config = os.environ.get("RCLONE_CONFIG")
        if not config:
            raise ValueError("Set RCLONE_CONFIG explicitly before using Drive.")
        self.script = PROJECT_ROOT / ".agents/skills/tennis-drive/scripts/drive.py"
        # resolve() is the skill library's duplicate-name-aware, optional lookup.
        name = "_court_scale_drive_core"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, self.script.with_name("drive_core.py")
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        core = sys.modules[name]
        self.remote_root = os.environ.get(
            "TENNIS_LAB_DRIVE_REMOTE", core.DEFAULT_REMOTE_ROOT
        )
        backend = core.RcloneBackend(
            remote_root=self.remote_root,
            executable=os.environ.get("RCLONE_BIN", "rclone"),
            config_path=Path(config),
        )
        self.session = core.RcloneSession(backend)
        self.folders = DriveFolders(self.session, self.remote_root)

    def __enter__(self) -> SkillDriveFiles:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.folders.close()
        finally:
            self.session.close()

    def _path(self, relative: str) -> str:
        return f"{self.drive_path}/{relative_path(relative)}"

    def exists(self, relative: str) -> bool:
        found = self.folders.lookup(self._path(relative))
        return found is not None and found[1] is not None

    @staticmethod
    def _content(local: Path) -> tuple[int, str]:
        if local.is_symlink() or not local.is_file():
            raise ValueError("Drive transfer requires an ordinary local file.")
        digest = hashlib.md5(usedforsecurity=False)
        with local.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return local.stat().st_size, digest.hexdigest()

    @staticmethod
    def _check(entry: dict[str, Any] | None, content: tuple[int, str]) -> None:
        if entry is None or entry.get("IsDir") or not entry.get("ID"):
            raise ValueError("Drive file is absent or has no file identity.")
        hashes = {k.lower(): v.lower() for k, v in entry.get("Hashes", {}).items()}
        if not hashes.get("md5"):
            raise ValueError("Drive file has no content hash; cannot acknowledge it.")
        if (entry.get("Size"), hashes["md5"]) != content:
            raise ValueError(
                "Drive bytes conflict with the immutable output (size/hash)."
            )

    def verify(self, local: Path, relative: str) -> dict[str, Any]:
        content = self._content(local)
        found = self.folders.lookup(self._path(relative))
        entry = found[1] if found else None
        self._check(entry, content)
        assert entry is not None
        return {"path": self._path(relative), "id": entry["ID"], "verified": True}

    def put(self, local: Path, relative: str) -> dict[str, Any]:
        content = self._content(local)
        found = self.folders.lookup(self._path(relative), create_parents=True)
        assert found is not None
        parent_fs, before = found
        if before is not None:
            self._check(before, content)
        else:
            self.session.call(
                "operations/copyfile",
                srcFs=str(local.resolve().parent),
                srcRemote=local.name,
                dstFs=parent_fs,
                dstRemote=PurePosixPath(relative).name,
                # IgnoreExisting protects the preflight-to-copy race. Success
                # alone is not acknowledgement: the fresh readback below must
                # still match. Immutable alone does not guard RC copyfile.
                # Rclone's post-copy mismatch cleanup deletes its destination.
                # Keep failed objects for diagnosis; our readback owns both checks.
                _config={
                    "Immutable": True,
                    "IgnoreExisting": True,
                    "IgnoreChecksum": True,
                    "IgnoreSize": True,
                },
            )
        result = self.verify(local, relative)
        if self._content(local) != content:
            raise ValueError("Local file changed during immutable put.")
        if before is not None and result["id"] != before["ID"]:
            raise ValueError("Drive file ID changed during immutable put.")
        return result

    def get(self, relative: str, local: Path, *, missing_ok: bool = False) -> bool:
        if local.exists() or local.is_symlink():
            raise ValueError("Drive download destination must be new.")
        found = self.folders.lookup(self._path(relative))
        if found is None or found[1] is None:
            if missing_ok:
                return False
            raise ValueError("Drive file does not exist.")
        parent_fs, before = found
        assert before is not None
        self.session.call(
            "operations/copyfile",
            srcFs=parent_fs,
            srcRemote=PurePosixPath(relative).name,
            dstFs=str(local.absolute().parent),
            dstRemote=local.name,
            _config={
                "Immutable": True,
                "IgnoreExisting": True,
                "IgnoreChecksum": True,
                "IgnoreSize": True,
            },
        )
        self._check(before, self._content(local))
        after = self.verify(local, relative)
        if after["id"] != before["ID"]:
            raise ValueError("Drive file ID changed during download.")
        return True


def tree_files(root: Path, directory: Path) -> list[str]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks cannot be saved in durable bundles.")
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("Only ordinary files/directories can be saved.")
    return files


class BundleStore:
    """An archive is committed only by its last, verified receipt.

    Unreferenced objects left by interrupted transfers are retained and ignored.
    Each receipt pins archive SHA-256, size, exact file names and Drive identity.
    """

    def __init__(self, root: Path, remote: RemoteFiles) -> None:
        self.root, self.remote = root, remote
        self.scratch = root / ".durability"
        self.scratch.mkdir(parents=True, exist_ok=True)

    def receipt(self, relative: str) -> dict[str, Any] | None:
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            local = Path(temporary) / "receipt.json"
            if not self.remote.get(relative, local, missing_ok=True):
                return None
            value: dict[str, Any] = read_json(local)
        if value.get("schema") != "court_scale_durable_bundle_v1":
            raise ValueError("Unknown durable bundle receipt schema.")
        return value

    def publish(
        self, receipt_path: str, files: Iterable[str], *, identity: dict[str, Any]
    ) -> dict[str, Any]:
        names = sorted(set(relative_path(name) for name in files))
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            archive = Path(temporary) / "bundle.tar"
            with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as stream:
                for name in names:
                    path = self.root / name
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or not path.resolve().is_relative_to(self.root.resolve())
                    ):
                        raise ValueError(
                            "Bundle input is not an ordinary contained file."
                        )
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = path.stat().st_size, 0o600
                    with path.open("rb") as source:
                        stream.addfile(info, source)
            sha = file_sha256(archive)
            value = {
                "schema": "court_scale_durable_bundle_v1",
                **identity,
                "archive_path": f"objects/{sha}.tar",
                "archive_sha256": sha,
                "archive_bytes": archive.stat().st_size,
                "files": names,
            }
            previous = self.receipt(receipt_path)
            if previous is not None:
                if any(previous.get(k) != v for k, v in value.items()):
                    raise ValueError(
                        "Existing durable receipt conflicts with local bytes."
                    )
                # The reconstructed archive already matches the pinned SHA-256.
                # Verify remote content afresh without downloading it a second time.
                self.remote.verify(archive, value["archive_path"])
                return previous
            value["archive_drive"] = self.remote.put(archive, value["archive_path"])
            value["verified_at"] = datetime.now(UTC).isoformat()
            local_receipt = Path(temporary) / "receipt.json"
            local_receipt.write_bytes(canonical_bytes(value))
            self.remote.put(local_receipt, receipt_path)
            if self.receipt(receipt_path) != value:
                raise ValueError(
                    "Durable receipt readback disagrees with the publication."
                )
        return value

    def restore(self, receipt: dict[str, Any]) -> None:
        if receipt.get("schema") != "court_scale_durable_bundle_v1":
            raise ValueError("Unknown durable bundle receipt schema.")
        sha = receipt["archive_sha256"]
        if receipt["archive_path"] != f"objects/{sha}.tar":
            raise ValueError("Durable archive path disagrees with its digest.")
        names = receipt["files"]
        if names != sorted(set(relative_path(name) for name in names)):
            raise ValueError("Invalid durable bundle file inventory.")
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            archive = Path(temporary) / "bundle.tar"
            self.remote.get(receipt["archive_path"], archive)
            if (
                archive.stat().st_size != receipt["archive_bytes"]
                or file_sha256(archive) != sha
            ):
                raise ValueError("Drive archive size/hash mismatch.")
            extracted = Path(temporary) / "extracted"
            with tarfile.open(archive) as stream:
                members = stream.getmembers()
                if [m.name for m in members] != names or any(
                    not m.isfile() for m in members
                ):
                    raise ValueError(
                        "Durable archive contains unexpected files or links."
                    )
                for member in members:
                    target = extracted / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = stream.extractfile(member)
                    assert source is not None
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
            # Check every collision before publishing any extracted file.
            for name in names:
                target = self.root / name
                if (
                    not target.resolve().is_relative_to(self.root.resolve())
                    or target.is_symlink()
                ):
                    raise ValueError("Durable output escapes its explicit root.")
                if target.exists() and (
                    not target.is_file()
                    or file_sha256(target) != file_sha256(extracted / name)
                ):
                    raise ValueError(
                        "Local bytes conflict with the durable bundle; preserve them in a new root."
                    )
            for name in names:
                target = self.root / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (extracted / name).rename(target)
