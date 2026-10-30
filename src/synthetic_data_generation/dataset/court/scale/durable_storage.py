"""Verified immutable bundles through the tennis-drive skill's public CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
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
from .publication import read_json


class RemoteFiles(Protocol):
    """Transfer boundary used by CPU resume tests and the Drive adapter."""

    def exists(self, relative: str) -> bool: ...

    def put(self, local: Path, relative: str) -> dict[str, Any]: ...

    def get(self, relative: str, local: Path) -> None: ...


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
    """No overwrite, deletion, or independent rclone command construction."""

    def __init__(self, drive_path: str) -> None:
        self.drive_path = relative_path(drive_path)
        if not self.drive_path.startswith("data/synthetic_data_generation/court-scale/"):
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
        self.backend = core.RcloneBackend(
            remote_root=self.remote_root,
            executable=os.environ.get("RCLONE_BIN", "rclone"),
            config_path=Path(config),
        )

    def _path(self, relative: str) -> str:
        return f"{self.drive_path}/{relative_path(relative)}"

    def _call(self, *args: str) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--remote-root",
                self.remote_root,
                *args,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("Unexpected tennis-drive CLI result schema.")
        return value

    def exists(self, relative: str) -> bool:
        return self.backend.resolve(self._path(relative), must_exist=False) is not None

    def put(self, local: Path, relative: str) -> dict[str, Any]:
        path = self._path(relative)
        if self.exists(relative):
            result = self._call("verify", str(local), path, "--download")
            if result.get("matches") is not True:
                raise ValueError(
                    "Existing Drive bytes conflict with the immutable output."
                )
        else:
            result = self._call("upload", str(local), path, "--verify")
            if (
                result.get("status") != "completed"
                or result.get("verified") is not True
            ):
                raise ValueError("Drive upload was not verified.")
        info = self._call("inspect", path, "--metadata-only")
        if not info.get("id"):
            raise ValueError("Drive did not return an output file ID.")
        return {"path": path, "id": info["id"], "verified": True}

    def get(self, relative: str, local: Path) -> None:
        result = self._call("download", self._path(relative), str(local), "--verify")
        if result.get("status") != "completed" or result.get("verified") is not True:
            raise ValueError("Drive download was not verified.")


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
        if not self.remote.exists(relative):
            return None
        with tempfile.TemporaryDirectory(dir=self.scratch) as temporary:
            local = Path(temporary) / "receipt.json"
            self.remote.get(relative, local)
            value = read_json(local)
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
                self.restore(previous)
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
