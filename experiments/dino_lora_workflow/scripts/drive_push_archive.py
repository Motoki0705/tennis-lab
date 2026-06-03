"""Overview:
Push a local workflow `.tar.zst` archive to a Drive path or rclone remote.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/drive_push_archive.py local_archive=data/dino_workflow/archives/guardrail_current.tar.zst drive_archive=/content/drive/MyDrive/tennis_lab/data/dino_workflow/guardrail_current.tar.zst
    .venv/bin/python experiments/dino_lora_workflow/scripts/drive_push_archive.py local_archive=data/dino_workflow/archives/guardrail_current.tar.zst drive_archive=google:tennis_lab/data/dino_workflow/guardrail_current.tar.zst

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/drive_push_archive.yaml`.
    - Plain filesystem paths use `shutil.copy2`; `name:path` locations use `rclone copyto`.
    - A sidecar `<archive>.transfer.json` is written next to the copied archive by default.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


REMOTE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+:.+")
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArchiveLocation:
    """A local filesystem path or an rclone remote path."""

    original: str
    path: str
    is_remote: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_rclone_remote(value: str) -> bool:
    return REMOTE_PATTERN.match(value) is not None


def resolve_location(value: Any) -> ArchiveLocation:
    text = str(value).strip()
    if not text:
        raise ValueError("Archive path cannot be empty")
    validate_tar_zst_path(text)
    if is_rclone_remote(text):
        return ArchiveLocation(original=text, path=text, is_remote=True)
    return ArchiveLocation(original=text, path=str(Path(to_absolute_path(text)).resolve()), is_remote=False)


def validate_tar_zst_path(path: str) -> None:
    if not path.endswith(".tar.zst"):
        raise ValueError(f"Archive path must end with .tar.zst: {path}")


def sidecar_path(location: ArchiveLocation) -> str:
    return f"{location.path}.transfer.json"


def local_path(location: ArchiveLocation) -> Path:
    if location.is_remote:
        raise ValueError(f"Expected local path, got rclone remote: {location.path}")
    return Path(location.path)


def remote_parent(remote_path: str) -> str:
    remote_name, remote_item = remote_path.split(":", 1)
    parent = remote_item.rsplit("/", 1)[0] if "/" in remote_item else ""
    return f"{remote_name}:{parent}" if parent else f"{remote_name}:"


def local_parent(location: ArchiveLocation) -> Path:
    return local_path(location).parent


def resolve_rclone_binary(configured_binary: str) -> str:
    if configured_binary and configured_binary != "auto":
        resolved = shutil.which(configured_binary) if os.sep not in configured_binary else configured_binary
    else:
        resolved = shutil.which("rclone")
    if resolved is None:
        raise RuntimeError("rclone remote was requested, but `rclone` was not found on PATH")
    if os.sep in resolved and not Path(resolved).exists():
        raise RuntimeError(f"Configured rclone binary was not found: {resolved}")
    return resolved


def run_rclone(command: list[str], *, rclone_binary: str) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        [rclone_binary, *command],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        detail = stderr or stdout or "no output"
        raise RuntimeError(f"rclone {' '.join(command)} failed with code {process.returncode}: {detail}")
    return process


def run_rclone_json(command: list[str], *, rclone_binary: str) -> Any:
    process = run_rclone(command, rclone_binary=rclone_binary)
    payload = process.stdout.strip()
    return json.loads(payload) if payload else None


def remote_exists(remote_path: str, *, rclone_binary: str) -> bool:
    process = subprocess.run(
        [rclone_binary, "lsjson", "--stat", remote_path],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.returncode == 0


def remote_size(remote_path: str, *, rclone_binary: str) -> int:
    payload = run_rclone_json(["size", "--json", remote_path], rclone_binary=rclone_binary)
    if not isinstance(payload, dict) or "bytes" not in payload:
        raise RuntimeError(f"Could not read remote size for {remote_path}")
    return int(payload["bytes"])


def local_exists(location: ArchiveLocation) -> bool:
    return local_path(location).exists()


def destination_exists(location: ArchiveLocation, *, backend: str, rclone_binary: str | None) -> bool:
    if location.is_remote:
        if backend != "rclone" or rclone_binary is None:
            raise RuntimeError("Internal error: remote destination without rclone backend")
        return remote_exists(location.path, rclone_binary=rclone_binary)
    return local_exists(location)


def location_size(location: ArchiveLocation, *, rclone_binary: str | None) -> int | None:
    if location.is_remote:
        if rclone_binary is None:
            return None
        return remote_size(location.path, rclone_binary=rclone_binary)
    path = local_path(location)
    return path.stat().st_size if path.exists() else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_sha256(location: ArchiveLocation) -> str | None:
    if location.is_remote:
        return None
    path = local_path(location)
    return sha256_file(path) if path.exists() else None


def source_sha256_for_manifest(source: ArchiveLocation, destination: ArchiveLocation) -> str | None:
    source_hash = local_sha256(source)
    if source_hash is not None:
        return source_hash
    return local_sha256(destination)


def ensure_source_exists(source: ArchiveLocation, *, backend: str, rclone_binary: str | None) -> None:
    if source.is_remote:
        if backend != "rclone" or rclone_binary is None:
            raise RuntimeError("Internal error: remote source without rclone backend")
        if not remote_exists(source.path, rclone_binary=rclone_binary):
            raise FileNotFoundError(f"Remote source archive not found: {source.path}")
        return
    path = local_path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Local source archive not found: {path}")


def ensure_destination_parent(destination: ArchiveLocation, *, backend: str, rclone_binary: str | None, dry_run: bool) -> None:
    if destination.is_remote:
        if backend != "rclone" or rclone_binary is None:
            raise RuntimeError("Internal error: remote destination without rclone backend")
        if not dry_run:
            run_rclone(["mkdir", remote_parent(destination.path)], rclone_binary=rclone_binary)
        return
    if not dry_run:
        local_parent(destination).mkdir(parents=True, exist_ok=True)


def choose_backend(source: ArchiveLocation, destination: ArchiveLocation, configured_rclone_binary: str) -> tuple[str, str | None]:
    if source.is_remote or destination.is_remote:
        return "rclone", resolve_rclone_binary(configured_rclone_binary)
    return "local-shutil", None


def copy_archive(
    *,
    source: ArchiveLocation,
    destination: ArchiveLocation,
    backend: str,
    rclone_binary: str | None,
    overwrite: bool,
) -> None:
    if backend == "rclone":
        if rclone_binary is None:
            raise RuntimeError("Internal error: rclone backend without binary")
        run_rclone(["copyto", source.path, destination.path], rclone_binary=rclone_binary)
        return

    destination_path = local_path(destination)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Destination archive already exists and overwrite=false: {destination_path}")
    shutil.copy2(local_path(source), destination_path)


def verify_copy(
    *,
    source: ArchiveLocation,
    destination: ArchiveLocation,
    rclone_binary: str | None,
) -> dict[str, Any]:
    source_size = location_size(source, rclone_binary=rclone_binary)
    destination_size = location_size(destination, rclone_binary=rclone_binary)
    size_match = source_size is not None and destination_size is not None and source_size == destination_size

    source_hash = local_sha256(source)
    destination_hash = local_sha256(destination)
    sha256_match = None
    if source_hash is not None and destination_hash is not None:
        sha256_match = source_hash == destination_hash

    verified = size_match if sha256_match is None else size_match and sha256_match
    return {
        "status": "ok" if verified else "failed",
        "verified": verified,
        "source_size": source_size,
        "destination_size": destination_size,
        "size_match": size_match,
        "source_sha256": source_hash,
        "destination_sha256": destination_hash,
        "sha256_match": sha256_match,
    }


def transfer_manifest_payload(
    *,
    direction: str,
    source: ArchiveLocation,
    destination: ArchiveLocation,
    backend: str,
    copied_at: str,
    source_size: int | None,
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    size_bytes = None
    if verification is not None:
        size_bytes = verification.get("destination_size") or verification.get("source_size")
    if size_bytes is None:
        size_bytes = source_size
    if size_bytes is None and not destination.is_remote:
        size_bytes = location_size(destination, rclone_binary=None)
    payload: dict[str, Any] = {
        "direction": direction,
        "source": source.path,
        "destination": destination.path,
        "size_bytes": size_bytes,
        "sha256": source_sha256_for_manifest(source, destination),
        "copied_at": copied_at,
        "backend": backend,
        "verified": verification["verified"] if verification is not None else None,
    }
    if verification is not None:
        payload["verification"] = verification
    return payload


def write_transfer_manifest(
    *,
    destination: ArchiveLocation,
    manifest: dict[str, Any],
    backend: str,
    rclone_binary: str | None,
) -> str:
    payload = json.dumps(manifest, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    output_path = sidecar_path(destination)
    if destination.is_remote:
        if backend != "rclone" or rclone_binary is None:
            raise RuntimeError("Internal error: remote sidecar without rclone backend")
        with tempfile.NamedTemporaryFile(prefix="dino-transfer-", suffix=".json", delete=False) as tmp_file:
            tmp_file.write(payload)
            tmp_path = Path(tmp_file.name)
        try:
            run_rclone(["copyto", str(tmp_path), output_path], rclone_binary=rclone_binary)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        Path(output_path).write_bytes(payload)
    return output_path


def transfer_archive(cfg: DictConfig, *, direction: str) -> dict[str, Any]:
    source = resolve_location(cfg.local_archive if direction == "push" else cfg.drive_archive)
    destination = resolve_location(cfg.drive_archive if direction == "push" else cfg.local_archive)
    overwrite = bool(cfg.overwrite)
    dry_run = bool(cfg.dry_run)

    backend, rclone_binary = choose_backend(source, destination, str(cfg.rclone_binary))
    ensure_source_exists(source, backend=backend, rclone_binary=rclone_binary)
    destination_already_exists = destination_exists(destination, backend=backend, rclone_binary=rclone_binary)
    if destination_already_exists and not overwrite:
        raise FileExistsError(f"Destination archive already exists and overwrite=false: {destination.path}")
    ensure_destination_parent(destination, backend=backend, rclone_binary=rclone_binary, dry_run=dry_run)

    source_size = location_size(source, rclone_binary=rclone_binary)
    source_sha256 = local_sha256(source)
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "direction": direction,
        "source": source.path,
        "destination": destination.path,
        "source_is_remote": source.is_remote,
        "destination_is_remote": destination.is_remote,
        "backend": backend,
        "overwrite": overwrite,
        "verify_after_copy": bool(cfg.verify_after_copy),
        "write_transfer_manifest": bool(cfg.write_transfer_manifest),
        "destination_already_exists": destination_already_exists,
        "source_size": source_size,
        "source_sha256": source_sha256,
        "sidecar_path": sidecar_path(destination),
    }
    if dry_run:
        summary["planned_action"] = "copy"
        summary["sidecar_would_be_written"] = bool(cfg.write_transfer_manifest)
        return summary

    copy_archive(
        source=source,
        destination=destination,
        backend=backend,
        rclone_binary=rclone_binary,
        overwrite=overwrite,
    )

    verification = None
    if bool(cfg.verify_after_copy):
        verification = verify_copy(source=source, destination=destination, rclone_binary=rclone_binary)
        if not verification["verified"]:
            raise RuntimeError(f"Archive copy verification failed: {verification}")

    manifest_path = None
    copied_at = now_iso()
    if bool(cfg.write_transfer_manifest):
        manifest = transfer_manifest_payload(
            direction=direction,
            source=source,
            destination=destination,
            backend=backend,
            copied_at=copied_at,
            source_size=source_size,
            verification=verification,
        )
        manifest_path = write_transfer_manifest(
            destination=destination,
            manifest=manifest,
            backend=backend,
            rclone_binary=rclone_binary,
        )

    summary["copied_at"] = copied_at
    summary["destination_size"] = location_size(destination, rclone_binary=rclone_binary)
    summary["destination_sha256"] = local_sha256(destination)
    summary["verification"] = verification
    summary["transfer_manifest"] = manifest_path
    return summary


def drive_push_archive(cfg: DictConfig) -> dict[str, Any]:
    return transfer_archive(cfg, direction="push")


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="drive_push_archive",
)
def main(cfg: DictConfig) -> None:
    summary = drive_push_archive(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
