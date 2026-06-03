"""Overview:
Unpack `.tar.zst` workflow archives safely and optionally verify the embedded manifest.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/archive_unpack.py input_archive=data/dino_workflow/archives/guardrail_current.tar.zst output_dir=/content/guardrail
    .venv/bin/python experiments/dino_lora_workflow/scripts/archive_unpack.py dry_run=true input_archive=outputs/tmp/example.tar.zst output_dir=outputs/tmp/unpacked

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/archive_unpack.yaml`.
    - Path traversal is blocked when `safe_extract=true`, which is the default.
    - If `archive_manifest.json` is present and `verify_manifest=true`, extracted file size and SHA-256 hashes are checked.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def validate_tar_zst_path(path: Path) -> None:
    if not str(path).endswith(".tar.zst"):
        raise ValueError(f"Archive path must end with .tar.zst: {path}")


def zstd_module() -> Any | None:
    try:
        return importlib.import_module("zstandard")
    except ModuleNotFoundError:
        return None


def resolve_zstd_binary(configured_binary: str) -> str:
    if configured_binary and configured_binary != "auto":
        return configured_binary
    resolved = shutil.which("zstd")
    if resolved is None:
        raise RuntimeError(
            "zstandard package is not installed and `zstd` CLI was not found. "
            "Install the Python `zstandard` package or make the `zstd` binary available."
        )
    return resolved


def strip_member_name(name: str, strip_components: int, safe_extract: bool) -> str | None:
    if strip_components < 0:
        raise ValueError("strip_components must be non-negative")

    path = PurePosixPath(name)
    if safe_extract and path.is_absolute():
        raise ValueError(f"Refusing to extract absolute archive path: {name}")

    parts = [part for part in path.parts if part not in {"", "."}]
    if safe_extract and any(part == ".." for part in parts):
        raise ValueError(f"Refusing to extract path traversal member: {name}")
    if len(parts) <= strip_components:
        return None
    return PurePosixPath(*parts[strip_components:]).as_posix()


def ensure_safe_destination(output_dir: Path, relative_path: str, safe_extract: bool) -> Path:
    destination = output_dir / relative_path
    if safe_extract:
        output_root = output_dir.resolve()
        resolved_destination = destination.resolve()
        if resolved_destination != output_root and not resolved_destination.is_relative_to(output_root):
            raise ValueError(f"Refusing to extract outside output_dir: {relative_path}")
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_decompressed_tar_with_python_zstandard(input_archive: Path) -> tarfile.TarFile:
    zstd = zstd_module()
    if zstd is None:
        raise RuntimeError("Python zstandard package is unavailable")
    input_file = input_archive.open("rb")
    reader = zstd.ZstdDecompressor().stream_reader(input_file)
    tar = tarfile.open(fileobj=reader, mode="r|")
    tar._dino_workflow_context = (input_file, reader)  # type: ignore[attr-defined]
    return tar


def close_python_zstandard_tar(tar: tarfile.TarFile) -> None:
    context = getattr(tar, "_dino_workflow_context", None)
    tar.close()
    if context is not None:
        input_file, reader = context
        reader.close()
        input_file.close()


def iter_tar_members_with_cli(input_archive: Path, zstd_binary: str) -> tuple[tarfile.TarFile, subprocess.Popen[bytes]]:
    process = subprocess.Popen(
        [resolve_zstd_binary(zstd_binary), "-q", "-d", "-c", str(input_archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to open zstd stdout")
    return tarfile.open(fileobj=process.stdout, mode="r|"), process


def finish_cli_process(process: subprocess.Popen[bytes]) -> None:
    if process.stdout is not None and not process.stdout.closed:
        process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"zstd decompression failed with code {return_code}: {message}")


def member_type(member: tarfile.TarInfo) -> str:
    if member.isdir():
        return "directory"
    if member.isfile():
        return "file"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    return "other"


def build_member_plan(
    *,
    member: tarfile.TarInfo,
    output_dir: Path,
    strip_components: int,
    safe_extract: bool,
) -> dict[str, Any]:
    stripped_name = strip_member_name(member.name, strip_components, safe_extract)
    if stripped_name is None:
        return {
            "archive_path": member.name,
            "output_path": None,
            "type": member_type(member),
            "size": member.size,
            "action": "skip_strip_components",
        }

    destination = ensure_safe_destination(output_dir, stripped_name, safe_extract)
    return {
        "archive_path": member.name,
        "output_path": str(destination),
        "type": member_type(member),
        "size": member.size,
        "action": "extract",
    }


def copy_file_member(
    *,
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
    overwrite: bool,
    capture_bytes: bool,
) -> bytes | None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists and overwrite=false: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    source = tar.extractfile(member)
    if source is None:
        raise RuntimeError(f"Failed to read archive member: {member.name}")

    captured = bytearray() if capture_bytes else None
    with destination.open("wb") as output_file:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            output_file.write(chunk)
            if captured is not None:
                captured.extend(chunk)

    if member.mtime:
        os.utime(destination, (member.mtime, member.mtime))
    return bytes(captured) if captured is not None else None


def extract_members(tar: tarfile.TarFile, cfg: DictConfig, output_dir: Path) -> dict[str, Any]:
    strip_components = int(cfg.strip_components)
    safe_extract = bool(cfg.safe_extract)
    overwrite = bool(cfg.overwrite)
    dry_run = bool(cfg.dry_run)
    max_summary_files = int(cfg.max_summary_files)

    plans: list[dict[str, Any]] = []
    files_planned = 0
    dirs_planned = 0
    skipped = 0
    archive_manifest_bytes: bytes | None = None

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for member in tar:
        plan = build_member_plan(
            member=member,
            output_dir=output_dir,
            strip_components=strip_components,
            safe_extract=safe_extract,
        )
        if len(plans) < max_summary_files:
            plans.append(plan)
        if plan["action"] == "skip_strip_components":
            skipped += 1
            continue

        output_path = Path(str(plan["output_path"]))
        if member.isdir():
            if not dry_run:
                if output_path.exists() and not output_path.is_dir():
                    raise FileExistsError(f"Cannot create directory over existing file: {output_path}")
                output_path.mkdir(parents=True, exist_ok=True)
            dirs_planned += 1
            continue

        if member.isfile():
            if not dry_run:
                captured = copy_file_member(
                    tar=tar,
                    member=member,
                    destination=output_path,
                    overwrite=overwrite,
                    capture_bytes=member.name == "archive_manifest.json",
                )
                if captured is not None:
                    archive_manifest_bytes = captured
            files_planned += 1
            continue

        if safe_extract:
            raise ValueError(f"Refusing to extract unsupported member type {member_type(member)}: {member.name}")
        skipped += 1

    return {
        "files_planned": files_planned,
        "dirs_planned": dirs_planned,
        "files_extracted": 0 if dry_run else files_planned,
        "dirs_extracted": 0 if dry_run else dirs_planned,
        "skipped": skipped,
        "members_preview": plans,
        "members_preview_count": len(plans),
        "archive_manifest_bytes": archive_manifest_bytes,
    }


def load_manifest_for_verification(
    *,
    output_dir: Path,
    archive_manifest_bytes: bytes | None,
) -> dict[str, Any] | None:
    if archive_manifest_bytes is not None:
        return json.loads(archive_manifest_bytes.decode("utf-8"))
    manifest_path = output_dir / "archive_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return None


def verify_manifest(
    *,
    manifest: dict[str, Any] | None,
    output_dir: Path,
    strip_components: int,
    safe_extract: bool,
) -> dict[str, Any]:
    if manifest is None:
        return {"status": "not_found", "checked_files": 0, "failed": []}

    failed: list[dict[str, Any]] = []
    checked = 0
    skipped = 0
    for row in manifest.get("files", []):
        archive_path = str(row["path"])
        stripped_name = strip_member_name(archive_path, strip_components, safe_extract)
        if stripped_name is None:
            skipped += 1
            continue
        target_path = ensure_safe_destination(output_dir, stripped_name, safe_extract)
        if not target_path.exists():
            failed.append({"path": stripped_name, "error": "missing"})
            continue
        expected_size = int(row["size"])
        actual_size = target_path.stat().st_size
        if actual_size != expected_size:
            failed.append(
                {
                    "path": stripped_name,
                    "error": "size_mismatch",
                    "expected": expected_size,
                    "actual": actual_size,
                }
            )
            continue
        expected_sha256 = str(row["sha256"])
        actual_sha256 = sha256_file(target_path)
        if actual_sha256 != expected_sha256:
            failed.append(
                {
                    "path": stripped_name,
                    "error": "sha256_mismatch",
                    "expected": expected_sha256,
                    "actual": actual_sha256,
                }
            )
            continue
        checked += 1

    return {
        "status": "ok" if not failed else "failed",
        "checked_files": checked,
        "skipped_by_strip_components": skipped,
        "failed": failed,
    }


def read_tar_archive(input_archive: Path, cfg: DictConfig, output_dir: Path) -> dict[str, Any]:
    backend = "python-zstandard" if zstd_module() is not None else "zstd-cli"
    if backend == "python-zstandard":
        tar = open_decompressed_tar_with_python_zstandard(input_archive)
        try:
            result = extract_members(tar, cfg, output_dir)
        finally:
            close_python_zstandard_tar(tar)
    else:
        tar, process = iter_tar_members_with_cli(input_archive, str(cfg.zstd_binary))
        try:
            result = extract_members(tar, cfg, output_dir)
        finally:
            tar.close()
            finish_cli_process(process)

    result["compression_backend"] = backend
    return result


def archive_unpack(cfg: DictConfig) -> dict[str, Any]:
    input_archive = Path(to_absolute_path(str(cfg.input_archive))).resolve()
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    validate_tar_zst_path(input_archive)
    if not input_archive.exists():
        raise FileNotFoundError(f"Archive not found: {input_archive}")

    result = read_tar_archive(input_archive, cfg, output_dir)
    archive_manifest_bytes = result.pop("archive_manifest_bytes")
    verification = None
    if bool(cfg.verify_manifest) and not bool(cfg.dry_run):
        manifest = load_manifest_for_verification(
            output_dir=output_dir,
            archive_manifest_bytes=archive_manifest_bytes,
        )
        verification = verify_manifest(
            manifest=manifest,
            output_dir=output_dir,
            strip_components=int(cfg.strip_components),
            safe_extract=bool(cfg.safe_extract),
        )

    summary = {
        "dry_run": bool(cfg.dry_run),
        "input_archive": str(input_archive),
        "output_dir": str(output_dir),
        "overwrite": bool(cfg.overwrite),
        "strip_components": int(cfg.strip_components),
        "safe_extract": bool(cfg.safe_extract),
        **result,
    }
    if verification is not None:
        summary["manifest_verification"] = verification
    return summary


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="archive_unpack",
)
def main(cfg: DictConfig) -> None:
    summary = archive_unpack(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
