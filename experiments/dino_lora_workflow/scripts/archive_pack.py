"""Overview:
Pack workflow files and directories into a streaming `.tar.zst` archive.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/archive_pack.py input_paths='[data/dino_workflow/guardrail/current]' output_archive=data/dino_workflow/archives/guardrail_current.tar.zst
    .venv/bin/python experiments/dino_lora_workflow/scripts/archive_pack.py dry_run=true input_paths='[data/youtube/videos/av1]' output_archive=outputs/tmp/videos.tar.zst

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/archive_pack.yaml`.
    - The archive is written as a tar stream compressed with Python `zstandard` when available, or the `zstd` CLI as a fallback.
    - `archive_manifest.json` records file paths, sizes, and SHA-256 hashes when `include_manifest=true`.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


DEFAULT_EXCLUDE_GLOBS = ["__pycache__", "__pycache__/**", "*.pyc", ".DS_Store"]


@dataclass(slots=True)
class ArchiveFile:
    """One regular file planned for the tar archive."""

    source_path: Path
    archive_path: str
    size: int
    sha256: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(to_absolute_path(text)).resolve()


def validate_tar_zst_path(path: Path) -> None:
    if not str(path).endswith(".tar.zst"):
        raise ValueError(f"Archive path must end with .tar.zst: {path}")


def resolve_input_paths(values: Any) -> list[Path]:
    paths = [Path(to_absolute_path(str(value))).resolve() for value in values]
    if not paths:
        raise ValueError("input_paths must contain at least one file or directory")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input paths do not exist: {missing}")
    return paths


def normalize_exclude_globs(values: Any) -> list[str]:
    patterns = [str(value) for value in values]
    return patterns if patterns else list(DEFAULT_EXCLUDE_GLOBS)


def archive_path_for(source_path: Path, input_root: Path, base_dir: Path | None) -> str:
    if base_dir is not None:
        try:
            relative_path = source_path.relative_to(base_dir)
        except ValueError as exc:
            raise ValueError(f"Input path {source_path} is not under base_dir {base_dir}") from exc
        return relative_path.as_posix()

    if input_root.is_file():
        return source_path.name
    return (Path(input_root.name) / source_path.relative_to(input_root)).as_posix()


def should_exclude(archive_path: str, patterns: list[str]) -> bool:
    name = Path(archive_path).name
    return any(
        fnmatch.fnmatch(archive_path, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in patterns
    )


def iter_regular_files(input_root: Path, base_dir: Path | None, exclude_globs: list[str]) -> list[ArchiveFile]:
    files: list[ArchiveFile] = []
    if input_root.is_file():
        archive_path = archive_path_for(input_root, input_root, base_dir)
        if not should_exclude(archive_path, exclude_globs):
            files.append(ArchiveFile(input_root, archive_path, input_root.stat().st_size))
        return files

    for root, dirnames, filenames in os.walk(input_root):
        root_path = Path(root)
        kept_dirnames: list[str] = []
        for dirname in sorted(dirnames):
            dir_path = root_path / dirname
            archive_path = archive_path_for(dir_path, input_root, base_dir)
            if not should_exclude(archive_path, exclude_globs):
                kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        for filename in sorted(filenames):
            source_path = root_path / filename
            archive_path = archive_path_for(source_path, input_root, base_dir)
            if should_exclude(archive_path, exclude_globs) or not source_path.is_file():
                continue
            files.append(ArchiveFile(source_path, archive_path, source_path.stat().st_size))
    return files


def collect_files(
    *,
    input_paths: list[Path],
    base_dir: Path | None,
    exclude_globs: list[str],
) -> list[ArchiveFile]:
    collected: list[ArchiveFile] = []
    seen: dict[str, Path] = {}
    for input_path in input_paths:
        for item in iter_regular_files(input_path, base_dir, exclude_globs):
            previous = seen.get(item.archive_path)
            if previous is not None:
                raise ValueError(
                    f"Archive path collision for {item.archive_path}: {previous} and {item.source_path}"
                )
            seen[item.archive_path] = item.source_path
            collected.append(item)
    return sorted(collected, key=lambda item: item.archive_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    files: list[ArchiveFile],
    input_paths: list[Path],
    output_archive: Path,
    base_dir: Path | None,
    exclude_globs: list[str],
) -> dict[str, Any]:
    file_rows: list[dict[str, Any]] = []
    for item in files:
        item.sha256 = sha256_file(item.source_path)
        file_rows.append({"path": item.archive_path, "size": item.size, "sha256": item.sha256})

    return {
        "created_at": now_iso(),
        "input_paths": [str(path) for path in input_paths],
        "output_archive": str(output_archive),
        "base_dir": str(base_dir) if base_dir is not None else None,
        "exclude_globs": exclude_globs,
        "files": file_rows,
        "total_files": len(files),
        "total_bytes": sum(item.size for item in files),
    }


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


def add_manifest_to_tar(tar: tarfile.TarFile, manifest: dict[str, Any]) -> None:
    payload = json.dumps(manifest, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    info = tarfile.TarInfo("archive_manifest.json")
    info.size = len(payload)
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))


def add_files_to_tar(tar: tarfile.TarFile, files: list[ArchiveFile], manifest: dict[str, Any] | None) -> None:
    if manifest is not None:
        if any(item.archive_path == "archive_manifest.json" for item in files):
            raise ValueError("archive_manifest.json would collide with an input file")
        add_manifest_to_tar(tar, manifest)
    for item in files:
        tar.add(item.source_path, arcname=item.archive_path, recursive=False)


def write_with_python_zstandard(
    *,
    tmp_archive: Path,
    files: list[ArchiveFile],
    manifest: dict[str, Any] | None,
    compression_level: int,
) -> None:
    zstd = zstd_module()
    if zstd is None:
        raise RuntimeError("Python zstandard package is unavailable")
    cctx = zstd.ZstdCompressor(level=compression_level)
    with tmp_archive.open("wb") as output_file:
        with cctx.stream_writer(output_file) as compressor:
            with tarfile.open(
                fileobj=compressor,
                mode="w|",
                format=tarfile.PAX_FORMAT,
                dereference=True,
            ) as tar:
                add_files_to_tar(tar, files, manifest)


def write_with_zstd_cli(
    *,
    tmp_archive: Path,
    files: list[ArchiveFile],
    manifest: dict[str, Any] | None,
    compression_level: int,
    zstd_binary: str,
) -> None:
    command = [resolve_zstd_binary(zstd_binary), "-q", f"-{compression_level}", "-T0", "-c"]
    process: subprocess.Popen[bytes] | None = None
    with tmp_archive.open("wb") as output_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=output_file,
            stderr=subprocess.PIPE,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("Failed to open zstd stdin")
            with tarfile.open(
                fileobj=process.stdin,
                mode="w|",
                format=tarfile.PAX_FORMAT,
                dereference=True,
            ) as tar:
                add_files_to_tar(tar, files, manifest)
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code != 0:
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"zstd compression failed with code {return_code}: {message}")
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise


def write_archive(
    *,
    output_archive: Path,
    files: list[ArchiveFile],
    manifest: dict[str, Any] | None,
    overwrite: bool,
    compression_level: int,
    zstd_binary: str,
) -> str:
    validate_tar_zst_path(output_archive)
    if output_archive.exists() and not overwrite:
        raise FileExistsError(f"Archive already exists and overwrite=false: {output_archive}")
    output_archive.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_archive.name}.",
        suffix=".tmp",
        dir=output_archive.parent,
        delete=False,
    ) as tmp_file:
        tmp_archive = Path(tmp_file.name)

    backend = "python-zstandard" if zstd_module() is not None else "zstd-cli"
    try:
        if backend == "python-zstandard":
            write_with_python_zstandard(
                tmp_archive=tmp_archive,
                files=files,
                manifest=manifest,
                compression_level=compression_level,
            )
        else:
            write_with_zstd_cli(
                tmp_archive=tmp_archive,
                files=files,
                manifest=manifest,
                compression_level=compression_level,
                zstd_binary=zstd_binary,
            )
        tmp_archive.replace(output_archive)
    except Exception:
        tmp_archive.unlink(missing_ok=True)
        raise
    return backend


def build_summary(
    *,
    dry_run: bool,
    input_paths: list[Path],
    output_archive: Path,
    base_dir: Path | None,
    exclude_globs: list[str],
    files: list[ArchiveFile],
    max_summary_files: int,
    backend: str | None = None,
) -> dict[str, Any]:
    files_preview = [
        {
            "source_path": str(item.source_path),
            "path": item.archive_path,
            "size": item.size,
            "sha256": item.sha256,
        }
        for item in files[:max_summary_files]
    ]
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "input_paths": [str(path) for path in input_paths],
        "output_archive": str(output_archive),
        "base_dir": str(base_dir) if base_dir is not None else None,
        "exclude_globs": exclude_globs,
        "total_files": len(files),
        "total_bytes": sum(item.size for item in files),
        "files_preview": files_preview,
        "files_preview_count": len(files_preview),
    }
    if backend is not None:
        summary["compression_backend"] = backend
    if len(files) > max_summary_files:
        summary["omitted_preview_files"] = len(files) - max_summary_files
    return summary


def archive_pack(cfg: DictConfig) -> dict[str, Any]:
    output_archive = Path(to_absolute_path(str(cfg.output_archive))).resolve()
    validate_tar_zst_path(output_archive)
    input_paths = resolve_input_paths(cfg.input_paths)
    base_dir = optional_path(cfg.base_dir)
    exclude_globs = normalize_exclude_globs(cfg.exclude_globs)
    files = collect_files(input_paths=input_paths, base_dir=base_dir, exclude_globs=exclude_globs)

    max_summary_files = int(cfg.max_summary_files)
    if bool(cfg.dry_run):
        return build_summary(
            dry_run=True,
            input_paths=input_paths,
            output_archive=output_archive,
            base_dir=base_dir,
            exclude_globs=exclude_globs,
            files=files,
            max_summary_files=max_summary_files,
        )

    manifest = (
        build_manifest(
            files=files,
            input_paths=input_paths,
            output_archive=output_archive,
            base_dir=base_dir,
            exclude_globs=exclude_globs,
        )
        if bool(cfg.include_manifest)
        else None
    )
    backend = write_archive(
        output_archive=output_archive,
        files=files,
        manifest=manifest,
        overwrite=bool(cfg.overwrite),
        compression_level=int(cfg.compression_level),
        zstd_binary=str(cfg.zstd_binary),
    )
    summary = build_summary(
        dry_run=False,
        input_paths=input_paths,
        output_archive=output_archive,
        base_dir=base_dir,
        exclude_globs=exclude_globs,
        files=files,
        max_summary_files=max_summary_files,
        backend=backend,
    )
    summary["include_manifest"] = bool(cfg.include_manifest)
    summary["archive_size"] = output_archive.stat().st_size
    return summary


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="archive_pack",
)
def main(cfg: DictConfig) -> None:
    summary = archive_pack(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
