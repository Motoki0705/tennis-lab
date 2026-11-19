"""Install reviewed Linux binaries without consulting system package repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MANIFEST = Path(__file__).with_name("system-deps.json")


def _command(argv: list[str], *, timeout: int = 20) -> str:
    result = subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=timeout
    )
    return result.stdout


def read_manifest(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    if (
        data["schema_version"] != 1
        or data["platform"] != f"{platform.system()}-{platform.machine()}"
    ):
        raise ValueError("unsupported system dependency manifest/platform")
    if [row["name"] for row in data["binaries"]] != ["ffmpeg", "rclone"]:
        raise ValueError("manifest must pin exactly ffmpeg and rclone")
    for row in data["binaries"]:
        url = urlparse(row["url"])
        if url.scheme != "https" or url.hostname not in {
            "github.com",
            "downloads.rclone.org",
        }:
            raise ValueError("unapproved binary URL")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or not row["version"]:
            raise ValueError("missing/invalid version or SHA-256 pin")
        if row["format"] not in {"tar.xz", "zip"}:
            raise ValueError("unsupported archive format")
    return data


def verify_rclone_signature(data: dict[str, Any], root: Path, temporary: Path) -> None:
    """Use an isolated keyring and the reviewed fingerprint, without key discovery."""
    keyring = temporary / "gnupg"
    keyring.mkdir(mode=0o700)
    prefix = ["gpg", "--homedir", str(keyring), "--batch", "--no-auto-key-retrieve"]
    _command([*prefix, "--import", str(root / data["rclone_key"])])
    sums = root / data["rclone_sums"]
    plaintext = temporary / "verified-sums"
    status = _command(
        [
            *prefix,
            "--status-fd",
            "1",
            "--output",
            str(plaintext),
            "--decrypt",
            str(sums),
        ]
    )
    if f"[GNUPG:] VALIDSIG {data['rclone_signer']} " not in status:
        raise ValueError("rclone checksum signature does not match the pinned signer")
    rclone = data["binaries"][1]
    expected = f"{rclone['sha256']}  {Path(urlparse(rclone['url']).path).name}"
    if expected not in plaintext.read_text().splitlines():
        raise ValueError("rclone SHA-256 pin differs from signed SHA256SUMS")
    print(f"Verified rclone SHA256SUMS signature: {data['rclone_signer']}", flush=True)


def _verify(archive: Path, expected: str) -> None:
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(
            f"SHA-256 mismatch: {archive.name}: expected {expected}, got {digest}"
        )


def install(manifest: Path, cache: Path, output: Path) -> None:
    started = time.monotonic()
    data = read_manifest(manifest)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as work:
        temporary = Path(work)
        verify_rclone_signature(data, manifest.parent, temporary)
        staged = temporary / "bin"
        staged.mkdir()
        for row in data["binaries"]:
            name = Path(urlparse(row["url"]).path).name
            archive = cache / name
            print(f"Pinned {row['name']} {row['version']}: {row['url']}", flush=True)
            if archive.exists():
                print(f"Revalidating cached archive: {name}", flush=True)
                _verify(archive, row["sha256"])
            else:
                downloaded = temporary / name
                _command(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--proto",
                        "=https",
                        "--proto-redir",
                        "=https",
                        "--retry",
                        "0",
                        "--connect-timeout",
                        "15",
                        "--max-time",
                        "180",
                        "--output",
                        str(downloaded),
                        row["url"],
                    ],
                    timeout=190,
                )
                _verify(downloaded, row["sha256"])
                shutil.copyfile(downloaded, archive)
            binary = staged / row["name"]
            # Extract only the named regular file, never paths, links or archive metadata.
            if row["format"] == "tar.xz":
                with tarfile.open(archive, "r:xz") as tar:
                    member = tar.getmember(row["member"])
                    if not member.isfile():
                        raise ValueError("ffmpeg archive member is not a regular file")
                    source = tar.extractfile(member)
                    assert source is not None
                    with source, binary.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
            else:
                with zipfile.ZipFile(archive) as zipped:
                    binary.write_bytes(zipped.read(row["member"]))
            binary.chmod(0o755)
            version = _command(
                [str(binary), "-version" if row["name"] == "ffmpeg" else "version"]
            )
            expected = (
                f"ffmpeg version {row['version']} "
                if row["name"] == "ffmpeg"
                else f"rclone {row['version']}\n"
            )
            if not version.startswith(expected):
                raise ValueError(
                    f"unexpected {row['name']} version: {version.splitlines()[:1]}"
                )
            print(version.splitlines()[0], flush=True)
            if row["name"] == "ffmpeg" and not re.search(
                r"\blibx264\b", _command([str(binary), "-encoders"])
            ):
                raise ValueError("pinned ffmpeg does not provide libx264")
        destination = output / "bin"
        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)
    if github_path := os.environ.get("GITHUB_PATH"):
        with Path(github_path).open("a") as stream:
            print((output / "bin").resolve(), file=stream)
    print(
        f"System dependency preparation: {time.monotonic() - started:.3f}s; PATH={(output / 'bin').resolve()}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--cache", type=Path, default=Path(".ci-system-deps-cache"))
    parser.add_argument("--output", type=Path, default=Path(".ci-system-deps"))
    args = parser.parse_args()
    try:
        install(args.manifest.resolve(), args.cache.resolve(), args.output.resolve())
    except (
        OSError,
        ValueError,
        KeyError,
        tarfile.TarError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as error:
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        print(f"::error::System dependency preparation failed: {detail}", flush=True)
        return (
            error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
