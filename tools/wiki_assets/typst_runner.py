"""Resolve the fixed Typst CLI without adding a project dependency."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import tarfile
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from .sources import require

VERSION = "0.15.1"
TARGET = "typst-x86_64-unknown-linux-musl"
ARCHIVE_SHA256 = "a6d077d0a95eed5a2eba715b2dae06be954f624ccbf85758a03f389ded33118c"
BINARY_SHA256 = "29273eaa04f6d00edd0c2bec578f565fc9c65be856bfbffc894567c68ed0b237"
URL = f"https://github.com/typst/typst/releases/download/v{VERSION}/{TARGET}.tar.xz"
BINARY_ENV = "WIKI_ASSETS_TYPST_BIN"
FONTS = ("Noto Sans CJK JP", "DejaVu Sans Mono")


def _digest(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _download(destination: Path) -> None:
    with urlopen(URL, timeout=20) as response, destination.open("wb") as file:
        # The pinned release is about 17 MiB. Bound reads before testing its digest.
        limit = 24 * 1024**2
        content = response.read(limit + 1)
        require(len(content) <= limit, "Typst archive exceeds the download limit")
        file.write(content)


def _cached_binary(cache: Path, offline: bool) -> Path:
    require(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "The pinned download targets x86_64 Linux. Set --typst-bin to your Typst 0.15.1 binary.",
    )
    directory = cache / f"{VERSION}-{TARGET}"
    binary = directory / "typst"
    if not binary.is_file():
        require(
            not offline,
            "Typst is missing from the offline cache. Set --typst-bin or WIKI_ASSETS_TYPST_BIN.",
        )
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=directory) as temporary:
            archive = Path(temporary) / "typst.tar.xz"
            _download(archive)
            require(
                _digest(archive) == ARCHIVE_SHA256, "Typst archive SHA-256 mismatch"
            )
            candidate = Path(temporary) / "typst"
            with tarfile.open(archive, mode="r:xz") as contents:
                member = contents.getmember(f"{TARGET}/typst")
                require(
                    member.isfile(),
                    "Expected a regular executable in the Typst archive",
                )
                executable = contents.extractfile(member)
                if executable is None:
                    raise ValueError("Typst archive has no executable bytes")
                with executable:
                    candidate.write_bytes(executable.read())
            require(
                _digest(candidate) == BINARY_SHA256,
                "Extracted Typst binary SHA-256 mismatch",
            )
            candidate.chmod(0o755)
            candidate.replace(binary)
    require(_digest(binary) == BINARY_SHA256, "Cached Typst binary SHA-256 mismatch")
    return binary


def _run(binary: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [binary, *arguments], capture_output=True, text=True, timeout=30, check=False
    )
    if completed.returncode or completed.stderr.strip():
        raise ValueError(f"Typst {arguments[0]} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def resolve(
    explicit: Path | None = None, *, offline: bool = False, cache: Path | None = None
) -> Path:
    value = os.getenv(BINARY_ENV)
    if explicit is None and value:
        explicit = Path(value)
    if explicit is None:
        cache_home = os.getenv("XDG_CACHE_HOME")
        if cache is None:
            cache = (
                (Path(cache_home) if cache_home else Path.home() / ".cache")
                / "tennis-lab"
                / "wiki-assets"
            )
        binary = _cached_binary(cache, offline)
    else:
        binary = explicit.expanduser().resolve()
        require(
            binary.is_file() and os.access(binary, os.X_OK),
            "The explicit Typst binary is missing or not executable",
        )
    version = _run(binary, "--version")
    require(
        re.fullmatch(rf"typst {re.escape(VERSION)}(?: \([^()]+\))?", version)
        is not None,
        f"Expected Typst {VERSION}",
    )
    fonts: dict[str, set[str]] = {}
    family = ""
    for line in _run(binary, "fonts", "--variants").splitlines():
        if line and not line[0].isspace():
            family = line
            fonts[family] = set()
        if variant := re.search(r"Style: Normal, Weight: (400|700)\b", line):
            fonts[family].add(variant[1])
    for font in FONTS:
        require(
            fonts.get(font) == {"400", "700"},
            f"Typst needs regular and bold faces of {font}",
        )
    return binary


def compile_table(binary: Path, source: Path, target: Path) -> None:
    for extension in (".svg", ".png"):
        output = target.with_suffix(extension)
        _run(
            binary,
            "compile",
            "--root",
            os.fspath(source.parent),
            "--creation-timestamp",
            "0",
            "--ppi",
            "144",
            os.fspath(source),
            os.fspath(output),
        )
        require(output.is_file(), f"Expected one Typst {extension} page")
