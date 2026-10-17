"""Bounded validation of retained JSON-only ZIP artifacts."""

import io
import re
import stat
import zipfile

from .contracts import loads_json

MAX_ZIP_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 256


def simple_name(name: str, suffix: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.()-]{0,199}", name
    ) or not name.endswith(suffix):
        raise ValueError(
            f"expected a plain {suffix} filename (at most 200 ASCII letters, digits, "
            "underscores, dots, hyphens or parentheses; start with a letter, digit or underscore)"
        )
    return name


def inspect_zip(data: bytes) -> list[str]:
    if not data or len(data) > MAX_ZIP_BYTES:
        raise ValueError("ZIP must be nonempty and at most 16 MiB")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= MAX_MEMBERS:
            raise ValueError("ZIP must contain 1..256 JSON files")
        if sum(member.file_size for member in members) > MAX_JSON_BYTES:
            raise ValueError("expanded JSON exceeds 64 MiB")
        names: list[str] = []
        for member in members:
            try:
                simple_name(member.filename, ".json")
            except ValueError as error:
                raise ValueError(
                    f"invalid ZIP member {member.filename!r}: {error}; "
                    "this check applies to entries inside the ZIP, not the submitted ZIP name. "
                    "Put only JSON files directly at the archive root, without directories."
                ) from None
            mode = member.external_attr >> 16
            if member.filename in names or member.flag_bits & 1 or stat.S_ISLNK(mode):
                raise ValueError("duplicate, encrypted or symlink ZIP member")
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("only stored/deflated ZIP members are supported")
            with archive.open(member) as handle:
                content = handle.read(MAX_JSON_BYTES + 1)
            if len(content) > MAX_JSON_BYTES:
                raise ValueError("JSON exceeds 64 MiB")
            if not isinstance(loads_json(content), dict):
                raise ValueError("each JSON must be an annotation object")
            names.append(member.filename)
        return names
