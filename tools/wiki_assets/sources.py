"""Read only the named files from a Git-pinned source revision."""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVISION = "8d629a35289f475e514291167435c333876afdd2"
REPOSITORY = "https://github.com/Motoki0705/tennis-lab"
RUNS = "knowledge/runs/"
CsvRow = dict[str, str]
JsonObject = dict[str, object]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(
            f"Git could not resolve the revision or read the pinned file: {args[-1]}"
        )
    return result.stdout


class Sources:
    def __init__(self, revision: str, read: Callable[[str], bytes]) -> None:
        require(
            re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
            "Expected a full Git commit SHA",
        )
        self.revision = revision
        self._read = read
        self.digests: dict[str, str] = {}

    @classmethod
    def from_git(cls, root: Path, revision: str) -> Sources:
        commit = (
            git(
                root,
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{revision}^{{commit}}",
            )
            .decode()
            .strip()
        )
        return cls(commit, lambda path: git(root, "show", f"{commit}:{path}"))

    def _bytes(self, relative: str) -> bytes:
        parts = PurePosixPath(relative).parts
        require(
            bool(parts)
            and re.fullmatch(r"[a-zA-Z0-9_./-]+", relative) is not None
            and all(part not in {"/", ".."} for part in parts),
            "Expected a relative path within knowledge/runs",
        )
        path = RUNS + relative
        blob = self._read(path)
        digest = sha256(blob).hexdigest()
        require(
            self.digests.setdefault(path, digest) == digest,
            f"{path}: source changed during read",
        )
        return blob

    def csv(self, relative: str) -> list[CsvRow]:
        reader = csv.reader(
            io.StringIO(self._bytes(relative).decode("utf-8"), newline=""), strict=True
        )
        header = next(reader, [])
        require(
            bool(header) and all(header) and len(header) == len(set(header)),
            f"{relative}: invalid CSV header",
        )
        rows = []
        for values in reader:
            require(
                len(values) == len(header),
                f"{relative}:{reader.line_num}: wrong number of CSV fields",
            )
            rows.append(dict(zip(header, values, strict=True)))
        require(bool(rows), f"{relative}: empty CSV")
        return rows

    def json(self, relative: str) -> JsonObject:
        return obj(json.loads(self._bytes(relative), parse_float=Decimal), relative)


def obj(value: object, label: str = "JSON field") -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    return cast(JsonObject, value)


def array(value: object, label: str = "JSON field") -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected an array")
    return cast(list[object], value)


def string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a non-empty string")
    return value


def number(value: object) -> Decimal:
    if not isinstance(value, (str, int, Decimal)) or isinstance(value, bool):
        raise ValueError("Expected a number")
    try:
        decimal = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Invalid numeric value") from error
    require(
        decimal.is_finite() and decimal >= 0, "Expected a finite non-negative number"
    )
    return decimal


def count(value: object) -> int:
    decimal = number(value)
    require(decimal == decimal.to_integral_value(), "Expected an integer count")
    return int(decimal)


def fraction(value: object) -> Decimal:
    decimal = number(value)
    require(decimal <= 1, "Expected a fraction in [0, 1]")
    return decimal


def mib(value: object) -> Decimal:
    return number(value) / (1024**2)


def unique(rows: Iterable[CsvRow], *keys: str) -> dict[tuple[str, ...], CsvRow]:
    index: dict[tuple[str, ...], CsvRow] = {}
    for row in rows:
        key = tuple(string(row[column]) for column in keys)
        require(key not in index, f"Duplicate source key for {', '.join(keys)}")
        index[key] = row
    return index
