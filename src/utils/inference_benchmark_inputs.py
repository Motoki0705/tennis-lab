"""Verify published input manifests and the ordered timing schedule they pin."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import cycle, islice
from pathlib import Path
from typing import Any

from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import BenchmarkProtocol

__all__ = [
    "PinnedBenchmarkInputs",
    "TimingSchedule",
    "read_pinned_json",
    "timed_batch_sha256",
]

_Row = dict[str, Any]
_Ids = tuple[tuple[str, ...], ...]


def read_pinned_json(path: Path, expected_sha256: str) -> Any:
    content = path.read_bytes()
    actual = sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch at {path}: {actual} != {expected_sha256}")
    return json.loads(content)


def timed_batch_sha256(batches: _Ids, repetitions: int) -> str:
    """Digest the JSON array of measured batches, including every repeated ID."""
    measured = list(islice(cycle(batches), repetitions))
    return sha256(
        json.dumps(measured, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TimingSchedule:
    protocol: BenchmarkProtocol
    batch_ids: _Ids
    timed_ids_sha256: str

    def require_batches(self, batches: Sequence[Sequence[str]]) -> None:
        if tuple(tuple(ids) for ids in batches) != self.batch_ids:
            raise ValueError(
                "Batch input IDs or order differ from the pinned timing schedule"
            )


class PinnedBenchmarkInputs:
    """Load input rows and schedules using independently published SHA-256 pins."""

    def __init__(
        self,
        directory: Path,
        timing_path: Path,
        *,
        manifest_sha256: str,
        timing_sha256: str,
    ) -> None:
        self.directory = directory.resolve(strict=True)
        self.manifest_sha256 = manifest_sha256
        self.timing_sha256 = timing_sha256
        self.manifest: dict[str, Any] = read_pinned_json(
            self.directory / "manifest.json", manifest_sha256
        )
        self.timing: dict[str, Any] = read_pinned_json(timing_path, timing_sha256)
        if (
            self.timing["schema_version"] != 1
            or self.timing["input_manifest_sha256"] != manifest_sha256
        ):
            raise ValueError("Timing protocol schema or input manifest SHA-256 differs")
        self._indexes: dict[str, tuple[_Row, ...]] = {}

    def _relative(self, value: str) -> Path:
        path = (self.directory / value).resolve()
        if Path(value).is_absolute() or not path.is_relative_to(self.directory):
            raise ValueError(f"Input file escapes the pinned directory: {value}")
        return path

    def rows(self, name: str) -> tuple[_Row, ...]:
        if name not in self._indexes:
            spec = self.manifest["indexes"][name]
            path = self._relative(spec["path"])
            rows = read_pinned_json(path, spec["sha256"])
            ids = [row["id"] for row in rows]
            if (
                len(ids) != spec["records"]
                or len(set(ids)) != len(ids)
                or any(type(key) is not str or not key for key in ids)
            ):
                raise ValueError(f"Invalid record count or input IDs in {path}")
            self._indexes[name] = tuple(rows)
        return self._indexes[name]

    def verified_asset(self, row: _Row) -> Path:
        path = self._relative(row["path"])
        if dual_sha256(path) != row["sha256"]:
            raise ValueError(f"Input asset SHA-256 mismatch: {path}")
        return path

    def schedule(self, index: str, batch_size: int) -> TimingSchedule:
        spec = self.timing["schedules"][index]
        batch_sizes = tuple(sorted(map(int, spec["batch_sizes"])))
        protocol = BenchmarkProtocol(
            batch_sizes=batch_sizes,
            warmup=self.timing["warmup"],
            iterations_per_block=self.timing["iterations_per_block"],
            blocks=self.timing["blocks"],
        )
        if batch_size not in batch_sizes:
            raise ValueError(
                f"Batch {batch_size} is outside the pinned sweep for {index}"
            )
        rows = self.rows(index)
        count = spec["records"]
        if type(count) is not int or not 0 < count <= len(rows) or count % batch_size:
            raise ValueError(
                f"Pinned input count does not form complete {index} batches"
            )
        ids = tuple(row["id"] for row in rows[:count])
        batches = tuple(ids[i : i + batch_size] for i in range(0, count, batch_size))
        repeats = protocol.iterations_per_block * protocol.blocks
        expected = spec["batch_sizes"][str(batch_size)]
        if timed_batch_sha256(batches, repeats) != expected:
            raise ValueError(
                f"Input IDs disagree with the pinned {index} B={batch_size} timing order"
            )
        return TimingSchedule(protocol, batches, expected)
