"""Shared real-file storage laboratory for safety and crash-boundary tests."""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@dataclass
class Lab:
    module: Any
    filesystem: Any
    layout: Any
    usage: Any
    config: Any
    engine: Any
    proc: Path
    now: float
    wanted: dict[str, bytes] | None = None
    orphan: Path | None = None

    def unit(self, path: str = "data/task/v1") -> Any:
        directory = self.config.repo / path
        (directory / "nested").mkdir(parents=True)
        (directory / "a.bin").write_bytes(b"apple\n")
        (directory / "nested/b.bin").write_bytes(b"\x00\x01\x04\x09")
        unit = self.layout.Unit(
            path,
            "checkpoint"
            if path.startswith("ckpt/")
            else "output"
            if "outputs/" in path
            else "dataset",
        )
        self.age(unit, 15)
        return unit

    def age(self, unit: Any, days: float) -> None:
        for name, entry in self.filesystem.snapshot(
            self.config.repo / unit.path
        ).entries.items():
            if entry["kind"] == "file":
                path = self.config.repo / unit.path / name
                os.utime(
                    path,
                    ns=(
                        int((self.now - days * 86400) * 10**9),
                        path.stat().st_mtime_ns,
                    ),
                )

    def run(
        self, command: str, unit: Any, *, dry_run: bool = False
    ) -> tuple[int, dict[str, Any]]:
        return cast(
            tuple[int, dict[str, Any]],
            self.engine.run(command, [unit.path], dry_run=dry_run),
        )

    def mirror(self, unit: Any) -> None:
        code, report = self.run("mirror", unit)
        assert code == 0, report
        assert report["results"][0]["action"] in {"mirrored", "verified", "unchanged"}

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.filesystem.read_bytes(
                self.config.state / "audit.jsonl"
            ).splitlines()
        ]


def make_lab(tmp_path: Path) -> Lab:
    module = importlib.import_module("storage_tier.engine")
    filesystem = importlib.import_module("storage_tier.filesystem")
    layout = importlib.import_module("storage_tier.layout")
    usage = importlib.import_module("storage_tier.usage")
    repo, mount, proc = tmp_path / "D/repo", tmp_path / "E", tmp_path / "proc"
    repo.mkdir(parents=True)
    mount.mkdir()
    proc.mkdir()
    config = layout.Config(repo, mount, ("data/task",))
    config.mirror.mkdir(parents=True)
    # Aged atimes are truly in the past: an ordinary read must update them.
    result = Lab(module, filesystem, layout, usage, config, None, proc, time.time())
    result.engine = module.Engine(
        config,
        inspector=usage.Inspector(proc),
        mount_check=lambda config: None,
        clock=lambda: result.now,
    )
    return result
