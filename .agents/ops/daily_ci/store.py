"""Private durable state; short state transactions and a separate repo merge mutex."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

from .model import LIVE, instant, request_id, sha, window


def state_directory() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return base / "tennis-lab-agents/daily-ci"


def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) != len({key for key, _ in pairs}):
        raise ValueError("duplicate JSON keys")
    return dict(pairs)


def load(body: str) -> Any:
    return json.loads(body, object_pairs_hook=unique)


class CycleStore:
    def __init__(self, repository: str, directory: Path | None = None) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("repository must be owner/name")
        self.repository = repository.casefold()
        self.root = directory if directory is not None else state_directory()
        self.directory = (
            self.root / hashlib.sha256(self.repository.encode()).hexdigest()
        )
        self.path = self.directory / "state.json"
        self.marker = self.directory / "initialized"

    @contextmanager
    def lock(self, name: str, *, blocking: bool = True) -> Iterator[None]:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (self.directory / f"{name}.lock").open("a") as handle:
            os.chmod(handle.name, 0o600)
            try:
                fcntl.flock(
                    handle,
                    fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB),
                )
            except BlockingIOError:
                raise ValueError(f"daily_ci {name} operation is busy") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def repository_lock(
        self, *, blocking: bool = False
    ) -> AbstractContextManager[None]:
        return self.lock("repository-merge", blocking=blocking)

    def read(self, *, required: bool = True) -> dict[str, Any] | None:
        if not self.path.exists():
            if self.marker.exists() or required:
                raise ValueError(
                    "daily_ci state is missing; restore it, do not reinitialize"
                )
            return None
        data: dict[str, Any] = load(self.path.read_text(encoding="utf-8"))
        self.validate(data)
        return data

    def validate(self, data: Any) -> None:
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != 1
            or type(data.get("schema_version")) is not int
            or data.get("repository") != self.repository
            or type(data.get("enabled")) is not bool
            or not isinstance(data.get("cycles"), dict)
            or not isinstance(data.get("integrations"), dict)
            or not isinstance(data.get("workflows"), list)
            or not data["workflows"]
            or any(
                not isinstance(w, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.ya?ml", w)
                for w in data["workflows"]
            )
            or len(set(data["workflows"])) != len(data["workflows"])
        ):
            raise ValueError("unknown or malformed daily_ci state")
        sha(data["baseline_sha"])
        for key, cycle in data["cycles"].items():
            expected_id, opens, deadline = window(instant(cycle["opens_at"]))
            if (
                key != cycle["cycle_id"]
                or key != expected_id
                or cycle["status"]
                not in LIVE | {"success", "expired", "missed", "disabled"}
                or instant(cycle["opens_at"]) != opens
                or instant(cycle["deadline_at"]) != deadline
                or not isinstance(cycle["generations"], list)
                or cycle["workflows"] != data["workflows"]
            ):
                raise ValueError("invalid daily_ci cycle")
            for number, revision in enumerate(cycle["generations"], 1):
                if revision["number"] != number:
                    raise ValueError("invalid daily_ci generation")
                sha(revision["target_sha"])
                if set(revision["requests"]) != set(cycle["workflows"]):
                    raise ValueError("daily_ci workflow set changed")
                for workflow, request in revision["requests"].items():
                    if request["request_id"] != request_id(
                        self.repository, key, number, workflow
                    ):
                        raise ValueError("daily_ci request identity changed")

    def save(self, data: dict[str, Any]) -> None:
        self.validate(data)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".state-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
            self.marker.touch(mode=0o600, exist_ok=True)
            directory_fd = os.open(self.directory, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @contextmanager
    def transaction(self) -> Iterator[dict[str, Any]]:
        with self.lock("state"):
            data = self.read()
            assert data is not None
            yield data
            self.save(data)
