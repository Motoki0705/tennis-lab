"""Persist AutoDream history under a single nonblocking writer lock."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from .git import is_sha
from .operations import load_json

SCHEMA_VERSION = 1
STATE_FILE = "autodream_state.json"
LOCK_FILE = "autodream.lock"
Outcome = Literal["merged", "noop", "skipped_trigger", "dry_run", "failed"]


@dataclass(frozen=True)
class State:
    schema_version: int = SCHEMA_VERSION
    last_run_at: str | None = None
    last_completed_at: str | None = None
    last_consolidated_sha: str | None = None
    last_outcome: Outcome | None = None
    last_trigger: dict[str, object] | None = None
    last_pr: int | None = None
    last_merge_sha: str | None = None
    last_error: str | None = None


def _timestamp(value: object) -> bool:
    return (
        isinstance(value, str) and datetime.fromisoformat(value).utcoffset() is not None
    )


def read_state(path: Path) -> State:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return State()
    try:
        value = load_json(raw)
    except (ValueError, RecursionError):
        raise ValueError(f"{path}: invalid AutoDream state JSON") from None
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"{path}: unsupported AutoDream state schema")
    try:
        if value.keys() != {item.name for item in fields(State)}:
            raise ValueError()
        state = State(**value)
        has_error = isinstance(state.last_error, str) and bool(state.last_error)
        if (
            not _timestamp(state.last_run_at)
            or (state.last_completed_at is None)
            != (state.last_consolidated_sha is None)
            or (
                state.last_completed_at is not None
                and (
                    not _timestamp(state.last_completed_at)
                    or not is_sha(state.last_consolidated_sha)
                )
            )
            or state.last_outcome
            not in ("merged", "noop", "skipped_trigger", "dry_run", "failed")
            or (
                state.last_trigger is not None
                and not isinstance(state.last_trigger, dict)
            )
            or (state.last_pr is None) != (state.last_merge_sha is None)
            or (
                state.last_pr is not None
                and (
                    type(state.last_pr) is not int
                    or state.last_pr <= 0
                    or not is_sha(state.last_merge_sha)
                )
            )
            or ((state.last_outcome == "failed") != has_error)
            or (state.last_error is not None and not has_error)
        ):
            raise ValueError()
    except ValueError:
        raise ValueError(f"{path}: invalid AutoDream state fields") from None
    return state


def write_state(path: Path, state: State) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(asdict(state), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def state_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / LOCK_FILE
    with path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The owner alone can write the status file. The CLI reports this to stderr.
            raise ValueError(
                f"AutoDream lock contention: {path} is already held"
            ) from None
        yield
