"""Detect replacement or edits while reading frozen review evidence outside registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Concatenate, ParamSpec, TypeVar

from parent.state import RegistryConflict

P = ParamSpec("P")
T = TypeVar("T")


def identity(path: Path) -> tuple[int, ...] | None:
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def fixed_files(paths: Iterable[Path]) -> Iterator[None]:
    """Metadata fences bracket reads/hashes; missing inputs remain missing."""
    before = {path: identity(path) for path in paths}
    try:
        yield
    finally:
        if any(identity(path) != observed for path, observed in before.items()):
            raise RegistryConflict("frozen review input changed during verification")


def fixed_run(
    function: Callable[Concatenate[Path, P], T],
) -> Callable[Concatenate[Path, P], T]:
    @wraps(function)
    def checked(directory: Path, *args: P.args, **kwargs: P.kwargs) -> T:
        paths = [
            directory / name
            for name in (
                "request.json",
                "execution.json",
                "result.json",
                "context.json",
                "retrieval-config.json",
                "prompt.md",
                "output.json",
                "agent_output.md",
                "agent-runtime/result.json",
                "agent-runtime/events.jsonl",
                "agent-runtime/final.md",
                "agent-runtime/output-schema.json",
            )
        ]
        with fixed_files([*paths, *directory.glob("pr-*.diff")]):
            return function(directory, *args, **kwargs)

    return checked
