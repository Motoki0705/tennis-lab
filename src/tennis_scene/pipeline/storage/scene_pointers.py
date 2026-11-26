"""Guard the paired dataset pointers during a recoverable publication switch."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.utils.io import write_json_atomic

_PENDING_FILE = ".scene-pointer-update.json"
_SCHEMA = "tennis_scene_pointer_update_v1"


class ScenePointerUpdateError(ValueError):
    """The live scene and annotation pointers cannot be used until recovery."""


@dataclass(frozen=True)
class PendingSceneUpdate:
    receipt: Path
    recovery: str

    def error(self) -> ScenePointerUpdateError:
        return ScenePointerUpdateError(
            f"Scene pointer update is pending ({self.receipt}). "
            f"If the writer has stopped, {self.recovery}"
        )


def read_pending_scene_update(root: Path) -> PendingSceneUpdate | None:
    """Read the owning receipt while the caller decides whether it can recover."""
    path = root / _PENDING_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        record = json.loads(text)
        if (
            record["schema"] != _SCHEMA
            or not isinstance(record["receipt"], str)
            or not isinstance(record["recovery"], str)
            or not record["recovery"].strip()
        ):
            raise ValueError("invalid record")
        relative = Path(record["receipt"])
        receipt = (root / relative).resolve()
        if relative.is_absolute() or not receipt.is_relative_to(root.resolve()):
            raise ValueError("receipt escapes the scene store")
    except (KeyError, TypeError, ValueError) as error:
        raise ScenePointerUpdateError(
            f"Invalid scene pointer update at {path}. "
            "Inspect its publication receipt before retrying publish or restore."
        ) from error
    return PendingSceneUpdate(receipt, record["recovery"])


def require_stable_scene_pointers(root: Path) -> None:
    pending = read_pending_scene_update(root)
    if pending is not None:
        raise pending.error()


@contextmanager
def stable_scene_pointers(root: Path) -> Iterator[None]:
    """Reject a pending switch on either side of reading the live pointers."""
    require_stable_scene_pointers(root)
    try:
        yield
    finally:
        require_stable_scene_pointers(root)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None


def _replace_text(path: Path, content: str | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
        return
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_scene_pointers(
    root: Path,
    *,
    scene: str,
    annotation: str | None,
    receipt: Path,
    recovery: str,
) -> None:
    """Switch to a validated snapshot under the caller's exclusive clip-store lock.

    The caller verifies the receipt and admits only its known before/after
    pointer bytes. An interruption leaves the receipt discoverable even if
    neither pointer changed. Only the owner can finish or restore that update.
    """
    root, receipt = root.resolve(), receipt.resolve()
    pending = read_pending_scene_update(root)
    if pending is not None and pending.receipt != receipt:
        raise pending.error()
    if (
        not receipt.is_relative_to(root)
        or not receipt.is_file()
        or not recovery.strip()
    ):
        raise ValueError(
            "Scene pointer update requires a saved receipt and recovery instructions"
        )

    pointers = root / "scene.json", root / "annotation.json"
    current = tuple(_read_text(path) for path in pointers)
    expected = scene, annotation
    if current == expected and pending is None:
        return
    guard = root / _PENDING_FILE
    if pending is None:
        write_json_atomic(
            guard,
            {
                "schema": _SCHEMA,
                "receipt": str(receipt.relative_to(root)),
                "recovery": recovery,
            },
        )
    for path, old, new in zip(pointers, current, expected, strict=True):
        if old != new:
            _replace_text(path, new)
    if tuple(_read_text(path) for path in pointers) != expected:
        raise ScenePointerUpdateError(
            f"Scene pointer readback differs from {receipt}. {recovery}"
        )
    guard.unlink()
