"""Diagnose archived inputs at the common path boundary after local migration."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from src.utils.configuration.errors import PathContractError


def require_hydrated_input(path: Path) -> None:
    """Give an actionable error for an E:-only input, without reading it.

    Activation is explicit and local to a migrated repository. Non-managed
    paths keep their normal existence contracts. No data is fetched here.
    """
    if path.exists():
        return
    for repository in path.parents:
        marker = repository / ".storage-tier/active.json"
        if not marker.exists():
            continue
        fd = os.open(marker, os.O_RDONLY | os.O_NOATIME | os.O_NOFOLLOW)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            record = json.load(stream)
        if (
            not isinstance(record, dict)
            or set(record) != {"schema_version", "repository", "mirror_mount"}
            or record["schema_version"] != 1
            or record["repository"] != str(repository)
        ):
            raise PathContractError(f"Invalid local storage activation: {marker}")
        mount = Path(record["mirror_mount"])
        if not mount.is_absolute() or mount == Path("/") or mount.resolve() != mount:
            raise PathContractError(f"Invalid local storage mirror mount: {marker}")
        relative = path.relative_to(repository)
        parts = relative.parts
        managed = bool(parts) and (
            parts[0] in {"data", "outputs", "ckpt"}
            or (
                len(parts) > 3
                and parts[:2] == (".claude", "worktrees")
                and parts[3] == "outputs"
            )
        )
        if not managed:
            return
        if not os.path.ismount(mount):
            raise PathContractError(
                f"Local storage input is missing on D: and E: is not mounted at {mount}: {path}. Mount E: before using the storage-tier hydrate CLI."
            )
        mirror = mount / repository.relative_to("/") / relative
        if mirror.exists():
            quoted_repo, quoted_relative = (
                shlex.quote(str(repository)),
                shlex.quote(str(relative)),
            )
            raise PathContractError(
                f"Input is archived on E: and is missing on D: {path}. "
                f"Hydrate it before use: cd {quoted_repo} && "
                "PYTHONPATH=.agents/ops .venv/bin/python .agents/ops/checkout_job.py --repo . -- "
                f".venv/bin/python -m storage_tier hydrate {quoted_relative}"
            )
        return
