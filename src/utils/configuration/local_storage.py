"""Diagnose archived inputs at the common path boundary after local migration."""

from __future__ import annotations

import json
import os
import shlex
import stat
import tomllib
from pathlib import Path

from src.utils.configuration.errors import PathContractError
from src.utils.paths import PROJECT_ROOT

# The runtime checkout and storage CLI read this same policy; no alias list is
# copied into src. The DATA root can belong to the main checkout of a worktree.
POLICY_CONFIG = PROJECT_ROOT / ".agents/ops/storage_tier/config.toml"


def declared_data_path(root: Path, path: Path, resolved: Path) -> Path | None:
    """Validate an explicit E: alias and return its logical D: DATA path.

    Keeping the logical name preserves DATA-relative consumer serialization.
    Only DATA has this additional authority; ordinary paths retain the usual
    containment checks. Configuration and alias contents are never cached.
    """
    if (
        root.name != "data"
        or path == root
        or (path == resolved and resolved.is_relative_to(root))
        or not POLICY_CONFIG.exists()
    ):
        return None
    try:
        fd = os.open(POLICY_CONFIG, os.O_RDONLY | os.O_NOATIME | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            policy = tomllib.load(stream)
        mount = Path(policy["mirror_mount"])
        links = policy.get("unmanaged_data_links", [])
        if not isinstance(links, list) or any(
            not isinstance(name, str)
            or not name.startswith("data/")
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or str(Path(name)) != name
            for name in links
        ):
            raise ValueError("invalid unmanaged_data_links")
        if any(
            Path(first).is_relative_to(second) or Path(second).is_relative_to(first)
            for index, first in enumerate(links)
            for second in links[index + 1 :]
        ):
            raise ValueError("overlapping unmanaged_data_links")
        if not mount.is_absolute() or mount == Path("/") or mount.resolve() != mount:
            raise ValueError("invalid mirror_mount")
        repository = root.parent
        mirror = mount / repository.relative_to("/")
        selected = None
        for name in links:
            alias, expected = repository / name, mirror / name
            if path.is_relative_to(alias):
                selected = alias, expected, path
                break
        if selected is None:
            for name in links:
                alias, expected = repository / name, mirror / name
                if path.is_relative_to(expected):
                    selected = alias, expected, alias / path.relative_to(expected)
                    break
        if selected is None:
            return None
        alias, expected, logical = selected
        if not alias.is_symlink() and resolved.is_relative_to(root):
            # Custom/local DATA roots can contain real directories with these
            # names. They do not need or receive the E: alias exception.
            return None
        if ".." in logical.parts:
            raise ValueError("parent traversal through an unmanaged data link")
        if not os.path.ismount(mount):
            raise ValueError(f"E: is not mounted at {mount}")
        parent = root
        for component in alias.parent.relative_to(root).parts:
            parent /= component
            if not stat.S_ISDIR(parent.lstat().st_mode):
                raise ValueError(
                    f"unmanaged data link has a non-directory ancestor: {parent}"
                )
        fd = os.open(alias, os.O_PATH | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            if not stat.S_ISLNK(before.st_mode):
                raise ValueError(f"unmanaged data link is not a symlink: {alias}")
            if os.readlink("", dir_fd=fd) != str(expected):
                raise ValueError(f"unmanaged data link target mismatch: {alias}")
            observed = alias.lstat()
            if (before.st_dev, before.st_ino, before.st_ctime_ns) != (
                observed.st_dev,
                observed.st_ino,
                observed.st_ctime_ns,
            ):
                raise ValueError(f"unmanaged data link changed: {alias}")
        finally:
            os.close(fd)
        info = expected.lstat()
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError(
                f"unmanaged E: target is not a real file/directory: {expected}"
            )
        if expected.resolve() != expected or not resolved.is_relative_to(expected):
            raise ValueError(f"unmanaged data link escapes its E: target: {path}")
        if logical.resolve(strict=False) != resolved:
            raise ValueError(f"unmanaged data link changed while resolving: {path}")
        return logical
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PathContractError(f"Invalid unmanaged DATA reference: {error}") from error


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
            parts[0] in {"data", "outputs", "ckpt", ".worktree-outputs"}
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
