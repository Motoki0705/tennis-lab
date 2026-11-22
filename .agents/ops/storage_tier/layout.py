"""Explicit dataset boundaries and standard run discovery; no legacy guessing."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .filesystem import StorageError, names, read_bytes, real_directory

PURPOSES = frozenset(
    {"train", "evaluate", "visualize", "analyze", "generate", "precompute"}
)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or any(part.startswith(".") for part in path.parts)
    ):
        raise StorageError(f"invalid managed relative path: {value!r}")
    return value


@dataclass(frozen=True)
class Config:
    repo: Path
    mount: Path
    containers: tuple[str, ...]
    datasets: tuple[str, ...] = ()
    residents: tuple[str, ...] = ()
    external_links: tuple[str, ...] = ()
    # Persistent worktree outputs live only on E:; D: holds the exact link.
    worktree_outputs_on_mirror: bool = False

    @property
    def mirror(self) -> Path:
        return self.mount / self.repo.relative_to("/")

    @property
    def state(self) -> Path:
        return self.repo / ".storage-tier"

    @classmethod
    def load(cls, repo: Path, path: Path) -> Config:
        data = tomllib.loads(read_bytes(path).decode())
        required = {
            "mirror_mount",
            "dataset_containers",
            "dataset_units",
        }
        if not required <= data.keys() or data.keys() - required - {
            "resident_units",
            "unmanaged_data_links",
            "worktree_outputs_on_mirror",
        }:
            raise StorageError(f"unknown/missing storage configuration keys: {path}")
        on_mirror = data.get("worktree_outputs_on_mirror", False)
        if not isinstance(on_mirror, bool):
            raise StorageError("worktree_outputs_on_mirror must be a boolean")
        mount = Path(data["mirror_mount"])
        if not mount.is_absolute() or mount == Path("/") or mount.resolve() != mount:
            raise StorageError("mirror_mount must be a real absolute non-root path")
        lists = []
        for key in (
            "dataset_containers",
            "dataset_units",
            "resident_units",
            "unmanaged_data_links",
        ):
            values = data.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(v, str) or not relative_path(v).startswith("data/")
                for v in values
            ):
                raise StorageError(f"{key} must contain data-relative directory paths")
            if len(set(values)) != len(values):
                raise StorageError(f"duplicate {key}")
            lists.append(tuple(values))
        boundaries = [value for values in lists for value in values]
        for index, first in enumerate(boundaries):
            if any(
                PurePosixPath(first).is_relative_to(second)
                or PurePosixPath(second).is_relative_to(first)
                for second in boundaries[index + 1 :]
            ):
                raise StorageError(
                    "dataset/resident/reference boundaries must not overlap"
                )
        return cls(repo.resolve(), mount, *lists, worktree_outputs_on_mirror=on_mirror)


WORKTREE_OUTPUTS = ".worktree-outputs"


def worktree_outputs_root(config: Config) -> Path:
    """Return the validated real root of persistent worktree outputs.

    The default keeps them in a real D: directory. With
    ``worktree_outputs_on_mirror`` the D: name must be a symlink whose literal
    target is the exact E: counterpart, itself a real directory on the mount.
    Any other shape is an error; neither layout is accepted for the other.
    """
    link = config.repo / WORKTREE_OUTPUTS
    if not config.worktree_outputs_on_mirror:
        return link
    expected = config.mirror / WORKTREE_OUTPUTS
    try:
        info = link.lstat()
    except FileNotFoundError as exc:
        raise StorageError(
            f"worktree_outputs_on_mirror requires a symlink to {expected}: {link}"
        ) from exc
    if not stat.S_ISLNK(info.st_mode):
        raise StorageError(
            f"worktree_outputs_on_mirror requires a symlink to {expected}: {link}"
        )
    if os.readlink(link) != str(expected):
        raise StorageError(f"worktree outputs link target mismatch: {link}")
    after = link.lstat()
    if (after.st_dev, after.st_ino, after.st_ctime_ns) != (
        info.st_dev,
        info.st_ino,
        info.st_ctime_ns,
    ):
        raise StorageError(f"worktree outputs link changed while reading: {link}")
    real_directory(expected, config.mount)
    return expected


@dataclass(frozen=True, order=True)
class Unit:
    path: str
    kind: str

    @property
    def sweepable(self) -> bool:
        return self.kind not in {"checkpoint", "checkpoint-files", "resident"}


@dataclass
class Inventory:
    units: list[Unit]
    unmanaged: list[str]
    errors: list[str]
    references: list[str] = field(default_factory=list)
    legacy_outputs: list[dict[str, str]] = field(default_factory=list)


def discover(config: Config, base: Path) -> Inventory:
    found: set[Unit] = set()
    unmanaged: list[str] = []
    errors: list[str] = []
    references: list[str] = []
    legacy_outputs: list[dict[str, str]] = []
    anchor = config.mount if base == config.mirror else base

    def directories(parent: str, *, reject_links: bool = False) -> list[str]:
        path = base / parent
        try:
            real_directory(path, anchor)
            result = []
            for name in names(path):
                candidate = path / name
                if name.startswith(".storage-tier-"):
                    errors.append(f"unfinished storage transaction: {candidate}")
                elif name.startswith("."):
                    unmanaged.append(str(candidate.relative_to(base)))
                elif reject_links and stat.S_ISLNK(candidate.lstat().st_mode):
                    errors.append(f"undeclared managed unit symlink: {candidate}")
                elif stat.S_ISDIR(candidate.lstat().st_mode):
                    result.append(str(candidate.relative_to(base)))
                else:
                    unmanaged.append(str(candidate.relative_to(base)))
            return result
        except FileNotFoundError:
            return []
        except (OSError, StorageError) as exc:
            errors.append(str(exc))
            return []

    if not base.exists():
        return Inventory([], [], [])
    try:
        real_directory(base, anchor)
    except (OSError, StorageError) as exc:
        return Inventory([], [], [str(exc)])
    for name in names(base):
        if name.startswith(
            (".storage-tier-stage-", ".storage-tier-hydrate-", ".storage-tier-trash-")
        ):
            errors.append(f"unfinished storage transaction: {base / name}")
    children = directories("data")
    known = {
        PurePosixPath(p).parts[1]
        for p in (*config.containers, *config.datasets, *config.residents)
    }
    unmanaged.extend(p for p in children if PurePosixPath(p).parts[1] not in known)
    for container in config.containers:
        for path in directories(container, reject_links=True):
            found.add(Unit(path, "dataset"))
    for path in (*config.datasets, *config.residents):
        try:
            real_directory(base / path, anchor)
            found.add(Unit(path, "resident" if path in config.residents else "dataset"))
        except FileNotFoundError:
            pass
        except (OSError, StorageError) as exc:
            errors.append(str(exc))
    # Inspect only explicitly implied containers, never resident payloads or
    # reference targets. Preserve unclassified siblings as unmanaged entries.
    from .references import data_parents

    declared = {
        *config.containers,
        *config.datasets,
        *config.residents,
        *config.external_links,
    }
    parents = data_parents(config)
    for parent in parents:
        if parent == "data" or parent in config.containers:
            continue
        try:
            real_directory(base / parent, anchor)
            for name in names(base / parent):
                relative = parent + "/" + name
                if (
                    relative not in declared
                    and relative not in parents
                    or relative in config.external_links
                ):
                    unmanaged.append(relative)
        except FileNotFoundError:
            pass
        except (OSError, StorageError) as exc:
            errors.append(str(exc))
    output_roots = ["outputs"]
    if config.worktree_outputs_on_mirror:
        # E:-resident outputs are outside mirror/evict/hydrate: D: and E: are
        # the same bytes through the link, so treating them as units could
        # delete the only copy. On D: only the link itself is validated.
        if base != config.mirror and os.path.lexists(base / WORKTREE_OUTPUTS):
            try:
                worktree_outputs_root(config)
            except (OSError, StorageError) as exc:
                errors.append(str(exc))
    else:
        output_roots.extend(directories(WORKTREE_OUTPUTS))
    # Worktree links are references, never separate units. Do not read their
    # targets: even stat-through-symlink changes an old symlink's atime.
    for worktree in directories(".claude/worktrees"):
        output = worktree + "/outputs"
        try:
            info = (base / output).lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            references.append(output)
        elif stat.S_ISDIR(info.st_mode):
            unmanaged.append(output)
            legacy_outputs.append(
                {
                    "source": output,
                    "destination": ".worktree-outputs/" + PurePosixPath(worktree).name,
                }
            )
        else:
            errors.append(f"unsupported legacy output path: {base / output}")
    for root in output_roots:
        for task in directories(root):
            if (
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", PurePosixPath(task).name)
                is None
            ):
                unmanaged.append(task)
                continue
            for purpose in directories(task):
                if PurePosixPath(purpose).name not in PURPOSES:
                    unmanaged.append(purpose)
                    continue
                for experiment in directories(purpose):
                    if (
                        re.fullmatch(
                            r"[A-Za-z0-9][A-Za-z0-9_.-]*",
                            PurePosixPath(experiment).name,
                        )
                        is None
                    ):
                        unmanaged.append(experiment)
                        continue
                    if any(
                        not stat.S_ISDIR((base / experiment / name).lstat().st_mode)
                        for name in names(base / experiment)
                    ):
                        unmanaged.append(experiment)
                        continue
                    for run in directories(experiment):
                        if (
                            re.fullmatch(
                                r"[A-Za-z0-9][A-Za-z0-9_.-]*", PurePosixPath(run).name
                            )
                            is None
                        ):
                            unmanaged.append(run)
                        else:
                            found.add(Unit(run, "output"))
    if os.path.lexists(base / "ckpt"):
        try:
            real_directory(base / "ckpt", anchor)
            found.add(Unit("ckpt", "checkpoint-files"))
            for path in directories("ckpt"):
                found.add(Unit(path, "checkpoint"))
            unmanaged[:] = [
                path
                for path in unmanaged
                if not (
                    Path(path).parent == Path("ckpt")
                    and stat.S_ISREG((base / path).lstat().st_mode)
                )
            ]
        except (OSError, StorageError) as exc:
            errors.append(str(exc))
    paths = sorted(found)
    for index, unit in enumerate(paths):
        if unit.kind != "checkpoint-files" and any(
            PurePosixPath(other.path).is_relative_to(unit.path)
            for other in paths[index + 1 :]
        ):
            errors.append(f"overlapping units: {unit.path}")
    return Inventory(
        paths,
        sorted(set(unmanaged)),
        sorted(set(errors)),
        sorted(references),
        legacy_outputs,
    )


def main_checkout(path: Path) -> Path:
    common = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        text=True,
    ).strip()
    return Path(common).parent


def prepare_worktree(
    config: Config, worktree: Path, *, dry_run: bool
) -> dict[str, Any]:
    worktree = worktree.resolve()
    if (
        worktree.parent != config.repo / ".claude/worktrees"
        or main_checkout(worktree) != config.repo
    ):
        raise StorageError(
            "worktree must be registered directly in the main checkout's .claude/worktrees"
        )
    if worktree.stat().st_dev != config.repo.stat().st_dev:
        raise StorageError("worktree is not on D:")
    for name in ("data", "outputs", "ckpt"):
        real_directory(config.repo / name, config.repo)
    actions = []
    output = worktree / "outputs"
    # The worktree link always names the main checkout path; with E:-resident
    # outputs that path resolves through the validated root link.
    persistent = config.repo / WORKTREE_OUTPUTS / worktree.name
    tier = "E:" if config.worktree_outputs_on_mirror else "D:"
    if config.worktree_outputs_on_mirror:
        real_root = worktree_outputs_root(config)
        resolved = real_root / worktree.name
        if os.path.lexists(resolved):
            real_directory(resolved, config.mount)
    else:
        resolved = persistent
        for path in (persistent.parent, persistent):
            if os.path.lexists(path):
                real_directory(path, config.repo)
                if path.stat().st_dev != config.repo.stat().st_dev:
                    raise StorageError("persistent worktree outputs are not on D:")
    if os.path.lexists(output):
        if dry_run and output.is_symlink():
            raise StorageError(
                "dry-run cannot inspect an existing outputs symlink target without changing its atime; use actual prepare-worktree for validation"
            )
        if not output.is_symlink() or output.resolve() not in {
            resolved,
            config.repo / "outputs",
        }:
            raise StorageError(
                f"existing worktree outputs must point to its persistent {tier} directory or main outputs"
            )
        target = output.resolve()
    else:
        target = resolved
        actions.append(f"link outputs to persistent {tier} {resolved}")
    for name in ("data", "ckpt", ".venv"):
        destination = worktree / name
        source = config.repo / name
        if os.path.lexists(destination):
            if dry_run and destination.is_symlink():
                raise StorageError(
                    f"dry-run cannot inspect an existing {name} symlink target without changing its atime; use actual prepare-worktree for validation"
                )
            if (
                not destination.is_symlink()
                or destination.resolve() != source.resolve()
            ):
                raise StorageError(
                    f"existing worktree path is not the shared {name} link: {destination}"
                )
        else:
            actions.append(f"link {name} to main checkout")
    # Validate the complete plan before making any changes.
    if not dry_run:
        if target == resolved:
            if config.worktree_outputs_on_mirror:
                # The validated root exists; never create E: ancestors.
                resolved.mkdir(exist_ok=True)
                real_directory(resolved, config.mount)
            else:
                persistent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(output):
            output.symlink_to(persistent, target_is_directory=True)
        for name in ("data", "ckpt", ".venv"):
            if not os.path.lexists(worktree / name):
                (worktree / name).symlink_to(
                    config.repo / name, target_is_directory=True
                )
    return {
        "worktree": str(worktree),
        "output_root": str(target),
        # E:-resident outputs have no separate mirror copy.
        "mirror_output_root": None
        if config.worktree_outputs_on_mirror and target == resolved
        else str(config.mirror / target.relative_to(config.repo)),
        "dry_run": dry_run,
        "actions": actions,
    }
