"""One production annotation root, regardless of the current worktree."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

from .contracts import ArtifactRef, relative_path

Area = Literal[
    "sources",
    "clips",
    "predictions",
    "annotations",
    "campaigns",
    "cache",
    "migration_plans",
    "migration_staging",
    "migration_archive",
]
_AREAS: dict[Area, str] = {
    "sources": "media/sources",
    "clips": "media/clips",
    "predictions": "predictions",
    "annotations": "annotations",
    "campaigns": "campaigns",
    "cache": "cache",
    "migration_plans": "migration/plans",
    "migration_staging": "migration/staging",
    "migration_archive": "migration/archive",
}
_COMPONENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\Z")


def _no_symlinks(path: Path, boundary: Path) -> None:
    if not path.is_relative_to(boundary):
        raise ValueError("path escapes the declared boundary")
    for entry in (path, *path.parents):
        if entry.is_symlink():
            raise ValueError(f"annotation paths must not contain symlinks: {entry}")
        if entry == boundary:
            break


@dataclass(frozen=True, slots=True)
class AnnotationLayout:
    """Construct with production() or testing(); construction performs no writes."""

    repository: Path
    root: Path
    mode: Literal["production", "test"]

    def __post_init__(self) -> None:
        if self.mode not in {"production", "test"}:
            raise ValueError("explicit production or test mode is required")
        if not self.repository.is_absolute() or not self.root.is_absolute():
            raise ValueError("layout roots must be absolute")
        if (
            self.repository != self.repository.resolve()
            or self.root != self.root.absolute()
        ):
            raise ValueError("layout roots must be normalized")
        if (
            self.mode == "production"
            and self.root != self.repository / "outputs/chat_annotation"
        ):
            raise ValueError("production root is fixed to outputs/chat_annotation")
        _no_symlinks(self.root, self.repository)

    @classmethod
    def production(cls, source_root: Path = PROJECT_ROOT) -> AnnotationLayout:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
        repository = Path(result.stdout.strip()).resolve().parent
        return cls(repository, repository / "outputs/chat_annotation", "production")

    @classmethod
    def testing(cls, root: Path) -> AnnotationLayout:
        if not root.is_absolute():
            raise ValueError("test root must be explicitly absolute")
        return cls(root.parent.resolve(), root, "test")

    @property
    def resolver(self) -> PathResolver:
        repo = self.repository
        return PathResolver(
            RuntimePathRoots(
                project_root=repo,
                data_root=repo / "data",
                checkpoint_root=repo / "ckpt",
                artifact_root=repo / "outputs",
                output_root=self.root,
                cache_root=self.root / "cache",
                external_asset_root=repo / "third_party",
            )
        )

    def path(self, area: Area, *components: str) -> Path:
        if area not in _AREAS:
            raise ValueError(f"unregistered annotation output area: {area}")
        for part in components:
            if not _COMPONENT.fullmatch(part):
                raise ValueError("output identity must be one filename component")
        candidate = self.root / _AREAS[area]
        for part in components:
            candidate /= part
        _no_symlinks(candidate, self.repository)
        resolved: Path
        if area == "cache":
            resolved = self.resolver.validate(PathRole.CACHE, candidate)
        else:
            resolved = self.resolver.resolve_beneath(
                PathRole.OUTPUT, self.root, candidate.relative_to(self.root)
            )
        return resolved

    @property
    def registry(self) -> Path:
        candidate = self.root / "registry.sqlite3"
        _no_symlinks(candidate, self.repository)
        return candidate

    def artifact_path(self, reference: ArtifactRef) -> Path:
        relative_path(reference.path)
        role = {
            "annotation": PathRole.OUTPUT,
            "data": PathRole.DATA,
            "checkpoint": PathRole.CHECKPOINT,
            "artifact": PathRole.ARTIFACT,
        }[reference.role]
        candidate = self.resolver.roots.root(role) / reference.path
        _no_symlinks(candidate, self.repository)
        resolved: Path = self.resolver.resolve(role, reference.path)
        return resolved
