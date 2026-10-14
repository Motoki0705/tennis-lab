"""Content-addressed migration plans; importing never authorizes source cleanup."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ...contracts import ArtifactRef, Digest, Document, Identifier, relative_path
from ..contracts import ArtifactKind
from ..contracts import after_validator as _after_validator

_Model = TypeVar("_Model")


class _AfterValidator(Protocol):
    def __call__(
        self, function: Callable[[_Model], _Model]
    ) -> Callable[[_Model], _Model]: ...


after_validator = cast(_AfterValidator, _after_validator)


class BundleFile(Document):
    path: str
    digest: Digest
    size_bytes: int = Field(ge=0)
    kind: ArtifactKind
    dependencies: tuple[str, ...] = ()

    @after_validator
    def paths_are_relative(self) -> Self:
        for path in (self.path, *self.dependencies):
            relative_path(path)
        if self.path in self.dependencies or len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise ValueError("bundle dependencies must be distinct other files")
        return self


class Bundle(Document):
    reader: Identifier
    selection_digest: Digest
    semantic_digest: Digest
    files: tuple[BundleFile, ...] = Field(min_length=1)

    @after_validator
    def complete_file_graph(self) -> Self:
        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("bundle files must have unique, sorted paths")
        known = set(paths)
        for item in self.files:
            if not set(item.dependencies) <= known:
                raise ValueError("bundle contains an unlisted dependency")
            if any(parent.as_posix() in known for parent in Path(item.path).parents):
                raise ValueError("a bundle file cannot also be a parent directory")
        return self


class MigrationReader(Protocol):
    """A task owns schema interpretation and explicitly selects a closed bundle."""

    def inspect(self, root: Path) -> Bundle: ...


class MigrationPlan(Document):
    schema_version: Literal["annotation.storage.migration-plan.v1"] = (
        "annotation.storage.migration-plan.v1"
    )
    root_id: Identifier
    source_root: str
    mode: Literal["in_place", "copy"]
    expected_previous: Digest | None
    bundle: Bundle
    # Upper bound for file payload copies; directories/journals need extra headroom.
    copy_payload_bytes: int = Field(ge=0)
    source_retained: Literal[True] = True

    @after_validator
    def explicit_source_and_size(self) -> Self:
        root = Path(self.source_root)
        if not root.is_absolute() or root.as_posix() != self.source_root:
            raise ValueError("migration source root must be a normalized absolute path")
        expected = (
            sum(item.size_bytes for item in self.bundle.files)
            if self.mode == "copy"
            else 0
        )
        if self.copy_payload_bytes != expected:
            raise ValueError("copy payload size differs from the selected files")
        return self


class MigrationReceipt(Document):
    schema_version: Literal["annotation.storage.migration-receipt.v1"] = (
        "annotation.storage.migration-receipt.v1"
    )
    plan: ArtifactRef
    bundle_digest: Digest
    files: tuple[ArtifactRef, ...] = Field(min_length=1)
    source_retained: Literal[True] = True
