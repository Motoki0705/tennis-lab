"""Typed roots and deterministic resolution for all runtime path roles."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

from src.utils.configuration.errors import PathContractError
from src.utils.configuration.schema import ConfigField, StrictConfigSchema

__all__ = [
    "BoundaryPathField",
    "NonHydraPathBoundary",
    "PathDirection",
    "PathKind",
    "PathResolver",
    "PathRole",
    "ResolvedBoundaryPath",
    "ResolvedBoundaryPaths",
    "RuntimePathRoots",
]


class PathRole(StrEnum):
    """Every root role supported by the shared runtime path contract."""

    PROJECT = "project"
    DATA = "data"
    CHECKPOINT = "checkpoint"
    ARTIFACT = "artifact"
    OUTPUT = "output"
    CACHE = "cache"
    EXTERNAL_ASSET = "external_asset"


class PathDirection(StrEnum):
    """Whether a boundary consumes or produces the declared path."""

    INPUT = "input"
    OUTPUT = "output"


class PathKind(StrEnum):
    """Filesystem object kind required by an explicit boundary path."""

    ANY = "any"
    FILE = "file"
    DIRECTORY = "directory"


_ROOT_SCHEMA = StrictConfigSchema(
    name="paths",
    fields={f"{role.value}_root": ConfigField.of(str) for role in PathRole},
)

_LEGACY_ROOT_ALIASES = frozenset(
    {
        ".cache",
        "artifact",
        "artifacts",
        "build",
        "cache",
        "checkpoint",
        "checkpoints",
        "ckpt",
        "data",
        "external",
        "external_asset",
        "output",
        "outputs",
        "project",
        "third_party",
    }
)


def _validated_root_text(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise AssertionError(f"Strict root schema returned non-string {name}.")
    if not value.strip() or value != value.strip():
        raise PathContractError(f"{name} must be a non-empty trimmed path string.")
    return value


def _validated_child_part(
    role: PathRole,
    value: str | Path,
    *,
    forbidden_prefixes: frozenset[str],
) -> Path:
    rendered = str(value)
    if not rendered.strip() or rendered != rendered.strip():
        raise PathContractError(
            f"Derived {role.value} path parts must be non-empty and trimmed; "
            f"got {value!r}."
        )
    path = Path(value)
    if path.is_absolute():
        raise PathContractError(
            f"Derived {role.value} path parts must be relative; got {value!r}."
        )
    if path in {Path("."), Path("..")}:
        raise PathContractError(
            f"Derived {role.value} paths must identify a child below the role root; "
            f"got {value!r}."
        )
    first_part = path.parts[0].casefold() if path.parts else ""
    if first_part in forbidden_prefixes:
        raise PathContractError(
            f"Derived {role.value} path uses a root-prefixed or legacy fragment "
            f"{value!r}; pass a role-relative child without {path.parts[0]!r}."
        )
    return path


def _absolute(path: str | Path, *, relative_to: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    return candidate.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class RuntimePathRoots:
    """Resolved absolute roots for project data and produced/third-party assets."""

    project_root: Path
    data_root: Path
    checkpoint_root: Path
    artifact_root: Path
    output_root: Path
    cache_root: Path
    external_asset_root: Path

    def __post_init__(self) -> None:
        for root_field in fields(self):
            value = getattr(self, root_field.name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise PathContractError(
                    f"{root_field.name} must be an absolute pathlib.Path; got {value!r}."
                )
            if value == Path(value.anchor):
                raise PathContractError(
                    f"{root_field.name} must not grant the filesystem root as a "
                    "runtime path authority."
                )
            if value.resolve(strict=False) != value:
                raise PathContractError(
                    f"{root_field.name} must be a resolved absolute path; got {value!r}."
                )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        repository_root: Path,
    ) -> RuntimePathRoots:
        """Validate root keys and resolve them independently of the process CWD.

        ``project_root`` is relative to the explicit repository root. Every
        other relative root is relative to the resolved project root.
        """
        if not repository_root.is_absolute():
            raise PathContractError("repository_root must be absolute.")
        validated = _ROOT_SCHEMA.validate(value)
        project = _absolute(
            _validated_root_text(validated["project_root"], name="project_root"),
            relative_to=repository_root,
        )
        resolved = {
            key: _absolute(
                _validated_root_text(raw, name=key),
                relative_to=project,
            )
            for key, raw in validated.items()
            if key != "project_root"
        }
        return cls(project_root=project, **resolved)

    def root(self, role: PathRole) -> Path:
        """Return the root assigned to ``role``."""
        value = getattr(self, f"{role.value}_root")
        if not isinstance(value, Path):
            raise AssertionError(f"Runtime path root {role.value!r} is not a Path.")
        return value

    def forbidden_child_prefixes(self) -> frozenset[str]:
        """Return every prefix reserved for a root rather than a child path.

        A child fragment may not repeat a static legacy directory, a role/root
        key, or the basename of any configured root.  Computing the set from
        all seven roots keeps custom layouts subject to the same policy as the
        repository defaults and prevents role selection through path spelling.
        """
        configured_basenames = {
            root.name.casefold() for role in PathRole if (root := self.root(role)).name
        }
        role_names = {
            name.casefold()
            for role in PathRole
            for name in (role.value, f"{role.value}_root")
        }
        return frozenset(
            alias.casefold() for alias in _LEGACY_ROOT_ALIASES
        ) | configured_basenames | role_names

    def as_mapping(self) -> Mapping[str, str]:
        """Serialize the complete absolute root contract for a subprocess."""
        return {f"{role.value}_root": str(self.root(role)) for role in PathRole}


@dataclass(frozen=True, slots=True)
class PathResolver:
    """Resolve derived runtime paths beneath one explicitly selected root."""

    roots: RuntimePathRoots

    def resolve(self, role: PathRole, *relative_parts: str | Path) -> Path:
        """Resolve a non-empty relative path without permitting root escape."""
        return self.resolve_beneath(role, self.roots.root(role), *relative_parts)

    def resolve_configured(self, role: PathRole, value: str | Path) -> Path:
        """Resolve one configured path without weakening its declared role root.

        Role-relative values are resolved beneath the configured root. Absolute
        values are accepted only when they already lie strictly below that same
        root. No process-CWD or project-root fallback is applied.
        """
        rendered = str(value)
        if not rendered.strip() or rendered != rendered.strip():
            raise PathContractError(
                f"Configured {role.value} path must be non-empty and trimmed; "
                f"got {value!r}."
            )
        candidate = Path(value)
        if not candidate.is_absolute():
            return self.resolve(role, candidate)
        resolved = self.validate(role, candidate)
        if resolved == self.roots.root(role):
            raise PathContractError(
                f"Configured {role.value} path must identify a child below its "
                f"declared root; got {value!r}."
            )
        return resolved

    def resolve_beneath(
        self,
        role: PathRole,
        parent: Path,
        *relative_parts: str | Path,
    ) -> Path:
        """Resolve role-relative children beneath an already validated parent."""
        if not relative_parts:
            raise PathContractError("At least one relative path part is required.")
        root = self.roots.root(role)
        resolved_parent = self.validate(role, parent)
        candidate = resolved_parent
        forbidden_prefixes = self.roots.forbidden_child_prefixes()
        for part in relative_parts:
            path_part = _validated_child_part(
                role,
                part,
                forbidden_prefixes=forbidden_prefixes,
            )
            candidate /= path_part
        resolved = candidate.resolve(strict=False)
        if resolved == resolved_parent:
            raise PathContractError(
                f"Derived {role.value} path must identify a child below its "
                "declared parent."
            )
        if not resolved.is_relative_to(resolved_parent):
            raise PathContractError(
                f"Derived {role.value} path escapes its declared parent: {resolved} "
                f"(parent: {resolved_parent}, role root: {root})."
            )
        return resolved

    def resolve_symlink_entry(
        self,
        role: PathRole,
        relative_path: str | Path,
    ) -> Path:
        """Resolve a contained entry while preserving its final symlink identity.

        Virtual-environment Python launchers must remain symlink paths so the
        interpreter discovers that environment's site-packages. Every parent
        component is still resolved and required to stay beneath the role root;
        only the final entry is intentionally not dereferenced.
        """
        path_part = _validated_child_part(
            role,
            relative_path,
            forbidden_prefixes=self.roots.forbidden_child_prefixes(),
        )
        root = self.roots.root(role)
        lexical_entry = root.joinpath(path_part)
        resolved_parent = lexical_entry.parent.resolve(strict=False)
        if not resolved_parent.is_relative_to(root):
            raise PathContractError(
                f"Derived {role.value} symlink entry has a parent outside its root: "
                f"{resolved_parent} (root: {root})."
            )
        entry = resolved_parent.joinpath(lexical_entry.name)
        if entry == root:
            raise PathContractError(
                f"Derived {role.value} symlink entry must identify a child."
            )
        return entry

    def validate(self, role: PathRole, path: Path) -> Path:
        """Validate an already resolved absolute path against its declared root."""
        if not path.is_absolute():
            raise PathContractError(
                f"Validated {role.value} path must be absolute: {path}"
            )
        resolved = path.resolve(strict=False)
        root = self.roots.root(role)
        if not resolved.is_relative_to(root):
            raise PathContractError(
                f"Resolved {role.value} path is outside its root: {resolved} (root: {root})."
            )
        return resolved


@dataclass(frozen=True, slots=True)
class BoundaryPathField:
    """One explicit non-Hydra path parameter and its complete contract.

    Explicit boundary paths are deliberately distinct from configured,
    role-relative fragments accepted by :meth:`PathResolver.resolve`.  This
    contract accepts absolute paths only, so a CLI or subprocess boundary
    cannot silently reinterpret a relative value against the process CWD.
    """

    name: str
    role: PathRole
    direction: PathDirection
    kind: PathKind = PathKind.ANY
    must_exist: bool = False
    allow_role_root: bool = False
    many: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip() or self.name != self.name.strip():
            raise ValueError("Boundary path field names must be non-empty and trimmed.")
        if type(self.many) is not bool:
            raise TypeError("Boundary path field many must be exactly bool.")
        if self.direction is PathDirection.OUTPUT and self.must_exist:
            raise ValueError(
                f"Output boundary path {self.name!r} cannot require prior existence."
            )


@dataclass(frozen=True, slots=True)
class ResolvedBoundaryPath:
    """A validated absolute path retaining its role and I/O direction."""

    field: BoundaryPathField
    path: Path


@dataclass(frozen=True, slots=True)
class ResolvedBoundaryPaths(Mapping[str, Path | tuple[Path, ...]]):
    """Immutable named result of one non-Hydra path-boundary validation."""

    boundary_name: str
    entries: tuple[ResolvedBoundaryPath, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("Resolved boundary paths must not be empty.")
        for name in self:
            declared = tuple(
                value for value in self.entries if value.field.name == name
            )
            field = declared[0].field
            if any(value.field != field for value in declared):
                raise ValueError(
                    f"Resolved boundary field {name!r} has conflicting declarations."
                )
            if field.many:
                if len({value.path for value in declared}) != len(declared):
                    raise ValueError(
                        f"Resolved repeatable boundary field {name!r} contains duplicates."
                    )
            elif len(declared) != 1:
                raise ValueError(
                    f"Resolved scalar boundary field {name!r} must contain one path."
                )

    def __getitem__(self, name: str) -> Path | tuple[Path, ...]:
        declared = self._declared_values(name)
        if declared[0].field.many:
            return tuple(value.path for value in declared)
        return declared[0].path

    def __iter__(self) -> Iterator[str]:
        return iter(dict.fromkeys(value.field.name for value in self.entries))

    def __len__(self) -> int:
        return len(set(value.field.name for value in self.entries))

    def _declared_values(self, name: str) -> tuple[ResolvedBoundaryPath, ...]:
        declared = tuple(value for value in self.entries if value.field.name == name)
        if not declared:
            raise KeyError(name)
        return declared

    def declared(self, name: str) -> ResolvedBoundaryPath:
        """Return one scalar path with its role and direction declaration."""
        declared = self._declared_values(name)
        if declared[0].field.many:
            raise TypeError(
                f"Boundary path {name!r} is repeatable; use declared_many()."
            )
        return declared[0]

    def declared_many(self, name: str) -> tuple[ResolvedBoundaryPath, ...]:
        """Return an immutable non-empty group for a repeatable path field."""
        declared = self._declared_values(name)
        if not declared[0].field.many:
            raise TypeError(f"Boundary path {name!r} is scalar; use declared().")
        return declared


@dataclass(frozen=True, slots=True)
class NonHydraPathBoundary:
    """Strict path contract for argparse, callable, and subprocess boundaries.

    Callers first parse syntax without opening files or creating directories,
    then pass the complete path mapping here.  Validation closes the key set,
    rejects non-path/relative/escaping values, and checks requested input
    existence and kind before the caller performs any side effect.
    """

    name: str
    fields: tuple[BoundaryPathField, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip() or self.name != self.name.strip():
            raise ValueError("Non-Hydra boundary names must be non-empty and trimmed.")
        if not self.fields:
            raise ValueError(f"Non-Hydra boundary {self.name!r} has no path fields.")
        field_names = tuple(field.name for field in self.fields)
        if len(field_names) != len(set(field_names)):
            raise ValueError(
                f"Non-Hydra boundary {self.name!r} has duplicate path fields."
            )

    def validate(
        self,
        arguments: Mapping[str, object],
        *,
        resolver: PathResolver,
    ) -> ResolvedBoundaryPaths:
        """Validate the exact path mapping without changing the filesystem."""
        expected = {field.name for field in self.fields}
        actual = set(arguments)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise PathContractError(
                f"Invalid {self.name!r} path arguments: {'; '.join(details)}."
            )

        resolved_values: list[ResolvedBoundaryPath] = []
        for field in self.fields:
            raw_value = arguments[field.name]
            if field.many:
                if (
                    isinstance(raw_value, (str, bytes, Path))
                    or not isinstance(raw_value, Sequence)
                    or not raw_value
                ):
                    raise PathContractError(
                        f"{self.name}.{field.name} must be a non-empty sequence "
                        "of explicit str or pathlib.Path values."
                    )
                raw_paths: Sequence[object] = raw_value
            else:
                raw_paths = (raw_value,)
            field_values = tuple(
                self._validate_path(field, raw_path, resolver=resolver)
                for raw_path in raw_paths
            )
            if len(set(field_values)) != len(field_values):
                raise PathContractError(
                    f"{self.name}.{field.name} must not contain duplicate paths."
                )
            resolved_values.extend(
                ResolvedBoundaryPath(field=field, path=resolved)
                for resolved in field_values
            )
        return ResolvedBoundaryPaths(self.name, tuple(resolved_values))

    def _validate_path(
        self,
        field: BoundaryPathField,
        raw_path: object,
        *,
        resolver: PathResolver,
    ) -> Path:
        if type(raw_path) is not str and not isinstance(raw_path, Path):
            raise PathContractError(
                f"{self.name}.{field.name} values must be exactly str or pathlib.Path; "
                f"got {type(raw_path).__name__}."
            )
        if isinstance(raw_path, str):
            if not raw_path or "\x00" in raw_path:
                raise PathContractError(
                    f"{self.name}.{field.name} must be a non-empty valid path."
                )
            candidate = Path(raw_path)
        else:
            candidate = raw_path
        if not candidate.is_absolute():
            raise PathContractError(
                f"{self.name}.{field.name} must be an explicit absolute "
                f"{field.direction.value} path for role {field.role.value}; "
                f"got {candidate}."
            )
        resolved = resolver.validate(field.role, candidate)
        if not field.allow_role_root and resolved == resolver.roots.root(field.role):
            raise PathContractError(
                f"{self.name}.{field.name} must identify a path below the "
                f"{field.role.value} root, not the role root itself."
            )
        if field.must_exist and not resolved.exists():
            raise PathContractError(
                f"Required {field.direction.value} path does not exist for "
                f"{self.name}.{field.name}: {resolved}."
            )
        if field.must_exist and field.kind is PathKind.FILE and not resolved.is_file():
            raise PathContractError(
                f"{self.name}.{field.name} must be an existing file: {resolved}."
            )
        if (
            field.must_exist
            and field.kind is PathKind.DIRECTORY
            and not resolved.is_dir()
        ):
            raise PathContractError(
                f"{self.name}.{field.name} must be an existing directory: {resolved}."
            )
        return resolved
