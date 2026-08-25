"""Strict root and role-relative child path contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathContractError,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)


def _root_mapping() -> dict[str, str]:
    return {
        "project_root": ".",
        "data_root": "data",
        "checkpoint_root": "ckpt",
        "artifact_root": "artifacts",
        "output_root": "outputs",
        "cache_root": ".cache",
        "external_asset_root": "third_party",
    }


@pytest.mark.parametrize("blank", ["", " ", "\t"])
@pytest.mark.parametrize("root_name", [f"{role.value}_root" for role in PathRole])
def test_runtime_roots_reject_blank_strings(
    tmp_path: Path,
    root_name: str,
    blank: str,
) -> None:
    mapping = _root_mapping()
    mapping[root_name] = blank

    with pytest.raises(PathContractError, match="non-empty trimmed"):
        RuntimePathRoots.from_mapping(mapping, repository_root=tmp_path.resolve())


def test_runtime_roots_reject_unresolved_absolute_values(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    with pytest.raises(PathContractError, match="must be a resolved absolute path"):
        RuntimePathRoots(
            project_root=root / "nested/..",
            data_root=root / "data",
            checkpoint_root=root / "ckpt",
            artifact_root=root / "artifacts",
            output_root=root / "outputs",
            cache_root=root / ".cache",
            external_asset_root=root / "third_party",
        )


@pytest.mark.parametrize("root_name", [f"{role.value}_root" for role in PathRole])
def test_runtime_roots_reject_filesystem_root_authority(
    tmp_path: Path,
    root_name: str,
) -> None:
    mapping = _root_mapping()
    mapping[root_name] = "/"

    with pytest.raises(PathContractError, match="filesystem root"):
        RuntimePathRoots.from_mapping(mapping, repository_root=tmp_path.resolve())


@pytest.mark.parametrize("role", list(PathRole))
@pytest.mark.parametrize(
    "legacy_fragment",
    [
        "outputs/legacy",
        "data/legacy",
        "ckpt/legacy",
        "artifacts/legacy",
        ".cache/legacy",
        "third_party/legacy",
        "output/legacy",
        "checkpoint/legacy",
        "checkpoints/legacy",
        "cache/legacy",
        "external/legacy",
        "build/legacy",
    ],
)
def test_every_role_rejects_legacy_root_prefixed_children(
    tmp_path: Path,
    role: PathRole,
    legacy_fragment: str,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )

    with pytest.raises(PathContractError, match="root-prefixed or legacy"):
        resolver.resolve(role, legacy_fragment)


@pytest.mark.parametrize("role", list(PathRole))
@pytest.mark.parametrize(
    "reserved_fragment",
    [
        "project/work",
        "data/work",
        "checkpoint/work",
        "artifact/work",
        "output/work",
        "cache/work",
        "external_asset/work",
        "data_root/work",
        "external_asset_root/work",
    ],
)
def test_every_role_rejects_role_named_child_prefixes(
    tmp_path: Path,
    role: PathRole,
    reserved_fragment: str,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )

    with pytest.raises(PathContractError, match="root-prefixed or legacy"):
        resolver.resolve(role, reserved_fragment)


@pytest.mark.parametrize("role", list(PathRole))
@pytest.mark.parametrize(
    "custom_root_prefix",
    [
        "workspace-root/child",
        "samples-root/child",
        "weights-root/child",
        "records-root/child",
        "runs-root/child",
        "scratch-root/child",
        "vendors-root/child",
    ],
)
def test_every_role_rejects_every_configured_root_basename(
    tmp_path: Path,
    role: PathRole,
    custom_root_prefix: str,
) -> None:
    mapping = {
        "project_root": "workspace-root",
        "data_root": "samples-root",
        "checkpoint_root": "weights-root",
        "artifact_root": "records-root",
        "output_root": "runs-root",
        "cache_root": "scratch-root",
        "external_asset_root": "vendors-root",
    }
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(mapping, repository_root=tmp_path.resolve())
    )

    with pytest.raises(PathContractError, match="root-prefixed or legacy"):
        resolver.resolve(role, custom_root_prefix)


@pytest.mark.parametrize("blank", ["", " ", "\t", "."])
@pytest.mark.parametrize("role", list(PathRole))
def test_every_role_rejects_root_collapsing_children(
    tmp_path: Path,
    role: PathRole,
    blank: str,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )

    with pytest.raises(PathContractError, match="child|non-empty"):
        resolver.resolve(role, blank)


def test_project_source_child_remains_explicit_and_role_contained(
    tmp_path: Path,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )

    assert resolver.resolve(PathRole.PROJECT, "src/package/module.py") == (
        tmp_path / "src/package/module.py"
    )


def test_resolve_configured_accepts_relative_and_contained_absolute_paths(
    tmp_path: Path,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )
    absolute = resolver.roots.checkpoint_root / "run/model.ckpt"

    assert resolver.resolve_configured(PathRole.CHECKPOINT, "run/model.ckpt") == (
        absolute
    )
    assert resolver.resolve_configured(PathRole.CHECKPOINT, absolute) == absolute


@pytest.mark.parametrize("configured", ["", " ", ".", "../escape.ckpt"])
def test_resolve_configured_rejects_blank_root_and_escape_values(
    tmp_path: Path,
    configured: str,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )

    with pytest.raises(PathContractError, match="non-empty|child|escapes"):
        resolver.resolve_configured(PathRole.CHECKPOINT, configured)


def test_resolve_configured_rejects_absolute_path_from_another_role(
    tmp_path: Path,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )
    wrong_role = resolver.roots.output_root / "run/model.ckpt"

    with pytest.raises(PathContractError, match="outside its root"):
        resolver.resolve_configured(PathRole.CHECKPOINT, wrong_role)


@pytest.mark.parametrize("child", ["", " ", ".", "..", "nested/../.."])
def test_resolve_beneath_rejects_blank_root_and_escape_children(
    tmp_path: Path,
    child: str,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )
    parent = resolver.resolve(PathRole.DATA, "datasets")

    with pytest.raises(PathContractError, match="child|non-empty|escapes"):
        resolver.resolve_beneath(PathRole.DATA, parent, child)


def test_resolve_beneath_rejects_parent_from_another_role(tmp_path: Path) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )
    output_parent = resolver.resolve(PathRole.OUTPUT, "run")

    with pytest.raises(PathContractError, match="outside its root"):
        resolver.resolve_beneath(PathRole.DATA, output_parent, "dataset.json")


def test_resolve_symlink_entry_preserves_venv_launcher_but_contains_parent(
    tmp_path: Path,
) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )
    interpreter = tmp_path / "system/python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("python", encoding="utf-8")
    launcher = resolver.roots.external_asset_root / "runtime/.venv/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(interpreter)

    resolved = resolver.resolve_symlink_entry(
        PathRole.EXTERNAL_ASSET,
        "runtime/.venv/bin/python",
    )

    assert resolved == launcher
    assert resolved.is_symlink()
    assert resolved.resolve() == interpreter


def test_resolve_symlink_entry_rejects_parent_symlink_escape(tmp_path: Path) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped_parent = resolver.roots.external_asset_root / "escaped"
    escaped_parent.parent.mkdir(parents=True)
    escaped_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(PathContractError, match="parent outside its root"):
        resolver.resolve_symlink_entry(
            PathRole.EXTERNAL_ASSET,
            "escaped/python",
        )


def _non_hydra_boundary_fixture(
    tmp_path: Path,
) -> tuple[NonHydraPathBoundary, PathResolver, dict[str, object]]:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _root_mapping(), repository_root=tmp_path.resolve()
        )
    )
    source = resolver.resolve(PathRole.DATA, "samples/source.json")
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    assets = (
        resolver.resolve(PathRole.EXTERNAL_ASSET, "models/one.bin"),
        resolver.resolve(PathRole.EXTERNAL_ASSET, "models/two.bin"),
    )
    for asset in assets:
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"weights")
    output = resolver.resolve(PathRole.OUTPUT, "run/result.json")
    boundary = NonHydraPathBoundary(
        "tests.non_hydra",
        (
            BoundaryPathField(
                "source",
                PathRole.DATA,
                PathDirection.INPUT,
                kind=PathKind.FILE,
                must_exist=True,
            ),
            BoundaryPathField(
                "assets",
                PathRole.EXTERNAL_ASSET,
                PathDirection.INPUT,
                kind=PathKind.FILE,
                must_exist=True,
                many=True,
            ),
            BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT),
        ),
    )
    return boundary, resolver, {
        "source": str(source),
        "assets": assets,
        "output": output,
    }


def test_non_hydra_boundary_returns_typed_role_aware_paths(tmp_path: Path) -> None:
    boundary, resolver, arguments = _non_hydra_boundary_fixture(tmp_path)

    resolved = boundary.validate(arguments, resolver=resolver)

    source = arguments["source"]
    assert isinstance(source, str)
    assert resolved["source"] == Path(source)
    assert resolved["assets"] == arguments["assets"]
    assert resolved["output"] == arguments["output"]
    assert resolved.declared("source").field.role is PathRole.DATA
    assert resolved.declared("output").field.direction is PathDirection.OUTPUT
    assert all(
        entry.field.role is PathRole.EXTERNAL_ASSET
        for entry in resolved.declared_many("assets")
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"source": "/tmp/source", "assets": ["/tmp/asset"]},
        {
            "source": "/tmp/source",
            "assets": ["/tmp/asset"],
            "output": "/tmp/output",
            "typo": "/tmp/typo",
        },
    ],
)
def test_non_hydra_boundary_rejects_missing_and_unknown_keys(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    boundary, resolver, _ = _non_hydra_boundary_fixture(tmp_path)

    with pytest.raises(PathContractError, match="missing|unknown"):
        boundary.validate(arguments, resolver=resolver)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("source", "relative/source.json", "explicit absolute"),
        ("source", object(), "exactly str or pathlib.Path"),
        ("assets", [], "non-empty sequence"),
        ("assets", "one.bin", "non-empty sequence"),
    ],
)
def test_non_hydra_boundary_rejects_invalid_path_shapes(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    boundary, resolver, arguments = _non_hydra_boundary_fixture(tmp_path)
    arguments[field] = invalid

    with pytest.raises(PathContractError, match=message):
        boundary.validate(arguments, resolver=resolver)


def test_non_hydra_boundary_rejects_wrong_role_root_and_duplicate_paths(
    tmp_path: Path,
) -> None:
    boundary, resolver, arguments = _non_hydra_boundary_fixture(tmp_path)
    arguments["source"] = resolver.resolve(PathRole.OUTPUT, "wrong/source.json")

    with pytest.raises(PathContractError, match="outside its root"):
        boundary.validate(arguments, resolver=resolver)

    arguments["source"] = resolver.resolve(PathRole.DATA, "samples/source.json")
    arguments["output"] = resolver.roots.output_root
    with pytest.raises(PathContractError, match="not the role root itself"):
        boundary.validate(arguments, resolver=resolver)

    arguments["output"] = resolver.resolve(PathRole.OUTPUT, "run/result.json")
    asset = resolver.resolve(PathRole.EXTERNAL_ASSET, "models/one.bin")
    arguments["assets"] = (asset, asset)
    with pytest.raises(PathContractError, match="duplicate paths"):
        boundary.validate(arguments, resolver=resolver)


def test_non_hydra_boundary_checks_input_existence_and_kind(tmp_path: Path) -> None:
    boundary, resolver, arguments = _non_hydra_boundary_fixture(tmp_path)
    arguments["source"] = resolver.resolve(PathRole.DATA, "missing.json")

    with pytest.raises(PathContractError, match="does not exist"):
        boundary.validate(arguments, resolver=resolver)

    directory = resolver.resolve(PathRole.DATA, "directory")
    directory.mkdir(parents=True)
    arguments["source"] = directory
    with pytest.raises(PathContractError, match="existing file"):
        boundary.validate(arguments, resolver=resolver)
