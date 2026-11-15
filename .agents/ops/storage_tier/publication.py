"""Durable publication receipts and explicit recovery without discarding evidence."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .filesystem import (
    Snapshot,
    StorageError,
    names,
    read_json,
    real_directory,
    rename,
    snapshot,
    write_json,
)
from .layout import Unit, discover

if TYPE_CHECKING:
    from .engine import Engine


def evidence(engine: Engine, name: str, value: dict[str, Any]) -> Path:
    archive = engine.config.state / "publication-evidence" / name
    keys = (
        "unit",
        "kind",
        "operation",
        "expected",
        "copied",
        "previous",
        "previous_manifest",
    )
    plan = {key: value.get(key) for key in keys}
    if archive.exists():
        if read_json(archive).get("plan") != plan:
            raise StorageError(
                "publication evidence archive conflicts with the journal"
            )
        return archive
    unit = Unit(value["unit"], value["kind"])
    current = engine.manifest_path(unit)
    write_json(
        archive,
        {
            "schema_version": 1,
            "plan": plan,
            "journal": value,
            "previous_manifest": value.get("previous_manifest"),
            "manifest_at_recovery": read_json(current) if current.exists() else None,
        },
    )
    return archive


def prepare(
    engine: Engine,
    unit: Unit,
    operation: str,
    source: Path,
    destination: Path,
    stage: Path,
    expected: Snapshot,
    copied: Snapshot,
    previous: Snapshot | None,
    prior_manifest: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    path = engine.config.state / "publications" / f"{uuid.uuid4().hex}.json"
    value = {
        "schema_version": 1,
        "unit": unit.path,
        "kind": unit.kind,
        "operation": operation,
        "phase": "prepared",
        "source": str(source),
        "destination": str(destination),
        "stage": str(stage),
        "expected": expected.entries,
        "copied": copied.entries,
        "previous": previous.entries if previous is not None else None,
        "previous_manifest": prior_manifest,
    }
    write_json(path, value)
    return path, value


def complete(path: Path, value: dict[str, Any]) -> None:
    value["phase"] = "complete"
    write_json(path, value)


def pending(engine: Engine) -> list[Path]:
    folder = engine.config.state / "publications"
    return (
        [
            folder / name
            for name in names(folder)
            if read_json(folder / name).get("phase") != "complete"
        ]
        if folder.exists()
        else []
    )


def fresh(engine: Engine, unit: Unit, path: Path) -> Snapshot:
    before = snapshot(path)
    engine.quiet(unit, path, before)
    tree = snapshot(path, hashes=True)
    if not before.same_metadata(tree) or not tree.same_metadata(snapshot(path)):
        raise StorageError("publication changed during SHA-256 recovery verification")
    engine.quiet(unit, path, tree)
    return tree


def root_matches(tree: Snapshot, expected: dict[str, Any]) -> bool:
    return all(tree.entries["."][k] == expected["."][k] for k in ("dev", "ino"))


def recover(engine: Engine) -> list[dict[str, Any]]:
    results = []
    for journal in pending(engine):
        value = read_json(journal)
        if (
            value.get("schema_version") != 1
            or value.get("phase") != "prepared"
            or value.get("operation") not in {"mirror", "hydrate"}
        ):
            raise StorageError("invalid publication recovery journal")
        unit = Unit(value["unit"], value["kind"])
        if (
            unit
            not in discover(engine.config, engine.config.repo).units
            + discover(engine.config, engine.config.mirror).units
        ):
            raise StorageError("publication unit is not independently discovered")
        hydrate = value["operation"] == "hydrate"
        source = engine.path(unit, mirror=hydrate)
        destination = (
            engine.config.repo if hydrate else engine.config.mirror
        ) / unit.path
        stage = Path(value["stage"])
        base = engine.config.repo if hydrate else engine.config.mirror
        anchor = base if hydrate else engine.config.mount
        prefix = ".storage-tier-hydrate-" if hydrate else ".storage-tier-stage-"
        if (
            value["source"] != str(source)
            or value["destination"] != str(destination)
            or stage.parent != destination.parent
            or not stage.name.startswith(prefix)
        ):
            raise StorageError("invalid publication recovery paths")
        expected, copied = Snapshot(value["expected"]), Snapshot(value["copied"])
        original = fresh(engine, unit, source)
        if original.content() != expected.content():
            raise StorageError(
                "publication source SHA-256 differs from its verified intent"
            )
        target = (
            fresh(engine, unit, engine.path(unit, mirror=not hydrate))
            if destination.exists()
            else None
        )
        published = target is not None and root_matches(target, value["copied"])
        if not published:
            if not stage.exists():
                raise StorageError(
                    "verified publication stage is missing; preserve and reconcile before recovery"
                )
            real_directory(stage, anchor)
            staged = fresh(engine, unit, stage)
            if (
                not root_matches(staged, value["copied"])
                or staged.content() != copied.content()
            ):
                raise StorageError(
                    "publication stage SHA-256/identity differs from its intent"
                )
            previous = value["previous"]
            if (previous is None and target is not None) or (
                previous is not None
                and (
                    target is None
                    or not root_matches(target, previous)
                    or target.content() != Snapshot(previous).content()
                )
            ):
                raise StorageError(
                    "publication destination differs from the prior verified copy"
                )
            if target is not None:
                engine.quiet(unit, destination, target, destructive=True)
            engine.check_mount()
            rename(stage, destination, exchange=target is not None)
            target = fresh(engine, unit, engine.path(unit, mirror=not hydrate))
        if target is None or target.content() != expected.content():
            raise StorageError(
                "publication destination SHA-256 differs from the verified intent"
            )
        archive = evidence(engine, journal.name, value)
        old_tree = None
        if value["previous"] is not None:
            old_tree = base / ".storage-tier/recovered-publications" / journal.stem
            current = stage if stage.exists() else old_tree
            real_directory(current, anchor)
            info = current.lstat()
            prior = value["previous"]["."]
            if (info.st_dev, info.st_ino) != (prior["dev"], prior["ino"]):
                raise StorageError(
                    "old publication tree identity changed; preserving it"
                )
            if current == stage:
                engine.ensure_parent(old_tree.parent, base)
                rename(stage, old_tree)
        # Re-hash both current sides immediately before issuing new evidence.
        local = fresh(engine, unit, engine.path(unit))
        mirror = fresh(engine, unit, engine.path(unit, mirror=True))
        if (
            local.content() != expected.content()
            or mirror.content() != expected.content()
        ):
            raise StorageError("publication SHA-256 changed before recovery commit")
        engine.record(unit, local, mirror, dry_run=False)
        engine.clear_recovery(unit)
        value["evidence_archive"], value["old_tree_archive"] = (
            str(archive),
            str(old_tree) if old_tree else None,
        )
        complete(journal, value)
        engine.audit(
            "publication_recovered",
            unit,
            operation=value["operation"],
            evidence_archive=str(archive),
            old_tree_archive=value["old_tree_archive"],
        )
        results.append(
            {
                "unit": unit.path,
                "action": "publication_recovered",
                "evidence_archive": str(archive),
                "old_tree_archive": value["old_tree_archive"],
            }
        )
    return results
