from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.registry import Registry
from src.tasks.base.annotation.storage import (
    InventoryScope,
    ManagedArtifact,
    Protection,
    Retention,
    StorageCatalog,
    plan_gc,
    save_dry_run,
)
from src.tasks.base.annotation.storage.contracts import GCPlan, Inventory
from src.tasks.base.annotation.storage.inventory import inventory, scope_path


@pytest.fixture
def catalog(tmp_path: Path) -> StorageCatalog:
    return StorageCatalog(AnnotationLayout.testing(tmp_path / "annotation"))


def artifact(
    catalog: StorageCatalog,
    name: str,
    *,
    retain: bool = False,
    expires_at: int = 0,
    dependencies: tuple[ArtifactRef, ...] = (),
    stage: str = "decode",
) -> ManagedArtifact:
    ref = write_asset(
        catalog.layout,
        catalog.layout.path("migration_staging", name),
        name.encode() * 1024,
    )
    item = ManagedArtifact(
        reference=ref,
        owner="fixture",
        stage=stage,
        kind="cache",
        regenerable=True,
        retention=Retention(mode="retain", reason="explicit fixture retention")
        if retain
        else Retention(
            mode="expire", expires_at_ns=expires_at, reason="explicit fixture expiry"
        ),
        dependencies=dependencies,
    )
    catalog.register(item)
    return item


def scan(catalog: StorageCatalog) -> Inventory:
    return inventory(
        catalog.layout,
        (InventoryScope(role="annotation", path="migration/staging"),),
        max_entries=100,
    )


def reasons(
    catalog: StorageCatalog, plan: GCPlan, item: ManagedArtifact
) -> tuple[str, ...]:
    return next(
        row.reasons for row in plan.decisions if row.path == catalog.key(item.reference)
    )


def test_unknown_and_missing_retention_are_protected(catalog: StorageCatalog) -> None:
    expired = artifact(catalog, "expired")
    retained = artifact(catalog, "retained", retain=True)
    future = artifact(catalog, "future", expires_at=time.time_ns() + 10**12)
    unknown = catalog.layout.path("migration_staging", "unknown")
    unknown.write_bytes(b"not registered")
    observed = scan(catalog)
    plan = plan_gc(catalog, observed)
    assert reasons(catalog, plan, expired) == ("expired_and_unreferenced_in_catalog",)
    assert reasons(catalog, plan, retained) == ("retention:retain",)
    assert reasons(catalog, plan, future) == ("retention:not_expired",)
    assert "ownership_unregistered" in next(
        item.reasons for item in plan.decisions if item.path == str(unknown)
    )
    assert plan.execution_ready is False
    assert (
        plan.eligible_allocated_bytes
        == Path(catalog.key(expired.reference)).stat().st_blocks * 512
    )
    with pytest.raises(ValueError, match="deadline"):
        Retention(mode="expire", reason="no chosen retention")
    with pytest.raises(ValueError, match="deadline"):
        Retention(mode="retain", expires_at_ns=0, reason="contradiction")


@pytest.mark.parametrize(
    "kind", ["adoption", "publication", "training", "model", "active", "reader"]
)
def test_protection_reaches_dependencies_and_release_is_audited(
    catalog: StorageCatalog, kind: str
) -> None:
    source = artifact(catalog, "source")
    clip = artifact(catalog, "clip", dependencies=(source.reference,))
    # Parse JSON to exercise the public strict schema with parametrized literal values.
    protection = Protection.model_validate_json(
        '{"protection_id":"live","kind":"' + kind + '","reason":"fixture",'
        '"references":[' + clip.reference.canonical_bytes().decode() + "]}"
    )
    catalog.protect(protection)
    before = catalog.snapshot().digest
    plan = plan_gc(catalog, scan(catalog))
    assert plan.eligible_allocated_bytes == 0
    assert any(
        item.startswith("referenced_by_retained:")
        for item in reasons(catalog, plan, source)
    )
    with pytest.raises(ValueError, match="identity"):
        catalog.release("live", expected_digest="0" * 64, reason="done")
    catalog.release("live", expected_digest=protection.sha256, reason="owner completed")
    catalog.release("live", expected_digest=protection.sha256, reason="owner completed")
    assert catalog.snapshot().digest != before
    assert plan_gc(catalog, scan(catalog)).eligible_allocated_bytes > 0
    with pytest.raises(ValueError, match="cannot be reused"):
        catalog.protect(protection)


def test_retained_manifest_outside_inventory_protects_dependency(
    catalog: StorageCatalog,
) -> None:
    shard = artifact(catalog, "shard")
    manifest = write_asset(
        catalog.layout, catalog.layout.path("models", "manifest"), b"manifest"
    )
    catalog.register(
        ManagedArtifact(
            reference=manifest,
            owner="fixture",
            stage="publish",
            kind="snapshot",
            regenerable=False,
            retention=Retention(
                mode="expire", expires_at_ns=0, reason="outside this inventory"
            ),
            dependencies=(shard.reference,),
        )
    )
    assert plan_gc(catalog, scan(catalog)).eligible_allocated_bytes == 0


def test_hardlinks_count_once_and_shared_stages_are_not_additive(
    catalog: StorageCatalog,
) -> None:
    original = artifact(catalog, "original", stage="decode")
    first = Path(catalog.key(original.reference))
    second = catalog.layout.path("migration_staging", "linked")
    os.link(first, second)
    linked = original.model_copy(
        update={
            "reference": annotation_reference(catalog.layout, second),
            "stage": "publish",
        }
    )
    catalog.register(linked)
    observed = scan(catalog)
    assert observed.files == 2
    assert observed.hardlinked_files == 2
    assert observed.unique_inodes == 2  # directory plus shared file inode
    plan = plan_gc(catalog, observed)
    assert plan.eligible_logical_bytes == first.stat().st_size * 2
    assert plan.eligible_allocated_bytes == first.stat().st_blocks * 512
    assert [
        item.shared_inode_bytes for item in plan.stages if item.stage != "unregistered"
    ] == [first.stat().st_blocks * 512] * 2
    assert plan.eligible_allocated_bytes < sum(
        item.allocated_bytes for item in plan.stages
    )


@pytest.mark.parametrize("external", [False, True])
def test_unowned_or_external_hardlink_protects_all_names(
    catalog: StorageCatalog, external: bool
) -> None:
    item = artifact(catalog, "shared")
    link = (
        catalog.layout.repository / "external"
        if external
        else catalog.layout.path("migration_staging", "unowned")
    )
    os.link(catalog.key(item.reference), link)
    plan = plan_gc(catalog, scan(catalog))
    assert plan.eligible_allocated_bytes == 0
    assert any("hardlink" in reason for reason in reasons(catalog, plan, item))


def test_hardlink_protection_propagates_back_through_references(
    catalog: StorageCatalog,
) -> None:
    leaf = artifact(catalog, "leaf")
    middle = artifact(catalog, "middle", dependencies=(leaf.reference,))
    os.link(
        catalog.key(middle.reference),
        catalog.layout.path("migration_staging", "unknown"),
    )
    assert plan_gc(catalog, scan(catalog)).eligible_allocated_bytes == 0


def test_symlinks_are_counted_without_following_targets(
    catalog: StorageCatalog,
) -> None:
    artifact(catalog, "managed")
    outside = catalog.layout.repository / "outside"
    outside.mkdir()
    (outside / "large").write_bytes(b"x" * 65536)
    catalog.layout.path("migration_staging", "symlink").symlink_to(
        outside, target_is_directory=True
    )
    observed = scan(catalog)
    assert observed.symlinks == 1 and observed.files == 1
    assert not any(item.path.startswith(str(outside)) for item in observed.entries)
    with pytest.raises(ValueError, match="symlink"):
        inventory(
            catalog.layout,
            (InventoryScope(role="annotation", path="migration/staging/symlink"),),
            max_entries=10,
        )
    with pytest.raises(ValueError, match="relative"):
        InventoryScope(role="annotation", path="../outside")


def test_inventory_bounded_and_rejects_overlapping_scopes(
    catalog: StorageCatalog,
) -> None:
    artifact(catalog, "item")
    scope = InventoryScope(role="annotation", path="migration/staging")
    with pytest.raises(ValueError, match="entry budget"):
        inventory(catalog.layout, (scope,), max_entries=1)
    with pytest.raises(ValueError, match="overlap"):
        inventory(catalog.layout, (scope, scope), max_entries=100)


@pytest.mark.parametrize("max_entries", [1, 3])
@pytest.mark.parametrize("nested", [False, True])
def test_inventory_stops_consuming_entries_when_budget_is_exceeded(
    catalog: StorageCatalog,
    monkeypatch: pytest.MonkeyPatch,
    max_entries: int,
    nested: bool,
) -> None:
    root = catalog.layout.path("migration_staging")
    directory = root / "nested" if nested else root
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(10):
        (directory / f"item-{index}").touch()
    original_scandir = os.scandir
    consumed = 0
    opened = 0
    closed = 0

    @contextmanager
    def bounded_scandir(descriptor: int) -> Iterator[Iterator[os.DirEntry[str]]]:
        nonlocal opened, closed

        def entries(iterator: Iterator[os.DirEntry[str]]) -> Iterator[os.DirEntry[str]]:
            nonlocal consumed
            for entry in iterator:
                consumed += 1
                # The scope root uses one slot; one extra entry detects overflow.
                assert consumed <= max_entries, "inventory consumed beyond its budget"
                yield entry

        with original_scandir(descriptor) as iterator:
            opened += 1
            try:
                yield entries(iterator)
            finally:
                closed += 1

    monkeypatch.setattr(os, "scandir", bounded_scandir)
    with pytest.raises(ValueError, match="entry budget"):
        inventory(
            catalog.layout,
            (InventoryScope(role="annotation", path="migration/staging"),),
            max_entries=max_entries,
        )
    assert consumed == max_entries
    assert opened == closed


@pytest.mark.parametrize("names", [(), ("z", "a")])
def test_inventory_accepts_exact_entry_budget_and_sorts_report(
    catalog: StorageCatalog, names: tuple[str, ...]
) -> None:
    root = catalog.layout.path("migration_staging")
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).touch()
    observed = inventory(
        catalog.layout,
        (InventoryScope(role="annotation", path="migration/staging"),),
        max_entries=1 + len(names),
    )
    assert [entry.path for entry in observed.entries] == sorted(
        str(path) for path in (root, *(root / name for name in names))
    )


@pytest.mark.parametrize("ancestor", ["", "migration", "migration/staging"])
def test_inventory_rejects_symlink_replacement_after_scope_validation(
    catalog: StorageCatalog, monkeypatch: pytest.MonkeyPatch, ancestor: str
) -> None:
    artifact(catalog, "managed")
    layout = catalog.layout
    replaced = layout.root / ancestor
    outside = layout.repository / "outside"
    outside_scope = outside / layout.path("migration_staging").relative_to(replaced)
    outside_scope.mkdir(parents=True)
    (outside_scope / "external").write_bytes(b"must not be inventoried")
    module = import_module("src.tasks.base.annotation.storage.inventory")

    def replace_ancestor(layout: AnnotationLayout, scope: InventoryScope) -> Path:
        root = scope_path(layout, scope)
        replaced.rename(layout.repository / "original")
        replaced.symlink_to(outside, target_is_directory=True)
        return root

    monkeypatch.setattr(module, "scope_path", replace_ancestor)
    with pytest.raises((OSError, ValueError)):
        scan(catalog)


@pytest.mark.parametrize("mutation", ["content", "unlink", "replace", "link"])
def test_inventory_and_content_changes_protect_candidate(
    catalog: StorageCatalog, mutation: str
) -> None:
    item = artifact(catalog, "item")
    observed = scan(catalog)
    path = Path(catalog.key(item.reference))
    if mutation == "unlink":
        path.unlink()
    elif mutation == "replace":
        path.unlink()
        path.write_bytes(b"other")
    elif mutation == "link":
        os.link(path, catalog.layout.repository / "newlink")
    else:
        path.write_bytes(b"edited")
    plan = plan_gc(catalog, observed)
    assert plan.eligible_allocated_bytes == 0
    assert "changed_since_inventory" in reasons(catalog, plan, item)
    if path.exists() and mutation != "link":
        assert "registered_content_unverified" in reasons(
            catalog, plan_gc(catalog, scan(catalog)), item
        )


def test_new_reference_invalidates_previous_catalog_view(
    catalog: StorageCatalog,
) -> None:
    item = artifact(catalog, "item")
    observed = scan(catalog)
    old = plan_gc(catalog, observed)
    catalog.protect(
        Protection(
            protection_id="new-reader",
            kind="reader",
            reason="reading",
            references=(item.reference,),
        )
    )
    new = plan_gc(catalog, observed)
    assert old.eligible_allocated_bytes > 0 and new.eligible_allocated_bytes == 0
    assert old.catalog_digest != new.catalog_digest
    assert old.execution_ready is False


def test_business_transaction_rolls_back_reference_registration(
    catalog: StorageCatalog,
) -> None:
    item = artifact(catalog, "item")
    protection = Protection(
        protection_id="published",
        kind="publication",
        reason="snapshot receipt",
        references=(item.reference,),
    )
    registry = Registry(catalog.layout)
    before = catalog.snapshot().digest
    with (
        pytest.raises(RuntimeError, match="simulated crash"),
        registry.transaction() as database,
    ):
        catalog.protect(protection, database=database)
        raise RuntimeError("simulated crash")
    assert catalog.snapshot().digest == before
    with registry.transaction() as database:
        catalog.protect(protection, database=database)
    assert catalog.snapshot().protections == (protection,)
    with sqlite3.connect(":memory:") as database:
        with pytest.raises(ValueError, match="active transaction"):
            catalog.protect(protection, database=database)
        database.execute("BEGIN")
        with pytest.raises(ValueError, match="another registry"):
            catalog.protect(protection, database=database)


def test_registration_is_verified_immutable_and_preserves_role_boundaries(
    catalog: StorageCatalog,
) -> None:
    item = artifact(catalog, "item")
    catalog.register(item)
    with pytest.raises(ValueError, match="immutable"):
        catalog.register(item.model_copy(update={"stage": "different"}))
    with pytest.raises(ValueError, match="hash"):
        catalog.register(
            item.model_copy(
                update={
                    "reference": item.reference.model_copy(update={"digest": "0" * 64})
                }
            )
        )
    with pytest.raises(ValueError, match="distinct other"):
        catalog.register(item.model_copy(update={"dependencies": (item.reference,)}))
    # A production-shaped test layout makes annotation/artifact roots overlap.
    layout = AnnotationLayout(
        repository=catalog.layout.repository,
        root=catalog.layout.repository / "outputs" / "annotation",
        mode="test",
    )
    ref = write_asset(layout, layout.path("migration_staging", "same"), b"same")
    alias = ref.model_copy(
        update={"role": "artifact", "path": "annotation/" + ref.path}
    )
    alias_catalog = StorageCatalog(layout)
    with pytest.raises(ValueError, match="root-prefixed"):
        alias_catalog.key(alias)


def test_reports_are_immutable_and_match_exact_inventory(
    catalog: StorageCatalog,
) -> None:
    artifact(catalog, "item")
    observed = scan(catalog)
    plan = plan_gc(catalog, observed)
    before = catalog.snapshot().digest
    refs = save_dry_run(catalog, observed, plan)
    assert read_document(catalog.layout, refs[0], Inventory) == observed
    assert read_document(catalog.layout, refs[1], GCPlan) == plan
    assert save_dry_run(catalog, observed, plan) == refs
    assert catalog.snapshot().digest == before
    with pytest.raises(ValueError, match="do not match"):
        save_dry_run(catalog, observed.model_copy(update={"created_at_ns": 0}), plan)
    with pytest.raises(ValueError, match="predate"):
        plan_gc(catalog, observed, now_ns=0)
