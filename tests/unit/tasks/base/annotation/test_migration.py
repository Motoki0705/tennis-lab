from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import content_digest
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage import InventoryScope, inventory, plan_gc
from src.tasks.base.annotation.storage.migration import (
    Bundle,
    MigrationPlan,
    MigrationReceipt,
    MigrationStore,
    describe_file,
    dry_run,
    imported_root,
)
from src.tasks.base.annotation.storage.migration import store as implementation
from src.tasks.base.annotation.storage.migration.files import read_file


class Reader:
    def inspect(self, root: Path) -> Bundle:
        files = tuple(
            describe_file(root, path, kind="source")
            for path in ("payload/a.bin", "payload/b.bin")
        )
        return Bundle(
            reader="synthetic",
            selection_digest="a" * 64,
            semantic_digest=content_digest(
                [(item.path, item.digest) for item in files]
            ),
            files=files,
        )


@pytest.fixture
def environment(tmp_path: Path) -> tuple[AnnotationLayout, Path, Reader, MigrationPlan]:
    source = tmp_path / "old"
    (source / "payload").mkdir(parents=True)
    (source / "payload/a.bin").write_bytes(b"first source" * 1024)
    (source / "payload/b.bin").write_bytes(b"second source" * 1024)
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    reader = Reader()
    plan = dry_run(
        layout,
        reader,
        root_id="old-campaign",
        source_root=source,
        mode="copy",
        max_files=2,
    )
    return layout, source, reader, plan


def apply(store: MigrationStore, plan: MigrationPlan, reader: Reader) -> Any:
    return store.apply(
        plan, reader, max_copy_bytes=plan.copy_payload_bytes, max_files=2
    )


def test_dry_run_does_not_create_registry_or_touch_originals(environment: Any) -> None:
    layout, source, reader, plan = environment
    assert not layout.root.exists()
    before = {
        path: (path.read_bytes(), path.stat())
        for path in (source / "payload").iterdir()
    }
    assert plan.copy_payload_bytes == sum(len(value[0]) for value in before.values())
    assert plan == dry_run(
        layout,
        reader,
        root_id="old-campaign",
        source_root=source,
        mode="copy",
        max_files=2,
    )
    assert not layout.root.exists()
    for path, (payload, metadata) in before.items():
        assert path.read_bytes() == payload
        assert path.stat().st_ino == metadata.st_ino
        assert path.stat().st_mtime_ns == metadata.st_mtime_ns


def test_copy_preserves_bytes_reader_graph_and_source_independence(
    environment: Any,
) -> None:
    layout, source, reader, plan = environment
    store = MigrationStore(layout)
    reference = apply(store, plan, reader)
    receipt = read_document(layout, reference, MigrationReceipt)
    root = imported_root(layout, plan.root_id, reader)
    assert reader.inspect(root) == reader.inspect(source)
    assert root.is_relative_to(layout.path("migration_archive"))
    assert receipt.source_retained
    for item in plan.bundle.files:
        assert (root / item.path).stat().st_ino != (source / item.path).stat().st_ino
    assert apply(store, plan, reader) == reference
    observed = inventory(
        layout,
        (InventoryScope(role="annotation", path="migration/archive"),),
        max_entries=20,
    )
    assert plan_gc(store.catalog, observed).eligible_allocated_bytes == 0
    snapshot = store.catalog.snapshot()
    receipt_artifact = snapshot.artifacts[store.catalog.key(reference)]
    assert set(receipt.files) <= set(receipt_artifact.dependencies)
    assert len(snapshot.protections) == 1
    # The imported reader no longer relies on the external source tree.
    source.rename(source.with_name("old-still-preserved"))
    assert imported_root(layout, plan.root_id, reader) == root


def test_in_place_import_keeps_inode_and_rejects_later_mutation(
    environment: Any,
) -> None:
    _, source, reader, _ = environment
    layout = AnnotationLayout.testing(source)
    plan = dry_run(
        layout,
        reader,
        root_id="existing",
        source_root=source,
        mode="in_place",
        max_files=2,
    )
    inode = (source / "payload/a.bin").stat().st_ino
    store = MigrationStore(layout)
    store.apply(plan, reader, max_copy_bytes=0, max_files=2)
    assert imported_root(layout, "existing", reader) == source
    assert (source / "payload/a.bin").stat().st_ino == inode
    assert not layout.path("migration_archive").exists()
    (source / "payload/a.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt|hash|differs|size"):
        imported_root(layout, "existing", reader)


@pytest.mark.parametrize("interruption", ["copy", "rename", "transaction"])
def test_restart_recovers_without_partial_pointer(
    environment: Any, monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    layout, source, reader, plan = environment
    store = MigrationStore(layout)
    original_bundle = reader.inspect(source)
    with monkeypatch.context() as scoped:
        if interruption == "copy":
            original_copy = implementation.copy_file

            def fail_copy(*args: Any, **kwargs: Any) -> None:
                original_copy(*args, **kwargs)
                raise RuntimeError("interrupted")

            scoped.setattr(implementation, "copy_file", fail_copy)
        elif interruption == "rename":

            def fail_publish(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("interrupted")

            scoped.setattr(store, "_publish", fail_publish)
        else:
            original_protect = store.catalog.protect

            def fail_transaction(*args: Any, **kwargs: Any) -> None:
                original_protect(*args, **kwargs)
                raise RuntimeError("interrupted")

            scoped.setattr(store.catalog, "protect", fail_transaction)
        with pytest.raises(RuntimeError, match="interrupted"):
            apply(store, plan, reader)
    with sqlite3.connect(layout.registry) as database:
        assert database.execute("SELECT state FROM storage_migrations").fetchall() == [
            ("planned",)
        ]
        assert database.execute("SELECT * FROM storage_imported_roots").fetchall() == []
    assert store.catalog.snapshot().protections == ()
    assert reader.inspect(source) == original_bundle
    apply(MigrationStore(layout), plan, reader)
    assert (
        reader.inspect(imported_root(layout, plan.root_id, reader)) == original_bundle
    )


def test_source_changed_after_dry_run_is_not_registered(environment: Any) -> None:
    layout, source, reader, plan = environment
    (source / "payload/b.bin").write_bytes(b"new content")
    store = MigrationStore(layout)
    with pytest.raises(ValueError, match="differs from dry-run"):
        apply(store, plan, reader)
    assert not layout.path("migration_archive").exists()


def test_destination_changed_before_database_commit_cannot_change_pinned_hashes(
    environment: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _, reader, plan = environment
    store = MigrationStore(layout)
    original_publish = store._publish

    def change_target(*args: Any, **kwargs: Any) -> None:
        root = layout.path("migration_archive", plan.sha256, "root")
        (root / plan.bundle.files[0].path).write_bytes(
            b"changed after reader validation"
        )
        original_publish(*args, **kwargs)

    monkeypatch.setattr(store, "_publish", change_target)
    with pytest.raises(ValueError, match="hash|size"):
        apply(store, plan, reader)
    with pytest.raises(ValueError, match="no completed import"):
        imported_root(layout, plan.root_id, reader)


def test_in_place_imports_can_share_the_same_original_files(environment: Any) -> None:
    _, source, reader, _ = environment
    layout = AnnotationLayout.testing(source)
    store = MigrationStore(layout)
    for root_id in ("first", "second"):
        plan = dry_run(
            layout,
            reader,
            root_id=root_id,
            source_root=source,
            mode="in_place",
            max_files=2,
        )
        apply(store, plan, reader)
        assert imported_root(layout, root_id, reader) == source
    assert len(store.catalog.snapshot().protections) == 2


def test_source_change_during_copy_prevents_publication(
    environment: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, source, reader, plan = environment
    store = MigrationStore(layout)
    original_copy = implementation.copy_file

    def change_source(*args: Any, **kwargs: Any) -> None:
        original_copy(*args, **kwargs)
        (source / "payload/a.bin").write_bytes(b"concurrently changed")

    monkeypatch.setattr(implementation, "copy_file", change_source)
    with pytest.raises(ValueError, match="differs from dry-run"):
        apply(store, plan, reader)
    with pytest.raises(ValueError, match="no completed import"):
        imported_root(layout, plan.root_id, reader)


def test_pointer_cas_requires_new_plan_and_retains_previous_import(
    environment: Any,
) -> None:
    layout, source, reader, first = environment
    second_root = source.with_name("second")
    (second_root / "payload").mkdir(parents=True)
    (second_root / "payload/a.bin").write_bytes(b"different")
    (second_root / "payload/b.bin").write_bytes(b"also different")
    second = dry_run(
        layout,
        reader,
        root_id=first.root_id,
        source_root=second_root,
        mode="copy",
        max_files=2,
    )
    store = MigrationStore(layout)
    apply(store, first, reader)
    old_root = imported_root(layout, first.root_id, reader)
    with pytest.raises(ValueError, match="pointer changed"):
        apply(store, second, reader)
    replacement = dry_run(
        layout,
        reader,
        root_id=first.root_id,
        source_root=second_root,
        mode="copy",
        max_files=2,
        expected_previous=first.sha256,
    )
    apply(store, replacement, reader)
    assert imported_root(layout, first.root_id, reader) != old_root
    assert reader.inspect(old_root) == first.bundle
    assert len(store.catalog.snapshot().protections) == 2
    with pytest.raises(ValueError, match="pointer changed"):
        apply(store, first, reader)


def test_copy_and_file_limits_are_required(environment: Any) -> None:
    layout, source, reader, plan = environment
    store = MigrationStore(layout)
    with pytest.raises(ValueError, match="payload budget"):
        store.apply(
            plan, reader, max_copy_bytes=plan.copy_payload_bytes - 1, max_files=2
        )
    with pytest.raises(ValueError, match="file count"):
        store.apply(plan, reader, max_copy_bytes=plan.copy_payload_bytes, max_files=1)
    with pytest.raises(ValueError, match="file count"):
        dry_run(
            layout,
            reader,
            root_id="too-many",
            source_root=source,
            mode="copy",
            max_files=1,
        )


def test_extra_staging_content_is_not_published(environment: Any) -> None:
    layout, _, reader, plan = environment
    store = MigrationStore(layout)
    staging = layout.path("migration_staging", plan.sha256, "root")
    staging.mkdir(parents=True)
    (staging / "unplanned").write_bytes(b"not selected")
    with pytest.raises(ValueError, match="entry budget|unplanned"):
        apply(store, plan, reader)
    assert not layout.path("migration_archive", plan.sha256, "root").exists()


def test_source_symlink_fifo_and_replacement_fail_closed(environment: Any) -> None:
    _, source, _, _ = environment
    (source / "link").symlink_to(source / "payload/a.bin")
    with pytest.raises((ValueError, OSError)):
        describe_file(source, "link", kind="source")
    os.mkfifo(source / "fifo")
    with pytest.raises(ValueError, match="regular files"):
        describe_file(source, "fifo", kind="source")
    with (
        pytest.raises(ValueError, match="changed during reading"),
        read_file(source, "payload/a.bin") as handle,
    ):
        assert handle.read(1)
        (source / "payload/a.bin").write_bytes(b"replacement")


def test_mode_and_root_boundaries_are_explicit(environment: Any) -> None:
    layout, source, reader, _ = environment
    with pytest.raises(ValueError, match="fixed annotation root"):
        dry_run(
            layout,
            reader,
            root_id="wrong",
            source_root=source,
            mode="in_place",
            max_files=2,
        )
    with pytest.raises(ValueError, match="disjoint"):
        dry_run(
            AnnotationLayout.testing(source),
            reader,
            root_id="wrong",
            source_root=source,
            mode="copy",
            max_files=2,
        )
    alias = source.with_name("alias")
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink-free"):
        dry_run(
            layout, reader, root_id="wrong", source_root=alias, mode="copy", max_files=2
        )


def test_unclosed_or_duplicate_bundle_cannot_be_planned(environment: Any) -> None:
    _, _, _, plan = environment
    payload = plan.bundle.model_dump()
    payload["files"] = (
        plan.bundle.files[0].model_copy(update={"dependencies": ("missing",)}),
    )
    with pytest.raises(ValueError, match="unlisted dependency"):
        Bundle.model_validate(payload)
    payload["files"] = (plan.bundle.files[0], plan.bundle.files[0])
    with pytest.raises(ValueError, match="unique, sorted"):
        Bundle.model_validate(payload)
