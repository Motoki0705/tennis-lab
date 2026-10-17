from __future__ import annotations

import multiprocessing
import os
import signal
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
    BundleFile,
    MigrationPlan,
    MigrationReceipt,
    MigrationStore,
    describe_file,
    dry_run,
    imported_root,
)
from src.tasks.base.annotation.storage.migration import files as file_implementation
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


@pytest.mark.parametrize(
    "mutation", ["append_before", "replace_before", "append_during"]
)
def test_source_growth_never_writes_beyond_planned_copy_size(
    environment: Any, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    layout, source, reader, _ = environment
    changed_source = source / "payload/b.bin"
    # More than one read block exercises the remaining budget after a valid write.
    original_payload = b"x" * (1024 * 1024 + 23)
    changed_source.write_bytes(original_payload)
    plan = dry_run(
        layout,
        reader,
        root_id="growing-source",
        source_root=source,
        mode="copy",
        max_files=2,
    )
    store = MigrationStore(layout)
    staging = layout.path("migration_staging", plan.sha256, "root")
    partials = staging.parent / "partials"
    completed = staging / plan.bundle.files[0].path
    original_copy = implementation.copy_file
    original_temporary = file_implementation.tempfile.NamedTemporaryFile
    changed = False
    peak_payload = 0

    def grow_source() -> None:
        nonlocal changed
        assert not changed
        extra = b"y" * (2 * 1024 * 1024)
        if mutation == "replace_before":
            replacement = changed_source.with_suffix(".replacement")
            replacement.write_bytes(original_payload + extra)
            replacement.replace(changed_source)
        else:
            with changed_source.open("ab") as handle:
                handle.write(extra)
        changed = True

    def copy_after_validation(
        source_root: Path, target_root: Path, item: BundleFile
    ) -> None:
        if item.path == plan.bundle.files[1].path and mutation != "append_during":
            grow_source()
        original_copy(source_root, target_root, item)

    def measured_temporary(*args: Any, **kwargs: Any) -> Any:
        output = original_temporary(*args, **kwargs)
        if kwargs.get("dir") != partials:
            return output
        copying_second = completed.exists()
        item = plan.bundle.files[1 if copying_second else 0]
        write = output.write

        def measured_write(block: bytes) -> int:
            nonlocal peak_payload
            count: int = write(block)
            output.flush()
            payload = sum(
                path.stat().st_size
                for path in staging.parent.rglob("*")
                if path.is_file()
            )
            peak_payload = max(peak_payload, payload)
            assert output.tell() <= item.size_bytes
            assert payload <= plan.copy_payload_bytes
            if copying_second and mutation == "append_during" and not changed:
                grow_source()
            return count

        monkeypatch.setattr(output, "write", measured_write)
        return output

    monkeypatch.setattr(implementation, "copy_file", copy_after_validation)
    monkeypatch.setattr(
        file_implementation.tempfile, "NamedTemporaryFile", measured_temporary
    )
    with pytest.raises(ValueError, match="dry-run"):
        apply(store, plan, reader)
    assert changed
    assert changed_source.stat().st_size > plan.bundle.files[1].size_bytes
    assert 0 < peak_payload <= plan.copy_payload_bytes
    assert completed.read_bytes() == (source / plan.bundle.files[0].path).read_bytes()
    assert not (staging / plan.bundle.files[1].path).exists()
    assert list(partials.iterdir()) == []
    assert not layout.path("migration_archive", plan.sha256).exists()
    with pytest.raises(ValueError, match="no completed import"):
        imported_root(layout, plan.root_id, reader)


@pytest.mark.parametrize("interruptions", [1, 2])
def test_sigkill_partial_recovery_stays_within_copy_budget(
    environment: Any, monkeypatch: pytest.MonkeyPatch, interruptions: int
) -> None:
    layout, source, reader, plan = environment
    store = MigrationStore(layout)
    staging = layout.path("migration_staging", plan.sha256, "root")
    partials = staging.parent / "partials"
    completed = staging / plan.bundle.files[0].path
    other_partial = layout.path("migration_staging", "other-plan", "partials", "tmp")
    other_partial.parent.mkdir(parents=True)
    other_partial.write_bytes(b"another plan owns this")
    original_temporary = file_implementation.tempfile.NamedTemporaryFile

    def interrupted_copy() -> None:
        def temporary(*args: Any, **kwargs: Any) -> Any:
            output = original_temporary(*args, **kwargs)
            if completed.exists():
                write = output.write

                def interrupted_write(block: bytes) -> int:
                    write(block[: len(block) // 2])
                    output.flush()
                    os.fsync(output.fileno())
                    os.kill(os.getpid(), signal.SIGKILL)
                    raise AssertionError("SIGKILL must not return")

                monkeypatch.setattr(output, "write", interrupted_write)
            return output

        monkeypatch.setattr(file_implementation.tempfile, "NamedTemporaryFile", temporary)
        apply(store, plan, reader)

    for _ in range(interruptions):
        process = multiprocessing.get_context("fork").Process(target=interrupted_copy)
        process.start()
        try:
            process.join(10)
            assert process.exitcode == -signal.SIGKILL
        finally:
            if process.is_alive():
                process.kill()
                process.join(5)
        assert completed.read_bytes() == (source / plan.bundle.files[0].path).read_bytes()
        leftovers = list(partials.iterdir())
        assert len(leftovers) == 1
        assert 0 < leftovers[0].stat().st_size < plan.bundle.files[1].size_bytes

    completed_inode = completed.stat().st_ino
    peak_payload = 0

    def budgeted_temporary(*args: Any, **kwargs: Any) -> Any:
        output = original_temporary(*args, **kwargs)
        write = output.write

        def budgeted_write(block: bytes) -> int:
            nonlocal peak_payload
            count: int = write(block)
            output.flush()
            payload = sum(
                path.stat().st_size
                for path in staging.parent.rglob("*")
                if path.is_file()
            )
            peak_payload = max(peak_payload, payload)
            assert payload <= plan.copy_payload_bytes
            return count

        monkeypatch.setattr(output, "write", budgeted_write)
        return output

    monkeypatch.setattr(
        file_implementation.tempfile, "NamedTemporaryFile", budgeted_temporary
    )
    apply(MigrationStore(layout), plan, reader)
    root = imported_root(layout, plan.root_id, reader)
    assert reader.inspect(root) == reader.inspect(source) == plan.bundle
    assert (root / plan.bundle.files[0].path).stat().st_ino == completed_inode
    assert peak_payload == plan.copy_payload_bytes
    assert list(partials.iterdir()) == []
    assert other_partial.read_bytes() == b"another plan owns this"


def test_published_import_reclaims_partial_link_without_changing_archive(
    environment: Any,
) -> None:
    layout, _, reader, plan = environment
    store = MigrationStore(layout)
    reference = apply(store, plan, reader)
    root = imported_root(layout, plan.root_id, reader)
    completed = root / plan.bundle.files[0].path
    inode = completed.stat().st_ino
    partials = layout.path("migration_staging", plan.sha256, "partials")
    os.link(completed, partials / "tmp-interrupted-publication")
    assert apply(store, plan, reader) == reference
    assert list(partials.iterdir()) == []
    assert completed.stat().st_ino == inode
    assert reader.inspect(root) == plan.bundle


@pytest.mark.parametrize("intrusion", ["directory", "symlink", "fifo", "linked-root"])
def test_partial_recovery_rejects_nonregular_entries_and_symlink_roots(
    environment: Any, intrusion: str
) -> None:
    layout, source, reader, plan = environment
    store = MigrationStore(layout)
    partials = layout.path("migration_staging", plan.sha256, "partials")
    partials.parent.mkdir(parents=True)
    if intrusion == "linked-root":
        partials.symlink_to(source / "payload", target_is_directory=True)
    else:
        partials.mkdir()
        entry = partials / "tmp-unexpected"
        if intrusion == "directory":
            entry.mkdir()
            (entry / "keep").write_bytes(b"not a partial")
        elif intrusion == "symlink":
            entry.symlink_to(source / plan.bundle.files[0].path)
        else:
            os.mkfifo(entry)
    with pytest.raises((ValueError, OSError)):
        apply(store, plan, reader)
    assert reader.inspect(source) == plan.bundle
    assert not layout.path("migration_archive", plan.sha256).exists()
    if intrusion != "linked-root":
        assert (partials / "tmp-unexpected").lstat()


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


@pytest.mark.parametrize("phase", ["receipt_before", "receipt_after", "transaction", "commit_after"])
def test_sigkill_at_migration_publication_boundaries(environment: Any, phase: str) -> None:
    from concurrent.futures import ThreadPoolExecutor

    layout, source, reader, plan = environment
    store = MigrationStore(layout)
    before = reader.inspect(source)

    def killed_run() -> None:
        def stop() -> None:
            os.kill(os.getpid(), signal.SIGKILL)

        with pytest.MonkeyPatch.context() as patch:
            if phase.startswith("receipt"):
                original = implementation.write_document

                def write(layout, path, document):
                    if isinstance(document, MigrationReceipt) and phase == "receipt_before":
                        stop()
                    result = original(layout, path, document)
                    if isinstance(document, MigrationReceipt):
                        stop()
                    return result

                patch.setattr(implementation, "write_document", write)
            elif phase == "transaction":
                original_protect = store.catalog.protect

                def protect(*args, **kwargs):
                    original_protect(*args, **kwargs)
                    stop()

                patch.setattr(store.catalog, "protect", protect)
            else:
                original_publish = store._publish

                def publish(*args, **kwargs):
                    original_publish(*args, **kwargs)
                    stop()

                patch.setattr(store, "_publish", publish)
            apply(store, plan, reader)

    process = multiprocessing.get_context("fork").Process(target=killed_run)
    process.start()
    process.join(20)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("migration failpoint process did not finish")
    assert process.exitcode == -signal.SIGKILL
    with sqlite3.connect(layout.registry) as database:
        rows = database.execute("SELECT * FROM storage_imported_roots").fetchall()
        assert bool(rows) == (phase == "commit_after")
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: apply(MigrationStore(layout), plan, reader), range(2)))
    assert receipts[0] == receipts[1]
    assert reader.inspect(source) == before
    assert reader.inspect(imported_root(layout, plan.root_id, reader)) == before
    assert len(store.catalog.snapshot().protections) == 1
