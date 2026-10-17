from __future__ import annotations

import multiprocessing
import os
import signal
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage import (
    InventoryScope,
    ManagedArtifact,
    Protection,
    Retention,
    inventory,
    plan_gc,
    save_dry_run,
)
from src.tasks.base.annotation.storage import gc as module
from src.tasks.base.annotation.storage.gc import GCReceipt, GCRequest, GCStore


@pytest.fixture
def store(tmp_path: Path) -> GCStore:
    return GCStore(AnnotationLayout.testing(tmp_path / "annotation"))


def artifact(
    store: GCStore, name: str, *, link: Path | None = None, expires: int = 0
) -> ManagedArtifact:
    path = store.layout.path("migration_staging", name)
    if link is None:
        ref = write_asset(store.layout, path, name.encode() * 4096)
    else:
        os.link(link, path)
        ref = annotation_reference(store.layout, path)
    managed = ManagedArtifact(
        reference=ref,
        owner="fixture",
        stage="decode",
        kind="cache",
        regenerable=True,
        retention=Retention(
            mode="expire", expires_at_ns=expires, reason="synthetic cache TTL"
        ),
    )
    store.catalog.register(managed)
    return managed


def request(store: GCStore) -> GCRequest:
    observed = inventory(
        store.layout,
        (InventoryScope(role="annotation", path="migration/staging"),),
        max_entries=100,
    )
    refs = save_dry_run(store.catalog, observed, plan_gc(store.catalog, observed))
    return GCRequest(
        inventory=refs[0],
        plan=refs[1],
        reason="Delete only the reviewed synthetic expired files",
    )


def test_gc_counts_shared_inode_once_and_replays_receipt(
    store: GCStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = artifact(store, "a")
    path = store.layout.artifact_path(first.reference)
    second = artifact(store, "b", link=path)
    allocated = path.stat().st_blocks * 512
    chosen = request(store)
    saved = store.execute(chosen, max_entries=100)
    receipt = read_document(store.layout, saved, GCReceipt)
    assert receipt.removed == (first.reference, second.reference)
    assert receipt.unlinked_logical_bytes == first.reference.size_bytes * 2
    assert receipt.planned_reclaimable_allocated_bytes == allocated
    assert not path.exists()
    monkeypatch.setattr(module.time, "time_ns", lambda: 0)
    assert GCStore(store.layout).execute(chosen, max_entries=100) == saved
    with store.registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM storage_gc_jobs").fetchone()[0] == 1
        assert [row[0] for row in db.execute("SELECT state FROM storage_retired")] == [
            "deleted",
            "deleted",
        ]


def test_new_gc_refuses_future_dated_dry_run(
    store: GCStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = artifact(store, "a")
    chosen = request(store)
    monkeypatch.setattr(module.time, "time_ns", lambda: 0)
    with pytest.raises(ValueError, match="future-dated"):
        store.execute(chosen, max_entries=100)
    assert store.layout.artifact_path(item.reference).is_file()
    assert not store.catalog.snapshot().retired


def test_competing_gc_decisions_cannot_retire_the_same_file_twice(
    store: GCStore,
) -> None:
    artifact(store, "a")
    chosen = request(store)

    def execute(reason: str):
        try:
            return store.execute(
                chosen.model_copy(update={"reason": reason}), max_entries=100
            )
        except ValueError as error:
            assert "new dry-run" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                execute, ("first explicit decision", "competing explicit decision")
            )
        )
    assert sum(result is not None for result in results) == 1
    with store.registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM storage_gc_jobs").fetchone()[0] == 1


@pytest.mark.parametrize(
    "kind", ["active", "adoption", "training", "publication", "reader", "model"]
)
def test_gc_keeps_live_references_and_future_expiry(store: GCStore, kind: str) -> None:
    kept = artifact(store, "kept")
    future = artifact(store, "future", expires=2**63 - 1)
    stale = artifact(store, "stale")
    pin = Protection.model_validate(
        {
            "protection_id": "live",
            "kind": kind,
            "reason": "reader/owner is alive",
            "references": (kept.reference,),
        }
    )
    store.catalog.protect(pin)
    before = {
        item.reference.path: store.layout.artifact_path(item.reference).read_bytes()
        for item in (kept, future)
    }
    saved = store.execute(request(store), max_entries=100)
    assert read_document(store.layout, saved, GCReceipt).removed == (stale.reference,)
    assert {
        item.reference.path: store.layout.artifact_path(item.reference).read_bytes()
        for item in (kept, future)
    } == before


def test_gc_rechecks_reference_added_after_saved_dry_run(store: GCStore) -> None:
    item = artifact(store, "a")
    chosen = request(store)
    store.catalog.protect(
        Protection(
            protection_id="new-reader",
            kind="reader",
            reason="started after dry-run",
            references=(item.reference,),
        )
    )
    with pytest.raises(ValueError, match="references changed"):
        store.execute(chosen, max_entries=100)
    assert store.layout.artifact_path(item.reference).is_file()
    assert not store.catalog.snapshot().retired


@pytest.mark.parametrize("change", ["content", "hardlink", "symlink"])
def test_gc_rechecks_files_and_never_follows_replacements(
    store: GCStore, change: str
) -> None:
    item = artifact(store, "a")
    chosen = request(store)
    path = store.layout.artifact_path(item.reference)
    outside = store.layout.repository / "outside"
    outside.write_bytes(b"outside evidence")
    if change == "content":
        path.write_bytes(b"changed")
    elif change == "hardlink":
        os.link(path, path.with_name("extra"))
    else:
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        store.execute(chosen, max_entries=100)
    assert outside.read_bytes() == b"outside evidence"
    assert not store.catalog.snapshot().retired


@pytest.mark.parametrize(
    "phase",
    [
        "retire_before",
        "retire_after",
        "unlink_before",
        "unlink_after",
        "finish_after",
        "receipt_before",
        "receipt_after",
    ],
)
def test_sigkill_gc_resumes_journal_without_duplicate_unlink(
    store: GCStore, phase: str
) -> None:
    first = artifact(store, "a")
    artifact(store, "b", link=store.layout.artifact_path(first.reference))
    chosen = request(store)

    def killed_run() -> None:
        def stop() -> None:
            os.kill(os.getpid(), signal.SIGKILL)

        with pytest.MonkeyPatch.context() as patch:
            if phase.startswith("retire"):
                original_retire = GCStore._retire

                def retire(self, *args, **kwargs):
                    if phase == "retire_before":
                        stop()
                    original_retire(self, *args, **kwargs)
                    stop()

                patch.setattr(GCStore, "_retire", retire)
            elif phase.startswith("unlink"):
                original_unlink = os.unlink

                def unlink(name, *args, **kwargs):
                    selected = name == "a" and "dir_fd" in kwargs
                    if selected and phase == "unlink_before":
                        stop()
                    original_unlink(name, *args, **kwargs)
                    if selected:
                        stop()

                patch.setattr(os, "unlink", unlink)
            elif phase == "finish_after":
                original_finish = GCStore._finish

                def finish(self, *args, **kwargs):
                    original_finish(self, *args, **kwargs)
                    stop()

                patch.setattr(GCStore, "_finish", finish)
            else:
                original_write = module.write_document

                def write(layout, path, document):
                    if isinstance(document, GCReceipt) and phase == "receipt_before":
                        stop()
                    result = original_write(layout, path, document)
                    if isinstance(document, GCReceipt):
                        stop()
                    return result

                patch.setattr(module, "write_document", write)
            GCStore(store.layout).execute(chosen, max_entries=100)

    process = multiprocessing.get_context("fork").Process(target=killed_run)
    process.start()
    process.join(20)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("GC failpoint process did not finish")
    assert process.exitcode == -signal.SIGKILL
    resumed = GCStore(store.layout)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: resumed.execute(chosen, max_entries=100), range(2))
        )
    assert results[0] == results[1]
    assert len(read_document(store.layout, results[0], GCReceipt).removed) == 2
    with sqlite3.connect(store.layout.registry) as db:
        assert db.execute("SELECT count(*) FROM storage_gc_jobs").fetchone()[0] == 1


def test_reference_writer_cannot_pin_retired_path_during_unlink_gap(
    store: GCStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = artifact(store, "a")
    chosen = request(store)
    retired = Event()
    proceed = Event()
    original = store._retire

    def paused(*args, **kwargs):
        original(*args, **kwargs)
        retired.set()
        assert proceed.wait(10)

    monkeypatch.setattr(store, "_retire", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(store.execute, chosen, max_entries=100)
        try:
            assert retired.wait(10)
            assert store.layout.artifact_path(item.reference).is_file()
            with pytest.raises(ValueError, match="retired"):
                store.catalog.protect(
                    Protection(
                        protection_id="too-late",
                        kind="reader",
                        reason="raced with GC",
                        references=(item.reference,),
                    )
                )
            with pytest.raises(ValueError, match="retired"):
                store.catalog.register(item)
        finally:
            proceed.set()
        future.result(timeout=10)


def test_gc_refuses_recreated_retired_paths_on_replay(store: GCStore) -> None:
    item = artifact(store, "a")
    chosen = request(store)
    store.execute(chosen, max_entries=100)
    path = store.layout.artifact_path(item.reference)
    path.write_bytes(b"new unrelated bytes")
    with pytest.raises(ValueError, match="recreated"):
        store.execute(chosen, max_entries=100)
    assert path.read_bytes() == b"new unrelated bytes"


def test_reader_context_keeps_inputs_until_exit(store: GCStore) -> None:
    item = artifact(store, "a")
    artifact(store, "stale")
    pin = Protection(
        protection_id="reading",
        kind="reader",
        reason="native reader inputs",
        references=(item.reference,),
    )
    with store.catalog.reader(pin):
        receipt = read_document(
            store.layout, store.execute(request(store), max_entries=100), GCReceipt
        )
        assert item.reference not in receipt.removed
        assert store.layout.artifact_path(item.reference).read_bytes() == b"a" * 4096
    receipt = read_document(
        store.layout, store.execute(request(store), max_entries=100), GCReceipt
    )
    assert receipt.removed == (item.reference,)


def test_killed_reader_is_not_expired_automatically(store: GCStore) -> None:
    item = artifact(store, "a")
    artifact(store, "stale")
    pin = Protection(
        protection_id="killed-reader",
        kind="reader",
        reason="must explicitly reconcile dead reader",
        references=(item.reference,),
    )

    def read_then_die() -> None:
        with store.catalog.reader(pin):
            os.kill(os.getpid(), signal.SIGKILL)

    process = multiprocessing.get_context("fork").Process(target=read_then_die)
    process.start()
    process.join(20)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("reader failpoint process did not finish")
    assert process.exitcode == -signal.SIGKILL
    receipt = read_document(
        store.layout, store.execute(request(store), max_entries=100), GCReceipt
    )
    assert item.reference not in receipt.removed
    assert pin in store.catalog.snapshot().protections


def test_gc_requires_decision_limits_and_annotation_boundary(store: GCStore) -> None:
    artifact(store, "a")
    chosen = request(store)
    with pytest.raises(ValueError, match="reason"):
        store.execute(chosen.model_copy(update={"reason": " "}), max_entries=100)
    with pytest.raises(ValueError, match="entry budget"):
        store.execute(chosen, max_entries=1)
    with pytest.raises(ValueError, match="fixed annotation"):
        store._target(chosen.plan.model_copy(update={"role": "data"}))
    with pytest.raises(ValueError, match="storage evidence"):
        store._target(chosen.plan)


@pytest.mark.parametrize("change", ["link", "ancestor"])
def test_retired_candidate_is_rechecked_immediately_before_unlink(
    store: GCStore, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    item = artifact(store, "a")
    chosen = request(store)
    target = store.layout.artifact_path(item.reference)
    original = store._retire
    outside = store.layout.repository / "outside"

    def retire(*args, **kwargs):
        original(*args, **kwargs)
        if change == "link":
            os.link(target, outside)
        else:
            target.parent.rename(outside)
            target.parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(store, "_retire", retire)
    with pytest.raises(ValueError, match="links changed|symlinks"):
        store.execute(chosen, max_entries=100)
    retained = outside if change == "link" else outside / target.name
    assert retained.read_bytes() == b"a" * 4096


def test_gc_rejects_data_root_before_retiring_any_file(store: GCStore) -> None:
    from src.tasks.base.annotation.artifacts import data_reference

    managed = artifact(store, "a")
    dataset = store.layout.repository / "data" / "fixture"
    dataset.mkdir(parents=True)
    payload = dataset / "pixels.bin"
    payload.write_bytes(b"published pixels")
    store.catalog.register(
        managed.model_copy(update={"reference": data_reference(store.layout, payload)})
    )
    observed = inventory(
        store.layout,
        (
            InventoryScope(role="data", path="fixture"),
            InventoryScope(role="annotation", path="migration/staging"),
        ),
        max_entries=100,
    )
    refs = save_dry_run(store.catalog, observed, plan_gc(store.catalog, observed))
    chosen = GCRequest(
        inventory=refs[0],
        plan=refs[1],
        reason="explicit fixture decision cannot override root boundary",
    )
    with pytest.raises(ValueError, match="fixed annotation"):
        store.execute(chosen, max_entries=100)
    assert not store.catalog.snapshot().retired
    assert payload.read_bytes() == b"published pixels"
    assert store.layout.artifact_path(managed.reference).exists()
