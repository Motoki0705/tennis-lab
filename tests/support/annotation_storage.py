"""Exercise reference protection even when individual test artifacts are expired."""

from __future__ import annotations

from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage.catalog import StorageCatalog
from src.tasks.base.annotation.storage.contracts import (
    InventoryScope,
    ManagedArtifact,
    Retention,
)
from src.tasks.base.annotation.storage.inventory import inventory
from src.tasks.base.annotation.storage.planner import plan_gc
from src.tasks.base.annotation.storage.references import distinct


def assert_expired_but_protected(
    layout: AnnotationLayout, references: tuple[ArtifactRef, ...], kind: str
) -> None:
    references = distinct(layout, references)
    catalog = StorageCatalog(layout)
    for reference in references:
        catalog.register(
            ManagedArtifact(
                reference=reference,
                owner="gc-fixture",
                stage="fixture",
                kind="cache",
                regenerable=False,
                retention=Retention(
                    mode="expire", expires_at_ns=0, reason="explicit test expiry"
                ),
            )
        )
    scopes = tuple(
        InventoryScope(role=role, path=path)
        for role, path in sorted({(r.role, r.path.split("/")[0]) for r in references})
    )
    report = plan_gc(catalog, inventory(layout, scopes, max_entries=10000))
    decisions = {row.path: row for row in report.decisions}
    for reference in references:
        decision = decisions[catalog.key(reference)]
        assert not decision.eligible and decision.reference_count > 0
        assert any(
            reason.startswith("protection:" + kind + ":") for reason in decision.reasons
        )
    assert report.execution_ready is False
