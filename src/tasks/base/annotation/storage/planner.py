"""Conservative GC simulation. This module has no deletion or move operation."""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path

from ..artifacts import verify_reference, write_document
from ..contracts import ArtifactRef
from .catalog import StorageCatalog
from .contracts import GCDecision, GCPlan, Inventory, InventoryEntry, StageUsage


def _unchanged(entry: InventoryEntry) -> bool:
    try:
        current = Path(entry.path).lstat()
    except FileNotFoundError:
        return False
    return (
        current.st_dev,
        current.st_ino,
        current.st_nlink,
        current.st_size,
        current.st_blocks * 512,
        current.st_mtime_ns,
        current.st_ctime_ns,
    ) == (
        entry.device,
        entry.inode,
        entry.links,
        entry.logical_bytes,
        entry.allocated_bytes,
        entry.modified_ns,
        entry.changed_ns,
    )


def plan_gc(
    catalog: StorageCatalog,
    observed: Inventory,
    *,
    now_ns: int | None = None,
    database: sqlite3.Connection | None = None,
) -> GCPlan:
    """An explicit expiry is necessary, but never sufficient to authorize removal."""
    now = time.time_ns() if now_ns is None else now_ns
    if now < observed.created_at_ns:
        raise ValueError("GC decision cannot predate its inventory")
    snapshot = catalog.snapshot(database=database)
    entries = {item.path: item for item in observed.entries}
    if len(entries) != len(observed.entries):
        raise ValueError("inventory contains duplicate paths")
    edges: dict[str, tuple[str, ...]] = {}
    incoming: dict[str, int] = defaultdict(int)
    reasons: dict[str, set[str]] = defaultdict(set)
    retired = set(snapshot.retired)
    for path, registered in snapshot.artifacts.items():
        if path in retired:
            reasons[path].add("gc:retired")
            continue
        dependencies = tuple(catalog.key(item) for item in registered.dependencies)
        edges[path] = dependencies
        for target in dependencies:
            incoming[target] += 1
        if path not in entries:
            reasons[path].add("outside_inventory")
        elif registered.retention.mode == "retain":
            reasons[path].add("retention:retain")
        elif (
            registered.retention.expires_at_ns is None
            or registered.retention.expires_at_ns > now
        ):
            reasons[path].add("retention:not_expired")
    for protection in snapshot.protections:
        for reference in protection.references:
            path = catalog.key(reference)
            incoming[path] += 1
            reasons[path].add(
                f"protection:{protection.kind}:{protection.protection_id}"
            )
    groups: dict[tuple[int, int], list[InventoryEntry]] = defaultdict(list)
    for entry in observed.entries:
        if entry.kind == "file":
            groups[(entry.device, entry.inode)].append(entry)
        else:
            reasons[entry.path].add("not_regular_file")
        artifact = snapshot.artifacts.get(entry.path)
        if artifact is None:
            reasons[entry.path].add("ownership_unregistered")
        if not _unchanged(entry):
            reasons[entry.path].add("changed_since_inventory")
        elif artifact is not None and entry.kind == "file":
            try:
                verify_reference(catalog.layout, artifact.reference)
            except (OSError, ValueError):
                reasons[entry.path].add("registered_content_unverified")
            if not _unchanged(entry):
                reasons[entry.path].add("changed_during_verification")
    for group in groups.values():
        if any(item.links != len(group) for item in group):
            for item in group:
                reasons[item.path].add("hardlinks_outside_inventory_or_changed")

    def propagate() -> None:
        pending = deque(sorted(path for path, why in reasons.items() if why))
        visited: set[str] = set()
        while pending:
            path = pending.popleft()
            if path in visited:
                continue
            visited.add(path)
            for target in edges.get(path, ()):
                if not reasons[target]:
                    reasons[target].add(f"referenced_by_retained:{path}")
                pending.append(target)

    # Reference and shared-inode protection can activate each other. Reach a fixed
    # point without treating an unrooted, expired dependency cycle as a live root.
    while True:
        before = {path for path, why in reasons.items() if why}
        propagate()
        for group in groups.values():
            if any(reasons[item.path] for item in group):
                for item in group:
                    if not reasons[item.path]:
                        reasons[item.path].add("hardlink_shared_with_protected_path")
        if before == {path for path, why in reasons.items() if why}:
            break

    decisions = []
    stage_entries: dict[str, list[InventoryEntry]] = defaultdict(list)
    inode_stages: dict[tuple[int, int], set[str]] = defaultdict(set)
    for entry in observed.entries:
        artifact = snapshot.artifacts.get(entry.path)
        stage = artifact.stage if artifact is not None else "unregistered"
        stage_entries[stage].append(entry)
        inode_stages[(entry.device, entry.inode)].add(stage)
        why = reasons[entry.path]
        decisions.append(
            GCDecision(
                path=entry.path,
                owner=artifact.owner if artifact else None,
                stage=artifact.stage if artifact else None,
                kind=artifact.kind if artifact else None,
                retention=artifact.retention if artifact else None,
                regenerable=artifact.regenerable if artifact else None,
                eligible=not why,
                reasons=tuple(sorted(why))
                if why
                else ("expired_and_unreferenced_in_catalog",),
                reference_count=incoming[entry.path],
            )
        )
    stages = []
    for stage, items in sorted(stage_entries.items()):
        unique = {(item.device, item.inode): item for item in items}
        stages.append(
            StageUsage(
                stage=stage,
                entries=len(items),
                logical_bytes=sum(item.logical_bytes for item in items),
                files=sum(item.kind == "file" for item in items),
                symlinks=sum(item.kind == "symlink" for item in items),
                unique_inodes=len(unique),
                hardlinked_files=sum(
                    item.kind == "file" and item.links > 1 for item in items
                ),
                allocated_bytes=sum(item.allocated_bytes for item in unique.values()),
                shared_inode_bytes=sum(
                    item.allocated_bytes
                    for inode, item in unique.items()
                    if len(inode_stages[inode]) > 1
                ),
            )
        )
    eligible = {item.path for item in decisions if item.eligible}
    reclaimable = [
        group[0]
        for group in groups.values()
        if all(item.path in eligible for item in group)
    ]
    return GCPlan(
        created_at_ns=now,
        inventory_digest=observed.sha256,
        catalog_digest=snapshot.digest,
        decisions=tuple(decisions),
        stages=tuple(stages),
        eligible_logical_bytes=sum(entries[path].logical_bytes for path in eligible),
        eligible_allocated_bytes=sum(item.allocated_bytes for item in reclaimable),
    )


def save_dry_run(
    catalog: StorageCatalog, observed: Inventory, plan: GCPlan
) -> tuple[ArtifactRef, ArtifactRef]:
    if plan.inventory_digest != observed.sha256:
        raise ValueError("GC plan and inventory do not match")
    # Evidence saving never executes collection. GCStore rechecks this evidence
    # under the reference transaction after an explicit, separate GCRequest.
    inventory_ref = write_document(
        catalog.layout,
        catalog.layout.path("storage_inventories", observed.sha256 + ".json"),
        observed,
    )
    plan_ref = write_document(
        catalog.layout,
        catalog.layout.path("storage_plans", plan.sha256 + ".json"),
        plan,
    )
    return inventory_ref, plan_ref
