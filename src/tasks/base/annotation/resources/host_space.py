"""Audited host-space exception; original scope and all usage history stay intact."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..artifacts import artifact_lock, read_document, verify_reference
from ..contracts import ArtifactRef
from ..plan import load_plan
from .contracts import ObservedHostSpacePolicy

if TYPE_CHECKING:
    from .scheduler import ResourceManager

SCHEMA = """CREATE TABLE IF NOT EXISTS observed_host_space_policies (
 scope TEXT PRIMARY KEY, reference TEXT NOT NULL
);"""


def current(manager: ResourceManager) -> ObservedHostSpacePolicy | None:
    with manager.registry.transaction() as db:
        row = db.execute(
            "SELECT reference FROM observed_host_space_policies WHERE scope=?",
            (manager.scope.scope_id,),
        ).fetchone()
    if row is None:
        return None
    reference = ArtifactRef.model_validate_json(row[0])
    policy: ObservedHostSpacePolicy = read_document(
        manager.layout, reference, ObservedHostSpacePolicy
    )
    if policy.scope != manager.reference:
        raise ValueError("observed host-space policy belongs to another scope")
    verify_reference(manager.layout, policy.authorization)
    return policy


def activate(manager: ResourceManager, reference: ArtifactRef) -> None:
    policy = read_document(manager.layout, reference, ObservedHostSpacePolicy)
    if policy.scope != manager.reference or not policy.reason.strip():
        raise ValueError("host-space authorization needs the exact scope and reason")
    plan = load_plan(verify_reference(manager.layout, policy.plan))
    manager.bind(plan)
    verify_reference(manager.layout, policy.authorization)
    mount = Path(policy.mount)
    if policy.mount not in manager.scope.free_space_mounts or any(
        Path(root).is_relative_to(mount) for root in manager.scope.roots
    ):
        raise ValueError("only an indirect monitored host mount can use observed space")
    with (
        artifact_lock(manager.layout, "resource-admission", blocking=False),
        manager.registry.transaction() as db,
    ):
        previous = db.execute(
            "SELECT reference FROM observed_host_space_policies WHERE scope=?",
            (manager.scope.scope_id,),
        ).fetchone()
        encoded = reference.canonical_bytes().decode()
        if previous is not None:
            if previous[0] != encoded:
                raise ValueError("observed host-space authorization is immutable")
            return
        active = db.execute(
            "SELECT count(*) FROM resource_requests WHERE scope=? AND state IN ('running','draining')",
            (manager.scope.scope_id,),
        ).fetchone()[0]
        if active:
            raise ValueError(
                "stop and drain resource owners before authorizing host space"
            )
        db.execute(
            "INSERT INTO observed_host_space_policies VALUES (?,?)",
            (manager.scope.scope_id, encoded),
        )
        # A new policy must not use the old watchdog's interpretation.
        db.execute(
            "DELETE FROM resource_watchdogs WHERE scope=?", (manager.scope.scope_id,)
        )
        db.execute(
            "UPDATE resource_samples SET measured_at=-1 WHERE scope=?",
            (manager.scope.scope_id,),
        )
