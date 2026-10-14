"""Borrow registered, unadopted predictions without changing canonical clip identity."""

from __future__ import annotations

import sqlite3

from ..artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from ..contracts import (
    ArtifactRef,
    ClipManifest,
    Digest,
    Document,
    Identifier,
    content_digest,
)
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, references, require_execution_layout
from ..predictions import PredictionAdapter, PredictionReceipt, Predictions
from ..registry import Registry
from ..storage.catalog import StorageCatalog
from ..storage.references import protection
from .validation import condition


class BorrowedPrediction(Document):
    task: Identifier
    clip_id: Digest
    source_plan: ArtifactRef
    receipt: ArtifactRef


class RegisteredInputs(Document):
    condition: ArtifactRef
    borrowed: tuple[BorrowedPrediction, ...]


def register_inputs(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    adapters: tuple[PredictionAdapter, ...],
) -> ArtifactRef:
    shared = layout.shared()
    spec, plan, _ = condition(shared, reference)
    if plan.phase != "benchmark":
        raise ValueError("prediction replay requires an explicit benchmark plan")
    local = layout.for_benchmark(plan.campaign_id)
    require_execution_layout(local, plan)
    by_task = {adapter.task: adapter for adapter in adapters}
    if len(by_task) != len(adapters):
        raise ValueError("benchmark prediction adapters must be unique")
    receipts: list[tuple[ArtifactRef, PredictionReceipt, ClipManifest]] = []
    retained: list[Document] = [plan, spec]
    for ref in spec.predictions:
        receipt = read_document(shared, ref, PredictionReceipt)
        context = receipt.context
        manifest = read_document(shared, context.manifest, ClipManifest)
        adapter = by_task.get(context.task)
        if adapter is None:
            raise ValueError("benchmark input has no matching task validator")
        if receipt.prediction.request_id != context.request_id:
            raise ValueError("benchmark prediction receipt/context disagree")
        for artifact in (*references(receipt), *references(manifest)):
            verify_reference(shared, artifact)
        adapter.validate(shared, context, manifest, receipt.prediction.artifact)
        receipts.append((ref, receipt, manifest))
        retained.extend((receipt, manifest))
    if not shared.registry.exists():
        raise ValueError("benchmark inputs require an existing production registry")
    source_registry = Registry(shared)
    catalog = StorageCatalog(shared)
    borrowed: list[BorrowedPrediction] = []
    # Publish input protection before domain state. A later failed import leaves
    # conservative protection, never a usable domain with unprotected inputs.
    with source_registry.transaction() as db:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {
            "resource_scopes",
            "annotation_predictions",
            "annotation_tasks",
            "clips",
            "sources",
            "campaigns",
        }.issubset(tables):
            raise ValueError(
                "benchmark inputs require registered scope and predictions"
            )
        scope = db.execute(
            "SELECT reference FROM resource_scopes WHERE reference=?",
            (spec.scope.canonical_bytes().decode(),),
        ).fetchone()
        if scope is None:
            raise ValueError("benchmark must reuse a registered shared scope")
        for ref, receipt, manifest in receipts:
            task_id = content_digest([receipt.context.task, manifest.clip_id])
            original = db.execute(
                "SELECT t.*,p.receipt,c.manifest,c.media_digest,c.source_id FROM annotation_tasks t "
                "JOIN annotation_predictions p ON p.task_id=t.id JOIN clips c ON c.id=t.clip_id WHERE t.id=?",
                (task_id,),
            ).fetchone()
            if original is None or (
                original["state"] not in {"predicted", "reviewing"}
                or ArtifactRef.model_validate_json(original["receipt"]) != ref
                or ArtifactRef.model_validate_json(original["manifest"])
                != receipt.context.manifest
                or original["model_digest"] != receipt.context.model.sha256
                or original["inference_digest"] != receipt.context.configuration.digest
                or original["media_digest"] != manifest.media.digest
                or original["source_id"] != manifest.source.canonical_id
            ):
                raise ValueError(
                    "benchmark can only borrow exact registered, unadopted task predictions"
                )
            source_ref = annotation_reference(
                shared, shared.path("campaigns", original["campaign_id"], "plan.json")
            )
            source_plan = read_document(shared, source_ref, CampaignPlan)
            if source_plan.phase == "benchmark":
                raise ValueError(
                    "benchmark inputs must originate in the shared collection workflow"
                )
            saved = db.execute(
                "SELECT digest FROM campaigns WHERE id=?", (source_plan.campaign_id,)
            ).fetchone()
            if saved is None or saved[0] != source_plan.sha256:
                raise ValueError(
                    "borrowed input source plan differs from its registration"
                )
            item = BorrowedPrediction(
                task=receipt.context.task,
                clip_id=manifest.clip_id,
                source_plan=source_ref,
                receipt=ref,
            )
            borrowed.append(item)
            retained.extend((item, source_plan))
        catalog.protect(
            protection(
                shared,
                identity="benchmark-inputs-v1-" + reference.digest,
                kind="reader",
                reason="fixed benchmark inputs retained across isolated conditions and restarts",
                roots=(reference,),
                documents=tuple(retained),
            ),
            database=db,
        )
    predictions = Predictions(local, plan)
    predictions.registry.initialize(
        "benchmark-inputs-v1",
        """
        CREATE TABLE IF NOT EXISTS benchmark_inputs (
            campaign TEXT PRIMARY KEY, specification TEXT NOT NULL
        );
    """,
    )
    with predictions.registry.transaction() as db:
        prior = db.execute(
            "SELECT specification FROM benchmark_inputs WHERE campaign=?",
            (plan.campaign_id,),
        ).fetchone()
        if prior is not None and ArtifactRef.model_validate_json(prior[0]) != reference:
            raise ValueError("benchmark condition input binding is immutable")
        for ref, receipt, manifest in receipts:
            _register(db, plan, ref, receipt, manifest)
        db.execute(
            "INSERT OR IGNORE INTO benchmark_inputs VALUES (?,?)",
            (plan.campaign_id, reference.canonical_bytes().decode()),
        )
    return write_document(
        local,
        local.path("campaigns", plan.campaign_id, "benchmark", "inputs.json"),
        RegisteredInputs(condition=reference, borrowed=tuple(borrowed)),
    )


def _register(
    db: sqlite3.Connection,
    plan: CampaignPlan,
    reference: ArtifactRef,
    receipt: PredictionReceipt,
    manifest: ClipManifest,
) -> None:
    source = manifest.source
    db.execute(
        "INSERT OR IGNORE INTO channels VALUES (?,?,?)",
        (source.provider, source.channel_id, plan.campaign_id),
    )
    db.execute(
        "INSERT OR IGNORE INTO sources(id,provider,channel_id,video_id,campaign_id,candidate,state) VALUES (?,?,?,?,?,?,'benchmark_input')",
        (
            source.canonical_id,
            source.provider,
            source.channel_id,
            source.video_id,
            plan.campaign_id,
            source.model_dump_json(),
        ),
    )
    existing = db.execute(
        "SELECT campaign_id,manifest,media_digest,source_id FROM clips WHERE id=?",
        (manifest.clip_id,),
    ).fetchone()
    expected = (
        plan.campaign_id,
        receipt.context.manifest.canonical_bytes().decode(),
        manifest.media.digest,
        source.canonical_id,
    )
    if existing is not None and tuple(existing) != expected:
        raise ValueError("benchmark clip binding changed")
    db.execute(
        "INSERT OR IGNORE INTO clips VALUES (?,?,?,?,?)",
        (manifest.clip_id, source.canonical_id, *expected[:3]),
    )
    task_id = content_digest([receipt.context.task, manifest.clip_id])
    task = db.execute(
        "SELECT campaign_id,task,clip_id,model_digest,inference_digest,state FROM annotation_tasks WHERE id=?",
        (task_id,),
    ).fetchone()
    values = (
        plan.campaign_id,
        receipt.context.task,
        manifest.clip_id,
        receipt.context.model.sha256,
        receipt.context.configuration.digest,
    )
    if task is not None and (
        tuple(task)[:5] != values
        or task["state"] not in {"predicted", "reviewing", "adopted"}
    ):
        raise ValueError("benchmark task binding/state changed")
    db.execute(
        "INSERT OR IGNORE INTO annotation_tasks VALUES (?,?,?,?,?,?,'predicted')",
        (task_id, *values),
    )
    previous = db.execute(
        "SELECT receipt FROM annotation_predictions WHERE task_id=?", (task_id,)
    ).fetchone()
    if (
        previous is not None
        and ArtifactRef.model_validate_json(previous[0]) != reference
    ):
        raise ValueError("benchmark prediction binding changed")
    db.execute(
        "INSERT OR IGNORE INTO annotation_predictions VALUES (?,?)",
        (task_id, reference.canonical_bytes().decode()),
    )
