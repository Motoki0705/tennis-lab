"""Transfer completed screening dependencies to durable JPEG preparation."""

from __future__ import annotations

from ..artifacts import read_document
from ..contracts import ArtifactRef, ClipManifest
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, references
from ..storage import StorageCatalog
from ..storage.contracts import Protection
from ..storage.references import distinct
from ..workers.contracts import AttemptResult, WorkRequest
from .contracts import PreparationReceipt


def handoff_screening_inputs(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    source_id: str,
    media: ArtifactRef,
    prepared_ref: ArtifactRef,
) -> None:
    prepared = read_document(layout, prepared_ref, PreparationReceipt)
    manifests = tuple(
        read_document(layout, ref, ClipManifest) for ref in prepared.clips
    )
    catalog = StorageCatalog(layout)
    with catalog.registry.transaction() as database:
        if (
            database.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='worker_tasks'"
            ).fetchone()
            is None
        ):
            return  # Explicit/manual selection API has no screening worker inputs.
        for row in database.execute(
            "SELECT * FROM worker_tasks WHERE campaign=?", (plan.campaign_id,)
        ).fetchall():
            request = WorkRequest.model_validate_json(row["request"])
            if request.adapter != "source_screening" or request.subject_id != source_id:
                continue
            if (
                row["state"] != "complete"
                or row["plan_digest"] != plan.sha256
                or row["result"] is None
            ):
                raise ValueError(
                    "source retirement requires a completed screening worker"
                )
            result_ref = ArtifactRef.model_validate_json(row["result"])
            result = read_document(layout, result_ref, AttemptResult)
            if (
                result.outcome.status != "complete"
                or result.outcome.receipt != prepared.selection
            ):
                raise ValueError(
                    "screening worker result differs from the prepared selection"
                )
            identity = "worker-inputs-v1-" + request.sha256
            stored = database.execute(
                "SELECT document,released_reason FROM storage_protections WHERE protection_id=?",
                (identity,),
            ).fetchone()
            if stored is None:
                raise ValueError("screening worker has no registered input protection")
            original = Protection.model_validate_json(stored[0])
            replacement = Protection(
                protection_id="prepared-source-inputs-v1-" + request.sha256,
                kind="active",
                reason="Completed selection evidence and verified JPEG preparation replace the temporary source video",
                references=distinct(
                    layout,
                    (
                        *(
                            ref
                            for ref in original.references
                            if layout.artifact_path(ref) != layout.artifact_path(media)
                        ),
                        prepared_ref,
                        result_ref,
                        *prepared.clips,
                        *(
                            ref
                            for manifest in manifests
                            for ref in references(manifest)
                        ),
                    ),
                ),
            )
            catalog.protect_once(replacement, database=database)
            catalog.release(
                identity,
                expected_digest=original.sha256,
                reason="handoff to " + replacement.protection_id,
                database=database,
            )
