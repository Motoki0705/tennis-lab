"""New-clip-only inference transactions; model and label meaning belong to adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from .collection.ledger import CollectionLedger
from .contracts import (
    ArtifactRef,
    ClipManifest,
    Document,
    ModelRef,
    PredictionRef,
    content_digest,
)
from .paths import AnnotationLayout
from .plan import CampaignPlan, references


class PredictionContext(Document):
    task: str
    manifest: ArtifactRef
    clip_id: str
    model: ModelRef
    configuration: ArtifactRef

    @property
    def request_id(self) -> str:
        result: str = content_digest(
            [
                self.task,
                self.clip_id,
                self.manifest.digest,
                self.model.sha256,
                self.configuration.digest,
            ]
        )
        return result


class PredictionReceipt(Document):
    context: PredictionContext
    prediction: PredictionRef


class PredictionAdapter(Protocol):
    task: str

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef: ...

    def validate(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        result: ArtifactRef,
    ) -> None: ...


class Predictions:
    def __init__(self, layout: AnnotationLayout, plan: CampaignPlan) -> None:
        self.layout, self.plan = layout, plan
        self.collection = CollectionLedger(layout)
        self.collection.register_campaign(plan)
        self.registry = self.collection.registry
        self.registry.initialize(
            "predictions.v1",
            """
            CREATE TABLE IF NOT EXISTS annotation_predictions (
                task_id TEXT PRIMARY KEY REFERENCES annotation_tasks(id), receipt TEXT NOT NULL
            );
        """,
        )

    def _commit(self, task_id: str, receipt: ArtifactRef) -> None:
        encoded = receipt.canonical_bytes().decode()
        with self.registry.transaction() as database:
            previous = database.execute(
                "SELECT receipt FROM annotation_predictions WHERE task_id=?", (task_id,)
            ).fetchone()
            if previous is not None and previous["receipt"] != encoded:
                raise ValueError(
                    "a clip prediction has already been registered with different bytes"
                )
            database.execute(
                "INSERT OR IGNORE INTO annotation_predictions VALUES (?,?)",
                (task_id, encoded),
            )
            database.execute(
                "UPDATE annotation_tasks SET state='predicted' WHERE id=? AND state='pending'",
                (task_id,),
            )

    def run(self, manifest_ref: ArtifactRef, adapter: PredictionAdapter) -> ArtifactRef:
        task = next(
            (task for task in self.plan.tasks if task.task == adapter.task), None
        )
        if task is None:
            raise ValueError("prediction task is absent from the frozen campaign")
        manifest = read_document(self.layout, manifest_ref, ClipManifest)
        context = PredictionContext(
            task=task.task,
            manifest=manifest_ref,
            clip_id=manifest.clip_id,
            model=task.model,
            configuration=task.inference,
        )
        task_id = content_digest([task.task, manifest.clip_id])
        with artifact_lock(self.layout, "prediction-" + task_id, blocking=False):
            with self.registry.transaction() as database:
                row = database.execute(
                    """SELECT t.*,c.manifest FROM annotation_tasks t
                    JOIN clips c ON c.id=t.clip_id WHERE t.id=?""",
                    (task_id,),
                ).fetchone()
                previous = database.execute(
                    "SELECT receipt FROM annotation_predictions WHERE task_id=?",
                    (task_id,),
                ).fetchone()
            if row is None or (
                row["campaign_id"],
                row["model_digest"],
                row["inference_digest"],
                row["manifest"],
            ) != (
                self.plan.campaign_id,
                task.model.sha256,
                task.inference.digest,
                manifest_ref.canonical_bytes().decode(),
            ):
                raise ValueError(
                    "normal inference only accepts newly registered clips in the exact campaign/model"
                )
            for reference in references(context):
                verify_reference(self.layout, reference)
            verify_reference(self.layout, manifest.media)
            directory = self.layout.path(
                "predictions", task.task, manifest.clip_id, context.request_id
            )
            if previous is not None:
                ref = ArtifactRef.model_validate_json(previous["receipt"])
                self._verify(ref, context, manifest, adapter)
                return ref
            if row["state"] != "pending":
                raise ValueError(
                    "annotation has advanced without a prediction receipt; refusing reinference"
                )
            receipt_path = directory / "receipt.json"
            if receipt_path.exists():
                ref = annotation_reference(self.layout, receipt_path)
                self._verify(ref, context, manifest, adapter)
            else:
                output = adapter.produce(self.layout, context, manifest, directory)
                adapter.validate(self.layout, context, manifest, output)
                verify_reference(self.layout, output)
                prediction = PredictionRef(
                    task=task.task,
                    clip_id=manifest.clip_id,
                    manifest_digest=manifest_ref.digest,
                    model_digest=task.model.sha256,
                    inference_digest=task.inference.digest,
                    artifact=output,
                )
                ref = write_document(
                    self.layout,
                    receipt_path,
                    PredictionReceipt(context=context, prediction=prediction),
                )
            self._commit(task_id, ref)
            return ref

    def _verify(
        self,
        reference: ArtifactRef,
        context: PredictionContext,
        manifest: ClipManifest,
        adapter: PredictionAdapter,
    ) -> None:
        receipt = read_document(self.layout, reference, PredictionReceipt)
        if (
            receipt.context != context
            or receipt.prediction.request_id != context.request_id
        ):
            raise ValueError("prediction receipt differs from the frozen request")
        adapter.validate(self.layout, context, manifest, receipt.prediction.artifact)
