"""New-clip-only inference transactions; model and label meaning belong to adapters."""

from __future__ import annotations

from pathlib import Path
from sqlite3 import Row
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
from .plan import CampaignPlan, TaskPlan, references


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

    def completed(self, adapter: PredictionAdapter) -> tuple[ArtifactRef, ...]:
        """Verify registered results for this campaign without producing predictions."""
        task = self._task(adapter)
        with self.registry.transaction() as database:
            rows = database.execute(
                "SELECT t.*,c.manifest,p.receipt FROM annotation_tasks t "
                "JOIN clips c ON c.id=t.clip_id "
                "LEFT JOIN annotation_predictions p ON p.task_id=t.id "
                "WHERE t.campaign_id=? AND t.task=? ORDER BY t.clip_id",
                (self.plan.campaign_id, task.task),
            ).fetchall()
        results = []
        for row in rows:
            if row["receipt"] is None:
                if row["state"] != "pending":
                    raise ValueError(
                        "advanced annotation task has no prediction receipt"
                    )
                continue
            self._completed_row(row, task, adapter)
            receipt_ref = ArtifactRef.model_validate_json(row["receipt"])
            results.append(receipt_ref)
        return tuple(results)

    def require_completed(
        self,
        reference: ArtifactRef,
        adapter: PredictionAdapter,
        *,
        unreviewed: bool = False,
    ) -> PredictionReceipt:
        """Validate one exact registered result without loading an inference model."""
        task = self._task(adapter)
        with self.registry.transaction() as database:
            rows = database.execute(
                "SELECT t.*,c.manifest,p.receipt FROM annotation_tasks t "
                "JOIN clips c ON c.id=t.clip_id "
                "JOIN annotation_predictions p ON p.task_id=t.id "
                "WHERE t.campaign_id=? AND t.task=? AND p.receipt=?",
                (
                    self.plan.campaign_id,
                    task.task,
                    reference.canonical_bytes().decode(),
                ),
            ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                "prediction receipt is not registered to this campaign/task"
            )
        if unreviewed and rows[0]["state"] != "predicted":
            raise ValueError("prediction preview requires an unreviewed prediction")
        return self._completed_row(rows[0], task, adapter)

    def _task(self, adapter: PredictionAdapter) -> TaskPlan:
        task = next(
            (task for task in self.plan.tasks if task.task == adapter.task), None
        )
        if task is None:
            raise ValueError("prediction task is absent from the frozen campaign")
        return task

    def _completed_row(
        self, row: Row, task: TaskPlan, adapter: PredictionAdapter
    ) -> PredictionReceipt:
        if row["state"] not in {"predicted", "reviewing", "adopted"} or (
            row["model_digest"],
            row["inference_digest"],
        ) != (task.model.sha256, task.inference.digest):
            raise ValueError("prediction ledger differs from the frozen campaign")
        manifest_ref = ArtifactRef.model_validate_json(row["manifest"])
        manifest = read_document(self.layout, manifest_ref, ClipManifest)
        if manifest.clip_id != row["clip_id"]:
            raise ValueError("registered clip identity differs from its manifest")
        context = PredictionContext(
            task=task.task,
            manifest=manifest_ref,
            clip_id=manifest.clip_id,
            model=task.model,
            configuration=task.inference,
        )
        for reference in references(context):
            verify_reference(self.layout, reference)
        verify_reference(self.layout, manifest.media)
        return self._verify(
            ArtifactRef.model_validate_json(row["receipt"]), context, manifest, adapter
        )

    def run(self, manifest_ref: ArtifactRef, adapter: PredictionAdapter) -> ArtifactRef:
        if self.plan.phase == "benchmark" or self.layout.review_namespace is not None:
            raise ValueError(
                "benchmark review must reuse predictions without inference"
            )
        task = self._task(adapter)
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
    ) -> PredictionReceipt:
        receipt: PredictionReceipt = read_document(
            self.layout, reference, PredictionReceipt
        )
        if (
            receipt.context != context
            or receipt.prediction.request_id != context.request_id
        ):
            raise ValueError("prediction receipt differs from the frozen request")
        adapter.validate(self.layout, context, manifest, receipt.prediction.artifact)
        return receipt
