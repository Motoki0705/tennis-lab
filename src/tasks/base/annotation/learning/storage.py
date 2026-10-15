"""Retain learning inputs and validated results independently of process lifetime."""

from __future__ import annotations

import sqlite3

from ..artifacts import annotation_reference, read_document
from ..contracts import ArtifactRef, PublicationReceipt
from ..execution.contracts import StageExit, StageLaunch
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, references
from ..storage import StorageCatalog
from ..storage.contracts import Protection
from ..storage.references import distinct, protection
from .contracts import (
    CampaignCompletion,
    CompletedStage,
    EvaluationReceipt,
    FixedEvaluation,
    LearningIntent,
    Retry,
    StageInput,
    TrainingReceipt,
)


class LearningReferences:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.catalog = StorageCatalog(layout)
        self.catalog.registry.initialize(
            "learning-references.v1",
            """
            CREATE TABLE IF NOT EXISTS learning_input_protections (
                intent TEXT PRIMARY KEY,
                protection TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS learning_result_protections (
                result TEXT PRIMARY KEY,
                protection TEXT NOT NULL
            );
            """,
        )

    def register(
        self,
        reference: ArtifactRef,
        intent: LearningIntent,
        dependencies: tuple[ArtifactRef, ...],
        *,
        database: sqlite3.Connection,
    ) -> None:
        completion = read_document(self.layout, intent.completion, CampaignCompletion)
        plan = read_document(self.layout, completion.plan, CampaignPlan)
        fixed = read_document(self.layout, intent.fixed_evaluation, FixedEvaluation)
        publications = tuple(
            read_document(self.layout, item.publication, PublicationReceipt)
            for task in completion.tasks
            for item in task.products
            if item.publication is not None
        )
        record = protection(
            self.layout,
            identity="learning-inputs-v1-" + reference.digest,
            kind="training",
            reason="registered cumulative teachers, fixed evaluation and model inputs",
            roots=(reference,),
            documents=(intent, completion, plan, fixed, *publications),
            dependencies=dependencies,
        )
        self.catalog.protect_once(record, database=database)
        database.execute(
            "INSERT OR IGNORE INTO learning_input_protections VALUES (?,?)",
            (reference.digest, record.canonical_bytes().decode()),
        )

    def inputs(self, reference: ArtifactRef) -> tuple[ArtifactRef, ...]:
        with self.catalog.registry.transaction() as database:
            row = database.execute(
                "SELECT protection FROM learning_input_protections WHERE intent=?",
                (reference.digest,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    "learning input protection is missing; register explicitly"
                )
            record = Protection.model_validate_json(row[0])
            if reference not in record.references:
                raise ValueError("learning input protection belongs to another intent")
            self.catalog.protect_once(record, database=database)
        retained: tuple[ArtifactRef, ...] = record.references
        return retained

    def _completed(
        self, reference: ArtifactRef, intent: ArtifactRef
    ) -> tuple[ArtifactRef, ...]:
        completed = read_document(self.layout, reference, CompletedStage)
        request = read_document(self.layout, completed.request, StageInput)
        execution = read_document(self.layout, completed.execution, StageExit)
        launch = read_document(self.layout, execution.launch, StageLaunch)
        if (
            request.intent != intent
            or execution.status != "succeeded"
            or execution.exit_code != 0
            or launch.input != completed.request
        ):
            raise ValueError(
                "retained result differs from its successful learning stage"
            )
        result = (
            read_document(self.layout, completed.result, TrainingReceipt)
            if request.stage == "train"
            else read_document(self.layout, completed.result, EvaluationReceipt)
        )
        if result.request != completed.request:
            raise ValueError("retained result belongs to another learning request")
        retained: tuple[ArtifactRef, ...] = distinct(
            self.layout,
            (
                reference,
                *(
                    ref
                    for doc in (completed, request, execution, launch, result)
                    for ref in references(doc)
                ),
            ),
        )
        return retained

    def completed_inputs(self, reference: ArtifactRef) -> tuple[ArtifactRef, ...]:
        with self.catalog.registry.transaction() as database:
            row = database.execute(
                "SELECT protection FROM learning_result_protections WHERE result=?",
                (reference.digest,),
            ).fetchone()
            if row is None:
                raise ValueError("learning result protection is missing")
            record = Protection.model_validate_json(row[0])
            if reference not in record.references:
                raise ValueError("learning result protection belongs to another result")
            self.catalog.protect_once(record, database=database)
        retained: tuple[ArtifactRef, ...] = record.references
        return retained

    def request(
        self, reference: ArtifactRef, request: StageInput
    ) -> tuple[ArtifactRef, ...]:
        dependencies = self.inputs(request.intent)
        if request.training is not None:
            completed = read_document(self.layout, request.training, CompletedStage)
            training = read_document(self.layout, completed.request, StageInput)
            if training.stage != "train":
                raise ValueError("evaluation requires a completed training stage")
            dependencies += self._completed(request.training, request.intent)
            dependencies += self.completed_inputs(request.training)
        if request.attempt > 1:
            retry_ref = annotation_reference(
                self.layout, self.layout.artifact_path(reference).parent / "retry.json"
            )
            retry = read_document(self.layout, retry_ref, Retry)
            previous = read_document(self.layout, retry.previous, StageInput)
            if (
                previous.intent != request.intent
                or previous.stage != request.stage
                or previous.attempt + 1 != request.attempt
                or previous.training != request.training
                or retry.resume_checkpoint != request.resume_checkpoint
            ):
                raise ValueError("learning retry differs from its retained request")
            dependencies += (retry_ref, *references(retry), *references(previous))
        record = protection(
            self.layout,
            identity="learning-request-v1-" + reference.digest,
            kind="training",
            reason="pending, resumable or completed learning request inputs",
            roots=(reference,),
            documents=(request,),
            dependencies=dependencies,
        )
        self.catalog.protect_once(record)
        retained: tuple[ArtifactRef, ...] = record.references
        return retained

    def complete(
        self,
        reference: ArtifactRef,
        completed: CompletedStage,
        dependencies: tuple[ArtifactRef, ...],
    ) -> None:
        request = read_document(self.layout, completed.request, StageInput)
        record = protection(
            self.layout,
            identity="learning-result-v1-" + reference.digest,
            kind="training",
            reason="validated learning result retained for evaluation and adoption",
            roots=(reference,),
            dependencies=(
                *self.request(completed.request, request),
                *self._completed(reference, request.intent),
                *dependencies,
            ),
        )
        with self.catalog.registry.transaction() as database:
            self.catalog.protect_once(record, database=database)
            database.execute(
                "INSERT OR IGNORE INTO learning_result_protections VALUES (?,?)",
                (reference.digest, record.canonical_bytes().decode()),
            )
