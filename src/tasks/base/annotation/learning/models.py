"""Atomic model adoption and replayable receipts, including the commit/file gap."""

from __future__ import annotations

from ..artifacts import read_document, verify_reference, write_document
from ..contracts import (
    ArtifactRef,
    Document,
    ModelRef,
    PublicationReceipt,
    content_digest,
)
from ..execution.contracts import StageExit, StageLaunch
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, references
from ..registry import Registry
from ..storage.catalog import StorageCatalog
from ..storage.references import protection
from .adapters import LearningAdapter
from .contracts import (
    CampaignCompletion,
    CompletedStage,
    EvaluationReceipt,
    FixedEvaluation,
    LearningIntent,
    ModelState,
    SelectionRecord,
    SelectionVerdict,
    StageInput,
    TrainingReceipt,
)
from .validation import check_evaluation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_models (
    task TEXT NOT NULL, version TEXT NOT NULL, state TEXT NOT NULL,
    PRIMARY KEY(task, version)
);
CREATE TABLE IF NOT EXISTS annotation_current_models (
    task TEXT PRIMARY KEY, state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS annotation_model_bootstrap (
    task TEXT PRIMARY KEY, request TEXT NOT NULL, evidence TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS annotation_model_selections (
    id TEXT PRIMARY KEY, record TEXT NOT NULL
);
"""


def _selection_documents(
    layout: AnnotationLayout,
    verdict: SelectionVerdict,
    intent: LearningIntent,
    evaluation: EvaluationReceipt,
) -> tuple[Document, ...]:
    """Retain the known evidence graph used by check_evaluation on every replay."""
    evaluated = read_document(layout, verdict.evaluation, CompletedStage)
    trained = read_document(layout, evaluation.training, CompletedStage)
    documents: list[Document] = [intent, evaluation, evaluated, trained]
    for completed in (evaluated, trained):
        request = read_document(layout, completed.request, StageInput)
        execution = read_document(layout, completed.execution, StageExit)
        launch = read_document(layout, execution.launch, StageLaunch)
        documents.extend((request, execution, launch))
    documents.append(read_document(layout, trained.result, TrainingReceipt))
    completion = read_document(layout, intent.completion, CampaignCompletion)
    documents.extend(
        (
            completion,
            read_document(layout, completion.plan, CampaignPlan),
            read_document(layout, intent.fixed_evaluation, FixedEvaluation),
        )
    )
    publications = {intent.publication}
    publications.update(
        product.publication
        for task in completion.tasks
        for product in task.products
        if product.publication is not None
    )
    documents.extend(
        read_document(layout, ref, PublicationReceipt) for ref in publications
    )
    return tuple(documents)


class ModelRegistry:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.registry = Registry(layout)
        self.registry.initialize("annotation-learning-models-v1", _SCHEMA)
        self.catalog = StorageCatalog(layout)

    def bootstrap(self, model: ModelRef, evidence: ArtifactRef) -> ModelState:
        """Explicit initial registration, never a newest-file or last lookup."""
        for ref in (*references(model), evidence):
            verify_reference(self.layout, ref)
        state = ModelState(generation=1, model=model)
        request = content_digest([model.model_dump(), evidence.model_dump()])
        with self.registry.transaction() as db:
            old = db.execute(
                "SELECT request FROM annotation_model_bootstrap WHERE task=?",
                (model.task,),
            ).fetchone()
            if old is not None and old["request"] != request:
                raise ValueError("initial model registration already differs")
            self.catalog.protect(
                protection(
                    self.layout,
                    identity="model-bootstrap-v1-" + request,
                    kind="model",
                    reason="explicit initial model weights, configuration, and evidence",
                    roots=(evidence,),
                    documents=(model,),
                ),
                database=db,
            )
            if old is not None:
                return state
            db.execute(
                "INSERT INTO annotation_model_bootstrap VALUES (?,?,?)",
                (model.task, request, evidence.model_dump_json()),
            )
            db.execute(
                "INSERT INTO annotation_models VALUES (?,?,?)",
                (model.task, model.version, state.model_dump_json()),
            )
            db.execute(
                "INSERT INTO annotation_current_models VALUES (?,?)",
                (model.task, state.model_dump_json()),
            )
        return state

    def current(self, task: str) -> ModelState:
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT state FROM annotation_current_models WHERE task=?", (task,)
            ).fetchone()
        if row is None:
            raise ValueError("task has no explicitly adopted initial model")
        state: ModelState = ModelState.model_validate_json(row["state"])
        return state

    def adopted_state(self, model: ModelRef) -> ModelState:
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT state FROM annotation_models WHERE task=? AND version=?",
                (model.task, model.version),
            ).fetchone()
        if row is None:
            raise ValueError("campaign model has never been adopted")
        state: ModelState = ModelState.model_validate_json(row["state"])
        if state.model != model:
            raise ValueError("model version already identifies different weights")
        return state

    def select(self, verdict_ref: ArtifactRef, adapter: LearningAdapter) -> ArtifactRef:
        """Task judgment is opaque; acceptance still requires an unchanged baseline."""
        verdict = read_document(self.layout, verdict_ref, SelectionVerdict)
        intent, evaluation = check_evaluation(self.layout, verdict.evaluation)
        if adapter.task != intent.task:
            raise ValueError("model selection reference adapter task mismatch")
        if intent.selection_policy is None or verdict.policy != intent.selection_policy:
            raise ValueError("model selection requires the frozen adoption policy")
        expected, candidate = evaluation.baseline, evaluation.candidate
        if expected.model.task != candidate.task or candidate == expected.model:
            raise ValueError("candidate must be a new model of the same task")
        for ref in (*references(candidate), verdict.policy, evaluation.metrics):
            verify_reference(self.layout, ref)
        if self.adopted_state(expected.model) != expected:
            raise ValueError("evaluation baseline generation is not the adopted state")
        evidence = _selection_documents(self.layout, verdict, intent, evaluation)
        dependencies = adapter.retained_references(self.layout, intent, evaluation)
        # The pointer and complete receipt payload commit together. A crash before
        # the file write is recovered from this exact row, without another CAS.
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT record FROM annotation_model_selections WHERE id=?",
                (verdict_ref.digest,),
            ).fetchone()
            if row is not None:
                record = SelectionRecord.model_validate_json(row["record"])
                if record.verdict != verdict_ref:
                    raise ValueError("selection identity changed")
            else:
                row = db.execute(
                    "SELECT state FROM annotation_current_models WHERE task=?",
                    (candidate.task,),
                ).fetchone()
                if row is None:
                    raise ValueError("candidate has no adopted baseline")
                current = ModelState.model_validate_json(row["state"])
                adopted = None
                if current != expected:
                    status = "reevaluation_required"
                    reason = "adopted baseline changed; evaluate against the new baseline explicitly"
                elif verdict.status == "rejected":
                    status, reason = "rejected", verdict.reason
                else:
                    exists = db.execute(
                        "SELECT 1 FROM annotation_models WHERE task=? AND version=?",
                        (candidate.task, candidate.version),
                    ).fetchone()
                    if exists is not None:
                        raise ValueError("candidate version has already been adopted")
                    adopted = ModelState(
                        generation=current.generation + 1, model=candidate
                    )
                    db.execute(
                        "INSERT INTO annotation_models VALUES (?,?,?)",
                        (candidate.task, candidate.version, adopted.model_dump_json()),
                    )
                    db.execute(
                        "UPDATE annotation_current_models SET state=? WHERE task=?",
                        (adopted.model_dump_json(), candidate.task),
                    )
                    status, reason = "adopted", verdict.reason
                record = SelectionRecord.model_validate(
                    {
                        "verdict": verdict_ref,
                        "status": status,
                        "expected": expected,
                        "candidate": candidate,
                        "observed": current,
                        "adopted": adopted,
                        "reason": reason,
                    }
                )
                db.execute(
                    "INSERT INTO annotation_model_selections VALUES (?,?)",
                    (verdict_ref.digest, record.model_dump_json()),
                )
            self.catalog.protect(
                protection(
                    self.layout,
                    identity="model-selection-v3-" + verdict_ref.digest,
                    kind="model",
                    reason="model decision, compared weights, and fixed learning inputs",
                    roots=(verdict_ref,),
                    documents=(verdict, record, *evidence),
                    dependencies=dependencies,
                ),
                database=db,
            )
        reference = write_document(
            self.layout,
            self.layout.path(
                "models", candidate.task, "selections", verdict_ref.digest + ".json"
            ),
            record,
        )
        # The model pointer/weights already committed atomically above. Retrying
        # the commit/file gap materializes and protects this exact stored record.
        self.catalog.protect(
            protection(
                self.layout,
                identity="model-selection-record-v1-" + verdict_ref.digest,
                kind="model",
                reason="materialized model decision receipt",
                roots=(reference,),
                documents=(record,),
            )
        )
        result: ArtifactRef = reference
        return result
