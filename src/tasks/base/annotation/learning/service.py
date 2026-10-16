"""Publication -> pinned intent -> shared queue -> evaluation -> atomic adoption."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from src.utils.paths import PROJECT_ROOT

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from ..contracts import ArtifactRef, PublicationReceipt
from ..execution.contracts import StageLaunch
from ..execution.runner import StageRunner, stage_directory
from ..paths import AnnotationLayout
from ..resources import BudgetStopped, CapacityUnavailable, ResourceRequest
from .adapters import EvaluationResolver, LearningAdapter, NoAdditionalTeachers
from .contracts import (
    CampaignCompletion,
    CompletedStage,
    EvaluationReceipt,
    FixedEvaluation,
    LearningIntent,
    LearningOutcome,
    LearningRegistration,
    LearningRuntime,
    Retry,
    SelectionRecord,
    SelectionVerdict,
    StageFailure,
    StageInput,
    TrainingReceipt,
)
from .models import ModelRegistry
from .storage import LearningReferences
from .validation import (
    check_completion,
    check_evaluation,
    check_evaluation_value,
    check_partitions,
    check_training_value,
    verify_inputs,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_learning_intents (
    task TEXT NOT NULL, publication TEXT NOT NULL, intent TEXT NOT NULL,
    PRIMARY KEY(task, publication)
);
"""


class StageExecutor(Protocol):
    def submit(self, launch: StageLaunch) -> dict[str, Any]: ...
    def status(self, ref: ArtifactRef) -> dict[str, Any]: ...


class Learning:
    def __init__(
        self, layout: AnnotationLayout, *, executor: StageExecutor | None = None
    ) -> None:
        self.layout = layout
        self.models = ModelRegistry(layout)
        self.registry = self.models.registry
        self.registry.initialize("annotation-learning-intents-v1", _SCHEMA)
        self.storage = LearningReferences(layout)
        self.executor: StageExecutor = (
            executor if executor is not None else StageRunner(layout)
        )

    def register(
        self, completion_ref: ArtifactRef, adapter: LearningAdapter
    ) -> LearningRegistration:
        completion = read_document(self.layout, completion_ref, CampaignCompletion)
        plan = check_completion(self.layout, completion)
        task = next((t for t in plan.tasks if t.task == adapter.task), None)
        if task is None or adapter.product not in {p.product for p in task.products}:
            raise ValueError("learning adapter does not own a planned training product")
        if not completion.complete:
            return LearningRegistration(
                task=adapter.task,
                status="annotation_incomplete",
                intent=None,
                reason="not all planned products are published or explicitly excluded",
            )
        products = next(t.products for t in completion.tasks if t.task == adapter.task)
        product = next(p for p in products if p.product == adapter.product)
        publication = (
            read_document(self.layout, product.publication, PublicationReceipt)
            if product.publication is not None
            else None
        )
        if publication is None or not publication.added_clips:
            return self._skip(
                completion_ref,
                adapter.task,
                "no_additional_teachers",
                product.reason or "publication contains no additional teachers",
            )
        if task.learning is None:
            return self._skip(
                completion_ref,
                adapter.task,
                "not_configured",
                "learning recipe and fixed validation are not configured",
            )
        assert product.publication is not None
        train = adapter.training_partition(self.layout, publication.snapshot)
        fixed_ref = task.learning.validation
        if task.learning.evaluation_recipe is not None:
            if not isinstance(adapter, EvaluationResolver):
                raise ValueError("task adapter does not support post-publication evaluation recipes")
            fixed_ref = adapter.resolve_evaluation(self.layout, task.learning, product.publication)
        fixed = read_document(self.layout, fixed_ref, FixedEvaluation)
        if (
            train.manifest != publication.snapshot
            or train.split != "train"
            or fixed.task != task.task
        ):
            raise ValueError(
                "training/fixed validation differ from the publication task"
            )
        if not set(publication.added_clips).intersection(train.clips):
            raise ValueError(
                "new teachers must include the cumulative training partition"
            )
        partitions = (train, fixed.validation, *((fixed.test,) if fixed.test else ()))
        if not set(publication.added_clips).issubset({clip for part in partitions for clip in part.clips}):
            raise ValueError("new teachers are missing from the declared train/validation/test partitions")
        check_partitions(*partitions)
        for partition in partitions:
            verify_reference(self.layout, partition.manifest)
            adapter.validate_partition(self.layout, partition)
        intent = LearningIntent(
            completion=completion_ref,
            task=task.task,
            publication=product.publication,
            train=train,
            parent=self.models.adopted_state(task.model),
            recipe=task.learning.recipe,
            fixed_evaluation=fixed_ref,
            selection_policy=task.learning.selection_policy,
            evaluation_recipe=task.learning.evaluation_recipe,
        )
        verify_inputs(self.layout, intent)
        eligibility = adapter.validate_intent(self.layout, intent)
        if isinstance(eligibility, NoAdditionalTeachers):
            return self._skip(
                completion_ref, task.task, "no_additional_teachers", eligibility.reason
            )
        dependencies = adapter.input_references(self.layout, intent)
        directory = self._directory(intent)
        with artifact_lock(
            self.layout,
            "learning-register-" + task.task + "-" + product.publication.digest,
        ):
            with self.registry.transaction() as db:
                old = db.execute(
                    "SELECT intent FROM annotation_learning_intents WHERE task=? AND publication=?",
                    (task.task, product.publication.digest),
                ).fetchone()
            if old is not None:
                ref = ArtifactRef.model_validate_json(old["intent"])
                if read_document(self.layout, ref, LearningIntent) != intent:
                    raise ValueError(
                        "publication already has a different learning intent"
                    )
            else:
                ref = write_document(self.layout, directory / "intent.json", intent)
            with self.registry.transaction() as db:
                self.storage.register(ref, intent, dependencies, database=db)
                db.execute(
                    "INSERT OR IGNORE INTO annotation_learning_intents VALUES (?,?,?)",
                    (task.task, product.publication.digest, ref.model_dump_json()),
                )
        return LearningRegistration(
            task=task.task,
            status="ready",
            intent=ref,
            reason="additional teachers and fixed learning inputs verified",
        )

    def _skip(
        self,
        completion_ref: ArtifactRef,
        task: str,
        status: Literal["no_additional_teachers", "not_configured"],
        reason: str,
    ) -> LearningRegistration:
        completion = read_document(self.layout, completion_ref, CampaignCompletion)
        plan = check_completion(self.layout, completion)
        result = LearningRegistration(
            task=task, status=status, intent=None, reason=reason
        )
        write_document(
            self.layout,
            self.layout.path(
                "campaigns", plan.campaign_id, "learning", task, "skipped.json"
            ),
            result,
        )
        return result

    def _directory(self, intent: LearningIntent) -> Path:
        completion = read_document(self.layout, intent.completion, CampaignCompletion)
        plan = check_completion(self.layout, completion)
        directory: Path = self.layout.path(
            "campaigns", plan.campaign_id, "learning", intent.task
        )
        return directory

    def _require_registered(
        self, ref: ArtifactRef, adapter: LearningAdapter
    ) -> LearningIntent:
        intent: LearningIntent = read_document(self.layout, ref, LearningIntent)
        if adapter.task != intent.task:
            raise ValueError("learning adapter task mismatch")
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT intent FROM annotation_learning_intents WHERE task=? AND publication=?",
                (intent.task, intent.publication.digest),
            ).fetchone()
        if row is None or ArtifactRef.model_validate_json(row["intent"]) != ref:
            raise ValueError("learning intent is not registered")
        publication = read_document(self.layout, intent.publication, PublicationReceipt)
        if publication.product != adapter.product:
            raise ValueError("learning adapter product mismatch")
        self.storage.inputs(ref)
        return intent

    def tick(
        self,
        intent_ref: ArtifactRef,
        adapter: LearningAdapter,
        runtime: LearningRuntime,
    ) -> LearningOutcome:
        intent = self._require_registered(intent_ref, adapter)
        with artifact_lock(
            self.layout, "learning-run-" + intent_ref.digest, blocking=False
        ):
            directory = self._directory(intent)
            write_document(self.layout, directory / "runtime.json", runtime)
            training = self._stage(intent_ref, intent, adapter, runtime, "train", None)
            if isinstance(training, LearningOutcome):
                return training
            evaluation = self._stage(
                intent_ref, intent, adapter, runtime, "evaluate", training
            )
            if isinstance(evaluation, LearningOutcome):
                return evaluation
            if intent.selection_policy is None:
                result = LearningOutcome(
                    intent=intent_ref,
                    state="not_configured",
                    stage="select",
                    training=training,
                    evaluation=evaluation,
                    reason="model adoption policy is not configured; current model retained",
                )
                write_document(
                    self.layout, directory / "selection-not-configured.json", result
                )
                return result
            path = directory / "verdict.json"
            if path.exists():
                verdict_ref = annotation_reference(self.layout, path)
                verdict = read_document(self.layout, verdict_ref, SelectionVerdict)
            else:
                check_evaluation(self.layout, evaluation)
                verdict = adapter.select(self.layout, intent, evaluation)
                if (
                    verdict.evaluation != evaluation
                    or verdict.policy != intent.selection_policy
                ):
                    raise ValueError(
                        "task verdict differs from the completed evaluation/frozen policy"
                    )
                verdict_ref = write_document(self.layout, path, verdict)
            if (
                verdict.evaluation != evaluation
                or verdict.policy != intent.selection_policy
            ):
                raise ValueError("cached verdict differs from the fixed inputs")
            selection_ref = self.models.select(verdict_ref)
            selection = read_document(self.layout, selection_ref, SelectionRecord)
            return LearningOutcome(
                intent=intent_ref,
                state=selection.status,
                stage="select",
                training=training,
                evaluation=evaluation,
                selection=selection_ref,
                reason=selection.reason,
            )

    def _stage_input(
        self,
        directory: Path,
        ref: ArtifactRef,
        stage: Literal["train", "evaluate"],
        training: ArtifactRef | None,
    ) -> tuple[Path, ArtifactRef, StageInput]:
        root = directory / stage
        attempts = sorted(
            int(p.name) for p in root.glob("*") if p.is_dir() and p.name.isdecimal()
        )
        attempt = max(attempts, default=1)
        current = root / str(attempt)
        request_path = current / "request.json"
        if request_path.exists():
            request_ref = annotation_reference(self.layout, request_path)
            request = read_document(self.layout, request_ref, StageInput)
            if (request.intent, request.stage, request.attempt, request.training) != (
                ref,
                stage,
                attempt,
                training,
            ):
                raise ValueError("stage request changed its fixed inputs")
        else:
            if attempt != 1:
                raise ValueError("retry directory has no durable stage request")
            request = StageInput(intent=ref, stage=stage, attempt=1, training=training)
            request_ref = write_document(self.layout, request_path, request)
        self.storage.request(request_ref, request)
        return current, request_ref, request

    def _launch(
        self,
        intent: LearningIntent,
        request_ref: ArtifactRef,
        request: StageInput,
        adapter: LearningAdapter,
        runtime: LearningRuntime,
        directory: Path,
    ) -> StageLaunch:
        completion = read_document(self.layout, intent.completion, CampaignCompletion)
        cost = runtime.train if request.stage == "train" else runtime.evaluate
        return StageLaunch(
            scope=runtime.scope,
            resource=ResourceRequest(
                request_id="learning-" + request.sha256,
                plan=completion.plan,
                stage=cost.stage,
                pss_bytes=cost.pss_bytes,
                storage_bytes=cost.storage_bytes,
                gpu_seconds=cost.gpu_seconds,
            ),
            command=adapter.command(
                self.layout, request_ref, directory / "result.json"
            ),
            code_root=str(PROJECT_ROOT),
            timeout_seconds=cost.timeout_seconds,
            input=request_ref,
            dependencies=self.storage.request(request_ref, request),
            queue=runtime.queue,
            test_root=str(self.layout.root) if self.layout.mode == "test" else None,
        )

    def _validated_result(
        self,
        completed: CompletedStage,
        intent: LearningIntent,
        adapter: LearningAdapter,
        stage: Literal["train", "evaluate"],
    ) -> TrainingReceipt | EvaluationReceipt:
        if stage == "train":
            _, training = check_training_value(self.layout, completed)
            adapter.validate_training(self.layout, intent, training)
            return training
        _, evaluation = check_evaluation_value(self.layout, completed)
        adapter.validate_evaluation(self.layout, intent, evaluation)
        return evaluation

    def _stage(
        self,
        ref: ArtifactRef,
        intent: LearningIntent,
        adapter: LearningAdapter,
        runtime: LearningRuntime,
        stage: Literal["train", "evaluate"],
        training: ArtifactRef | None,
    ) -> ArtifactRef | LearningOutcome:
        directory, request_ref, request = self._stage_input(
            self._directory(intent), ref, stage, training
        )
        completed_path, failed_path = (
            directory / "completed.json",
            directory / "failure.json",
        )
        if completed_path.exists():
            completed_ref = annotation_reference(self.layout, completed_path)
            completed = read_document(self.layout, completed_ref, CompletedStage)
            if completed.request != request_ref:
                raise ValueError("completed stage belongs to another request")
            result = self._validated_result(completed, intent, adapter, stage)
            self.storage.complete(
                completed_ref, completed, adapter.result_references(self.layout, result)
            )
            return completed_ref
        if failed_path.exists():
            failure = read_document(
                self.layout,
                annotation_reference(self.layout, failed_path),
                StageFailure,
            )
            if failure.request != request_ref:
                raise ValueError("failure receipt belongs to another request")
            return LearningOutcome(
                intent=ref,
                stage=stage,
                state="failed",
                launch=failure.launch,
                training=training,
                reason=failure.reason,
            )
        launch = self._launch(intent, request_ref, request, adapter, runtime, directory)
        launch_ref = write_document(
            self.layout, stage_directory(self.layout, launch) / "launch.json", launch
        )
        pending = "training_pending" if stage == "train" else "evaluation_pending"
        active = "training" if stage == "train" else "evaluating"
        try:
            state = self.executor.status(launch_ref)
            if state["state"] in {"pending", "unregistered"}:
                state = self.executor.submit(launch)
        except (BudgetStopped, CapacityUnavailable, BlockingIOError) as error:
            return LearningOutcome.model_validate(
                dict(
                    intent=ref,
                    state=pending,
                    stage=stage,
                    training=training,
                    launch=launch_ref,
                    reason=str(error),
                )
            )
        if state["state"] in {"pending", "queued", "running", "draining", "unfinished"}:
            return LearningOutcome.model_validate(
                dict(
                    intent=ref,
                    state=pending
                    if state["state"] in {"pending", "queued"}
                    else active,
                    stage=stage,
                    training=training,
                    launch=launch_ref,
                    reason=state.get("reason"),
                )
            )
        if state["state"] != "succeeded":
            failure = StageFailure(
                request=request_ref,
                launch=launch_ref,
                reason=state.get("reason") or "learning stage " + str(state["state"]),
            )
            write_document(self.layout, failed_path, failure)
            return LearningOutcome(
                intent=ref,
                stage=stage,
                state="failed",
                launch=launch_ref,
                training=training,
                reason=failure.reason,
            )
        try:
            completed = CompletedStage(
                request=request_ref,
                result=annotation_reference(self.layout, directory / "result.json"),
                execution=ArtifactRef.model_validate(state["exit"]),
            )
            result = self._validated_result(completed, intent, adapter, stage)
        except (ValueError, OSError) as error:
            failure = StageFailure(
                request=request_ref,
                launch=launch_ref,
                reason=f"invalid learning result: {error}",
            )
            write_document(self.layout, failed_path, failure)
            return LearningOutcome(
                intent=ref,
                stage=stage,
                state="failed",
                launch=launch_ref,
                training=training,
                reason=failure.reason,
            )
        completed_ref = write_document(self.layout, completed_path, completed)
        self.storage.complete(
            completed_ref, completed, adapter.result_references(self.layout, result)
        )
        return completed_ref

    def retry(
        self,
        ref: ArtifactRef,
        adapter: LearningAdapter,
        stage: Literal["train", "evaluate"],
        *,
        reason: str,
        resume_checkpoint: ArtifactRef | None = None,
    ) -> ArtifactRef:
        intent = self._require_registered(ref, adapter)
        if not reason.strip():
            raise ValueError("retry needs an explicit reason")
        with artifact_lock(self.layout, "learning-run-" + ref.digest, blocking=False):
            directory = self._directory(intent)
            if stage == "train" and (directory / "evaluate").exists():
                raise ValueError("cannot retrain after evaluation has started")
            root = directory / stage
            attempts = [
                int(p.name) for p in root.glob("*") if p.is_dir() and p.name.isdecimal()
            ]
            if not attempts:
                raise ValueError("no failed stage to retry")
            current = root / str(max(attempts))
            if (current / "retry.json").exists() and not (
                current / "request.json"
            ).exists():
                partial = read_document(
                    self.layout,
                    annotation_reference(self.layout, current / "retry.json"),
                    Retry,
                )
                if (
                    partial.reason != reason
                    or partial.resume_checkpoint != resume_checkpoint
                ):
                    raise ValueError(
                        "incomplete retry must be resumed with the same explicit inputs"
                    )
                old = read_document(self.layout, partial.previous, StageInput)
                if (
                    old.intent != ref
                    or old.stage != stage
                    or old.attempt + 1 != int(current.name)
                ):
                    raise ValueError(
                        "incomplete retry belongs to another learning stage"
                    )
                restored = StageInput(
                    intent=ref,
                    stage=stage,
                    attempt=old.attempt + 1,
                    training=old.training,
                    resume_checkpoint=resume_checkpoint,
                )
                restored_ref = write_document(
                    self.layout, current / "request.json", restored
                )
                self.storage.request(restored_ref, restored)
                return restored_ref
            if int(current.name) > 1 and not any(
                (current / name).exists()
                for name in ("retry.json", "request.json", "failure.json", "completed.json")
            ):
                # write_document may create the next directory before publishing
                # retry.json. Only an explicit retry can recover this gap, and
                # the immediately preceding attempt must still be a failure.
                current = root / str(int(current.name) - 1)
            if (
                not (current / "failure.json").exists()
                or (current / "completed.json").exists()
            ):
                raise ValueError("only a recorded failed stage can be retried")
            previous = annotation_reference(self.layout, current / "request.json")
            request = read_document(self.layout, previous, StageInput)
            if resume_checkpoint is not None:
                if stage != "train":
                    raise ValueError("evaluation cannot resume training")
                verify_reference(self.layout, resume_checkpoint)
                adapter.validate_resume(self.layout, intent, resume_checkpoint)
            next_request = StageInput(
                intent=ref,
                stage=stage,
                attempt=request.attempt + 1,
                training=request.training,
                resume_checkpoint=resume_checkpoint,
            )
            target = root / str(next_request.attempt)
            # The reason is durable first; a crash here is recovered by replaying
            # this same explicit retry, not by silently starting a new attempt.
            write_document(
                self.layout,
                target / "retry.json",
                Retry(
                    previous=previous,
                    reason=reason,
                    resume_checkpoint=resume_checkpoint,
                ),
            )
            next_ref = write_document(self.layout, target / "request.json", next_request)
            self.storage.request(next_ref, next_request)
            return next_ref
