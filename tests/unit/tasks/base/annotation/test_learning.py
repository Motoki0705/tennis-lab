from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ModelRef,
    ProductOutcome,
    PublicationReceipt,
    content_digest,
)
from src.tasks.base.annotation.execution.contracts import (
    QueueConfiguration,
    StageExit,
    StageLaunch,
)
from src.tasks.base.annotation.execution.runner import stage_directory
from src.tasks.base.annotation.learning import Learning, ModelRegistry
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    CompletedStage,
    DatasetPartition,
    EvaluationReceipt,
    FixedEvaluation,
    LearningIntent,
    LearningRuntime,
    Retry,
    SelectionRecord,
    SelectionVerdict,
    StageCost,
    StageFailure,
    StageInput,
    TaskProducts,
    TrainingReceipt,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, file_digest, save_plan
from src.tasks.base.annotation.resources import BudgetLimits, ResourceManager
from src.tasks.base.annotation.storage import (
    InventoryScope,
    ManagedArtifact,
    Retention,
    StorageCatalog,
    inventory,
    plan_gc,
)


class Crash(BaseException):
    pass


class Adapter:
    def retained_references(
        self,
        layout: AnnotationLayout,
        intent: LearningIntent,
        evaluation: EvaluationReceipt,
    ) -> tuple[ArtifactRef, ...]:
        return self.dependencies

    task = "detector"
    product = "boxes"

    def __init__(self) -> None:
        self.accept = True
        self.complete_metrics = True
        self.use_test = False
        self.selections = 0
        self.dependencies: tuple[ArtifactRef, ...] = ()
        self.result_dependencies: tuple[ArtifactRef, ...] = ()
        self.resume_dependencies: tuple[ArtifactRef, ...] = ()

    def training_partition(
        self, layout: AnnotationLayout, snapshot: ArtifactRef
    ) -> DatasetPartition:
        members = json.loads(layout.artifact_path(snapshot).read_bytes())
        return DatasetPartition(
            manifest=snapshot,
            split="train",
            sources=tuple(members["sources"]),
            clips=tuple(members["clips"]),
        )

    def validate_partition(
        self, layout: AnnotationLayout, partition: DatasetPartition
    ) -> None:
        members = json.loads(layout.artifact_path(partition.manifest).read_bytes())
        assert members == {
            "sources": list(partition.sources),
            "clips": list(partition.clips),
        }

    def validate_intent(self, layout: AnnotationLayout, intent: LearningIntent) -> None:
        assert layout.artifact_path(intent.recipe).read_bytes() == b"fresh optimizer"

    def input_references(
        self, layout: AnnotationLayout, intent: LearningIntent
    ) -> tuple[ArtifactRef, ...]:
        return self.dependencies

    def result_references(
        self, layout: AnnotationLayout, result: TrainingReceipt | EvaluationReceipt
    ) -> tuple[ArtifactRef, ...]:
        return self.result_dependencies

    def command(
        self, layout: AnnotationLayout, request: ArtifactRef, result: Path
    ) -> tuple[str, ...]:
        return (
            sys.executable,
            "fixture-only",
            str(layout.artifact_path(request)),
            str(result),
        )

    def validate_training(
        self, layout: AnnotationLayout, intent: LearningIntent, receipt: TrainingReceipt
    ) -> None:
        assert (
            layout.artifact_path(receipt.provenance).read_bytes()
            == intent.canonical_bytes()
        )

    def validate_evaluation(
        self,
        layout: AnnotationLayout,
        intent: LearningIntent,
        receipt: EvaluationReceipt,
    ) -> None:
        metrics = json.loads(layout.artifact_path(receipt.metrics).read_bytes())
        if not metrics["complete"]:
            raise ValueError("required task validation metric is missing")

    def resume_references(
        self, layout: AnnotationLayout, intent: LearningIntent, checkpoint: ArtifactRef
    ) -> tuple[ArtifactRef, ...]:
        if layout.artifact_path(checkpoint).read_bytes() != intent.canonical_bytes():
            raise ValueError("checkpoint belongs to another training intent")
        return self.resume_dependencies

    def select(
        self, layout: AnnotationLayout, intent: LearningIntent, evaluation: ArtifactRef
    ) -> SelectionVerdict:
        self.selections += 1
        assert intent.selection_policy is not None
        return SelectionVerdict(
            evaluation=evaluation,
            policy=intent.selection_policy,
            status="passed" if self.accept else "rejected",
            reason="fixture task judgment",
        )


class Executor:
    """Persist fake queue admission; never execute a command or allocate a GPU."""

    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.accepted: list[ArtifactRef] = []
        self.crash_after_accept = False

    def status(self, ref: ArtifactRef) -> dict[str, Any]:
        launch = read_document(self.layout, ref, StageLaunch)
        directory = stage_directory(self.layout, launch)
        if (directory / "exit.json").exists():
            exit_ref = annotation_reference(self.layout, directory / "exit.json")
            receipt = read_document(self.layout, exit_ref, StageExit)
            return {
                "state": receipt.status,
                "exit": exit_ref.model_dump(),
                "reason": receipt.reason,
            }
        return {
            "state": "queued" if (directory / "queue.accepted").exists() else "pending"
        }

    def submit(self, launch: StageLaunch) -> dict[str, Any]:
        directory = stage_directory(self.layout, launch)
        ref = write_document(self.layout, directory / "launch.json", launch)
        if not (directory / "queue.accepted").exists():
            write_asset(
                self.layout, directory / "queue.accepted", ref.canonical_bytes()
            )
            self.accepted.append(ref)
        if self.crash_after_accept:
            self.crash_after_accept = False
            raise Crash("queue accepted, parent response lost")
        return {"state": "queued"}

    def finish(
        self, ref: ArtifactRef, adapter: Adapter, *, failed: bool = False
    ) -> ArtifactRef:
        launch = read_document(self.layout, ref, StageLaunch)
        assert launch.input is not None
        request = read_document(self.layout, launch.input, StageInput)
        intent = read_document(self.layout, request.intent, LearningIntent)
        result_path = Path(launch.command[-1])
        if request.stage == "train":
            weights = write_asset(
                self.layout,
                result_path.parent / "weights.bin",
                (intent.sha256 + "weights").encode(),
            )
            candidate = ModelRef(
                task=intent.task,
                version="candidate-" + intent.sha256[:12],
                training_weights=weights,
                inference_weights=weights,
                configuration=intent.parent.model.configuration,
            )
            provenance = write_asset(
                self.layout,
                result_path.parent / "provenance.json",
                intent.canonical_bytes(),
            )
            result: TrainingReceipt | EvaluationReceipt = TrainingReceipt(
                request=launch.input, candidate=candidate, provenance=provenance
            )
        else:
            assert request.training is not None
            completed = read_document(self.layout, request.training, CompletedStage)
            training = read_document(self.layout, completed.result, TrainingReceipt)
            fixed = read_document(self.layout, intent.fixed_evaluation, FixedEvaluation)
            metrics = write_asset(
                self.layout,
                result_path.parent / "metrics.json",
                json.dumps(
                    {"complete": adapter.complete_metrics, "task_metric": 2.5}
                ).encode(),
            )
            partition = fixed.test if adapter.use_test else fixed.validation
            assert partition is not None
            result = EvaluationReceipt(
                request=launch.input,
                training=request.training,
                candidate=training.candidate,
                baseline=intent.parent,
                validation=partition,
                metrics=metrics,
            )
        result_ref = write_document(self.layout, result_path, result)
        directory = stage_directory(self.layout, launch)
        stdout = write_asset(
            self.layout, directory / "stdout.log", b"synthetic execution"
        )
        stderr = write_asset(self.layout, directory / "stderr.log", b"")
        write_document(
            self.layout,
            directory / "exit.json",
            StageExit(
                launch=ref,
                status="failed" if failed else "succeeded",
                exit_code=1 if failed else 0,
                started_at="2026-01-01T00:00:00Z",
                ended_at="2026-01-01T00:00:01Z",
                elapsed_seconds=1.0,
                stdout=stdout,
                stderr=stderr,
                reason="interrupted after checkpoint" if failed else None,
            ),
        )
        return result_ref


class Environment:
    def __init__(self, root: Path) -> None:
        self.layout = AnnotationLayout.testing(root)
        self.executor = Executor(self.layout)
        self.adapter = Adapter()
        self.learning = Learning(self.layout, executor=self.executor)
        self.common = write_asset(
            self.layout, self.layout.path("models", "fixture.bin"), b"fixture"
        )
        self.model = ModelRef(
            task="detector",
            version="initial",
            training_weights=self.common,
            inference_weights=self.common,
            configuration=self.common,
        )
        self.learning.models.bootstrap(self.model, self.common)
        self.val = self.partition("validation", "val")
        self.test = self.partition("test", "test")
        self.fixed = write_document(
            self.layout,
            self.layout.path("models", "fixed-evaluation.json"),
            FixedEvaluation(task="detector", validation=self.val, test=self.test),
        )
        self.recipe = write_asset(
            self.layout, self.layout.path("models", "recipe.txt"), b"fresh optimizer"
        )
        self.adapter.dependencies = (
            write_asset(
                self.layout,
                self.layout.path("models", "nested-input.bin"),
                b"task input",
            ),
        )
        self.runtime = LearningRuntime(
            scope=self.common,
            queue=QueueConfiguration(
                script="/fixture/queue.sh",
                script_digest=self.common.digest,
                directory="/fixture/queue",
                session="fixture",
                issue=1135,
            ),
            train=StageCost(
                stage="train",
                pss_bytes=1,
                storage_bytes=1,
                gpu_seconds=60.0,
                timeout_seconds=40.0,
            ),
            evaluate=StageCost(
                stage="evaluate",
                pss_bytes=1,
                storage_bytes=1,
                gpu_seconds=60.0,
                timeout_seconds=40.0,
            ),
        )

    def partition(
        self, split: Literal["train", "validation", "test"], identity: str
    ) -> DatasetPartition:
        source, clip = (
            content_digest(["source", identity]),
            content_digest(["clip", identity]),
        )
        ref = write_asset(
            self.layout,
            self.layout.path("models", identity + "-dataset.json"),
            json.dumps({"sources": [source], "clips": [clip]}).encode(),
        )
        return DatasetPartition(
            manifest=ref, split=split, sources=(source,), clips=(clip,)
        )

    def campaign(
        self,
        name: str = "cycle-one",
        *,
        policy: bool = True,
        configured: bool = True,
        added: bool = True,
        incomplete: bool = False,
        overlap: str | None = None,
        model: ModelRef | None = None,
    ) -> ArtifactRef:
        train = self.partition("train", overlap or name)
        ref = self.common.model_dump()
        selected = model or self.model
        products = [
            {
                "product": p,
                "dataset_task": "detector",
                "dataset_version": name,
                "component": p,
                "base_snapshot": None,
                "generation_policy": ref,
            }
            for p in ("boxes", "poses", "ball-join")
        ]
        plan = CampaignPlan.model_validate_json(
            json.dumps(
                {
                    "campaign_id": name,
                    "phase": "production",
                    "annotation_root": str(self.layout.root),
                    "counts": {
                        "channels": 1,
                        "videos_per_channel": 1,
                        "max_clips_per_video": 1,
                    },
                    "selection": ref,
                    "preparation": ref,
                    "code_commit": "a" * 40,
                    "code_diff_digest": "b" * 64,
                    "agent_model": "gpt-6.1-sol",
                    "agent_effort": "max",
                    "resources": {
                        "codex_workers": 1,
                        "additional_storage_bytes": 1_000_000,
                        "minimum_free_bytes": 1,
                        "total_pss_bytes": 2 * 1024**3,
                        "gpu_seconds": 120,
                        "stages": [
                            {
                                "stage": stage,
                                "kind": "gpu",
                                "parallel": 1,
                                "threads_per_process": 1,
                            }
                            for stage in ("train", "evaluate")
                        ],
                    },
                    "tasks": [
                        {
                            "task": "detector",
                            "model": selected.model_dump(),
                            "inference": ref,
                            "review": {
                                "policy": ref,
                                "preview_evidence": ref,
                                "decision": ref,
                            },
                            "learning": {
                                "recipe": self.recipe.model_dump(),
                                "validation": self.fixed.model_dump(),
                                "selection_policy": ref if policy else None,
                            }
                            if configured
                            else None,
                            "products": products,
                        }
                    ],
                }
            )
        )
        plan_ref = annotation_reference(self.layout, save_plan(self.layout, plan))
        publication = PublicationReceipt(
            task="detector",
            product="boxes",
            request_id=content_digest(name),
            base_snapshot=None,
            snapshot=train.manifest,
            adoptions=(self.common,) if added else (),
            added_clips=train.clips if added else (),
            reused_clips=() if added else train.clips,
        )
        publication_ref = write_document(
            self.layout,
            self.layout.path("campaigns", name, "publication.json"),
            publication,
        )
        outcomes = (
            ProductOutcome(
                product="boxes",
                status="published",
                publication=publication_ref,
                reason=None,
            ),
            ProductOutcome(
                product="poses",
                status="pending" if incomplete else "not_applicable",
                publication=None,
                reason=None if incomplete else "fixture has no pose targets",
            ),
            ProductOutcome(
                product="ball-join",
                status="not_applicable",
                publication=None,
                reason="fixture has no joined target",
            ),
        )
        return write_document(
            self.layout,
            self.layout.path("campaigns", name, "completion.json"),
            CampaignCompletion(
                plan=plan_ref, tasks=(TaskProducts(task="detector", products=outcomes),)
            ),
        )

    def register(self, **kwargs: Any) -> ArtifactRef:
        result = self.learning.register(self.campaign(**kwargs), self.adapter)
        assert result.intent is not None
        return result.intent

    def train_and_evaluate(
        self, intent: ArtifactRef, *, finish_selection: bool = True
    ) -> Any:
        outcome = self.learning.tick(intent, self.adapter, self.runtime)
        assert outcome.state == "training_pending" and outcome.launch is not None
        self.executor.finish(outcome.launch, self.adapter)
        outcome = self.learning.tick(intent, self.adapter, self.runtime)
        assert outcome.state == "evaluation_pending" and outcome.launch is not None
        self.executor.finish(outcome.launch, self.adapter)
        return (
            self.learning.tick(intent, self.adapter, self.runtime)
            if finish_selection
            else outcome
        )


@pytest.fixture
def env(tmp_path: Path) -> Environment:
    return Environment(tmp_path / "annotation")


def retained(env: Environment) -> set[ArtifactRef]:
    return {
        ref
        for record in env.learning.storage.catalog.snapshot().protections
        for ref in record.references
    }


def test_registered_learning_protects_nested_input_before_any_queue_job(
    env: Environment,
) -> None:
    dependency = write_asset(
        env.layout, env.layout.path("models", "nested-shard"), b"teachers"
    )
    env.adapter.dependencies = (dependency,)
    catalog = env.learning.storage.catalog
    catalog.register(
        ManagedArtifact(
            reference=dependency,
            owner="fixture",
            stage="train",
            kind="shard",
            regenerable=False,
            retention=Retention(mode="expire", expires_at_ns=0, reason="fixture TTL"),
        )
    )

    def eligible() -> bool:
        observed = inventory(
            env.layout,
            (InventoryScope(role="annotation", path="models"),),
            max_entries=100,
        )
        result: bool = next(
            row.eligible
            for row in plan_gc(catalog, observed).decisions
            if row.path == catalog.key(dependency)
        )
        return result

    assert eligible()
    intent = env.register()
    assert not eligible() and env.executor.accepted == []
    assert {dependency, env.val.manifest, env.test.manifest, intent} <= set(
        env.learning.storage.inputs(intent)
    )
    env.executor.crash_after_accept = True
    with pytest.raises(Crash):
        env.learning.tick(intent, env.adapter, env.runtime)
    assert not eligible()
    assert (
        Learning(env.layout, executor=Executor(env.layout))
        .tick(intent, env.adapter, env.runtime)
        .state
        == "training_pending"
    )
    assert not eligible() and len(env.executor.accepted) == 1


def test_learning_input_failure_rolls_back_registration(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = env.campaign()
    before = env.learning.storage.catalog.snapshot().digest
    original = env.learning.storage.register

    def fail(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("input protection commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(env.learning.storage, "register", fail)
        with pytest.raises(RuntimeError, match="input protection commit failure"):
            env.learning.register(completion, env.adapter)
    with env.learning.registry.transaction() as db:
        assert (
            db.execute("SELECT count(*) FROM annotation_learning_intents").fetchone()[0]
            == 0
        )
        assert (
            db.execute("SELECT count(*) FROM learning_input_protections").fetchone()[0]
            == 0
        )
    assert env.learning.storage.catalog.snapshot().digest == before
    assert env.learning.register(completion, env.adapter).status == "ready"


def test_learning_input_set_cannot_change_on_registration_replay(
    env: Environment,
) -> None:
    completion = env.campaign()
    registered = env.learning.register(completion, env.adapter)
    before = env.learning.storage.catalog.snapshot().digest
    env.adapter.dependencies = (
        write_asset(
            env.layout, env.layout.path("models", "different-shard"), b"different"
        ),
    )
    with pytest.raises(ValueError, match="protection changed"):
        env.learning.register(completion, env.adapter)
    assert env.learning.storage.catalog.snapshot().digest == before
    assert registered.intent is not None
    assert (
        env.learning.tick(registered.intent, env.adapter, env.runtime).state
        == "training_pending"
    )


def test_released_learning_inputs_block_dispatch_instead_of_silent_repin(
    env: Environment,
) -> None:
    intent = env.register()
    catalog = env.learning.storage.catalog
    record = next(
        row
        for row in catalog.snapshot().protections
        if row.protection_id == "learning-inputs-v1-" + intent.digest
    )
    catalog.release(
        record.protection_id, expected_digest=record.sha256, reason="fixture revocation"
    )
    with pytest.raises(ValueError, match="released"):
        env.learning.tick(intent, env.adapter, env.runtime)
    assert env.executor.accepted == []


def test_completed_result_pin_recovers_commit_failure_before_evaluation(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent = env.register()
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.launch is not None
    result_ref = env.executor.finish(state.launch, env.adapter)
    training = read_document(env.layout, result_ref, TrainingReceipt)
    dependency = write_asset(
        env.layout,
        env.layout.path("models", "training-evidence"),
        b"validated evidence",
    )
    env.adapter.result_dependencies = (dependency,)
    original = StorageCatalog.protect

    def fail(self: StorageCatalog, record: Any, **kwargs: Any) -> None:
        original(self, record, **kwargs)
        if record.protection_id.startswith("learning-result-v1-"):
            raise RuntimeError("result protection commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(StorageCatalog, "protect", fail)
        with pytest.raises(RuntimeError, match="result protection commit failure"):
            env.learning.tick(intent, env.adapter, env.runtime)
    with env.learning.registry.transaction() as db:
        assert (
            db.execute("SELECT count(*) FROM learning_result_protections").fetchone()[0]
            == 0
        )
    assert len(env.executor.accepted) == 1 and dependency not in retained(env)
    evaluated = env.learning.tick(intent, env.adapter, env.runtime)
    assert evaluated.state == "evaluation_pending" and evaluated.launch is not None
    assert len(env.executor.accepted) == 2
    launch = read_document(env.layout, evaluated.launch, StageLaunch)
    assert {
        dependency,
        training.provenance,
        training.candidate.training_weights,
    } <= set(launch.dependencies)
    assert dependency in retained(env)


@pytest.mark.parametrize("accept", [True, False])
def test_model_decisions_protect_both_compared_weights_and_fixed_inputs(
    env: Environment,
    accept: bool,
) -> None:
    from src.tasks.base.annotation.plan import references
    from tests.support.annotation_storage import assert_expired_but_protected

    env.adapter.accept = accept
    intent_ref = env.register()
    outcome = env.train_and_evaluate(intent_ref)
    assert outcome.state == ("adopted" if accept else "rejected")
    selection = read_document(env.layout, outcome.selection, SelectionRecord)
    intent = read_document(env.layout, intent_ref, LearningIntent)
    verdict = read_document(env.layout, selection.verdict, SelectionVerdict)
    assert_expired_but_protected(
        env.layout,
        (
            outcome.selection,
            selection.verdict,
            verdict.evaluation,
            intent.recipe,
            intent.fixed_evaluation,
            *(references(selection.expected.model)),
            *(references(selection.candidate)),
        ),
        "model",
    )


def test_model_bootstrap_protection_failure_cannot_publish_a_pointer(
    env: Environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.base.annotation.storage.catalog import StorageCatalog

    registry = env.learning.models
    candidate = env.model.model_copy(update={"task": "another"})
    original = registry.catalog.protect
    before = StorageCatalog(env.layout).snapshot().digest

    def crash(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("after model protection")

    with monkeypatch.context() as patch:
        patch.setattr(registry.catalog, "protect", crash)
        with pytest.raises(RuntimeError, match="after model protection"):
            registry.bootstrap(candidate, env.common)
    assert StorageCatalog(env.layout).snapshot().digest == before
    with pytest.raises(ValueError, match="no explicitly adopted"):
        registry.current("another")
    assert registry.bootstrap(candidate, env.common).model == candidate


@pytest.mark.parametrize(
    ("options", "state"),
    [
        ({"incomplete": True}, "annotation_incomplete"),
        ({"added": False}, "no_additional_teachers"),
        ({"configured": False}, "not_configured"),
    ],
)
def test_missing_products_and_teachers_do_not_enqueue(
    env: Environment, options: dict, state: str
) -> None:
    completion = env.campaign(**options)
    for _ in range(2):
        result = env.learning.register(completion, env.adapter)
        assert result.status == state and result.intent is None
    assert env.executor.accepted == []
    assert env.learning.models.current("detector").model == env.model


@pytest.mark.parametrize("identity", ["val", "test"])
def test_source_and_clip_leakage_is_rejected_before_registration(
    env: Environment, identity: str
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        env.learning.register(env.campaign(overlap=identity), env.adapter)
    assert env.executor.accepted == []


def test_completion_must_cover_all_products_once(env: Environment) -> None:
    original = env.campaign()
    value = read_document(env.layout, original, CampaignCompletion)
    invalid = value.model_copy(
        update={
            "tasks": (
                value.tasks[0].model_copy(
                    update={"products": value.tasks[0].products[:1]}
                ),
            )
        }
    )
    ref = write_document(
        env.layout, env.layout.path("campaigns", "incomplete.json"), invalid
    )
    with pytest.raises(ValueError, match="exactly once"):
        env.learning.register(ref, env.adapter)


def test_registration_replay_and_two_generations_preserve_campaign_pin(
    env: Environment,
) -> None:
    completion = env.campaign()
    with ThreadPoolExecutor(max_workers=4) as pool:
        registrations = list(
            pool.map(lambda _: env.learning.register(completion, env.adapter), range(8))
        )
    assert len(set(r.intent for r in registrations)) == 1
    intent = registrations[0].intent
    assert intent is not None
    result = env.train_and_evaluate(intent)
    assert result.state == "adopted" and len(env.executor.accepted) == 2
    adopted = env.learning.models.current("detector")
    assert adopted.generation == 2
    replay = Learning(env.layout, executor=env.executor).tick(
        intent, env.adapter, env.runtime
    )
    assert (
        replay == result
        and len(env.executor.accepted) == 2
        and env.adapter.selections == 1
    )
    old = read_document(env.layout, completion, CampaignCompletion)
    plan = read_document(env.layout, old.plan, CampaignPlan)
    assert plan.tasks[0].model == env.model
    second = env.register(name="cycle-two", model=adopted.model)
    assert env.train_and_evaluate(second).state == "adopted"
    assert env.learning.models.current("detector").generation == 3
    assert len(env.executor.accepted) == 4


@pytest.mark.parametrize(
    "mode", ["reject", "no_policy", "partial_metrics", "test_evaluation"]
)
def test_unqualified_results_never_change_current_model(
    env: Environment, mode: str
) -> None:
    env.adapter.accept = mode != "reject"
    env.adapter.complete_metrics = mode != "partial_metrics"
    env.adapter.use_test = mode == "test_evaluation"
    intent = env.register(policy=mode != "no_policy")
    result = env.train_and_evaluate(intent)
    expected = {
        "reject": "rejected",
        "no_policy": "not_configured",
        "partial_metrics": "failed",
        "test_evaluation": "failed",
    }[mode]
    assert result.state == expected
    assert env.learning.models.current("detector").model == env.model
    assert env.learning.tick(intent, env.adapter, env.runtime) == result
    assert len(env.executor.accepted) == 2


def test_queue_acceptance_response_loss_does_not_duplicate_job(
    env: Environment,
) -> None:
    intent = env.register()
    env.executor.crash_after_accept = True
    with pytest.raises(Crash):
        env.learning.tick(intent, env.adapter, env.runtime)
    restarted = Learning(env.layout, executor=Executor(env.layout))
    assert restarted.tick(intent, env.adapter, env.runtime).state == "training_pending"
    assert len(env.executor.accepted) == 1


def test_failed_training_with_checkpoint_requires_explicit_same_intent_retry(
    env: Environment,
) -> None:
    intent = env.register()
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.launch is not None
    result_ref = env.executor.finish(state.launch, env.adapter, failed=True)
    training = read_document(env.layout, result_ref, TrainingReceipt)
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.state == "failed"
    assert env.learning.tick(intent, env.adapter, env.runtime).state == "failed"
    with pytest.raises(ValueError, match="another training intent"):
        env.learning.retry(
            intent, env.adapter, "train", reason="recover", resume_checkpoint=env.common
        )
    retry = env.learning.retry(
        intent,
        env.adapter,
        "train",
        reason="recover after interrupted checkpoint",
        resume_checkpoint=training.provenance,
    )
    assert read_document(env.layout, retry, StageInput).attempt == 2
    assert {retry, training.provenance} <= retained(env)
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.state == "training_pending" and len(env.executor.accepted) == 2
    assert env.learning.models.current("detector").generation == 1


@pytest.mark.parametrize("boundary", [None, "retry", "protection"])
def test_retry_retains_checkpoint_dependencies_before_dispatch(
    env: Environment, monkeypatch: pytest.MonkeyPatch, boundary: str | None
) -> None:
    from src.tasks.base.annotation.learning import service
    from tests.support.annotation_storage import assert_expired_but_protected

    intent = env.register()
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.launch is not None
    result_ref = env.executor.finish(state.launch, env.adapter, failed=True)
    training = read_document(env.layout, result_ref, TrainingReceipt)
    assert env.learning.tick(intent, env.adapter, env.runtime).state == "failed"
    dependency = write_asset(
        env.layout, env.layout.path("models", "initial-config.json"), b"run config"
    )
    assert dependency not in retained(env)
    env.adapter.resume_dependencies = (dependency,)
    real_write = service.write_document
    persisted_request: ArtifactRef | None = None

    def crash(layout: AnnotationLayout, path: Path, document: Any) -> ArtifactRef:
        reference = real_write(layout, path, document)
        if path.name == "retry.json":
            raise Crash("retry dependencies persisted before request")
        return reference

    def fail_protection(
        reference: ArtifactRef, request: StageInput
    ) -> tuple[ArtifactRef, ...]:
        nonlocal persisted_request
        assert read_document(env.layout, reference, StageInput) == request
        persisted_request = reference
        raise sqlite3.OperationalError("request persisted before protection")

    if boundary is not None:
        with monkeypatch.context() as patch:
            if boundary == "retry":
                patch.setattr(service, "write_document", crash)
            else:
                patch.setattr(env.learning.storage, "request", fail_protection)
            error = Crash if boundary == "retry" else sqlite3.OperationalError
            with pytest.raises(error, match="persisted"):
                env.learning.retry(
                    intent,
                    env.adapter,
                    "train",
                    reason="resume failed run",
                    resume_checkpoint=training.provenance,
                )
        # Recovery must use the dependencies already frozen in retry.json.
        env.adapter.resume_dependencies = ()
        assert dependency not in retained(env)

    restarted = Learning(env.layout, executor=env.executor)
    retry = restarted.retry(
        intent,
        env.adapter,
        "train",
        reason="resume failed run",
        resume_checkpoint=training.provenance,
    )
    assert len(env.executor.accepted) == 1
    assert {retry, training.provenance, dependency} <= retained(env)
    if boundary == "protection":
        assert retry == persisted_request
        assert (
            restarted.retry(
                intent,
                env.adapter,
                "train",
                reason="resume failed run",
                resume_checkpoint=training.provenance,
            )
            == retry
        )
        assert not (env.layout.artifact_path(retry).parent.parent / "3").exists()
    assert_expired_but_protected(
        env.layout, (training.provenance, dependency), "training"
    )
    state = restarted.tick(intent, env.adapter, env.runtime)
    assert state.state == "training_pending" and state.launch is not None
    launch = read_document(env.layout, state.launch, StageLaunch)
    assert {training.provenance, dependency} <= set(launch.dependencies)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("reason", "same explicit inputs"),
        ("checkpoint", "same explicit inputs"),
        ("request", "immutable document already has different content"),
        ("previous", "immediately preceding failed stage"),
        ("failure", "immediately preceding failed stage"),
        ("missing_failure", "immediately preceding failed stage"),
        ("completed", "immediately preceding failed stage"),
    ],
)
def test_retry_replay_rejects_mismatched_durable_state(
    env: Environment,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    message: str,
) -> None:
    intent = env.register()
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.launch is not None
    env.executor.finish(state.launch, env.adapter, failed=True)
    assert env.learning.tick(intent, env.adapter, env.runtime).state == "failed"
    saved = env.learning.retry(intent, env.adapter, "train", reason="explicit retry")
    request_path = env.layout.artifact_path(saved)
    retry_path = request_path.parent / "retry.json"
    retry = read_document(
        env.layout, annotation_reference(env.layout, retry_path), Retry
    )
    previous = env.layout.artifact_path(retry.previous).parent
    if mismatch == "request":
        changed_request = read_document(env.layout, saved, StageInput).model_copy(
            update={"attempt": 3}
        )
        request_path.write_bytes(changed_request.canonical_bytes() + b"\n")
    elif mismatch == "previous":
        copied = write_document(
            env.layout,
            env.layout.path("models", "copied-request.json"),
            read_document(env.layout, retry.previous, StageInput),
        )
        retry_path.write_bytes(
            retry.model_copy(update={"previous": copied}).canonical_bytes() + b"\n"
        )
    elif mismatch == "failure":
        failure_path = previous / "failure.json"
        failure = read_document(
            env.layout, annotation_reference(env.layout, failure_path), StageFailure
        )
        failure_path.write_bytes(
            failure.model_copy(update={"request": saved}).canonical_bytes() + b"\n"
        )
    elif mismatch == "missing_failure":
        (previous / "failure.json").unlink()
    elif mismatch == "completed":
        write_document(
            env.layout,
            previous / "completed.json",
            CompletedStage(
                request=retry.previous, result=env.common, execution=env.common
            ),
        )

    def unexpected_protection(*args: Any, **kwargs: Any) -> None:
        pytest.fail("mismatched retry must be rejected before protection")

    monkeypatch.setattr(env.learning.storage, "request", unexpected_protection)
    before = (request_path.read_bytes(), retry_path.read_bytes())
    with pytest.raises(ValueError, match=message):
        env.learning.retry(
            intent,
            env.adapter,
            "train",
            reason="changed reason" if mismatch == "reason" else "explicit retry",
            resume_checkpoint=env.common if mismatch == "checkpoint" else None,
        )
    assert (request_path.read_bytes(), retry_path.read_bytes()) == before
    assert not (request_path.parent.parent / "3").exists()
    assert len(env.executor.accepted) == 1


@pytest.mark.parametrize("after_write", [False, True])
def test_model_commit_file_gap_recovers_without_second_promotion(
    env: Environment, monkeypatch: pytest.MonkeyPatch, after_write: bool
) -> None:
    from src.tasks.base.annotation.learning import models
    from tests.support.annotation_storage import assert_expired_but_protected

    intent = env.register()
    env.train_and_evaluate(intent, finish_selection=False)
    real_write = models.write_document

    def interrupted(*args: Any, **kwargs: Any) -> ArtifactRef:
        if after_write:
            real_write(*args, **kwargs)
        raise Crash("pointer committed, receipt file not written")

    monkeypatch.setattr(models, "write_document", interrupted)
    with pytest.raises(Crash):
        env.learning.tick(intent, env.adapter, env.runtime)
    assert env.learning.models.current("detector").generation == 2
    assert_expired_but_protected(
        env.layout, selection_dependencies(env, intent), "model"
    )
    monkeypatch.setattr(models, "write_document", real_write)
    result = Learning(env.layout, executor=env.executor).tick(
        intent, env.adapter, env.runtime
    )
    assert (
        result.state == "adopted"
        and env.learning.models.current("detector").generation == 2
    )
    assert len(env.executor.accepted) == 2 and env.adapter.selections == 1


def test_parallel_candidates_cannot_overwrite_newer_adopted_model(
    env: Environment,
) -> None:
    from tests.support.annotation_storage import assert_expired_but_protected

    first, second = env.register(name="parallel-one"), env.register(name="parallel-two")
    env.train_and_evaluate(first, finish_selection=False)
    env.train_and_evaluate(second, finish_selection=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda ref: Learning(env.layout, executor=env.executor).tick(
                    ref, env.adapter, env.runtime
                ),
                (first, second),
            )
        )
    assert {r.state for r in results} == {"adopted", "reevaluation_required"}
    assert env.learning.models.current("detector").generation == 2
    for result in results:
        assert_expired_but_protected(
            env.layout, selection_dependencies(env, result.intent), "model"
        )
        assert result.selection is not None
        selection = read_document(env.layout, result.selection, SelectionRecord)
        assert selection.expected.model == env.model
        if result.state == "reevaluation_required":
            assert selection.observed.generation == 2 and selection.adopted is None


def test_selection_protection_failure_rolls_back_model_decision(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.annotation_storage import assert_expired_but_protected

    intent = env.register()
    env.train_and_evaluate(intent, finish_selection=False)
    registry = env.learning.models
    before: str | None = None
    original = registry.catalog.protect
    original_select = registry.select

    def select(verdict_ref: ArtifactRef, adapter: Adapter) -> ArtifactRef:
        nonlocal before
        # Completed stage results are protected before the model decision begins.
        before = registry.catalog.snapshot().digest
        selected: ArtifactRef = original_select(verdict_ref, adapter)
        return selected

    def crash(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("after selection protection")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "select", select)
        patch.setattr(registry.catalog, "protect", crash)
        with pytest.raises(RuntimeError, match="after selection protection"):
            env.learning.tick(intent, env.adapter, env.runtime)
    assert registry.current("detector").generation == 1
    assert before is not None and registry.catalog.snapshot().digest == before
    with registry.registry.transaction() as db:
        assert (
            db.execute("SELECT * FROM annotation_model_selections").fetchone() is None
        )
        assert db.execute("SELECT count(*) FROM annotation_models").fetchone()[0] == 1
    assert env.learning.tick(intent, env.adapter, env.runtime).state == "adopted"
    assert_expired_but_protected(
        env.layout, selection_dependencies(env, intent), "model"
    )


def test_registered_intent_and_runtime_cannot_be_silently_replaced(
    env: Environment,
) -> None:
    intent = env.register()
    env.learning.tick(intent, env.adapter, env.runtime)
    changed = env.runtime.model_copy(
        update={"train": env.runtime.train.model_copy(update={"timeout_seconds": 20.0})}
    )
    with pytest.raises(ValueError, match="immutable"):
        env.learning.tick(intent, env.adapter, changed)
    with pytest.raises(ValueError, match="registration already differs"):
        ModelRegistry(env.layout).bootstrap(
            env.model.model_copy(update={"version": "latest"}), env.common
        )


@pytest.mark.parametrize("collect_failure", [False, True])
@pytest.mark.parametrize("move_checkout", [False, True])
def test_code_update_after_learning_failure_preserves_history_and_retries(
    env: Environment,
    monkeypatch: pytest.MonkeyPatch,
    collect_failure: bool,
    move_checkout: bool,
) -> None:
    from src.tasks.base.annotation.learning import service
    from src.tasks.base.annotation.reproducibility import CodeRevision

    command = env.adapter.command

    def checkout_command(layout, request, result):
        return (
            str(service.PROJECT_ROOT / ".venv/bin/python"),
            *command(layout, request, result)[1:],
        )

    monkeypatch.setattr(env.adapter, "command", checkout_command)
    intent_ref = env.register()
    intent = read_document(env.layout, intent_ref, LearningIntent)
    completion = read_document(env.layout, intent.completion, CampaignCompletion)
    first = env.learning.tick(intent_ref, env.adapter, env.runtime)
    assert first.launch is not None
    env.executor.finish(first.launch, env.adapter, failed=True)
    if collect_failure:
        assert env.learning.tick(intent_ref, env.adapter, env.runtime).state == "failed"
    directory = env.layout.path("campaigns", "cycle-one", "learning", "detector")
    original_runtime = (directory / "runtime.json").read_bytes()
    original_launch = env.layout.artifact_path(first.launch).read_bytes()
    revision = write_document(
        env.layout,
        env.layout.path("campaigns", "cycle-one", "code-update.json"),
        CodeRevision(
            plan=completion.plan,
            scope=env.runtime.scope,
            previous_runtime=annotation_reference(
                env.layout, directory / "runtime.json"
            ),
            code_commit="c" * 40,
            code_diff_digest=hashlib.sha256(b"").hexdigest(),
            verification=env.common,
            reason="verified fix for the failed training command",
        ),
    )
    runtime = env.runtime.model_copy(update={"code_revision": revision})
    if move_checkout:
        monkeypatch.setattr(
            service, "PROJECT_ROOT", env.layout.repository / "fixed-code"
        )
    learning = Learning(env.layout, executor=env.executor)
    changed_cost = runtime.model_copy(
        update={
            "train": runtime.train.model_copy(update={"storage_bytes": 2}),
        }
    )
    with pytest.raises(ValueError, match="immutable"):
        learning.tick(intent_ref, env.adapter, changed_cost)
    wrong_scope = write_document(
        env.layout,
        env.layout.path("campaigns", "cycle-one", "wrong-scope-update.json"),
        read_document(env.layout, revision, CodeRevision).model_copy(
            update={"scope": env.recipe}
        ),
    )
    with pytest.raises(ValueError, match="another plan or scope"):
        learning.tick(
            intent_ref,
            env.adapter,
            runtime.model_copy(update={"code_revision": wrong_scope}),
        )
    assert learning.tick(intent_ref, env.adapter, runtime).state == "failed"
    assert len(env.executor.accepted) == 1
    retry = learning.retry(intent_ref, env.adapter, "train", reason="retry fixed code")
    assert read_document(env.layout, retry, StageInput).attempt == 2
    state = learning.tick(intent_ref, env.adapter, runtime)
    assert state.launch is not None and state.launch != first.launch
    assert (
        read_document(env.layout, state.launch, StageLaunch).code_revision == revision
    )
    assert learning.tick(intent_ref, env.adapter, runtime).launch == state.launch
    env.executor.finish(state.launch, env.adapter)
    state = learning.tick(intent_ref, env.adapter, runtime)
    assert state.state == "evaluation_pending" and state.launch is not None
    assert (
        read_document(env.layout, state.launch, StageLaunch).code_revision == revision
    )
    env.executor.finish(state.launch, env.adapter)
    assert learning.tick(intent_ref, env.adapter, runtime).state == "adopted"
    assert len(env.executor.accepted) == 3
    assert (directory / "runtime.json").read_bytes() == original_runtime
    assert env.layout.artifact_path(first.launch).read_bytes() == original_launch
    saved = annotation_reference(
        env.layout, directory / "code-updates" / revision.digest / "runtime.json"
    )
    assert read_document(env.layout, saved, LearningRuntime) == runtime


def test_real_stage_runner_binds_scope_and_one_queue_registration(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_ref = env.register()
    intent = read_document(env.layout, intent_ref, LearningIntent)
    completion = read_document(env.layout, intent.completion, CampaignCompletion)
    plan = read_document(env.layout, completion.plan, CampaignPlan)
    manager = ResourceManager.create(
        env.layout,
        scope_id="fixture-learning",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(env.layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0.0,
        max_pending_per_stage=5,
    )
    script = env.layout.repository / "queue.sh"
    script.write_text("# fixture protocol --idempotency-key\n")
    queue = QueueConfiguration(
        script=str(script),
        script_digest=file_digest(script),
        directory=str(env.layout.repository / ".training_queue"),
        session="fixture",
        issue=1135,
    )
    runtime = env.runtime.model_copy(
        update={"scope": manager.reference, "queue": queue}
    )
    calls: list[list[str]] = []
    # Patch only the queue CLI call; other subprocess tools retain their behavior.
    from src.tasks.base.annotation.execution import queue as queue_module

    original_run = subprocess.run

    def register(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        if command[:2] != ["bash", str(script)]:
            return original_run(command, **kwargs)
        calls.append(command)
        assert command[command.index("--resource") + 1] == "all"
        key = command[command.index("--idempotency-key") + 1]
        receipt = (
            Path(queue.directory)
            / "control/registrations"
            / (hashlib.sha256(key.encode()).hexdigest() + ".receipt")
        )
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("a" * 64 + " fixture.job\n")
        job = Path(queue.directory) / "jobs/fixture.job"
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_text("fixture; not executed")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(queue_module.subprocess, "run", register)
    learning = Learning(env.layout)
    first = learning.tick(intent_ref, env.adapter, runtime)
    second = Learning(env.layout).tick(intent_ref, env.adapter, runtime)
    assert (
        first.state == second.state == "training_pending"
        and first.launch == second.launch
    )
    assert len(calls) == 1
    sample = manager.sample()
    assert sample.gpu_charged_seconds == 0 and sample.pending_by_stage == {"train": 1}


def test_direct_selection_still_requires_the_frozen_policy(env: Environment) -> None:
    intent = env.register(policy=False)
    outcome = env.train_and_evaluate(intent)
    assert outcome.evaluation is not None
    verdict = write_document(
        env.layout,
        env.layout.path("models", "unconfigured-verdict.json"),
        SelectionVerdict(
            evaluation=outcome.evaluation,
            policy=env.common,
            status="passed",
            reason="must not bypass missing policy",
        ),
    )
    with pytest.raises(ValueError, match="frozen adoption policy"):
        env.learning.models.select(verdict, env.adapter)
    assert env.learning.models.current("detector").generation == 1


def test_evaluation_retry_reuses_completed_training(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent = env.register()
    env.adapter.complete_metrics = False
    assert env.train_and_evaluate(intent).state == "failed"
    with pytest.raises(ValueError, match="evaluation has started"):
        env.learning.retry(
            intent, env.adapter, "train", reason="must not replace trained weights"
        )
    env.adapter.complete_metrics = True
    saved: ArtifactRef | None = None

    def fail_protection(
        reference: ArtifactRef, request: StageInput
    ) -> tuple[ArtifactRef, ...]:
        nonlocal saved
        assert read_document(env.layout, reference, StageInput) == request
        saved = reference
        raise sqlite3.OperationalError("request persisted before protection")

    with monkeypatch.context() as patch:
        patch.setattr(env.learning.storage, "request", fail_protection)
        with pytest.raises(sqlite3.OperationalError, match="before protection"):
            env.learning.retry(
                intent, env.adapter, "evaluate", reason="repair evaluation runtime"
            )
    restored = env.learning.retry(
        intent, env.adapter, "evaluate", reason="repair evaluation runtime"
    )
    assert restored == saved
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.state == "evaluation_pending" and state.launch is not None
    env.executor.finish(state.launch, env.adapter)
    assert env.learning.tick(intent, env.adapter, env.runtime).state == "adopted"
    assert len(env.executor.accepted) == 3


@pytest.mark.parametrize("boundary", ["intent", "evaluation", "retry"])
def test_durable_file_gaps_resume_without_new_job(
    env: Environment, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    from src.tasks.base.annotation.learning import service

    real_write = service.write_document

    def interrupted(layout: AnnotationLayout, path: Path, document: Any) -> ArtifactRef:
        ref = real_write(layout, path, document)
        match = (
            (boundary == "intent" and path.name == "intent.json")
            or (
                boundary == "evaluation"
                and path.name == "completed.json"
                and path.parent.parent.name == "evaluate"
            )
            or (boundary == "retry" and path.name == "retry.json")
        )
        if match:
            raise Crash("durable write before parent acknowledgment")
        return ref

    completion = env.campaign()
    if boundary == "intent":
        monkeypatch.setattr(service, "write_document", interrupted)
        with pytest.raises(Crash):
            env.learning.register(completion, env.adapter)
        monkeypatch.setattr(service, "write_document", real_write)
        assert env.learning.register(completion, env.adapter).status == "ready"
        assert not env.executor.accepted
        return
    registered = env.learning.register(completion, env.adapter)
    assert registered.intent is not None
    intent = registered.intent
    if boundary == "evaluation":
        env.train_and_evaluate(intent, finish_selection=False)
        monkeypatch.setattr(service, "write_document", interrupted)
        with pytest.raises(Crash):
            env.learning.tick(intent, env.adapter, env.runtime)
        monkeypatch.setattr(service, "write_document", real_write)
        assert env.learning.tick(intent, env.adapter, env.runtime).state == "adopted"
        assert len(env.executor.accepted) == 2
    else:
        state = env.learning.tick(intent, env.adapter, env.runtime)
        assert state.launch is not None
        env.executor.finish(state.launch, env.adapter, failed=True)
        assert env.learning.tick(intent, env.adapter, env.runtime).state == "failed"
        monkeypatch.setattr(service, "write_document", interrupted)
        with pytest.raises(Crash):
            env.learning.retry(intent, env.adapter, "train", reason="explicit retry")
        monkeypatch.setattr(service, "write_document", real_write)
        restored = env.learning.retry(
            intent, env.adapter, "train", reason="explicit retry"
        )
        assert read_document(env.layout, restored, StageInput).attempt == 2
        assert (
            env.learning.tick(intent, env.adapter, env.runtime).state
            == "training_pending"
        )
        assert len(env.executor.accepted) == 2


@pytest.mark.parametrize("stage", ["train", "evaluate"])
@pytest.mark.parametrize("error", [Crash, OSError])
def test_retry_recovers_directory_created_before_retry_publication(
    env: Environment,
    monkeypatch: pytest.MonkeyPatch,
    stage: Literal["train", "evaluate"],
    error: type[BaseException],
) -> None:
    from src.tasks.base.annotation import artifacts

    intent = env.register()
    if stage == "train":
        state = env.learning.tick(intent, env.adapter, env.runtime)
        assert state.launch is not None
        env.executor.finish(state.launch, env.adapter, failed=True)
        assert env.learning.tick(intent, env.adapter, env.runtime).state == "failed"
    else:
        env.adapter.complete_metrics = False
        assert env.train_and_evaluate(intent).state == "failed"
        env.adapter.complete_metrics = True
    accepted = len(env.executor.accepted)
    real_replace = artifacts.os.replace
    directory = env.layout.artifact_path(intent).parent / stage / "2"

    def interrupted(source: Path, target: Path) -> None:
        if target == directory / "retry.json":
            assert directory.is_dir()
            raise error("interrupted before retry publication")
        real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(artifacts.os, "replace", interrupted)
        with pytest.raises(error, match="before retry publication"):
            env.learning.retry(intent, env.adapter, stage, reason="explicit retry")
    assert directory.is_dir()
    assert not (directory / "retry.json").exists()
    assert not (directory / "request.json").exists()
    executor = Executor(env.layout)
    restarted = Learning(env.layout, executor=executor)
    with pytest.raises(ValueError, match="no durable stage request"):
        restarted.tick(intent, env.adapter, env.runtime)
    assert len(env.executor.accepted) == accepted
    restored = restarted.retry(intent, env.adapter, stage, reason="explicit retry")
    request = read_document(env.layout, restored, StageInput)
    assert request.attempt == 2 and request.stage == stage
    assert env.layout.artifact_path(restored) == directory / "request.json"
    outcome = restarted.tick(intent, env.adapter, env.runtime)
    assert outcome.state == (
        "training_pending" if stage == "train" else "evaluation_pending"
    )
    assert len(executor.accepted) == 1
    assert restarted.tick(intent, env.adapter, env.runtime) == outcome
    assert len(executor.accepted) == 1


def selection_dependencies(
    env: Environment, intent_ref: ArtifactRef
) -> tuple[ArtifactRef, ...]:
    """Expected fixture files needed to revalidate a completed model decision."""
    intent = read_document(env.layout, intent_ref, LearningIntent)
    completion = read_document(env.layout, intent.completion, CampaignCompletion)
    publication = read_document(env.layout, intent.publication, PublicationReceipt)
    refs = [
        intent_ref,
        intent.completion,
        completion.plan,
        intent.publication,
        intent.recipe,
        intent.fixed_evaluation,
        intent.train.manifest,
        env.val.manifest,
        env.test.manifest,
        *publication.adoptions,
        *env.adapter.dependencies,
    ]
    directory = env.layout.artifact_path(intent_ref).parent
    for stage in ("train", "evaluate"):
        completed_ref = annotation_reference(
            env.layout, directory / stage / "1" / "completed.json"
        )
        completed = read_document(env.layout, completed_ref, CompletedStage)
        execution = read_document(env.layout, completed.execution, StageExit)
        refs.extend(
            (
                completed_ref,
                completed.request,
                completed.result,
                completed.execution,
                execution.launch,
                execution.stdout,
                execution.stderr,
            )
        )
    training = read_document(
        env.layout,
        annotation_reference(env.layout, directory / "train/1/result.json"),
        TrainingReceipt,
    )
    evaluation = read_document(
        env.layout,
        annotation_reference(env.layout, directory / "evaluate/1/result.json"),
        EvaluationReceipt,
    )
    refs.extend((training.provenance, evaluation.metrics))
    return tuple(refs)


@pytest.mark.parametrize("accept", [True, False])
def test_model_decisions_protect_complete_selection_evidence(
    env: Environment, accept: bool
) -> None:
    from src.tasks.base.annotation.plan import references
    from tests.support.annotation_storage import assert_expired_but_protected

    env.adapter.accept = accept
    intent_ref = env.register()
    outcome = env.train_and_evaluate(intent_ref)
    assert outcome.state == ("adopted" if accept else "rejected")
    assert outcome.selection is not None
    selection = read_document(env.layout, outcome.selection, SelectionRecord)
    assert_expired_but_protected(
        env.layout,
        (
            *selection_dependencies(env, intent_ref),
            outcome.selection,
            selection.verdict,
            *references(selection.expected.model),
            *references(selection.candidate),
        ),
        "model",
    )


def stale_failure(
    env: Environment, stage: Literal["train", "evaluate"]
) -> tuple[ArtifactRef, ArtifactRef, Path]:
    """Persist the false interruption produced by the previous queue observer."""
    intent = env.register()
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.launch is not None
    if stage == "evaluate":
        env.executor.finish(state.launch, env.adapter)
        state = env.learning.tick(intent, env.adapter, env.runtime)
        assert state.launch is not None
    launch = read_document(env.layout, state.launch, StageLaunch)
    assert launch.input is not None
    directory = env.layout.artifact_path(launch.input).parent
    write_document(
        env.layout,
        directory / "failure.json",
        StageFailure(
            request=launch.input,
            launch=state.launch,
            reason="queue registration has no unique lifecycle state",
        ),
    )
    return intent, state.launch, directory


@pytest.mark.parametrize("stage", ["train", "evaluate"])
def test_tick_recovers_success_after_stale_queue_failure(
    env: Environment, stage: Literal["train", "evaluate"]
) -> None:
    dependency = write_asset(
        env.layout, env.layout.path("models", "recovered-result-input"), b"evidence"
    )
    env.adapter.result_dependencies = (dependency,)
    intent, launch, directory = stale_failure(env, stage)
    failure = (directory / "failure.json").read_bytes()
    env.executor.finish(launch, env.adapter)
    outcome = Learning(env.layout, executor=env.executor).tick(
        intent, env.adapter, env.runtime
    )
    assert outcome.state == ("evaluation_pending" if stage == "train" else "adopted")
    assert (directory / "completed.json").is_file()
    assert (directory / "failure.json").read_bytes() == failure
    completed_ref = annotation_reference(env.layout, directory / "completed.json")
    assert dependency in env.learning.storage.completed_inputs(completed_ref)
    assert len(env.executor.accepted) == 2


@pytest.mark.parametrize("partial_retry", [False, True])
@pytest.mark.parametrize(
    "state",
    [
        "pending",
        "unregistered",
        "queued",
        "waiting",
        "running",
        "draining",
        "unfinished",
    ],
)
def test_retry_rechecks_old_job_before_creating_or_restoring_attempt(
    env: Environment,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    partial_retry: bool,
) -> None:
    intent, launch, directory = stale_failure(env, "train")
    if partial_retry:
        request = read_document(env.layout, launch, StageLaunch).input
        assert request is not None
        write_document(
            env.layout,
            directory.parent / "2/retry.json",
            Retry(
                previous=request,
                reason="recover",
                resume_checkpoint=None,
                resume_dependencies=(),
            ),
        )
    monkeypatch.setattr(env.executor, "status", lambda _ref: {"state": state})
    if not partial_retry:
        outcome = env.learning.tick(intent, env.adapter, env.runtime)
        assert outcome.state == (
            "training_pending"
            if state in {"pending", "unregistered", "queued", "waiting"}
            else "training"
        )
    with pytest.raises(ValueError, match="not terminated"):
        env.learning.retry(intent, env.adapter, "train", reason="recover")
    assert not (directory.parent / "2/request.json").exists()
    assert len(env.executor.accepted) == 1


@pytest.mark.parametrize("stage", ["train", "evaluate"])
@pytest.mark.parametrize("partial_retry", ["none", "directory", "receipt"])
def test_retry_collects_old_success_instead_of_duplicate_execution(
    env: Environment, stage: Literal["train", "evaluate"], partial_retry: str
) -> None:
    intent, launch, directory = stale_failure(env, stage)
    target = directory.parent / "2"
    if partial_retry != "none":
        target.mkdir()
    if partial_retry == "receipt":
        request = read_document(env.layout, launch, StageLaunch).input
        assert request is not None
        write_document(
            env.layout,
            target / "retry.json",
            Retry(
                previous=request,
                reason="recover",
                resume_checkpoint=None,
                resume_dependencies=(),
            ),
        )
    env.executor.finish(launch, env.adapter)
    accepted = len(env.executor.accepted)
    with pytest.raises(ValueError, match="completed successfully"):
        env.learning.retry(intent, env.adapter, stage, reason="recover")
    assert (directory / "completed.json").exists()
    assert not (target / "request.json").exists()
    assert len(env.executor.accepted) == accepted
    outcome = env.learning.tick(intent, env.adapter, env.runtime)
    assert outcome.state == ("evaluation_pending" if stage == "train" else "adopted")
    assert len(env.executor.accepted) == 2


def test_unconfirmed_interruption_does_not_become_durable_failure(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent = env.register()
    state = env.learning.tick(intent, env.adapter, env.runtime)
    assert state.launch is not None
    launch = read_document(env.layout, state.launch, StageLaunch)
    assert launch.input is not None
    directory = env.layout.artifact_path(launch.input).parent
    monkeypatch.setattr(env.executor, "status", lambda _ref: {"state": "interrupted"})
    with pytest.raises(ValueError, match="termination is unconfirmed"):
        env.learning.tick(intent, env.adapter, env.runtime)
    assert not (directory / "failure.json").exists()
    assert len(env.executor.accepted) == 1


def test_retry_rejects_unconfirmed_old_job(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, _launch, directory = stale_failure(env, "train")
    monkeypatch.setattr(env.executor, "status", lambda _ref: {"state": "interrupted"})
    with pytest.raises(ValueError, match="termination is unconfirmed"):
        env.learning.retry(intent, env.adapter, "train", reason="recover")
    assert not (directory.parent / "2").exists()
    assert len(env.executor.accepted) == 1


def test_missing_task_dependency_rolls_back_model_selection(env: Environment) -> None:
    intent = env.register()
    env.train_and_evaluate(intent, finish_selection=False)
    registry = env.learning.models
    before = registry.catalog.snapshot().digest
    dependency = env.adapter.dependencies[0]
    path = env.layout.artifact_path(dependency)
    payload = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="missing or size changed"):
        env.learning.tick(intent, env.adapter, env.runtime)
    assert registry.current("detector").generation == 1
    assert registry.catalog.snapshot().digest == before
    with registry.registry.transaction() as db:
        assert (
            db.execute("SELECT * FROM annotation_model_selections").fetchone() is None
        )
    write_asset(env.layout, path, payload)
    assert env.learning.tick(intent, env.adapter, env.runtime).state == "adopted"
