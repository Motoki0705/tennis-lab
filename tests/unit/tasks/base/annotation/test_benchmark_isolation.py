from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest

from src.tasks.ball_detection.annotation.review.contracts import BallEdits
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    PublishRequest,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
)
from src.tasks.base.annotation.benchmarks.capture import observe
from src.tasks.base.annotation.benchmarks.contracts import (
    ConditionSpec,
    Observation,
    WorkerBinding,
)
from src.tasks.base.annotation.benchmarks.inputs import (
    RegisteredInputs,
    register_inputs,
)
from src.tasks.base.annotation.collection.service import Collection
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ProductOutcome,
)
from src.tasks.base.annotation.editing import ReviewContext, Revision, RevisionStore
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    TaskProducts,
)
from src.tasks.base.annotation.learning.models import ModelRegistry
from src.tasks.base.annotation.learning.validation import check_completion
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    require_execution_layout,
    save_plan,
)
from src.tasks.base.annotation.predictions import Predictions
from src.tasks.base.annotation.resources import BudgetLimits, ResourceManager
from src.tasks.base.annotation.resources.contracts import GPUUsage
from src.tasks.base.annotation.storage.catalog import StorageCatalog
from src.tasks.base.annotation.workers.contracts import (
    PreparedWorker,
    WorkOutcome,
    WorkRequest,
)
from src.tasks.base.annotation.workers.pool import WorkerPool
from tests.support.tasks.ball_detection.annotation import Environment, ProposalFixture


class Setup:
    def __init__(self, root: Path) -> None:
        self.env = Environment(root)
        self.shared = self.env.layout
        self.manager = ResourceManager.create(
            self.shared,
            scope_id="all-conditions",
            limits=BudgetLimits.from_budget(self.env.plan.resources),
            roots=(self.shared.repository,),
            previously_allocated_bytes=17,
            previously_used_gpu_seconds=7.0,
            max_pending_per_stage=50,
        )

    def condition(
        self, number: int
    ) -> tuple[CampaignPlan, ArtifactRef, AnnotationLayout]:
        env = self.env
        plan: CampaignPlan = env.plan.model_copy(
            update={
                "campaign_id": f"condition-{number}",
                "phase": "benchmark",
                "resources": env.plan.resources.model_copy(
                    update={
                        "codex_workers": number,
                        "stages": tuple(
                            stage.model_copy(update={"parallel": number})
                            for stage in env.plan.resources.stages
                        ),
                    }
                ),
            }
        )
        local = self.shared.for_benchmark(plan.campaign_id)
        ref = env.save(
            ConditionSpec(
                benchmark_id="comparison",
                condition_id=plan.campaign_id,
                plan=annotation_reference(local, save_plan(local, plan)),
                scope=self.manager.reference,
                predictions=(env.context.prediction,),
                bindings=(
                    WorkerBinding(
                        stage="review",
                        adapter="ball_review",
                        adapter_version="v1",
                        task="ball",
                    ),
                ),
                target_workers=number,
                maximum_observations=10,
            )
        )
        return plan, ref, local


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path / "annotation")


def test_layout_changes_domain_state_but_retains_fixed_roots(setup: Setup) -> None:
    _, _, local = setup.condition(10)
    assert (
        local.root == setup.shared.root
        and local.resolver.roots == setup.shared.resolver.roots
    )
    assert (
        local.registry
        == setup.shared.root / "campaigns/condition-10/benchmark/registry.sqlite3"
    )
    assert (
        local.path("annotations", "ball", "clip")
        == setup.shared.root / "campaigns/condition-10/benchmark/annotations/ball/clip"
    )
    assert local.artifact_path(
        setup.env.context.prediction
    ) == setup.shared.artifact_path(setup.env.context.prediction)
    assert local.shared() == setup.shared
    with pytest.raises(ValueError, match="namespace"):
        local.for_benchmark("condition-25")
    with pytest.raises(ValueError):
        setup.shared.for_benchmark("../outside")
    with pytest.raises(ValueError, match="model"):
        local.path("models", "ball")


def test_two_conditions_reuse_exact_predictions_without_mutating_original(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = setup.env.context_ref
    original_plan_bytes = setup.shared.artifact_path(setup.env.plan_ref).read_bytes()
    monkeypatch.setattr(
        ProposalFixture,
        "produce",
        lambda *_args: pytest.fail("benchmark must never infer"),
    )
    contexts = []
    for number in (10, 25):
        plan, spec_ref, local = setup.condition(number)
        receipt = register_inputs(setup.shared, spec_ref, (ProposalFixture(),))
        assert register_inputs(setup.shared, spec_ref, (ProposalFixture(),)) == receipt
        provenance = read_document(local, receipt, RegisteredInputs)
        assert provenance.borrowed[0].receipt == setup.env.context.prediction
        assert provenance.borrowed[0].clip_id == setup.env.manifest.clip_id
        assert provenance.borrowed[0].source_plan == setup.env.plan_ref
        predictions = Predictions(local, plan)
        assert predictions.completed(ProposalFixture()) == (
            setup.env.context.prediction,
        )
        context = setup.env.context.model_copy(
            update={"plan": read_document(local, spec_ref, ConditionSpec).plan}
        )
        revision = RevisionStore(local).initialize(context, setup.env.adapter)
        contexts.append(read_document(local, revision, Revision).context)
        with pytest.raises(ValueError, match="without inference"):
            predictions.run(setup.env.manifest_ref, ProposalFixture())
    assert contexts[0] != contexts[1] and original not in contexts
    assert setup.env.store.current(original, setup.env.adapter)[0] == setup.env.initial
    assert (
        setup.shared.artifact_path(setup.env.plan_ref).read_bytes()
        == original_plan_bytes
    )
    with setup.env.store.registry.transaction() as db:
        assert (
            db.execute("SELECT state FROM annotation_tasks").fetchone()[0]
            == "reviewing"
        )
        assert db.execute("SELECT count(*) FROM campaigns").fetchone()[0] == 1
    protected = StorageCatalog(setup.shared).snapshot().protections
    assert any(setup.env.manifest.media in item.references for item in protected)


def test_mixed_phase_and_layout_are_rejected(setup: Setup) -> None:
    plan, _, local = setup.condition(10)
    with pytest.raises(ValueError, match="own campaign state"):
        require_execution_layout(setup.shared, plan)
    with pytest.raises(ValueError, match="normal campaigns"):
        require_execution_layout(local, setup.env.plan)
    with pytest.raises(ValueError, match="own campaign state"):
        Predictions(setup.shared, plan)
    with pytest.raises(ValueError, match="cannot collect"):
        Collection(local, plan)


def test_new_budget_creation_is_rejected_and_existing_counters_are_shared(
    setup: Setup,
) -> None:
    plan, _, local = setup.condition(10)
    with setup.manager.registry.transaction() as db:
        db.execute("UPDATE resource_scopes SET stop_reason='disk reserve'")
    scoped = ResourceManager(local, setup.manager.reference)
    scoped.bind(plan)
    assert scoped.registry.layout == setup.shared
    assert scoped.scope == setup.manager.scope
    with scoped.registry.transaction() as db:
        assert (
            db.execute("SELECT stop_reason FROM resource_scopes").fetchone()[0]
            == "disk reserve"
        )
    with pytest.raises(ValueError, match="reuse"):
        ResourceManager.create(
            local,
            scope_id="reset",
            limits=scoped.scope.limits,
            roots=(local.repository,),
            previously_allocated_bytes=0,
            previously_used_gpu_seconds=0,
            max_pending_per_stage=50,
        )


def test_already_adopted_source_prediction_is_not_replayed(setup: Setup) -> None:
    _, ref, local = setup.condition(10)
    with setup.env.store.registry.transaction() as db:
        db.execute("UPDATE annotation_tasks SET state='adopted'")
    with pytest.raises(ValueError, match="unadopted"):
        register_inputs(setup.shared, ref, (ProposalFixture(),))
    assert not local.registry.exists()


def test_source_receipt_tampering_and_changed_condition_do_not_reset_state(
    setup: Setup,
) -> None:
    _, ref, local = setup.condition(10)
    register_inputs(setup.shared, ref, (ProposalFixture(),))
    changed = setup.env.save(
        read_document(local, ref, ConditionSpec).model_copy(
            update={"maximum_observations": 9}
        )
    )
    with pytest.raises(ValueError, match="immutable"):
        register_inputs(setup.shared, changed, (ProposalFixture(),))
    with setup.env.store.registry.transaction() as db:
        db.execute(
            "UPDATE annotation_predictions SET receipt=?",
            (setup.env.blob.canonical_bytes().decode(),),
        )
    with pytest.raises(ValueError, match="exact registered"):
        register_inputs(setup.shared, ref, (ProposalFixture(),))


def test_sealed_benchmark_adoption_cannot_publish_or_start_learning(
    setup: Setup,
) -> None:
    env = setup.env
    original_context, original_initial = env.context_ref, env.initial
    plan, ref, local = setup.condition(10)
    register_inputs(setup.shared, ref, (ProposalFixture(),))
    spec = read_document(local, ref, ConditionSpec)
    env.layout, env.plan, env.plan_ref = local, plan, spec.plan
    env.context = ReviewContext(
        plan=spec.plan,
        task="ball",
        manifest=env.manifest_ref,
        prediction=env.context.prediction,
    )
    env.store = RevisionStore(local)
    env.initial = env.store.initialize(env.context, env.adapter)
    env.context_ref = read_document(local, env.initial, Revision).context
    env.actor = env.new_actor("benchmark-review")
    image = env.page((1,))
    env.apply(
        BallEdits(
            operations=(env.replace(1, image, presence="absent"),), images=(image,)
        )
    )
    adoption = env.adapter.prepare_adoption(
        local,
        env.context,
        env.manifest,
        env.head(),
        (env.telemetry((image,)),),
        reason="synthetic isolated review",
    )
    env.store.seal(env.head(), adoption, env.adapter)
    assert env.store.current(env.context_ref, env.adapter)[1] == adoption
    assert (
        RevisionStore(setup.shared).current(original_context, env.adapter)[0]
        == original_initial
    )
    with pytest.raises(ValueError, match="cannot publish"):
        BallPublisher(local)
    # Carrying the immutable benchmark plan/adoption to the normal layout also fails.
    with pytest.raises(ValueError, match="production plan"):
        BallPublisher(setup.shared).publish(
            PublishRequest(plan=spec.plan, product="frames", adoptions=(adoption,))
        )
    completion = CampaignCompletion(
        plan=spec.plan,
        tasks=(
            TaskProducts(
                task="ball",
                products=(
                    ProductOutcome(
                        product="frames",
                        status="not_applicable",
                        publication=None,
                        reason="synthetic evidence",
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="production plan"):
        check_completion(setup.shared, completion)
    with pytest.raises(ValueError, match="model adoption"):
        ModelRegistry(local)
    assert not (setup.shared.repository / "data/datasets/ball_detection").exists()


class _Worker:
    name, version = "ball_review", "v1"

    def prepare(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> PreparedWorker:
        return PreparedWorker(
            prompt=write_asset(
                layout, directory / "prompt.md", b"synthetic isolated worker"
            )
        )

    def validate(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
        result: ArtifactRef,
        execution: Any,
        telemetry: Any,
    ) -> WorkOutcome:
        return WorkOutcome(status="complete", receipt=result)

    def telemetry_images(self, *args: Any) -> tuple:
        return ()

    def classify_tool(self, name: str | None, raw_input: str) -> str:
        return "fixture"


def test_same_subject_conditions_use_distinct_shared_resource_requests(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The full-suite pytest process is not part of this synthetic workload.
    # Supervisor/child ownership and their real PSS measurements remain active.
    monkeypatch.setattr(ResourceManager, "observe", lambda self, pid: None)
    monkeypatch.setattr(
        "src.tasks.base.annotation.resources.scheduler.gpu_usage",
        lambda _pids: GPUUsage(
            status="unavailable",
            owned_vram_bytes=None,
            device_vram_bytes=None,
            maximum_utilization_percent=None,
            reason="CPU fixture",
        ),
    )
    command = (
        sys.executable,
        "-c",
        'import pathlib; pathlib.Path(\'result.json\').write_text(\'{}\'); print(\'{"type":"thread.started","thread_id":"fixture"}\'); print(\'{"type":"turn.completed"}\')',
    )
    launches = []
    for number in (10, 25):
        plan, ref, local = setup.condition(number)
        register_inputs(setup.shared, ref, (ProposalFixture(),))
        request = WorkRequest(
            adapter="ball_review",
            adapter_version="v1",
            stage="review",
            subject_id=setup.env.manifest.clip_id,
            payload=setup.env.blob,
            inputs=(),
        )
        pool = WorkerPool(
            local,
            plan,
            (_Worker(),),
            test_command=command,
            resources=ResourceManager(local, setup.manager.reference),
            worker_pss_bytes=16 * 1024**2,
            worker_storage_bytes=1024**2,
            timeout_seconds=10,
        )
        assert pool.start(request) is not None
        until = time.monotonic() + 10
        while time.monotonic() < until:
            state = pool.reconcile(request)
            if state["state"] not in {"running", "exited"}:
                break
            time.sleep(0.02)
        assert state["state"] == "complete", state
        _, launch = pool._launch(pool.ledger.task(request.work_id))
        launches.append(launch)
        observation = read_document(
            setup.shared, observe(setup.shared, ref), Observation
        )
        assert observation.task_states == {"complete": 1}
        assert len(observation.attempts) == 1
        assert observation.attempts[0].launch == annotation_reference(
            local, Path(launch.directory) / "launch.json"
        )
    assert launches[0].work_id == launches[1].work_id
    assert (
        launches[0].resource_request.request_id
        != launches[1].resource_request.request_id
    )
    with setup.manager.registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM resource_requests").fetchone()[0] == 2
    assert setup.manager.scope.previously_used_gpu_seconds == 7.0
