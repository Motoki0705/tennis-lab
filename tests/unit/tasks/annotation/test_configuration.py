from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.tasks.annotation.configuration import (
    Configuration,
    ConfigurationReceipt,
    ExistingBudget,
    NewBudget,
    configure,
)
from src.tasks.annotation.contracts import Runtime, RuntimeSettings, StageCost
from src.tasks.annotation.controller import Controller
from src.tasks.base.annotation.artifacts import (
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import PreparationConfig
from src.tasks.base.annotation.contracts import ArtifactRef, ModelRef
from src.tasks.base.annotation.execution.contracts import QueueConfiguration
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    CollectionCounts,
    ProductPlan,
    ResourceBudget,
    StageCapacity,
    TaskPlan,
    file_digest,
)
from src.tasks.base.annotation.resources import ResourceManager
from src.tasks.base.annotation.screening.contracts import ScreeningPolicy
from src.utils.paths import PROJECT_ROOT


@dataclass
class Example:
    layout: AnnotationLayout
    configuration: Configuration
    blob: ArtifactRef

    def receipt(self, reference: ArtifactRef) -> ConfigurationReceipt:
        result: ConfigurationReceipt = read_document(
            self.layout, reference, ConfigurationReceipt
        )
        return result


@pytest.fixture
def example(tmp_path: Path) -> Example:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    root = layout.path("campaigns", "inputs")
    blob = write_asset(layout, root / "fixture.bin", b"configuration fixture only")
    selection = write_document(layout, root / "selection.json", ScreeningPolicy())
    preparation = write_document(
        layout,
        root / "preparation.json",
        PreparationConfig(
            format_selector="fixture",
            crf=16,
            preset="ultrafast",
            minimum_clip_frames=32,
            maximum_unprepared_sources=1,
        ),
    )
    script = layout.repository / "queue.sh"
    script.write_text("# fixture --idempotency-key\n")
    stages = tuple(
        StageCapacity(stage=name, kind=kind, parallel=1, threads_per_process=1)
        for name, kind in (
            ("acquire", "io"),
            ("screen", "codex"),
            ("render", "cpu"),
            ("prepare", "cpu"),
            ("infer_ball", "gpu"),
        )
    )
    plan = CampaignPlan(
        campaign_id="fixture",
        phase="preview",
        annotation_root=str(layout.root),
        counts=CollectionCounts(
            channels=1, videos_per_channel=1, max_clips_per_video=1
        ),
        selection=selection,
        preparation=preparation,
        code_commit="a" * 40,
        code_diff_digest="b" * 64,
        agent_model="gpt-6.1-sol",
        agent_effort="low",
        resources=ResourceBudget(
            codex_workers=1,
            additional_storage_bytes=64 * 1024**2,
            minimum_free_bytes=1,
            total_pss_bytes=2 * 1024**3,
            gpu_seconds=60,
            stages=stages,
        ),
        tasks=(
            TaskPlan(
                task="ball",
                model=ModelRef(
                    task="ball",
                    version="fixture",
                    training_weights=blob,
                    inference_weights=blob,
                    configuration=blob,
                ),
                inference=blob,
                review=None,
                learning=None,
                products=(
                    ProductPlan(
                        product="frames",
                        dataset_task="ball_detection",
                        dataset_version="fixture",
                        component="frames",
                        base_snapshot=None,
                        generation_policy=blob,
                    ),
                ),
            ),
        ),
    )
    settings = RuntimeSettings(
        queue=QueueConfiguration(
            script=str(script),
            script_digest=file_digest(script),
            directory=str(layout.repository / ".training_queue"),
            session="fixture",
            issue=1088,
        ),
        codex_binary="codex",
        codex_home=str(layout.path("cache", "codex")),
        costs=tuple(
            StageCost(
                stage=stage.stage,
                pss_bytes=128 * 1024**2,
                storage_bytes=1024**2,
                timeout_seconds=10,
                gpu_seconds=20 if stage.kind == "gpu" else 0,
            )
            for stage in stages
        ),
    )
    configuration = Configuration(
        plan=plan,
        budget=NewBudget(
            scope_id="shared",
            roots=(str(layout.repository),),
            previously_allocated_bytes=1024,
            previously_used_gpu_seconds=7,
            max_pending_per_stage=10,
        ),
        runtime=settings,
    )
    return Example(layout, configuration, blob)


def test_configuration_freezes_inputs_without_launching_or_resetting(
    example: Example,
) -> None:
    ref = configure(example.layout, example.configuration)
    receipt = example.receipt(ref)
    runtime = read_document(example.layout, receipt.runtime, Runtime)
    assert runtime.plan == receipt.plan and runtime.scope == receipt.scope
    resources = ResourceManager(example.layout, receipt.scope)
    baseline = example.layout.artifact_path(resources.scope.baseline)
    inode = baseline.stat().st_ino
    assert configure(example.layout, example.configuration) == ref
    assert baseline.stat().st_ino == inode
    assert resources.scope.previously_used_gpu_seconds == 7
    assert resources.scope.previously_allocated_bytes == 1024
    controller = Controller(example.layout, runtime)
    assert controller.status()["state"] == "pending"
    assert controller.status()["stages"] == []
    assert not (example.layout.repository / ".training_queue").exists()


def test_next_campaign_reuses_stopped_scope_and_cumulative_usage(
    example: Example,
) -> None:
    first = example.receipt(configure(example.layout, example.configuration))
    resources = ResourceManager(example.layout, first.scope)
    with resources.registry.transaction() as db:
        db.execute(
            "UPDATE resource_scopes SET stop_reason='fixture stop' WHERE id='shared'"
        )
    second = example.configuration.model_copy(
        update={
            "plan": example.configuration.plan.model_copy(
                update={"campaign_id": "next"}
            ),
            "budget": ExistingBudget(scope=first.scope),
        }
    )
    receipt = example.receipt(configure(example.layout, second))
    assert receipt.scope == first.scope
    assert receipt.plan != first.plan
    assert (
        ResourceManager(example.layout, receipt.scope).scope.previously_used_gpu_seconds
        == 7
    )
    with resources.registry.transaction() as db:
        assert (
            db.execute(
                "SELECT stop_reason FROM resource_scopes WHERE id='shared'"
            ).fetchone()[0]
            == "fixture stop"
        )
        assert [
            row[0] for row in db.execute("SELECT scope FROM resource_campaigns")
        ] == ["shared", "shared"]
    assert len(list(example.layout.path("resources").glob("*/baseline.jsonl"))) == 1


@pytest.mark.parametrize(
    "damage",
    ["missing_cost", "gpu_kind", "pss", "codex_home", "queue", "selection", "root"],
)
def test_invalid_configuration_does_not_create_budget_or_plan(
    example: Example, damage: str
) -> None:
    data = example.configuration
    if damage == "missing_cost":
        data = data.model_copy(
            update={
                "runtime": data.runtime.model_copy(
                    update={"costs": data.runtime.costs[:-1]}
                )
            }
        )
    elif damage == "gpu_kind":
        costs = (
            *data.runtime.costs[:-1],
            data.runtime.costs[-1].model_copy(update={"gpu_seconds": 0}),
        )
        data = data.model_copy(
            update={"runtime": data.runtime.model_copy(update={"costs": costs})}
        )
    elif damage == "pss":
        costs = (
            data.runtime.costs[0].model_copy(update={"pss_bytes": 3 * 1024**3}),
            *data.runtime.costs[1:],
        )
        data = data.model_copy(
            update={"runtime": data.runtime.model_copy(update={"costs": costs})}
        )
    elif damage == "codex_home":
        data = data.model_copy(
            update={
                "runtime": data.runtime.model_copy(
                    update={
                        "codex_home": str(
                            example.layout.repository.parent / "unmeasured"
                        )
                    }
                )
            }
        )
    elif damage == "queue":
        data = data.model_copy(
            update={
                "runtime": data.runtime.model_copy(
                    update={
                        "queue": data.runtime.queue.model_copy(
                            update={"script_digest": "0" * 64}
                        )
                    }
                )
            }
        )
    elif damage == "selection":
        data = data.model_copy(
            update={"plan": data.plan.model_copy(update={"selection": example.blob})}
        )
    elif damage == "root":
        data = data.model_copy(
            update={
                "budget": NewBudget(
                    scope_id="invalid",
                    roots=(str(example.layout.repository / "absent"),),
                    previously_allocated_bytes=0,
                    previously_used_gpu_seconds=0,
                    max_pending_per_stage=1,
                )
            }
        )
    with pytest.raises(ValueError):
        configure(example.layout, data)
    assert not example.layout.path("resources").exists()
    assert not example.layout.path("campaigns", data.plan.campaign_id).exists()


def test_frozen_configuration_cannot_change_or_rebaseline(example: Example) -> None:
    ref = configure(example.layout, example.configuration)
    receipt = example.receipt(ref)
    resources = ResourceManager(example.layout, receipt.scope)
    inode = example.layout.artifact_path(resources.scope.baseline).stat().st_ino
    changed = example.configuration.model_copy(
        update={
            "plan": example.configuration.plan.model_copy(
                update={
                    "counts": CollectionCounts(
                        channels=2, videos_per_channel=1, max_clips_per_video=1
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="immutable"):
        configure(example.layout, changed)
    assert example.layout.artifact_path(resources.scope.baseline).stat().st_ino == inode
    assert configure(example.layout, example.configuration) == ref


def test_interrupted_configuration_recovers_the_original_scope(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.tasks.annotation.configuration as module

    original = module.write_document

    def interrupted(layout: Any, path: Path, document: Any) -> Any:
        if path.name == "runtime.json":
            raise RuntimeError("fixture interruption after scope binding")
        return original(layout, path, document)

    monkeypatch.setattr(module, "write_document", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        configure(example.layout, example.configuration)
    baseline = example.layout.path("resources", "shared", "baseline.jsonl")
    inode, digest = baseline.stat().st_ino, file_digest(baseline)
    monkeypatch.setattr(module, "write_document", original)
    receipt = example.receipt(configure(example.layout, example.configuration))
    resources = ResourceManager(example.layout, receipt.scope)
    assert baseline.stat().st_ino == inode and file_digest(baseline) == digest
    assert resources.scope.previously_used_gpu_seconds == 7


def test_cli_schema_and_configuration_round_trip(example: Example) -> None:
    schema = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.annotation",
            "schema",
            "--type",
            "configuration",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    document = json.loads(schema.stdout)
    assert set(document["required"]) == {"plan", "budget", "runtime"}
    assert document["properties"]["budget"]["discriminator"]["propertyName"] == "action"
    request = write_document(
        example.layout,
        example.layout.path("campaigns", "inputs", "configure.json"),
        example.configuration,
    )
    command = [
        sys.executable,
        "-m",
        "src.tasks.annotation",
        "configure",
        "--configuration",
        str(example.layout.artifact_path(request)),
        "--sha256",
        request.digest,
        "--test-root",
        str(example.layout.root),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    result = ArtifactRef.model_validate(json.loads(completed.stdout))
    receipt = example.receipt(result)
    assert (
        read_document(example.layout, receipt.request, Configuration)
        == example.configuration
    )
    assert configure(example.layout, example.configuration) == result
    assert not (example.layout.repository / ".training_queue").exists()


@pytest.mark.parametrize("damage", ["limits", "missing_reference"])
def test_reuse_never_creates_a_replacement_budget(
    example: Example, damage: str
) -> None:
    first = example.receipt(configure(example.layout, example.configuration))
    plan = example.configuration.plan.model_copy(update={"campaign_id": "next"})
    reference = first.scope
    if damage == "limits":
        plan = plan.model_copy(
            update={
                "resources": plan.resources.model_copy(
                    update={"gpu_seconds": plan.resources.gpu_seconds + 1}
                )
            }
        )
    else:
        reference = reference.model_copy(
            update={"path": "resources/missing/scope.json"}
        )
    changed = example.configuration.model_copy(
        update={"plan": plan, "budget": ExistingBudget(scope=reference)}
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        configure(example.layout, changed)
    assert len(list(example.layout.path("resources").glob("*/baseline.jsonl"))) == 1
    assert not example.layout.path("campaigns", "next").exists()


def test_production_code_pin_is_checked_before_any_configuration_write(
    example: Example,
) -> None:
    production = AnnotationLayout(
        example.layout.repository,
        example.layout.repository / "outputs/chat_annotation",
        "production",
    )
    data = example.configuration.model_copy(
        update={
            "plan": example.configuration.plan.model_copy(
                update={"annotation_root": str(production.root)}
            )
        }
    )
    with pytest.raises(ValueError, match="frozen commit/diff"):
        configure(production, data)
    assert not production.root.exists()
