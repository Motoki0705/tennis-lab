from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
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
    annotation_reference,
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
from src.tasks.player_detection.annotation.contracts import DinoInferenceConfig
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


def test_configuration_freezes_extra_free_space_mount_without_scanning_it(
    example: Example,
) -> None:
    assert isinstance(example.configuration.budget, NewBudget)
    configured = example.configuration.model_copy(
        update={
            "budget": example.configuration.budget.model_copy(
                update={"free_space_mounts": ("/",)}
            )
        }
    )
    reference = configure(example.layout, configured)
    receipt = example.receipt(reference)
    guard = ResourceManager(example.layout, receipt.scope)
    assert guard.scope.free_space_mounts == ("/",)
    assert guard.scope.roots == (str(example.layout.repository),)
    for line in (
        example.layout.artifact_path(guard.scope.baseline).read_text().splitlines()
    ):
        assert Path(json.loads(line)[3]).is_relative_to(example.layout.repository)
    assert configure(example.layout, configured) == reference


@pytest.mark.parametrize(
    "variant", ["valid", "video_decoder", "video_output", "quality"]
)
def test_jpeg_configuration_requires_one_explicit_frame_contract(
    example: Example, variant: str
) -> None:
    from src.tasks.ball_detection.annotation.contracts import MDDInferenceConfig
    from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
        BallPublicationPolicy,
    )

    layout = example.layout
    inputs = layout.path("campaigns", "jpeg-inputs")
    preparation = write_document(
        layout,
        inputs / "preparation.json",
        PreparationConfig(
            format_selector="fixture",
            retained_media="jpeg_q90",
            source_retention="delete_after_verified_frames",
            crf=None,
            preset=None,
            minimum_clip_frames=32,
            maximum_unprepared_sources=1,
        ),
    )
    inference = write_document(
        layout,
        inputs / "inference.json",
        MDDInferenceConfig(
            stride=32,
            device="cpu",
            precision="float32",
            decode=(
                "pyav_rgb24_native"
                if variant == "video_decoder"
                else "opencv_jpeg_rgb24_native"
            ),
        ),
    )
    publication = write_document(
        layout,
        inputs / "publication.json",
        BallPublicationPolicy(
            jpeg_quality=80 if variant == "quality" else 90,
            image_storage="video_q90" if variant == "video_output" else "jpeg",
            source_splits=(),
        ),
    )
    task = example.configuration.plan.tasks[0]
    plan = example.configuration.plan.model_copy(
        update={
            "preparation": preparation,
            "tasks": (
                task.model_copy(
                    update={
                        "inference": inference,
                        "products": (
                            task.products[0].model_copy(
                                update={"generation_policy": publication}
                            ),
                        ),
                    }
                ),
            ),
        }
    )
    config = example.configuration.model_copy(update={"plan": plan})
    if variant == "valid":
        assert example.receipt(configure(layout, config)).plan
    else:
        with pytest.raises(ValueError, match="JPEG preparation requires"):
            configure(layout, config)
        assert not layout.path("resources", "shared", "scope.json").exists()


@pytest.mark.parametrize("remove_file", [False, True])
def test_reuse_retains_file_roots_baseline_and_stop(
    example: Example, remove_file: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = example.layout.repository.with_name(
        example.layout.repository.name + ".wal"
    )
    external.write_bytes(b"original allocation" * 1024)
    assert isinstance(example.configuration.budget, NewBudget)
    configured = example.configuration.model_copy(
        update={
            "budget": example.configuration.budget.model_copy(
                update={"roots": (*example.configuration.budget.roots, str(external))}
            )
        }
    )
    first_ref = configure(example.layout, configured)
    first = example.receipt(first_ref)
    resources = ResourceManager(example.layout, first.scope)
    baseline = resources.scope.baseline
    inode = example.layout.artifact_path(baseline).stat().st_ino
    if remove_file:
        external.unlink()
    else:
        with external.open("ab") as handle:
            handle.write(b"added allocation" * 1024)
    with resources.registry.transaction() as db:
        db.execute(
            "UPDATE resource_scopes SET stop_reason='existing stop' WHERE id='shared'"
        )
    # Replaying the original create request must also reuse its existing scope.
    assert configure(example.layout, configured) == first_ref

    def no_new_baseline(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("scope reuse must not create a new baseline")

    monkeypatch.setattr(ResourceManager, "create", no_new_baseline)
    second = configured.model_copy(
        update={
            "plan": configured.plan.model_copy(update={"campaign_id": "next"}),
            "budget": ExistingBudget(scope=first.scope),
        }
    )
    receipt = example.receipt(configure(example.layout, second))
    reused = ResourceManager(example.layout, receipt.scope)
    assert receipt.scope == first.scope
    assert reused.scope == resources.scope
    assert reused.scope.baseline == baseline
    assert example.layout.artifact_path(baseline).stat().st_ino == inode
    sample = reused.sample()
    assert sample.stopped_reason == "existing stop"
    assert sample.gpu_charged_seconds == 7
    assert sample.storage.additional_bytes >= 1024


@pytest.mark.parametrize("kind", ["missing", "fifo"])
def test_new_scope_rejects_unmeasurable_roots(example: Example, kind: str) -> None:
    external = example.layout.repository.with_name(
        example.layout.repository.name + ".state"
    )
    if kind == "fifo":
        os.mkfifo(external)
    assert isinstance(example.configuration.budget, NewBudget)
    configured = example.configuration.model_copy(
        update={
            "budget": example.configuration.budget.model_copy(
                update={"roots": (*example.configuration.budget.roots, str(external))}
            )
        }
    )
    with pytest.raises(ValueError, match="regular files or directories"):
        configure(example.layout, configured)
    assert not example.layout.path("campaigns", "fixture", "configuration").exists()


def test_file_inventory_does_not_cover_a_codex_home(example: Example) -> None:
    external = example.layout.repository.with_name(
        example.layout.repository.name + ".state"
    )
    external.write_text("a file cannot contain Codex state")
    assert isinstance(example.configuration.budget, NewBudget)
    configured = example.configuration.model_copy(
        update={
            "budget": example.configuration.budget.model_copy(
                update={"roots": (*example.configuration.budget.roots, str(external))}
            ),
            "runtime": example.configuration.runtime.model_copy(
                update={"codex_home": str(external / "codex")}
            ),
        }
    )
    with pytest.raises(ValueError, match="inside the resource inventory"):
        configure(example.layout, configured)


def test_existing_scope_mismatch_is_rejected_before_new_campaign_is_saved(
    example: Example,
) -> None:
    configure(example.layout, example.configuration)
    assert isinstance(example.configuration.budget, NewBudget)
    changed = example.configuration.model_copy(
        update={
            "plan": example.configuration.plan.model_copy(
                update={"campaign_id": "changed"}
            ),
            "budget": example.configuration.budget.model_copy(
                update={"previously_used_gpu_seconds": 0.0}
            ),
        }
    )
    with pytest.raises(ValueError, match="existing resource scope differs"):
        configure(example.layout, changed)
    assert not example.layout.path("campaigns", "changed").exists()


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


@pytest.mark.parametrize(
    ("kind", "gpu_seconds"), [("cpu", 0), ("io", 0), ("gpu", 0), ("cpu", 20)]
)
def test_player_inference_requires_gpu_reservation_before_freezing_configuration(
    example: Example, kind: str, gpu_seconds: float
) -> None:
    data = example.configuration.model_dump()
    task = data["plan"]["tasks"][0]
    task["task"] = task["model"]["task"] = "player"
    task["products"][0].update(
        product="bbox", dataset_task="player_detection", component="bbox"
    )
    inference = write_document(
        example.layout,
        example.layout.path("campaigns", "inputs", "player-inference.json"),
        DinoInferenceConfig(
            confidence_threshold=0.5, short_side=800, max_long_side=1333
        ),
    )
    task["inference"] = inference.model_dump()
    data["plan"]["resources"]["stages"][-1]["stage"] = "infer_player"
    data["runtime"]["costs"][-1]["stage"] = "infer_player"
    valid = Configuration.model_validate(data)
    data["plan"]["resources"]["stages"][-1]["kind"] = kind
    data["runtime"]["costs"][-1]["gpu_seconds"] = gpu_seconds
    invalid = Configuration.model_validate(data)

    with pytest.raises(ValueError, match="GPU|stage kinds"):
        configure(example.layout, invalid)
    assert not example.layout.path("resources").exists()
    assert not example.layout.path("campaigns", valid.plan.campaign_id).exists()

    reference = configure(example.layout, valid)
    receipt = example.receipt(reference)
    assert read_document(example.layout, receipt.request, Configuration) == valid
    runtime = read_document(example.layout, receipt.runtime, Runtime)
    assert runtime.cost("infer_player").gpu_seconds == 20
    assert configure(example.layout, valid) == reference


@pytest.mark.parametrize("field", ["previously_allocated_bytes", "previously_used_gpu_seconds"])
@pytest.mark.parametrize("recovery", ["create", "reuse"])
def test_scope_input_conflict_allows_corrected_configuration(
    example: Example, field: str, recovery: str
) -> None:
    original = example.configuration
    assert isinstance(original.budget, NewBudget)
    first = example.receipt(configure(example.layout, original))
    resources = ResourceManager(example.layout, first.scope)
    baseline = example.layout.artifact_path(resources.scope.baseline)
    inode, digest = baseline.stat().st_ino, file_digest(baseline)
    plan = original.plan.model_copy(update={"campaign_id": "next"})
    conflicting = original.model_copy(
        update={
            "plan": plan,
            "budget": original.budget.model_copy(update={field: 0}),
        }
    )
    with pytest.raises(ValueError, match="existing resource scope differs"):
        configure(example.layout, conflicting)
    assert not example.layout.path("campaigns", "next").exists()
    corrected = original.model_copy(
        update={
            "plan": plan,
            "budget": original.budget
            if recovery == "create"
            else ExistingBudget(scope=first.scope),
        }
    )
    reference = configure(example.layout, corrected)
    receipt = example.receipt(reference)
    assert receipt.scope == first.scope
    assert read_document(example.layout, receipt.request, Configuration) == corrected
    assert configure(example.layout, corrected) == reference
    assert baseline.stat().st_ino == inode and file_digest(baseline) == digest


@pytest.mark.parametrize("action", ["create", "reuse"])
def test_campaign_binding_conflict_allows_corrected_configuration(
    example: Example, action: str
) -> None:
    original = example.configuration
    assert isinstance(original.budget, NewBudget)
    first = example.receipt(configure(example.layout, original))
    resources = ResourceManager(example.layout, first.scope)
    plan = original.plan.model_copy(update={"campaign_id": "next"})
    resources.bind(plan)
    other_budget = original.budget.model_copy(update={"scope_id": "other"})
    other = example.receipt(
        configure(
            example.layout,
            original.model_copy(
                update={
                    "plan": original.plan.model_copy(update={"campaign_id": "other"}),
                    "budget": other_budget,
                }
            ),
        )
    )
    conflicting = original.model_copy(
        update={
            "plan": plan,
            "budget": other_budget if action == "create" else ExistingBudget(scope=other.scope),
        }
    )
    with pytest.raises(ValueError, match="different budget scope or plan"):
        configure(example.layout, conflicting)
    assert not example.layout.path("campaigns", "next").exists()
    corrected = original.model_copy(
        update={"plan": plan, "budget": ExistingBudget(scope=first.scope)}
    )
    receipt = example.receipt(configure(example.layout, corrected))
    assert receipt.scope == first.scope
    assert read_document(example.layout, receipt.request, Configuration) == corrected


@pytest.mark.parametrize("boundary", ["request.json", "runtime.json"])
def test_interrupted_configuration_recovers_the_original_scope(
    example: Example, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    import src.tasks.annotation.configuration as module

    original = module.write_document

    def interrupted(layout: Any, path: Path, document: Any) -> Any:
        if path.name == boundary:
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


def test_explicit_code_update_keeps_plan_runtime_inputs_and_budget(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.code_update import CodeUpdateReceipt, update_code
    from src.tasks.base.annotation.reproducibility import (
        CodeRevision,
        require_frozen_code,
    )

    layout = example.layout
    configured = example.receipt(configure(layout, example.configuration))
    runtime = read_document(layout, configured.runtime, Runtime)
    controller = Controller(layout, runtime)
    original_plan = layout.artifact_path(configured.plan).read_bytes()
    original_runtime = layout.artifact_path(configured.runtime).read_bytes()
    original_scope = controller.resources.scope
    identity = ("c" * 40, hashlib.sha256(b"").hexdigest())
    monkeypatch.setattr(
        "src.tasks.annotation.code_update.checkout_identity", lambda _: identity
    )
    monkeypatch.setattr(
        "src.tasks.base.annotation.reproducibility.checkout_identity",
        lambda _: identity,
    )
    reference = update_code(
        controller,
        configured.runtime,
        example.blob,
        reason="synthetic verified bug fix",
    )
    assert (
        update_code(
            controller,
            configured.runtime,
            example.blob,
            reason="synthetic verified bug fix",
        )
        == reference
    )
    receipt = read_document(layout, reference, CodeUpdateReceipt)
    updated = read_document(layout, receipt.runtime, Runtime)
    revision = read_document(layout, receipt.revision, CodeRevision)
    assert updated == runtime.model_copy(update={"code_revision": receipt.revision})
    assert (
        revision.plan,
        revision.scope,
        revision.previous_runtime,
        revision.verification,
    ) == (configured.plan, configured.scope, configured.runtime, example.blob)
    assert layout.artifact_path(configured.plan).read_bytes() == original_plan
    assert layout.artifact_path(configured.runtime).read_bytes() == original_runtime
    assert b'"code_revision"' not in original_runtime
    assert controller.resources.scope == original_scope
    require_frozen_code(
        PROJECT_ROOT,
        controller.plan,
        layout=layout,
        revision=receipt.revision,
        scope=runtime.scope,
    )
    with pytest.raises(ValueError, match="frozen commit/diff"):
        require_frozen_code(PROJECT_ROOT, controller.plan)
    with pytest.raises(ValueError, match="another plan or resource scope"):
        require_frozen_code(
            PROJECT_ROOT,
            controller.plan,
            layout=layout,
            revision=receipt.revision,
            scope=example.blob,
        )
    layout.artifact_path(example.blob).write_bytes(b"changed verification evidence")
    with pytest.raises(ValueError, match="size changed|hash changed"):
        require_frozen_code(
            PROJECT_ROOT,
            controller.plan,
            layout=layout,
            revision=receipt.revision,
            scope=runtime.scope,
        )


def test_code_update_refuses_live_resource_owner(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.code_update import update_code
    from src.tasks.base.annotation.resources import ResourceRequest

    configured = example.receipt(configure(example.layout, example.configuration))
    controller = Controller(
        example.layout, read_document(example.layout, configured.runtime, Runtime)
    )
    monkeypatch.setattr(
        "src.tasks.annotation.code_update.checkout_identity",
        lambda _: ("c" * 40, hashlib.sha256(b"").hexdigest()),
    )
    request = ResourceRequest(
        request_id="live-before-code-update",
        plan=configured.plan,
        stage="acquire",
        pss_bytes=1024,
        storage_bytes=1024,
        gpu_seconds=0.0,
    )
    with (
        controller.resources.lease(request),
        pytest.raises(ValueError, match="stop and drain"),
    ):
        update_code(
            controller,
            configured.runtime,
            example.blob,
            reason="synthetic live guard",
        )
    assert not list(
        example.layout.path(
            "campaigns", controller.plan.campaign_id, "code-updates"
        ).glob("*/result.json")
    )


def test_code_update_reads_old_attempt_without_launching_or_changing_its_cost(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.code_update import CodeUpdateReceipt, update_code
    from src.tasks.annotation.contracts import OperationInput, ScreenOperation
    from src.tasks.base.annotation.execution.contracts import StageLaunch
    from src.tasks.base.annotation.execution.runner import stage_directory

    layout = example.layout
    configured = example.receipt(configure(layout, example.configuration))
    runtime = read_document(layout, configured.runtime, Runtime)
    controller = Controller(layout, runtime)
    identity = ("c" * 40, hashlib.sha256(b"").hexdigest())
    monkeypatch.setattr(
        "src.tasks.annotation.code_update.checkout_identity", lambda _: identity
    )
    monkeypatch.setattr(
        "src.tasks.base.annotation.reproducibility.checkout_identity",
        lambda _: identity,
    )
    revision = read_document(
        layout,
        update_code(
            controller, configured.runtime, example.blob, reason="synthetic update"
        ),
        CodeUpdateReceipt,
    )
    operation = OperationInput(
        plan=runtime.plan, operation=ScreenOperation(source_id="a" * 64)
    )

    def record_pending(launch: StageLaunch) -> dict[str, object]:
        ref = write_document(
            layout, stage_directory(layout, launch) / "launch.json", launch
        )
        return {"state": "pending", "launch": ref.model_dump()}

    monkeypatch.setattr(controller.stages, "submit", record_pending)
    first = controller.operation(operation)
    old_ref = ArtifactRef.model_validate(first["launch"])
    original = layout.artifact_path(old_ref).read_bytes()
    updated = Controller(layout, read_document(layout, revision.runtime, Runtime))
    monkeypatch.setattr(updated.stages, "status", lambda _: first)
    monkeypatch.setattr(
        updated.stages, "submit", lambda _: pytest.fail("old code must not relaunch")
    )
    assert updated.operation(operation) == first
    assert layout.artifact_path(old_ref).read_bytes() == original
    updated.jobs.runtime = updated.runtime.model_copy(
        update={
            "costs": tuple(
                cost.model_copy(update={"storage_bytes": cost.storage_bytes + 1})
                for cost in runtime.costs
            )
        }
    )
    with pytest.raises(ValueError, match="resource/command configuration changed"):
        updated.operation(operation)


@pytest.mark.parametrize("registered_resource", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_code_update_cancels_unregistered_worker_launch_before_explicit_retry(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    registered_resource: bool,
    interrupted: bool,
) -> None:
    from src.tasks.annotation.code_update import CodeUpdateReceipt, update_code
    from src.tasks.base.annotation.resources import CapacityUnavailable
    from src.tasks.base.annotation.workers.contracts import (
        Launch,
        PreparedWorker,
        WorkOutcome,
        WorkRequest,
    )
    from src.tasks.base.annotation.workers.ledger import WorkerStopped

    layout = example.layout
    configured = example.receipt(configure(layout, example.configuration))
    controller = Controller(layout, read_document(layout, configured.runtime, Runtime))
    pool = controller.pool
    adapter = pool.adapters["source_screening"]
    monkeypatch.setattr(adapter, "input_references", lambda *_: ())
    monkeypatch.setattr(
        adapter,
        "prepare",
        lambda _layout, _plan, _request, directory: PreparedWorker(
            prompt=write_asset(layout, directory / "prompt.md", b"synthetic worker"),
            inputs=(example.blob,),
        ),
    )
    monkeypatch.setattr(
        adapter,
        "validate",
        lambda *_: WorkOutcome(
            status="complete", receipt=example.blob, adoption=None, reason=None
        ),
    )
    pool.test_command = (
        sys.executable,
        "-c",
        "import sys,json,pathlib; sys.stdin.read(); "
        "pathlib.Path('result.json').write_text('{}'); "
        "print(json.dumps({'type':'thread.started','thread_id':'fixture-thread'})); "
        "print(json.dumps({'type':'turn.completed','usage':{}}))",
    )
    request = WorkRequest(
        adapter=adapter.name,
        adapter_version=adapter.version,
        stage="screen",
        subject_id="a" * 64,
        payload=example.blob,
        inputs=(example.blob,),
    )

    def full(*_args, **_kwargs):
        raise CapacityUnavailable("synthetic resource backpressure")

    with monkeypatch.context() as patch:
        patch.setattr(
            controller.resources, "_admit" if registered_resource else "submit", full
        )
        with pytest.raises(CapacityUnavailable, match="backpressure"):
            pool.start(request)
    assert pool.ledger.task(request.work_id)["state"] == "ready"
    assert pool.ledger.history(controller.plan, request) == ()
    path = layout.path(
        "campaigns",
        controller.plan.campaign_id,
        "workers",
        request.work_id,
        "attempt-0001",
        "launch.json",
    )
    old_ref = annotation_reference(layout, path)
    original = path.read_bytes()
    old = read_document(layout, old_ref, Launch)
    assert old.resource_request is not None
    identity = ("c" * 40, hashlib.sha256(b"").hexdigest())
    monkeypatch.setattr(
        "src.tasks.annotation.code_update.checkout_identity", lambda _: identity
    )
    monkeypatch.setattr(
        "src.tasks.base.annotation.reproducibility.checkout_identity",
        lambda _: identity,
    )
    if interrupted:
        from src.tasks.annotation import code_update

        cancel = controller.resources.cancel_pending

        def interrupt_cancel(request, *, reason):
            cancel(request, reason=reason)
            raise RuntimeError("interrupted after cancellation")

        def interrupt_result(layout, path, document):
            if path.name == "result.json":
                raise RuntimeError("interrupted after cancellation")
            return write_document(layout, path, document)

        with monkeypatch.context() as patch:
            if registered_resource:
                patch.setattr(controller.resources, "cancel_pending", interrupt_cancel)
            else:
                patch.setattr(code_update, "write_document", interrupt_result)
            with pytest.raises(RuntimeError, match="interrupted after cancellation"):
                update_code(
                    controller,
                    configured.runtime,
                    example.blob,
                    reason="fix waiting worker",
                )
    result = update_code(
        controller, configured.runtime, example.blob, reason="fix waiting worker"
    )
    receipt = read_document(layout, result, CodeUpdateReceipt)
    updated = Controller(layout, read_document(layout, receipt.runtime, Runtime))
    updated.pool.adapters[adapter.name] = adapter
    updated.pool.test_command = pool.test_command
    assert pool.ledger.task(request.work_id)["state"] == "cancelled"
    with pytest.raises(WorkerStopped, match="explicit retry"):
        updated.pool.start(request)
    history = updated.pool.ledger.history(controller.plan, request)
    assert len(history) == 1 and history[0].state == "cancelled"
    assert history[0].launch == old_ref
    if registered_resource:
        assert controller.resources._row(old.resource_request)["state"] == "cancelled"
    assert (
        update_code(
            controller, configured.runtime, example.blob, reason="fix waiting worker"
        )
        == result
    )
    assert updated.pool.start(request, retry=True) == request.work_id + "-0002"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = updated.pool.reconcile(request)
        if state["state"] != "running":
            break
        time.sleep(0.05)
    assert state["state"] == "complete"
    history = updated.pool.ledger.history(controller.plan, request)
    assert [item.state for item in history] == ["cancelled", "complete"]
    assert (
        read_document(layout, history[1].launch, Launch).code_revision
        == receipt.revision
    )
    assert path.read_bytes() == original


def test_explicit_pending_cancellation_preserves_request_and_recovers_receipt(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation import recovery
    from src.tasks.annotation.contracts import CancelPendingStage, RecoveryInput
    from src.tasks.base.annotation.execution.contracts import StageLaunch
    from src.tasks.base.annotation.resources import ResourceRequest

    configured = example.receipt(configure(example.layout, example.configuration))
    controller = Controller(
        example.layout, read_document(example.layout, configured.runtime, Runtime)
    )
    request = ResourceRequest(
        request_id="cancel-pending",
        plan=configured.plan,
        stage="acquire",
        pss_bytes=1024,
        storage_bytes=1024,
        gpu_seconds=0.0,
    )
    controller.resources.submit(request)
    launch = write_document(
        example.layout,
        example.layout.path("campaigns", "fixture", "pending-launch.json"),
        StageLaunch(
            scope=configured.scope,
            resource=request,
            command=("fixture-command-never-executed",),
            code_root=str(PROJECT_ROOT),
            timeout_seconds=10.0,
        ),
    )
    data = RecoveryInput(
        plan=configured.plan,
        reason="old code attempt replaced after a verified fix",
        operation=CancelPendingStage(launch=launch),
    )
    original = recovery.write_document

    def interrupt(layout: Any, path: Path, document: Any) -> Any:
        if path.name == "result.json":
            raise RuntimeError("fixture interruption after cancellation")
        return original(layout, path, document)

    monkeypatch.setattr(recovery, "write_document", interrupt)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        recovery.recover(controller, data)
    monkeypatch.setattr(recovery, "write_document", original)
    ref = recovery.recover(controller, data)
    assert recovery.recover(controller, data) == ref
    row = controller.resources._row(request)
    assert row["state"] == "cancelled" and row["reason"] == data.reason
    assert row["request"] == request.canonical_bytes().decode()
