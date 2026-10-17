from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    CollectionCounts,
    load_plan,
    save_plan,
)


def payload(root: Path) -> dict[str, object]:
    artifact = root.parent / "outputs/configs/policy.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"fixture input")
    ref = {
        "role": "artifact",
        "path": "configs/policy.json",
        "digest": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "size_bytes": artifact.stat().st_size,
    }
    return {
        "campaign_id": "new-clips",
        "phase": "preview",
        "annotation_root": str(root),
        "counts": {"channels": 10, "videos_per_channel": 10, "max_clips_per_video": 5},
        "selection": ref,
        "preparation": ref,
        "code_commit": "a" * 40,
        "code_diff_digest": "b" * 64,
        "agent_model": "gpt-6.1-sol",
        "agent_effort": "high",
        "resources": {
            "codex_workers": 50,
            "additional_storage_bytes": 30 * 1024**3,
            "minimum_free_bytes": 30 * 1024**3,
            "total_pss_bytes": 22 * 1024**3,
            "gpu_seconds": 12 * 3600,
            "stages": [
                {
                    "stage": "review",
                    "kind": "codex",
                    "parallel": 50,
                    "threads_per_process": 1,
                }
            ],
        },
        "tasks": [
            {
                "task": "detector",
                "model": {
                    "task": "detector",
                    "version": "adopted-v1",
                    "training_weights": ref,
                    "inference_weights": ref,
                    "configuration": ref,
                },
                "inference": ref,
                "review": None,
                "learning": None,
                "products": [
                    {
                        "product": "boxes",
                        "dataset_task": "detector",
                        "dataset_version": "v2",
                        "component": "frames",
                        "base_snapshot": ref,
                        "generation_policy": ref,
                    }
                ],
            }
        ],
    }


def test_independent_collection_counts_and_strict_values() -> None:
    counts = CollectionCounts(channels=10, videos_per_channel=10, max_clips_per_video=5)
    assert (counts.videos, counts.max_clips) == (100, 500)
    with pytest.raises(ValidationError):
        CollectionCounts(channels=True, videos_per_channel=10, max_clips_per_video=5)
    with pytest.raises(ValidationError):
        CollectionCounts(channels=0, videos_per_channel=10, max_clips_per_video=5)


def test_preview_never_allows_adoption_and_production_requires_policy(
    tmp_path: Path,
) -> None:
    raw = payload(tmp_path / "annotation")
    plan = CampaignPlan.model_validate_json(json.dumps(raw))
    assert not plan.allows_adoption
    raw["phase"] = "production"
    with pytest.raises(ValidationError, match="review policies"):
        CampaignPlan.model_validate_json(json.dumps(raw))


def test_nested_values_are_immutable(tmp_path: Path) -> None:
    plan = CampaignPlan.model_validate_json(
        json.dumps(payload(tmp_path / "annotation"))
    )
    with pytest.raises(ValidationError):
        plan.tasks[0].model.version = "changed"
    assert isinstance(plan.resources.stages, tuple)


def test_concurrent_registration_reuses_exact_plan_and_refuses_changes(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = CampaignPlan.model_validate_json(json.dumps(payload(layout.root)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: save_plan(layout, plan), range(8)))
    assert len(set(paths)) == 1
    assert load_plan(paths[0], expected_digest=plan.sha256) == plan
    raw = plan.model_dump(mode="json")
    raw["tasks"][0]["model"]["version"] = "changed"
    changed = CampaignPlan.model_validate_json(json.dumps(raw))
    before = paths[0].read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        save_plan(layout, changed)
    assert paths[0].read_bytes() == before


def test_plan_rejects_root_change_and_duplicate_json_keys(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    raw = payload(tmp_path / "other")
    with pytest.raises(ValueError, match="root differs"):
        save_plan(layout, CampaignPlan.model_validate_json(json.dumps(raw)))
    path = tmp_path / "duplicate.json"
    path.write_text('{"campaign_id":"first","campaign_id":"second"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_plan(path)


def test_pinned_input_tampering_prevents_plan_registration(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = CampaignPlan.model_validate_json(json.dumps(payload(layout.root)))
    artifact = tmp_path / "outputs/configs/policy.json"
    artifact.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="input hash/size mismatch"):
        save_plan(layout, plan)
    assert not layout.root.exists()


def test_published_components_can_share_a_dataset_version(tmp_path: Path) -> None:
    original = CampaignPlan.model_validate_json(
        json.dumps(payload(tmp_path / "annotation"))
    )
    raw = original.model_dump(mode="json")
    product = dict(raw["tasks"][0]["products"][0])
    product.update(product="pose", component="player_pose")
    raw["tasks"][0]["products"].append(product)
    plan = CampaignPlan.model_validate_json(json.dumps(raw))
    assert len(plan.tasks[0].products) == 2
    raw["tasks"][0]["products"][1]["component"] = "frames"
    with pytest.raises(ValidationError, match="destinations"):
        CampaignPlan.model_validate_json(json.dumps(raw))


def test_unknown_fields_and_overallocated_resources_fail(tmp_path: Path) -> None:
    plan = CampaignPlan.model_validate_json(
        json.dumps(payload(tmp_path / "annotation"))
    )
    raw = plan.model_dump(mode="json")
    raw["qa_required"] = True
    with pytest.raises(ValidationError):
        CampaignPlan.model_validate_json(json.dumps(raw))
    del raw["qa_required"]
    raw["resources"]["codex_workers"] = 25
    with pytest.raises(ValidationError, match="allocations"):
        CampaignPlan.model_validate_json(json.dumps(raw))


def test_interrupted_publish_leaves_no_partial_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = CampaignPlan.model_validate_json(json.dumps(payload(layout.root)))

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("injected before atomic publish")

    with monkeypatch.context() as context:
        context.setattr("src.tasks.base.annotation.plan.os.replace", fail_replace)
        with pytest.raises(OSError, match="injected"):
            save_plan(layout, plan)
    directory = layout.path("campaigns", plan.campaign_id)
    assert [p.name for p in directory.iterdir()] == ["plan.lock"]
    assert load_plan(save_plan(layout, plan)) == plan
