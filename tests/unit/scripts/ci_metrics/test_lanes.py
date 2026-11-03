"""Registered lane dependencies must request entire files without losing CPU cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from scripts.ci_metrics.lanes import (
    REGISTRY,
    load_registry,
    make_plan,
    partition_nodeids,
    validate_plan,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[4]
# Independent examples from the import/CLI audit, including cross-domain users.
# These must remain useful when a registry pattern is accidentally narrowed.
DEPENDENCY_CHANGES = {
    "javascript": ("src/tennis_scene/clip_studio/web/static/app.js",),
    "ops-scenarios": (
        ".agents/ops/lib/agent_exec.sh",
        ".agents/ops/review/merge_gate.py",
        ".agents/skills/parent-session/SKILL.md",
        "tests/e2e/agents_ops/board_browser.cjs",
    ),
    "agents-documentation": ("AGENTS.md", ".agents/memory/SUMMARY.md"),
    "ball-learning": (
        "src/tasks/ball_detection/scripts/train.py",
        "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml",
        "tests/integration/tasks/ball_detection/mdd_training_contracts.py",
        "src/tennis_scene/pipeline/storage/codec.py",
        "src/submodules/models/dino/architecture.py",
        "src/synthetic_data_generation/alignment/contracts.py",
        "tests/benchmarks/ball_detection/detection_holdout.py",
    ),
    "configuration-audit": ("scripts/audit_configuration.py",),
    "precommit-wrapper": (
        ".pre-commit-config.yaml",
        "scripts/run_in_repo_venv.sh",
    ),
    "github-labels": ("scripts/sync_github_labels.py", ".github/labels.yml"),
    "public-documentation": (
        "README.ja.md",
        "LICENSES.zh-CN.md",
        "docs/installation/README.md",
        "assets/branding/tennis-lab-header.png",
        "src/tennis_scene/installation_assets.py",
        "src/tennis_scene/__init__.py",
    ),
    "geometry-documentation": (
        "src/tasks/blcs/README.md",
        "src/tasks/plcs/README.md",
    ),
    "real-rgb-entrypoints": (
        "src/tasks/slcs/scripts/evaluate_run.py",
        "src/tasks/ball_detection/models/mdd/detectors.py",
        # Reintroducing a removed root CLI must rerun the ownership check too.
        "scripts/datasets/build_real_rgb.sh",
    ),
    "dsh-runtime": ("tools/dsh/runner.py", "tools/dsh/package-lock.json"),
    "agent-skills": (
        ".agents/skills/codex-scheduled-followup/scripts/cli_heartbeat.py",
        ".agents/skills/research-loop/scripts/lit_search.py",
        ".github/ISSUE_TEMPLATE/05_research_cycle.md",
    ),
    "drive-cli": (".agents/skills/tennis-drive/scripts/drive.py",),
    "synthetic-publication": (
        "src/synthetic_data_generation/scripts/run_scene_pipeline.py",
        "src/tennis_scene/generate_dataset/manifest.py",
        "src/tasks/court_detection/configs/train.yaml",
        "third_party/nht",
    ),
    "scene-cli": (
        "src/tennis_scene/scripts/generate_dataset.py",
        "scripts/prepare_clean_install.py",
        "src/submodules/models/dino/architecture.py",
    ),
    "ball-learning-visualization": (
        "src/tasks/ball_detection/scripts/visualize_learning_stages.py",
        "src/tasks/ball_detection/visualization/README.md",
    ),
    "blcs-generation": (
        "src/tasks/blcs/scripts/generate_dataset.py",
        "src/synthetic_data_generation/scene_contract.py",
    ),
    "plcs-operations": (
        "src/tasks/plcs/scripts/generate_dataset.py",
        "src/tasks/plcs/scripts/analysis/analyze_dataset_distribution.py",
        "src/synthetic_data_generation/scene_contract.py",
    ),
    "slcs-analysis": (
        "src/tasks/slcs/scripts/analyze_predictions.py",
        "src/synthetic_data_generation/alignment/contracts.py",
        "src/tasks/person_tracking/contracts.py",
    ),
    "tracking-migration": (
        "src/tasks/person_tracking/scripts/migrate_features.py",
        "src/tasks/person_tracking/archive.py",
    ),
    "player-publication": (
        "src/tasks/player_detection/generate_dataset/publishing/publisher.py",
        "src/tasks/person_tracking/contracts.py",
        "src/tasks/annotation/migration/legacy_format/manifests.py",
        "third_party/DINO",
    ),
    "github-automation": (".github/dependabot.yml", ".gitmodules"),
    "court-publication": (
        "src/tasks/court_detection/scaling/training.py",
        "src/synthetic_data_generation/dataset/court/scale/recipes.py",
        "third_party/nht",
    ),
    "scene-visualization": (
        "src/tennis_scene/visualization/blender/export/bundle.py",
        "src/tennis_scene/clip_studio/web/app.py",
    ),
    "scene-benchmarks": (
        "tests/benchmarks/pipeline_qualification.py",
        "tests/benchmarks/court_side_clip000_counterfactuals.py",
        "src/tasks/court_side/hypothesis.py",
    ),
    "tracking-benchmarks": (
        "tests/benchmarks/person_unseen.py",
        "tests/benchmarks/association_recalibration_dev.py",
        "src/tasks/player_association/association/config.py",
        "src/tasks/player_detection/evaluation/far_archive.py",
        "src/tennis_scene/pipeline/definition.py",
    ),
}


@pytest.fixture(scope="module")
def repository_lanes() -> tuple[dict[str, Any], list[str]]:
    return cast(tuple[dict[str, Any], list[str]], load_registry(ROOT))


def _plan(
    repository_lanes: tuple[dict[str, Any], list[str]],
    *,
    path: str | None = None,
    event: str = "pull_request",
) -> dict[str, Any]:
    registry, files = repository_lanes
    plan: dict[str, Any] = validate_plan(
        make_plan(
            registry,
            files,
            source_sha="head",
            event=event,
            ref="refs/heads/main" if event == "push" else "refs/pull/1/merge",
            base_sha="base",
            head_sha="head",
            changes=[{"status": "M", "paths": [path]}] if path else [],
        )
    )
    return plan


def _suite_nodes(registry: dict[str, Any], suite_id: str) -> list[str]:
    suite = next(s for s in registry["suites"] if s["id"] == suite_id)
    # New cases in an existing file inherit its lane, without a nodeid allowlist.
    return [f"{f}::{case}" for f in suite["files"] for case in ("test_old", "test_new")]


@pytest.mark.parametrize("suite_id", DEPENDENCY_CHANGES)
def test_registered_dependencies_and_own_tests_request_whole_suite(
    repository_lanes: tuple[dict[str, Any], list[str]], suite_id: str
) -> None:
    registry, _ = repository_lanes
    nodes = _suite_nodes(registry, suite_id)
    own_files = sorted({node.split("::", 1)[0] for node in nodes})
    for path in (*DEPENDENCY_CHANGES[suite_id], *own_files):
        plan = _plan(repository_lanes, path=path)
        assert plan["scope"] == "pr", path
        assert suite_id in plan["requested_suites"], path
        assert partition_nodeids(nodes, plan) == (nodes, {}), path


@pytest.mark.parametrize("suite_id", [*DEPENDENCY_CHANGES, "architecture-boundaries"])
def test_unrelated_pr_defers_each_registered_suite_with_owner_and_reason(
    repository_lanes: tuple[dict[str, Any], list[str]], suite_id: str
) -> None:
    registry, _ = repository_lanes
    plan = _plan(repository_lanes, path="paper/court_robustness/notes.txt")
    nodes = _suite_nodes(registry, suite_id)
    assert plan["scope"] == "pr"
    assert suite_id in plan["deferred_suites"]
    assert partition_nodeids(nodes, plan) == ([], dict.fromkeys(nodes, suite_id))
    suite = next(s for s in plan["registry"]["suites"] if s["id"] == suite_id)
    assert suite["owner"].strip() and suite["reason"].strip()


@pytest.mark.parametrize("suite_id", [*DEPENDENCY_CHANGES, "architecture-boundaries"])
def test_main_push_requests_each_registered_suite_without_changes(
    repository_lanes: tuple[dict[str, Any], list[str]], suite_id: str
) -> None:
    registry, _ = repository_lanes
    nodes = _suite_nodes(registry, suite_id)
    plan = _plan(repository_lanes, event="push")
    assert plan["scope"] == "full"
    assert suite_id in plan["requested_suites"]
    assert partition_nodeids(nodes, plan) == (nodes, {})


def test_registry_keeps_merge_queue_and_unregistered_tests_always_requested(
    repository_lanes: tuple[dict[str, Any], list[str]],
) -> None:
    registry, files = repository_lanes
    assert {suite["id"] for suite in registry["suites"]} == {
        *DEPENDENCY_CHANGES,
        "architecture-boundaries",
    }
    registered = {f for suite in registry["suites"] for f in suite["files"]}
    assert not any(f.startswith("tests/unit/") or "test_knowledge" in f for f in registered)
    protected = {
        "tests/e2e/agents_ops/test_merge_gate.py",
        "tests/e2e/agents_ops/test_review_policy.py",
        "tests/e2e/skills/test_training_queue.py",
        "tests/integration/github/test_local_gpu_actions.py",
        "tests/integration/tasks/test_court_keypoint_contract_smoke.py",
    }
    assert protected <= set(files) - registered
    nodes = [f"{f}::test_contract" for f in files if f not in registered]
    assert partition_nodeids(nodes, _plan(repository_lanes)) == (nodes, {})


@pytest.mark.parametrize(
    "path",
    [
        "src/tasks/ball_detection/models/mdd/detectors.py",
        "tests/e2e/development/test_architecture_boundaries.py",
    ],
)
def test_main_only_architecture_sweep_is_deferred_for_ordinary_prs(
    repository_lanes: tuple[dict[str, Any], list[str]], path: str
) -> None:
    registry, _ = repository_lanes
    nodes = _suite_nodes(registry, "architecture-boundaries")
    plan = _plan(repository_lanes, path=path)
    assert plan["scope"] == "pr"
    assert partition_nodeids(nodes, plan) == (
        [],
        dict.fromkeys(nodes, "architecture-boundaries"),
    )


@pytest.mark.parametrize("path", [REGISTRY, "scripts/ci_metrics/lanes.py"])
def test_this_registry_pr_still_requests_every_suite_on_shared_changes(
    repository_lanes: tuple[dict[str, Any], list[str]], path: str
) -> None:
    registry, files = repository_lanes
    plan = _plan(repository_lanes, path=path)
    assert plan["scope"] == "full"
    assert plan["deferred_suites"] == []
    nodes = [f"{f}::test_contract" for f in files]
    assert partition_nodeids(nodes, plan) == (nodes, {})
    assert set(plan["requested_suites"]) == {suite["id"] for suite in registry["suites"]}


def test_empty_paths_are_an_explicit_main_only_exception() -> None:
    registry = {
        "schema_version": 1,
        "suites": [
            {
                "id": "architecture",
                "files": ["tests/test_architecture.py"],
                "paths": [],
                "reason": "Whole-repository sweep on full runs",
                "owner": "Motoki0705",
            }
        ],
    }
    assert validate_registry(registry, ["tests/test_architecture.py"]) == registry
