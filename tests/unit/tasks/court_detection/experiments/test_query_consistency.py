"""Determinism and staged readiness tests for the Issue #790 manifest."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.experiments.query_consistency import (
    PHASE_ORDER,
    PYTHON_EXECUTABLE,
    SHARED_DATA_ROOT,
    SHARED_EXTERNAL_ASSET_ROOT,
    V3_DERIVED_TARGET_ROOT,
    V3_WORKSPACE_ROOT,
    QueryConsistencyAblationConfig,
    build_query_consistency_manifest,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _compose(*overrides: str) -> QueryConsistencyAblationConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="run_query_consistency_ablation",
            overrides=list(overrides),
        )
    return QueryConsistencyAblationConfig.from_config(config)


def test_manifest_is_deterministic_ordered_and_only_phase_one_is_ready() -> None:
    first = build_query_consistency_manifest(_compose())
    second = build_query_consistency_manifest(_compose())
    runs = cast(list[dict[str, object]], first["runs"])

    assert first == second
    assert first["phase_order"] == list(PHASE_ORDER)
    assert len(runs) == 51
    assert [run["seed"] for run in runs[:6]] == [42, 43, 44, 42, 43, 44]
    assert sum(bool(run["queue_ready"]) for run in runs) == 12
    assert all(run["command_argv"] is None for run in runs[12:])
    assert all(run["profile_command_argv"] is None for run in runs[12:])
    fixed_contract = cast(dict[str, object], first["fixed_contract"])
    assert fixed_contract["data_root"] == SHARED_DATA_ROOT
    assert fixed_contract["external_asset_root"] == SHARED_EXTERNAL_ASSET_ROOT
    assert fixed_contract["derived_target_root"] == V3_DERIVED_TARGET_ROOT
    assert fixed_contract["workspace_root"] == V3_WORKSPACE_ROOT


def test_selection_resolution_exposes_exact_phase_counts_and_formal_order() -> None:
    encoder_manifest = build_query_consistency_manifest(
        _compose("consistency_ablation.selected.encoder_depth=4")
    )
    complete_manifest = build_query_consistency_manifest(
        _compose(
            "consistency_ablation.selected.encoder_depth=4",
            "consistency_ablation.selected.decoder_family=linear",
            "consistency_ablation.selected.decoder_size=tiny",
        )
    )
    encoder_runs = cast(list[dict[str, object]], encoder_manifest["runs"])
    complete_runs = cast(list[dict[str, object]], complete_manifest["runs"])

    assert sum(bool(run["queue_ready"]) for run in encoder_runs) == 39
    assert sum(bool(run["queue_ready"]) for run in complete_runs) == 51
    assert [run["condition"] for run in complete_runs[39::3]] == [
        "direct-all",
        "joint-both",
        "joint-stopgrad-pose",
        "joint-stopgrad-dense",
    ]


def test_queue_ready_argv_fixes_all_targets_auxiliary_and_evidence_paths() -> None:
    manifest = build_query_consistency_manifest(
        _compose(
            "consistency_ablation.selected.encoder_depth=4",
            "consistency_ablation.selected.decoder_family=dpt",
            "consistency_ablation.selected.decoder_size=base",
        )
    )
    runs = cast(list[dict[str, object]], manifest["runs"])
    direct = runs[39]
    joint = runs[42]
    direct_argv = cast(list[str], direct["command_argv"])
    joint_argv = cast(list[str], joint["command_argv"])

    for argv in (direct_argv, joint_argv):
        assert argv[:3] == [
            PYTHON_EXECUTABLE,
            "-m",
            "src.tasks.court_detection.scripts.train",
        ]
        assert "data/processing=all" in argv
        assert f"paths.data_root={SHARED_DATA_ROOT}" in argv
        assert f"paths.external_asset_root={SHARED_EXTERNAL_ASSET_ROOT}" in argv
        assert f"data.source.workspace_root={V3_WORKSPACE_ROOT}" in argv
        assert f"data.processing.derived_target_root={V3_DERIVED_TARGET_ROOT}" in argv
        assert "data/augmentation=pose_safe" in argv
        assert "model.heads.dense_targets=[kp,seg,line]" in argv
        assert "run.test_after_fit=true" in argv
        assert any(value.startswith("run.output_dir=") for value in argv)
        assert any(value.startswith("paths.artifact_root=") for value in argv)
    assert "loss.consistency.weight=0.0" in direct_argv
    assert "loss.consistency.cheirality_weight=0.0" in direct_argv
    assert "loss.consistency.weight=1.0" in joint_argv
    assert "loss.consistency.cheirality_weight=0.1" in joint_argv


def test_depth_one_selection_fails_instead_of_substituting_dpt_architecture() -> None:
    with pytest.raises(ValueError, match="depth 1 cannot resolve"):
        _compose("consistency_ablation.selected.encoder_depth=1")


@pytest.mark.parametrize(
    "workspace_root",
    (
        "/tmp/scenes",
        "../scenes",
        "issue-779/scenes/../other",
        "issue-779//scenes",
        "issue-779/./scenes",
    ),
)
def test_v3_workspace_requires_a_normalized_relative_descendant(
    workspace_root: str,
) -> None:
    with pytest.raises(ValueError, match="normalized relative descendant"):
        _compose(
            f"consistency_ablation.composition.workspace_root={workspace_root}"
        )


def test_manifest_rejects_a_worktree_local_data_root() -> None:
    with pytest.raises(
        ValueError, match="shared declared data root|escapes its declared parent"
    ):
        _compose("paths.data_root=data")


@pytest.mark.parametrize(
    "derived_target_root",
    ("/tmp/targets", "../targets", "court_detection/../targets"),
)
def test_v3_derived_store_requires_a_normalized_relative_descendant(
    derived_target_root: str,
) -> None:
    with pytest.raises(ValueError, match="normalized relative descendant"):
        _compose(
            "consistency_ablation.composition.derived_target_root="
            f"{derived_target_root}"
        )
