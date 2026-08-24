"""Determinism and dependency-resolution tests for query ablation manifests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir

from src.tasks.court_detection.experiments.configuration import QueryAblationConfig
from src.tasks.court_detection.experiments.query_ablation import (
    PHASE_ORDER,
    build_ablation_manifest,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _compose(*overrides: str) -> QueryAblationConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="run_query_ablation", overrides=list(overrides))
    return QueryAblationConfig.from_config(config)


def test_manifest_is_byte_identity_deterministic_and_phase_ordered() -> None:
    first = build_ablation_manifest(_compose())
    second = build_ablation_manifest(_compose())
    runs = cast(list[dict[str, object]], first["runs"])

    assert first == second
    assert first["phase_order"] == list(PHASE_ORDER)
    assert [run["seed"] for run in runs[:2]] == [42, 42]
    assert [run["phase"] for run in runs] == sorted(
        (cast(str, run["phase"]) for run in runs),
        key=PHASE_ORDER.index,
    )


def test_unresolved_dependencies_never_claim_queue_readiness() -> None:
    manifest = build_ablation_manifest(_compose())
    runs = cast(list[dict[str, object]], manifest["runs"])

    assert all(bool(run["queue_ready"]) for run in runs[:2])
    assert all(not bool(run["queue_ready"]) for run in runs[2:])
    assert all(
        "selected_encoder_depth" in cast(list[str], run["unresolved"])
        for run in runs[2:]
    )


def test_depth_one_selection_supports_single_tap_dpt_matrix() -> None:
    runtime = _compose(
        "ablation.selected.encoder_depth=1",
        "ablation.selected.decoder_family=dpt",
        "ablation.selected.decoder_size=tiny",
    )

    manifest = build_ablation_manifest(runtime)
    assert all(bool(run["queue_ready"]) for run in cast(list[dict[str, object]], manifest["runs"]))
