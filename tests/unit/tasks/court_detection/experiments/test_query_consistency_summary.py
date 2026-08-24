"""Collector, selection, adoption, and artifact tests for Issue #790."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from torch.utils.tensorboard import SummaryWriter

from src.tasks.court_detection.experiments.query_consistency import (
    RESULT_METRIC_NAMES,
    TRAIN_DIAGNOSTIC_NAMES,
    QueryConsistencyAblationConfig,
    build_query_consistency_manifest,
)
from src.tasks.court_detection.experiments.query_consistency_summary import (
    RESULTS_SCHEMA,
    build_query_consistency_result_record,
    collect_tensorboard_training_evidence,
    summarize_query_consistency,
    write_query_consistency_summary_artifacts,
)
from src.tasks.court_detection.models.query_encoder.profiling import (
    DECODER_MAC_DEFINITION,
    PROFILE_SCHEMA,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _manifest() -> dict[str, object]:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="run_query_consistency_ablation",
            overrides=[
                "consistency_ablation.selected.encoder_depth=8",
                "consistency_ablation.selected.decoder_family=dpt",
                "consistency_ablation.selected.decoder_size=tiny",
            ],
        )
    return cast(
        dict[str, object],
        build_query_consistency_manifest(
            QueryConsistencyAblationConfig.from_config(config)
        ),
    )


def _profile(family: str, size: str, *, capacity: int) -> dict[str, object]:
    return {
        "schema": PROFILE_SCHEMA,
        "candidate": {"family": family, "size": size},
        "evidence": {
            "kind": "gpu_runtime",
            "device_name": "fixture-gpu",
            "latency_is_adoption_evidence": True,
        },
        "execution_contract": {
            "model_mode": "eval",
            "autograd_enabled": False,
            "latency_statistic": "arithmetic_mean_and_population_std_ms",
            "peak_scope": "end_to_end_forward",
        },
        "input_contract": {
            "batch_size": 1,
            "channels": 3,
            "height": 256,
            "width": 256,
            "dtype": "float32",
            "device": "cuda",
        },
        "parameters": {
            "decoder": capacity,
            "trainable": capacity + 100,
            "total": capacity + 200,
        },
        "decoder_macs": {
            "count": capacity * 10,
            "definition": DECODER_MAC_DEFINITION,
        },
        "latency_ms": {
            "warmup": 20,
            "repeats": 100,
            "decoder_mean": capacity / 100.0,
            "decoder_std": 0.01,
            "end_to_end_mean": capacity / 100.0 + 1.0,
            "end_to_end_std": 0.02,
        },
        "peak_memory": {
            "bytes": capacity * 100,
            "method": "cuda.max_memory_allocated",
        },
    }


def _metrics(kp: float, *, condition: str) -> dict[str, float]:
    pose = 9.4 if condition == "joint-both" else kp
    return {
        "kp_mean_distance_px": kp,
        "kp_median_distance_px": kp * 0.8,
        "pose_reprojection_mean_distance_px": pose,
        "pose_translation_l2_m": 1.0,
        "pose_rotation_geodesic_deg": 2.0,
        "pose_focal_relative_error": 0.1,
        "line_dice": 0.8,
        "seg_miou": 0.7,
        "kp_pose_consistency_distance_px": kp * 0.5,
        "invalid_depth_rate": 0.01,
        "visible_point_count": 140.0,
    }


def _diagnostics(*, condition: str) -> dict[str, float]:
    overhead = 1.08 if condition == "joint-both" else 1.0
    return {
        "kp_gradient_finite": 1.0,
        "seg_gradient_finite": 1.0,
        "line_gradient_finite": 1.0,
        "pose_gradient_finite": 1.0,
        "train_step_time_ms": 100.0 * overhead,
        "cuda_peak_memory_bytes": 1000.0 * overhead,
    }


def _results(manifest: dict[str, object]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for run in cast(list[dict[str, object]], manifest["runs"]):
        architecture = cast(dict[str, object], run["architecture"])
        family = cast(str, architecture["decoder_family"])
        size = cast(str, architecture["decoder_size"])
        condition = cast(str, run["condition"])
        if run["phase"] == "encoder_scaling":
            depth = cast(int, architecture["encoder_depth"])
            kp = {1: 11.0, 2: 10.7, 4: 10.4, 8: 10.0}[depth]
            capacity = depth * 100
        elif run["phase"] == "decoder_scaling":
            size_index = {"tiny": 0, "small": 1, "base": 2, "large": 3}[size]
            kp = 10.4 - 0.05 * size_index
            if family == "dpt" and size == "base":
                kp = 10.0
            capacity = 100 + 50 * size_index
        else:
            kp = 9.4 if condition == "joint-both" else 10.0
            capacity = 100
        records.append(
            build_query_consistency_result_record(
                run,
                test_metrics=_metrics(kp, condition=condition),
                diagnostics=_diagnostics(condition=condition),
                loss_curve=[{"step": 0, "loss": 2.0}, {"step": 1, "loss": 1.0}],
                profile=_profile(family, size, capacity=capacity),
                require_gpu_profile=True,
            )
        )
    return {
        "schema": RESULTS_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "phase": "consistency_ablation",
        "runs": records,
    }


def test_summary_applies_frozen_scaling_and_adoption_rules(tmp_path: Path) -> None:
    manifest = _manifest()
    results = _results(manifest)
    summary = summarize_query_consistency(
        manifest,
        results,
        phase="consistency_ablation",
    )

    assert cast(dict[str, object], summary["encoder_selection"])[
        "selected_depth"
    ] == 4
    decoder = cast(dict[str, object], summary["decoder_selection"])
    assert (decoder["selected_family"], decoder["selected_size"]) == (
        "dpt",
        "tiny",
    )
    decision = cast(dict[str, object], summary["adoption_decision"])
    assert decision["status"] == "adopted"
    outputs = write_query_consistency_summary_artifacts(
        summary,
        results,
        output_dir=tmp_path,
    )
    assert {path.name for path in outputs} == {
        "summary.json",
        "all_runs.csv",
        "scaling_table.csv",
        "pareto_table.csv",
        "encoder_scaling.png",
        "decoder_pareto.png",
        "consistency_comparison.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_result_fixture_rejects_line_iou_alias_and_missing_diagnostic() -> None:
    run = cast(list[dict[str, object]], _manifest()["runs"])[0]
    architecture = cast(dict[str, object], run["architecture"])
    metrics = _metrics(10.0, condition="joint-both")
    metrics["line_iou"] = 0.8
    profile = _profile(
        cast(str, architecture["decoder_family"]),
        cast(str, architecture["decoder_size"]),
        capacity=100,
    )

    with pytest.raises(ValueError, match="line_iou"):
        build_query_consistency_result_record(
            run,
            test_metrics=metrics,
            diagnostics=_diagnostics(condition="joint-both"),
            loss_curve=[{"step": 0, "loss": 1.0}],
            profile=profile,
            require_gpu_profile=True,
        )
    diagnostics = _diagnostics(condition="joint-both")
    del diagnostics["pose_gradient_finite"]
    with pytest.raises(ValueError, match="every exact"):
        build_query_consistency_result_record(
            run,
            test_metrics=_metrics(10.0, condition="joint-both"),
            diagnostics=diagnostics,
            loss_curve=[{"step": 0, "loss": 1.0}],
            profile=profile,
            require_gpu_profile=True,
        )


def test_consistency_only_improvement_is_explicit_non_adoption() -> None:
    manifest = _manifest()
    results = _results(manifest)
    for record in cast(list[dict[str, object]], results["runs"]):
        if record["condition"] != "joint-both" or record["phase"] != "consistency_ablation":
            continue
        metrics = cast(dict[str, float], record["metrics"])
        metrics["kp_mean_distance_px"] = 10.0
        metrics["pose_reprojection_mean_distance_px"] = 10.0
        metrics["kp_pose_consistency_distance_px"] = 4.0

    summary = summarize_query_consistency(
        manifest,
        results,
        phase="consistency_ablation",
    )
    decision = cast(dict[str, object], summary["adoption_decision"])

    assert decision["status"] == "not_adopted"
    assert decision["consistency_only_improvement"] is True
    assert "gt_improvement" in cast(list[str], decision["failed_requirements"])


def test_tensorboard_collector_uses_explicit_reductions(tmp_path: Path) -> None:
    writer = SummaryWriter(log_dir=tmp_path)
    for name in TRAIN_DIAGNOSTIC_NAMES[:4]:
        writer.add_scalar(f"train/{name}", 1.0, 0)
        writer.add_scalar(f"train/{name}", 1.0, 1)
    writer.add_scalar("train/train_step_time_ms", 90.0, 0)
    writer.add_scalar("train/train_step_time_ms", 110.0, 1)
    writer.add_scalar("train/cuda_peak_memory_bytes", 1000.0, 0)
    writer.add_scalar("train/cuda_peak_memory_bytes", 1200.0, 1)
    writer.add_scalar("train/loss", 2.0, 0)
    writer.add_scalar("train/loss", 1.0, 1)
    writer.close()

    diagnostics, loss_curve = collect_tensorboard_training_evidence(tmp_path)

    assert tuple(diagnostics) == TRAIN_DIAGNOSTIC_NAMES
    assert diagnostics["train_step_time_ms"] == 100.0
    assert diagnostics["cuda_peak_memory_bytes"] == 1200.0
    assert loss_curve == [{"step": 0, "loss": 2.0}, {"step": 1, "loss": 1.0}]
    assert tuple(_metrics(10.0, condition="joint-both")) == RESULT_METRIC_NAMES
