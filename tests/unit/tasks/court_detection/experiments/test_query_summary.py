"""Completeness, selection, Pareto, and artifact tests for query summaries."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.experiments.configuration import QueryAblationConfig
from src.tasks.court_detection.experiments.query_ablation import (
    build_ablation_manifest,
)
from src.tasks.court_detection.experiments.query_summary import (
    RESULTS_SCHEMA,
    summarize_ablation,
    write_summary_artifacts,
)
from src.tasks.court_detection.models.query_encoder.profiling import (
    DECODER_MAC_DEFINITION,
    PROFILE_SCHEMA,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _manifest() -> dict[str, object]:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="run_query_ablation",
            overrides=[
                "ablation.selected.encoder_depth=8",
                "ablation.selected.decoder_family=dpt",
                "ablation.selected.decoder_size=tiny",
            ],
        )
    return cast(
        dict[str, object],
        build_ablation_manifest(QueryAblationConfig.from_config(config)),
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


def _results(manifest: dict[str, object]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for run in cast(list[dict[str, object]], manifest["runs"]):
        architecture = cast(dict[str, object], run["architecture"])
        family = str(architecture["decoder_family"])
        size = str(architecture["decoder_size"])
        if run["phase"] == "encoder_first":
            depth = cast(int, architecture["encoder_depth"])
            kp = {1: 12.0, 2: 10.6, 4: 10.4, 8: 10.0}[depth]
            capacity = depth * 100
        elif run["phase"] == "decoder_second":
            size_index = {"tiny": 0, "small": 1, "base": 2, "large": 3}[size]
            kp = 10.4 - 0.2 * size_index
            if family == "dpt" and size == "base":
                kp = 10.0
            capacity = 100 + 50 * size_index
        else:
            supervision_index = {"kp": 0, "kp+pose": 1, "all": 2, "all+pose": 3}[
                cast(str, run["supervision"])
            ]
            kp = 10.0 - supervision_index * 0.1
            capacity = 100
        supervision = cast(str, run["supervision"])
        metrics: dict[str, float] = {"kp_mean_distance_px": kp}
        if "pose" in supervision:
            metrics.update(
                {
                    "pose_translation_l2_m": kp / 10.0,
                    "pose_rotation_geodesic_deg": kp * 2.0,
                    "pose_focal_relative_error": kp / 100.0,
                    "pose_log_focal_abs_error": kp / 200.0,
                }
            )
        if supervision.startswith("all"):
            metrics.update({"seg_miou": 0.7, "line_iou": 0.8})
        records.append(
            {
                "run_id": run["run_id"],
                "phase": run["phase"],
                "seed": run["seed"],
                "metrics": metrics,
                "profile": _profile(family, size, capacity=capacity),
            }
        )
    return {
        "schema": RESULTS_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "runs": records,
    }


def test_summary_selects_by_frozen_rules_and_writes_all_artifacts(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    summary = summarize_ablation(
        manifest,
        _results(manifest),
        adopted_supervision="all+pose",
        adoption_rationale="Complete evidence selected the multimodal joint target.",
        require_gpu_profiles=True,
    )

    assert cast(dict[str, object], summary["encoder_selection"])["selected_depth"] == 4
    decoder = cast(dict[str, object], summary["decoder_selection"])
    assert (decoder["selected_family"], decoder["selected_size"]) == (
        "dpt",
        "tiny",
    )
    candidates = cast(list[dict[str, object]], decoder["candidates"])
    assert any(candidate["pareto_front"] for candidate in candidates)
    assert sum(bool(candidate["adopted"]) for candidate in candidates) == 1
    outputs = write_summary_artifacts(summary, output_dir=tmp_path)

    assert {path.name for path in outputs} == {
        "summary.json",
        "scaling_table.csv",
        "pareto_table.csv",
        "encoder_scaling.png",
        "decoder_pareto.png",
        "supervision_comparison.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_summary_fails_closed_on_one_missing_seed_run() -> None:
    manifest = _manifest()
    results = _results(manifest)
    cast(list[object], results["runs"]).pop()

    with pytest.raises(ValueError, match="incomplete"):
        summarize_ablation(
            manifest,
            results,
            adopted_supervision="kp+pose",
            adoption_rationale="Fixture decision.",
            require_gpu_profiles=True,
        )


def test_summary_fails_closed_on_missing_metric() -> None:
    manifest = _manifest()
    results = _results(manifest)
    broken = deepcopy(results)
    records = cast(list[dict[str, object]], broken["runs"])
    metrics = cast(dict[str, float], records[0]["metrics"])
    del metrics["pose_rotation_geodesic_deg"]

    with pytest.raises(ValueError, match="metrics must be exactly"):
        summarize_ablation(
            manifest,
            broken,
            adopted_supervision="kp+pose",
            adoption_rationale="Fixture decision.",
            require_gpu_profiles=True,
        )
