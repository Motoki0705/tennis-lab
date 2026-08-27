"""Smoke coverage for Court matrix evidence production and collection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from scripts.training import court_detection_matrix as matrix
from src.tasks.court_detection.data.contracts import CourtTargetKind
from src.tasks.court_detection.geometry.pose import CourtDecodedPose
from src.tasks.court_detection.model_io.contracts import (
    CourtConsistencyResult,
    CourtModelOutput,
    CourtPoseLossKind,
    CourtPoseTrainingResult,
    CourtRawPoseOutput,
)
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _loss_terms(condition: str) -> dict[str, float]:
    terms = {
        name: 1.0 for name in matrix._LOSSES_BY_CONDITION[condition]
    }
    active_branches = matrix._GRADIENT_BRANCHES_BY_CONDITION[condition]
    for branch in active_branches:
        if branch == "pose":
            for pose_name in ("pose_translation", "pose_rotation", "pose_focal"):
                terms.setdefault(f"{pose_name}_direct", 1.0)
                terms[f"{pose_name}_configured_weight"] = 1.0
                terms[f"{pose_name}_effective_weight"] = 1.0
                terms[f"{pose_name}_weighted"] = 1.0
        else:
            terms.setdefault(f"{branch}_direct", 1.0)
            terms[f"{branch}_configured_weight"] = 1.0
            terms[f"{branch}_effective_weight"] = 1.0
            terms[f"{branch}_weighted"] = 1.0
    if condition in {"pure", "weighted"}:
        terms.update(
            {
                "consistency_auxiliary_unweighted": 1.0,
                "consistency_configured_weight": 1.0,
                "consistency_effective_weight": 1.0,
                "consistency_auxiliary_weighted": 1.0,
            }
        )
    return terms


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest_path = tmp_path / "manifest.json"
    manifest = matrix.build_manifest(
        matrix.MatrixOptions(manifest_path=str(manifest_path))
    )
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _complete_queue(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], Path]:
    manifest_path, manifest = _manifest(tmp_path)
    queue_dir = tmp_path / "queue"
    entries = cast(list[dict[str, object]], manifest["entries"])
    manifest_sha256 = cast(str, manifest["manifest_sha256"])
    for index, entry in enumerate(entries):
        run_id = f"run-{index:02d}"
        repro_dir = queue_dir / "repro" / run_id
        prediction_dir = repro_dir / "predictions"
        prediction_dir.mkdir(parents=True)
        condition = cast(str, entry["condition"])
        _write_json(
            repro_dir / "run.json",
            {
                "run_id": run_id,
                "name": entry["queue_name"],
                "command": entry["command"],
                "issue": "790",
            },
        )
        done_marker = queue_dir / "done" / f"{run_id}.job"
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.write_text("done\n", encoding="utf-8")
        np.savez_compressed(prediction_dir / "pred_test.npz", value=np.ones(1))
        _write_json(
            prediction_dir / "metrics.json",
            {
                name: 1.0
                for name in matrix._METRICS_BY_CONDITION[condition]
            },
        )
        gradients = {
            name: True
            for name in matrix._GRADIENT_BRANCHES_BY_CONDITION[condition]
        }
        _write_json(
            repro_dir / "court_matrix_evidence.json",
            {
                "schema": matrix.RUN_EVIDENCE_SCHEMA,
                "phase": matrix.PHASE,
                "manifest_sha256": manifest_sha256,
                "entry_id": entry["entry_id"],
                "complete": True,
                "loss_terms": _loss_terms(condition),
                "diagnostics": {
                    "gradient_finite": gradients,
                    "parameter_count": 1,
                    "train_step_time_ms": 1.0,
                    "peak_memory_bytes": 1,
                },
            },
        )
    return manifest_path, manifest, queue_dir


def test_manifest_covers_standalone_multimodal_and_scaling_protocol(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(tmp_path)

    assert matrix.validate_manifest(manifest) == manifest
    entries = cast(list[dict[str, object]], manifest["entries"])
    condition_entries = {
        cast(str, entry["condition"]): entry
        for entry in entries
        if "condition" in cast(list[str], entry["roles"])
    }
    assert set(condition_entries) == set(matrix.CONDITIONS)
    assert {
        name: entry["processing"] for name, entry in condition_entries.items()
    } == {
        "kp-only": "kp",
        "line-only": "line",
        "seg-only": "seg",
        "pose-only": "kp",
        "pure": "all",
        "weighted": "all",
    }
    pose_only_overrides = set(
        cast(list[str], condition_entries["pose-only"]["overrides"])
    )
    assert {
        "loss=default",
        "loss.kp.weight=0.0",
        "loss.seg.weight=0.0",
        "loss.line.weight=0.0",
        "loss.pose.enabled=true",
        "loss.pose.translation_weight=1.0",
        "loss.pose.rotation_weight=1.0",
        "loss.pose.focal_weight=1.0",
        "loss.consistency.enabled=false",
    } <= pose_only_overrides
    assert "loss=pose_only" not in pose_only_overrides
    assert "model/transformer_encoder=none" in cast(
        list[str], condition_entries["kp-only"]["overrides"]
    )
    scaling = {
        (entry["depth"], entry["input_size"], entry["dpt_size"])
        for entry in entries
        if "scaling" in cast(list[str], entry["roles"])
    }
    assert scaling == {
        (depth, input_size, dpt_size)
        for depth in matrix.DEPTHS
        for input_size in matrix.INPUT_SIZES
        for dpt_size in matrix.DPT_CHANNELS
    }


def test_lightning_producer_writes_collector_compatible_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    entries = cast(list[dict[str, object]], manifest["entries"])
    entry = next(item for item in entries if item["condition"] == "pure")
    repro_dir = tmp_path / "queue/repro/run-pure"
    _write_json(
        repro_dir / "run.json",
        {
            "name": entry["queue_name"],
            "command": entry["command"],
            "issue": "790",
        },
    )
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))

    module = object.__new__(CourtDetectionLightningModule)
    torch.nn.Module.__init__(module)
    module.model = torch.nn.Linear(2, 2)
    module._matrix_manifest_path = manifest_path
    module.model_io = SimpleNamespace(
        pose_loss_config=SimpleNamespace(
            pose=SimpleNamespace(
                translation_weight=1.0,
                rotation_weight=1.0,
                focal_weight=1.0,
            ),
            consistency=SimpleNamespace(weight=1.0),
        )
    )
    one = torch.tensor(1.0)
    dense_losses: dict[CourtTargetKind, torch.Tensor] = {
        name: one for name in ("kp", "seg", "line")
    }
    pose_losses: dict[CourtPoseLossKind, torch.Tensor] = {
        name: one
        for name in ("pose_translation", "pose_rotation", "pose_focal")
    }
    consistency = CourtConsistencyResult(
        coordinate_loss=one,
        cheirality_loss=one,
        auxiliary_loss=torch.tensor(2.0),
        weighted_auxiliary_loss=one,
        configured_weight=torch.tensor(2.0),
        effective_weight=torch.tensor(0.5),
        visible_point_count=one,
        mean_distance_px=one,
        invalid_depth_rate=one,
        dense_points_xy=torch.zeros(1, 14, 2),
        pose_points_xy=torch.zeros(1, 14, 2),
        pose_depth_m=torch.ones(1, 14),
    )
    result = CourtPoseTrainingResult(
        loss=torch.tensor(13.0),
        raw_dense_loss=torch.tensor(3.0),
        direct_dense_loss=torch.tensor(3.0),
        direct_pose_loss=torch.tensor(9.0),
        raw_dense_losses=dense_losses,
        dense_losses=dense_losses,
        dense_configured_weights=dense_losses,
        dense_effective_weights=dense_losses,
        weighted_dense_losses=dense_losses,
        pose_losses=pose_losses,
        weighted_pose_losses={
            "pose_translation": torch.tensor(2.0),
            "pose_rotation": torch.tensor(3.0),
            "pose_focal": torch.tensor(4.0),
        },
        pose_configured_weights={
            "pose_translation": torch.tensor(2.0),
            "pose_rotation": torch.tensor(3.0),
            "pose_focal": torch.tensor(4.0),
        },
        pose_effective_weights={
            "pose_translation": torch.tensor(2.0),
            "pose_rotation": torch.tensor(3.0),
            "pose_focal": torch.tensor(4.0),
        },
        consistency=consistency,
        output=CourtModelOutput(
            dense_logits={"kp": torch.zeros(1, 14, 2, 2)},
            pose=CourtRawPoseOutput(torch.zeros(1, 10)),
        ),
        decoded_pose=CourtDecodedPose(
            translation_m=torch.zeros(1, 3),
            rotation=torch.eye(3).unsqueeze(0),
            focal_px=torch.ones(1),
            log_focal=torch.zeros(1),
        ),
    )
    module._matrix_loss_sums = {}
    module._matrix_loss_counts = {}
    module._record_matrix_loss_result(result)
    module._matrix_active_gradient_branches = frozenset(
        matrix._GRADIENT_BRANCHES_BY_CONDITION["pure"]
    )
    module._matrix_gradient_finite = {
        name: True for name in module._matrix_active_gradient_branches
    }
    module._matrix_step_time_ms = 12.5
    module._matrix_step_count = 1
    module._matrix_peak_memory_bytes = 4096
    loss_terms = _loss_terms("pure")
    loss_terms.update(
        {
            "pose_translation_configured_weight": 2.0,
            "pose_translation_effective_weight": 2.0,
            "pose_translation_weighted": 2.0,
            "pose_rotation_configured_weight": 3.0,
            "pose_rotation_effective_weight": 3.0,
            "pose_rotation_weighted": 3.0,
            "pose_focal_configured_weight": 4.0,
            "pose_focal_effective_weight": 4.0,
            "pose_focal_weighted": 4.0,
            "consistency_auxiliary_unweighted": 2.0,
            "consistency_configured_weight": 2.0,
            "consistency_effective_weight": 0.5,
            "weighted_total": 13.0,
        }
    )

    evidence_path = module._write_matrix_evidence()

    assert evidence_path == repro_dir / "court_matrix_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    validated = matrix._validate_run_evidence(
        evidence,
        entry=entry,
        manifest_sha256=cast(str, manifest["manifest_sha256"]),
        path=evidence_path,
    )
    assert validated["complete"] is True
    assert validated["loss_terms"] == loss_terms
    recorded_terms = cast(dict[str, float], validated["loss_terms"])
    assert (
        recorded_terms["consistency_configured_weight"]
        != recorded_terms["consistency_effective_weight"]
    )
    assert recorded_terms["consistency_auxiliary_weighted"] == (
        recorded_terms["consistency_auxiliary_unweighted"]
        * recorded_terms["consistency_effective_weight"]
    )


def test_collector_accepts_complete_producer_schema_for_all_matrix_runs(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, queue_dir = _complete_queue(tmp_path)

    results = matrix.collect_results(
        manifest_path=manifest_path,
        queue_dir=queue_dir,
    )

    assert results["complete"] is True
    assert len(cast(list[object], results["rows"])) == 21
    summary, flattened = matrix.summarize_results(
        results,
        decision={
            "decision": "inconclusive",
            "rationale": "Synthetic comparison requires human review.",
            "selected_entry_id": None,
        },
    )
    assert summary["complete"] is True
    assert summary["row_count"] == 21
    assert summary["weighted_strategy"] == cast(
        dict[str, object], manifest["protocol"]
    )["weighted_strategy"]
    assert summary["decision_status"] == "reviewed"
    assert len(flattened) == 21
    assert all("parameter_count" in row for row in flattened)


def test_collector_rejects_one_missing_required_metric(tmp_path: Path) -> None:
    manifest_path, manifest, queue_dir = _complete_queue(tmp_path)
    entries = cast(list[dict[str, object]], manifest["entries"])
    first_condition = cast(str, entries[0]["condition"])
    metrics_path = queue_dir / "repro/run-00/predictions/metrics.json"
    metrics = cast(dict[str, object], json.loads(metrics_path.read_text(encoding="utf-8")))
    metrics.pop(next(iter(matrix._METRICS_BY_CONDITION[first_condition])))
    _write_json(metrics_path, metrics)

    with pytest.raises(matrix.MatrixError, match="missing required metrics"):
        matrix.collect_results(
            manifest_path=manifest_path,
            queue_dir=queue_dir,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("phase", "foreign-phase", "old-phase or foreign"),
        ("complete", False, "incomplete run evidence"),
    ],
)
def test_run_evidence_rejects_foreign_or_incomplete_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    _, manifest = _manifest(tmp_path)
    entry = next(
        item
        for item in cast(list[dict[str, object]], manifest["entries"])
        if item["condition"] == "pure"
    )
    evidence = {
        "schema": matrix.RUN_EVIDENCE_SCHEMA,
        "phase": matrix.PHASE,
        "manifest_sha256": manifest["manifest_sha256"],
        "entry_id": entry["entry_id"],
        "complete": True,
        "loss_terms": _loss_terms("pure"),
        "diagnostics": {
            "gradient_finite": {
                name: True
                for name in matrix._GRADIENT_BRANCHES_BY_CONDITION["pure"]
            },
            "parameter_count": 1,
            "train_step_time_ms": 1.0,
            "peak_memory_bytes": 1,
        },
    }
    evidence[field] = value

    with pytest.raises(matrix.MatrixError, match=message):
        matrix._validate_run_evidence(
            evidence,
            entry=entry,
            manifest_sha256=cast(str, manifest["manifest_sha256"]),
            path=tmp_path / "court_matrix_evidence.json",
        )
