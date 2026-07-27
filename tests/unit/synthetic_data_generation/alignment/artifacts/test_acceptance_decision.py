"""Tests for explicit alignment acceptance decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.synthetic_data_generation.alignment.artifacts.acceptance_decision import (
    ALIGNMENT_ACCEPTANCE_DECISION_SCHEMA,
    USER_OVERRIDE_DECISION,
    AlignmentAcceptanceDecision,
    load_alignment_acceptance_decision,
    publish_alignment_acceptance_decision,
    verify_machine_evidence,
)
from src.synthetic_data_generation.scene_contract import ArtifactRef


def _artifact(artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=f"data/{artifact_id}.json",
        sha256="a" * 64,
        size_bytes=123,
    )


def _decision() -> AlignmentAcceptanceDecision:
    return AlignmentAcceptanceDecision(
        schema=ALIGNMENT_ACCEPTANCE_DECISION_SCHEMA,
        decision_id="alignment-user-override-v1",
        created_at_utc="2026-07-25T13:00:00+00:00",
        decision=USER_OVERRIDE_DECISION,
        authority="user",
        reason="Qualitative confirmation for the controlled synthetic pilot.",
        provider_bundle_fingerprint="b" * 64,
        selected_court_cluster="court-0",
        selected_symmetry="positive-court-y",
        machine_validation_status="rejected",
        failed_gates=(
            "distance_weighted_q95",
            "every_group_template_coverage",
        ),
        decision_source=_artifact("user-decision-source"),
        calibration=_artifact("fit-calibration"),
        holdout_validation=_artifact("holdout-validation"),
        git_revision="abc123",
        git_dirty=True,
        command=(
            "python -m "
            "src.synthetic_data_generation.scripts.alignment.finalize_court_alignment"
        ),
        code_sha256="c" * 64,
    )


def _calibration() -> dict[str, object]:
    return {"status": "fit_calibration_passed"}


def _validation() -> dict[str, object]:
    return {
        "status": "rejected",
        "provider": {"bundle_fingerprint": "b" * 64},
        "geometry": {
            "selected_candidate_id": "court-0",
            "selected_symmetry": "positive-court-y",
        },
        "gate_results": {
            "accepted_view_fraction": True,
            "distance_weighted_q95": False,
            "every_group_template_coverage": False,
        },
    }


def test_decision_round_trip_preserves_machine_rejection(tmp_path: Path) -> None:
    decision = _decision()
    verify_machine_evidence(
        decision,
        calibration=_calibration(),
        holdout_validation=_validation(),
    )

    path = publish_alignment_acceptance_decision(decision, output_dir=tmp_path)
    loaded, fingerprint = load_alignment_acceptance_decision(path)

    assert loaded == decision
    assert loaded.machine_validation_status == "rejected"
    assert path.name.endswith(f"{fingerprint[:16]}.json")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_alignment_acceptance_decision(decision, output_dir=tmp_path)


def test_decision_rejects_non_user_authority() -> None:
    payload = _decision().to_dict()
    payload["authority"] = "agent"

    with pytest.raises(ValueError, match="requires user authority"):
        AlignmentAcceptanceDecision.from_dict(payload)


def test_decision_rejects_mismatched_failed_gates() -> None:
    payload = _decision().to_dict()
    payload["failed_gates"] = ["distance_weighted_q95"]
    decision = AlignmentAcceptanceDecision.from_dict(payload)

    with pytest.raises(ValueError, match="differ from machine evidence"):
        verify_machine_evidence(
            decision,
            calibration=_calibration(),
            holdout_validation=_validation(),
        )


def test_decision_detects_tampering(tmp_path: Path) -> None:
    path = publish_alignment_acceptance_decision(
        _decision(),
        output_dir=tmp_path,
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["reason"] = "tampered"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_alignment_acceptance_decision(path)
