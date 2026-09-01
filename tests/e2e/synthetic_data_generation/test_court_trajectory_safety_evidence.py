from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from src.synthetic_data_generation.scripts.evaluate_court_trajectory_safety import (
    _load_frozen_benchmark_authority,
    _load_pilot_manifest,
)
from src.utils.paths import PROJECT_ROOT

_ROOT = PROJECT_ROOT / "experiments/synthetic_data_generation/court_trajectory_safety"
_MANIFEST_SHA256 = "6d65ff89729a866491d97f62dff8c76650d0dff793544baf26533c168a75dffa"
_PILOT_FEATURES_SHA256 = (
    "3d31139f8ea97e90e7d13b659038f466641ab4ebc5360fb12b6c78708d03ea81"
)
_BLIND_REVIEW_MANIFEST_SHA256 = (
    "79c7f35b45be82ceab3b732db53c841078c4393544dab317265a5cdb93140f43"
)
_REVIEWER_A_SHA256 = (
    "4eae289a094586b613fa6d7d29fe09fb8da00b8a14bc5e164c7ed5a072dad079"
)
_REVIEWER_B_SHA256 = (
    "8e774f559ca5bc831a71e26f350af003f38530535ac671c12200a3f612dc54ef"
)
_ADJUDICATION_SHA256 = (
    "5685dc2b0b84cef434bd5a597209e399378facbb9a60f70d1ddd8355eab0d58b"
)
_RGB_PREVIEW_INVENTORY_SHA256 = (
    "7d13309ad19ee31d5e8a3f16889137affbb7275d7e046c70a3f99c455341898d"
)
_CONSENSUS_SHA256 = (
    "feddf40c03de40eabadbcbcbda45ad4a931bd9c34fd5064e3670623e12d44836"
)
_QUALITY_DECISION_SHA256 = (
    "551787b8ffcf78437e21c09cd50b3087f57b6354a5aec518140fdb7e4ddaa776"
)
_INVENTORY_SHA256 = "4280156d286359e34f020701ee67df269741c60a91815964378b76f17bd4b839"
_FINAL_EVIDENCE_ROOT = "outputs/court_trajectory_safety/B00-required-coverage-final"
_FINAL_DATASET_SHA256 = (
    "7eb953056cb417671dd9f5d1066aa5b61e13b1e5ecec60953f73553bba193ad1"
)
_COMPACT_EVIDENCE_SHA256 = (
    "72e8417ae2c7621eee01e0b4d2ee8173cc5b26fb4f67cd146428ab744f40882a"
)
_CANONICAL_EVIDENCE_SHA256 = (
    "7db8e4c4727f13f65789d288358596dd708f2534931b5b24a69926f330aa97d3"
)
_SUMMARY_FILE_SHA256 = (
    "da90393ecc454e09afdd0736e1b61c9636fb225611c2f675d75657bd8b39356e"
)
_REPORT_FILE_SHA256 = (
    "18fcb6645e122e1decdd96e02b7f47c2cfac86aa3b2fe6b440cce0af6238149a"
)


def _expected_required_coverage() -> dict[str, object]:
    return {
        "constructors": ["free_space_cycle", "anchored_rounded_rectangle"],
        "path_families": ["rounded_rectangle"],
        "vertical_profiles": ["planar", "raised_phases"],
        "target_modes": ["court_center"],
        "minimum_total_groups": 24,
        "minimum_free_space_cycle_groups": 12,
        "minimum_anchored_rounded_rectangle_groups": 6,
        "minimum_unique_anchors": 6,
        "minimum_anchored_planar_groups": 3,
        "minimum_anchored_raised_groups": 3,
        "required_raised_lift_m": 0.25,
        "minimum_anchored_frame_share": 0.08,
    }


def _expected_selected_coverage() -> dict[str, object]:
    anchor_indices = [
        1,
        6,
        9,
        10,
        11,
        12,
        13,
        15,
        16,
        17,
        18,
        20,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        31,
        80,
        83,
        89,
        94,
        95,
        98,
        101,
    ]
    return {
        "total_group_count": 39,
        "total_frame_count": 2_016,
        "constructors": ["anchored_rounded_rectangle", "free_space_cycle"],
        "constructor_group_counts": {
            "anchored_rounded_rectangle": 27,
            "free_space_cycle": 12,
        },
        "constructor_frame_counts": {
            "anchored_rounded_rectangle": 648,
            "free_space_cycle": 1_368,
        },
        "path_families": ["free_space_cycle", "rounded_rectangle"],
        "family_group_counts": {"free_space_cycle": 12, "rounded_rectangle": 27},
        "family_frame_counts": {
            "free_space_cycle": 1_368,
            "rounded_rectangle": 648,
        },
        "vertical_profiles": ["free_space_cycle", "planar", "raised_phases"],
        "profile_group_counts": {
            "free_space_cycle": 12,
            "planar": 13,
            "raised_phases": 14,
        },
        "profile_frame_counts": {
            "free_space_cycle": 1_368,
            "planar": 312,
            "raised_phases": 336,
        },
        "target_modes": ["court_center"],
        "target_group_counts": {"court_center": 39},
        "target_frame_counts": {"court_center": 2_016},
        "anchor_camera_indices": anchor_indices,
        "anchor_camera_ids": [f"frame_{index:06d}" for index in anchor_indices],
        "unique_anchor_count": 27,
        "anchored_group_count": 27,
        "anchored_frame_count": 648,
        "anchored_frame_share": 648 / 2_016,
        "anchored_planar_group_count": 13,
        "anchored_raised_group_count": 14,
        "anchored_required_lift_group_count": 14,
    }


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Tracked evidence must be a mapping: {path}")
    return value


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_b00_attempt2_cpu_manifest_is_fresh_and_required_coverage_complete() -> None:
    frozen_path = _ROOT / "frozen-config.json"
    authority = _load_frozen_benchmark_authority(
        frozen_path,
        allow_unfrozen_observation_lock=True,
    )
    assert authority.observation_lock is not None
    lock = authority.observation_lock
    assert lock.pilot_manifest_sha256 == _MANIFEST_SHA256
    assert lock.pilot_features_sha256 == _PILOT_FEATURES_SHA256
    assert lock.blind_review_manifest_sha256 == _BLIND_REVIEW_MANIFEST_SHA256
    assert lock.reviewer_a_sha256 == _REVIEWER_A_SHA256
    assert lock.reviewer_b_sha256 == _REVIEWER_B_SHA256
    assert lock.adjudication_sha256 == _ADJUDICATION_SHA256
    assert lock.rgb_preview_inventory_sha256 == _RGB_PREVIEW_INVENTORY_SHA256
    pilot_path = _ROOT / "pilot-manifest.json"
    assert hashlib.sha256(pilot_path.read_bytes()).hexdigest() == _MANIFEST_SHA256
    entries = _load_pilot_manifest(
        pilot_path,
        expected_scene_id=authority.scene_id,
        expected_seed=authority.pilot_seed,
        minimum_view_count=authority.minimum_pilot_views,
    )
    pilot = _json(pilot_path)
    inputs = _mapping(pilot["decision_inputs"], name="pilot decision inputs")
    required = _mapping(inputs["required_coverage"], name="required coverage")
    selected = _mapping(inputs["selected_coverage"], name="selected coverage")
    assert required == _expected_required_coverage()
    assert selected == _expected_selected_coverage()
    assert len(entries) == 128
    assert inputs["legacy_proposal_budget"] == 4_800
    assert inputs["candidate_proposal_budget"] == 4_800
    assert inputs["equal_proposal_budget"] is True
    assert inputs["required_coverage_shortfall"] == []
    assert required["minimum_total_groups"] == 24
    assert required["minimum_free_space_cycle_groups"] == 12
    assert required["minimum_anchored_rounded_rectangle_groups"] == 6
    assert required["minimum_unique_anchors"] == 6
    assert required["minimum_anchored_planar_groups"] == 3
    assert required["minimum_anchored_raised_groups"] == 3
    assert required["required_raised_lift_m"] == 0.25
    assert required["minimum_anchored_frame_share"] == 0.08
    assert selected["total_group_count"] == inputs["selected_group_count"] == 39
    assert selected["total_frame_count"] == inputs["planned_frame_count"] == 2_016
    assert selected["constructor_group_counts"] == {
        "anchored_rounded_rectangle": 27,
        "free_space_cycle": 12,
    }
    assert selected["unique_anchor_count"] == 27
    assert selected["anchored_planar_group_count"] == 13
    assert selected["anchored_raised_group_count"] == 14
    assert selected["anchored_required_lift_group_count"] == 14
    assert selected["anchored_frame_share"] == pytest.approx(648 / 2_016)
    assert inputs["semantic_phase_inventory_digest"] == _INVENTORY_SHA256

    summary = _json(_ROOT / "summary.json")
    geometry = _mapping(summary["geometry_metrics"], name="geometry metrics")
    assert summary["status"] == "complete"
    assert summary["decision"] == "quality_only_rejected"
    assert summary["production_authority"] == "geometry_only"
    assert geometry["required_coverage"] == required
    assert geometry["selected_coverage"] == selected
    assert geometry["required_coverage_shortfall"] == []
    assert geometry["final_release_status"] == "passed"
    assert geometry["accepted_frame_count"] == 2_016


def test_b00_complete_evidence_is_hash_locked_and_canonical() -> None:
    bundle = tuple(sorted(_ROOT.iterdir()))
    assert {path.name for path in bundle} == {
        "README.md",
        "frozen-config.json",
        "pilot-manifest.json",
        "report.md",
        "summary.json",
    }
    assert all(path.is_file() and not path.is_symlink() for path in bundle)

    frozen_path = _ROOT / "frozen-config.json"
    authority = _load_frozen_benchmark_authority(
        frozen_path,
        allow_unfrozen_observation_lock=True,
    )
    assert authority.observation_lock is not None
    lock = authority.observation_lock
    pilot_path = _ROOT / "pilot-manifest.json"
    assert hashlib.sha256(pilot_path.read_bytes()).hexdigest() == (
        lock.pilot_manifest_sha256
    )
    pilot = _json(pilot_path)
    inputs = _mapping(pilot["decision_inputs"], name="pilot decision inputs")
    summary = _json(_ROOT / "summary.json")
    assert hashlib.sha256((_ROOT / "summary.json").read_bytes()).hexdigest() == (
        _SUMMARY_FILE_SHA256
    )
    assert hashlib.sha256((_ROOT / "report.md").read_bytes()).hexdigest() == (
        _REPORT_FILE_SHA256
    )
    assert summary["schema"] == "court_trajectory_safety_evidence_v2"
    assert summary["status"] == "complete"
    assert summary["scene_id"] == "B00"
    assert summary["decision"] == "quality_only_rejected"
    assert summary["production_authority"] == "geometry_only"

    pilot_summary = _mapping(summary["pilot"], name="pilot summary")
    assert pilot_summary["record_count"] == 128
    assert pilot_summary["manifest_sha256"] == _MANIFEST_SHA256
    assert pilot_summary["features_sha256"] == _PILOT_FEATURES_SHA256
    assert (
        pilot_summary["blind_review_manifest_sha256"]
        == _BLIND_REVIEW_MANIFEST_SHA256
    )
    assert pilot_summary["stratum_counts"] == {
        "captured_control": 22,
        "legacy_orbit": 22,
        "safe_v4_candidate": 21,
        "support_boundary": 21,
        "support_exterior": 21,
        "support_interior": 21,
    }
    assert pilot_summary["calibration_record_count"] == 66
    assert pilot_summary["held_out_record_count"] == 62
    assert pilot_summary["calibration_group_count"] == 44
    assert pilot_summary["held_out_group_count"] == 44

    annotations = _mapping(summary["annotations"], name="annotations")
    assert annotations == {
        "reviewer_a": {
            "reviewer_id": "attempt2-reviewer-a",
            "record_count": 128,
            "positive_count": 11,
            "sha256": _REVIEWER_A_SHA256,
        },
        "reviewer_b": {
            "reviewer_id": "attempt2-reviewer-b",
            "record_count": 128,
            "positive_count": 3,
            "sha256": _REVIEWER_B_SHA256,
        },
        "adjudication": {
            "reviewer_id": "attempt2-adjudicator",
            "record_count": 10,
            "sha256": _ADJUDICATION_SHA256,
        },
        "disagreement_count": 10,
        "consensus_positive_count": 5,
        "calibration_positive_count": 2,
        "held_out_positive_count": 3,
        "label_inventory": {
            "calibration": {
                "positive_count": 2,
                "negative_count": 64,
                "record_count": 66,
            },
            "held_out": {
                "positive_count": 3,
                "negative_count": 59,
                "record_count": 62,
            },
        },
        "consensus_sha256": _CONSENSUS_SHA256,
    }

    quality_calibration = _mapping(
        summary["quality_calibration"], name="quality calibration"
    )
    assert quality_calibration == {
        "selection": "best_frozen_gate_passing_adjacent_midpoint_v1",
        "status": "no_calibration_threshold_family_passes_frozen_gates",
        "evaluated_candidate_count": 800,
        "eligible_feature_names": [],
        "rule": None,
        "predictive_metrics": None,
        "threshold_bounds": None,
    }
    quality_decision = _mapping(summary["quality_decision"], name="quality decision")
    assert quality_decision["schema"] == "court_trajectory_quality_decision_v2"
    assert quality_decision["feature_definition_id"] == (
        "court_public_quality_features_v1"
    )
    assert quality_decision["thresholds"] == {
        "minimum_recall": 0.9,
        "minimum_precision": 0.8,
        "maximum_valid_control_false_positive_rate": 0.1,
        "minimum_positive_labels": 12,
        "minimum_negative_labels": 12,
    }
    assert quality_decision["rule"] is None
    assert quality_decision["predictive_metrics"] is None
    assert quality_decision["decision"] == "quality_only_rejected"
    assert quality_decision["failure_reasons"] == [
        "no_calibration_threshold_family_passes_frozen_gates",
        "insufficient_held_out_positive_labels",
    ]
    assert quality_decision["calibration_group_ids"] == pilot_summary[
        "calibration_group_ids"
    ]
    assert quality_decision["held_out_group_ids"] == pilot_summary[
        "held_out_group_ids"
    ]
    assert quality_decision["production_authority"] == "geometry_only"
    assert quality_decision["decision_sha256"] == _QUALITY_DECISION_SHA256

    geometry = _mapping(summary["geometry_metrics"], name="geometry metrics")
    for key in (
        "legacy_proposal_budget",
        "candidate_proposal_budget",
        "equal_proposal_budget",
        "required_coverage",
        "selected_coverage",
        "required_coverage_shortfall",
        "optional_candidate_coverage_shortfall",
    ):
        assert geometry[key] == inputs[key]
    assert geometry["required_coverage"] == _expected_required_coverage()
    assert geometry["selected_coverage"] == _expected_selected_coverage()
    assert geometry["planned_frame_count"] == 2_016
    assert geometry["trajectory_group_count"] == 39
    assert geometry["projected_semantic_valid_frame_count"] == 2_016
    assert geometry["projected_semantic_rejected_frame_count"] == 0
    assert geometry["split_group_counts"] == {
        "test": 4,
        "train": 31,
        "validation": 4,
    }
    assert geometry["group_disjoint_splits"] is True
    assert geometry["selected_support_violation_count"] == 0
    assert geometry["accepted_frame_count"] == 2_016

    assert summary["artifact_comparison"] == {
        "split": "held_out",
        "candidate": {
            "stratum": "safe_v4_candidate",
            "artifact_heavy_count": 0,
            "record_count": 9,
            "artifact_heavy_rate": 0.0,
        },
        "legacy": {
            "stratum": "legacy_orbit",
            "artifact_heavy_count": 2,
            "record_count": 12,
            "artifact_heavy_rate": 2 / 12,
        },
        "candidate_to_legacy_rate_ratio": 0.0,
        "maximum_allowed_ratio": 0.5,
        "passed": True,
    }

    final_dataset = _mapping(summary["final_dataset"], name="final dataset")
    assert final_dataset == {
        "schema": "canonical_court_dataset_v4",
        "status": "complete",
        "path": _FINAL_EVIDENCE_ROOT,
        "proposal_count": 2_016,
        "accepted_frame_count": 2_016,
        "rejected_frame_count": 0,
        "accepted_fraction": 1.0,
        "trajectory_group_count": 39,
        "resolved_shard_count": 8,
        "split_leakage_count": 0,
        "selected_support_violation_count": 0,
        "group_disjoint_splits": True,
        "renderer_error_count": 0,
        "renderer_invocation_count": 8,
        "dataset_manifest_sha256": _FINAL_DATASET_SHA256,
        "compact_evidence_sha256": _COMPACT_EVIDENCE_SHA256,
        "compact_evidence_file_count": 12,
    }
    assert summary["canonical_evidence_sha256"] == _CANONICAL_EVIDENCE_SHA256
    _validate_complete_evidence(summary, frozen=_json(frozen_path))
    sheets = summary["representative_contact_sheets"]
    assert sheets == [
        {
            "kind": "consensus_artifact_heavy",
            "path": (
                "outputs/court_trajectory_safety/issue-823-required-coverage-blind/"
                "evidence/contact-sheets/consensus_artifact_heavy.png"
            ),
            "sha256": (
                "74814410a9615e1185f87933a31ede90c0ee404447f49d16c19bbee8adf4092b"
            ),
            "image_count": 5,
            "mode": "RGB",
            "width": 1_280,
            "height": 400,
        },
        {
            "kind": "non_artifact_controls",
            "path": (
                "outputs/court_trajectory_safety/issue-823-required-coverage-blind/"
                "evidence/contact-sheets/non_artifact_controls.png"
            ),
            "sha256": (
                "505fcf727fff8551abfd7063242ac82c6993a794d8a7ac1a4c40c168cf966d6e"
            ),
            "image_count": 22,
            "mode": "RGB",
            "width": 1_280,
            "height": 1_200,
        },
        {
            "kind": "disagreements_adjudication",
            "path": (
                "outputs/court_trajectory_safety/issue-823-required-coverage-blind/"
                "evidence/contact-sheets/disagreements_adjudication.png"
            ),
            "sha256": (
                "f8fc72b3b7cedb423786ae9534f0b4fb0dad504f474f7f51b8e2d4f4401303d0"
            ),
            "image_count": 10,
            "mode": "RGB",
            "width": 1_280,
            "height": 600,
        },
    ]


def test_tracked_attempt2_evidence_does_not_reuse_prior_observations() -> None:
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(_ROOT.iterdir())
        if path.suffix in {".json", ".md"}
    )
    for stale_value in (
        "quality_evidence_complete_final_v4_pending",
        "pending_final_gpu_dataset",
        "final V4 GPU dataset is still pending",
        "V4 remains non-default pending",
        "## Pending final GPU dataset",
        "1fcc53f13cdbd4ddc29a93fb357020ea32e04c08b0cec784a8fc91f18823939b",
        "bd483259f44be4c5332ff11719901a77cb9a8ec473f97c7009c6a810318e0ffe",
        "c3abacc95b01d71a7a2de932076d9a8cf4023b1c86451bea4b04dd9d367185c6",
        "8637f0faa494c9ad04cb2bb1c47250799a420a6e921b32daa13ce5b38970eb5e",
    ):
        assert stale_value not in serialized


def _complete_summary_fixture() -> dict[str, object]:
    required = _expected_required_coverage()
    selected = _expected_selected_coverage()
    summary: dict[str, object] = {
        "status": "complete",
        "geometry_metrics": {
            "final_release_status": "passed",
            "legacy_proposal_budget": 4_800,
            "candidate_proposal_budget": 4_800,
            "equal_proposal_budget": True,
            "required_coverage": required,
            "selected_coverage": selected,
            "required_coverage_shortfall": [],
            "optional_candidate_coverage_shortfall": [],
            "planned_frame_count": 2_016,
            "accepted_frame_count": 2_016,
            "projected_semantic_valid_frame_count": 2_016,
            "projected_semantic_rejected_frame_count": 0,
            "trajectory_group_count": 39,
            "split_group_counts": {"train": 31, "validation": 4, "test": 4},
            "group_disjoint_splits": True,
            "selected_support_violation_count": 0,
        },
        "final_dataset": {
            "schema": "canonical_court_dataset_v4",
            "status": "complete",
            "path": _FINAL_EVIDENCE_ROOT,
            "proposal_count": 2_016,
            "accepted_frame_count": 2_016,
            "rejected_frame_count": 0,
            "accepted_fraction": 1.0,
            "trajectory_group_count": 39,
            "resolved_shard_count": 8,
            "split_leakage_count": 0,
            "selected_support_violation_count": 0,
            "group_disjoint_splits": True,
            "renderer_error_count": 0,
            "renderer_invocation_count": 8,
            "dataset_manifest_sha256": _FINAL_DATASET_SHA256,
            "compact_evidence_sha256": _COMPACT_EVIDENCE_SHA256,
            "compact_evidence_file_count": 12,
        },
        "artifact_comparison": {
            "candidate": {
                "artifact_heavy_count": 0,
                "record_count": 10,
                "artifact_heavy_rate": 0.0,
            },
            "legacy": {
                "artifact_heavy_count": 5,
                "record_count": 10,
                "artifact_heavy_rate": 0.5,
            },
            "candidate_to_legacy_rate_ratio": 0.0,
            "maximum_allowed_ratio": 0.5,
            "passed": True,
        },
    }
    summary["canonical_evidence_sha256"] = _canonical_sha256(summary)
    return summary


@pytest.mark.parametrize(
    ("path", "mutated_value"),
    (
        (("status",), "pending"),
        (("geometry_metrics", "final_release_status"), "pending"),
        (("geometry_metrics", "equal_proposal_budget"), False),
        (("geometry_metrics", "candidate_proposal_budget"), 4_799),
        (("geometry_metrics", "required_coverage_shortfall"), ["missing"]),
        (("geometry_metrics", "selected_coverage", "unique_anchor_count"), 5),
        (("geometry_metrics", "selected_coverage", "anchored_frame_share"), 0.07),
        (("final_dataset", "status"), "pending"),
        (("final_dataset", "proposal_count"), 1_999),
        (("final_dataset", "accepted_frame_count"), 1_900),
        (("final_dataset", "rejected_frame_count"), 8),
        (("final_dataset", "accepted_fraction"), 0.99),
        (("final_dataset", "trajectory_group_count"), 23),
        (("final_dataset", "resolved_shard_count"), 0),
        (("final_dataset", "split_leakage_count"), 1),
        (("final_dataset", "selected_support_violation_count"), 1),
        (("final_dataset", "group_disjoint_splits"), False),
        (("final_dataset", "renderer_error_count"), 1),
        (("final_dataset", "renderer_invocation_count"), 7),
        (("final_dataset", "dataset_manifest_sha256"), "a" * 64),
        (("final_dataset", "compact_evidence_sha256"), "b" * 64),
        (("artifact_comparison", "candidate_to_legacy_rate_ratio"), 0.75),
        (("artifact_comparison", "passed"), False),
    ),
)
def test_complete_evidence_release_gates_fail_closed_on_tampering(
    path: tuple[str, ...],
    mutated_value: object,
) -> None:
    frozen = _json(_ROOT / "frozen-config.json")
    baseline = _complete_summary_fixture()
    _validate_complete_evidence(baseline, frozen=frozen)
    summary = copy.deepcopy(baseline)
    cursor = summary
    for key in path[:-1]:
        cursor = _mapping(cursor[key], name="tamper target")
    cursor[path[-1]] = mutated_value
    canonical = dict(summary)
    canonical.pop("canonical_evidence_sha256")
    summary["canonical_evidence_sha256"] = _canonical_sha256(canonical)

    with pytest.raises(AssertionError):
        _validate_complete_evidence(summary, frozen=frozen)


def _validate_complete_evidence(
    summary: dict[str, object],
    *,
    frozen: dict[str, object],
) -> None:
    assert summary["status"] == "complete"
    geometry = _mapping(summary["geometry_metrics"], name="geometry metrics")
    final_dataset = _mapping(summary["final_dataset"], name="final dataset")
    gates = _mapping(frozen["geometry_release_gates"], name="geometry gates")
    comparison = _mapping(summary["artifact_comparison"], name="artifact comparison")
    candidate = _mapping(comparison["candidate"], name="candidate comparison")
    legacy = _mapping(comparison["legacy"], name="legacy comparison")
    required = _mapping(geometry["required_coverage"], name="required coverage")
    selected = _mapping(geometry["selected_coverage"], name="selected coverage")
    constructor_groups = _mapping(
        selected["constructor_group_counts"],
        name="constructor group counts",
    )
    split_group_counts = _mapping(
        geometry["split_group_counts"], name="split group counts"
    )

    minimum_frames = _integer(gates["minimum_frames"], name="minimum frames")
    maximum_frames = _integer(gates["maximum_frames"], name="maximum frames")
    minimum_accepted_fraction = _number(
        gates["minimum_accepted_fraction"], name="minimum accepted fraction"
    )
    minimum_groups = _integer(
        gates["minimum_trajectory_groups"], name="minimum trajectory groups"
    )
    proposal_count = _integer(
        final_dataset["proposal_count"], name="proposal count"
    )
    accepted_count = _integer(
        final_dataset["accepted_frame_count"], name="accepted frame count"
    )
    rejected_count = _integer(
        final_dataset["rejected_frame_count"], name="rejected frame count"
    )
    accepted_fraction = _number(
        final_dataset["accepted_fraction"], name="accepted fraction"
    )
    trajectory_group_count = _integer(
        final_dataset["trajectory_group_count"], name="trajectory group count"
    )
    resolved_shard_count = _integer(
        final_dataset["resolved_shard_count"], name="resolved shard count"
    )

    assert geometry["final_release_status"] == "passed"
    assert geometry["legacy_proposal_budget"] == 4_800
    assert geometry["candidate_proposal_budget"] == 4_800
    assert geometry["equal_proposal_budget"] is True
    assert geometry["required_coverage_shortfall"] == []
    assert required == _expected_required_coverage()
    assert selected == _expected_selected_coverage()
    assert _integer(selected["total_group_count"], name="selected groups") >= _integer(
        required["minimum_total_groups"], name="minimum total groups"
    )
    assert _integer(
        constructor_groups["free_space_cycle"], name="free-space groups"
    ) >= _integer(
        required["minimum_free_space_cycle_groups"],
        name="minimum free-space groups",
    )
    assert _integer(
        constructor_groups["anchored_rounded_rectangle"], name="anchored groups"
    ) >= _integer(
        required["minimum_anchored_rounded_rectangle_groups"],
        name="minimum anchored groups",
    )
    assert _integer(selected["unique_anchor_count"], name="unique anchors") >= _integer(
        required["minimum_unique_anchors"], name="minimum unique anchors"
    )
    assert _integer(
        selected["anchored_planar_group_count"], name="anchored planar groups"
    ) >= _integer(
        required["minimum_anchored_planar_groups"],
        name="minimum anchored planar groups",
    )
    assert _integer(
        selected["anchored_raised_group_count"], name="anchored raised groups"
    ) >= _integer(
        required["minimum_anchored_raised_groups"],
        name="minimum anchored raised groups",
    )
    assert _integer(
        selected["anchored_required_lift_group_count"],
        name="required-lift groups",
    ) >= _integer(
        required["minimum_anchored_raised_groups"],
        name="minimum required-lift groups",
    )
    assert _number(
        selected["anchored_frame_share"], name="anchored frame share"
    ) >= _number(
        required["minimum_anchored_frame_share"],
        name="minimum anchored frame share",
    )

    assert final_dataset["status"] == "complete"
    assert final_dataset["schema"] == "canonical_court_dataset_v4"
    assert final_dataset["path"] == _FINAL_EVIDENCE_ROOT
    assert minimum_frames <= proposal_count <= maximum_frames
    assert minimum_frames <= accepted_count <= maximum_frames
    assert accepted_count >= math.ceil(minimum_accepted_fraction * proposal_count)
    assert accepted_count + rejected_count == proposal_count
    assert accepted_fraction == pytest.approx(accepted_count / proposal_count)
    assert accepted_fraction >= minimum_accepted_fraction
    assert proposal_count == geometry["planned_frame_count"]
    assert proposal_count == selected["total_frame_count"]
    assert accepted_count == geometry["accepted_frame_count"]
    assert accepted_count == geometry["projected_semantic_valid_frame_count"]
    assert rejected_count == geometry["projected_semantic_rejected_frame_count"]
    assert trajectory_group_count == geometry["trajectory_group_count"]
    assert trajectory_group_count == selected["total_group_count"]
    assert trajectory_group_count >= minimum_groups
    assert sum(
        _integer(count, name="split group count")
        for count in split_group_counts.values()
    ) == trajectory_group_count
    assert resolved_shard_count > 0
    assert final_dataset["renderer_invocation_count"] == resolved_shard_count
    assert final_dataset["renderer_error_count"] == 0
    assert final_dataset["split_leakage_count"] == 0
    assert final_dataset["group_disjoint_splits"] is gates[
        "require_group_disjoint_splits"
    ]
    assert geometry["group_disjoint_splits"] is final_dataset[
        "group_disjoint_splits"
    ]
    assert final_dataset["selected_support_violation_count"] == gates[
        "required_selected_support_violations"
    ]
    assert geometry["selected_support_violation_count"] == final_dataset[
        "selected_support_violation_count"
    ]
    assert final_dataset["dataset_manifest_sha256"] == _FINAL_DATASET_SHA256
    assert final_dataset["compact_evidence_sha256"] == _COMPACT_EVIDENCE_SHA256
    assert _integer(
        final_dataset["compact_evidence_file_count"],
        name="compact evidence file count",
    ) == 12

    candidate_count = _integer(
        candidate["artifact_heavy_count"], name="candidate artifact count"
    )
    candidate_records = _integer(
        candidate["record_count"], name="candidate record count"
    )
    candidate_rate = _number(
        candidate["artifact_heavy_rate"], name="candidate artifact rate"
    )
    legacy_count = _integer(
        legacy["artifact_heavy_count"], name="legacy artifact count"
    )
    legacy_records = _integer(legacy["record_count"], name="legacy record count")
    legacy_rate = _number(legacy["artifact_heavy_rate"], name="legacy artifact rate")
    observed_ratio = _number(
        comparison["candidate_to_legacy_rate_ratio"], name="artifact rate ratio"
    )
    maximum_ratio = _number(
        comparison["maximum_allowed_ratio"], name="maximum artifact rate ratio"
    )
    assert candidate_rate == pytest.approx(candidate_count / candidate_records)
    assert legacy_rate == pytest.approx(legacy_count / legacy_records)
    assert observed_ratio == pytest.approx(candidate_rate / legacy_rate)
    assert maximum_ratio == gates[
        "maximum_candidate_to_legacy_artifact_rate_ratio"
    ]
    assert observed_ratio <= maximum_ratio
    assert comparison["passed"] is True

    canonical = dict(summary)
    observed_digest = canonical.pop("canonical_evidence_sha256")
    assert observed_digest == _canonical_sha256(canonical)
