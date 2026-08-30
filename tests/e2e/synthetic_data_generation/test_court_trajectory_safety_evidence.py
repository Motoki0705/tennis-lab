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
_SHA256 = set("0123456789abcdef")
_MANIFEST_SHA256 = "6d65ff89729a866491d97f62dff8c76650d0dff793544baf26533c168a75dffa"
_INVENTORY_SHA256 = "4280156d286359e34f020701ee67df269741c60a91815964378b76f17bd4b839"
_FINAL_EVIDENCE_ROOT = "outputs/court_trajectory_safety/B00-required-coverage-final"


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


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


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
    assert authority.observation_lock is None
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
    geometry = _mapping(summary["geometry_metrics"], name="pending geometry")
    assert summary["status"] == "attempt_2_pending_gpu_pilot_and_blind_annotations"
    assert summary["decision"] is None
    assert summary["annotations"] is None
    assert summary["final_dataset"] is None
    assert geometry["required_coverage"] == required
    assert geometry["selected_coverage"] == selected
    assert geometry["required_coverage_shortfall"] == []
    assert geometry["final_release_status"] == "pending_fresh_gpu_evidence"


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
    assert authority.observation_lock is not None, (
        "Attempt-2 complete evidence is blocked pending a fresh GPU pilot, "
        "blind annotations/adjudication, and final V4 render."
    )
    lock = authority.observation_lock
    pilot_path = _ROOT / "pilot-manifest.json"
    assert hashlib.sha256(pilot_path.read_bytes()).hexdigest() == (
        lock.pilot_manifest_sha256
    )
    pilot = _json(pilot_path)
    inputs = _mapping(pilot["decision_inputs"], name="pilot decision inputs")
    summary = _json(_ROOT / "summary.json")
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
    _validate_complete_evidence(summary, frozen=_json(frozen_path))
    sheets = summary["representative_contact_sheets"]
    assert isinstance(sheets, list) and sheets
    assert all(
        isinstance(sheet, dict)
        and _is_sha256(sheet.get("sha256"))
        and _integer(sheet.get("image_count"), name="contact-sheet image count") > 0
        for sheet in sheets
    )


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
    required = {
        "minimum_total_groups": 24,
        "minimum_free_space_cycle_groups": 12,
        "minimum_anchored_rounded_rectangle_groups": 6,
        "minimum_unique_anchors": 6,
        "minimum_anchored_planar_groups": 3,
        "minimum_anchored_raised_groups": 3,
        "required_raised_lift_m": 0.25,
        "minimum_anchored_frame_share": 0.08,
    }
    selected = {
        "total_group_count": 39,
        "total_frame_count": 2_016,
        "constructor_group_counts": {
            "anchored_rounded_rectangle": 27,
            "free_space_cycle": 12,
        },
        "unique_anchor_count": 27,
        "anchored_planar_group_count": 13,
        "anchored_raised_group_count": 14,
        "anchored_required_lift_group_count": 14,
        "anchored_frame_share": 648 / 2_016,
    }
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
            "dataset_manifest_sha256": "a" * 64,
            "compact_evidence_sha256": "b" * 64,
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
        (("artifact_comparison", "candidate_to_legacy_rate_ratio"), 0.75),
        (("artifact_comparison", "passed"), False),
    ),
)
def test_complete_evidence_release_gates_fail_closed_on_tampering(
    path: tuple[str, ...],
    mutated_value: object,
) -> None:
    summary = copy.deepcopy(_complete_summary_fixture())
    frozen = _json(_ROOT / "frozen-config.json")
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
    assert _is_sha256(final_dataset["dataset_manifest_sha256"])
    assert _is_sha256(final_dataset["compact_evidence_sha256"])
    assert _integer(
        final_dataset["compact_evidence_file_count"],
        name="compact evidence file count",
    ) > 0

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
