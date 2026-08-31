from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image

from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scripts import (
    evaluate_court_trajectory_safety as benchmark,
)
from src.synthetic_data_generation.scripts.evaluate_court_trajectory_safety import (
    BenchmarkAction,
    BenchmarkConfiguration,
    ObservationLock,
    ValidatedPilotEvidence,
    _load_pilot_manifest,
    _render_contact_sheet,
    _require_matching_observation_lock,
    _validate_blind_review_manifest,
    validate_complete_evidence,
)

_STRATA = (
    "captured_control",
    "legacy_orbit",
    "support_interior",
    "support_boundary",
    "support_exterior",
    "safe_v4_candidate",
)
_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "synthetic_data_generation"
    / "configs"
)


class _WrappedEntrypoint(Protocol):
    __wrapped__: Callable[[DictConfig], int]


def _required_coverage() -> dict[str, object]:
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


def _selected_coverage() -> dict[str, object]:
    return {
        "total_group_count": 24,
        "total_frame_count": 2_712,
        "constructors": [
            "anchored_rounded_rectangle",
            "free_space_cycle",
        ],
        "constructor_group_counts": {
            "anchored_rounded_rectangle": 6,
            "free_space_cycle": 18,
        },
        "constructor_frame_counts": {
            "anchored_rounded_rectangle": 226,
            "free_space_cycle": 2_486,
        },
        "path_families": ["free_space_cycle", "rounded_rectangle"],
        "family_group_counts": {
            "free_space_cycle": 18,
            "rounded_rectangle": 6,
        },
        "family_frame_counts": {
            "free_space_cycle": 2_486,
            "rounded_rectangle": 226,
        },
        "vertical_profiles": ["free_space_cycle", "planar", "raised_phases"],
        "profile_group_counts": {
            "free_space_cycle": 18,
            "planar": 3,
            "raised_phases": 3,
        },
        "profile_frame_counts": {
            "free_space_cycle": 2_486,
            "planar": 113,
            "raised_phases": 113,
        },
        "target_modes": ["court_center"],
        "target_group_counts": {"court_center": 24},
        "target_frame_counts": {"court_center": 2_712},
        "anchor_camera_indices": [0, 1, 2, 3, 4, 5],
        "anchor_camera_ids": [f"camera-{index}" for index in range(6)],
        "unique_anchor_count": 6,
        "anchored_group_count": 6,
        "anchored_frame_count": 226,
        "anchored_frame_share": 226 / 2_712,
        "anchored_planar_group_count": 3,
        "anchored_raised_group_count": 3,
        "anchored_required_lift_group_count": 3,
    }


def _manifest() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for index in range(128):
        opaque_id = f"review-{index:016x}"
        stratum = _STRATA[index % len(_STRATA)]
        records.append(
            {
                "opaque_id": opaque_id,
                "trajectory_group_id": f"group-{index % 8:05d}",
                "stratum": stratum,
                "valid_control": stratum == "captured_control",
                "support_margin_m": 0.5,
                "obstacle_clearance_m": 1.0,
                "captured_camera_distance_m": 0.25,
                "camera": {
                    "camera_id": opaque_id,
                    "width": 64,
                    "height": 48,
                    "intrinsics": {
                        "model": "PINHOLE",
                        "distortion_model": "NONE",
                        "params": [100.0, 101.0, 31.5, 23.5],
                        "matrix": [
                            [100.0, 0.0, 31.5],
                            [0.0, 101.0, 23.5],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                    "camera_to_scene": [
                        [1.0, 0.0, 0.0, float(index)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 2.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            }
        )
    selected_phases = [
        {
            "trajectory_group_id": f"group-{index:05d}",
            "phase_index": index % 6,
            "phase_count": 6,
            "coverage_mode": ("full", "near_full", "partial")[index % 3],
            "look_at_height_m": 1.5 if index % 2 else 0.0,
            "expected_frame_count": 113,
            "expected_valid_frame_count": 104 if index == 0 else 113,
            "rejection_counts": (
                [
                    {
                        "reason": "insufficient_pre_render_semantic_coverage",
                        "count": 9,
                    }
                ]
                if index == 0
                else []
            ),
            "disposition_digest": f"{index:064x}",
        }
        for index in range(24)
    ]
    return {
        "schema": "court_trajectory_safety_pilot_manifest_v1",
        "status": "frozen",
        "scene_id": "B00",
        "seed": 823,
        "required_strata": sorted(_STRATA),
        "tracked_evidence_path": (
            "experiments/synthetic_data_generation/court_trajectory_safety/"
            "pilot-manifest.json"
        ),
        "final_evidence_root": (
            "outputs/court_trajectory_safety/B00-required-coverage-final"
        ),
        "provenance": {
            "scene_export_schema": "nht_standard_scene_v1",
            "captured_camera_count": 491,
            "public_point_count": 217_407,
            "nht_scene_units_per_metre": 0.0698050915802639,
            "metric_metres_per_nht_scene_unit": 14.325602579436074,
            "support_input_digest": "a" * 64,
            "captured_camera_occupied_count": 0,
        },
        "decision_inputs": {
            "feature_definition_id": "court_public_quality_features_v1",
            "legacy_plan_seed": 695,
            "candidate_plan_seed": 695,
            "legacy_proposal_budget": 4_800,
            "candidate_proposal_budget": 4_800,
            "equal_proposal_budget": True,
            "candidate_count": 256,
            "safe_candidate_count": 64,
            "selected_group_count": 24,
            "planned_frame_count": 2_712,
            "selected_support_violation_count": 0,
            "selected_trajectory_group_ids": [
                f"group-{index:05d}" for index in range(24)
            ],
            "required_coverage": _required_coverage(),
            "selected_coverage": _selected_coverage(),
            "required_coverage_shortfall": [],
            "optional_candidate_coverage_shortfall": [],
            "semantic_phase_evaluation_count": 384,
            "semantic_phase_count": 6,
            "semantic_phase_inventory_digest": "b" * 64,
            "projected_semantic_valid_frame_count": 2_703,
            "projected_semantic_rejected_frame_count": 9,
            "projected_semantic_valid_fraction": 2_703 / 2_712,
            "selected_semantic_phases": selected_phases,
        },
        "records": records,
    }


def _write_manifest(tmp_path: Path, manifest: dict[str, object]) -> Path:
    path = tmp_path / "pilot-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _unfrozen_authority() -> dict[str, object]:
    return {
        "schema": "court_trajectory_safety_frozen_config_v2",
        "scene_id": "B00",
        "observation_lock": None,
        "pilot_seed": 823,
        "minimum_pilot_views": 128,
        "required_strata": sorted(_STRATA),
        "feature_definition_id": "court_public_quality_features_v1",
        "group_split": {
            "calibration_fraction": 0.5,
            "held_out_fraction": 0.5,
        },
        "quality_only_thresholds": {
            "minimum_recall": 0.9,
            "minimum_precision": 0.8,
            "maximum_valid_control_false_positive_rate": 0.1,
            "minimum_positive_labels": 12,
            "minimum_negative_labels": 12,
        },
        "geometry_release_gates": {
            "minimum_frames": 2_000,
            "maximum_frames": 5_000,
            "minimum_accepted_fraction": 0.9,
            "minimum_trajectory_groups": 24,
            "maximum_candidate_to_legacy_artifact_rate_ratio": 0.5,
            "required_selected_support_violations": 0,
            "require_group_disjoint_splits": True,
        },
    }


def _entrypoint_config(tmp_path: Path, *, action: str) -> DictConfig:
    scene_path = tmp_path / "data" / "export" / "scene.json"
    alignment_path = tmp_path / "alignment.json"
    frozen_config_path = tmp_path / "frozen-config.json"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("{}\n", encoding="utf-8")
    alignment_path.write_text("{}\n", encoding="utf-8")
    frozen_config_path.write_text(
        json.dumps(_unfrozen_authority()),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(tmp_path, _manifest())
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="evaluate_court_trajectory_safety",
            overrides=[
                f"action={action}",
                f"scene_path={scene_path}",
                f"alignment_path={alignment_path}",
                f"pilot_manifest_path={manifest_path}",
                f"pilot_output_root={tmp_path / 'pilot-output'}",
                f"final_evidence_root={tmp_path / 'final-output'}",
                f"annotation_root={tmp_path / 'annotations'}",
                f"source_video_path={tmp_path / 'B00.mp4'}",
                f"frozen_config_path={frozen_config_path}",
            ],
        )


def test_entrypoint_dispatches_unfrozen_pilot_to_renderer_and_binds_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _entrypoint_config(tmp_path, action="render_frozen_pilot")
    scene_path = tmp_path / "data" / "export" / "scene.json"
    scene = cast(
        StandardSceneExport,
        SimpleNamespace(scene_id="B00", scene_path=scene_path),
    )
    client = Mock()
    client.validate_scene.return_value = scene
    client.render.return_value = SimpleNamespace(records=())
    monkeypatch.setattr(benchmark, "NHTRenderClient", lambda: client)
    entrypoint = cast(_WrappedEntrypoint, benchmark.main).__wrapped__

    assert entrypoint(config) == 0
    client.render.assert_called_once()
    features_path = tmp_path / "pilot-output" / "features.json"
    features = json.loads(features_path.read_text(encoding="utf-8"))
    assert features["pilot_manifest_sha256"] == hashlib.sha256(
        (tmp_path / "pilot-manifest.json").read_bytes()
    ).hexdigest()

    with pytest.raises(FileExistsError, match="must be a new ordinary directory"):
        entrypoint(config)
    assert client.render.call_count == 1


def test_entrypoint_consumer_rejects_unfrozen_pilot_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_path = tmp_path / "scene.json"
    runtime = cast(
        BenchmarkConfiguration,
        SimpleNamespace(
            action=BenchmarkAction.VALIDATE_COMPLETE_EVIDENCE,
            scene_path=scene_path,
            frozen_authority=SimpleNamespace(observation_lock=None),
        ),
    )
    scene = cast(StandardSceneExport, SimpleNamespace(scene_id="B00"))
    client = Mock()
    client.validate_scene.return_value = scene
    monkeypatch.setattr(
        benchmark.BenchmarkConfiguration,
        "from_config",
        lambda _config: runtime,
    )
    monkeypatch.setattr(benchmark, "NHTRenderClient", lambda: client)
    entrypoint = cast(_WrappedEntrypoint, benchmark.main).__wrapped__

    with pytest.raises(ValueError, match="Pilot observations are not frozen"):
        entrypoint(DictConfig({}))


def test_frozen_pilot_manifest_preserves_order_strata_and_camera_intrinsics(
    tmp_path: Path,
) -> None:
    path = _write_manifest(tmp_path, _manifest())

    entries = _load_pilot_manifest(
        path,
        expected_scene_id="B00",
        expected_seed=823,
        minimum_view_count=128,
    )

    assert len(entries) == 128
    assert tuple(entry.opaque_id for entry in entries) == tuple(
        sorted(entry.opaque_id for entry in entries)
    )
    assert {entry.stratum for entry in entries} == set(_STRATA)
    assert (
        min(sum(entry.stratum == stratum for entry in entries) for stratum in _STRATA)
        >= 12
    )
    assert entries[0].camera.width == 64
    assert entries[0].camera.height == 48
    assert entries[0].camera.intrinsics == (
        100.0,
        0.0,
        31.5,
        0.0,
        101.0,
        23.5,
        0.0,
        0.0,
        1.0,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_proposal_budget", 4_799),
        ("equal_proposal_budget", False),
        ("required_coverage_shortfall", ["minimum_unique_anchors"]),
    ),
)
def test_frozen_pilot_manifest_rejects_budget_and_required_coverage_tampering(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest = _manifest()
    decision_inputs = manifest["decision_inputs"]
    assert isinstance(decision_inputs, dict)
    decision_inputs[field] = value
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="decision inputs are not feasible"):
        _load_pilot_manifest(
            path,
            expected_scene_id="B00",
            expected_seed=823,
            minimum_view_count=128,
        )


@pytest.mark.parametrize("case", ("unrelated", "order"))
def test_frozen_pilot_manifest_binds_selected_groups_to_semantic_phases(
    tmp_path: Path,
    case: str,
) -> None:
    manifest = _manifest()
    decision_inputs = manifest["decision_inputs"]
    assert isinstance(decision_inputs, dict)
    selected_group_ids = decision_inputs["selected_trajectory_group_ids"]
    selected_phases = decision_inputs["selected_semantic_phases"]
    assert isinstance(selected_group_ids, list)
    assert isinstance(selected_phases, list)
    assert {
        phase["trajectory_group_id"]
        for phase in selected_phases
        if isinstance(phase, dict)
    } == set(selected_group_ids)
    if case == "unrelated":
        selected_group_ids[0] = "group-unrelated-to-selected-phases"
    elif case == "order":
        selected_group_ids[0], selected_group_ids[1] = (
            selected_group_ids[1],
            selected_group_ids[0],
        )
    else:  # pragma: no cover - parametrization is intentionally exhaustive
        raise AssertionError(case)
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="decision inputs are not feasible"):
        _load_pilot_manifest(
            path,
            expected_scene_id="B00",
            expected_seed=823,
            minimum_view_count=128,
        )


@pytest.mark.parametrize(
    ("case", "error", "match"),
    (
        ("seed", ValueError, "identity is invalid"),
        ("duplicate", ValueError, "opaque IDs must be unique"),
        ("order", ValueError, "opaque-ID order"),
        ("missing_stratum", ValueError, "does not cover every frozen stratum"),
        ("under_sampled", ValueError, "strata are under-sampled"),
        ("valid_control", ValueError, "must identify captured controls exactly"),
        ("intrinsics", ValueError, "camera values are invalid"),
        ("dimensions", ValueError, "camera values are invalid"),
    ),
)
def test_frozen_pilot_manifest_rejects_identity_strata_and_camera_mutations(
    tmp_path: Path,
    case: str,
    error: type[Exception],
    match: str,
) -> None:
    manifest = copy.deepcopy(_manifest())
    records = manifest["records"]
    assert isinstance(records, list)
    if case == "seed":
        manifest["seed"] = 824
    elif case == "duplicate":
        records[1]["opaque_id"] = records[0]["opaque_id"]
        records[1]["camera"]["camera_id"] = records[0]["opaque_id"]
    elif case == "order":
        records.reverse()
    elif case == "missing_stratum":
        for record in records:
            if record["stratum"] == "safe_v4_candidate":
                record["stratum"] = "support_interior"
    elif case == "under_sampled":
        safe_records = [
            record for record in records if record["stratum"] == "safe_v4_candidate"
        ]
        for record in safe_records[:-11]:
            record["stratum"] = "support_interior"
    elif case == "valid_control":
        records[0]["valid_control"] = False
    elif case == "intrinsics":
        records[0]["camera"]["intrinsics"]["params"][0] = 99.0
    elif case == "dimensions":
        records[0]["camera"]["width"] = True
    else:  # pragma: no cover - parametrization is intentionally exhaustive
        raise AssertionError(case)

    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(error, match=match):
        _load_pilot_manifest(
            path,
            expected_scene_id="B00",
            expected_seed=823,
            minimum_view_count=128,
        )


def test_contact_sheet_is_deterministic_and_rgb_only(tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilot"
    opaque_ids = ("review-0000000000000000", "review-0000000000000001")
    for index, opaque_id in enumerate(opaque_ids):
        root = pilot_root / "renders" / opaque_id
        root.mkdir(parents=True)
        Image.new("RGBA", (32, 18), (index * 127, 25, 200, 128)).save(root / "rgb.png")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    first_size = _render_contact_sheet(
        first,
        pilot_root=pilot_root,
        opaque_ids=opaque_ids,
        caption=lambda opaque_id: opaque_id,
    )
    second_size = _render_contact_sheet(
        second,
        pilot_root=pilot_root,
        opaque_ids=opaque_ids,
        caption=lambda opaque_id: opaque_id,
    )

    assert first_size == second_size == (1280, 200)
    assert (
        hashlib.sha256(first.read_bytes()).digest()
        == hashlib.sha256(second.read_bytes()).digest()
    )
    with Image.open(first) as image:
        assert image.mode == "RGB"


def test_observation_lock_rejects_stale_hashes_and_preview_or_annotation_tampering(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path, _manifest())
    entries = _load_pilot_manifest(
        manifest_path,
        expected_scene_id="B00",
        expected_seed=823,
        minimum_view_count=128,
    )
    pilot_root = tmp_path / "pilot"
    blind_records: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        preview = pilot_root / "renders" / entry.opaque_id / "rgb.png"
        preview.parent.mkdir(parents=True)
        Image.new("RGB", (4, 3), (index % 256, 25, 200)).save(preview)
        blind_records.append(
            {
                "opaque_id": entry.opaque_id,
                "rgb_preview": f"renders/{entry.opaque_id}/rgb.png",
            }
        )
    blind_path = pilot_root / "blind-review-manifest.json"
    blind_path.write_text(
        json.dumps(
            {
                "schema": "court_trajectory_blind_review_manifest_v1",
                "scene_id": "B00",
                "records": blind_records,
            }
        ),
        encoding="utf-8",
    )
    preview_digest = _validate_blind_review_manifest(
        blind_path,
        pilot_root=pilot_root,
        entries=entries,
        scene_id="B00",
    )
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text('{"labels":[]}', encoding="utf-8")
    annotation_sha256 = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    expected = ObservationLock(
        pilot_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        pilot_features_sha256="a" * 64,
        blind_review_manifest_sha256=hashlib.sha256(
            blind_path.read_bytes()
        ).hexdigest(),
        reviewer_a_sha256=annotation_sha256,
        reviewer_b_sha256="b" * 64,
        adjudication_sha256="c" * 64,
        rgb_preview_inventory_sha256=preview_digest,
    )
    _require_matching_observation_lock(observed=expected, expected=expected)

    with pytest.raises(ValueError, match="explicit tracked observation lock"):
        _require_matching_observation_lock(
            observed=expected,
            expected=replace(expected, pilot_manifest_sha256="d" * 64),
        )

    Image.new("RGB", (4, 3), (255, 0, 0)).save(
        pilot_root / "renders" / entries[0].opaque_id / "rgb.png"
    )
    changed_preview_digest = _validate_blind_review_manifest(
        blind_path,
        pilot_root=pilot_root,
        entries=entries,
        scene_id="B00",
    )
    with pytest.raises(ValueError, match="explicit tracked observation lock"):
        _require_matching_observation_lock(
            observed=replace(
                expected,
                rgb_preview_inventory_sha256=changed_preview_digest,
            ),
            expected=expected,
        )

    annotation_path.write_text('{"labels":[true]}', encoding="utf-8")
    changed_annotation_sha256 = hashlib.sha256(
        annotation_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="explicit tracked observation lock"):
        _require_matching_observation_lock(
            observed=replace(
                expected,
                reviewer_a_sha256=changed_annotation_sha256,
            ),
            expected=expected,
        )


def test_validate_complete_evidence_is_strictly_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_root = tmp_path / "tracked"
    audit_root = tmp_path / "annotations" / "evidence"
    audit_root.mkdir(parents=True)
    manifest_path = tracked_root / "pilot-manifest.json"
    tracked_root.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    lock = ObservationLock(
        pilot_manifest_sha256="a" * 64,
        pilot_features_sha256="b" * 64,
        blind_review_manifest_sha256="c" * 64,
        reviewer_a_sha256="d" * 64,
        reviewer_b_sha256="e" * 64,
        adjudication_sha256="f" * 64,
        rgb_preview_inventory_sha256="0" * 64,
    )
    consensus_payload = {
        "records": [],
        "pilot_manifest_sha256": lock.pilot_manifest_sha256,
    }
    decision_payload = {
        "schema": "court_trajectory_safety_derived_evidence_v2",
        "pilot_manifest_sha256": lock.pilot_manifest_sha256,
        "calibration": {"status": "calibrated"},
        "held_out_decision": {"decision": "quality_only_rejected"},
    }
    summary: dict[str, object] = {
        "status": "complete",
        "decision": "quality_only_rejected",
        "representative_contact_sheets": [],
    }
    derived_payload = {
        "schema": "court_trajectory_safety_derived_evidence_v2",
        "tracked_summary": summary,
        "consensus": consensus_payload,
        "quality_decision": decision_payload,
    }
    for path, payload in (
        (audit_root / "consensus.json", consensus_payload),
        (audit_root / "quality-decision.json", decision_payload),
        (audit_root / "evidence.json", derived_payload),
        (tracked_root / "summary.json", summary),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    (tracked_root / "report.md").write_text("complete\n", encoding="utf-8")

    evidence = cast(
        ValidatedPilotEvidence,
        SimpleNamespace(
            consensus=SimpleNamespace(to_dict=lambda: {"records": []}),
            calibration=SimpleNamespace(
                to_dict=lambda: {"status": "calibrated"}
            ),
            decision=SimpleNamespace(
                to_dict=lambda: {"decision": "quality_only_rejected"}
            ),
        ),
    )
    runtime = cast(
        BenchmarkConfiguration,
        SimpleNamespace(
            pilot_manifest_path=manifest_path,
            audit_output_root=audit_root,
            source_video_path=tmp_path / "B00.mp4",
            frozen_authority=SimpleNamespace(observation_lock=lock),
        ),
    )
    monkeypatch.setattr(benchmark, "validate_pilot_evidence", lambda **_kwargs: evidence)
    monkeypatch.setattr(
        benchmark,
        "_validate_source_video_authority",
        lambda **_kwargs: {"status": "valid"},
    )
    monkeypatch.setattr(
        benchmark,
        "_final_dataset_evidence",
        lambda **_kwargs: {"status": "complete"},
    )
    monkeypatch.setattr(
        benchmark,
        "_validate_existing_contact_sheets",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        benchmark,
        "_tracked_summary",
        lambda **_kwargs: summary,
    )
    monkeypatch.setattr(benchmark, "_evidence_report", lambda _summary: "complete\n")

    before = (
        tmp_path.stat().st_mtime_ns,
        tuple(
            (
                path.relative_to(tmp_path).as_posix(),
                path.is_dir(),
                path.stat().st_mtime_ns,
                path.read_bytes() if path.is_file() else b"",
            )
            for path in sorted(tmp_path.rglob("*"))
        ),
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only evidence validation attempted a write")

    monkeypatch.setattr(benchmark, "save_json_atomic", forbidden)
    monkeypatch.setattr(benchmark, "_write_contact_sheets", forbidden)
    monkeypatch.setattr(benchmark, "_render_contact_sheet", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    result = validate_complete_evidence(
        scene=cast(StandardSceneExport, object()),
        runtime=runtime,
    )
    after = (
        tmp_path.stat().st_mtime_ns,
        tuple(
            (
                path.relative_to(tmp_path).as_posix(),
                path.is_dir(),
                path.stat().st_mtime_ns,
                path.read_bytes() if path.is_file() else b"",
            )
            for path in sorted(tmp_path.rglob("*"))
        ),
    )

    assert result == summary
    assert after == before
