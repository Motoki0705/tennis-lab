"""Human authority, geometry, publication rollback, stale edits and local API tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from src.synthetic_data_generation.alignment.contracts import (
    ALIGNMENT_SCHEMA,
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
    AlignmentResult,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    AlignmentLineHeatmapView,
)
from src.synthetic_data_generation.alignment.manual.artifacts import (
    read_json,
    validate_manual_outputs,
    write_json,
)
from src.synthetic_data_generation.alignment.manual.geometry import build_manual_result
from src.synthetic_data_generation.alignment.manual.models import (
    ApplyRequest,
    CourtPlacement,
    EditRequest,
    LayoutEdit,
)
from src.synthetic_data_generation.alignment.manual.service import AlignmentEditor
from src.synthetic_data_generation.alignment.manual.source import (
    import_source,
    owner_digest,
)
from src.synthetic_data_generation.alignment.manual.web import create_app
from src.synthetic_data_generation.alignment.validation import (
    load_accepted_layout,
    validate_alignment_outputs,
    write_alignment_outputs,
)
from src.synthetic_data_generation.pipeline.contracts import StageName, StageStatus
from src.synthetic_data_generation.pipeline.locking import scene_write_lock
from src.synthetic_data_generation.pipeline.run_manifest import (
    MutableRunManifest,
    StageRecord,
)


@pytest.fixture
def editor(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> AlignmentEditor:
    root = tmp_path / "B01"
    owner = root / "alignment"
    owner.mkdir(parents=True)
    evidence = alignment_evidence
    heatmaps = AlignmentLineHeatmaps(
        bounds_uv=evidence.ground_plane_frame.bounds_uv_metres,
        grid_spacing=0.5,
        proximity_scale=1.0,
        proximity_power=2.0,
        views=tuple(
            AlignmentLineHeatmapView(
                camera_id=camera_id,
                probability=np.ones((2, 2), np.float32),
                points_uv=evidence.ground_plane_frame.to_uv(
                    evidence.metric_adapter.metric_from_nht_points(
                        line.points_nht_scene
                    )
                ),
                projected_probabilities=np.ones(len(line.points_nht_scene), np.float32),
                proximity_weights=np.ones(len(line.points_nht_scene)),
                included_in_aggregate=camera_id in evidence.partitions.fit_camera_ids,
            )
            for camera_id in evidence.diagnostics.selection.camera_prefix_ids
            for line in evidence.measured_camera_lines
            if camera_id == line.camera_id
        ),
    )
    write_alignment_outputs(
        owner,
        evidence=evidence,
        result=fit_alignment(evidence, policy=alignment_policy),
        heatmaps=heatmaps,
    )
    export = root / "reconstruction" / "export"
    (export / "images").mkdir(parents=True)
    pose = np.eye(4)
    pose[2, 3] = -50
    write_json(export / "scene.json", {"scene_id": "B01"})
    write_json(
        export / "cameras.json",
        {
            "cameras": [
                {
                    "camera_id": "camera-0",
                    "image": "images/test.png",
                    "width": 64,
                    "height": 64,
                    "intrinsics": {"matrix": [[30, 0, 32], [0, 30, 32], [0, 0, 1]]},
                    "camera_to_scene": pose.tolist(),
                }
            ]
        },
    )
    np.save(export / "points_scene.npy", np.zeros((4, 3)))
    Image.new("RGB", (64, 64)).save(export / "images/test.png")
    manifest = MutableRunManifest(
        scene_id="B01",
        config_schema="canonical_scene_pipeline_v1",
        source_video="video.mp4",
        targets=["court"],
        stages={
            stage: StageRecord(status=StageStatus.COMPLETED, attempt=1)
            for stage in StageName
        },
    )
    manifest.save(root / "run.json")
    for name in (
        "datasets/court",
        "datasets/blcs",
        "datasets/plcs",
        "report",
        "publication",
    ):
        path = root / name
        path.mkdir(parents=True)
        (path / "old.txt").write_text(name)
    return AlignmentEditor(root)


def edit_layout() -> LayoutEdit:
    return LayoutEdit(
        scale=0.55,
        courts=[
            CourtPlacement(
                court_id=f"court-{index}",
                u=100.0 + index * 12,
                v=50.0,
                angle_degrees=32.0,
            )
            for index in range(3)
        ],
        primary_court_id="court-2",
    )


def apply_request(editor: AlignmentEditor) -> ApplyRequest:
    return ApplyRequest(
        revision=editor.revision, layout=edit_layout(), human_confirmed=True
    )


def test_human_confirmation_keeps_rejected_metrics_and_round_trips(
    editor: AlignmentEditor,
) -> None:
    result = build_manual_result(editor.source, editor.heatmaps, edit_layout())
    assert all(
        candidate.human_confirmed and candidate.accepted
        for candidate in result.candidates
    )
    assert all(
        candidate.fit.status.value == "rejected" for candidate in result.candidates
    )
    assert all(
        not all(candidate.fit.threshold_checks.values())
        for candidate in result.candidates
    )
    assert all(court.fit_status == "human_confirmed" for court in result.layout.courts)
    assert AlignmentResult.from_dict(result.to_dict()).to_dict() == result.to_dict()
    forged = result.to_dict()
    forged["schema"] = ALIGNMENT_SCHEMA
    with pytest.raises(ValueError):
        AlignmentResult.from_dict(forged)
    with pytest.raises(ValueError, match="cannot be mixed"):
        replace(
            result,
            candidates=(
                result.candidates[0],
                replace(result.candidates[1], human_confirmed=False),
            ),
        )


def test_common_scale_preserves_regulation_size_and_correct_nht_binding(
    editor: AlignmentEditor,
) -> None:
    edit = edit_layout()
    result = build_manual_result(editor.source, editor.heatmaps, edit)
    local = np.asarray(
        [[-5.485, -11.885, 0], [5.485, -11.885, 0], [-5.485, 11.885, 0]], np.float64
    )
    for placement, court in zip(edit.courts, result.layout.courts, strict=True):
        metric = court.scene_from_court.apply(local)
        assert np.linalg.norm(metric[1] - metric[0]) == pytest.approx(10.97)
        assert np.linalg.norm(metric[2] - metric[0]) == pytest.approx(23.77)
        nht_center = result.metric_adapter.nht_from_metric_points(
            court.scene_from_court.apply(np.zeros((1, 3)))
        )
        source_center = editor.source.initial.metric_adapter.metric_from_nht_points(
            nht_center
        )
        np.testing.assert_allclose(
            editor.source.plane.to_uv(source_center),
            [[placement.u, placement.v]],
            atol=1e-10,
        )
        nht_points = result.metric_adapter.nht_from_metric_points(metric)
        original_metric = editor.source.initial.metric_adapter.metric_from_nht_points(
            nht_points
        )
        assert np.linalg.norm(original_metric[1] - original_metric[0]) == pytest.approx(
            10.97 * edit.scale
        )
        np.testing.assert_allclose(
            editor.source.plane.signed_distances(original_metric), 0, atol=1e-10
        )


def test_apply_publishes_canonical_owner_invalidates_descendants_and_retains_history(
    editor: AlignmentEditor,
) -> None:
    old_owner = owner_digest(editor.owner)
    old_revision = editor.revision
    response = editor.apply(apply_request(editor))
    assert editor.revision != old_revision
    assert owner_digest(Path(response["history"]) / "alignment") == old_owner
    assert len(validate_alignment_outputs(editor.owner).layout.courts) == 3
    assert load_accepted_layout(editor.owner).primary_court_instance_id == "court-2"
    manifest = MutableRunManifest.load(editor.root / "run.json")
    assert manifest.stages[StageName.ALIGNMENT].status is StageStatus.COMPLETED
    for stage in (
        StageName.COURT_DATASET,
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
        StageName.REPORT,
    ):
        assert manifest.stages[stage].status is StageStatus.INVALIDATED
    assert not (editor.root / "datasets/court").exists()
    assert not (editor.root / "publication").exists()
    reopened = AlignmentEditor(editor.root)
    assert reopened.layout == edit_layout()
    assert reopened.revision == editor.revision
    # Subsequent edits retain the ORIGINAL source frame rather than compounding scale.
    reopened.apply(apply_request(reopened))
    assert (
        validate_alignment_outputs(editor.owner).metric_adapter
        == build_manual_result(
            editor.source, editor.heatmaps, edit_layout()
        ).metric_adapter
    )


def test_failed_exchange_rolls_back_manifest_and_every_owner(
    editor: AlignmentEditor, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_owner = owner_digest(editor.owner)
    run = (editor.root / "run.json").read_bytes()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated exchange failure")

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.manual.service.exchange_owner_directories",
        fail,
    )
    with pytest.raises(OSError, match="simulated"):
        editor.apply(apply_request(editor))
    assert owner_digest(editor.owner) == old_owner
    assert (editor.root / "run.json").read_bytes() == run
    for name in (
        "datasets/court",
        "datasets/blcs",
        "datasets/plcs",
        "report",
        "publication",
    ):
        assert (editor.root / name / "old.txt").read_text() == name


def test_stale_revision_running_stage_and_writer_lock_fail_without_mutation(
    editor: AlignmentEditor,
) -> None:
    before = owner_digest(editor.owner)
    with pytest.raises(RuntimeError, match="changed"):
        editor.apply(
            ApplyRequest(revision="stale", layout=edit_layout(), human_confirmed=True)
        )
    with scene_write_lock(editor.root), pytest.raises(RuntimeError, match="busy"):
        editor.apply(apply_request(editor))
    manifest = MutableRunManifest.load(editor.root / "run.json")
    manifest.stages[StageName.BLCS_DATASET].status = StageStatus.RUNNING
    manifest.save(editor.root / "run.json")
    editor.revision = editor.current_revision()
    with pytest.raises(RuntimeError, match="running"):
        editor.apply(apply_request(editor))
    assert owner_digest(editor.owner) == before


def test_changed_reconstruction_rejects_old_browser_and_reopen(
    editor: AlignmentEditor,
) -> None:
    request = apply_request(editor)
    (editor.camera_export / "scene.json").write_text('{"scene_id":"changed"}')
    with pytest.raises(RuntimeError, match="changed"):
        editor.apply(request)


def test_draft_is_not_application_and_zero_courts_cannot_be_applied(
    editor: AlignmentEditor,
) -> None:
    before = owner_digest(editor.owner)
    empty = LayoutEdit(scale=1.0, courts=[], primary_court_id=None)
    editor.save_draft(EditRequest(revision=editor.revision, layout=empty))
    reopened = AlignmentEditor(editor.root)
    assert reopened.state()["layout"]["courts"] == []
    assert owner_digest(editor.owner) == before
    with pytest.raises(ValueError, match="at least one"):
        editor.apply(
            ApplyRequest(revision=editor.revision, layout=empty, human_confirmed=True)
        )
    with pytest.raises(ValueError, match="Explicit human"):
        editor.apply(
            ApplyRequest(
                revision=editor.revision, layout=edit_layout(), human_confirmed=False
            )
        )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True])
def test_invalid_scale_rejected(scale: Any) -> None:
    with pytest.raises(ValidationError):
        LayoutEdit(scale=scale, courts=[], primary_court_id=None)


def test_duplicate_ids_and_dangling_primary_rejected() -> None:
    edit = edit_layout().model_dump()
    edit["courts"].append(edit["courts"][0])
    with pytest.raises(ValidationError, match="unique"):
        LayoutEdit.model_validate(edit)
    with pytest.raises(ValidationError, match="Primary"):
        LayoutEdit(scale=1.0, courts=[], primary_court_id="missing")


def test_confirmation_and_geometry_tampering_rejected(editor: AlignmentEditor) -> None:
    editor.apply(apply_request(editor))
    path = editor.owner / "manual-confirmation.json"
    confirmation = read_json(path)
    write_json(path, {**confirmation, "human_confirmed": False})
    with pytest.raises(ValueError, match="confirmation"):
        validate_manual_outputs(editor.owner)
    write_json(path, confirmation)
    result = read_json(editor.owner / "alignment.json")
    result["candidates"][0]["scene_from_court"][3] += 1
    write_json(editor.owner / "alignment.json", result)
    with pytest.raises(ValueError, match="disagrees"):
        validate_manual_outputs(editor.owner)


def test_api_requires_session_token_and_supports_preview_draft_apply(
    editor: AlignmentEditor,
) -> None:
    with TestClient(create_app(editor)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/editor.js").status_code == 200
        assert client.get("/static/resize.mjs").status_code == 200
        assert client.get("/api/heatmap").headers["content-type"] == "image/png"
        assert (
            client.post(
                "/api/apply", json=apply_request(editor).model_dump()
            ).status_code
            == 403
        )
        headers = {"x-editor-token": client.get("/api/state").json()["token"]}
        preview = client.post(
            "/api/preview",
            json={"layout": edit_layout().model_dump(), "camera_index": 0},
            headers=headers,
        )
        assert preview.status_code == 200, preview.text
        assert len(preview.json()["courts"]) == 3
        assert client.get("/api/cameras/0/image").status_code == 200
        assert client.get("/api/cameras/-1/image").status_code == 404
        assert (
            client.get("/api/state", headers={"Host": "evil.invalid"}).status_code
            == 400
        )
        draft = client.post(
            "/api/draft",
            json=EditRequest(
                revision=editor.revision, layout=edit_layout()
            ).model_dump(),
            headers=headers,
        )
        assert draft.status_code == 200, draft.text
        applied = client.post(
            "/api/apply", json=apply_request(editor).model_dump(), headers=headers
        )
        assert applied.status_code == 200, applied.text


def test_missing_ground_frame_requires_explicit_import_option(
    editor: AlignmentEditor,
) -> None:
    path = editor.owner / "ground-line-map.npz"
    with np.load(path) as archive:
        arrays = {
            name: archive[name]
            for name in archive.files
            if name != "ground_plane_frame_json"
        }
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="recover-ground-frame"):
        import_source(editor.root, recover_ground_frame=False)
    # Collinear UV evidence cannot define a plane; recovery must reject it.
    with pytest.raises(ValueError, match="rank="):
        import_source(editor.root, recover_ground_frame=True)


def test_final_manifest_failure_rolls_back_exchanged_alignment(
    editor: AlignmentEditor, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_owner = owner_digest(editor.owner)
    old_manifest = (editor.root / "run.json").read_bytes()
    original_save = MutableRunManifest.save

    def save(manifest: MutableRunManifest, path: Path) -> None:
        if (
            manifest.stages[StageName.ALIGNMENT].summary.get("authority")
            == "human_confirmed"
        ):
            raise OSError("simulated final manifest failure")
        original_save(manifest, path)

    monkeypatch.setattr(MutableRunManifest, "save", save)
    with pytest.raises(OSError, match="simulated final"):
        editor.apply(apply_request(editor))
    assert owner_digest(editor.owner) == old_owner
    assert (editor.root / "run.json").read_bytes() == old_manifest
    assert (editor.root / "datasets/court/old.txt").exists()


def test_another_editor_apply_is_visible_after_browser_reload(
    editor: AlignmentEditor,
) -> None:
    old_request = apply_request(editor)
    second = AlignmentEditor(editor.root)
    second.apply(apply_request(second))
    with pytest.raises(RuntimeError, match="changed"):
        editor.apply(old_request)
    state = editor.state()
    assert state["revision"] == second.revision
    assert state["layout"] == edit_layout().model_dump()


def test_manual_owner_rejects_foreign_reconstruction(editor: AlignmentEditor) -> None:
    editor.apply(apply_request(editor))
    (editor.camera_export / "scene.json").write_text('{"scene_id":"different"}')
    with pytest.raises(ValueError, match="different reconstruction"):
        validate_alignment_outputs(editor.owner)
    with pytest.raises(ValueError, match="Reconstruction has changed"):
        AlignmentEditor(editor.root)


def test_excluded_diagnostic_views_do_not_change_manual_fit_partitions(
    editor: AlignmentEditor,
) -> None:
    excluded = AlignmentLineHeatmapView(
        camera_id="excluded-view",
        probability=np.zeros((2, 2), dtype=np.float32),
        points_uv=np.zeros((0, 2), dtype=np.float64),
        projected_probabilities=np.zeros(0, dtype=np.float32),
        proximity_weights=np.zeros(0, dtype=np.float64),
        included_in_aggregate=False,
    )
    heatmaps = replace(editor.heatmaps, views=editor.heatmaps.views + (excluded,))
    result = build_manual_result(editor.source, heatmaps, edit_layout())
    assert (
        result.to_dict()
        == build_manual_result(editor.source, editor.heatmaps, edit_layout()).to_dict()
    )
