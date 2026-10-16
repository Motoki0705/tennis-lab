"""Observable data-flow, missing-state, integrity and read-only regressions."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.tasks.player_association.review.reader import identity_spans
from src.tasks.player_association.review.service import AssociationReviewService
from src.tasks.player_association.review.web import create_app
from src.tasks.player_association.scripts.review_dataset import PATH_BOUNDARY
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathDirection
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES


def test_same_raw_id_changes_anonymous_person_and_preserves_gaps(
    service: AssociationReviewService,
) -> None:
    detail = service.detail("video_000/clip_000")
    track = next(
        t for t in detail["timelines"] if t["camera"] == "cam0" and t["track_id"] == 7
    )
    assert [
        (s["start"], s["end"], s["label_state"], s["person_id"]) for s in track["spans"]
    ] == [
        (0, 1, "player", "A"),
        (1, 2, "ambiguous", None),
        (3, 4, "player", "B"),
        (4, 5, "unmatched", None),
    ]
    assert track["label_transitions"] == [3]
    assert not any(
        o["camera"] == "cam0"
        for o in service.frame("video_000/clip_000", 2)["observations"]
    )


def test_coverage_uses_label_boxes_and_reports_unmatched_separately(
    service: AssociationReviewService,
) -> None:
    camera = service.detail("video_000/clip_000")["coverage"][0]
    assert camera["label_boxes"] == camera["matched_boxes"] == 4
    assert camera["raw_observed_boxes"] == 5
    assert camera["unmatched_raw_boxes"] == camera["ambiguous_matches"] == 1
    assert camera["label_coverage"] == 1
    assert "未検出" in service.catalog()["label_scope"]


def test_boundary_footpoint_is_invalid_and_not_origin(
    service: AssociationReviewService,
) -> None:
    frame = service.frame("video_000/clip_000", 0)
    excluded = next(o for o in frame["observations"] if o["track_id"] == 11)
    assert excluded["label_state"] == "non_player"
    assert excluded["footpoint"] is None
    assert "境界" in excluded["footpoint_reason"]
    player = next(o for o in frame["observations"] if o["track_id"] == 7)
    assert player["footpoint"] is not None


def test_missing_side_does_not_imply_missing_tracks(
    review_inputs: dict[str, Any],
) -> None:
    service = AssociationReviewService(
        review_inputs["dataset"], review_inputs["artifacts"]
    )
    frame = service.frame(review_inputs["clip_id"], 0)
    assert len(frame["observations"]) == 3
    assert all(o["footpoint"] is None for o in frame["observations"])
    assert "未指定" in service.detail(review_inputs["clip_id"])["geometry_reason"]


def test_missing_raw_store_is_label_only_with_unknown_coverage(
    review_inputs: dict[str, Any],
) -> None:
    (review_inputs["store"] / "scene.json").unlink()
    service = AssociationReviewService(
        review_inputs["dataset"], review_inputs["artifacts"]
    )
    assert (
        service.detail(review_inputs["clip_id"])["coverage"][0]["label_coverage"]
        is None
    )
    assert all(
        o["track_id"] is None
        for o in service.frame(review_inputs["clip_id"], 0)["observations"]
    )


def test_no_store_or_annotation_mutation(
    service: AssociationReviewService, review_inputs: dict[str, Any]
) -> None:
    root = review_inputs["dataset"].parent
    before = {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, dual_sha256(p))
        for p in root.rglob("*")
        if p.is_file()
    }
    service.detail(review_inputs["clip_id"])
    service.frame(review_inputs["clip_id"], 0, report_id=0, camera="cam0", track_id=7)
    service.image(review_inputs["clip_id"], "cam0", 0, 7)
    after = {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, dual_sha256(p))
        for p in root.rglob("*")
        if p.is_file()
    }
    assert before == after
    assert not (review_inputs["store"] / ".scene.lock").exists()


def test_saved_pairs_use_candidate_mapping_and_exact_observation_run(
    service: AssociationReviewService,
) -> None:
    scores = service.frame(
        "video_000/clip_000", 0, report_id=0, camera="cam0", track_id=7
    )["scores"]
    assert scores["available"]
    pair = scores["pairs"][0]
    assert (pair["from"]["track_id"], pair["to"]["track_id"]) == (7, 12)
    assert pair["geometry"] == 1.2
    assert "appearance" not in pair
    assert "historical" in scores["method"]


def test_score_binding_mismatch_does_not_reuse_same_ids(
    review_inputs: dict[str, Any],
) -> None:
    path = review_inputs["report"]
    value = json.loads(path.read_text())
    value["observe"] = str(review_inputs["artifacts"] / "different-observation")
    path.write_text(json.dumps(value))
    service = AssociationReviewService(
        review_inputs["dataset"], review_inputs["artifacts"], score_reports=(path,)
    )
    result = service.frame(review_inputs["clip_id"], 0, report_id=0)["scores"]
    assert not result["available"]
    assert "異なる" in result["reason"]


def test_corrupt_pair_index_stops(review_inputs: dict[str, Any]) -> None:
    path = review_inputs["report"]
    value = json.loads(path.read_text())
    value["clips"][review_inputs["clip_id"]]["diagnostics"]["pairs"][0]["a"] = 9
    path.write_text(json.dumps(value))
    service = AssociationReviewService(
        review_inputs["dataset"], review_inputs["artifacts"], score_reports=(path,)
    )
    with pytest.raises(ValueError, match="unknown candidate"):
        service.frame(review_inputs["clip_id"], 0, report_id=0)


def test_array_corruption_stops_instead_of_label_fallback(
    service: AssociationReviewService, review_inputs: dict[str, Any]
) -> None:
    descriptor = review_inputs["track_paths"][0]
    data = json.loads(descriptor.read_text())
    array = descriptor.parent / data["payload"]["boxes_xyxy"]["array"]
    with array.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        service.detail(review_inputs["clip_id"])


def test_unsupported_track_version_stops(
    service: AssociationReviewService, review_inputs: dict[str, Any]
) -> None:
    path = review_inputs["store"] / "scene.json"
    value = json.loads(path.read_text())
    value["artifacts"]["person_tracking/cam0"]["version"] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsupported review component"):
        service.detail(review_inputs["clip_id"])


def test_raw_rgb_frame_and_crop_share_requested_time(
    service: AssociationReviewService,
) -> None:
    def decode(data: bytes) -> np.ndarray:
        return cast(
            np.ndarray, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        )

    first = decode(service.image("video_000/clip_000", "cam0", 0))
    later = decode(service.image("video_000/clip_000", "cam0", 3))
    crop = decode(service.image("video_000/clip_000", "cam0", 3, 7))
    assert first.shape == later.shape == (96, 160, 3)
    assert crop.shape == (30, 20, 3)
    assert abs(float(crop[:, :, 0].mean()) - float(later[:, :, 0].mean())) < 3
    assert float(later[:, :, 0].mean()) - float(first[:, :, 0].mean()) > 70
    with pytest.raises(ValueError, match="no observed box"):
        service.image("video_000/clip_000", "cam0", 2, 7)


def test_http_routes_are_read_only_and_reject_unknown_ids(
    service: AssociationReviewService,
) -> None:
    client = TestClient(create_app(service))
    assert (
        client.get("/").status_code == client.get("/static/app.js").status_code == 200
    )
    assert client.get("/static/secret.txt").status_code == 404
    assert client.get("/api/clip", params={"clip": "../clip"}).status_code == 404
    assert (
        client.get(
            "/api/frame", params={"clip": "video_000/clip_000", "frame": 5}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/image",
            params={"clip": "video_000/clip_000", "camera": "missing", "frame": 0},
        ).status_code
        == 422
    )
    assert client.post("/api/frame").status_code == 405
    response = client.get(
        "/api/frame", params={"clip": "video_000/clip_000", "frame": 1}
    )
    assert response.status_code == 200
    assert any(o["label_state"] == "ambiguous" for o in response.json()["observations"])


def test_cli_paths_are_registered_inputs_only() -> None:
    assert all(field.direction is PathDirection.INPUT for field in PATH_BOUNDARY.fields)
    assert (
        next(
            b
            for b in EXPECTED_RUNTIME_BOUNDARIES
            if b.module == "src.tasks.player_association.scripts.review_dataset"
        ).validator_key
        == PATH_BOUNDARY.name
    )


def test_spans_preserve_gaps_even_if_same_anonymous_id() -> None:
    spans = identity_spans(np.array([0, 0, 0, 0]), np.array([True, False, False, True]))
    assert spans == [
        {"start": 0, "end": 1, "person_index": 0},
        {"start": 3, "end": 4, "person_index": 0},
    ]


def test_saved_score_observation_count_mismatch_stops(
    review_inputs: dict[str, Any],
) -> None:
    path = review_inputs["report"]
    value = json.loads(path.read_text())
    value["clips"][review_inputs["clip_id"]]["diagnostics"]["segments"][1][
        "observed_frames"
    ] = 99
    path.write_text(json.dumps(value))
    service = AssociationReviewService(
        review_inputs["dataset"], review_inputs["artifacts"], score_reports=(path,)
    )
    with pytest.raises(ValueError, match="observed-frame count"):
        service.frame(review_inputs["clip_id"], 0, report_id=0)


def test_source_grid_mismatch_stops_before_showing_crops(
    service: AssociationReviewService, review_inputs: dict[str, Any]
) -> None:
    path = review_inputs["store"] / "scene.json"
    index = json.loads(path.read_text())
    index["source"]["videos"][0]["num_frames"] = 6
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="time grid"):
        service.detail(review_inputs["clip_id"])


def test_array_symlink_escape_is_rejected(
    service: AssociationReviewService, review_inputs: dict[str, Any]
) -> None:
    descriptor = review_inputs["track_paths"][0]
    data = json.loads(descriptor.read_text())
    array = descriptor.parent / data["payload"]["boxes_xyxy"]["array"]
    outside = review_inputs["dataset"].parent / "outside.npy"
    array.rename(outside)
    array.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        service.detail(review_inputs["clip_id"])


def test_v5_raw_reference_uses_saved_receipts_and_excludes_gsi(
    review_inputs: dict[str, Any],
) -> None:
    run, clip_id = review_inputs["run"], review_inputs["clip_id"]
    (run / "observe.json").unlink()
    target = run / clip_id
    target.mkdir(parents=True)
    store = target / "store"
    shutil.move(str(review_inputs["store"]), store)
    index_path = store / "scene.json"
    index = json.loads(index_path.read_text())
    for node in ("person_tracking/cam0", "person_tracking/cam1"):
        reference = index["artifacts"][node]
        path = store / reference["path"]
        descriptor = json.loads(path.read_text())
        descriptor["output_version"] = reference["version"] = 5
        path.write_text(json.dumps(descriptor))
        reference["sha256"] = dual_sha256(path)
    index_path.write_text(json.dumps(index))
    receipt = target / "person-execute.json"
    receipt.write_text("{}")
    labels = json.loads(review_inputs["label"].read_text())
    labels["provenance"]["raw_receipt"] = {
        "path": str(receipt),
        "sha256": dual_sha256(receipt),
    }
    review_inputs["label"].write_text(json.dumps(labels))
    court = {
        c: {
            "status": "ok",
            "calibration": {
                "views": [
                    {
                        "camera": {
                            "camera_id": c,
                            "intrinsic": [[100, 0, 80], [0, 100, 48], [0, 0, 1]],
                            "rotation": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                            "translation": [0, 0, 10],
                        }
                    }
                ]
            },
        }
        for c in ("cam0", "cam1")
    }
    (target / "court-execute.json").write_text(json.dumps(court))
    (target / "prediction.json").write_text(
        json.dumps({"clip": clip_id, "view_half_turns": [False, True]})
    )
    service = AssociationReviewService(
        review_inputs["dataset"], review_inputs["artifacts"]
    )
    detail = service.detail(clip_id)
    assert detail["provenance"]["layout"] == "raw_reference_v5"
    assert detail["provenance"]["track_versions"] == {"cam0": 5, "cam1": 5}
    assert not any(
        o["camera"] == "cam0" for o in service.frame(clip_id, 2)["observations"]
    )
    players = [
        o
        for o in service.frame(clip_id, 0)["observations"]
        if o["label_state"] == "player"
    ]
    assert np.allclose(players[0]["footpoint"], -np.array(players[1]["footpoint"]))


def test_concurrent_source_frame_requests_do_not_mix_time(
    service: AssociationReviewService,
) -> None:
    frames = [3, 0, 4, 1, 2, 4, 0, 3]
    with ThreadPoolExecutor(max_workers=3) as pool:
        images = list(
            pool.map(
                lambda frame: service.image("video_000/clip_000", "cam0", frame), frames
            )
        )
    for frame, image in zip(frames, images, strict=True):
        decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert abs(float(decoded[:, :, 0].mean()) - (30 + frame * 30)) < 4
