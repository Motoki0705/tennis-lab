"""Negative identity checks and mask semantics for the read-only review UI.

These tiny fixtures are test inputs only. Campaign screenshots use real caches.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.refiner_2d.review.artifacts import ReviewArtifacts
from src.tasks.ball_refiner.refiner_2d.review.web import create_app
from src.tasks.ball_refiner.scripts.review_2d_dataset import PATH_BOUNDARY
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathResolver, RuntimePathRoots
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def review_inputs(tmp_path: Path) -> dict[str, Path]:
    store_root = write_store_clip(tmp_path / "rgb", "tracknet/game9/Clip1", [
        frame(10, ball()), frame(12, ball("interpolated")), frame(14, ball("occlusion_estimated")),
        frame(16, ball("unresolved", xy=None)), frame(18), frame(20, ball("out_of_frame", xy=None)),
        frame(22, annotated=False), frame(24, ball(), ball(track="b002")),
    ], split="val")
    metadata = json.loads((store_root / "metadata.json").read_text())
    metadata["clips"][0].update(source_width=128, source_height=96)
    write_json(store_root / "metadata.json", metadata)
    store = BallFrameStore(store_root)
    clip = store.clips[0]
    target = project_store_targets(store, clip)
    maps = torch.zeros(1, 8, 5, 7)
    maps[:, :, 2, 3] = .6
    maps[:, :, 0, 0] = .1
    config = BallCandidateConfig(max_candidates=3, nms_kernel=3, patch_size=3)
    candidates = decode_candidates(maps, config=config, subpixel_refine=False)
    evidence = ClipEvidence(target.frame_index, target.pts, target.timestamps_seconds,
                            np.arange(8, dtype=np.int64), np.zeros(8, np.int64),
                            candidates.coords[0, :, 0].numpy(), candidates.scores[0, :, 0].numpy(),
                            candidates, (5, 7), 1)
    root = tmp_path / "evidence"
    (root / "clips").mkdir(parents=True)
    path = root / "clips/clip-00000.npz"
    np.savez_compressed(path, **evidence.arrays())
    record = {**asdict(clip), "track_ids": list(clip.track_ids)}
    store_hashes = {name: dual_sha256(store_root / name) for name in ("metadata.json", "index.npz")}
    jpeg_hash = dual_sha256(store_root / "shards" / shard_name(0))
    write_json(root / "manifest.json", {
        "schema": "ball_refiner_detector_evidence.v1", "status": "complete",
        "coordinate_system": "source_xy_div_size_minus_one",
        "store": {"directory": str(tmp_path / "retired_original_store"), "sha256": store_hashes},
        "selection": {"clip_ids": [clip.clip_id]},
        "detector": {"candidates": asdict(config), "window_length": 1,
                     "window_selection": "nearest_window_centre_then_earlier_start", "tail_policy": "backfill_real_frames_no_padding"},
        "clips": [{"clip": record, "file": "clips/clip-00000.npz", "sha256": dual_sha256(path),
                   "jpeg_shard_sha256": jpeg_hash, "heatmap_size_hw": [5, 7]}],
    })
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    files = []
    for condition in ("observed", "evidence_gap"):
        factor = np.tile(np.asarray([[.01, 0], [.004, .02]], np.float32), (8, 2, 1, 1))
        gap: NDArray[np.bool_] = np.zeros(8, np.bool_)
        if condition == "evidence_gap":
            gap[0] = True
        path = predictions / f"{condition}.npz"
        np.savez_compressed(path, frame_index=target.frame_index, pts=target.pts, target_uv=target.uv,
                            target_reason=target.reason, presence=target.presence, presence_valid=target.presence_valid,
                            gap_mask=gap, means=np.full((8, 2, 2), .5, np.float32), scale_tril=factor,
                            mixture_logits=np.zeros((8, 2), np.float32), presence_logits=np.zeros(8, np.float32))
        files.append({"clip_id": clip.clip_id, "condition": condition, "source": clip.source,
                      "camera": None, "frames": 8, "path": path.name, "sha256": dual_sha256(path)})
    write_json(predictions / "manifest.json", {
        "schema": "ball_refiner_cached_comparison.v1",
        "input_sha256": {str(root / "manifest.json"): dual_sha256(root / "manifest.json"),
                         **{str(tmp_path / "retired_original_store" / name): digest for name, digest in store_hashes.items()}},
        "artifacts": files,
    })
    write_json(predictions / "run_state.json", {"status": "complete"})
    context = tmp_path / "context"
    context.mkdir()
    points: NDArray[np.float32] = np.zeros((8, 1, 17, 3), np.float32)
    points[..., 0] = 10
    points[..., 1] = 20
    points[..., 2] = .8
    points[:, :, 7, 0] = -2  # Finite outside-image pose must not be clamped.
    path = context / "context.npz"
    np.savez_compressed(path, frame_index=target.frame_index, pts=target.pts,
                        detection_count=np.ones(8, np.int32), track_ids=np.asarray([1], np.int64),
                        boxes_xys=np.ones((8, 1, 3), np.float32), track_observed=np.ones((8, 1), np.bool_),
                        keypoints=points, court_points=np.full((14, 2), 12, np.float32), court_valid=np.ones(14, np.bool_))
    write_json(context / "manifest.json", {
        "schema": "ball_refiner_context.v1", "status": "complete",
        "rgb_condition": "unmodified_stored_jpeg_bgr.v1",
        "coordinate_system": "stored_jpeg_pixels; source_xy=stored_xy/clip.scale",
        "store": {"sha256": store_hashes}, "evidence": {"manifest_sha256": "different-detector-same-jpeg"},
        "selection": {"clip_ids": [clip.clip_id]},
        "clips": [{"clip": record, "file": path.name, "sha256": dual_sha256(path),
                   "jpeg_shard_sha256": jpeg_hash,
                   "execution": {"person_detection_frames": 8, "tracking_frames": 8, "pose_crops": 8,
                                 "court_frame_indices": [0], "person_region_policy": "full_frame",
                                 "status": "complete", "court_diagnostics": {}}}],
    })
    return {"evidence": root, "predictions": predictions, "context": context, "rgb": store_root}


def service(paths: dict[str, Path], *, rgb: bool = True) -> ReviewArtifacts:
    return ReviewArtifacts(paths["evidence"], predictions_root=paths["predictions"],
                           context_root=paths["context"], rgb_store=paths["rgb"] if rgb else None)


def test_retired_store_is_optional_and_rgb_does_not_replace_teachers(review_inputs: dict[str, Path]) -> None:
    # Change RGB-provider labels; JPEG/timeline/geometry still match.
    p = review_inputs["rgb"] / "metadata.json"
    m = json.loads(p.read_text())
    m["clips"][0]["annotation_sha256"] = "new-label-revision"
    write_json(p, m)
    review = service(review_inputs)
    row = review.frame("tracknet/game9/Clip1", 0, "observed")
    assert row["rgb"]["available"]
    assert row["target"]["position_valid"]
    assert row["target"]["uv"] == pytest.approx([10 / .5 / 127, 20 / .5 / 95])
    assert row["context"]["people"][0]["uv"][7][0] == pytest.approx(-2 / .5 / 127)
    assert row["context"]["used_by_saved_gmm"] is False
    assert row["gmm"]["presence_probability"] == .5
    assert review.image("tracknet/game9/Clip1", 0).startswith(b"\xff\xd8")


@pytest.mark.parametrize("index,reason,pos,known,presence", [
    (0, "observed", True, True, True), (1, "interpolated", False, False, None),
    (2, "occlusion_estimated", False, False, None), (3, "unresolved", False, False, None),
    (4, "no_instance", False, False, None), (5, "out_of_frame", False, True, False),
    (6, "unreviewed", False, False, None), (7, "multiple_instances", False, False, None),
])
def test_masks_do_not_promote_estimates_or_unknowns(review_inputs: dict[str, Path], index: int,
                                                  reason: str, pos: bool, known: bool, presence: bool | None) -> None:
    row = service(review_inputs, rgb=False).frame("tracknet/game9/Clip1", index, "observed")
    assert row["target"]["reason"] == reason
    assert row["target"]["position_valid"] is pos
    assert row["target"]["presence_valid"] is known
    assert row["target"]["presence"] is presence
    assert row["rgb"]["available"] is False


def test_artificial_gap_keeps_saved_patches_but_hides_effective_input(review_inputs: dict[str, Path]) -> None:
    row = service(review_inputs, rgb=False).frame("tracknet/game9/Clip1", 0, "evidence_gap")
    assert row["gap_active"]
    assert len(row["candidates"]) == 2
    assert row["effective_candidates"] == []
    assert row["target"]["position_valid"]
    boundary = row["candidates"][1]
    assert boundary["patch_valid"][0] == [False, False, False]


@pytest.mark.parametrize("mutation", ["jpeg", "pts", "geometry"])
def test_rgb_mismatch_never_overlays_unverified_image(review_inputs: dict[str, Path], mutation: str) -> None:
    p = review_inputs["rgb"]
    if mutation == "jpeg":
        shard = p / "shards" / shard_name(0)
        payload = bytearray(shard.read_bytes())
        payload[-30] ^= 1
        shard.write_bytes(payload)
    elif mutation == "pts":
        with np.load(p / "index.npz") as archive:
            values = {key: archive[key] for key in archive.files}
        values["pts"] += 1
        np.savez(p / "index.npz", **values)
    else:
        m = json.loads((p / "metadata.json").read_text())
        m["clips"][0].update(source_width=256, source_height=192)
        write_json(p / "metadata.json", m)
    review = service(review_inputs)
    row = review.frame("tracknet/game9/Clip1", 0, "observed")
    assert row["rgb"]["available"] is False
    assert row["target"] is not None
    with pytest.raises(FileNotFoundError):
        review.image("tracknet/game9/Clip1", 0)


def test_missing_saved_teachers_is_distinct_from_unknown(review_inputs: dict[str, Path]) -> None:
    review = ReviewArtifacts(review_inputs["evidence"])
    row = review.frame("tracknet/game9/Clip1", 0, "observed")
    assert row["target"] is None and row["gmm"] is None and row["context"] is None
    assert review.timeline("tracknet/game9/Clip1")["presence_valid"] is None
    with pytest.raises(ValueError, match="No saved evidence_gap"):
        review.frame("tracknet/game9/Clip1", 0, "evidence_gap")


def test_wrong_gmm_binding_and_corrupt_evidence_are_errors(review_inputs: dict[str, Path]) -> None:
    path = review_inputs["evidence"] / "clips/clip-00000.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        service(review_inputs).frame("tracknet/game9/Clip1", 0, "observed")
    path = review_inputs["predictions"] / "manifest.json"
    m = json.loads(path.read_text())
    m["input_sha256"][str(review_inputs["evidence"] / "manifest.json")] = "wrong"
    write_json(path, m)
    with pytest.raises(ValueError, match="different detector evidence"):
        service(review_inputs)


def test_partial_context_and_timeline_mismatch_are_rejected(review_inputs: dict[str, Path]) -> None:
    path = review_inputs["context"] / "manifest.json"
    m = json.loads(path.read_text())
    m["clips"][0]["jpeg_shard_sha256"] = "wrong"
    write_json(path, m)
    with pytest.raises(ValueError, match="Context clip/JPEG identity"):
        service(review_inputs)
    m["status"] = "building"
    write_json(path, m)
    with pytest.raises(ValueError, match="Context is incomplete"):
        service(review_inputs)


def test_web_is_get_only_and_rejects_invalid_frames(review_inputs: dict[str, Path]) -> None:
    before = {p: dual_sha256(p) for root in review_inputs.values() for p in root.rglob("*") if p.is_file()}
    client = TestClient(create_app(service(review_inputs)))
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/catalog").json()["counts"]["teacher_gmm_clips"] == 1
    params = {"clip": "tracknet/game9/Clip1", "frame": 5}
    assert client.get("/api/frame", params=params).json()["target"]["presence"] is False
    assert client.get("/api/frame", params={**params, "frame": -1}).status_code == 422
    assert client.get("/api/frame", params={**params, "condition": "made_up"}).status_code == 422
    assert client.get("/api/image", params=params).headers["content-type"] == "image/jpeg"
    assert client.post("/api/frame", params=params).status_code == 405
    assert before == {p: dual_sha256(p) for root in review_inputs.values() for p in root.rglob("*") if p.is_file()}


def test_cli_path_boundary_rejects_relative_paths_before_read(tmp_path: Path) -> None:
    roots = RuntimePathRoots(**{f"{name}_root": tmp_path for name in
                               ("project", "data", "artifact", "output", "checkpoint", "cache", "external_asset")})
    with pytest.raises(ValueError, match="absolute"):
        PATH_BOUNDARY.validate({"evidence": Path("relative")}, resolver=PathResolver(roots), independent_artifact_inputs=True)
