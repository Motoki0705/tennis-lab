"""Small labelled video/store inputs for contract tests, never review evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from src.tasks.player_association.review.service import AssociationReviewService
from src.tennis_scene.pipeline.storage.codec import encode_value
from src.utils.checksum import dual_sha256


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def review_inputs(tmp_path: Path) -> dict[str, Any]:
    dataset, artifacts = tmp_path / "dataset", tmp_path / "outputs"
    clip_id, frames, width, height = "video_000/clip_000", 5, 160, 96
    clip = dataset / "videos/video_000/clips/clip_000"
    cameras = ["cam0", "cam1"]
    media = [f"media/{camera}.avi" for camera in cameras]
    write_json(
        dataset / "dataset.json",
        {
            "version": 2,
            "dataset_id": "test",
            "created_at": "test",
            "updated_at": "test",
            "clips": [
                {
                    "clip_id": clip_id,
                    "video_id": "video_000",
                    "clip_name": "clip_000",
                    "path": "videos/video_000/clips/clip_000",
                    "num_cameras": 2,
                    "num_frames": frames,
                    "fps": 30,
                    "width": width,
                    "height": height,
                }
            ],
        },
    )
    write_json(
        clip / "clip.json",
        {
            "version": 2,
            "dataset_id": "test",
            "clip_id": clip_id,
            "video_id": "video_000",
            "clip_name": "clip_000",
            "fps": 30,
            "num_frames": frames,
            "width": width,
            "height": height,
            "global_start_sec": 0,
            "global_end_sec": frames / 30,
            "camera_ids": cameras,
            "video_paths": media,
            "cameras": [],
            "sync_source": "test",
            "exported_at": "test",
        },
    )
    for relative in media:
        path = clip / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30, (width, height)
        )
        assert writer.isOpened()
        for frame in range(frames):
            writer.write(
                np.full((height, width, 3), (30 + frame * 30, 50, 100), np.uint8)
            )
        writer.release()
    boxes = np.tile(np.array([20, 10, 40, 40], np.float32), (2, frames, 1))
    boxes[1] = [60, 60, 90, 95]
    observed = np.array(
        [[True, True, False, True, True], [True, False, False, False, False]]
    )
    people = [
        {"person_id": "A", "role": "player", "description": "test A"},
        {"person_id": "B", "role": "player", "description": "test B"},
        {"person_id": "X", "role": "non_player", "description": "test exclusion"},
    ]
    label = clip / "annotations/player_association/labels.json"
    write_json(
        label,
        {
            "schema": "player_association_labels_v1",
            "clip_id": clip_id,
            "num_frames": frames,
            "people": people,
            "provenance": {},
            "cameras": {
                "cam0": {
                    "frames": [0, 0, 1, 3],
                    "person": ["A", "X", None, "B"],
                    "boxes_xyxy": [
                        boxes[0, 0].tolist(),
                        boxes[1, 0].tolist(),
                        boxes[0, 1].tolist(),
                        boxes[0, 3].tolist(),
                    ],
                },
                "cam1": {
                    "frames": list(range(frames)),
                    "person": ["A"] * frames,
                    "boxes_xyxy": boxes[0].tolist(),
                },
            },
        },
    )
    run = artifacts / "player_association/test_observe"
    review = {
        "observe_run": "outputs/player_association/test_observe",
        "clips": {clip_id: {}},
    }
    (label.parent / "review.yaml").write_text(yaml.safe_dump(review))
    write_json(
        run / "observe.json", {"schema": "player_association_observe_v1", "clips": {}}
    )
    store = run / "stores" / clip_id
    source = {
        "clip_id": clip_id,
        "videos": [
            {
                "camera_id": camera,
                "path": str((clip / relative).resolve()),
                "width": width,
                "height": height,
                "num_frames": frames,
                "fps": 30,
            }
            for camera, relative in zip(cameras, media, strict=True)
        ],
    }
    index: dict[str, Any] = {
        "schema": "tennis_scene_index_v1",
        "source": source,
        "source_sha256": "test-source",
        "artifacts": {},
    }

    def publish(node: str, payload: Any, schema: str, version: int) -> Path:
        directory = store / "components" / node.split("/")[0] / node.replace("/", "-")
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}
        encoded = encode_value(payload, directory, arrays)
        path = directory / (node.split("/")[0] + ".json")
        write_json(
            path,
            {
                "node": node,
                "artifact_id": node,
                "output_schema": schema,
                "output_version": version,
                "source_sha256": "test-source",
                "payload": encoded,
                "arrays": arrays,
            },
        )
        index["artifacts"][node] = {
            "artifact_id": node,
            "path": str(path.relative_to(store)),
            "schema": schema,
            "version": version,
            "sha256": dual_sha256(path),
        }
        return path

    paths = []
    for camera, ids, bb, obs in [
        ("cam0", [7, 11], boxes, observed),
        ("cam1", [12], boxes[:1], np.ones((1, frames), np.bool_)),
    ]:
        payload = {
            "camera_id": camera,
            "track_ids": np.array(ids, np.int64),
            "boxes_xyxy": bb,
            "observed": obs,
            "source_track_ids": [[x] for x in ids],
            "reconstruction": {
                "boxes": bb,
                "observed": np.ones(obs.shape, np.bool_),
                "interpolated": ~obs,
            },
        }
        paths.append(publish(f"person_tracking/{camera}", payload, "person_tracks", 3))
    camera_values = [
        {
            "camera_id": c,
            "intrinsic": np.array([[100.0, 0, 80], [0, 100, 48], [0, 0, 1]]),
            "rotation": np.diag([1.0, -1, -1]),
            "translation": np.array([0.0, 0, 10]),
        }
        for c in cameras
    ]
    publish(
        "court_calibration",
        {
            "calibration": {
                "views": [{"camera": x} for x in camera_values],
                "excluded": {},
            }
        },
        "local_court_calibration",
        1,
    )
    write_json(store / "scene.json", index)
    sides = artifacts / "sides.json"
    write_json(
        sides,
        {
            "clips": [
                {
                    "clip_id": clip_id,
                    "camera_ids": cameras,
                    "annotation": {"decided": True, "view_half_turns": [False, False]},
                }
            ]
        },
    )
    report = artifacts / "test_score/evaluate.json"
    segments = [
        {"camera": "cam0", "track_id": 11, "start": 0, "end": frames},
        {"camera": "cam0", "track_id": 7, "start": 0, "end": frames},
        {"camera": "cam1", "track_id": 12, "start": 0, "end": frames},
    ]
    write_json(
        report,
        {
            "schema": "player_association_evaluate_v1",
            "observe": str(run),
            "geometry_only": True,
            "sides": str(sides),
            "config_path": "saved.yaml",
            "clips": {
                clip_id: {
                    "status": "ok",
                    "metrics": {},
                    "diagnostics": {
                        "frames": frames,
                        "fps": 30,
                        "segments": [
                            {
                                **seg,
                                "observed_frames": 1
                                if seg["track_id"] == 11
                                else 4
                                if seg["track_id"] == 7
                                else 5,
                            }
                            for seg in segments
                        ],
                        "candidates": [1, 2],
                        "pairs": [
                            {
                                "a": 0,
                                "b": 1,
                                "geometry": 1.2,
                                "median_m": 0.7,
                                "shared_frames": 3,
                            }
                        ],
                    },
                }
            },
        },
    )
    return {
        "dataset": dataset,
        "artifacts": artifacts,
        "clip_id": clip_id,
        "clip": clip,
        "store": store,
        "label": label,
        "sides": sides,
        "report": report,
        "track_paths": paths,
        "run": run,
    }


@pytest.fixture
def service(review_inputs: dict[str, Any]) -> AssociationReviewService:
    return AssociationReviewService(
        review_inputs["dataset"],
        review_inputs["artifacts"],
        sides=review_inputs["sides"],
        score_reports=(review_inputs["report"],),
    )
