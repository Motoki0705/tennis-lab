"""Small contract fixtures, never used as dataset screenshot evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.tennis_scene.pipeline.storage.codec import encode_value
from src.utils.checksum import dual_sha256


def write_artifact(
    root: Path, node: str, schema: str, version: int, payload: dict[str, Any]
) -> dict[str, Any]:
    directory = root / "components" / node.replace("/", "_")
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    packed = encode_value(payload, directory, arrays)
    path = directory / "artifact.json"
    descriptor = {
        "artifact_id": node,
        "node": node,
        "output_schema": schema,
        "output_version": version,
        "source_sha256": "test-source",
        "dependencies": {},
        "arrays": arrays,
        "payload": packed,
        "provenance": {"origin": "test_fixture"},
    }
    path.write_text(json.dumps(descriptor))
    return {
        "artifact_id": node,
        "schema": schema,
        "version": version,
        "execution_key": node,
        "path": str(path.relative_to(root)),
        "sha256": dual_sha256(path),
    }


@pytest.fixture
def stored_case(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    videos, references = [], {}
    for camera in ("cam0", "cam1"):
        path = tmp_path / f"{camera}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"mp4v"), 60, (128, 72)
        )
        assert writer.isOpened()
        for frame in range(4):
            writer.write(np.full((72, 128, 3), frame * 50, np.uint8))
        writer.release()
        videos.append(
            {
                "camera_id": camera,
                "fps": 60.0,
                "num_frames": 4,
                "width": 128,
                "height": 72,
                "path": str(path),
                "sha256": dual_sha256(path),
            }
        )
        node = f"ball_detection/{camera}"
        references[node] = write_artifact(
            root,
            node,
            "ball_detections",
            1,
            {
                "camera_id": camera,
                "frame_indices": np.arange(4, dtype=np.int64),
                "uv_px": np.full((4, 2), 30, np.float32),
                "observed": np.array([False, True, True, False]),
                "point_kind": np.array([0, 1, 1, 2], np.uint8),
                "confidence": np.array([0, 0.7, 0.8, 0], np.float32),
                "score_semantics": "model_score",
            },
        )
    references["court_calibration"] = write_artifact(
        root,
        "court_calibration",
        "local_court_calibration",
        1,
        {
            "reference_camera": "cam0",
            "calibration": {
                "excluded": {},
                "views": [
                    {
                        "camera": {
                            "camera_id": c,
                            "rotation": np.eye(3),
                            "translation": np.array([0.0, 10.0, 2.0]),
                        },
                        "frame_index": 0,
                        "rmse_px": 0.5,
                    }
                    for c in ("cam0", "cam1")
                ],
            },
        },
    )
    document = {
        "schema": "tennis_scene_index_v1",
        "source_sha256": "test-source",
        "source": {"clip_id": "test/fixture", "videos": videos},
        "artifacts": references,
    }
    (root / "scene.json").write_text(json.dumps(document))
    return root


@pytest.fixture
def saved_diagnostics(stored_case: Path) -> Path:
    directory = stored_case.parent / "diagnostics"
    directory.mkdir()
    hypotheses = [
        {"view_half_turns": [False, False], "cost": 0.2, "support": 1.0, "frames": 1},
        {"view_half_turns": [False, True], "cost": 0.8, "support": 0.0, "frames": 1},
    ]
    report = {
        "decided": False,
        "reason": "ambiguous_margin",
        "frames": 1,
        "hypotheses": hypotheses,
        "pair_frames": {"cam0-cam1": 1},
        "sampled_frames": 2,
        "margin": 0.6,
        "thresholds": {
            "min_margin": 0.7,
            "max_cost": 0.8,
            "min_support": 0.2,
            "min_frames": 1,
        },
        "view_half_turns": None,
    }
    document = json.loads((stored_case / "scene.json").read_text())
    source = document["source"]
    execute = directory / "execute.json"
    execute.write_text(
        json.dumps(
            {
                "run": {
                    "source": source,
                    "artifacts": document["artifacts"],
                    "scene_index": str(stored_case / "scene.json"),
                    "error_reason": "court_side_ambiguous_margin",
                    "error_diagnostics": {
                        key: report[key]
                        for key in (
                            "frames",
                            "hypotheses",
                            "pair_frames",
                            "sampled_frames",
                        )
                    },
                }
            }
        )
    )
    report["inputs"] = [{"path": str(execute), "sha256": dual_sha256(execute)}]
    (directory / "production.json").write_text(json.dumps(report))
    np.savez(
        directory / "production-observations.npz",
        uv_px=np.full((2, 4, 2), 30, np.float32),
        visible=np.array([[False, True, True, False]] * 2),
        sampled_frame_indices=np.array([0, 2], np.int64),
        distinct_mask=np.array([True, True]),
        scored_mask=np.array([False, True]),
    )
    (directory / "production-frames.csv").write_text(
        "frame,view_mask,cost_0,cost_1,support_0,support_1\n2,3,0.2,0.8,1,0\n"
    )
    return directory
