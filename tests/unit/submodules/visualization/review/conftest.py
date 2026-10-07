"""Small numeric fixtures for contract tests only; never screenshot evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.pipeline.storage.codec import encode_value
from src.utils.checksum import dual_sha256


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    references: dict[str, Any] = {}

    def publish(node: str, schema: str, version: int, value: Any, dependencies: dict[str, Any]) -> None:
        directory = tmp_path / node.replace("/", "_")
        directory.mkdir()
        arrays: dict[str, Any] = {}
        payload = encode_value(value, directory, arrays)
        descriptor = {"node": node, "artifact_id": node, "output_schema": schema, "output_version": version, "source_sha256": "a" * 64, "payload": payload, "arrays": arrays, "dependencies": dependencies, "provenance": {"origin": "inference"}}
        path = directory / "descriptor.json"
        path.write_text(json.dumps(descriptor))
        references[node] = {"artifact_id": node, "schema": schema, "version": version, "path": str(path.relative_to(tmp_path)), "sha256": dual_sha256(path)}

    publish("player_association", "person_identities", 3, {"camera_ids": ["cam0"], "local_track_ids": np.array([[42]], np.int64), "player_ids": np.array([[[7, 7, 7, 7, -1]]], np.int64)}, {})
    observed = np.array([[[True], [True], [False], [True], [True]]])
    confidence: np.ndarray = np.full((1, 5, 1, 17), .8, np.float32)
    confidence[0, 0, 0, 10] = .1
    boxes: np.ndarray = np.full((1, 5, 1, 3), 20, np.float32)
    publish("pose_estimation/cam0", "person_poses", 1, {"camera_ids": ["cam0"], "local_track_ids": np.array([[42]], np.int64), "observed": observed, "uv_px": np.full((1, 5, 1, 17, 2), 30, np.float32), "confidence": confidence, "boxes_xys": boxes, "size": [100, 80], "fps": 30.0}, {})
    frames = np.array([0, 2, 4], np.int64)
    request = {"source_frames": frames, "keypoints": np.full((3, 17, 3), .8, np.float32), "boxes_xys": np.full((3, 3), 20, np.float32), "video_path": str(tmp_path / "missing.mp4"), "size": [100, 80], "intrinsic": np.eye(3)}
    publish("body_view_selection", "body_view_selection", 1, {"selections": [{"person_id": 7, "camera_id": "cam0", "requests": [request], "observed_samples": [3]}]}, {"pose": references["pose_estimation/cam0"]})
    parameters: dict[str, np.ndarray] = {name: np.zeros((3, width), np.float32) for name, width in (("body_pose", 63), ("global_orient", 3), ("betas", 10), ("transl", 3))}
    publish("gvhmr", "body_parameters", 1, {"bodies": [{"person_id": 7, "camera_id": "cam0", "segments": [{"source_frames": frames, "parameters": parameters, "observed_samples": 3}]}]}, {"selection": references["body_view_selection"]})
    valid = np.array([[True, True, True, False, True]])
    publish("body_placement", "placed_bodies", 1, {"players": {"position": np.full((1, 5, 3), 37, np.float32), "yaw": np.zeros((1, 5), np.float32), "global_orient": np.zeros((1, 5, 3), np.float32), "root_valid": valid, "heading_valid": valid, "smpl_valid": valid, "root_reasons": np.array([[0, 0, 0, 101, 0]], np.uint8), "vertices_local": np.zeros((1, 5, 6890, 3), np.float32), "mesh_reprojection_px": np.full((1, 5), 2.5, np.float32), "diagnostics": {"rejection_codes": {"INSUFFICIENT_JOINTS": 101}}}}, {"recovered": references["gvhmr"]})
    joint_valid: np.ndarray = np.ones((1, 5, 17), bool)
    joint_valid[0, 3, 10] = False
    reasons: np.ndarray = (~joint_valid).astype(np.uint8)
    publish("player_triangulation", "player_skeletons", 1, {"skeleton": {"valid": joint_valid, "reasons": reasons, "positions": np.full((1, 5, 17, 3), 1.5, np.float32)}}, {})
    document = {"schema": "tennis_scene_index_v1", "source_sha256": "a" * 64, "source": {"clip_id": "unit-test-only", "videos": [{"camera_id": "cam0", "path": str(tmp_path / "missing.mp4"), "sha256": "b" * 64, "num_frames": 5, "width": 100, "height": 80, "fps": 30.0}]}, "artifacts": references}
    (tmp_path / "scene.json").write_text(json.dumps(document))
    return tmp_path


def replace_payload(root: Path, node: str, transform: Any, *, version: int | None = None) -> None:
    """Create an intentional alternate published fixture with valid checksums."""
    from src.tennis_scene.pipeline.storage.codec import unpack_value

    index = root / "scene.json"
    document = json.loads(index.read_text())
    reference = document["artifacts"][node]
    path = root / reference["path"]
    descriptor = json.loads(path.read_text())
    value = copy.deepcopy(unpack_value(descriptor["payload"], path.parent, descriptor["arrays"]))
    value = transform(value)
    descriptor["arrays"] = {}
    descriptor["payload"] = encode_value(value, path.parent, descriptor["arrays"])
    if version is not None:
        descriptor["output_version"] = reference["version"] = version
    path.write_text(json.dumps(descriptor))
    reference["sha256"] = dual_sha256(path)
    # Update dependent references in the fixture so lineage remains consistent.
    for dependent in document["artifacts"].values():
        dependent_path = root / dependent["path"]
        payload = json.loads(dependent_path.read_text())
        changed = False
        for port, dependency in payload["dependencies"].items():
            if dependency["artifact_id"] == reference["artifact_id"]:
                payload["dependencies"][port] = reference
                changed = True
        if changed:
            dependent_path.write_text(json.dumps(payload))
            dependent["sha256"] = dual_sha256(dependent_path)
    index.write_text(json.dumps(document))
