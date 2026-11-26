import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import NODE, SCHEMA, VERSION
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.event_detection import EVENT_DETECTION
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.visualization.pose_comparison.inputs import (
    ClipPlan,
    _read_scene,
    bind_sources,
    check_candidate_pin,
    select_clips,
)
from src.tennis_scene.visualization.review.store import StoreReader


def test_select_original_stops_and_sample_from_sorted_successes() -> None:
    ids = [f"video_000/clip_{n:03d}" for n in (6, 3, 2, 5, 1, 4)]
    failed = ids[2]
    records = {
        clip: {"status": "generated"}
        if clip != failed
        else {"status": "failed", "reason": "example"}
        for clip in ids
    }
    progress = {
        "clips": records,
        "targets": [
            {"record": {"clip_id": clip, "path": f"dataset/{clip}", "num_frames": 4}}
            for clip in ids
        ],
        "nodes": {
            clip: {
                "used": {"status": "executed", "seconds": 1.5},
                "blocked": {"status": "blocked", "seconds": 7},
            }
            for clip in ids
        },
    }
    plans, selection = select_clips(progress, seed=42)
    assert [p.clip_id for p in plans] == [
        failed,
        "video_000/clip_001",
        "video_000/clip_006",
    ]
    assert plans[0].baseline_reason == "example"
    assert plans[0].node_seconds == 1.5
    assert selection["success_population"] == 5


@pytest.fixture
def source(tmp_path: Path) -> ClipSource:
    return ClipSource(
        "video_000/clip_012",
        tuple(
            SourceVideo(
                f"cam{i}", tmp_path / f"cam{i}.mp4", str(i) * 64, 3, 60.0, 320, 180
            )
            for i in range(4)
        ),
    )


def _publish(
    store: ClipStore,
    node: str,
    payload: dict,
    schema: str,
    version: int,
    dependencies: dict | None = None,
) -> ArtifactRef:
    return store.publish(
        node,
        payload,
        ArtifactCodec(dict),
        schema=schema,
        version=version,
        identity={"node": node},
        dependencies=dependencies or {},
        provenance={},
    )


def test_reject_order_mismatch_even_with_the_same_cameras(
    tmp_path: Path, source: ClipSource
) -> None:
    old = ClipStore(tmp_path / "old", json_value(source))
    changed = replace(
        source, videos=(source.videos[1], source.videos[0], *source.videos[2:])
    )
    new = ClipStore(tmp_path / "new", json_value(changed))
    plan = ClipPlan(source.clip_id, Path(source.clip_id), 3, None, None, 0.0)
    with pytest.raises(ValueError, match="camera order"):
        bind_sources(StoreReader(old.index_path), StoreReader(new.index_path), plan)


def test_candidate_pin_verifies_published_identity_and_descriptor(
    tmp_path: Path, source: ClipSource
) -> None:
    store = ClipStore(tmp_path, json_value(source))
    reference = _publish(store, NODE, {}, SCHEMA, VERSION)
    reader = StoreReader(store.index_path)
    pin = {
        "artifact_id": reference.artifact_id,
        "descriptor_sha256": reference.sha256,
        "source_sha256": store.source_key,
    }
    assert check_candidate_pin(reader, pin)["node"] == NODE
    with pytest.raises(ValueError, match="pinned run record"):
        check_candidate_pin(reader, {**pin, "artifact_id": "f" * 64})
    with (tmp_path / reference.path).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="descriptor path/checksum"):
        check_candidate_pin(reader, pin)


@pytest.mark.parametrize("version", [2, 3])
def test_ball_only_scene_uses_its_own_adopted_camera_component(
    tmp_path: Path, source: ClipSource, version: int
) -> None:
    store = ClipStore(tmp_path, json_value(source))
    cameras = [
        {
            "camera_id": video.camera_id,
            "intrinsic": np.eye(3),
            "rotation": np.eye(3),
            "translation": np.ones(3),
        }
        for video in source.videos
    ]
    _publish(store, "camera_alignment", {"geometry": None}, "aligned_cameras", 1)
    actual = _publish(
        store,
        "external_ball_camera_geometry",
        {"geometry": {"cameras": cameras}},
        "aligned_cameras",
        1,
    )
    ball_ref = _publish(store, "ball_triangulation", {}, "external_ball_trajectory", 1)
    assembly = _publish(
        store,
        "scene_assembly",
        {},
        "scene_result",
        2,
        {"alignment": actual, "ball": ball_ref},
    )
    inputs = (
        {"scene_assembly": assembly}
        if version == 2
        else {
            EVENT_DETECTION: _publish(
                store, EVENT_DETECTION, {}, "scene_result", 3, {"scene": assembly}
            ),
        }
    )
    archive = tmp_path / "scene.npz"
    sidecar = tmp_path / "scene.metadata.json"
    sidecar.write_text(
        json.dumps(
            {
                "track_ids": [],
                "camera_ids": source.camera_ids,
                "dataset_clip_id": source.clip_id,
                "scene_schema_version": version,
            }
        )
    )
    np.savez_compressed(
        archive,
        width=320,
        height=180,
        num_frames=3,
        fps=60.0,
        player_kp_3d=np.zeros((0, 3, 17, 3), dtype=np.float32),
        player_kp_3d_vis=np.zeros((0, 3, 17), dtype=bool),
        ball_3d=np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]], dtype=np.float32),
        ball_3d_valid=[False, True, False],
    )
    store.record_export("scene", {"scene": archive, "metadata": sidecar}, inputs)
    scene = _read_scene(StoreReader(store.index_path), source)
    assert (
        scene is not None and scene.alignment_ref["artifact_id"] == actual.artifact_id
    )
    assert scene.schema_version == version
    assert (
        scene.ball_ref is not None
        and scene.ball_ref["artifact_id"] == ball_ref.artifact_id
    )
    assert np.array_equal(scene.ball[scene.ball_valid], [[1, 2, 3]])
    assert not scene.valid.any()
    assert scene.unused_view_errors is None


@pytest.mark.parametrize("tracked", [False, True], ids=["v2", "v3"])
def test_saved_scene_uses_its_skeleton_for_unused_view_errors(
    tmp_path: Path, source: ClipSource, tracked: bool
) -> None:
    store = ClipStore(tmp_path, json_value(source))
    cameras = [
        {
            "camera_id": camera,
            "intrinsic": np.array([[100.0, 0, 160 + 10 * i], [0, 100, 90], [0, 0, 1]]),
            "rotation": np.eye(3),
            "translation": np.array([0.0, 0.0, 1.0]),
        }
        for i, camera in enumerate(source.camera_ids)
    ]
    xyz: NDArray[np.float32] = np.zeros((1, 3, 17, 3), dtype=np.float32)
    xyz[0, 0, 5:7, 2] = 1.0
    xyz[0, 1, 5, 2] = -2.0
    valid = np.zeros(xyz.shape[:-1], dtype=bool)
    valid[0, 0, 5:7] = True
    valid[0, 1, 5] = True
    inliers: NDArray[np.bool_] = np.zeros((1, 4, 3, 17), dtype=bool)
    inliers[0, :2, 0, 5:7] = True
    visibility: NDArray[np.float32] = np.zeros(inliers.shape, dtype=np.float32)
    visibility[0, :, :, 5] = 1  # Includes behind-camera and invalid 3D joints.
    points: NDArray[np.float32] = np.zeros((*inliers.shape, 2), dtype=np.float32)
    points[0, 2, 0, 5] = (177 / 320, 86 / 180)  # Error (3, 4) px.
    points[0, 3, 0, 5] = (199 / 320, 90 / 180)  # Error (9, 0) px.
    geometry = {"geometry": {"cameras": cameras}}
    skeleton = {"positions": xyz, "valid": valid, "inliers": inliers}
    if tracked:
        frozen = _publish(
            store,
            "multiview_scene_inputs",
            {"alignment": geometry, "people": {"skeleton": skeleton}},
            "multiview_scene_inputs",
            1,
        )
        dependencies = {"tracked": frozen}
    else:
        dependencies = {
            "alignment": _publish(
                store, "camera_alignment", geometry, "aligned_cameras", 1
            ),
            "skeleton": _publish(
                store,
                "player_triangulation",
                {"skeleton": skeleton},
                "player_skeletons",
                1,
            ),
        }
    assembly = _publish(store, "scene_assembly", {}, "scene_result", 2, dependencies)
    version = 3 if tracked else 2
    inputs = (
        {"scene_assembly": assembly}
        if version == 2
        else {
            EVENT_DETECTION: _publish(
                store, EVENT_DETECTION, {}, "scene_result", 3, {"scene": assembly}
            ),
        }
    )
    archive = tmp_path / "scene.npz"
    sidecar = archive.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "track_ids": [14],
                "camera_ids": source.camera_ids,
                "dataset_clip_id": source.clip_id,
                "scene_schema_version": version,
            }
        )
    )
    np.savez_compressed(
        archive,
        width=320,
        height=180,
        num_frames=3,
        fps=60.0,
        player_kp_3d=xyz,
        player_kp_3d_vis=valid,
        human_kp_2d=points,
        human_kp_vis=visibility,
        ball_3d=np.zeros((3, 3), dtype=np.float32),
        ball_3d_valid=np.zeros(3, dtype=bool),
    )
    files = {"scene": archive, "metadata": sidecar}
    store.record_export("scene", files, inputs)
    scene = _read_scene(StoreReader(store.index_path), source)
    assert scene is not None and scene.schema_version == version
    assert scene.player_ids == (14,)
    np.testing.assert_allclose(scene.unused_view_errors, (5.0, 9.0), atol=1e-5)

    if tracked:
        # The generic lineage can be current while the export uses a different assembly.
        other_scene = _publish(
            store, "other_scene", {}, "scene_result", 2, dependencies
        )
        event = _publish(
            store, EVENT_DETECTION, {}, "scene_result", 3, {"scene": other_scene}
        )
        store.record_export("scene", files, {EVENT_DETECTION: event})
        with pytest.raises(ValueError, match="adopted scene and its export disagree"):
            _read_scene(StoreReader(store.index_path), source)
