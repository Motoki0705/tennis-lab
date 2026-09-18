"""Unit tests for the PLCS inference-UI scene catalog and request validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.tasks.plcs.visualization.inference.service import (
    InferenceService,
    PredictionRequest,
    SceneCatalogError,
    ground_truth_world_joints,
    window_scene,
    yaw_error_degrees,
)


def _write_dataset(
    data_root: Path,
    family: str,
    *,
    selector: str,
    objects: str,
    cameras: int,
    persons: int,
    scenes: int = 1,
    frames: int = 300,
    fps: float = 120.0,
) -> None:
    root = data_root / family
    (root / "scenes").mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "court_keypoints": {"selector": selector},
                    "generation": {"mode": objects},
                    "camera": {"layout": "fixed"},
                }
            }
        ),
        encoding="utf-8",
    )
    ids = [f"scene_{index:06d}" for index in range(scenes)]
    (root / "val.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (root / "train.txt").write_text("", encoding="utf-8")
    (root / "test.txt").write_text("", encoding="utf-8")
    for scene_id in ids:
        scene_dir = root / "scenes" / scene_id
        scene_dir.mkdir(parents=True)
        (scene_dir / "meta.json").write_text(
            json.dumps(
                {
                    "scene_id": scene_id,
                    "fps": fps,
                    "num_frames": frames,
                    "court_keypoint_views": [
                        {"camera_id": f"camera_{index}"} for index in range(cameras)
                    ],
                }
            ),
            encoding="utf-8",
        )
        (scene_dir / "scalars.json").write_text(
            json.dumps({"num_cameras": cameras, "num_persons": persons}),
            encoding="utf-8",
        )


def _write_checkpoint(
    checkpoint_root: Path,
    run: str,
    *,
    model_name: str,
    selector: str,
    input_profile: str | None,
    scene_dir: str,
    max_views: int | None = None,
    max_seq_len: int | None = None,
    num_queries: int | None = None,
    view_range: tuple[int, int] | None = None,
    camera_candidates: list[int] | None = None,
) -> str:
    model: dict[str, object] = {"name": model_name}
    if input_profile is not None:
        model["io"] = {"input_profile": input_profile}
    if max_views is not None:
        model["max_views"] = max_views
    if max_seq_len is not None:
        model["max_seq_len"] = max_seq_len
    if num_queries is not None:
        model["num_queries"] = num_queries
    data: dict[str, object] = {"scene_dir": scene_dir}
    if view_range is not None:
        data["num_views_range"] = list(view_range)
    if camera_candidates is not None:
        data["camera_candidates"] = list(camera_candidates)
    config = {
        "model": model,
        "court_keypoints": {"selector": selector},
        "data": data,
    }
    checkpoint = (
        checkpoint_root / run / "logs" / "version_0" / "checkpoints" / "best.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    (checkpoint.parent.parent / "hparams.yaml").write_text(
        yaml.safe_dump({"config": config}), encoding="utf-8"
    )
    return f"{run}/logs/version_0/checkpoints/best.ckpt"


def _make_service(tmp_path: Path) -> tuple[InferenceService, Path]:
    data_root = tmp_path / "data" / "plcs"
    checkpoint_root = tmp_path / "outputs" / "plcs"
    checkpoint_root.mkdir(parents=True)
    _write_dataset(
        data_root,
        "single_object",
        selector="physical_v1",
        objects="single_object",
        cameras=6,
        persons=1,
        scenes=3,
        frames=300,
    )
    _write_dataset(
        data_root,
        "single_object_camera_view_v2",
        selector="camera_view_v2",
        objects="single_object",
        cameras=4,
        persons=1,
        frames=500,
    )
    service = InferenceService(
        data_root=data_root, checkpoint_root=checkpoint_root, device="cpu"
    )
    return service, checkpoint_root


def test_families_report_selector_objects_and_splits(tmp_path: Path) -> None:
    service, _ = _make_service(tmp_path)
    families = {item.id: item for item in service.families()}
    assert set(families) == {"single_object", "single_object_camera_view_v2"}
    single = families["single_object"]
    assert single.selector == "physical_v1"
    assert single.objects == "single_object"
    assert single.num_cameras == 6
    assert single.splits == {"train": 0, "val": 3, "test": 0}
    v2 = families["single_object_camera_view_v2"]
    assert v2.selector == "camera_view_v2"
    assert v2.num_cameras == 4


def test_scenes_listing_filters_and_limits(tmp_path: Path) -> None:
    service, _ = _make_service(tmp_path)
    listing = service.scenes("single_object", split="val")
    assert listing["total"] == 3
    assert listing["returned"] == 3
    assert listing["scenes"][0]["id"] == "scene_000000"
    filtered = service.scenes("single_object", split="val", query="000002")
    assert [item["id"] for item in filtered["scenes"]] == ["scene_000002"]
    limited = service.scenes("single_object", split="val", limit=2)
    assert limited["returned"] == 2
    with pytest.raises(SceneCatalogError):
        service.scenes("single_object", split="val", limit=0)
    with pytest.raises(SceneCatalogError):
        service.scenes("does_not_exist")


def test_predictor_root_and_revision_follow_selected_checkpoint(tmp_path: Path) -> None:
    original, checkpoint_root = _make_service(tmp_path)
    extra = tmp_path / "ckpt" / "plcs"
    extra.mkdir(parents=True)
    checkpoint = extra / "model.ckpt"
    checkpoint.write_bytes(b"model")
    service = InferenceService(
        data_root=original.data_root,
        checkpoint_root=checkpoint_root,
        checkpoint_roots=[extra],
        device="cpu",
    )
    assert service._checkpoint_resolver(checkpoint).roots.checkpoint_root == extra
    with pytest.raises(SceneCatalogError, match="outside"):
        service._checkpoint_resolver(tmp_path / "elsewhere.ckpt")
    first = service._predictor_key("single", checkpoint, "cpu")
    stat = checkpoint.stat()
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert first != service._predictor_key("single", checkpoint, "cpu")


def test_scene_detail_without_checkpoint(tmp_path: Path) -> None:
    service, _ = _make_service(tmp_path)
    detail = service.scene_detail("single_object_camera_view_v2", "scene_000000")
    assert detail["num_frames"] == 500
    assert detail["num_cameras"] == 4
    assert detail["num_persons"] == 1
    assert detail["selector"] == "camera_view_v2"
    assert detail["reference_required"] is True
    assert detail["window"] is None
    assert [entry["id"] for entry in detail["cameras"]] == [
        "camera_0",
        "camera_1",
        "camera_2",
        "camera_3",
    ]


def test_scene_detail_windows_use_checkpoint_limit(tmp_path: Path) -> None:
    service, checkpoint_root = _make_service(tmp_path)
    _write_checkpoint(
        checkpoint_root,
        "run_phys",
        model_name="plcs_multiview_axial",
        selector="physical_v1",
        input_profile="multiview",
        scene_dir="plcs/single_object",
        max_views=6,
    )
    _write_checkpoint(
        checkpoint_root,
        "run_v2",
        model_name="plcs_multiview_axial_reference",
        selector="camera_view_v2",
        input_profile="multiview",
        scene_dir="plcs/single_object_camera_view_v2",
        max_views=4,
        max_seq_len=256,
    )
    unrestricted = service.scene_detail(
        "single_object",
        "scene_000000",
        checkpoint="run_phys/logs/version_0/checkpoints/best.ckpt",
    )
    assert unrestricted["supported"] is True
    assert unrestricted["window"]["max_length"] == 300
    assert unrestricted["reference_required"] is False
    limited = service.scene_detail(
        "single_object_camera_view_v2",
        "scene_000000",
        checkpoint="run_v2/logs/version_0/checkpoints/best.ckpt",
    )
    assert limited["supported"] is True
    assert limited["window"]["max_length"] == 256
    assert limited["window"]["default_length"] == 256
    assert limited["window"]["scene_frames"] == 500
    assert limited["reference_required"] is True


def test_scene_detail_rejects_family_not_in_allowlist(tmp_path: Path) -> None:
    service, checkpoint_root = _make_service(tmp_path)
    checkpoint = _write_checkpoint(
        checkpoint_root,
        "run_phys",
        model_name="plcs_multiview_axial",
        selector="physical_v1",
        input_profile="multiview",
        scene_dir="plcs/single_object",
    )
    detail = service.scene_detail(
        "single_object_camera_view_v2", "scene_000000", checkpoint=checkpoint
    )
    assert detail["supported"] is False
    assert detail["unsupported_reason"] is not None


def _request(checkpoint: str, **overrides: object) -> PredictionRequest:
    payload: dict[str, object] = {
        "checkpoint": checkpoint,
        "family": "single_object_camera_view_v2",
        "scene": "scene_000000",
        "cameras": (0, 1, 2, 3),
        "reference_camera_id": "camera_0",
        "window_start": 0,
        "window_length": 256,
        "canonical_pose_source": "prediction",
        "device": "cpu",
    }
    payload.update(overrides)
    return PredictionRequest(**payload)  # type: ignore[arg-type]


def _v2_checkpoint(service: InferenceService, checkpoint_root: Path) -> str:
    return _write_checkpoint(
        checkpoint_root,
        "run_v2",
        model_name="plcs_multiview_axial_reference",
        selector="camera_view_v2",
        input_profile="multiview",
        scene_dir="plcs/single_object_camera_view_v2",
        max_views=4,
        max_seq_len=256,
    )


def test_resolve_request_accepts_valid_settings(tmp_path: Path) -> None:
    service, checkpoint_root = _make_service(tmp_path)
    checkpoint = _v2_checkpoint(service, checkpoint_root)
    resolved = service._resolve_request(_request(checkpoint))
    assert resolved["cameras"] == (0, 1, 2, 3)
    assert resolved["reference_camera_id"] == "camera_0"
    assert resolved["window_start"] == 0
    assert resolved["window_length"] == 256
    assert resolved["scene_id"] == "scene_000000"


def test_resolve_request_rejects_invalid_settings(tmp_path: Path) -> None:
    service, checkpoint_root = _make_service(tmp_path)
    checkpoint = _v2_checkpoint(service, checkpoint_root)
    with pytest.raises(SceneCatalogError, match="out of range"):
        service._resolve_request(_request(checkpoint, cameras=(0, 1, 2, 9)))
    with pytest.raises(SceneCatalogError, match="distinct"):
        service._resolve_request(_request(checkpoint, cameras=(0, 0, 1)))
    with pytest.raises(SceneCatalogError, match="3〜4"):
        service._resolve_request(_request(checkpoint, cameras=(0, 1)))
    with pytest.raises(SceneCatalogError, match="reference_camera_id"):
        service._resolve_request(_request(checkpoint, reference_camera_id="camera_9"))
    with pytest.raises(SceneCatalogError, match="reference_camera_id"):
        service._resolve_request(_request(checkpoint, reference_camera_id=None))
    with pytest.raises(SceneCatalogError, match="max_seq_len"):
        service._resolve_request(_request(checkpoint, window_length=257))
    with pytest.raises(SceneCatalogError, match="num_frames"):
        service._resolve_request(_request(checkpoint, window_start=400))
    with pytest.raises(SceneCatalogError, match="num_frames"):
        service._resolve_request(
            _request(checkpoint, window_start=300, window_length=201)
        )
    with pytest.raises(SceneCatalogError, match="canonical_pose_source"):
        service._resolve_request(_request(checkpoint, canonical_pose_source="nope"))


def test_physical_checkpoint_rules(tmp_path: Path) -> None:
    service, checkpoint_root = _make_service(tmp_path)
    checkpoint = _write_checkpoint(
        checkpoint_root,
        "run_phys",
        model_name="plcs_multiview_axial",
        selector="physical_v1",
        input_profile="multiview",
        scene_dir="plcs/single_object",
        max_views=3,
    )
    with pytest.raises(SceneCatalogError, match="at most"):
        service._resolve_request(
            _request(
                checkpoint,
                family="single_object",
                cameras=(0, 1, 2, 3),
                reference_camera_id=None,
            )
        )
    with pytest.raises(SceneCatalogError, match="physical_v1"):
        service._resolve_request(
            _request(
                checkpoint,
                family="single_object",
                cameras=(0, 1),
                reference_camera_id="camera_0",
            )
        )
    resolved = service._resolve_request(
        _request(
            checkpoint,
            family="single_object",
            cameras=(0, 1, 2),
            reference_camera_id=None,
            window_length=300,
        )
    )
    assert resolved["reference_camera_id"] is None
    assert resolved["window_length"] == 300


def test_predict_rejects_cross_family_and_unknown_scene(tmp_path: Path) -> None:
    service, checkpoint_root = _make_service(tmp_path)
    checkpoint = _v2_checkpoint(service, checkpoint_root)
    with pytest.raises(SceneCatalogError, match="cannot consume family"):
        service.predict(_request(checkpoint, family="single_object"))
    with pytest.raises(SceneCatalogError, match="not listed"):
        service.predict(_request(checkpoint, scene="scene_099999"))


def test_predict_rejects_track_query_on_single_object_family(tmp_path: Path) -> None:
    service, checkpoint_root = _make_service(tmp_path)
    checkpoint = _write_checkpoint(
        checkpoint_root,
        "run_track",
        model_name="plcs_track_query",
        selector="physical_v1",
        input_profile=None,
        scene_dir="plcs/multi_object",
    )
    with pytest.raises(SceneCatalogError, match="cannot consume family"):
        service.predict(_request(checkpoint))


def _make_tracking_service(tmp_path: Path) -> tuple[InferenceService, Path]:
    data_root = tmp_path / "data" / "plcs"
    checkpoint_root = tmp_path / "outputs" / "plcs"
    checkpoint_root.mkdir(parents=True)
    _write_dataset(
        data_root,
        "multi_object",
        selector="physical_v1",
        objects="multi_object",
        cameras=6,
        persons=9,
        scenes=2,
        frames=600,
    )
    _write_dataset(
        data_root,
        "multi_object_broadcast",
        selector="physical_v1",
        objects="multi_object",
        cameras=2,
        persons=9,
        scenes=1,
        frames=600,
    )
    service = InferenceService(
        data_root=data_root, checkpoint_root=checkpoint_root, device="cpu"
    )
    return service, checkpoint_root


def _tracking_checkpoint(checkpoint_root: Path, run: str = "run_track") -> str:
    return _write_checkpoint(
        checkpoint_root,
        run,
        model_name="plcs_track_query",
        selector="physical_v1",
        input_profile=None,
        scene_dir="plcs/multi_object",
        num_queries=4,
        view_range=(3, 5),
        camera_candidates=[0, 1, 2, 3],
    )


def _tracking_request(checkpoint: str, **overrides: object) -> PredictionRequest:
    payload: dict[str, object] = {
        "checkpoint": checkpoint,
        "family": "multi_object",
        "scene": "scene_000000",
        "cameras": (0, 1, 2),
        "reference_camera_id": None,
        "window_start": 0,
        "window_length": 8,
        "canonical_pose_source": "gt",
        "device": "cpu",
    }
    payload.update(overrides)
    return PredictionRequest(**payload)  # type: ignore[arg-type]


def test_tracking_mode_and_views_are_validated(tmp_path: Path) -> None:
    service, checkpoint_root = _make_tracking_service(tmp_path)
    checkpoint = _tracking_checkpoint(checkpoint_root)
    resolved = service.validate_prediction_request(_tracking_request(checkpoint))
    assert resolved["mode"] == "tracking"
    # Training sampling ranges do not limit dynamic-view tracking models.
    broadcast = service.validate_prediction_request(
        _tracking_request(checkpoint, family="multi_object_broadcast", cameras=(0, 1))
    )
    assert broadcast["cameras"] == (0, 1)
    assert "training sampling range" in broadcast["warnings"][0]
    with pytest.raises(SceneCatalogError, match="camera_candidates"):
        service.validate_prediction_request(
            _tracking_request(checkpoint, cameras=(0, 1, 5))
        )


def test_tracking_rejects_reference_camera_on_physical_selector(
    tmp_path: Path,
) -> None:
    service, checkpoint_root = _make_tracking_service(tmp_path)
    checkpoint = _tracking_checkpoint(checkpoint_root)
    with pytest.raises(SceneCatalogError, match="physical_v1"):
        service.validate_prediction_request(
            _tracking_request(checkpoint, reference_camera_id="camera_1")
        )


def test_validate_prediction_request_rejects_unavailable_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, checkpoint_root = _make_tracking_service(tmp_path)
    checkpoint = _tracking_checkpoint(checkpoint_root)
    with pytest.raises(ValueError, match="Invalid device"):
        service.validate_prediction_request(
            _tracking_request(checkpoint, device="not-a-device")
        )
    # An explicit CUDA request never silently falls back to CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        service.validate_prediction_request(
            _tracking_request(checkpoint, device="cuda")
        )


def test_yaw_error_degrees_matches_known_angles() -> None:
    gt = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pred = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    error = yaw_error_degrees(gt, pred)
    assert error is not None
    assert error.shape == (2,)
    assert error[0] == pytest.approx(0.0, abs=1e-6)
    assert error[1] == pytest.approx(90.0, abs=1e-4)
    assert yaw_error_degrees(None, pred) is None


def test_ground_truth_joints_requires_coco17() -> None:
    joints: np.ndarray = np.zeros((4, 17, 3), dtype=np.float32)
    assert ground_truth_world_joints({"human_kp_3d": joints}).shape == (4, 17, 3)
    with pytest.raises(SceneCatalogError, match="human_kp_3d"):
        ground_truth_world_joints({})
    with pytest.raises(SceneCatalogError, match="17,3"):
        ground_truth_world_joints(
            {"human_kp_3d": np.zeros((4, 52, 3), dtype=np.float32)}
        )


def _synthetic_scene(frames: int, cameras: int) -> dict[str, object]:
    return {
        "meta": {"num_frames": frames, "fps": 120.0},
        "position": np.zeros((frames, 3), dtype=np.float32),
        "rotation": np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (frames, 1)),
        "canonical_pose_3d": np.zeros((frames, 73, 3), dtype=np.float32),
        "human_kp_3d": np.zeros((frames, 17, 3), dtype=np.float32),
        "num_cameras": cameras,
        "num_persons": 1,
        "court_keypoint_contract": None,
        "court_keypoint_validation": None,
        "cameras": [
            {
                "params": {},
                "human_kp_uv": np.zeros((frames, 17, 2), dtype=np.float32),
                "human_kp_vis": np.ones((frames, 17), dtype=bool),
                "human_visibility_ratio": 1.0,
                "court_kp_uv": np.zeros((frames, 20, 2), dtype=np.float32),
                "court_kp_vis": np.ones((frames, 20), dtype=bool),
                "court_visibility_count": 20.0,
                "court_view": None,
            }
            for _ in range(cameras)
        ],
    }


def test_window_scene_slices_every_track() -> None:
    scene = _synthetic_scene(frames=10, cameras=3)
    windowed = window_scene(scene, 2, 4)
    assert windowed["position"].shape == (4, 3)
    assert windowed["canonical_pose_3d"].shape == (4, 73, 3)
    assert windowed["human_kp_3d"].shape == (4, 17, 3)
    assert windowed["meta"]["num_frames"] == 4
    assert len(windowed["cameras"]) == 3
    assert windowed["cameras"][0]["human_kp_uv"].shape == (4, 17, 2)
    assert windowed["cameras"][0]["court_kp_uv"].shape == (4, 20, 2)
