from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.base.generate_dataset import (
    COURT_VIEW_METADATA_KEY,
    CourtKeypointContractMismatchError,
    InvalidCourtKeypointMetadataError,
    apply_court_view_record,
    build_court_view_record,
    resolve_court_keypoint_contract,
    validate_dataset_court_keypoint_contract,
)
from src.tasks.plcs.court_keypoint_contract import (
    PLCS_GENERATED_DATASET_SCHEMA_ID,
    PLCSCourtKeypointRuntimeConfig,
)
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene
from src.tasks.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.utils.schema.court_normalization import normalize_court_position


def _camera(camera_index: int, center_y: float, frames: int = 2) -> CameraData:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    view = build_court_view_record(
        camera_id=f"camera_{camera_index}",
        camera_center_court_m=(2.0 + camera_index, center_y, 4.0),
        contract=contract,
    )
    physical_uv = np.stack(
        [
            np.linspace(0.01, 0.20, 20, dtype=np.float32),
            np.linspace(0.21, 0.40, 20, dtype=np.float32),
        ],
        axis=-1,
    )
    physical_vis = np.arange(20) % 3 != 0
    disk_uv = apply_court_view_record(physical_uv, view, keypoint_axis=0)
    disk_vis = apply_court_view_record(physical_vis, view, keypoint_axis=0)
    return CameraData(
        camera_params={
            "C": [2.0 + camera_index, center_y, 4.0],
            "R": np.eye(3, dtype=np.float32).tolist(),
            "f": 1.0,
            "cx": 0.5,
            "cy": 0.5,
            "w": 1,
            "h": 1,
            "image_size": [1, 1],
        },
        human_kp_uv=np.full((frames, 17, 2), 0.5, dtype=np.float32),
        court_kp_uv=np.repeat(disk_uv[None], frames, axis=0),
        human_kp_vis=np.ones((frames, 17), dtype=np.bool_),
        court_kp_vis=np.repeat(disk_vis[None], frames, axis=0),
        human_visibility_ratio=1.0,
        court_visibility_count=float(disk_vis.sum()),
        court_view=view,
    )


def make_camera_view_scene(scene_id: str = "scene_000000") -> SceneData:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    frames = 2
    canonical: np.ndarray = np.zeros((frames, 17, 3), dtype=np.float32)
    canonical[..., 0] = 0.25
    world = canonical.copy()
    world[..., 0] += 1.0
    world[..., 1] += 2.0
    world[..., 2] += 0.5
    position = normalize_court_position(
        np.repeat(np.array([[1.0, 2.0, 0.5]], dtype=np.float32), frames, axis=0)
    )
    return SceneData(
        meta={
            "scene_id": scene_id,
            "motion_source": "fixture",
            "motion_category": "test",
            "gender": "neutral",
            "fps": 30,
            "num_frames": frames,
            "initial_position": [1.0, 2.0],
            "initial_yaw": 0.0,
            "num_cameras_sampled": 2,
        },
        position=np.asarray(position, dtype=np.float32),
        rotation=np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), frames, axis=0),
        canonical_pose_3d=canonical,
        cameras=[_camera(0, -12.0), _camera(1, 12.0)],
        num_persons=1,
        human_kp_3d=world,
        court_keypoint_contract=contract,
    )


def _make_physical_scene(scene_id: str = "scene_000000") -> SceneData:
    scene = make_camera_view_scene(scene_id)
    contract = resolve_court_keypoint_contract("physical_v1")
    for camera_index, camera in enumerate(scene.cameras):
        assert camera.court_view is not None
        physical_indices = np.argsort(camera.court_view.semantic_to_physical)
        camera.court_kp_uv = np.take(camera.court_kp_uv, physical_indices, axis=1)
        camera.court_kp_vis = np.take(camera.court_kp_vis, physical_indices, axis=1)
        camera.court_view = build_court_view_record(
            camera_id=f"camera_{camera_index}",
            camera_center_court_m=camera.camera_params["C"],
            contract=contract,
        )
    scene.court_keypoint_contract = contract
    return scene


def _make_legacy_scene(scene_id: str = "scene_000000") -> SceneData:
    scene = _make_physical_scene(scene_id)
    scene.court_keypoint_contract = None
    for camera in scene.cameras:
        camera.court_view = None
    return scene


def _writer_for_selector(root: Path, selector: str | None) -> PLCSDatasetWriter:
    if selector is None:
        return PLCSDatasetWriter(root, legacy_metadata_free_v1=True)
    return PLCSDatasetWriter(
        root,
        court_keypoint_contract=resolve_court_keypoint_contract(selector),
    )


def _scene_for_selector(selector: str | None, scene_id: str) -> SceneData:
    if selector is None:
        return _make_legacy_scene(scene_id)
    if selector == "physical_v1":
        return _make_physical_scene(scene_id)
    return make_camera_view_scene(scene_id)


def _scene_inventory(root: Path) -> dict[str, bytes]:
    scenes_dir = root / "scenes"
    return {
        str(path.relative_to(scenes_dir)): path.read_bytes()
        for path in sorted(scenes_dir.rglob("*"))
        if path.is_file()
    }


def write_camera_view_dataset(root: Path) -> Path:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    writer = PLCSDatasetWriter(
        root,
        court_keypoint_contract=contract,
    )
    writer.save_meta_json(config={})
    scene_path: Path = writer.save_scene(make_camera_view_scene())
    writer.save_meta_json(config={})
    (root / "train.txt").write_text("scene_000000\n", encoding="utf-8")
    return scene_path


def test_writer_publishes_task_schema_and_exact_camera_records(tmp_path: Path) -> None:
    scene_path = write_camera_view_dataset(tmp_path)
    root = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    scene = json.loads((scene_path / "meta.json").read_text(encoding="utf-8"))

    assert root["court_keypoints"]["dataset_schema_id"] == (
        PLCS_GENERATED_DATASET_SCHEMA_ID
    )
    assert scene["court_keypoint_views"][0]["semantic_to_physical"] == list(range(20))
    assert scene["court_keypoint_views"][1]["semantic_to_physical"] == [
        3,
        2,
        1,
        0,
        7,
        6,
        5,
        4,
        11,
        10,
        9,
        8,
        13,
        12,
        14,
        17,
        18,
        15,
        16,
        19,
    ]
    validated = validate_dataset_court_keypoint_contract(
        tmp_path,
        resolve_court_keypoint_contract("camera_view_v2"),
        expected_dataset_schema_id=PLCS_GENERATED_DATASET_SCHEMA_ID,
        scene_paths=(scene_path,),
    )
    assert not validated.legacy_metadata_free
    assert [view.camera_id for view in validated.scenes[0].court_views] == [
        "camera_0",
        "camera_1",
    ]


def test_writer_keeps_v1_array_names_and_physical_order(tmp_path: Path) -> None:
    contract = resolve_court_keypoint_contract("physical_v1")
    scene = make_camera_view_scene()
    physical_cameras: list[CameraData] = []
    for index, camera in enumerate(scene.cameras):
        center = camera.camera_params["C"]
        view = build_court_view_record(
            camera_id=f"camera_{index}",
            camera_center_court_m=center,
            contract=contract,
        )
        physical = np.arange(40, dtype=np.float32).reshape(20, 2) / 100.0
        physical_cameras.append(
            CameraData(
                **{
                    **camera.__dict__,
                    "court_kp_uv": np.repeat(physical[None], 2, axis=0),
                    "court_kp_vis": np.ones((2, 20), dtype=np.bool_),
                    "court_view": view,
                }
            )
        )
    scene.cameras = physical_cameras
    scene.court_keypoint_contract = contract
    writer = PLCSDatasetWriter(
        tmp_path,
        court_keypoint_contract=contract,
    )
    writer.save_meta_json(config={})
    scene_path = writer.save_scene(scene)
    np.testing.assert_array_equal(
        np.load(scene_path / "cam_0_court_kp_uv.npy"),
        physical_cameras[0].court_kp_uv,
    )
    np.testing.assert_array_equal(
        np.load(scene_path / "position.npy"),
        scene.position,
    )
    np.testing.assert_array_equal(
        np.load(scene_path / "rotation.npy"),
        scene.rotation,
    )
    np.testing.assert_array_equal(
        np.load(scene_path / "canonical_pose_3d.npy"),
        scene.canonical_pose_3d,
    )
    scalars = json.loads((scene_path / "scalars.json").read_text(encoding="utf-8"))
    assert scalars["cam_0_params"] == physical_cameras[0].camera_params
    assert not (scene_path / "cam_0_court_kp_visible.npy").exists()


def test_runtime_config_requires_explicit_court_selector() -> None:
    with pytest.raises(ValueError, match="court_keypoints"):
        PLCSCourtKeypointRuntimeConfig.from_config({})


def test_reader_rejects_camera_center_mismatch_before_array_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_path = write_camera_view_dataset(tmp_path)
    scalars_path = scene_path / "scalars.json"
    scalars = json.loads(scalars_path.read_text(encoding="utf-8"))
    scalars["cam_1_params"]["C"][1] = 15.0
    scalars_path.write_text(json.dumps(scalars), encoding="utf-8")

    def _forbid_array_payload(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("PLCS arrays were read before CourtKP20 headers.")

    monkeypatch.setattr(np, "load", _forbid_array_payload)
    with pytest.raises(CourtKeypointContractMismatchError, match="does not match"):
        load_scene(
            scene_path,
            court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
        )


def test_reader_rejects_unstable_camera_slot_id_before_array_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_path = write_camera_view_dataset(tmp_path)
    metadata_path = scene_path / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    views = metadata[COURT_VIEW_METADATA_KEY]
    assert isinstance(views, list) and isinstance(views[0], dict)
    views[0]["camera_id"] = "camera_alias"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def _forbid_array_payload(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("PLCS arrays were read before CourtKP20 headers.")

    monkeypatch.setattr(np, "load", _forbid_array_payload)
    with pytest.raises(
        CourtKeypointContractMismatchError,
        match="camera slot 0 requires stable ID 'camera_0'",
    ):
        load_scene(
            scene_path,
            court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
        )


def test_reader_rejects_extra_camera_parameter_slot_before_array_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_path = write_camera_view_dataset(tmp_path)
    scalars_path = scene_path / "scalars.json"
    scalars = json.loads(scalars_path.read_text(encoding="utf-8"))
    scalars["cam_2_params"] = scalars["cam_0_params"]
    scalars_path.write_text(json.dumps(scalars), encoding="utf-8")

    def _forbid_array_payload(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("PLCS arrays were read before CourtKP20 headers.")

    monkeypatch.setattr(np, "load", _forbid_array_payload)
    with pytest.raises(
        CourtKeypointContractMismatchError,
        match="camera parameter slots must exactly match",
    ):
        load_scene(
            scene_path,
            court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
        )


@pytest.mark.parametrize("mutation", ["missing", "renamed"])
def test_reader_rejects_missing_or_renamed_camera_parameter_slot_before_array_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    scene_path = write_camera_view_dataset(tmp_path)
    scalars_path = scene_path / "scalars.json"
    scalars = json.loads(scalars_path.read_text(encoding="utf-8"))
    removed = scalars.pop("cam_1_params")
    if mutation == "renamed":
        scalars["cam_01_params"] = removed
    scalars_path.write_text(json.dumps(scalars), encoding="utf-8")

    def _forbid_array_payload(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("PLCS arrays were read before CourtKP20 headers.")

    monkeypatch.setattr(np, "load", _forbid_array_payload)
    with pytest.raises(
        CourtKeypointContractMismatchError,
        match="camera parameter slots must exactly match",
    ):
        load_scene(
            scene_path,
            court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
        )


@pytest.mark.parametrize("malformed", [None, [], "not-a-camera-object"])
def test_reader_rejects_malformed_camera_parameter_slot_before_array_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: object,
) -> None:
    scene_path = write_camera_view_dataset(tmp_path)
    scalars_path = scene_path / "scalars.json"
    scalars = json.loads(scalars_path.read_text(encoding="utf-8"))
    scalars["cam_1_params"] = malformed
    scalars_path.write_text(json.dumps(scalars), encoding="utf-8")

    def _forbid_array_payload(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("PLCS arrays were read before CourtKP20 headers.")

    monkeypatch.setattr(np, "load", _forbid_array_payload)
    with pytest.raises(
        InvalidCourtKeypointMetadataError,
        match="expected a camera parameter object",
    ):
        load_scene(
            scene_path,
            court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
        )


def test_writer_rejects_camera_center_mismatch_before_array_payload(
    tmp_path: Path,
) -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    scene = make_camera_view_scene()
    scene.cameras[0].camera_params["C"] = [2.0, -15.0, 4.0]
    writer = PLCSDatasetWriter(
        tmp_path,
        court_keypoint_contract=contract,
    )
    writer.save_meta_json(config={})

    with pytest.raises(CourtKeypointContractMismatchError, match="does not match"):
        writer.save_scene(scene)

    assert not (tmp_path / "scenes" / "scene_000000").exists()


@pytest.mark.parametrize("explicit_none", [False, True])
def test_writer_requires_an_explicit_output_mode_for_v2_scene(
    tmp_path: Path,
    explicit_none: bool,
) -> None:
    scene = make_camera_view_scene()

    with pytest.raises(
        CourtKeypointContractMismatchError,
        match="requires an explicit CourtKP20 contract",
    ):
        if explicit_none:
            PLCSDatasetWriter(tmp_path, court_keypoint_contract=None)
        else:
            PLCSDatasetWriter(tmp_path)

    assert scene.court_keypoint_contract == resolve_court_keypoint_contract(
        "camera_view_v2"
    )
    assert list(tmp_path.iterdir()) == []


def test_writer_rejects_contract_mismatch_before_creating_scene(
    tmp_path: Path,
) -> None:
    writer = PLCSDatasetWriter(
        tmp_path,
        court_keypoint_contract=resolve_court_keypoint_contract("physical_v1"),
    )
    writer.save_meta_json(config={})

    with pytest.raises(CourtKeypointContractMismatchError, match="does not match"):
        writer.save_scene(make_camera_view_scene())

    assert not (tmp_path / "scenes" / "scene_000000").exists()


@pytest.mark.parametrize("selector", ["physical_v1", "camera_view_v2"])
def test_explicit_writer_rejects_scene_before_root_contract_publication(
    tmp_path: Path,
    selector: str,
) -> None:
    writer = _writer_for_selector(tmp_path, selector)

    with pytest.raises(
        CourtKeypointContractMismatchError,
        match="root metadata to be published before saving a scene",
    ):
        writer.save_scene(_scene_for_selector(selector, "scene_000000"))

    assert not (tmp_path / "meta.json").exists()
    assert not (tmp_path / "scenes" / "scene_000000").exists()
    assert writer.scene_records == []
    assert writer.scene_counter == 0


def test_legacy_writer_rejects_scene_and_camera_contract_records_atomically(
    tmp_path: Path,
) -> None:
    scene_with_contract = make_camera_view_scene()
    writer = PLCSDatasetWriter(tmp_path, legacy_metadata_free_v1=True)

    with pytest.raises(
        CourtKeypointContractMismatchError,
        match="scene CourtKP20 contract to be absent",
    ):
        writer.save_scene(scene_with_contract)
    assert not (tmp_path / "scenes" / "scene_000000").exists()

    scene_with_camera_record = make_camera_view_scene("scene_000001")
    scene_with_camera_record.court_keypoint_contract = None
    with pytest.raises(
        CourtKeypointContractMismatchError,
        match="every camera CourtKP20 record to be absent",
    ):
        writer.save_scene(scene_with_camera_record)
    assert not (tmp_path / "scenes" / "scene_000001").exists()


def test_explicit_legacy_writer_saves_metadata_free_v1_scene(tmp_path: Path) -> None:
    scene = make_camera_view_scene()
    scene.court_keypoint_contract = None
    for camera in scene.cameras:
        assert camera.court_view is not None
        physical_indices = np.argsort(camera.court_view.semantic_to_physical)
        camera.court_kp_uv = np.take(camera.court_kp_uv, physical_indices, axis=1)
        camera.court_kp_vis = np.take(camera.court_kp_vis, physical_indices, axis=1)
        camera.court_view = None
    writer = PLCSDatasetWriter(tmp_path, legacy_metadata_free_v1=True)

    scene_path = writer.save_scene(scene)
    writer.save_meta_json()

    root_metadata = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    scene_metadata = json.loads((scene_path / "meta.json").read_text(encoding="utf-8"))
    assert "court_keypoints" not in root_metadata
    assert "court_keypoints" not in scene_metadata
    assert COURT_VIEW_METADATA_KEY not in scene_metadata


def test_writer_modes_are_mutually_exclusive_without_filesystem_side_effects(
    tmp_path: Path,
) -> None:
    with pytest.raises(CourtKeypointContractMismatchError, match="mutually exclusive"):
        PLCSDatasetWriter(
            tmp_path,
            court_keypoint_contract=resolve_court_keypoint_contract("physical_v1"),
            legacy_metadata_free_v1=True,
        )

    assert not (tmp_path / "scenes").exists()


@pytest.mark.parametrize(
    ("existing_selector", "requested_selector"),
    [
        ("physical_v1", "camera_view_v2"),
        ("camera_view_v2", "physical_v1"),
        ("physical_v1", None),
        ("camera_view_v2", None),
        (None, "physical_v1"),
        (None, "camera_view_v2"),
    ],
)
def test_writer_rejects_cross_contract_append_without_mutating_dataset(
    tmp_path: Path,
    existing_selector: str | None,
    requested_selector: str | None,
) -> None:
    _writer_for_selector(tmp_path, existing_selector).save_meta_json()
    root_metadata_before = (tmp_path / "meta.json").read_bytes()

    with pytest.raises(CourtKeypointContractMismatchError):
        _writer_for_selector(tmp_path, requested_selector)

    assert (tmp_path / "meta.json").read_bytes() == root_metadata_before
    assert not (tmp_path / "scenes" / "scene_000000").exists()


@pytest.mark.parametrize("selector", ["camera_view_v2", "physical_v1"])
def test_compatible_explicit_writer_reopen_rejects_without_mutation(
    tmp_path: Path,
    selector: str,
) -> None:
    writer = _writer_for_selector(tmp_path, selector)
    writer.save_meta_json()
    writer.save_scene(_scene_for_selector(selector, "scene_000000"))
    writer.save_meta_json()
    root_metadata_before = (tmp_path / "meta.json").read_bytes()
    scene_inventory_before = _scene_inventory(tmp_path)

    with pytest.raises(FileExistsError, match="does not support reopening"):
        _writer_for_selector(tmp_path, selector)

    assert (tmp_path / "meta.json").read_bytes() == root_metadata_before
    assert _scene_inventory(tmp_path) == scene_inventory_before


def test_compatible_legacy_writer_reopen_rejects_without_mutation(
    tmp_path: Path,
) -> None:
    writer = _writer_for_selector(tmp_path, None)
    writer.save_scene(_scene_for_selector(None, "scene_000000"))
    writer.save_meta_json()
    root_metadata_before = (tmp_path / "meta.json").read_bytes()
    scene_inventory_before = _scene_inventory(tmp_path)

    with pytest.raises(FileExistsError, match="does not support reopening"):
        _writer_for_selector(tmp_path, None)

    assert (tmp_path / "meta.json").read_bytes() == root_metadata_before
    assert _scene_inventory(tmp_path) == scene_inventory_before


@pytest.mark.parametrize("selector", ["camera_view_v2", "physical_v1", None])
def test_writer_allows_compatible_empty_preinitialized_root(
    tmp_path: Path,
    selector: str | None,
) -> None:
    _writer_for_selector(tmp_path, selector).save_meta_json()

    writer = _writer_for_selector(tmp_path, selector)
    scene_path = writer.save_scene(
        _scene_for_selector(selector, "scene_000000")
    )
    writer.save_meta_json()

    assert scene_path.is_dir()
    root_metadata = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert root_metadata["stats"]["total_scenes"] == 1
    assert [record["scene_id"] for record in root_metadata["scenes"]] == [
        "scene_000000"
    ]


def test_same_writer_saves_multiple_scenes_and_refreshes_root_metadata(
    tmp_path: Path,
) -> None:
    writer = _writer_for_selector(tmp_path, "camera_view_v2")
    writer.save_meta_json()
    writer.save_scene(make_camera_view_scene("scene_000000"))
    writer.save_meta_json()
    writer.save_scene(make_camera_view_scene("scene_000001"))
    writer.save_meta_json()

    root_metadata = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert root_metadata["stats"]["total_scenes"] == 2
    assert [record["scene_id"] for record in root_metadata["scenes"]] == [
        "scene_000000",
        "scene_000001",
    ]


def test_root_metadata_refresh_is_atomic_for_versioned_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _writer_for_selector(tmp_path, "camera_view_v2")
    writer.save_meta_json(config={"phase": "initial"})
    writer.save_scene(make_camera_view_scene())
    root_path = tmp_path / "meta.json"
    root_metadata_before = root_path.read_bytes()
    original_replace = Path.replace

    def _interrupt_root_replace(source: Path, target: Path) -> Path:
        if source.name == ".meta.json.tmp":
            raise OSError("simulated interruption")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", _interrupt_root_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        writer.save_meta_json(config={"phase": "refresh"})

    assert root_path.read_bytes() == root_metadata_before
    assert not (tmp_path / ".meta.json.tmp").exists()


@pytest.mark.parametrize("selector", ["camera_view_v2", "physical_v1", None])
@pytest.mark.parametrize("nonempty_field", ["scenes", "stats"])
def test_writer_rejects_nonempty_root_metadata_without_scene_files(
    tmp_path: Path,
    selector: str | None,
    nonempty_field: str,
) -> None:
    _writer_for_selector(tmp_path, selector).save_meta_json()
    root_path = tmp_path / "meta.json"
    root_metadata = json.loads(root_path.read_text(encoding="utf-8"))
    if nonempty_field == "scenes":
        root_metadata["scenes"] = [{"scene_id": "missing_scene"}]
    else:
        root_metadata["stats"]["total_scenes"] = 1
    root_path.write_text(json.dumps(root_metadata, indent=2), encoding="utf-8")
    root_metadata_before = root_path.read_bytes()

    with pytest.raises(FileExistsError, match="does not support reopening"):
        _writer_for_selector(tmp_path, selector)

    assert root_path.read_bytes() == root_metadata_before
    assert list((tmp_path / "scenes").iterdir()) == []
