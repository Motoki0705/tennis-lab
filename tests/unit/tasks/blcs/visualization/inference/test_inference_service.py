"""Unit tests for the BLCS inference service catalog, scenes, and inference."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.base.generate_dataset import (
    CourtKeypointArtifactMetadata,
    CourtKeypointContract,
    build_court_view_record,
    build_physical_court_provenance,
    inject_court_keypoint_artifact_metadata,
    inject_scene_court_keypoint_metadata,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCS_DATASET_SCHEMA_ID
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    BLCSTrajectoryPrediction,
    blcs_reference_metadata_from_batch,
)
from src.tasks.blcs.model_io.checkpoints import (
    resolve_blcs_track_query_reference_contract,
)
from src.tasks.blcs.visualization.inference.service import (
    InferenceService,
    derive_checkpoint_metadata,
    forms_for_model,
)
from src.tasks.blcs.visualization.inference.tracking import (
    build_tracking_input,
    tracking_metrics,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import court_coordinate_normalization_metadata

REPO_ROOT = Path("/home/kamimura/projects/tennis-lab")
REAL_DATA_ROOT = REPO_ROOT / "data"
REAL_CHECKPOINTS_ROOT = REPO_ROOT / "ckpt" / "blcs"
_AUGMENTATION_CONFIG = (
    Path(__file__).resolve().parents[6]
    / "src/tasks/blcs/configs/data/_augmentation.yaml"
)

SINGLE_FORMS = ("single_object", "single_object_broadcast")
REFERENCE_SINGLE_FORMS = ("single_object_camera_view_v2",)
MULTI_FORMS = ("multi_object", "multi_object_broadcast")
REFERENCE_MULTI_FORMS = ("multi_object_camera_view_v2",)


# --------------------------------------------------------------------- configs


def _trajectory_config(
    *,
    name: str,
    selector: str,
    scene_dir: str = "blcs/single_object_camera_view_v2",
    input_profile: str = "multiview",
    num_court_tokens: int = 14,
    max_seq_len: int = 256,
    max_num_cameras: int = 4,
    seq_len_range: Sequence[int] = (128, 128),
    num_views_range: Sequence[int] = (3, 4),
) -> dict[str, Any]:
    model: dict[str, Any] = {
        "name": name,
        "io": {"input_profile": input_profile},
        "num_court_tokens": num_court_tokens,
        "max_seq_len": max_seq_len,
    }
    if input_profile == "multiview":
        model["max_num_cameras"] = max_num_cameras
    return {
        "model": model,
        "court_keypoints": {"selector": selector},
        "data": {
            "scene_dir": scene_dir,
            "seq_len_range": list(seq_len_range),
            "num_views_range": list(num_views_range),
        },
    }


def _tracking_data_config(
    *,
    selector: str,
    scene_dir: str,
    reference: bool,
    num_queries: int = 2,
    num_views_range: Sequence[int] = (6, 6),
) -> dict[str, Any]:
    model: dict[str, Any] = {
        "name": "blcs_track_query_reference" if reference else "blcs_track_query",
        "num_queries": num_queries,
    }
    if reference:
        model.update(
            {
                "target_frame_contract": "reference_camera_court_rzpi_v1",
                "track_query_rope_contract": "time_camera_reference_selector_v1",
                "reference_selector_mode": "reference",
            }
        )
    return {
        "court_keypoints": {"selector": selector},
        "model": model,
        "data": {
            "seq_len_range": [6, 6],
            "num_views_range": list(num_views_range),
            "camera_mode": "first",
            "scene_dir": scene_dir,
            "lifecycle": {"pack_to_query_slots": True, "min_reuse_gap_frames": 0},
            "association": {
                "max_distance": 0.04,
                "max_missed_frames": 2,
                "min_reuse_gap_frames": 4,
                "use_velocity_prediction": True,
                "min_common_keypoints": 1,
                "cost_reduction": "mean",
                "overflow_policy": "error",
            },
            "augmentation": OmegaConf.load(_AUGMENTATION_CONFIG).augmentation,
        },
    }


def _write_checkpoint(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hyper_parameters": {"config": config}}, path)


# ------------------------------------------------------------------ scene data


def _write_form(
    data_root: Path,
    form: str,
    *,
    scene_ids: Sequence[str],
    num_cameras: int,
    num_balls: int,
    frames: int,
    split_files: bool = True,
    present: bool = False,
) -> Path:
    contract = resolve_court_keypoint_contract(
        "camera_view_v2" if form.endswith("_camera_view_v2") else "physical_v1"
    )
    form_dir = data_root / "blcs" / form
    (form_dir / "scenes").mkdir(parents=True)
    artifact = CourtKeypointArtifactMetadata.from_contract(
        contract,
        dataset_schema_id=BLCS_DATASET_SCHEMA_ID,
    )
    (form_dir / "meta.json").write_text(
        json.dumps(
            inject_court_keypoint_artifact_metadata(
                {},
                artifact,
                location=str(form_dir),
            )
        ),
        encoding="utf-8",
    )
    for scene_id in scene_ids:
        _write_scene(
            form_dir,
            scene_id,
            contract=contract,
            artifact=artifact,
            num_cameras=num_cameras,
            num_balls=num_balls,
            frames=frames,
            present=present,
        )
    if split_files:
        (form_dir / "test.txt").write_text(
            "\n".join(scene_ids[:2]) + "\n",
            encoding="utf-8",
        )
        (form_dir / "split_info.json").write_text(
            json.dumps({"n_scenes": {"train": 0, "val": 0, "test": len(scene_ids)}}),
            encoding="utf-8",
        )
    return form_dir


def _write_scene(
    form_dir: Path,
    scene_id: str,
    *,
    contract: CourtKeypointContract,
    artifact: CourtKeypointArtifactMetadata,
    num_cameras: int,
    num_balls: int,
    frames: int,
    present: bool,
    fps: float = 30.0,
) -> None:
    scene_dir = form_dir / "scenes" / scene_id
    scene_dir.mkdir()
    records = [
        build_court_view_record(
            camera_id=f"cam_{index}",
            camera_center_court_m=(float(index), 16.0, 3.0),
            contract=contract,
        )
        for index in range(num_cameras)
    ]
    meta: dict[str, Any] = {
        "scene_id": scene_id,
        "rally_length": 1,
        "end_reason": "out",
        "winner_side": None,
        "shots": [],
        "fps_out": fps,
        "sim_fps": 240,
        "num_frames": frames,
        "num_cameras_sampled": num_cameras,
        "num_cameras": num_cameras,
        "court_coordinate_normalization": court_coordinate_normalization_metadata(),
        "physics_config": {},
        "court_config": {"net_post_offset_x": 0.914},
        "track_instances": [],
    }
    meta = inject_scene_court_keypoint_metadata(
        meta,
        artifact,
        records,
        location=str(scene_dir),
    )
    (scene_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    slots = 10 if num_balls > 1 else 1
    world: np.ndarray = np.arange(frames * slots * 3, dtype=np.float32).reshape(
        frames, slots, 3
    )
    np.save(scene_dir / "ball_pos_world.npy", world)
    np.save(scene_dir / "ball_pos_norm.npy", world / 11.885)
    np.save(scene_dir / "ball_vel_world.npy", np.zeros_like(world))
    np.save(scene_dir / "ball_vel_norm.npy", np.zeros_like(world))
    if present:
        presence: np.ndarray = np.zeros((frames, slots), dtype=bool)
        presence[:, 0] = True
        presence[:-1, 1] = True
        np.save(scene_dir / "ball_present.npy", presence)
    for index in range(num_cameras):
        prefix = f"cam_{index}_"
        np.save(
            scene_dir / f"{prefix}ball_uv.npy",
            np.full((frames, slots, 2), 0.5, dtype=np.float32),
        )
        np.save(
            scene_dir / f"{prefix}ball_vis.npy",
            np.ones((frames, slots), dtype=bool),
        )
        np.save(
            scene_dir / f"{prefix}ball_visibility_ratio.npy",
            np.array(0.75, dtype=np.float32),
        )
        np.save(
            scene_dir / f"{prefix}court_kp_uv.npy",
            np.full((20, 2), 0.4, dtype=np.float32),
        )
        np.save(
            scene_dir / f"{prefix}court_kp_vis.npy",
            np.ones((20,), dtype=bool),
        )
        np.save(
            scene_dir / f"{prefix}court_visibility_count.npy",
            np.array(20.0, dtype=np.float32),
        )
    scalars: dict[str, Any] = {"num_cameras": num_cameras, "num_balls": num_balls}
    for index, record in enumerate(records):
        scalars[f"cam_{index}_params"] = {
            "C": list(record.camera_center_court_m),
            "R": np.eye(3).tolist(),
            "f": 1108.5,
            "cx": 640.0,
            "cy": 360.0,
            "w": 1280,
            "h": 720,
        }
    (scene_dir / "scalars.json").write_text(json.dumps(scalars), encoding="utf-8")


def _build_dataset(tmp_path: Path) -> None:
    _write_form(
        tmp_path / "data",
        "single_object",
        scene_ids=["scene_000000", "scene_000001"],
        num_cameras=6,
        num_balls=1,
        frames=6,
    )
    _write_form(
        tmp_path / "data",
        "single_object_broadcast",
        scene_ids=["scene_000000", "scene_000001"],
        num_cameras=2,
        num_balls=1,
        frames=6,
    )
    _write_form(
        tmp_path / "data",
        "single_object_camera_view_v2",
        scene_ids=["scene_000000", "scene_000001"],
        num_cameras=4,
        num_balls=1,
        frames=6,
    )
    for form in MULTI_FORMS:
        _write_form(
            tmp_path / "data",
            form,
            scene_ids=["scene_000000", "scene_000001"],
            num_cameras=6 if form == "multi_object" else 2,
            num_balls=2,
            frames=6,
            present=True,
        )
    _write_form(
        tmp_path / "data",
        "multi_object_camera_view_v2",
        scene_ids=["scene_000000", "scene_000001"],
        num_cameras=6,
        num_balls=2,
        frames=6,
        present=True,
    )


def _service(tmp_path: Path, **kwargs: Any) -> InferenceService:
    return InferenceService(
        tmp_path / "outputs",
        tmp_path / "ckpt",
        tmp_path / "data",
        **kwargs,
    )


def _fail_load(*args: object, **kwargs: object) -> None:
    pytest.fail("no predictor may be loaded during request validation")


# ------------------------------------------------------------------- metadata


@pytest.mark.parametrize(
    ("name", "selector", "family", "mode", "reference", "profile", "scene_dir"),
    [
        (
            "blcs",
            "physical_v1",
            "trajectory",
            "single",
            False,
            "single",
            "blcs/single_object",
        ),
        (
            "blcs_multiview_axial",
            "physical_v1",
            "trajectory",
            "single",
            False,
            "multiview",
            "blcs/single_object",
        ),
        (
            "blcs_multiview_axial_reference",
            "camera_view_v2",
            "trajectory",
            "single",
            True,
            "multiview",
            "blcs/single_object_camera_view_v2",
        ),
        (
            "blcs_track_query",
            "physical_v1",
            "tracking",
            "multi",
            False,
            "tracking",
            "blcs/multi_object",
        ),
        (
            "blcs_track_query_reference",
            "camera_view_v2",
            "tracking",
            "multi",
            True,
            "tracking",
            "blcs/multi_object_camera_view_v2",
        ),
    ],
)
def test_derive_checkpoint_metadata_classifies_every_family(
    name: str,
    selector: str,
    family: str,
    mode: str,
    reference: bool,
    profile: str,
    scene_dir: str,
) -> None:
    if family == "tracking":
        config: dict[str, Any] = _tracking_data_config(
            selector=selector,
            scene_dir=scene_dir,
            reference=reference,
        )
        config["model"]["num_queries"] = 4
    else:
        config = _trajectory_config(
            name=name,
            selector=selector,
            scene_dir=scene_dir,
            input_profile=profile,
        )
    metadata = derive_checkpoint_metadata(config)
    assert metadata["model_family"] == family
    assert metadata["object_mode"] == mode
    assert metadata["reference"] is reference
    assert metadata["input_profile"] == profile
    assert metadata["runnable"] is True
    assert metadata["unavailable_reason"] is None
    assert metadata["seq_len"] == (6 if family == "tracking" else 128)


def test_derive_checkpoint_metadata_rejects_selector_mismatch() -> None:
    with pytest.raises(ValueError, match="requires court_keypoints.selector"):
        derive_checkpoint_metadata(
            _trajectory_config(
                name="blcs_multiview_axial_reference",
                selector="physical_v1",
                scene_dir="blcs/single_object",
            )
        )
    with pytest.raises(ValueError, match="requires court_keypoints.selector"):
        derive_checkpoint_metadata(
            _trajectory_config(
                name="blcs",
                selector="camera_view_v2",
                scene_dir="blcs/single_object_camera_view_v2",
                input_profile="single",
            )
        )


def test_derive_checkpoint_metadata_rejects_unknown_model_and_scene_dir() -> None:
    unknown = _trajectory_config(name="blcs_unknown", selector="physical_v1")
    with pytest.raises(ValueError, match="Unsupported BLCS checkpoint model name"):
        derive_checkpoint_metadata(unknown)
    mismatched = _trajectory_config(
        name="blcs",
        selector="physical_v1",
        scene_dir="blcs/multi_object",
        input_profile="single",
    )
    with pytest.raises(ValueError, match="contradicts model.name"):
        derive_checkpoint_metadata(mismatched)


@pytest.mark.parametrize(
    ("name", "selector", "expected"),
    [
        ("blcs", "physical_v1", SINGLE_FORMS),
        ("blcs_multiview_axial", "physical_v1", SINGLE_FORMS),
        ("blcs_multiview_axial_reference", "camera_view_v2", REFERENCE_SINGLE_FORMS),
        ("blcs_track_query", "physical_v1", MULTI_FORMS),
        ("blcs_track_query_reference", "camera_view_v2", REFERENCE_MULTI_FORMS),
    ],
)
def test_forms_for_model_maps_every_model_to_its_forms(
    name: str,
    selector: str,
    expected: tuple[str, ...],
) -> None:
    assert forms_for_model(name, selector) == expected


# -------------------------------------------------------------------- catalog


def test_catalog_classifies_all_six_forms_and_runnable_checkpoints(
    tmp_path: Path,
) -> None:
    _build_dataset(tmp_path)
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "single.ckpt",
        _trajectory_config(
            name="blcs",
            selector="physical_v1",
            scene_dir="blcs/single_object",
            input_profile="single",
        ),
    )
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "axial.ckpt",
        _trajectory_config(
            name="blcs_multiview_axial",
            selector="physical_v1",
            scene_dir="blcs/single_object",
        ),
    )
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "reference.ckpt",
        _trajectory_config(
            name="blcs_multiview_axial_reference",
            selector="camera_view_v2",
            scene_dir="blcs/single_object_camera_view_v2",
        ),
    )
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "track_physical.ckpt",
        _tracking_data_config(
            selector="physical_v1",
            scene_dir="blcs/multi_object",
            reference=False,
        ),
    )
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "track_reference.ckpt",
        _tracking_data_config(
            selector="camera_view_v2",
            scene_dir="blcs/multi_object_camera_view_v2",
            reference=True,
        ),
    )
    catalog = _service(tmp_path).catalog()
    forms = {form["id"]: form for form in catalog["scene_forms"]}
    assert set(forms) == {
        "single_object",
        "single_object_broadcast",
        "single_object_camera_view_v2",
        "multi_object",
        "multi_object_broadcast",
        "multi_object_camera_view_v2",
    }
    assert forms["single_object"]["object_mode"] == "single"
    assert forms["single_object"]["reference"] is False
    assert forms["single_object"]["num_cameras"] == [6, 6]
    assert forms["multi_object"]["object_mode"] == "multi"
    assert forms["multi_object"]["num_cameras"] == [6, 6]
    assert forms["multi_object_camera_view_v2"]["reference"] is True
    assert forms["single_object_camera_view_v2"]["reference"] is True

    by_id = {entry["id"]: entry for entry in catalog["checkpoints"]}
    assert all(entry["runnable"] for entry in by_id.values()), by_id
    assert by_id["checkpoints:blcs/single.ckpt"]["allowed_forms"] == list(SINGLE_FORMS)
    assert by_id["checkpoints:blcs/axial.ckpt"]["allowed_forms"] == list(SINGLE_FORMS)
    assert by_id["checkpoints:blcs/reference.ckpt"]["allowed_forms"] == list(
        REFERENCE_SINGLE_FORMS
    )
    assert by_id["checkpoints:blcs/track_physical.ckpt"]["allowed_forms"] == list(
        MULTI_FORMS
    )
    assert by_id["checkpoints:blcs/track_reference.ckpt"]["allowed_forms"] == list(
        REFERENCE_MULTI_FORMS
    )
    assert by_id["checkpoints:blcs/track_physical.ckpt"]["num_queries"] == 2
    assert by_id["checkpoints:blcs/track_reference.ckpt"]["reference"] is True
    assert by_id["checkpoints:blcs/reference.ckpt"]["num_views_range"] == (3, 4)
    assert catalog["world"]["units"] == "metres"


def test_catalog_reports_unreadable_checkpoint_without_failing(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    broken = tmp_path / "ckpt" / "blcs" / "broken.ckpt"
    broken.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"not_hyper_parameters": True}, broken)
    entry = _service(tmp_path).catalog()["checkpoints"][0]
    assert entry["runnable"] is False
    assert entry["unavailable_reason"]


def test_catalog_reports_missing_roots(tmp_path: Path) -> None:
    (tmp_path / "data" / "blcs").mkdir(parents=True)
    service = _service(tmp_path)
    roots = service.catalog()["roots"]
    assert roots == [
        {"id": "outputs", "path": str(tmp_path / "outputs"), "exists": False},
        {"id": "checkpoints", "path": str(tmp_path / "ckpt"), "exists": False},
    ]


# ---------------------------------------------------------------------- scenes


def test_scenes_listing_and_ground_truth(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    service = _service(tmp_path)
    default = service.scenes(form="single_object_camera_view_v2")
    assert default["split"] == "test"
    assert default["total"] == 2
    assert default["scenes"][0]["num_cameras"] == 4
    assert default["scenes"][0]["num_balls"] == 1

    multi = service.scenes(form="multi_object")
    assert multi["split"] == "test"
    assert multi["scenes"][0]["num_balls"] == 2

    single = service.scene(form="single_object", scene_id="scene_000000")
    assert single["frame_count"] == 6
    assert single["gt"]["tracks"] == 1
    assert len(single["gt"]["positions"]) == 18
    assert single["gt"]["active"] == [1] * 6
    assert single["reference_candidate_ids"] == [f"cam_{i}" for i in range(6)]

    multi_scene = service.scene(form="multi_object", scene_id="scene_000000")
    assert multi_scene["gt"]["tracks"] == 2
    assert len(multi_scene["gt"]["positions"]) == 6 * 2 * 3
    assert multi_scene["gt"]["active"] == [1, 1] * 5 + [1, 0]


def test_scene_rejects_path_traversal_and_unknown_form(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="Unknown scene form"):
        service.scenes(form="does_not_exist")
    with pytest.raises(ValueError, match="single 'scene_"):
        service.scene(form="single_object", scene_id="../scene_000000")
    with pytest.raises(FileNotFoundError, match="Scene directory is missing"):
        service.scene(form="single_object", scene_id="scene_999999")


# ------------------------------------------------------- request validation


def _single_checkpoint(tmp_path: Path) -> str:
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "single.ckpt",
        _trajectory_config(
            name="blcs",
            selector="physical_v1",
            scene_dir="blcs/single_object",
            input_profile="single",
            max_seq_len=8,
            seq_len_range=(4, 4),
            num_views_range=(1, 1),
        ),
    )
    return "checkpoints:blcs/single.ckpt"


def test_broadcast_views_are_not_rejected_by_training_sampling_range(
    tmp_path: Path,
) -> None:
    _build_dataset(tmp_path)
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "axial.ckpt",
        _trajectory_config(
            name="blcs_multiview_axial",
            selector="physical_v1",
            scene_dir="blcs/single_object",
            num_views_range=(3, 4),
        ),
    )
    result = _service(tmp_path).validate_inference_request(
        checkpoint="checkpoints:blcs/axial.ckpt",
        form="single_object_broadcast",
        scene_id="scene_000000",
        cameras=[0, 1],
        device="cpu",
    )
    assert result.cameras == (0, 1)
    assert any("sampling range" in warning for warning in result.warnings)


def test_validate_inference_request_resolves_without_loading_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_dataset(tmp_path)
    reference = _single_checkpoint(tmp_path)
    service = _service(tmp_path)
    monkeypatch.setattr(service, "_load_predictor", _fail_load)
    resolved = service.validate_inference_request(
        checkpoint=reference,
        form="single_object",
        scene_id="scene_000000",
        cameras=[0],
        device="cpu",
    )
    assert resolved.cameras == (0,)
    assert resolved.window == 4
    assert resolved.reference_camera_id is None
    assert resolved.device_key == "cpu"
    assert resolved.warnings == ()


def test_validate_inference_request_rejects_bad_requests_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_dataset(tmp_path)
    reference = _single_checkpoint(tmp_path)
    service = _service(tmp_path)
    monkeypatch.setattr(service, "_load_predictor", _fail_load)
    base: dict[str, Any] = {
        "checkpoint": reference,
        "form": "single_object",
        "scene_id": "scene_000000",
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        service.validate_inference_request(**{**base, "checkpoint": "missing.ckpt"})
    with pytest.raises(ValueError, match="Unknown scene form"):
        service.validate_inference_request(**{**base, "form": "nope"})
    with pytest.raises(ValueError, match="cannot run form"):
        service.validate_inference_request(**{**base, "form": "multi_object"})
    with pytest.raises(ValueError, match="outside 0"):
        service.validate_inference_request(**{**base, "cameras": [9]})
    with pytest.raises(ValueError, match="exactly one camera"):
        service.validate_inference_request(**{**base, "cameras": [0, 1]})
    with pytest.raises(ValueError, match="must be a positive integer"):
        service.validate_inference_request(**{**base, "cameras": [0], "window": 0})
    with pytest.raises(ValueError, match="reference_camera_id=null"):
        service.validate_inference_request(
            **{**base, "cameras": [0], "reference_camera_id": "cam_0"}
        )
    with pytest.raises(ValueError, match="CUDA"):
        service.validate_inference_request(
            **{**base, "cameras": [0], "device": "cuda:99"}
        )


def test_validate_inference_request_warns_on_window_clamp_and_shrink(
    tmp_path: Path,
) -> None:
    _build_dataset(tmp_path)
    reference = _single_checkpoint(tmp_path)
    service = _service(tmp_path)
    clamped = service.validate_inference_request(
        checkpoint=reference,
        form="single_object",
        scene_id="scene_000000",
        cameras=[0],
        device="cpu",
        window=999,
    )
    assert clamped.window == 8
    assert any("clamped" in warning for warning in clamped.warnings)
    shrunk = service.validate_inference_request(
        checkpoint=reference,
        form="single_object",
        scene_id="scene_000000",
        cameras=[0],
        device="cpu",
        window=2,
    )
    assert shrunk.window == 2
    assert any("smaller than trained clip length" in w for w in shrunk.warnings)


def test_validate_inference_request_requires_reference_camera(
    tmp_path: Path,
) -> None:
    _build_dataset(tmp_path)
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "reference.ckpt",
        _trajectory_config(
            name="blcs_multiview_axial_reference",
            selector="camera_view_v2",
            scene_dir="blcs/single_object_camera_view_v2",
            seq_len_range=(4, 4),
            num_views_range=(3, 4),
        ),
    )
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="explicit reference_camera_id"):
        service.validate_inference_request(
            checkpoint="checkpoints:blcs/reference.ckpt",
            form="single_object_camera_view_v2",
            scene_id="scene_000000",
            cameras=[0, 1, 2],
            device="cpu",
        )
    with pytest.raises(ValueError, match="must be one of the selected cameras"):
        service.validate_inference_request(
            checkpoint="checkpoints:blcs/reference.ckpt",
            form="single_object_camera_view_v2",
            scene_id="scene_000000",
            cameras=[0, 1, 2],
            reference_camera_id="cam_5",
            device="cpu",
        )


# -------------------------------------------------------------------- inference


class _StubTrajectoryPredictor:
    """Return a zero trajectory in physical metres for orchestration tests."""

    def __init__(self) -> None:
        self.calls = 0

    def predict_scene(
        self,
        scene: dict[str, Any],
        cameras: list[int],
        *,
        denormalize: bool,
        reference_camera_id: str | None = None,
    ) -> BLCSTrajectoryPrediction:
        del cameras, denormalize, reference_camera_id
        frames = int(np.asarray(scene["cameras"][0]["ball_uv"]).shape[0])
        self.calls += 1
        return BLCSTrajectoryPrediction(
            position=torch.zeros(1, frames, 3),
            velocity=None,
            court_reference_provenance=(build_physical_court_provenance(),),
            coordinates_in_metres=True,
        )


def test_infer_trajectory_windowed_single_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_dataset(tmp_path)
    reference = _single_checkpoint(tmp_path)
    service = _service(tmp_path, device="cpu")
    stub = _StubTrajectoryPredictor()
    monkeypatch.setattr(service, "_predictor", lambda *a, **k: stub)
    response = service.infer(
        checkpoint=reference,
        form="single_object",
        scene_id="scene_000000",
        cameras=[0],
        window=4,
    )
    assert stub.calls == 2
    assert response["frames"] == 6
    assert response["prediction"]["tracks"] == 1
    assert response["prediction"]["presence"] is None
    assert len(response["prediction"]["positions"]) == 18
    assert set(response["metrics"]) == {
        "position_error_m",
        "endpoint_error_m",
        "accuracy_0p3m",
    }
    assert response["device"] == "cpu"
    assert response["cameras"] == [0]
    assert response["warnings"] == []
    # The GPU queue worker serialises with allow_nan=False.
    json.dumps(response, allow_nan=False)


class _StubTrackingPredictor:
    """Return constant query tracks so the matched-metric path can be checked."""

    def __init__(self, num_queries: int, presence: bool = True) -> None:
        self.num_queries = num_queries
        self._presence = presence
        self.calls = 0

    def predict_batch(
        self,
        batch: dict[str, Any],
        *,
        denormalize: bool,
        court_reference_provenance: Any = None,
        reference_metadata: Any = None,
    ) -> BLCSTrackQueryPrediction:
        del denormalize, reference_metadata
        ball_uv = batch["ball_uv"]
        frames = int(ball_uv.shape[2])
        self.calls += 1
        position = torch.ones(1, frames, self.num_queries, 3)
        logits = torch.full((1, frames, self.num_queries), 4.0)
        if not self._presence:
            logits = torch.full((1, frames, self.num_queries), -4.0)
        probability = logits.sigmoid()
        return BLCSTrackQueryPrediction(
            position=position,
            presence_logits=logits,
            presence_probability=probability,
            presence=probability >= 0.5,
            court_reference_provenance=court_reference_provenance,
            coordinates_in_metres=True,
        )


def test_infer_tracking_multi_object_uses_canonical_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_dataset(tmp_path)
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "track.ckpt",
        _tracking_data_config(
            selector="physical_v1",
            scene_dir="blcs/multi_object",
            reference=False,
        ),
    )
    service = _service(tmp_path, device="cpu")
    stub = _StubTrackingPredictor(num_queries=2, presence=True)
    monkeypatch.setattr(service, "_tracking_predictor", lambda *a, **k: stub)
    response = service.infer(
        checkpoint="checkpoints:blcs/track.ckpt",
        form="multi_object",
        scene_id="scene_000000",
        cameras=[0, 1, 2, 3, 4, 5],
        window=6,
    )
    assert stub.calls == 1
    assert response["frames"] == 6
    assert response["prediction"]["tracks"] == 2
    assert response["prediction"]["presence"] is not None
    assert len(response["prediction"]["positions"]) == 6 * 2 * 3
    assert len(response["prediction"]["presence"]) == 6 * 2
    metrics = response["metrics"]
    assert metrics["matched_pairs"] > 0
    assert metrics["gt_present"] == 11.0
    assert metrics["predicted_present"] == 12.0
    assert 0.0 < metrics["presence_precision"] < 1.0
    assert 0.0 < metrics["presence_recall"] <= 1.0
    assert response["prediction"]["presence"][-2:] == [1, 1]
    json.dumps(response, allow_nan=False)


def test_tracking_metrics_match_ignoring_query_slot_order() -> None:
    target: np.ndarray = np.zeros((2, 2, 3), dtype=np.float64)
    target[0, 0] = (0.0, 0.0, 0.0)
    target[0, 1] = (5.0, 0.0, 0.0)
    target[1, 0] = (0.0, 1.0, 0.0)
    target[1, 1] = (5.0, 1.0, 0.0)
    present: np.ndarray = np.ones((2, 2), dtype=bool)
    prediction = target[:, ::-1, :].copy() + 0.05
    order_swapped = tracking_metrics(
        predicted_position=prediction,
        predicted_present=present.copy(),
        target_position=target,
        target_present=present,
    )
    straight = tracking_metrics(
        predicted_position=target + 0.05,
        predicted_present=present.copy(),
        target_position=target,
        target_present=present,
    )
    assert order_swapped["position_error_m"] == pytest.approx(
        straight["position_error_m"]
    )
    assert order_swapped["position_error_m"] == pytest.approx(np.sqrt(3) * 0.05)
    assert order_swapped["accuracy_0p3m"] == 1.0
    assert order_swapped["matched_pairs"] == 4.0

    absent = tracking_metrics(
        predicted_position=target + 0.05,
        predicted_present=present.copy(),
        target_position=target,
        target_present=np.zeros((2, 2), dtype=bool),
    )
    assert absent["matched_pairs"] == 0.0
    assert absent["position_error_m"] is None
    assert absent["accuracy_0p3m"] is None
    assert absent["presence_precision"] == 0.0


def test_tracking_metrics_counts_false_positives_and_negatives() -> None:
    target: np.ndarray = np.zeros((1, 2, 3), dtype=np.float64)
    present = np.array([[True, False]])
    predicted_present = np.array([[True, True]])
    metrics = tracking_metrics(
        predicted_position=np.zeros((1, 2, 3), dtype=np.float64),
        predicted_present=predicted_present,
        target_position=target,
        target_present=present,
    )
    assert metrics["matched_pairs"] == 1.0
    assert metrics["predicted_present"] == 2.0
    assert metrics["gt_present"] == 1.0
    assert metrics["presence_precision"] == pytest.approx(0.5)
    assert metrics["presence_recall"] == pytest.approx(1.0)


# --------------------------------------------- canonical dataset integration


@pytest.mark.parametrize(
    ("form", "reference"),
    [
        ("multi_object", False),
        ("multi_object_broadcast", False),
        ("multi_object_camera_view_v2", True),
    ],
)
def test_build_tracking_input_from_every_multi_form(
    tmp_path: Path,
    form: str,
    reference: bool,
) -> None:
    _build_dataset(tmp_path)
    config = _tracking_data_config(
        selector="camera_view_v2" if reference else "physical_v1",
        scene_dir=f"blcs/{form}",
        reference=reference,
    )
    num_cameras = 3 if reference else 2
    tracking_input = build_tracking_input(
        scene_dir=tmp_path / "data" / "blcs" / form,
        scene_id="scene_000000",
        config=config,
        reference_camera_id="cam_0" if reference else None,
        camera_indices=tuple(range(num_cameras)),
        window_start=0,
        window_length=4,
        seed=0,
    )
    assert tracking_input.window_length == 4
    assert tracking_input.ground_truth.shape == (4, 2, 3)
    assert tracking_input.ground_truth_present.shape == (4, 2)
    assert tracking_input.batch["ball_uv"].shape == (1, num_cameras, 4, 2, 2)
    assert tracking_input.batch["ball_vis"].shape == (1, num_cameras, 4, 2)
    assert tracking_input.batch["court_kp"].shape == (1, num_cameras, 4, 14, 2)
    assert tracking_input.batch["padding_mask"].shape == (1, num_cameras, 4)
    metadata = blcs_reference_metadata_from_batch(tracking_input.batch)
    if reference:
        assert metadata is not None
        assert metadata.track_query_contract is not None
        assert metadata.track_query_contract == (
            resolve_blcs_track_query_reference_contract(config)
        )
    else:
        assert metadata is None


@pytest.mark.parametrize(
    "form",
    SINGLE_FORMS + REFERENCE_SINGLE_FORMS,
)
def test_every_single_form_loads_as_a_scene(tmp_path: Path, form: str) -> None:
    _build_dataset(tmp_path)
    payload = _service(tmp_path).scene(form=form, scene_id="scene_000000")
    assert payload["gt"]["tracks"] == 1
    assert payload["gt"]["frames"] == 6
    expected_cameras = (
        4 if form.endswith("camera_view_v2") else (2 if "broadcast" in form else 6)
    )
    assert len(payload["cameras"]) == expected_cameras


# ----------------------------------------------------------- local data


@pytest.mark.local_data
def test_real_curated_checkpoints_are_classified_runnable() -> None:
    if not (REAL_CHECKPOINTS_ROOT.is_dir() and (REAL_DATA_ROOT / "blcs").is_dir()):
        pytest.skip("Local BLCS checkpoints or datasets are unavailable.")
    service = InferenceService(
        PROJECT_ROOT / "outputs" / "blcs",
        REAL_CHECKPOINTS_ROOT,
        REAL_DATA_ROOT,
    )
    catalog = service.catalog()
    runnable = [entry for entry in catalog["checkpoints"] if entry["runnable"]]
    assert runnable, "expected at least one runnable curated checkpoint"
    references = [
        entry
        for entry in runnable
        if entry["model_name"] == "blcs_multiview_axial_reference"
    ]
    assert references
    for entry in references:
        assert entry["allowed_forms"] == ["single_object_camera_view_v2"]
        assert entry["seq_len"] == 128
        assert entry["num_court_tokens"] in {14, 20}
    kp14 = next(entry for entry in references if "kp14" in entry["name"])
    assert kp14["num_court_tokens"] == 14


@pytest.mark.local_data
@pytest.mark.parametrize(
    "form",
    SINGLE_FORMS + REFERENCE_SINGLE_FORMS + MULTI_FORMS + REFERENCE_MULTI_FORMS,
)
def test_real_form_scene_payloads(form: str) -> None:
    form_dir = REAL_DATA_ROOT / "blcs" / form
    if not (form_dir / "scenes").is_dir():
        pytest.skip(f"Local BLCS form {form!r} is unavailable.")
    service = InferenceService(
        PROJECT_ROOT / "outputs" / "blcs",
        REAL_CHECKPOINTS_ROOT,
        REAL_DATA_ROOT,
    )
    listing = service.scenes(form=form, limit=1)
    assert listing["total"] >= 1
    scene_id = listing["scenes"][0]["id"]
    payload = service.scene(form=form, scene_id=scene_id)
    gt = payload["gt"]
    assert gt["tracks"] == payload["num_balls"]
    assert payload["frame_count"] == gt["frames"]
    assert len(gt["positions"]) == gt["frames"] * gt["tracks"] * 3
    assert len(gt["active"]) == gt["frames"] * gt["tracks"]
