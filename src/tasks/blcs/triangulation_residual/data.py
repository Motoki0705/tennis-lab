"""Single-ball physical simulation scenes for triangulation-residual training."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.generate_dataset import (
    CourtKeypointArtifactMetadata,
    build_court_view_record,
    validate_dataset_court_keypoint_contract_documents,
)
from src.tasks.base.triangulation_residual.contracts import (
    CameraRig,
    CleanResidualScene,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCS_DATASET_SCHEMA_ID
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    validate_court_coordinate_normalization,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


@lru_cache(maxsize=32)
def _validated_root_contract(
    metadata_path: Path, mtime_ns: int, size: int
) -> CourtKeypointArtifactMetadata:
    """Keep only validated root semantics, keyed by the source file revision."""
    root = _read_object(metadata_path)
    artifact = CourtKeypointArtifactMetadata.from_mapping(
        root.get("court_keypoints"),
        location=f"{metadata_path}.court_keypoints",
    )
    validate_dataset_court_keypoint_contract_documents(
        root_metadata=root,
        scene_metadata={},
        runtime_contract=artifact.contract,
        expected_dataset_schema_id=BLCS_DATASET_SCHEMA_ID,
        dataset_location=str(metadata_path.parent),
    )
    after = metadata_path.stat()
    if (after.st_mtime_ns, after.st_size) != (mtime_ns, size):
        raise ValueError(f"{metadata_path}: metadata changed during validation")
    return artifact


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def list_scene_paths(dataset_root: Path, split: str) -> list[Path]:
    """Preserve the generated dataset's fixed split order without resplitting."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    names = [
        line.strip()
        for line in (dataset_root / f"{split}.txt").read_text().splitlines()
        if line.strip()
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{split} must contain nonempty, unique scene IDs")
    paths: list[Path] = []
    for name in names:
        if name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"Invalid split scene ID: {name!r}")
        scene_path = dataset_root / "scenes" / name
        if not scene_path.is_dir():
            raise FileNotFoundError(f"Split scene is missing: {scene_path}")
        paths.append(scene_path)
    return paths


def load_clean_scene(scene_dir: Path) -> CleanResidualScene:
    """Read complete single-ball GT in metres and unchanged physical cameras.

    Camera-view-v2 only permutes the stored court observations. The ball GT and
    saved camera centres/rotations are already in the physical court frame.
    Inactive or padded GT and multi-ball scenes require another target contract
    and are rejected instead of being silently cropped or filled.
    """
    if scene_dir.parent.name != "scenes":
        raise ValueError("Expected a BLCS <dataset>/scenes/<scene> directory")
    meta = _read_object(scene_dir / "meta.json")
    scalars = _read_object(scene_dir / "scalars.json")
    validate_court_coordinate_normalization(meta, artifact=str(scene_dir))
    artifact = CourtKeypointArtifactMetadata.from_mapping(
        meta.get("court_keypoints"), location=f"{scene_dir}/meta.json.court_keypoints"
    )
    root_path = (scene_dir.parent.parent / "meta.json").resolve()
    root_stat = root_path.stat()
    root_artifact = _validated_root_contract(
        root_path, root_stat.st_mtime_ns, root_stat.st_size
    )
    court_contract = validate_dataset_court_keypoint_contract_documents(
        root_metadata={"court_keypoints": root_artifact.to_dict()},
        scene_metadata={scene_dir.name: meta},
        runtime_contract=artifact.contract,
        expected_dataset_schema_id=BLCS_DATASET_SCHEMA_ID,
        dataset_location=str(scene_dir.parent.parent),
    )
    camera_records = court_contract.scenes[0].court_views
    scene_id = meta.get("scene_id")
    if not isinstance(scene_id, str) or scene_id != scene_dir.name:
        raise ValueError("scene_id must match the scene directory name")
    frames = _positive_integer(meta.get("num_frames"), "num_frames")
    fps = _finite_number(meta.get("fps_out"), "fps_out")
    if fps <= 0:
        raise ValueError("fps_out must be positive")
    if type(scalars.get("num_balls")) is not int or scalars["num_balls"] != 1:
        raise ValueError("BLCS residual training requires exactly one ball")

    normalized = np.load(scene_dir / "ball_pos_norm.npy", allow_pickle=False)
    world = np.load(scene_dir / "ball_pos_world.npy", allow_pickle=False)
    for name, array in (("ball_pos_norm", normalized), ("ball_pos_world", world)):
        if (
            array.shape != (frames, 3)
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} must contain finite floating ({frames}, 3) GT")
    denormalized = denormalize_court_position(normalized)
    if not np.allclose(denormalized, world, rtol=1e-6, atol=1e-5):
        raise ValueError("ball_pos_norm and metre-valued ball_pos_world disagree")
    presence_path = scene_dir / "ball_present.npy"
    if presence_path.exists():
        present = np.load(presence_path, allow_pickle=False)
        if present.shape != (frames,) or present.dtype != np.bool_ or not present.all():
            raise ValueError("Inactive or padded ball GT is unsupported")

    views = _positive_integer(scalars.get("num_cameras"), "num_cameras")
    if (
        views != len(camera_records)
        or _positive_integer(meta.get("num_cameras"), "meta.num_cameras") != views
    ):
        raise ValueError("num_cameras must match the ordered court camera records")
    actual_slots = {
        key for key in scalars if key.startswith("cam_") and key.endswith("_params")
    }
    if actual_slots != {f"cam_{index}_params" for index in range(views)}:
        raise ValueError("Camera parameter slots must match the court camera records")
    intrinsics, rotations, translations, sizes = [], [], [], []
    for index in range(views):
        camera_id = f"cam_{index}"
        record = camera_records[index]
        if record.camera_id != camera_id:
            raise ValueError(f"Camera slot {index} requires stable ID {camera_id!r}")
        params = scalars[f"cam_{index}_params"]
        if isinstance(params, str):
            params = json.loads(params)
        if not isinstance(params, dict):
            raise ValueError(f"cam_{index}_params must be an object")
        parameter_record = build_court_view_record(
            camera_id=camera_id,
            camera_center_court_m=params["C"],
            contract=artifact.contract,
        )
        if parameter_record.camera_center_court_m != record.camera_center_court_m:
            raise ValueError(
                f"{camera_id}: camera center does not match CourtKP metadata"
            )
        rotation = np.asarray(params["R"], dtype=np.float64)
        center = np.asarray(params["C"], dtype=np.float64)
        if rotation.shape != (3, 3) or center.shape != (3,):
            raise ValueError("Camera R/C must have shape (3, 3)/(3,)")
        focal = _finite_number(params.get("f"), "camera.f")
        cx = _finite_number(params.get("cx"), "camera.cx")
        cy = _finite_number(params.get("cy"), "camera.cy")
        width = _positive_integer(params.get("w"), "camera.w")
        height = _positive_integer(params.get("h"), "camera.h")
        intrinsics.append([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        rotations.append(rotation)
        translations.append(-rotation @ center)
        sizes.append([width, height])
    rig = CameraRig(
        K=np.asarray(intrinsics, dtype=np.float64),
        R=np.asarray(rotations, dtype=np.float64),
        t=np.asarray(translations, dtype=np.float64),
        image_size=np.asarray(sizes, dtype=np.int64),
    )
    return CleanResidualScene(
        scene_id=scene_id,
        world_m=np.asarray(denormalized[:, None, :], dtype=np.float32),
        fps=fps,
        rig=rig,
        source_group=scene_id,
    )
