"""Read ACCAD scenes without loading SMPL or applying legacy root transforms."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from src.tasks.base.generate_dataset import (
    CourtKeypointArtifactMetadata,
    extract_court_view_records,
)
from src.tasks.base.triangulation_residual.contracts import (
    CameraRig,
    CleanResidualScene,
)
from src.tasks.plcs.court_keypoint_contract import PLCS_GENERATED_DATASET_SCHEMA_ID
from src.utils.schema.court_normalization import (
    validate_court_coordinate_normalization,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


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


def _source_group(meta: dict[str, Any], location: Path) -> str:
    """Identify the source file independently of a machine's dataset prefix."""
    source = meta.get("motion_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{location}: motion_source must be a nonempty ACCAD path")
    parts = PurePosixPath(source.replace("\\", "/")).parts
    if parts.count("ACCAD") != 1 or ".." in parts:
        raise ValueError(f"{location}: only ACCAD motion sources are supported")
    relative = parts[parts.index("ACCAD") :]
    if len(relative) < 3 or PurePosixPath(relative[-1]).suffix != ".npz":
        raise ValueError(f"{location}: expected an ACCAD subject/motion .npz path")
    if meta.get("motion_source_kind") not in (None, "accad"):
        raise ValueError(f"{location}: motion_source_kind must be accad")
    return PurePosixPath(*relative).as_posix()


def list_scene_paths(dataset_root: Path, split: str) -> list[Path]:
    """Return the existing fixed split in file order; never create a new split."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    names = [
        line.strip()
        for line in (dataset_root / f"{split}.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{split} must contain nonempty, unique scene IDs")
    paths = []
    for name in names:
        if name in {".", ".."} or Path(name).name != name or "\\" in name:
            raise ValueError(f"Invalid split scene ID: {name!r}")
        path = dataset_root / "scenes" / name
        if not path.is_dir():
            raise FileNotFoundError(f"Split scene is missing: {path}")
        paths.append(path)
    return paths


def load_clean_scene(scene_dir: Path) -> CleanResidualScene:
    """Load finite COCO-17 world joints and fixed physical cameras in metres.

    ``human_kp_3d`` is already physical, Z-up COCO-17. The normalization marker
    describes other legacy arrays, including the SMPL-root ``position.npy``;
    neither that array nor canonical pose, vertices, or disk court UV is read.
    The residual profile derives its root from the two COCO hip joints later.
    """
    meta = _read_object(scene_dir / "meta.json")
    scalars = _read_object(scene_dir / "scalars.json")
    scene_id = meta.get("scene_id")
    if not isinstance(scene_id, str) or scene_id != scene_dir.name:
        raise ValueError("scene_id must match the scene directory name")
    source_group = _source_group(meta, scene_dir)
    frames = _positive_integer(meta.get("num_frames"), "num_frames")
    fps = _finite_number(meta.get("fps"), "fps")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if type(scalars.get("num_persons")) is not int or scalars["num_persons"] != 1:
        raise ValueError("PLCS residual training requires exactly one person")
    validate_court_coordinate_normalization(meta, artifact=str(scene_dir))
    artifact = CourtKeypointArtifactMetadata.from_mapping(
        meta.get("court_keypoints"), location=f"{scene_dir}/meta.json.court_keypoints"
    )
    if artifact.dataset_schema_id != PLCS_GENERATED_DATASET_SCHEMA_ID:
        raise ValueError("Expected the PLCS generated-dataset schema")
    records = extract_court_view_records(
        meta, contract=artifact.contract, location=str(scene_dir / "meta.json")
    )

    views = _positive_integer(scalars.get("num_cameras"), "num_cameras")
    if meta.get("num_cameras") != views or records is None or len(records) != views:
        raise ValueError("Scene, scalar, and court-view camera counts must agree")
    intrinsics, rotations, translations, sizes = [], [], [], []
    for index, record in enumerate(records):
        params = scalars[f"cam_{index}_params"]
        if isinstance(params, str):
            params = json.loads(params)
        if not isinstance(params, dict):
            raise ValueError(f"cam_{index}_params must be an object")
        rotation = np.asarray(params["R"], dtype=np.float64)
        center = np.asarray(params["C"], dtype=np.float64)
        if rotation.shape != (3, 3) or center.shape != (3,):
            raise ValueError("Camera R/C must have shape (3, 3)/(3,)")
        if record.camera_id != f"camera_{index}" or not np.array_equal(
            center, record.camera_center_court_m
        ):
            raise ValueError("Camera order/centres disagree with court-view metadata")
        focal = _finite_number(params.get("f"), "camera.f")
        cx = _finite_number(params.get("cx"), "camera.cx")
        cy = _finite_number(params.get("cy"), "camera.cy")
        width = _positive_integer(params.get("w"), "camera.w")
        height = _positive_integer(params.get("h"), "camera.h")
        if "image_size" in params and params["image_size"] != [width, height]:
            raise ValueError("camera.image_size disagrees with camera.w/h")
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
    world = np.load(scene_dir / "human_kp_3d.npy", mmap_mode="r", allow_pickle=False)
    if (
        world.shape != (frames, 17, 3)
        or world.dtype != np.float32
        or not np.isfinite(world).all()
    ):
        raise ValueError(
            f"human_kp_3d must contain finite float32 ({frames}, 17, 3) GT"
        )
    presence_path = scene_dir / "person_present.npy"
    if presence_path.exists():
        present = np.load(presence_path, mmap_mode="r", allow_pickle=False)
        if present.shape != (frames,) or present.dtype != np.bool_ or not present.all():
            raise ValueError("Inactive or padded person GT is unsupported")
    return CleanResidualScene(
        scene_id=scene_id,
        world_m=np.asarray(world),
        fps=fps,
        rig=rig,
        source_group=source_group,
    )


def audit_source_splits(dataset_root: Path) -> dict[str, Any]:
    """Check actual scene metadata for scene/source leakage, reading no arrays.

    ACCAD directories such as ``Male1Walking_c3d`` and ``Male1Running_c3d``
    belong to the same subject. Subject overlap is reported, not rejected:
    the existing dataset is split by source motion, not by subject.
    """
    paths = {
        split: list_scene_paths(dataset_root, split)
        for split in ("train", "val", "test")
    }
    scene_ids = {
        split: {path.name for path in entries} for split, entries in paths.items()
    }
    source_groups: dict[str, set[str]] = {split: set() for split in paths}
    subjects: dict[str, set[str]] = {split: set() for split in paths}
    for left, right in combinations(paths, 2):
        overlap = scene_ids[left] & scene_ids[right]
        if overlap:
            raise ValueError(
                f"Scene overlap between {left}/{right}: {sorted(overlap)[:5]}"
            )
    for split, entries in paths.items():
        for path in entries:
            meta = _read_object(path / "meta.json")
            if meta.get("scene_id") != path.name:
                raise ValueError(
                    f"{path}: scene_id must match the scene directory name"
                )
            group = _source_group(meta, path)
            source_groups[split].add(group)
            subject_directory = PurePosixPath(group).parts[1]
            match = re.match(r"(?:Male|Female)\d+", subject_directory)
            subjects[split].add(match.group() if match else subject_directory)
    for left, right in combinations(paths, 2):
        overlap = source_groups[left] & source_groups[right]
        if overlap:
            raise ValueError(
                f"Source motion overlap between {left}/{right}: {sorted(overlap)[:5]}"
            )
    subject_overlap = {
        f"{left}/{right}": sorted(subjects[left] & subjects[right])
        for left, right in combinations(paths, 2)
    }
    return {
        "dataset_root": str(dataset_root.resolve()),
        "source_dataset": "ACCAD",
        "split_group": "motion_source",
        "scene_counts": {split: len(entries) for split, entries in paths.items()},
        "source_counts": {
            split: len(groups) for split, groups in source_groups.items()
        },
        "scene_disjoint": True,
        "source_motion_disjoint": True,
        "subjects": {split: sorted(values) for split, values in subjects.items()},
        "subject_overlap": subject_overlap,
        "subject_disjoint": not any(subject_overlap.values()),
    }
