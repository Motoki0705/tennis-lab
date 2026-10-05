"""Stored PLCS observations and their correspondence to physical 3D teachers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.base.visualization.review.camera import CameraRecord, parse_cameras
from src.tasks.base.visualization.review.service import load_json_object
from src.utils.projection.camera_projector import project_points
from src.utils.schema.court_normalization import denormalize_court_position

SPLITS = ("train", "val", "test")


def split_inventory(dataset: Path) -> dict[str, Any]:
    """Report saved assignments and source sharing without changing splits."""
    scenes = {
        path.name: path for path in (dataset / "scenes").iterdir() if path.is_dir()
    }
    assignments: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int | None] = {}
    missing: list[str] = []
    unknown: list[str] = []
    duplicates: list[str] = []
    for split in SPLITS:
        path = dataset / f"{split}.txt"
        if not path.is_file():
            counts[split] = None
            missing.append(path.name)
            continue
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        counts[split] = len(names)
        for name, count in Counter(names).items():
            if count > 1:
                duplicates.append(f"{split}:{name}")
            if name not in scenes:
                unknown.append(f"{split}:{name}")
            assignments[name].append(split)

    source_splits: dict[str, set[str]] = defaultdict(set)
    categories: Counter[str] = Counter()
    for name, path in scenes.items():
        meta = load_json_object(path / "meta.json")
        source = meta.get("motion_source")
        if isinstance(source, str) and source:
            source_splits[source].update(assignments.get(name, []))
        categories[str(meta.get("motion_category", "未記録"))] += 1
    return {
        "counts": counts,
        "missing_files": missing,
        "unknown_scenes": sorted(unknown),
        "duplicate_entries": sorted(duplicates),
        "unassigned_scenes": sorted(set(scenes) - set(assignments)),
        "overlapping_scenes": sorted(
            name for name, splits in assignments.items() if len(splits) > 1
        ),
        "source_count": len(source_splits),
        "shared_source_count": sum(
            len(splits) > 1 for splits in source_splits.values()
        ),
        "categories": dict(sorted(categories.items())),
        "scene_splits": dict(assignments),
        "source_splits": {
            source: sorted(splits, key=SPLITS.index)
            for source, splits in source_splits.items()
        },
    }


@dataclass(frozen=True)
class InspectionPayload:
    document: dict[str, Any]
    binary: bytes


def _float_array(path: Path, shape: tuple[int, ...]) -> NDArray[np.float32]:
    raw = np.load(path, allow_pickle=False)
    if raw.shape != shape or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(
            f"{path.name}: expected floating array with shape {shape}, got {raw.shape}/{raw.dtype}."
        )
    values = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: contains NaN or infinity.")
    return values


def _visibility(path: Path, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    raw = np.load(path, allow_pickle=False)
    if raw.shape != shape or not np.isin(raw, (0, 1)).all():
        raise ValueError(f"{path.name}: expected 0/1 visibility with shape {shape}.")
    return np.asarray(raw, dtype=np.bool_)


def _project(
    camera: CameraRecord, points: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    pixels, front = project_points(
        camera.to_camera(), torch.from_numpy(points.reshape(-1, 3))
    )
    pixels = pixels / torch.tensor(camera.image_size, dtype=torch.float32)
    return (
        pixels.numpy().reshape(*points.shape[:-1], 2),
        front.numpy().reshape(points.shape[:-1]),
    )


def _in_image(uv: NDArray[np.float32]) -> NDArray[np.bool_]:
    return np.asarray(((uv >= 0.0) & (uv <= 1.0)).all(axis=-1), dtype=np.bool_)


def _comparison(
    saved: NDArray[np.float32],
    visible: NDArray[np.bool_],
    projected: NDArray[np.float32],
    front: NDArray[np.bool_],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    expected = front & _in_image(projected)
    matched = visible & front
    pixel_difference = (saved - projected) * np.asarray(image_size, dtype=np.float32)
    error = np.linalg.norm(pixel_difference, axis=-1)[matched]
    return {
        "visible_fraction": float(visible.mean()),
        "empty_frames": int((~visible.any(axis=-1)).sum()),
        "visibility_mismatch_count": int((visible != expected).sum()),
        "visible_outside_count": int((visible & ~_in_image(saved)).sum()),
        "compared_points": int(matched.sum()),
        "max_error_px": float(error.max()) if error.size else None,
        "rms_error_px": float(np.sqrt(np.square(error).mean())) if error.size else None,
    }


def build_inspection(
    scene_path: Path, scene: dict[str, Any], inventory: dict[str, Any]
) -> InspectionPayload:
    """Build an explicit float32 observation buffer with JSON field descriptors."""
    if scene["mode"] != "single":
        raise ValueError("PLCS observation review supports single_object scenes only.")
    frames = int(scene["frame_count"])
    meta = load_json_object(scene_path / "meta.json")
    scalars = load_json_object(scene_path / "scalars.json")
    world = _float_array(scene_path / "human_kp_3d.npy", (frames, 17, 3))
    root = _float_array(scene_path / "position.npy", (frames, 3))
    heading = _float_array(scene_path / "rotation.npy", (frames, 2))
    if not np.allclose(np.linalg.norm(heading, axis=-1), 1.0, atol=1e-4, rtol=0):
        raise ValueError("rotation.npy: cos/sin heading must have unit length.")
    fields: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0

    def add(name: str, array: NDArray[Any]) -> None:
        nonlocal offset
        data = np.asarray(array, dtype="<f4", order="C").tobytes()
        fields[name] = {
            "byte_offset": offset,
            "shape": list(array.shape),
            "dtype": "float32",
        }
        chunks.append(data)
        offset += len(data)

    add("root_m", np.asarray(denormalize_court_position(root), dtype=np.float32))
    add("heading", heading)
    cameras: list[dict[str, Any]] = []
    court = np.asarray(scene["court"]["keypoints"], dtype=np.float32)
    for camera in parse_cameras(scalars):
        prefix = f"cam_{camera.index}_"
        human_uv = _float_array(
            scene_path / f"{prefix}human_kp_uv.npy", (frames, 17, 2)
        )
        human_vis = _visibility(scene_path / f"{prefix}human_kp_vis.npy", (frames, 17))
        court_uv = _float_array(
            scene_path / f"{prefix}court_kp_uv.npy", (frames, 20, 2)
        )
        court_vis = _visibility(scene_path / f"{prefix}court_kp_vis.npy", (frames, 20))
        human_projected, human_front = _project(camera, world)
        view = meta["court_keypoint_views"][camera.index]
        physical_order = np.asarray(view["semantic_to_physical"], dtype=np.intp)
        court_projected, court_front = _project(camera, court[physical_order])
        for name, array in (
            ("human_uv", human_uv),
            ("human_vis", human_vis),
            ("human_projected", human_projected),
            ("human_front", human_front),
            ("court_uv", court_uv),
            ("court_vis", court_vis),
            ("court_projected", court_projected),
            ("court_front", court_front),
        ):
            add(prefix + name, array)
        cameras.append(
            {
                "index": camera.index,
                "id": view["camera_id"],
                "view_id": camera.id,
                "image_size": list(camera.image_size),
                "human": _comparison(
                    human_uv, human_vis, human_projected, human_front, camera.image_size
                ),
                "court": _comparison(
                    court_uv, court_vis, court_projected, court_front, camera.image_size
                ),
            }
        )

    canonical_path = scene_path / "canonical_pose_3d.npy"
    canonical_shape = None
    if canonical_path.is_file():
        raw = np.load(canonical_path, mmap_mode="r", allow_pickle=False)
        if (
            raw.ndim != 3
            or raw.shape[0] != frames
            or raw.shape[1] <= 0
            or raw.shape[-1] != 3
            or not np.issubdtype(raw.dtype, np.floating)
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                "canonical_pose_3d.npy: expected finite [T, J, 3] local joints."
            )
        canonical_shape = list(raw.shape)
    source = meta.get("motion_source")
    if source is not None and (not isinstance(source, str) or not source):
        raise ValueError(
            "meta.json: motion_source must be a nonempty string when recorded."
        )
    document = {
        "scene_id": scene["scene_id"],
        "form": scene["form"],
        "revision": scene["revision"],
        "frame_count": frames,
        "source": {
            "motion": source,
            "category": meta.get("motion_category"),
            "gender": meta.get("gender"),
        },
        "splits": inventory["scene_splits"].get(scene["scene_id"], []),
        "source_splits": inventory["source_splits"].get(source, []),
        "split_summary": {
            key: value
            for key, value in inventory.items()
            if key not in {"scene_splits", "source_splits"}
        },
        "canonical_shape": canonical_shape,
        "skeleton": scene["entity"]["skeleton"],
        "joint_names": scene["entity"]["joint_names"],
        "court_edges": scene["court"]["edges"],
        "cameras": cameras,
        "buffer_fields": fields,
        "byte_length": offset,
        "rgb_available": False,
        "observation_stage": "stored_before_training_augmentation",
    }
    return InspectionPayload(document, b"".join(chunks))
