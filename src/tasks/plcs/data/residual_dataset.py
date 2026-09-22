"""Fixed source-disjoint splits with deterministic epoch-dependent corruption."""

from __future__ import annotations

import json
import multiprocessing
import re
from collections import OrderedDict
from collections.abc import Callable
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.tasks.base.generate_dataset import (
    CourtKeypointArtifactMetadata,
    extract_court_view_records,
)
from src.tasks.plcs.configuration_contracts import ResidualConfig
from src.tasks.plcs.court_keypoint_contract import PLCS_GENERATED_DATASET_SCHEMA_ID
from src.tasks.plcs.data.augmentation.residual import (
    corrupt_candidates,
    fixed_six_camera_rig,
)
from src.tasks.plcs.data.residual_types import CameraRig, CleanResidualScene
from src.tasks.plcs.geometry.residual_features import (
    InsufficientGeometryError,
    prepare_geometry,
)
from src.utils.schema.court_normalization import validate_court_coordinate_normalization


def _initialize_residual_worker(_worker_id: int) -> None:
    """Keep native OpenCV work inside each fresh worker on one thread."""
    import cv2

    cv2.setNumThreads(1)


class ResidualDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        scenes: list[Path],
        loader: Callable[[Path], CleanResidualScene],
        config: ResidualConfig,
        split: str,
    ) -> None:
        self.scenes, self.loader, self.config, self.split = (
            scenes,
            loader,
            config,
            split,
        )
        # The shared semaphore must use the same context as DataLoader workers.
        self.epoch = multiprocessing.get_context("spawn").Value("q", 0)
        self.cache: OrderedDict[Path, CleanResidualScene] = OrderedDict()

    def __len__(self) -> int:
        return len(self.scenes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.value = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.scenes[index]
        if path in self.cache:
            scene = self.cache.pop(path)
            self.cache[path] = scene
        else:
            scene = self.loader(path)
            if self.config.data.cache_scenes:
                self.cache[path] = scene
                while len(self.cache) > self.config.data.cache_scenes:
                    self.cache.popitem(last=False)
        if scene.world_m.shape[1:] != (self.config.joints, 3):
            raise ValueError("Scene joints do not match selected residual profile")
        epoch = self.epoch.value if self.split == "train" else 0
        seed = (
            self.config.runtime.run.seed
            + index * 101
            + epoch * 1000003
            + {"train": 0, "val": 100000000, "test": 200000000}[self.split]
        )
        rng = np.random.default_rng(seed)
        length = self.config.data.sequence_length
        stride = max(1, round(scene.fps / self.config.data.target_fps))
        fps = scene.fps / stride
        span = (length - 1) * stride + 1
        max_start = max(0, len(scene.world_m) - span)
        start = (
            int(rng.integers(max_start + 1))
            if self.split == "train"
            else max_start // 2
        )
        indices: np.ndarray = start + np.arange(length) * stride
        frame_valid = indices < len(scene.world_m)
        world = np.asarray(
            scene.world_m[np.minimum(indices, len(scene.world_m) - 1)], dtype=np.float32
        ).copy()
        values = sample_residual_window(
            world,
            frame_valid,
            fps,
            scene.rig,
            rng,
            self.config,
            scene_id=scene.scene_id,
            split=self.split,
        )
        return tensor_sample(values, scene.scene_id)


def tensor_sample(values: dict[str, Any], scene_id: str) -> dict[str, Any]:
    tensors: dict[str, Any] = {
        key: torch.from_numpy(np.ascontiguousarray(value))
        for key, value in values.items()
    }
    tensors["scene_id"] = scene_id
    return tensors


VIEW_FIELDS = frozenset(
    {
        "features",
        "view_valid",
        "true_projection",
        "clean_uv",
        "clean_visible",
        "selected_camera_indices",
        "persistent_kind",
    }
)


def collate_residual(samples: list[dict[str, Any]]) -> dict[str, Any]:
    views = max(sample["features"].shape[0] for sample in samples)
    batch: dict[str, Any] = {}
    for key in samples[0]:
        if key == "scene_id":
            batch[key] = [s[key] for s in samples]
            continue
        values: list[Tensor] = []
        for sample in samples:
            value = sample[key]
            if key in VIEW_FIELDS and value.shape[0] < views:
                padding = torch.full(
                    (views - value.shape[0], *value.shape[1:]),
                    -1 if key == "selected_camera_indices" else 0,
                    dtype=value.dtype,
                )
                value = torch.cat((value, padding))
            values.append(value)
        batch[key] = torch.stack(values)
    return batch


def sample_residual_window(
    world: np.ndarray,
    frame_valid: np.ndarray,
    fps: float,
    source_rig: CameraRig,
    rng: np.random.Generator,
    config: ResidualConfig,
    *,
    scene_id: str,
    split: str,
) -> dict[str, Any]:
    """Targets/window/split stay fixed across bounded observation-only retries."""
    augmentation = config.augmentation
    if not np.all(source_rig.image_size == source_rig.image_size[0]):
        raise ValueError(
            "The synthetic six-camera profile requires one common image size"
        )
    base_rig = fixed_six_camera_rig(
        (int(source_rig.image_size[0, 0]), int(source_rig.image_size[0, 1]))
    )
    n_views = (
        int(rng.integers(config.data.min_views, config.data.max_views + 1))
        if split == "train"
        else augmentation.evaluation_views
    )
    # Keep view ordering independent of the corruption mode for paired probes.
    subset_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    corruption_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    all_subsets = [
        subset_rng.permutation(candidate).astype(np.int64)
        for candidate in combinations(range(6), n_views)
    ]
    subset_rng.shuffle(all_subsets)
    draw_count = 0
    failed_cameras: np.ndarray = np.zeros(6, dtype=np.int64)
    failure_reasons: dict[str, int] = {}
    fixed_error: tuple[int, float] | None = None
    for attempt in range(8):
        candidates = corrupt_candidates(
            world,
            base_rig,
            corruption_rng,
            augmentation,
            fps=fps,
            fixed_error=fixed_error,
        )
        fixed_error = (int(candidates.family), candidates.severity)
        valid_camera_ids = set(int(i) for i in candidates.valid_indices)
        for index, failure in enumerate(candidates.court_fit.failures):
            if failure is not None:
                failed_cameras[index] += 1
                reason = str(failure.reason)
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        for candidate in all_subsets:
            if not set(candidate).issubset(valid_camera_ids):
                continue
            selected = candidate
            noisy = candidates.subset(selected)
            noisy.scores[:, ~frame_valid] = 0
            noisy.observations_px[:, ~frame_valid] = np.nan
            draw_count += 1
            try:
                geometry = prepare_geometry(
                    noisy.observations_px,
                    noisy.scores,
                    noisy.court_px,
                    noisy.court_scores,
                    noisy.estimated_rig,
                    root_indices=config.root_indices,
                    fps=fps,
                    min_score=config.initializer.min_score,
                    refinement_steps=config.initializer.refinement_steps,
                )
            except InsufficientGeometryError:
                continue
            true_p = noisy.true_rig.matrices.copy()
            true_p[:, 0] /= noisy.true_rig.image_size[:, 0, None]
            true_p[:, 1] /= noisy.true_rig.image_size[:, 1, None]
            event_mask = (
                candidates.persistent_mask[selected] & frame_valid[None, :, None]
            )
            return {
                "features": geometry.features,
                "view_valid": geometry.view_valid & frame_valid[None],
                "time_positions": geometry.time_positions,
                "root_init": geometry.root_init_m,
                "relative_init": geometry.relative_init_m,
                "init_world": geometry.init_world_m,
                "init_valid": geometry.init_valid & frame_valid[:, None],
                "target_world": world,
                "frame_valid": frame_valid,
                "true_projection": true_p.astype(np.float32),
                "clean_uv": noisy.clean_uv,
                "clean_visible": noisy.clean_visible & frame_valid[None, :, None],
                "fps": np.array(fps, np.float32),
                "severity": np.array(noisy.severity, np.float32),
                "geometry_attempts": np.array(draw_count, np.int64),
                "corruption_rounds": np.array(attempt + 1, np.int64),
                "selected_camera_indices": selected,
                "num_views": np.array(n_views, np.int64),
                "corruption_family": np.array(candidates.family, np.int64),
                "calibration_attempts": np.array(6 * (attempt + 1), np.int64),
                "calibration_failed_candidates": failed_cameras,
                "persistent_fraction": np.array(event_mask.mean(), np.float32),
                "persistent_mask": event_mask.any(axis=0),
                "persistent_kind": candidates.persistent_kind[selected],
            }
    raise InsufficientGeometryError(
        f"{scene_id}: no feasible {n_views}-camera subset after 8 full-rig draws; "
        f"geometry_attempts={draw_count}, calibration_failures={failure_reasons}"
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
