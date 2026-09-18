"""Scene catalog and GPU inference service for the PLCS inference UI.

Everything here is read-only with respect to datasets and checkpoints.  The
service loads one scene window, runs the standard PLCS predictor once, and
returns ground-truth and predicted tracks in world court metres so the browser
never has to know about normalized representations or reference frames.
"""

from __future__ import annotations

import json
import math
import struct
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    CourtKeypointContract,
    resolve_court_keypoint_contract,
)
from src.tasks.base.visualization.review.camera import parse_cameras
from src.tasks.plcs.court_keypoint_contract import (
    headings_target_to_physical,
    normalized_points_target_to_physical,
)
from src.tasks.plcs.generate_dataset.io.scene_loader import AttrDict, load_scene
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.inference.tracking_predictor import PLCSTrackingPredictor
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.device import DeviceSelectionError, resolve_device
from src.utils.geometry.court_pose import (
    canonical_pose_to_world_pose,
    world_pose_to_canonical_pose,
)
from src.utils.schema.court import (
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)
from src.utils.schema.court_normalization import normalize_court_position
from src.utils.schema.player import COCO17_SKELETON, COCO_KP_NAMES

from .checkpoints import (
    OBJECTNESS_MULTI,
    CheckpointInfo,
    describe_checkpoint,
    family_name,
    load_checkpoint_config,
    scan_checkpoints,
)
from .loader import load_standard_predictor, load_tracking_predictor
from .tracking import (
    TrackingSceneError,
    TrackingWindow,
    TrackMatch,
    build_tracking_batch,
    match_tracks,
    reference_metadata_from_batch,
    tracking_metric_config,
)

SPLITS: Final = ("train", "val", "test")
DEFAULT_SPLIT: Final = "val"
MAX_SCENE_LIMIT: Final = 1000
DEFAULT_CAMERA_DEPTH_M: Final = 4.0
MODE_SINGLE: Final = "single"
MODE_TRACKING: Final = "tracking"
MODE_PREVIEW: Final = "preview"
_SCALE_M: Final = np.asarray(
    (COURT_COORD_SCALE_X, COURT_COORD_SCALE_Y, COURT_COORD_SCALE_Z),
    dtype=np.float64,
)


class SceneCatalogError(ValueError):
    """Raised when a requested family or scene is not a valid PLCS dataset."""


def _as_path_component(value: str, *, name: str) -> str:
    stripped = value.strip()
    if not stripped or Path(stripped).name != stripped or stripped in {".", ".."}:
        raise SceneCatalogError(f"{name} must be a single path component: {value!r}.")
    return stripped


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@lru_cache(maxsize=64)
def split_ids(data_root: Path, family: str, split: str) -> tuple[str, ...]:
    """Return the scene ids of one split file, cached per root/family/split."""
    if split not in SPLITS:
        raise SceneCatalogError(f"split must be one of {list(SPLITS)}, got {split!r}.")
    path = Path(data_root) / family / f"{split}.txt"
    if not path.is_file():
        return ()
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


@lru_cache(maxsize=64)
def scene_meta_index(data_root: Path, family: str) -> dict[str, dict[str, Any]]:
    """Return the ``scenes_meta.json`` index for one family, cached and read-only."""
    path = Path(data_root) / family / "scenes_meta.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, list):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for entry in document:
        if isinstance(entry, Mapping) and isinstance(entry.get("scene_id"), str):
            index[entry["scene_id"]] = dict(entry)
    return index


@dataclass(frozen=True, slots=True)
class FamilyInfo:
    """One scene dataset family discovered under the data root."""

    id: str
    selector: str
    objects: str
    camera_layout: str
    scene_count: int
    splits: dict[str, int]
    num_cameras: int | None
    num_persons: int | None
    available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "selector": self.selector,
            "objects": self.objects,
            "camera_layout": self.camera_layout,
            "scene_count": self.scene_count,
            "splits": dict(self.splits),
            "num_cameras": self.num_cameras,
            "num_persons": self.num_persons,
            "available": self.available,
        }


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """Validated HTTP request payload for one inference run."""

    checkpoint: str
    family: str
    scene: str
    cameras: tuple[int, ...]
    reference_camera_id: str | None
    window_start: int
    window_length: int
    canonical_pose_source: str
    device: str


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Framed binary response: JSON header plus float32 element payload."""

    header: dict[str, Any]
    payload: NDArray[np.float32]

    def to_bytes(self) -> bytes:
        encoded = json.dumps(self.header, ensure_ascii=False).encode("utf-8")
        body: bytes = self.payload.astype("<f4", copy=False).tobytes()
        return struct.pack("<I", len(encoded)) + encoded + body


class PayloadBuilder:
    """Accumulate named ``float32`` arrays into one flat little-endian buffer."""

    def __init__(self) -> None:
        self._chunks: list[NDArray[np.float32]] = []
        self._total = 0

    def append(self, array: NDArray[np.float32]) -> dict[str, Any]:
        contiguous = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        descriptor = {
            "offset": self._total,
            "count": int(contiguous.size),
            "shape": list(array.shape),
        }
        self._chunks.append(contiguous)
        self._total += int(contiguous.size)
        return descriptor

    @property
    def total(self) -> int:
        return self._total

    def build(self) -> NDArray[np.float32]:
        if not self._chunks:
            return np.zeros(0, dtype="<f4")
        return np.concatenate(self._chunks).astype("<f4", copy=False)


def checkpoint_mode(checkpoint: CheckpointInfo) -> str:
    """Return the inference mode a checkpoint's saved config selects."""
    if checkpoint.objects == OBJECTNESS_MULTI:
        return MODE_TRACKING
    return MODE_SINGLE


class InferenceService:
    """Read-only scene catalog plus single-shot PLCS scene inference."""

    def __init__(
        self,
        *,
        data_root: Path,
        checkpoint_root: Path,
        device: str = "cuda",
        project_root: Path | None = None,
        checkpoint_roots: Sequence[str | Path] | None = None,
        camera_depth: float = DEFAULT_CAMERA_DEPTH_M,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"data root is not a directory: {self.data_root}")
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        if not self.checkpoint_root.is_dir():
            raise NotADirectoryError(
                f"checkpoint root is not a directory: {self.checkpoint_root}"
            )
        self.extra_checkpoint_roots: tuple[Path, ...] = tuple(
            Path(item).expanduser().resolve()
            for item in (checkpoint_roots or ())
            if Path(item).expanduser().resolve() != self.checkpoint_root
        )
        if not math.isfinite(camera_depth) or camera_depth <= 0.0:
            raise SceneCatalogError(
                f"camera_depth must be positive and finite, got {camera_depth!r}."
            )
        self.camera_depth = float(camera_depth)
        self.device = device
        self._resolver = self._build_resolver(project_root)
        self._predictor_cache: OrderedDict[
            str, PLCSPredictor | PLCSTrackingPredictor
        ] = OrderedDict()

    # ------------------------------------------------------------------ setup

    def _build_resolver(self, project_root: Path | None) -> PathResolver:
        resolved_project = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else self._infer_project_root()
        )
        roots = RuntimePathRoots(
            project_root=resolved_project,
            data_root=self.data_root,
            checkpoint_root=self.checkpoint_root,
            artifact_root=self.checkpoint_root,
            output_root=self.checkpoint_root,
            cache_root=self.data_root,
            external_asset_root=resolved_project,
        )
        return PathResolver(roots)

    def _infer_project_root(self) -> Path:
        """Pick the shallowest root that contains both data and checkpoints."""
        candidates = [self.data_root, self.data_root.parent, self.checkpoint_root]
        candidates.extend(self.checkpoint_root.parents[:3])
        common = Path(self._common_ancestor(self.data_root, self.checkpoint_root))
        return common if common != Path(common.anchor) else candidates[0]

    @staticmethod
    def _common_ancestor(left: Path, right: Path) -> Path:
        left_parts = left.resolve().parts
        right_parts = right.resolve().parts
        shared: list[str] = []
        for a, b in zip(left_parts, right_parts, strict=False):
            if a != b:
                break
            shared.append(a)
        if len(shared) <= 1:
            return Path(shared[0]) if shared else Path("/")
        return Path(*shared)

    # ---------------------------------------------------------------- catalog

    def catalog(self) -> dict[str, Any]:
        """Return scene families and suggested checkpoints."""
        return {
            "data_root": str(self.data_root),
            "checkpoint_root": str(self.checkpoint_root),
            "checkpoint_roots": [
                str(root)
                for root in (self.checkpoint_root, *self.extra_checkpoint_roots)
            ],
            "device": self.device,
            "splits": list(SPLITS),
            "families": [item.to_dict() for item in self.families()],
            "checkpoints": [item.to_dict() for item in self._scan_checkpoints()],
        }

    def _scan_checkpoints(self) -> list[CheckpointInfo]:
        return scan_checkpoints(
            self.checkpoint_root, extra_roots=self.extra_checkpoint_roots
        )

    def _describe_checkpoint(self, value: str | Path) -> CheckpointInfo:
        return describe_checkpoint(
            self.checkpoint_root, value, extra_roots=self.extra_checkpoint_roots
        )

    def _checkpoint_config(self, value: str | Path) -> Mapping[str, Any]:
        return load_checkpoint_config(
            self.checkpoint_root, value, extra_roots=self.extra_checkpoint_roots
        )

    def families(self) -> list[FamilyInfo]:
        """Discover scene dataset families under the data root."""
        families: list[FamilyInfo] = []
        for directory in sorted(self.data_root.iterdir()):
            if not directory.is_dir() or directory.name.startswith("_"):
                continue
            if not (directory / "scenes").is_dir():
                continue
            split_counts = {
                split: split_ids(self.data_root, directory.name, split)
                for split in SPLITS
            }
            available = any(len(ids) > 0 for ids in split_counts.values())
            if not available:
                continue
            summary = self._family_config(directory)
            num_cameras, num_persons = self._family_scene_shape(directory)
            families.append(
                FamilyInfo(
                    id=directory.name,
                    selector=summary["selector"],
                    objects=summary["objects"],
                    camera_layout=summary["camera_layout"],
                    scene_count=sum(len(ids) for ids in split_counts.values()),
                    splits={k: len(v) for k, v in split_counts.items()},
                    num_cameras=num_cameras,
                    num_persons=num_persons,
                    available=True,
                )
            )
        return families

    def family(self, family: str) -> FamilyInfo:
        name = family_name(family)
        for item in self.families():
            if item.id == name:
                return item
        raise SceneCatalogError(f"unknown scene family: {name!r}")

    def _family_config(self, directory: Path) -> dict[str, str]:
        """Read the dataset-level selector, object mode, and camera layout."""
        meta_path = directory / "meta.json"
        selector: str | None = None
        objects: str | None = None
        layout: str | None = None
        if meta_path.is_file():
            with meta_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            config = document.get("config")
            if isinstance(config, Mapping):
                court = config.get("court_keypoints")
                if isinstance(court, Mapping) and isinstance(
                    court.get("selector"), str
                ):
                    selector = court["selector"]
                generation = config.get("generation")
                if isinstance(generation, Mapping) and isinstance(
                    generation.get("mode"), str
                ):
                    objects = generation["mode"]
                camera = config.get("camera")
                if isinstance(camera, Mapping) and isinstance(
                    camera.get("layout"), str
                ):
                    layout = camera["layout"]
        if selector is None:
            raise SceneCatalogError(
                f"scene family {directory.name!r} has no court_keypoints.selector "
                "in meta.json; refusing to guess semantics."
            )
        if objects is None:
            raise SceneCatalogError(
                f"scene family {directory.name!r} has no generation.mode in "
                "meta.json; refusing to guess object count."
            )
        return {
            "selector": selector,
            "objects": objects,
            "camera_layout": layout or "unknown",
        }

    def _family_scene_shape(self, directory: Path) -> tuple[int | None, int | None]:
        """Return camera/person counts from the first available scene."""
        for split in SPLITS:
            ids = split_ids(self.data_root, directory.name, split)
            if not ids:
                continue
            scalars_path = directory / "scenes" / ids[0] / "scalars.json"
            if not scalars_path.is_file():
                continue
            with scalars_path.open(encoding="utf-8") as handle:
                scalars = json.load(handle)
            return int(scalars["num_cameras"]), int(scalars.get("num_persons", 1))
        return None, None

    def scene_meta_index(self, family: str) -> dict[str, dict[str, Any]]:
        """Return the cached ``scenes_meta.json`` index for one family."""
        return scene_meta_index(self.data_root, family_name(family))

    def scenes(
        self,
        family: str,
        *,
        split: str = DEFAULT_SPLIT,
        query: str = "",
        limit: int = 200,
    ) -> dict[str, Any]:
        """List scene ids of one split, optionally filtered by id substring."""
        info = self.family(family)
        if limit < 1 or limit > MAX_SCENE_LIMIT:
            raise SceneCatalogError(f"limit must be within 1..{MAX_SCENE_LIMIT}.")
        ids = split_ids(self.data_root, info.id, split)
        needle = query.strip().lower()
        matched = (
            [item for item in ids if needle in item.lower()] if needle else list(ids)
        )
        index = scene_meta_index(self.data_root, info.id)
        scenes = [
            self._scene_summary(info.id, scene_id, index)
            for scene_id in matched[:limit]
        ]
        return {
            "family": info.id,
            "split": split,
            "total": len(matched),
            "returned": len(scenes),
            "selector": info.selector,
            "objects": info.objects,
            "scenes": scenes,
        }

    def _scene_summary(
        self, family: str, scene: str, index: Mapping[str, dict[str, Any]]
    ) -> dict[str, Any]:
        entry = index.get(scene, {})
        return {
            "id": scene,
            "family": family,
            "num_frames": entry.get("num_frames"),
            "fps": entry.get("fps"),
            "num_cameras_sampled": entry.get("num_cameras_sampled"),
            "motion_category": entry.get("motion_category"),
            "motion_source": entry.get("motion_source"),
            "gender": entry.get("gender"),
        }

    def scene_detail(
        self, family: str, scene: str, *, checkpoint: str | None = None
    ) -> dict[str, Any]:
        """Describe one scene and the window/camera rules for a checkpoint."""
        info = self.family(family)
        scene_id = _as_path_component(scene, name="scene")
        scene_dir = self.data_root / info.id / "scenes" / scene_id
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"scene not found: {info.id}/{scene_id}")
        with (scene_dir / "meta.json").open(encoding="utf-8") as handle:
            meta = json.load(handle)
        with (scene_dir / "scalars.json").open(encoding="utf-8") as handle:
            scalars = json.load(handle)

        detail: dict[str, Any] = {
            "id": scene_id,
            "family": info.id,
            "num_frames": int(meta["num_frames"]),
            "fps": float(meta["fps"]),
            "num_cameras": int(scalars["num_cameras"]),
            "num_persons": int(scalars.get("num_persons", 1)),
            "selector": info.selector,
            "objects": info.objects,
            "camera_layout": info.camera_layout,
            "cameras": self._camera_ids(meta, int(scalars["num_cameras"])),
            "window": None,
            "mode": MODE_PREVIEW,
            "checkpoint": None,
            "reference_required": info.selector == CAMERA_VIEW_V2_SELECTOR,
            "supported": True,
            "unsupported_reason": None,
        }
        if checkpoint is None:
            return detail

        checkpoint_info = self._describe_checkpoint(checkpoint)
        detail["checkpoint"] = checkpoint_info.to_dict()
        detail["mode"] = (
            checkpoint_mode(checkpoint_info)
            if checkpoint_info.supported
            else MODE_PREVIEW
        )
        if not checkpoint_info.supported:
            detail["supported"] = False
            detail["unsupported_reason"] = checkpoint_info.unsupported_reason
        elif info.id not in checkpoint_info.families:
            detail["supported"] = False
            detail["unsupported_reason"] = (
                f"このチェックポイントは {info.id!r} を推論できません "
                f"(trained: {checkpoint_info.trained_scene_dir!r}, "
                f"allowed: {list(checkpoint_info.families)!r})."
            )
        elif checkpoint_info.selector != info.selector:
            detail["supported"] = False
            detail["unsupported_reason"] = (
                f"Court selector が一致しません: checkpoint="
                f"{checkpoint_info.selector!r}, scene={info.selector!r}."
            )
        else:
            detail["reference_required"] = (
                checkpoint_info.selector == CAMERA_VIEW_V2_SELECTOR
            )
            detail["window"] = self._window_hint(
                detail["num_frames"],
                checkpoint_info.max_seq_len,
                max_views=checkpoint_info.max_views,
                view_range=(3, 4)
                if checkpoint_info.reference
                and checkpoint_mode(checkpoint_info) != MODE_TRACKING
                else None,
            )
        return detail

    # -------------------------------------------------------- scene preview

    def scene_preview(
        self,
        family: str,
        scene: str,
        *,
        cameras: Sequence[int] | None = None,
        window_start: int = 0,
        window_length: int | None = None,
        reference_camera_id: str | None = None,
    ) -> PredictionResult:
        """Return GPU-free ground truth and camera geometry for one scene.

        The browser can draw a scene the moment it is selected, before any
        checkpoint is chosen or any GPU work is queued.  Ground truth is emitted
        as per-physical-object tracks in world metres, exactly like the
        inference payload, so the frontend only implements one track decoder.
        """
        resolved = self._resolve_preview(
            family,
            scene,
            cameras=cameras,
            window_start=window_start,
            window_length=window_length,
            reference_camera_id=reference_camera_id,
        )
        scene_dir = cast("Path", resolved["scene_dir"])
        contract = resolve_court_keypoint_contract(
            cast("FamilyInfo", resolved["family"]).selector
        )
        loaded = load_scene(scene_dir, court_keypoint_contract=contract)
        start = cast("int", resolved["window_start"])
        length = cast("int", resolved["window_length"])
        windowed = window_scene(loaded, start, length)

        position = np.asarray(windowed["position"], dtype=np.float64) * _SCALE_M
        rotation = np.asarray(windowed["rotation"], dtype=np.float64)
        if position.ndim == 2:
            position = position[:, None]
            rotation = rotation[:, None]
        joints = ground_truth_world_joints(windowed)
        if joints.ndim == 3:
            joints = joints[:, None]
        if "person_present" in windowed:
            presence = np.asarray(windowed["person_present"], dtype=bool)
        else:
            presence = np.ones(position.shape[:2], dtype=bool)
        if presence.shape != position.shape[:2]:
            raise SceneCatalogError(
                "person_present must match the physical (T,P) axes: "
                f"{presence.shape} vs {position.shape[:2]}."
            )
        if joints.shape[:2] != position.shape[:2]:
            raise SceneCatalogError(
                "human_kp_3d must match the physical (T,P) axes: "
                f"{joints.shape[:2]} vs {position.shape[:2]}."
            )

        builder = PayloadBuilder()
        tracks: list[dict[str, Any]] = []
        for index in range(position.shape[1]):
            tracks.append(
                {
                    "kind": "gt",
                    "label": f"GT {index}",
                    "object_index": index,
                    "has_joints": True,
                    "position": builder.append(position[:, index].astype(np.float32)),
                    "rotation": builder.append(rotation[:, index].astype(np.float32)),
                    "presence": builder.append(presence[:, index].astype(np.float32)),
                    "joints": builder.append(joints[:, index].astype(np.float32)),
                }
            )
        header: dict[str, Any] = {
            "scene": self._scene_header(resolved),
            "checkpoint": None,
            "request": {
                "cameras": list(cast("tuple[int, ...]", resolved["cameras"])),
                "reference_camera_id": resolved["reference_camera_id"],
                "window_start": start,
                "window_length": length,
            },
            "mode": MODE_PREVIEW,
            "court": self._court_document(),
            "cameras": self._camera_geometry(scene_dir),
            "skeleton": {
                "names": list(COCO_KP_NAMES),
                "edges": [list(edge) for edge in COCO17_SKELETON],
            },
            "tracks": tracks,
            "payload_dtype": "float32",
            "payload_elements": builder.total,
            "metrics": None,
            "warnings": cast("list[str]", resolved["warnings"]),
        }
        return PredictionResult(header=header, payload=builder.build())

    def _resolve_preview(
        self,
        family: str,
        scene: str,
        *,
        cameras: Sequence[int] | None,
        window_start: int,
        window_length: int | None,
        reference_camera_id: str | None,
    ) -> dict[str, Any]:
        info = self.family(family)
        scene_id = _as_path_component(scene, name="scene")
        known = {
            item
            for split in SPLITS
            for item in split_ids(self.data_root, info.id, split)
        }
        if scene_id not in known:
            raise SceneCatalogError(
                f"scene {scene_id!r} is not listed in any split of family {info.id!r}."
            )
        scene_dir = self.data_root / info.id / "scenes" / scene_id
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"scene directory is missing: {scene_dir}")
        with (scene_dir / "meta.json").open(encoding="utf-8") as handle:
            scene_meta = json.load(handle)
        with (scene_dir / "scalars.json").open(encoding="utf-8") as handle:
            scalars = json.load(handle)
        num_frames = int(scene_meta["num_frames"])
        num_cameras = int(scalars["num_cameras"])
        camera_ids = [
            entry["id"] for entry in self._camera_ids(scene_meta, num_cameras)
        ]
        selected = (
            tuple(range(num_cameras))
            if cameras is None
            else tuple(int(i) for i in cameras)
        )
        if not selected:
            raise SceneCatalogError("at least one camera index is required.")
        if len(set(selected)) != len(selected):
            raise SceneCatalogError("camera indices must be distinct.")
        for index in selected:
            if index < 0 or index >= num_cameras:
                raise SceneCatalogError(
                    f"camera index {index} is out of range 0..{num_cameras - 1}."
                )
        if reference_camera_id is not None:
            if reference_camera_id not in camera_ids:
                raise SceneCatalogError(
                    f"reference_camera_id {reference_camera_id!r} is not a scene "
                    f"camera id; available: {camera_ids!r}."
                )
            selected_ids = [camera_ids[index] for index in selected]
            if reference_camera_id not in selected_ids:
                raise SceneCatalogError(
                    f"reference_camera_id {reference_camera_id!r} must be one of "
                    f"the selected cameras {selected_ids!r}."
                )
        length = num_frames if window_length is None else int(window_length)
        start, length = self._validate_window(
            int(window_start), length, num_frames=num_frames, max_seq_len=None
        )
        warnings: list[str] = []
        if start != 0 or length != num_frames:
            warnings.append(
                f"シーン全 {num_frames} frame のうち "
                f"[{start}, {start + length}) のみを表示します。"
            )
        return {
            "family": info,
            "scene_id": scene_id,
            "scene_dir": scene_dir,
            "num_frames": num_frames,
            "num_cameras": num_cameras,
            "camera_ids": camera_ids,
            "fps": float(scene_meta["fps"]),
            "num_persons": int(scalars.get("num_persons", 1)),
            "cameras": selected,
            "reference_camera_id": reference_camera_id,
            "window_start": start,
            "window_length": length,
            "warnings": warnings,
        }

    def _camera_geometry(self, scene_dir: Path) -> list[dict[str, Any]]:
        with (scene_dir / "scalars.json").open(encoding="utf-8") as handle:
            scalars = json.load(handle)
        with (scene_dir / "meta.json").open(encoding="utf-8") as handle:
            meta = json.load(handle)
        ids = self._camera_ids(meta, int(scalars["num_cameras"]))
        return [
            {**camera.to_dict(depth=self.camera_depth), "id": ids[camera.index]["id"]}
            for camera in parse_cameras(scalars)
        ]

    @staticmethod
    def _camera_ids(meta: Mapping[str, Any], num_cameras: int) -> list[dict[str, Any]]:
        views = meta.get("court_keypoint_views")
        ids: list[str] = []
        if isinstance(views, Sequence) and not isinstance(views, (str, bytes)):
            for view in views:
                if isinstance(view, Mapping) and isinstance(view.get("camera_id"), str):
                    ids.append(view["camera_id"])
        while len(ids) < num_cameras:
            ids.append(f"camera_{len(ids)}")
        return [{"index": index, "id": ids[index]} for index in range(num_cameras)]

    @staticmethod
    def _window_hint(
        num_frames: int,
        max_seq_len: int | None,
        *,
        max_views: int | None = None,
        view_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        max_length = num_frames if max_seq_len is None else min(max_seq_len, num_frames)
        if view_range is not None:
            min_views: int | None = int(view_range[0])
            max_views = (
                int(view_range[1])
                if max_views is None
                else min(max_views, int(view_range[1]))
            )
        else:
            min_views = 1 if max_views is not None else None
        return {
            "max_length": int(max_length),
            "min_length": 1,
            "default_length": int(max_length),
            "scene_frames": int(num_frames),
            "model_max_seq_len": max_seq_len,
            "min_views": min_views,
            "max_views": max_views,
        }

    # --------------------------------------------------------------- inference

    def validate_prediction_request(self, request: PredictionRequest) -> dict[str, Any]:
        """Run the same preflight as :meth:`predict` without touching the GPU.

        The queue-backed HTTP path calls this before enqueueing so malformed
        checkpoint/family/scene/camera/window/device requests fail with a 422
        instead of blocking behind GPU work.
        """
        return self._resolve_request(request)

    def predict(self, request: PredictionRequest) -> PredictionResult:
        """Run one scene window through the checkpoint and build the payload."""
        window = self._resolve_request(request)
        return self._run_prediction(request, window)

    def _resolve_request(self, request: PredictionRequest) -> dict[str, Any]:
        """Validate the request against checkpoint and scene contracts."""
        info = self.family(request.family)
        checkpoint_info = self._describe_checkpoint(request.checkpoint)
        if not checkpoint_info.supported:
            raise SceneCatalogError(
                checkpoint_info.unsupported_reason or "checkpoint is unsupported."
            )
        if info.id not in checkpoint_info.families:
            raise SceneCatalogError(
                f"checkpoint {checkpoint_info.id!r} cannot consume family {info.id!r}; "
                f"allowed families: {list(checkpoint_info.families)!r}."
            )
        if checkpoint_info.selector != info.selector:
            raise SceneCatalogError(
                "court selector mismatch between checkpoint and scene family: "
                f"{checkpoint_info.selector!r} vs {info.selector!r}."
            )
        if request.canonical_pose_source not in {"gt", "prediction"}:
            raise SceneCatalogError(
                "canonical_pose_source must be 'gt' or 'prediction', got "
                f"{request.canonical_pose_source!r}."
            )
        # Fail an explicitly requested, unavailable device here so the queue
        # path reports it before enqueueing instead of as a GPU-time error.
        try:
            device = resolve_device(request.device)
        except DeviceSelectionError as error:
            raise SceneCatalogError(str(error)) from error
        if device.type not in {"cpu", "cuda"}:
            raise SceneCatalogError("The inference UI supports CPU and CUDA devices.")
        mode = checkpoint_mode(checkpoint_info)

        scene_id = _as_path_component(request.scene, name="scene")
        known = {
            item
            for split in SPLITS
            for item in split_ids(self.data_root, info.id, split)
        }
        if scene_id not in known:
            raise SceneCatalogError(
                f"scene {scene_id!r} is not listed in any split of family {info.id!r}."
            )
        scene_dir = self.data_root / info.id / "scenes" / scene_id
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"scene directory is missing: {scene_dir}")
        with (scene_dir / "meta.json").open(encoding="utf-8") as handle:
            scene_meta = json.load(handle)
        with (scene_dir / "scalars.json").open(encoding="utf-8") as handle:
            scalars = json.load(handle)
        num_frames = int(scene_meta["num_frames"])
        num_cameras = int(scalars["num_cameras"])
        camera_ids = [
            entry["id"] for entry in self._camera_ids(scene_meta, num_cameras)
        ]

        cameras = self._validate_cameras(
            request.cameras, checkpoint_info, num_cameras, mode=mode
        )
        reference_camera_id = self._validate_reference(
            request.reference_camera_id,
            checkpoint_info,
            cameras,
            camera_ids,
        )
        start, length = self._validate_window(
            request.window_start,
            request.window_length,
            num_frames=num_frames,
            max_seq_len=checkpoint_info.max_seq_len,
        )
        warnings: list[str] = []
        if checkpoint_info.num_views_range is not None:
            low, high = checkpoint_info.num_views_range
            if not low <= len(cameras) <= high:
                warnings.append(
                    "Selected camera count is outside the training sampling range."
                )
        return {
            "warnings": warnings,
            "family": info,
            "checkpoint": checkpoint_info,
            "device": str(device),
            "mode": mode,
            "scene_id": scene_id,
            "scene_dir": scene_dir,
            "num_frames": num_frames,
            "num_cameras": num_cameras,
            "camera_ids": camera_ids,
            "fps": float(scene_meta["fps"]),
            "num_persons": int(scalars.get("num_persons", 1)),
            "cameras": cameras,
            "reference_camera_id": reference_camera_id,
            "window_start": start,
            "window_length": length,
        }

    @staticmethod
    def _validate_cameras(
        cameras: Sequence[int],
        checkpoint: CheckpointInfo,
        num_cameras: int,
        *,
        mode: str,
    ) -> tuple[int, ...]:
        if not cameras:
            raise SceneCatalogError("at least one camera index is required.")
        if len(set(cameras)) != len(cameras):
            raise SceneCatalogError("camera indices must be distinct.")
        for index in cameras:
            if index < 0 or index >= num_cameras:
                raise SceneCatalogError(
                    f"camera index {index} is out of range 0..{num_cameras - 1}."
                )
        if mode == MODE_TRACKING:
            if checkpoint.camera_candidates is not None:
                outside = [
                    index
                    for index in cameras
                    if index not in checkpoint.camera_candidates
                ]
                if outside:
                    raise SceneCatalogError(
                        "このチェックポイントの camera_candidates "
                        f"{list(checkpoint.camera_candidates)!r} に含まれない"
                        f"カメラが指定されました: {outside!r}."
                    )
            return tuple(int(index) for index in cameras)
        if checkpoint.max_views is not None and len(cameras) > checkpoint.max_views:
            raise SceneCatalogError(
                f"checkpoint accepts at most {checkpoint.max_views} cameras, "
                f"got {len(cameras)}."
            )
        if checkpoint.reference and not 3 <= len(cameras) <= 4:
            raise SceneCatalogError(
                f"reference モデルは 3〜4 カメラを要求します (received {len(cameras)})."
            )
        return tuple(int(index) for index in cameras)

    @staticmethod
    def _validate_reference(
        reference_camera_id: str | None,
        checkpoint: CheckpointInfo,
        cameras: Sequence[int],
        camera_ids: Sequence[str],
    ) -> str | None:
        if checkpoint.selector != CAMERA_VIEW_V2_SELECTOR:
            if reference_camera_id is not None:
                raise SceneCatalogError(
                    "physical_v1 の推論に reference_camera_id は指定できません。"
                )
            return None
        if not isinstance(reference_camera_id, str) or not reference_camera_id.strip():
            raise SceneCatalogError(
                "camera_view_v2 の推論には reference_camera_id が必須です。"
            )
        if reference_camera_id not in camera_ids:
            raise SceneCatalogError(
                f"reference_camera_id {reference_camera_id!r} is not a scene camera id; "
                f"available: {list(camera_ids)!r}."
            )
        selected_ids = [camera_ids[index] for index in cameras]
        if reference_camera_id not in selected_ids:
            raise SceneCatalogError(
                f"reference_camera_id {reference_camera_id!r} must be one of the "
                f"selected cameras {selected_ids!r}."
            )
        return reference_camera_id

    @staticmethod
    def _validate_window(
        start: int,
        length: int,
        *,
        num_frames: int,
        max_seq_len: int | None,
    ) -> tuple[int, int]:
        if start < 0:
            raise SceneCatalogError(f"window_start must be >= 0, got {start}.")
        if start >= num_frames:
            raise SceneCatalogError(
                f"window_start {start} is outside the scene (num_frames={num_frames})."
            )
        if length < 1:
            raise SceneCatalogError(f"window_length must be >= 1, got {length}.")
        if max_seq_len is not None and length > max_seq_len:
            raise SceneCatalogError(
                f"window_length {length} exceeds the model max_seq_len={max_seq_len}."
            )
        if start + length > num_frames:
            raise SceneCatalogError(
                f"window [{start}, {start + length}) exceeds num_frames={num_frames}."
            )
        return start, length

    def _run_prediction(
        self, request: PredictionRequest, window: Mapping[str, Any]
    ) -> PredictionResult:
        if cast("str", window["mode"]) == MODE_TRACKING:
            return self._run_tracking_prediction(request, window)
        return self._run_single_prediction(request, window)

    def _run_single_prediction(
        self, request: PredictionRequest, window: Mapping[str, Any]
    ) -> PredictionResult:
        checkpoint_info = cast("CheckpointInfo", window["checkpoint"])
        family = cast("FamilyInfo", window["family"])
        contract = resolve_court_keypoint_contract(family.selector)
        scene = load_scene(
            cast("Path", window["scene_dir"]),
            court_keypoint_contract=contract,
        )
        start = cast("int", window["window_start"])
        length = cast("int", window["window_length"])
        cameras = cast("tuple[int, ...]", window["cameras"])
        reference_camera_id = cast("str | None", window["reference_camera_id"])

        windowed = window_scene(scene, start, length)
        warnings = list(cast("list[str]", window.get("warnings", [])))
        if start != 0 or length != int(window["num_frames"]):
            warnings.append(
                f"シーン全 {window['num_frames']} frame のうち "
                f"[{start}, {start + length}) のみを推論しました。"
            )

        predictor = self._predictor(checkpoint_info, contract, request.device)
        with torch.no_grad():
            decoded = predictor.predict_scene(
                windowed, cameras, reference_camera_id=reference_camera_id
            )
        provenance = decoded.court_reference_provenance
        if provenance is None or len(provenance) != 1:
            raise SceneCatalogError(
                "PLCS scene prediction must return exactly one reference provenance."
            )
        record = provenance[0]

        pred_position_m = normalized_points_target_to_physical(
            decoded.position, record
        ).squeeze(0)
        pred_position_norm = normalize_court_position(pred_position_m)
        pred_rotation = headings_target_to_physical(decoded.rotation, record).squeeze(0)

        gt_position_m = np.asarray(windowed["position"], dtype=np.float64) * _SCALE_M
        gt_rotation = np.asarray(windowed["rotation"], dtype=np.float64)
        gt_joints_m = ground_truth_world_joints(windowed)
        gt_canonical = world_pose_to_canonical_pose(
            torch.from_numpy(gt_joints_m),
            torch.from_numpy(np.asarray(windowed["position"], dtype=np.float32)),
            torch.from_numpy(np.asarray(windowed["rotation"], dtype=np.float32)),
        )

        if request.canonical_pose_source == "prediction":
            if decoded.canonical_pose is None:
                warnings.append(
                    "canonical_pose_source=prediction ですが、このモデルは "
                    "canonical pose を出力しません。予測骨格は表示できません。"
                )
                pred_canonical = None
            else:
                pred_canonical = decoded.canonical_pose.squeeze(0)
        else:
            pred_canonical = gt_canonical

        pred_joints_m: NDArray[np.float32] | None = None
        if pred_canonical is not None:
            pred_joints_m = (
                canonical_pose_to_world_pose(
                    pred_canonical,
                    torch.from_numpy(np.asarray(pred_position_norm, dtype=np.float32)),
                    torch.from_numpy(np.asarray(pred_rotation, dtype=np.float32)),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

        return self._build_result(
            request=request,
            window=window,
            warnings=warnings,
            gt_position_m=gt_position_m.astype(np.float32, copy=False),
            gt_rotation=gt_rotation.astype(np.float32, copy=False),
            gt_joints_m=gt_joints_m.astype(np.float32, copy=False),
            pred_position_m=pred_position_m.numpy().astype(np.float32, copy=False),
            pred_rotation=np.asarray(pred_rotation, dtype=np.float32),
            pred_joints_m=pred_joints_m,
        )

    # ----------------------------------------------------------- track-query

    def _run_tracking_prediction(
        self, request: PredictionRequest, window: Mapping[str, Any]
    ) -> PredictionResult:
        """Assemble the shared track-query batch, predict, and pack tracks."""
        checkpoint_info = cast("CheckpointInfo", window["checkpoint"])
        family = cast("FamilyInfo", window["family"])
        contract = resolve_court_keypoint_contract(family.selector)
        config = self._checkpoint_config(request.checkpoint)
        start = cast("int", window["window_start"])
        length = cast("int", window["window_length"])
        cameras = cast("tuple[int, ...]", window["cameras"])
        reference_camera_id = cast("str | None", window["reference_camera_id"])

        warnings = list(cast("list[str]", window.get("warnings", [])))
        if start != 0 or length != int(window["num_frames"]):
            warnings.append(
                f"シーン全 {window['num_frames']} frame のうち "
                f"[{start}, {start + length}) のみを推論しました。"
            )

        tracking_window = TrackingWindow(
            family_dir=self.data_root / family.id,
            scene_id=cast("str", window["scene_id"]),
            cameras=cameras,
            start=start,
            length=length,
            reference_camera_id=reference_camera_id,
        )
        try:
            batch = build_tracking_batch(config=config, window=tracking_window)
        except TrackingSceneError as error:
            raise SceneCatalogError(
                f"track-query バッチを構築できません: {error}"
            ) from error
        metrics_config = tracking_metric_config(config)
        reference_metadata = reference_metadata_from_batch(batch)

        predictor = self._tracking_predictor(checkpoint_info, contract, request.device)
        result = predictor.predict(
            human_kp=cast("torch.Tensor", batch["human_kp"]),
            human_vis=cast("torch.Tensor", batch["human_vis"]),
            court_kp=cast("torch.Tensor", batch["court_kp"]),
            court_vis=cast("torch.Tensor", batch["court_vis"]),
            padding_mask=cast("torch.Tensor", batch["padding_mask"]),
            tracking_metrics=metrics_config,
            denormalize=True,
            court_keypoint_metadata=batch["court_keypoint_metadata"],
            court_reference_provenance=batch.get("court_reference_provenance"),
            reference_metadata=reference_metadata,
        )
        pred_position = (
            cast("torch.Tensor", result["position_meters"])
            .numpy()[0]
            .astype(np.float64)
        )
        pred_yaw = (
            cast("torch.Tensor", result["yaw_radians"]).numpy()[0].astype(np.float64)
        )
        pred_presence = cast("torch.Tensor", result["presence"]).numpy()[0].astype(bool)
        pred_rotation = np.stack((np.cos(pred_yaw), np.sin(pred_yaw)), axis=-1)

        gt_position, gt_rotation, gt_joints, gt_presence = self._tracking_ground_truth(
            window, batch
        )

        warnings.append(
            "multi-object の予測骨格は query slot が時間方向で人物を再利用するため"
            "表示しません (位置と heading のみ)。"
        )
        if request.canonical_pose_source == "prediction":
            warnings.append(
                "canonical_pose_source=prediction は multi-object では未使用です "
                "(予測骨格を表示しないため)。"
            )

        metrics = match_tracks(
            pred_position_m=pred_position,
            pred_present=pred_presence,
            pred_rotation=pred_rotation,
            gt_position_m=gt_position,
            gt_present=gt_presence,
            gt_rotation=gt_rotation,
        )
        return self._build_tracking_result(
            request=request,
            window=window,
            warnings=warnings,
            gt_position=gt_position.astype(np.float32, copy=False),
            gt_rotation=gt_rotation.astype(np.float32, copy=False),
            gt_joints=gt_joints.astype(np.float32, copy=False),
            gt_presence=gt_presence.astype(np.float32, copy=False),
            pred_position=pred_position.astype(np.float32, copy=False),
            pred_rotation=pred_rotation.astype(np.float32, copy=False),
            pred_presence=pred_presence.astype(np.float32, copy=False),
            metrics=metrics,
            presence_threshold=metrics_config.presence_threshold,
        )

    def _tracking_ground_truth(
        self, window: Mapping[str, Any], batch: Mapping[str, Any]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float32],
        NDArray[np.bool_],
    ]:
        """Return physical ``(T,P,...)`` ground truth for the windowed scene."""
        contract = resolve_court_keypoint_contract(
            cast("FamilyInfo", window["family"]).selector
        )
        scene = load_scene(
            cast("Path", window["scene_dir"]), court_keypoint_contract=contract
        )
        start = cast("int", window["window_start"])
        stop = start + cast("int", window["window_length"])
        windowed = window_scene(scene, start, stop - start)
        position = np.asarray(windowed["position"], dtype=np.float64) * _SCALE_M
        rotation = np.asarray(windowed["rotation"], dtype=np.float64)
        if position.ndim == 2:
            position = position[:, None]
            rotation = rotation[:, None]
        joints = ground_truth_world_joints(windowed)
        if joints.ndim == 3:
            joints = joints[:, None]
        if joints.shape[:2] != position.shape[:2]:
            raise SceneCatalogError(
                "scene human_kp_3d tracks do not match position tracks: "
                f"{joints.shape[:2]} vs {position.shape[:2]}."
            )
        if "person_present" in windowed:
            presence = np.asarray(windowed["person_present"], dtype=bool)
        else:
            presence = np.ones(position.shape[:2], dtype=bool)
        if presence.shape != position.shape[:2]:
            raise SceneCatalogError(
                "person_present must match the physical (T,P) axes: "
                f"{presence.shape} vs {position.shape[:2]}."
            )
        # Cross-check the packing the model was fed so GT slots and rows agree.
        instance_id = cast("torch.Tensor", batch["target_instance_id"])
        if int(instance_id.max().item()) >= position.shape[1]:
            raise SceneCatalogError(
                "lifecycle packing references a physical track outside the scene."
            )
        return position, rotation, joints, presence

    def _build_tracking_result(
        self,
        *,
        request: PredictionRequest,
        window: Mapping[str, Any],
        warnings: list[str],
        gt_position: NDArray[np.float32],
        gt_rotation: NDArray[np.float32],
        gt_joints: NDArray[np.float32],
        gt_presence: NDArray[np.float32],
        pred_position: NDArray[np.float32],
        pred_rotation: NDArray[np.float32],
        pred_presence: NDArray[np.float32],
        metrics: Any,
        presence_threshold: float,
    ) -> PredictionResult:
        builder = PayloadBuilder()
        gt_tracks: list[dict[str, Any]] = []
        for index in range(gt_position.shape[1]):
            gt_tracks.append(
                {
                    "kind": "gt",
                    "label": f"GT {index}",
                    "object_index": index,
                    "has_joints": True,
                    "position": builder.append(gt_position[:, index]),
                    "rotation": builder.append(gt_rotation[:, index]),
                    "presence": builder.append(gt_presence[:, index]),
                    "joints": builder.append(gt_joints[:, index]),
                }
            )
        pred_tracks: list[dict[str, Any]] = []
        for index in range(pred_position.shape[1]):
            pred_tracks.append(
                {
                    "kind": "pred",
                    "label": f"query {index}",
                    "object_index": index,
                    "has_joints": False,
                    "position": builder.append(pred_position[:, index]),
                    "rotation": builder.append(pred_rotation[:, index]),
                    "presence": builder.append(pred_presence[:, index]),
                    "joints": None,
                }
            )
        header: dict[str, Any] = {
            "scene": self._scene_header(window),
            "checkpoint": self._checkpoint_header(
                cast("CheckpointInfo", window["checkpoint"])
            ),
            "request": {
                "cameras": list(request.cameras),
                "reference_camera_id": request.reference_camera_id,
                "canonical_pose_source": request.canonical_pose_source,
                "device": request.device,
                "window_start": window["window_start"],
                "window_length": window["window_length"],
            },
            "mode": MODE_TRACKING,
            "court": self._court_document(),
            "skeleton": {
                "names": list(COCO_KP_NAMES),
                "edges": [list(edge) for edge in COCO17_SKELETON],
            },
            "tracks": [*gt_tracks, *pred_tracks],
            "payload_dtype": "float32",
            "payload_elements": builder.total,
            "metrics": self._tracking_metrics(metrics, window, presence_threshold),
            "warnings": warnings,
        }
        return PredictionResult(header=header, payload=builder.build())

    @staticmethod
    def _tracking_metrics(
        match: TrackMatch, window: Mapping[str, Any], presence_threshold: float
    ) -> dict[str, Any]:
        return {
            "scope": "multi_object_tracks",
            "matching": "hungarian_per_frame",
            "note": (
                "フレームごとに物理座標の Hungarian 最小コスト割当で予測slotと"
                "GT trackを対応付けています。学習時の lifecycle metric とは"
                "定義が異なるため直接比較しないでください。"
            ),
            "presence_threshold": float(presence_threshold),
            "window_start": window["window_start"],
            "window_length": window["window_length"],
            "scene_frames": window["num_frames"],
            "cameras": len(cast("tuple[int, ...]", window["cameras"])),
            "matched_frames": match.matched_frames,
            "matched_pairs": match.matched_pairs,
            "unmatched_prediction": match.unmatched_prediction,
            "unmatched_ground_truth": match.unmatched_ground_truth,
            "position_error_m": _distribution(match.position_error_m),
            "yaw_error_deg": _distribution(match.yaw_error_deg),
        }

    @staticmethod
    def _scene_header(window: Mapping[str, Any]) -> dict[str, Any]:
        family = cast("FamilyInfo", window["family"])
        return {
            "family": family.id,
            "id": window["scene_id"],
            "fps": window["fps"],
            "num_frames": window["num_frames"],
            "num_cameras": window["num_cameras"],
            "num_persons": window["num_persons"],
            "selector": family.selector,
            "objects": family.objects,
        }

    @staticmethod
    def _checkpoint_header(checkpoint_info: CheckpointInfo) -> dict[str, Any]:
        return {
            "id": checkpoint_info.id,
            "model_name": checkpoint_info.model_name,
            "selector": checkpoint_info.selector,
            "input_profile": checkpoint_info.input_profile,
            "reference": checkpoint_info.reference,
            "max_views": checkpoint_info.max_views,
            "max_seq_len": checkpoint_info.max_seq_len,
        }

    @staticmethod
    def _court_document() -> dict[str, Any]:
        court = court_keypoints_3d(STANDARD_COURT_CONFIG).detach().cpu().numpy()
        return {
            "keypoints": court.astype(float).tolist(),
            "edges": [list(edge) for edge in COURT_SKELETON],
        }

    def _predictor(
        self,
        checkpoint_info: CheckpointInfo,
        contract: CourtKeypointContract,
        device: str,
    ) -> PLCSPredictor:
        key = self._predictor_key("single", checkpoint_info.path, device)
        cached = self._predictor_cache.get(key)
        if cached is not None:
            self._predictor_cache.move_to_end(key)
            return cast("PLCSPredictor", cached)
        resolved_device = resolve_device(device)
        predictor = load_standard_predictor(
            checkpoint_path=checkpoint_info.path,
            resolver=self._checkpoint_resolver(checkpoint_info.path),
            device=resolved_device,
            court_keypoint_contract=contract,
        )
        self._remember_predictor(key, predictor)
        return predictor

    def _tracking_predictor(
        self,
        checkpoint_info: CheckpointInfo,
        contract: CourtKeypointContract,
        device: str,
    ) -> PLCSTrackingPredictor:
        key = self._predictor_key("tracking", checkpoint_info.path, device)
        cached = self._predictor_cache.get(key)
        if cached is not None:
            self._predictor_cache.move_to_end(key)
            return cast("PLCSTrackingPredictor", cached)
        resolved_device = resolve_device(device)
        predictor = load_tracking_predictor(
            checkpoint_path=checkpoint_info.path,
            resolver=self._checkpoint_resolver(checkpoint_info.path),
            device=resolved_device,
            court_keypoint_contract=contract,
        )
        self._remember_predictor(key, predictor)
        return predictor

    @staticmethod
    def _predictor_key(kind: str, path: Path, device: str) -> str:
        stat = path.stat()
        return f"{kind}|{path}|{stat.st_size}|{stat.st_mtime_ns}|{device}"

    def _checkpoint_resolver(self, path: Path) -> PathResolver:
        for root in (self.checkpoint_root, *self.extra_checkpoint_roots):
            if path.resolve().is_relative_to(root):
                return PathResolver(replace(self._resolver.roots, checkpoint_root=root))
        raise SceneCatalogError("Checkpoint is outside the configured roots.")

    def _remember_predictor(
        self, key: str, predictor: PLCSPredictor | PLCSTrackingPredictor
    ) -> None:
        self._predictor_cache[key] = predictor
        self._predictor_cache.move_to_end(key)
        while len(self._predictor_cache) > 2:
            self._predictor_cache.popitem(last=False)

    # ----------------------------------------------------------------- payload

    def _build_result(
        self,
        *,
        request: PredictionRequest,
        window: Mapping[str, Any],
        warnings: list[str],
        gt_position_m: NDArray[np.float32],
        gt_rotation: NDArray[np.float32],
        gt_joints_m: NDArray[np.float32],
        pred_position_m: NDArray[np.float32],
        pred_rotation: NDArray[np.float32],
        pred_joints_m: NDArray[np.float32] | None,
    ) -> PredictionResult:
        checkpoint_info = cast("CheckpointInfo", window["checkpoint"])
        builder = PayloadBuilder()

        gt_track = {
            "kind": "gt",
            "label": "GT",
            "has_joints": True,
            "position": builder.append(gt_position_m),
            "joints": builder.append(gt_joints_m),
        }
        pred_track: dict[str, Any] = {
            "kind": "pred",
            "label": "推論",
            "has_joints": pred_joints_m is not None,
            "position": builder.append(pred_position_m),
            "joints": None,
        }
        if pred_joints_m is not None:
            pred_track["joints"] = builder.append(pred_joints_m)

        header: dict[str, Any] = {
            "scene": self._scene_header(window),
            "checkpoint": self._checkpoint_header(checkpoint_info),
            "request": {
                "cameras": list(request.cameras),
                "reference_camera_id": request.reference_camera_id,
                "canonical_pose_source": request.canonical_pose_source,
                "device": request.device,
                "window_start": window["window_start"],
                "window_length": window["window_length"],
            },
            "mode": MODE_SINGLE,
            "court": self._court_document(),
            "skeleton": {
                "names": list(COCO_KP_NAMES),
                "edges": [list(edge) for edge in COCO17_SKELETON],
            },
            "tracks": [gt_track, pred_track],
            "payload_dtype": "float32",
            "payload_elements": builder.total,
            "metrics": self._metrics(
                gt_position_m=gt_position_m,
                gt_rotation=gt_rotation,
                gt_joints_m=gt_joints_m,
                pred_position_m=pred_position_m,
                pred_rotation=pred_rotation,
                pred_joints_m=pred_joints_m,
                window=window,
            ),
            "warnings": warnings,
        }
        return PredictionResult(header=header, payload=builder.build())

    @staticmethod
    def _metrics(
        *,
        gt_position_m: NDArray[np.float32],
        gt_rotation: NDArray[np.float32] | None,
        gt_joints_m: NDArray[np.float32],
        pred_position_m: NDArray[np.float32],
        pred_rotation: NDArray[np.float32],
        pred_joints_m: NDArray[np.float32] | None,
        window: Mapping[str, Any],
    ) -> dict[str, Any]:
        frames = int(gt_position_m.shape[0])
        position_error = np.linalg.norm(
            gt_position_m.astype(np.float64) - pred_position_m.astype(np.float64),
            axis=1,
        )
        metrics: dict[str, Any] = {
            "scope": "single_object",
            "window_start": window["window_start"],
            "window_length": window["window_length"],
            "scene_frames": window["num_frames"],
            "cameras": len(cast("tuple[int, ...]", window["cameras"])),
            "gt_frames": frames,
            "pred_frames": int(pred_position_m.shape[0]),
            "position_error_m": _distribution(position_error),
        }
        yaw_error = yaw_error_degrees(gt_rotation, pred_rotation)
        if yaw_error is not None:
            metrics["yaw_error_deg"] = _distribution(yaw_error)
        if pred_joints_m is not None and pred_joints_m.shape == gt_joints_m.shape:
            per_frame = np.linalg.norm(
                gt_joints_m.astype(np.float64) - pred_joints_m.astype(np.float64),
                axis=-1,
            ).mean(axis=1)
            metrics["joints_error_m"] = _distribution(per_frame)
        return metrics


def _distribution(values: NDArray[np.float64]) -> dict[str, float | None]:
    """Return mean/median/max of a finite sample, or ``None`` when empty."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": _finite_or_none(float(np.mean(finite))),
        "median": _finite_or_none(float(np.median(finite))),
        "max": _finite_or_none(float(np.max(finite))),
    }


def yaw_error_degrees(
    gt_rotation: NDArray[np.float32] | None,
    pred_rotation: NDArray[np.float32],
) -> NDArray[np.float64] | None:
    """Return per-frame absolute yaw error in degrees for ``(cos, sin)`` rows."""
    if gt_rotation is None:
        return None
    gt = np.asarray(gt_rotation, dtype=np.float64)
    pred = np.asarray(pred_rotation, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 2 or gt.shape[1] != 2:
        return None
    cross = gt[:, 0] * pred[:, 1] - gt[:, 1] * pred[:, 0]
    dot = gt[:, 0] * pred[:, 0] + gt[:, 1] * pred[:, 1]
    error: NDArray[np.float64] = np.abs(np.degrees(np.arctan2(cross, dot)))
    return error


def ground_truth_world_joints(scene: Mapping[str, Any]) -> NDArray[np.float32]:
    """Return COCO17 world joints in metres, without inventing pose data.

    Accepts both ``(T,17,3)`` single-object and ``(T,P,17,3)`` multi-object
    layouts so one helper serves the single and track-query payloads.
    """
    if "human_kp_3d" in scene:
        joints = np.asarray(scene["human_kp_3d"], dtype=np.float32)
        if joints.ndim == 3 and joints.shape[1:] == (17, 3):
            return joints
        if joints.ndim == 4 and joints.shape[2:] == (17, 3):
            return joints
        raise SceneCatalogError(
            "human_kp_3d must have shape (T,17,3) or (T,P,17,3) metres, "
            f"got {joints.shape}."
        )
    raise SceneCatalogError(
        "scene has no human_kp_3d; the inference UI requires stored COCO17 "
        "world joints for the ground-truth overlay."
    )


def window_scene(scene: Mapping[str, Any], start: int, length: int) -> AttrDict:
    """Return a frame-sliced copy of a loaded scene for windowed inference."""
    stop = start + length
    cameras: list[AttrDict] = []
    for camera in cast("Sequence[Mapping[str, Any]]", scene["cameras"]):
        cameras.append(
            AttrDict(
                params=camera["params"],
                human_kp_uv=np.asarray(camera["human_kp_uv"])[start:stop],
                human_kp_vis=np.asarray(camera["human_kp_vis"])[start:stop],
                human_visibility_ratio=camera["human_visibility_ratio"],
                court_kp_uv=np.asarray(camera["court_kp_uv"])[start:stop],
                court_kp_vis=np.asarray(camera["court_kp_vis"])[start:stop],
                court_visibility_count=camera["court_visibility_count"],
                court_view=camera["court_view"],
            )
        )
    meta = dict(cast("Mapping[str, Any]", scene["meta"]))
    meta["num_frames"] = int(length)
    windowed = AttrDict(
        meta=meta,
        position=np.asarray(scene["position"])[start:stop],
        rotation=np.asarray(scene["rotation"])[start:stop],
        canonical_pose_3d=np.asarray(scene["canonical_pose_3d"])[start:stop],
        num_cameras=int(cast("int", scene["num_cameras"])),
        cameras=cameras,
        num_persons=int(cast("int", scene["num_persons"])),
        court_keypoint_contract=scene["court_keypoint_contract"],
        court_keypoint_validation=scene["court_keypoint_validation"],
    )
    if "human_kp_3d" in scene:
        windowed["human_kp_3d"] = np.asarray(scene["human_kp_3d"])[start:stop]
    if "person_present" in scene:
        windowed["person_present"] = np.asarray(scene["person_present"])[start:stop]
    return windowed


__all__ = [
    "DEFAULT_SPLIT",
    "DEFAULT_CAMERA_DEPTH_M",
    "MAX_SCENE_LIMIT",
    "MODE_PREVIEW",
    "MODE_SINGLE",
    "MODE_TRACKING",
    "SPLITS",
    "FamilyInfo",
    "InferenceService",
    "PayloadBuilder",
    "PredictionRequest",
    "PredictionResult",
    "SceneCatalogError",
    "checkpoint_mode",
    "ground_truth_world_joints",
    "window_scene",
    "yaw_error_degrees",
]
