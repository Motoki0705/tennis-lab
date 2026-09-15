"""Read-only catalog and GPU inference orchestration for the BLCS web UI.

Everything is derived from the configured roots: ``outputs_root`` and
``checkpoints_root`` supply ``*.ckpt`` files, and ``data_root/blcs/<form>``
supplies generated scenes. Inference runs the repository's
:class:`~src.tasks.blcs.inference.predictor.BLCSPredictor` in non-overlapping
windows and reports the metric against the scene's ground-truth trajectory.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Final, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.inference.tracking_predictor import BLCSTrackingPredictor
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    blcs_reference_metadata_from_batch,
    blcs_track_query_prediction_to_physical,
    blcs_trajectory_prediction_to_physical,
)
from src.tasks.blcs.visualization.inference.tracking import (
    DEFAULT_POSITION_THRESHOLD_M,
    TrackingInput,
    build_tracking_input,
    position_threshold_from_config,
    tracking_metrics,
    tracking_prediction_payload,
)
from src.tasks.blcs.visualization.io.scene import SceneBundle, load_scene_bundle
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.device import DeviceSelectionError, resolve_device
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court import (
    COURT_KP_NAMES,
    COURT_SKELETON,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    SERVICE_LINE_DISTANCE,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

PHYSICAL_SELECTOR: Final = "physical_v1"
CAMERA_VIEW_SELECTOR: Final = "camera_view_v2"
CAMERA_VIEW_SUFFIX: Final = "_camera_view_v2"
TRAJECTORY_MODEL_NAMES: Final[frozenset[str]] = frozenset(
    {"blcs", "blcs_multiview_axial", "blcs_multiview_axial_reference"}
)
TRACKING_MODEL_NAMES: Final[frozenset[str]] = frozenset(
    {"blcs_track_query", "blcs_track_query_reference"}
)
AXIAL_REFERENCE_MODEL_NAME: Final = "blcs_multiview_axial_reference"
SINGLE_OBJECT_FORMS: Final[tuple[str, ...]] = (
    "single_object",
    "single_object_broadcast",
)
SINGLE_REFERENCE_FORMS: Final[tuple[str, ...]] = ("single_object_camera_view_v2",)
MULTI_OBJECT_FORMS: Final[tuple[str, ...]] = (
    "multi_object",
    "multi_object_broadcast",
)
MULTI_REFERENCE_FORMS: Final[tuple[str, ...]] = ("multi_object_camera_view_v2",)
KNOWN_FORMS: Final[tuple[str, ...]] = (
    SINGLE_OBJECT_FORMS
    + SINGLE_REFERENCE_FORMS
    + MULTI_OBJECT_FORMS
    + MULTI_REFERENCE_FORMS
)
SPLITS: Final[tuple[str, ...]] = ("all", "train", "val", "test")
LIMIT_MIN: Final = 1
LIMIT_MAX: Final = 2000
FORM_SAMPLE_SIZE: Final = 20


def world_payload() -> dict[str, Any]:
    """Return the world-frame identity shared by every scene and prediction."""
    return {
        "units": "metres",
        "up_axis": "z",
        "coordinate_system": "right_handed",
        "court_contract": PHYSICAL_SELECTOR,
    }


def court_payload() -> dict[str, Any]:
    """Return the physical court geometry the frontend draws."""
    return {
        "keypoints": court_keypoints_3d(STANDARD_COURT_CONFIG).tolist(),
        "names": list(COURT_KP_NAMES),
        "lines": [list(edge) for edge in COURT_SKELETON],
        "half_length": float(HALF_LENGTH),
        "half_doubles_width": float(HALF_DOUBLES_WIDTH),
        "service_line_distance": float(SERVICE_LINE_DISTANCE),
        "net_height_center": float(NET_HEIGHT_CENTER),
        "net_height_post": float(NET_HEIGHT_POST),
    }


def _as_int(value: object, *, default: int | None) -> int | None:
    """Return ``value`` as an int when it is an integral number, else ``default``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _int_pair(value: object) -> tuple[int, int] | None:
    """Return an ``[lo, hi]`` integer pair, or ``None`` when it is malformed."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = _as_int(value[0], default=None)
    high = _as_int(value[1], default=None)
    if low is None or high is None or low <= 0 or high < low:
        return None
    return (low, high)


def _config_container(config: object) -> Mapping[str, object]:
    """Return a plain read-only mapping for a checkpoint's stored config."""
    if isinstance(config, DictConfig):
        converted = OmegaConf.to_container(config, resolve=False)
        if isinstance(converted, dict):
            return cast("Mapping[str, object]", converted)
        raise ValueError("Checkpoint config must convert to a mapping.")
    if isinstance(config, Mapping):
        return cast("Mapping[str, object]", config)
    raise ValueError("Checkpoint config must be a mapping.")


def _lookup(container: object, *keys: str) -> object | None:
    """Walk nested mapping keys, returning ``None`` at the first missing key."""
    current = container
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        if key not in current:
            return None
        current = cast("Mapping[str, object]", current)[key]
    return current


def _model_family(name: str) -> str:
    """Classify one checkpoint ``model.name`` into its inference family."""
    if name in TRACKING_MODEL_NAMES:
        return "tracking"
    if name in TRAJECTORY_MODEL_NAMES:
        return "trajectory"
    raise ValueError(f"Unsupported BLCS checkpoint model name {name!r}.")


def _validate_model_selector(name: str, selector: str) -> bool:
    """Validate the Court selector against the model and report reference usage.

    ``model.name`` is authoritative: only ``*_reference`` models consume a
    reference camera, so those checkpoints must select ``camera_view_v2`` while
    every other model must select ``physical_v1``. Parsing this from
    ``model.name`` plus ``court_keypoints.selector`` rather than trusting the
    ``data.scene_dir`` basename keeps a mismatched checkpoint from silently
    importing the wrong Court contract.
    """
    if selector not in {PHYSICAL_SELECTOR, CAMERA_VIEW_SELECTOR}:
        raise ValueError(
            f"Unsupported court_keypoints.selector {selector!r}; expected "
            f"{PHYSICAL_SELECTOR!r} or {CAMERA_VIEW_SELECTOR!r}."
        )
    reference = name.endswith("_reference")
    expected_selector = CAMERA_VIEW_SELECTOR if reference else PHYSICAL_SELECTOR
    if selector != expected_selector:
        raise ValueError(
            f"Checkpoint model.name {name!r} requires court_keypoints.selector "
            f"{expected_selector!r}, got {selector!r}."
        )
    return reference


def forms_for_model(name: str, selector: str) -> tuple[str, ...]:
    """Return the dataset forms a checkpoint may run from its model contract."""
    family = _model_family(name)
    reference = _validate_model_selector(name, selector)
    if family == "tracking":
        return MULTI_REFERENCE_FORMS if reference else MULTI_OBJECT_FORMS
    return SINGLE_REFERENCE_FORMS if reference else SINGLE_OBJECT_FORMS


def _validate_scene_dir_form(scene_dir: str | None, forms: tuple[str, ...]) -> None:
    """Reject a saved ``data.scene_dir`` that contradicts the model contract."""
    if scene_dir is None:
        return
    basename = Path(scene_dir).name
    if basename in KNOWN_FORMS and basename not in forms:
        raise ValueError(
            f"Checkpoint data.scene_dir {scene_dir!r} contradicts model.name and "
            f"court_keypoints.selector; expected one of {list(forms)!r}."
        )


def derive_checkpoint_metadata(config: object) -> dict[str, Any]:
    """Derive every catalog field a checkpoint exposes from its Lightning config."""
    container = _config_container(config)
    name = _lookup(container, "model", "name")
    if not isinstance(name, str) or not name:
        raise ValueError("Checkpoint config.model.name is required.")
    selector = _lookup(container, "court_keypoints", "selector")
    if not isinstance(selector, str) or not selector:
        raise ValueError("Checkpoint config.court_keypoints.selector must be a string.")
    family = _model_family(name)
    reference = _validate_model_selector(name, selector)
    object_mode = "multi" if family == "tracking" else "single"
    if family == "tracking":
        input_profile = "tracking"
    else:
        raw_profile = _lookup(container, "model", "io", "input_profile")
        if not isinstance(raw_profile, str) or not raw_profile:
            raise ValueError("Checkpoint config.model.io.input_profile is required.")
        input_profile = raw_profile
    max_seq_len = _as_int(_lookup(container, "model", "max_seq_len"), default=None)
    if family == "trajectory" and max_seq_len is None:
        raise ValueError("Checkpoint config.model.max_seq_len is required.")
    num_court_tokens = _as_int(
        _lookup(container, "model", "num_court_tokens"),
        default=_as_int(_lookup(container, "data", "num_court_kp"), default=14),
    )
    if num_court_tokens is None or num_court_tokens <= 0:
        raise ValueError("Checkpoint num_court_tokens must be positive.")
    seq_range = _lookup(container, "data", "seq_len_range")
    seq_len = max_seq_len
    if isinstance(seq_range, (list, tuple)) and len(seq_range) >= 2:
        candidate = _as_int(seq_range[1], default=None)
        if candidate is not None:
            seq_len = candidate
    if seq_len is None or seq_len <= 0:
        raise ValueError(
            "Checkpoint config.data.seq_len_range[1] or model.max_seq_len is "
            "required to size the inference window."
        )
    num_queries: int | None = None
    max_num_cameras: int | None = None
    if family == "tracking":
        num_queries = _as_int(_lookup(container, "model", "num_queries"), default=None)
        if num_queries is None or num_queries <= 0:
            raise ValueError("Tracking checkpoints require model.num_queries.")
    else:
        max_num_cameras = _as_int(
            _lookup(container, "model", "max_num_cameras"), default=None
        )
        if input_profile == "multiview" and (
            max_num_cameras is None or max_num_cameras <= 0
        ):
            raise ValueError("Multiview checkpoints require model.max_num_cameras.")
    scene_dir = _lookup(container, "data", "scene_dir")
    scene_dir_value = scene_dir if isinstance(scene_dir, str) and scene_dir else None
    _validate_scene_dir_form(scene_dir_value, forms_for_model(name, selector))
    num_views_range = _int_pair(_lookup(container, "data", "num_views_range"))
    return {
        "model_name": name,
        "model_family": family,
        "object_mode": object_mode,
        "reference": reference,
        "input_profile": input_profile,
        "court_keypoint_selector": selector,
        "scene_dir": scene_dir_value,
        "num_court_tokens": num_court_tokens,
        "max_seq_len": max_seq_len,
        "max_num_cameras": max_num_cameras,
        "num_queries": num_queries,
        "num_views_range": num_views_range,
        "seq_len": seq_len,
        "runnable": True,
        "unavailable_reason": None,
    }


@dataclass(frozen=True, slots=True)
class _ResolvedInference:
    """A validated inference request ready to execute on a compute device."""

    entry: Mapping[str, Any]
    form: str
    scene_id: str
    bundle: SceneBundle
    scene: Mapping[str, Any]
    cameras: tuple[int, ...]
    reference_camera_id: str | None
    device_key: str
    window: int
    warnings: tuple[str, ...]


class InferenceService:
    """Read-only catalog plus windowed BLCS inference for one pair of roots."""

    def __init__(
        self,
        outputs_root: Path,
        checkpoints_root: Path,
        data_root: Path,
        *,
        device: str = "auto",
        scene_cache_size: int = 4,
        predictor_cache_size: int = 2,
    ) -> None:
        self.outputs_root = self._resolve_root(outputs_root, must_exist=False)
        self.checkpoints_root = self._resolve_root(checkpoints_root, must_exist=False)
        self.data_root = self._resolve_root(data_root, must_exist=True)
        self.device = device
        self._lock = RLock()
        self._json_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._checkpoint_meta_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._predictors: OrderedDict[
            tuple[str, int, str],
            BLCSPredictor | BLCSTrackingPredictor,
        ] = OrderedDict()
        self._predictor_cache_size = max(1, predictor_cache_size)
        self._scenes: OrderedDict[tuple[str, str, str], SceneBundle] = OrderedDict()
        self._scene_cache_size = max(1, scene_cache_size)

    # ------------------------------------------------------------- roots

    @staticmethod
    def _resolve_root(path: Path, *, must_exist: bool) -> Path:
        resolved = Path(path).expanduser().resolve(strict=False)
        if must_exist and not resolved.is_dir():
            raise FileNotFoundError(f"Root is not a directory: {resolved}")
        return resolved

    def _read_json(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns)
        with self._lock:
            cached = self._json_cache.get(key)
        if cached is not None:
            return cached
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        with self._lock:
            self._json_cache[key] = loaded
        return loaded

    # --------------------------------------------------------- catalog

    def catalog(self) -> dict[str, Any]:
        """Return the checkpoint catalog, scene forms, and shared geometry."""
        form_ids = set(self._form_paths())
        return {
            "roots": [
                {
                    "id": "outputs",
                    "path": str(self.outputs_root),
                    "exists": self.outputs_root.is_dir(),
                },
                {
                    "id": "checkpoints",
                    "path": str(self.checkpoints_root),
                    "exists": self.checkpoints_root.is_dir(),
                },
            ],
            "checkpoints": self._checkpoints(form_ids),
            "scene_forms": self.scene_forms(),
            "court": court_payload(),
            "world": world_payload(),
        }

    def _checkpoints(self, form_ids: set[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for root_id, root in (
            ("outputs", self.outputs_root),
            ("checkpoints", self.checkpoints_root),
        ):
            if not root.is_dir():
                continue
            for path in sorted(p for p in root.rglob("*.ckpt") if p.is_file()):
                rel_path = path.relative_to(root).as_posix()
                entries.append(
                    self._checkpoint_entry(root_id, path, rel_path, form_ids)
                )
        return entries

    def _checkpoint_entry(
        self,
        root_id: str,
        path: Path,
        rel_path: str,
        form_ids: set[str],
    ) -> dict[str, Any]:
        stat = path.stat()
        entry: dict[str, Any] = {
            "id": f"{root_id}:{rel_path}",
            "name": path.name,
            "path": str(path),
            "rel_path": rel_path,
            "root": root_id,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        try:
            metadata = self._checkpoint_metadata(path, stat.st_size, stat.st_mtime_ns)
        except Exception as error:  # noqa: BLE001 - surfaced as an explicit reason
            entry.update(
                {
                    "model_name": None,
                    "model_family": None,
                    "object_mode": None,
                    "reference": False,
                    "input_profile": None,
                    "court_keypoint_selector": None,
                    "scene_dir": None,
                    "allowed_forms": [],
                    "runnable": False,
                    "unavailable_reason": (
                        f"Checkpoint metadata could not be read: {error}"
                    ),
                    "num_court_tokens": None,
                    "max_seq_len": None,
                    "max_num_cameras": None,
                    "num_queries": None,
                    "num_views_range": None,
                    "seq_len": None,
                }
            )
            return entry
        entry.update(metadata)
        entry["allowed_forms"] = self._allowed_forms(metadata, form_ids)
        return entry

    @staticmethod
    def _allowed_forms(
        metadata: Mapping[str, Any],
        form_ids: set[str],
    ) -> list[str]:
        candidates = forms_for_model(
            str(metadata["model_name"]),
            str(metadata["court_keypoint_selector"]),
        )
        return [name for name in candidates if name in form_ids]

    def _checkpoint_metadata(
        self,
        path: Path,
        size_bytes: int,
        mtime_ns: int,
    ) -> dict[str, Any]:
        key = (str(path), size_bytes, mtime_ns)
        with self._lock:
            cached = self._checkpoint_meta_cache.get(key)
        if cached is not None:
            return cached
        checkpoint = torch.load(
            str(path), map_location="cpu", weights_only=False, mmap=True
        )
        try:
            if not isinstance(checkpoint, Mapping):
                raise ValueError("Checkpoint file must contain a mapping.")
            hyper_parameters = checkpoint.get("hyper_parameters")
            if (
                not isinstance(hyper_parameters, Mapping)
                or "config" not in hyper_parameters
            ):
                raise ValueError("Checkpoint hyper_parameters.config is required.")
            config = cast("Mapping[str, object]", hyper_parameters)["config"]
            metadata = derive_checkpoint_metadata(config)
        finally:
            del checkpoint
        with self._lock:
            self._checkpoint_meta_cache[key] = metadata
        return metadata

    # ------------------------------------------------------ scene forms

    def _blcs_root(self) -> Path:
        return self.data_root / "blcs"

    def _form_paths(self) -> dict[str, Path]:
        base = self._blcs_root()
        if not base.is_dir():
            return {}
        forms: dict[str, Path] = {}
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "scenes").is_dir():
                forms[child.name] = child
        return forms

    def _require_form(self, form: str) -> Path:
        path = self._form_paths().get(form)
        if path is None:
            raise ValueError(f"Unknown scene form {form!r}.")
        return path

    def scene_forms(self) -> list[dict[str, Any]]:
        """Return one entry per dataset form discovered under ``data/blcs``."""
        return [
            self._scene_form(form_id, path)
            for form_id, path in self._form_paths().items()
        ]

    def _scene_form(self, form_id: str, path: Path) -> dict[str, Any]:
        scenes_dir = path / "scenes"
        scene_dirs = sorted(
            child
            for child in scenes_dir.iterdir()
            if child.is_dir() and child.name.startswith("scene_")
        )
        splits: dict[str, int] | None = None
        split_info = path / "split_info.json"
        if split_info.is_file():
            raw = self._read_json(split_info).get("n_scenes")
            if isinstance(raw, Mapping):
                splits = {
                    str(key): int(value)
                    for key, value in raw.items()
                    if _as_int(value, default=None) is not None
                }
        camera_counts: list[int] = []
        for scene_dir in scene_dirs[:FORM_SAMPLE_SIZE]:
            value = _as_int(
                self._read_json(scene_dir / "meta.json").get("num_cameras"),
                default=None,
            )
            if value is not None:
                camera_counts.append(value)
        return {
            "id": form_id,
            "path": str(path),
            "scene_count": len(scene_dirs),
            "object_mode": "multi" if form_id.startswith("multi") else "single",
            "reference": form_id.endswith(CAMERA_VIEW_SUFFIX),
            "broadcast": form_id.endswith("_broadcast"),
            "splits": splits,
            "has_split_files": (path / "test.txt").is_file(),
            "num_cameras": (
                [min(camera_counts), max(camera_counts)] if camera_counts else None
            ),
        }

    # ----------------------------------------------------------- scenes

    def scenes(
        self,
        *,
        form: str,
        split: str | None = None,
        query: str = "",
        offset: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Return one page of scene ids for a form and split."""
        form_path = self._require_form(form)
        if offset < 0:
            raise ValueError("offset must be non-negative.")
        clamped_limit = max(LIMIT_MIN, min(int(limit), LIMIT_MAX))
        effective_split = self._effective_split(form_path, split)
        ids = self._scene_ids(form_path, effective_split)
        needle = query.strip().lower()
        if needle:
            ids = [scene_id for scene_id in ids if needle in scene_id.lower()]
        total = len(ids)
        page = ids[offset : offset + clamped_limit]
        return {
            "form": form,
            "split": effective_split,
            "query": query,
            "total": total,
            "offset": offset,
            "limit": clamped_limit,
            "scenes": [self._scene_summary(form_path, scene_id) for scene_id in page],
        }

    def _effective_split(self, form_path: Path, split: str | None) -> str:
        if split is None or not split.strip():
            return "test" if (form_path / "test.txt").is_file() else "all"
        if split not in SPLITS:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {list(SPLITS)}."
            )
        return split

    def _scene_ids(self, form_path: Path, split: str) -> list[str]:
        scenes_dir = form_path / "scenes"
        if split == "all":
            found = sorted(
                child.name
                for child in scenes_dir.iterdir()
                if child.is_dir() and child.name.startswith("scene_")
            )
        else:
            split_file = form_path / f"{split}.txt"
            if not split_file.is_file():
                raise ValueError(
                    f"Split {split!r} has no scene list file for form "
                    f"{form_path.name!r}."
                )
            found = [
                line.strip()
                for line in split_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return [scene_id for scene_id in found if (scenes_dir / scene_id).is_dir()]

    def _scene_summary(self, form_path: Path, scene_id: str) -> dict[str, Any]:
        scene_dir = form_path / "scenes" / scene_id
        meta = self._read_json(scene_dir / "meta.json")
        scalars = self._read_json(scene_dir / "scalars.json")
        frame_count = _as_int(meta.get("num_frames"), default=None)
        if frame_count is None:
            raise ValueError(f"Scene {scene_id!r} meta.num_frames is required.")
        fps = meta.get("fps_out")
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError(f"Scene {scene_id!r} meta.fps_out must be positive.")
        num_cameras = _as_int(meta.get("num_cameras"), default=None)
        if num_cameras is None:
            num_cameras = _as_int(scalars.get("num_cameras"), default=None)
        if num_cameras is None:
            raise ValueError(f"Scene {scene_id!r} num_cameras is required.")
        num_balls = _as_int(scalars.get("num_balls"), default=None)
        if num_balls is None:
            raise ValueError(f"Scene {scene_id!r} scalars.num_balls is required.")
        return {
            "id": scene_id,
            "frame_count": frame_count,
            "num_cameras": num_cameras,
            "num_balls": num_balls,
            "duration_s": frame_count / float(fps),
        }

    def _scene_dir(self, form_path: Path, scene_id: str) -> Path:
        if (
            not scene_id
            or Path(scene_id).name != scene_id
            or scene_id in {".", ".."}
            or not scene_id.startswith("scene_")
        ):
            raise ValueError("scene id must be a single 'scene_*' path component.")
        scenes_dir = form_path / "scenes"
        scene_dir = (scenes_dir / scene_id).resolve(strict=False)
        if not scene_dir.is_relative_to(scenes_dir.resolve(strict=False)):
            raise ValueError("scene id escapes the dataset scenes directory.")
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory is missing: {scene_dir}")
        return scene_dir

    def _contract_for_form(self, form: str) -> CourtKeypointContract:
        selector = (
            CAMERA_VIEW_SELECTOR
            if form.endswith(CAMERA_VIEW_SUFFIX)
            else PHYSICAL_SELECTOR
        )
        return resolve_court_keypoint_contract(selector)

    def _load_bundle(
        self,
        form: str,
        scene_id: str,
        contract: CourtKeypointContract,
    ) -> SceneBundle:
        key = (form, scene_id, contract.contract_id)
        with self._lock:
            cached = self._scenes.get(key)
            if cached is not None:
                self._scenes.move_to_end(key)
                return cached
        form_path = self._require_form(form)
        bundle = load_scene_bundle(
            self._scene_dir(form_path, scene_id),
            0,
            "all",
            contract,
        )
        with self._lock:
            self._scenes[key] = bundle
            while len(self._scenes) > self._scene_cache_size:
                self._scenes.popitem(last=False)
        return bundle

    def scene(self, *, form: str, scene_id: str) -> dict[str, Any]:
        """Return ground truth and the camera list for one scene (no model)."""
        form_path = self._require_form(form)
        self._scene_dir(form_path, scene_id)
        bundle = self._load_bundle(form, scene_id, self._contract_for_form(form))
        scene = bundle.scene
        cameras: list[dict[str, Any]] = []
        reference_ids: list[str] = []
        for index, camera in enumerate(scene["cameras"]):
            camera_id = str(camera["camera_id"])
            reference_ids.append(camera_id)
            entry: dict[str, Any] = {
                "index": index,
                "camera_id": camera_id,
                "ball_visibility_ratio": float(camera["ball_visibility_ratio"]),
                "court_visibility_count": int(
                    round(float(camera["court_visibility_count"]))
                ),
            }
            params = camera.get("params")
            if isinstance(params, dict) and _is_json_value(params):
                entry["params"] = params
            cameras.append(entry)
        return {
            "form": form,
            "scene_id": scene_id,
            "fps": float(bundle.fps),
            "frame_count": int(np.asarray(scene["ball_pos_world"]).shape[0]),
            "num_balls": int(scene["num_balls"]),
            "world": world_payload(),
            "cameras": cameras,
            "reference_candidate_ids": reference_ids,
            "gt": self._gt_payload(scene),
        }

    @staticmethod
    def _gt_payload(scene: Mapping[str, Any]) -> dict[str, Any]:
        raw = np.asarray(scene["ball_pos_world"], dtype=np.float64)
        if raw.ndim == 2:
            raw = raw[:, None, :]
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError(f"Unsupported ball_pos_world shape {raw.shape}.")
        frames = int(raw.shape[0])
        tracks = int(scene["num_balls"])
        if tracks <= 0 or tracks > raw.shape[1]:
            raise ValueError(
                f"Scene num_balls {tracks} is inconsistent with ball_pos_world "
                f"{raw.shape}."
            )
        positions = raw[:, :tracks, :].reshape(-1)
        present = scene.get("ball_present")
        active: np.ndarray
        if present is None:
            active = np.ones((frames, tracks), dtype=np.int64)
        else:
            raw_present = np.asarray(present)
            if raw_present.ndim == 1:
                raw_present = raw_present[:, None]
            if raw_present.shape[0] != frames or raw_present.shape[1] < tracks:
                raise ValueError("ball_present is inconsistent with ball_pos_world.")
            active = (raw_present[:, :tracks] != 0).astype(np.int64)
        return {
            "frames": frames,
            "tracks": tracks,
            "positions": [float(value) for value in positions],
            "active": [int(value) for value in active.reshape(-1)],
        }

    # ---------------------------------------------------------- infer

    def validate_inference_request(
        self,
        *,
        checkpoint: str,
        form: str,
        scene_id: str,
        cameras: Sequence[int] | None = None,
        reference_camera_id: str | None = None,
        device: str | None = None,
        window: int | None = None,
    ) -> _ResolvedInference:
        """Resolve and validate one inference request without loading a model.

        Every request-level rejection (unknown checkpoint, disallowed form,
        missing scene, invalid cameras/window/reference, or an unavailable
        explicitly requested device) is raised here so a caller can answer with
        HTTP 422 before waiting for a GPU slot or loading model weights.
        """
        entry = self._resolve_checkpoint(checkpoint)
        if not entry["runnable"]:
            reason = entry["unavailable_reason"] or "Checkpoint is not runnable."
            raise ValueError(str(reason))
        form_path = self._require_form(form)
        if form not in entry["allowed_forms"]:
            raise ValueError(
                f"Checkpoint {entry['id']!r} cannot run form {form!r}; allowed "
                f"forms are {entry['allowed_forms']!r}."
            )
        self._scene_dir(form_path, scene_id)
        contract = self._contract_for_form(form)
        bundle = self._load_bundle(form, scene_id, contract)
        scene = bundle.scene
        num_cameras = int(scene["num_cameras"])
        selected = self._select_cameras(cameras, num_cameras)
        reference = self._resolve_reference(
            entry,
            scene,
            selected,
            reference_camera_id,
        )
        try:
            resolved_device = resolve_device(device if device else self.device)
        except DeviceSelectionError as error:
            raise ValueError(str(error)) from error
        device_key = str(resolved_device)
        trained_seq_len = int(entry["seq_len"])
        max_seq_len = entry["max_seq_len"]
        capacity = trained_seq_len if max_seq_len is None else int(max_seq_len)
        requested_window = trained_seq_len if window is None else int(window)
        warnings: list[str] = []
        training_views = entry.get("num_views_range")
        if (
            training_views is not None
            and not training_views[0] <= len(selected) <= training_views[1]
        ):
            warnings.append(
                "Selected camera count is outside the training sampling range."
            )
        if requested_window < 1:
            raise ValueError("window must be a positive integer.")
        effective_window = min(requested_window, capacity)
        if effective_window < trained_seq_len:
            warnings.append("WINDOW smaller than trained clip length")
        if window is not None and int(window) > capacity:
            warnings.append(f"WINDOW clamped to {capacity}")
        return _ResolvedInference(
            entry=entry,
            form=form,
            scene_id=scene_id,
            bundle=bundle,
            scene=scene,
            cameras=tuple(selected),
            reference_camera_id=reference,
            device_key=device_key,
            window=effective_window,
            warnings=tuple(warnings),
        )

    def infer(
        self,
        *,
        checkpoint: str,
        form: str,
        scene_id: str,
        cameras: Sequence[int] | None = None,
        reference_camera_id: str | None = None,
        device: str | None = None,
        window: int | None = None,
    ) -> dict[str, Any]:
        """Run windowed inference for one scene through the validated request."""
        started = time.perf_counter()
        request = self.validate_inference_request(
            checkpoint=checkpoint,
            form=form,
            scene_id=scene_id,
            cameras=cameras,
            reference_camera_id=reference_camera_id,
            device=device,
            window=window,
        )
        entry = request.entry
        scene = request.scene
        if entry["model_family"] == "tracking":
            return self._infer_tracking(request, started=started)
        prediction = self._run_windows(
            self._predictor(entry, request.device_key),
            scene,
            list(request.cameras),
            request.reference_camera_id,
            request.window,
        )
        gt = self._gt_payload(scene)
        frames = int(gt["frames"])
        if prediction.shape != (frames, 3):
            raise RuntimeError(
                f"Predicted trajectory shape {prediction.shape} does not match "
                f"the ground truth ({frames}, 3)."
            )
        gt_track = np.asarray(gt["positions"], dtype=np.float64).reshape(frames, 3)
        error = np.linalg.norm(prediction - gt_track, axis=-1)
        elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
        return self._prediction_response(
            request,
            prediction={
                "frames": frames,
                "tracks": 1,
                "positions": [float(value) for value in prediction.reshape(-1)],
                "presence": None,
            },
            metrics={
                "position_error_m": float(error.mean()),
                "endpoint_error_m": float(error[-1]),
                "accuracy_0p3m": float(np.mean(error <= 0.3)),
            },
            gt=gt,
            frames=frames,
            elapsed_ms=elapsed_ms,
        )

    def _prediction_response(
        self,
        request: _ResolvedInference,
        *,
        prediction: dict[str, Any],
        metrics: Mapping[str, float | None],
        gt: Mapping[str, Any],
        frames: int,
        elapsed_ms: int,
        warnings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Assemble the shared inference response envelope."""
        return {
            "checkpoint": request.entry["id"],
            "checkpoint_name": request.entry["name"],
            "form": request.form,
            "scene_id": request.scene_id,
            "device": request.device_key,
            "fps": float(request.bundle.fps),
            "frames": frames,
            "window": request.window,
            "reference_camera_id": request.reference_camera_id,
            "cameras": list(request.cameras),
            "gt": gt,
            "prediction": prediction,
            "metrics": metrics,
            "elapsed_ms": elapsed_ms,
            "warnings": list(request.warnings) if warnings is None else list(warnings),
        }

    def _infer_tracking(
        self,
        request: _ResolvedInference,
        *,
        started: float,
    ) -> dict[str, Any]:
        """Run one canonical tracking window and score it against matched GT."""
        entry = request.entry
        gt = self._gt_payload(request.scene)
        scene_frames = int(gt["frames"])
        window_start = 0
        window_length = min(request.window, scene_frames)
        if window_length <= 0:
            raise ValueError("Scene must contain at least one frame.")
        warnings = list(request.warnings)
        if window_length < scene_frames:
            warnings.append(
                f"Tracked the first {window_length} of {scene_frames} frames."
            )
        config = self._checkpoint_config(Path(str(entry["path"])))
        tracking_input = build_tracking_input(
            scene_dir=self._require_form(request.form),
            scene_id=request.scene_id,
            config=config,
            reference_camera_id=request.reference_camera_id,
            camera_indices=request.cameras,
            window_start=window_start,
            window_length=window_length,
            seed=self._seed_for(config),
        )
        if tracking_input.window_length != window_length or (
            int(tracking_input.ground_truth.shape[0]) != window_length
        ):
            raise RuntimeError(
                "The canonical tracking dataset did not produce the requested "
                f"window of {window_length} frames."
            )
        prediction = self._run_tracking(entry, request.device_key, tracking_input)
        frames = int(prediction.position.shape[1])
        if frames != window_length:
            raise RuntimeError(
                f"Tracking prediction produced {frames} frames for a window of "
                f"{window_length}."
            )
        if prediction.position.shape[0] != 1:
            raise RuntimeError("Tracking inference must decode exactly one scene.")
        gt_tracks = int(gt["tracks"])
        gt_positions = np.asarray(gt["positions"], dtype=np.float64).reshape(
            scene_frames,
            gt_tracks,
            3,
        )
        gt_active = np.asarray(gt["active"], dtype=bool).reshape(
            scene_frames,
            gt_tracks,
        )
        predicted_position = (
            prediction.position[0].detach().cpu().numpy().astype(np.float64)
        )
        predicted_present = prediction.presence[0].detach().cpu().numpy().astype(bool)
        metrics = tracking_metrics(
            predicted_position=predicted_position,
            predicted_present=predicted_present,
            target_position=gt_positions[:frames],
            target_present=gt_active[:frames],
            position_threshold_m=position_threshold_from_config(
                config,
                default=DEFAULT_POSITION_THRESHOLD_M,
            ),
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
        return self._prediction_response(
            request,
            prediction=tracking_prediction_payload(
                prediction.position,
                prediction.presence,
            ),
            metrics=metrics,
            gt=gt,
            frames=scene_frames,
            elapsed_ms=elapsed_ms,
            warnings=warnings,
        )

    def _run_tracking(
        self,
        entry: Mapping[str, Any],
        device_key: str,
        tracking_input: TrackingInput,
    ) -> BLCSTrackQueryPrediction:
        """Execute one validated tracking batch on the tracking predictor."""
        predictor = self._tracking_predictor(entry, device_key)
        batch = tracking_input.batch
        reference_metadata = blcs_reference_metadata_from_batch(batch)
        model_batch = {
            key: cast("Tensor", batch[key])
            for key in ("ball_uv", "ball_vis", "court_kp", "court_vis", "padding_mask")
        }
        prediction = predictor.predict_batch(
            model_batch,
            denormalize=True,
            court_reference_provenance=cast(
                "tuple[CourtReferenceFrameProvenance, ...]",
                batch["court_reference_provenance"],
            ),
            reference_metadata=reference_metadata,
        )
        return blcs_track_query_prediction_to_physical(prediction)

    def _resolve_checkpoint(self, reference: str) -> dict[str, Any]:
        form_ids = set(self._form_paths())
        for entry in self._checkpoints(form_ids):
            if entry["id"] == reference:
                return entry
        candidate = Path(reference)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
            for root_id, root in (
                ("outputs", self.outputs_root),
                ("checkpoints", self.checkpoints_root),
            ):
                if (
                    resolved.is_relative_to(root)
                    and resolved.is_file()
                    and resolved.suffix == ".ckpt"
                ):
                    rel_path = resolved.relative_to(root).as_posix()
                    return self._checkpoint_entry(
                        root_id,
                        resolved,
                        rel_path,
                        form_ids,
                    )
        raise ValueError(f"Unknown checkpoint {reference!r}.")

    @staticmethod
    def _select_cameras(
        cameras: Sequence[int] | None,
        num_cameras: int,
    ) -> list[int]:
        if cameras is None:
            return list(range(num_cameras))
        if len(cameras) == 0:
            raise ValueError("At least one camera must be selected.")
        selected = [int(index) for index in cameras]
        if len(set(selected)) != len(selected):
            raise ValueError("Camera indices must be unique.")
        for index in selected:
            if not 0 <= index < num_cameras:
                raise ValueError(
                    f"Camera index {index} is outside 0..{num_cameras - 1}."
                )
        return selected

    def _resolve_reference(
        self,
        entry: Mapping[str, Any],
        scene: Mapping[str, Any],
        selected: list[int],
        reference_camera_id: str | None,
    ) -> str | None:
        if entry["input_profile"] == "single" and len(selected) != 1:
            raise ValueError("Single-view checkpoints require exactly one camera.")
        capacity = entry.get("max_num_cameras")
        if capacity is not None and len(selected) > int(capacity):
            raise ValueError(f"Checkpoint accepts at most {capacity} cameras.")
        if entry["model_name"] == AXIAL_REFERENCE_MODEL_NAME and not (
            3 <= len(selected) <= 4
        ):
            raise ValueError(
                "Axial reference checkpoints require 3 or 4 selected cameras."
            )
        if not entry["reference"]:
            if reference_camera_id is not None:
                raise ValueError(
                    "Non-reference checkpoints require reference_camera_id=null."
                )
            return None
        raw_cameras = cast("Sequence[Mapping[str, Any]]", scene["cameras"])
        selected_ids = [str(raw_cameras[index]["camera_id"]) for index in selected]
        if reference_camera_id is None or not reference_camera_id.strip():
            raise ValueError(
                "Reference checkpoints require an explicit reference_camera_id."
            )
        if reference_camera_id not in selected_ids:
            raise ValueError(
                f"reference_camera_id {reference_camera_id!r} must be one of the "
                "selected cameras."
            )
        return reference_camera_id

    def _predictor(self, entry: Mapping[str, Any], device_key: str) -> BLCSPredictor:
        """Load and cache the trajectory predictor bound to one checkpoint."""
        predictor = self._load_predictor(entry, device_key)
        if not isinstance(predictor, BLCSPredictor):
            raise TypeError("Checkpoint does not compose a trajectory predictor.")
        return predictor

    def _tracking_predictor(
        self,
        entry: Mapping[str, Any],
        device_key: str,
    ) -> BLCSTrackingPredictor:
        """Load and cache the tracking predictor bound to one checkpoint."""
        predictor = self._load_predictor(entry, device_key)
        if not isinstance(predictor, BLCSTrackingPredictor):
            raise TypeError("Checkpoint does not compose a tracking predictor.")
        return predictor

    def _load_predictor(
        self,
        entry: Mapping[str, Any],
        device_key: str,
    ) -> BLCSPredictor | BLCSTrackingPredictor:
        key = (str(entry["path"]), int(entry["mtime_ns"]), device_key)
        with self._lock:
            cached = self._predictors.get(key)
            if cached is not None:
                self._predictors.move_to_end(key)
                return cached
        resolver = self._resolver_for(str(entry["root"]))
        checkpoint_path = Path(str(entry["path"]))
        if entry["model_family"] == "tracking":
            predictor: BLCSPredictor | BLCSTrackingPredictor = (
                BLCSTrackingPredictor.load_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    resolver=resolver,
                    device=device_key,
                    court_keypoints=None,
                )
            )
        else:
            predictor = BLCSPredictor.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                resolver=resolver,
                device=device_key,
                court_keypoints=None,
            )
        with self._lock:
            self._predictors[key] = predictor
            while len(self._predictors) > self._predictor_cache_size:
                self._predictors.popitem(last=False)
        return predictor

    def _checkpoint_config(self, path: Path) -> Any:
        """Return the authoritative Lightning config stored in a checkpoint."""
        checkpoint = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        try:
            if not isinstance(checkpoint, Mapping):
                raise ValueError("Checkpoint file must contain a mapping.")
            hyper_parameters = checkpoint.get("hyper_parameters")
            if (
                not isinstance(hyper_parameters, Mapping)
                or "config" not in hyper_parameters
            ):
                raise ValueError("Checkpoint hyper_parameters.config is required.")
            return cast("Mapping[str, object]", hyper_parameters)["config"]
        finally:
            del checkpoint

    @staticmethod
    def _seed_for(config: Any) -> int:
        """Return the checkpoint's run seed, defaulting to zero when absent."""
        seed = _lookup(_config_container(config), "run", "seed")
        value = _as_int(seed, default=None)
        return 0 if value is None else value

    def _resolver_for(self, root_id: str) -> PathResolver:
        checkpoint_root = (
            self.outputs_root if root_id == "outputs" else self.checkpoints_root
        )
        return PathResolver(
            RuntimePathRoots(
                project_root=PROJECT_ROOT,
                data_root=self.data_root,
                checkpoint_root=checkpoint_root,
                artifact_root=self.outputs_root,
                output_root=self.outputs_root,
                cache_root=PROJECT_ROOT / ".cache",
                external_asset_root=PROJECT_ROOT / "third_party",
            )
        )

    def _run_windows(
        self,
        predictor: BLCSPredictor,
        scene: Mapping[str, Any],
        cameras: list[int],
        reference_camera_id: str | None,
        window: int,
    ) -> np.ndarray:
        frames = int(np.asarray(scene["ball_pos_world"]).shape[0])
        if frames <= 0:
            raise ValueError("Scene must contain at least one frame.")
        chunks: list[np.ndarray] = []
        for start in range(0, frames, window):
            stop = min(start + window, frames)
            window_scene = self._slice_scene(scene, start, stop)
            prediction = predictor.predict_scene(
                window_scene,
                cameras,
                denormalize=True,
                reference_camera_id=reference_camera_id,
            )
            physical = blcs_trajectory_prediction_to_physical(prediction)
            chunks.append(physical.position.squeeze(0).detach().cpu().numpy())
        stacked: np.ndarray = np.concatenate(chunks, axis=0)
        return stacked

    @staticmethod
    def _slice_scene(
        scene: Mapping[str, Any],
        start: int,
        stop: int,
    ) -> dict[str, Any]:
        raw_cameras = cast("Sequence[Mapping[str, Any]]", scene["cameras"])
        sliced: list[dict[str, Any]] = []
        for camera in raw_cameras:
            window_camera = dict(camera)
            window_camera["ball_uv"] = np.asarray(camera["ball_uv"])[start:stop]
            window_camera["ball_vis"] = np.asarray(camera["ball_vis"])[start:stop]
            sliced.append(window_camera)
        return {**scene, "cameras": sliced}


def _is_json_value(value: object) -> bool:
    """Return whether ``value`` contains only JSON-serialisable primitives."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "InferenceService",
    "court_payload",
    "derive_checkpoint_metadata",
    "world_payload",
]
