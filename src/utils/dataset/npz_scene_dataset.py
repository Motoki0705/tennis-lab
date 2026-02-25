"""Base dataset utilities for BLCS-style NPZ scene loading."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from torch.utils.data import Dataset

from src.utils.data.scene_cache import SceneCache, get_scene_cache

SampleT = TypeVar("SampleT")


def _load_npz_payload(scene_path: Path) -> dict[str, Any]:
    """Load a full NPZ payload as copied arrays/scalars."""
    payload: dict[str, Any] = {}
    with np.load(scene_path, allow_pickle=True) as data:
        for key in data.files:
            payload[key] = data[key].copy()
    return payload


@dataclass(frozen=True)
class SceneDatasetConfig:
    """Configuration shared by NPZ scene datasets (split-file only)."""

    scene_dir: Path
    split_file: Path
    seq_len_range: tuple[int, int] = (1, 1024)
    num_views_range: tuple[int, int] = (1, 1)
    cache_max_scenes: int = 128
    camera_mode: str | int = "random"
    crop_mode: Literal["random", "center"] = "random"
    min_num_frames: int = 1
    min_num_cameras: int = 1


@dataclass(frozen=True)
class NPZSceneHeader:
    """Lightweight header for scene filtering/indexing."""

    path: Path
    meta: dict[str, Any]
    num_frames: int
    num_cameras: int


@dataclass(frozen=True)
class TemporalWindow:
    """Temporal crop window selected from a scene."""

    start: int
    end: int
    seq_len: int
    full_len: int

    @property
    def sl(self) -> slice:
        return slice(self.start, self.end)


@dataclass(frozen=True)
class CameraSelection:
    """Selected camera indices for a scene."""

    indices: tuple[int, ...]

    @property
    def primary(self) -> int:
        if not self.indices:
            raise ValueError("CameraSelection is empty.")
        return int(self.indices[0])


@dataclass(frozen=True)
class CameraViewArrays:
    """Per-camera BLCS NPZ arrays."""

    ball_uv: np.ndarray
    ball_visible: np.ndarray
    court_kp_uv: np.ndarray
    court_kp_visible: np.ndarray


@dataclass(frozen=True)
class NPZScene:
    """Loaded NPZ scene payload with BLCS schema accessors."""

    path: Path
    data: dict[str, Any]
    meta: dict[str, Any]
    num_frames: int
    num_cameras: int

    def has_key(self, key: str) -> bool:
        return key in self.data

    def require_key(self, key: str) -> None:
        if key not in self.data:
            available = ", ".join(sorted(self.data.keys()))
            raise KeyError(f"Missing NPZ key '{key}' in {self.path}. Available: {available}")

    @property
    def scene_id(self) -> str | None:
        value = self.meta.get("scene_id")
        return str(value) if value is not None else None

    @property
    def shots(self) -> list[dict[str, Any]]:
        shots = self.meta.get("shots", [])
        if not isinstance(shots, list):
            return []
        return [s for s in shots if isinstance(s, dict)]

    @property
    def rally_length(self) -> int | None:
        value = self.meta.get("rally_length")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def effective_num_frames(self, *candidate_lengths: int) -> int:
        lengths = [int(self.num_frames)]
        for n in candidate_lengths:
            if int(n) > 0:
                lengths.append(int(n))
        return min(lengths) if lengths else int(self.num_frames)

    def _camera_prefix(self, cam_idx: int) -> str:
        if cam_idx < 0 or cam_idx >= self.num_cameras:
            raise ValueError(
                f"cam_idx={cam_idx} out of range for scene with {self.num_cameras} cameras"
            )
        return f"cam_{cam_idx}_"

    def _copy_array(self, key: str) -> np.ndarray:
        self.require_key(key)
        return np.asarray(self.data[key]).copy()

    def _copy_temporal_array(
        self,
        key: str,
        window: TemporalWindow | None = None,
    ) -> np.ndarray:
        arr = self._copy_array(key)
        if window is None:
            return arr
        if arr.ndim == 0:
            raise ValueError(f"NPZ key '{key}' is scalar and cannot be temporally sliced.")
        return arr[window.sl].copy()

    def get_ball_uv(
        self,
        cam_idx: int,
        *,
        window: TemporalWindow | None = None,
    ) -> np.ndarray:
        prefix = self._camera_prefix(cam_idx)
        return self._copy_temporal_array(f"{prefix}ball_uv", window=window)

    def get_ball_visible(
        self,
        cam_idx: int,
        *,
        window: TemporalWindow | None = None,
    ) -> np.ndarray:
        prefix = self._camera_prefix(cam_idx)
        return self._copy_temporal_array(f"{prefix}ball_visible", window=window)

    def get_court_kp_uv(self, cam_idx: int) -> np.ndarray:
        prefix = self._camera_prefix(cam_idx)
        return self._copy_array(f"{prefix}court_kp_uv")

    def get_court_kp_visible(self, cam_idx: int) -> np.ndarray:
        prefix = self._camera_prefix(cam_idx)
        return self._copy_array(f"{prefix}court_kp_visible")

    def get_camera_view(
        self,
        cam_idx: int,
        *,
        window: TemporalWindow | None = None,
    ) -> CameraViewArrays:
        return CameraViewArrays(
            ball_uv=self.get_ball_uv(cam_idx, window=window),
            ball_visible=self.get_ball_visible(cam_idx, window=window),
            court_kp_uv=self.get_court_kp_uv(cam_idx),
            court_kp_visible=self.get_court_kp_visible(cam_idx),
        )

    def get_ball_pos_norm(self, *, window: TemporalWindow | None = None) -> np.ndarray:
        return self._copy_temporal_array("ball_pos_norm", window=window)

    def get_ball_pos_world(self, *, window: TemporalWindow | None = None) -> np.ndarray:
        return self._copy_temporal_array("ball_pos_world", window=window)

    def get_ball_vel_world(self, *, window: TemporalWindow | None = None) -> np.ndarray:
        return self._copy_temporal_array("ball_vel_world", window=window)


class NPZSceneDatasetBase(Dataset, Generic[SampleT]):
    """Base dataset for BLCS-style NPZ scenes."""

    def __init__(
        self,
        *,
        config: SceneDatasetConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.scene_dir = config.scene_dir

        self._validate_range(config.seq_len_range, name="seq_len_range")
        self._validate_range(config.num_views_range, name="num_views_range")

        all_paths = self._resolve_scene_files(self.scene_dir, config.split_file)
        if not all_paths:
            raise RuntimeError(f"No scenes found from split_file={config.split_file}")

        all_headers = self._index_scene_headers(all_paths)
        self.scene_headers = [h for h in all_headers if self._passes_filters(h)]
        self.scenes = [h.path for h in self.scene_headers]
        self._headers_by_path = {h.path: h for h in self.scene_headers}
        if not self.scenes:
            raise RuntimeError(
                "No scenes remain after filtering: "
                f"min_num_frames={self._required_min_frames()}, "
                f"min_num_cameras={self._required_min_cameras()}"
            )

        self._scene_cache: SceneCache | None = (
            get_scene_cache(load_fn=_load_npz_payload, maxsize=config.cache_max_scenes)
            if config.cache_max_scenes > 0
            else None
        )

    @staticmethod
    def _validate_range(value: tuple[int, int], *, name: str) -> None:
        if len(value) != 2:
            raise ValueError(f"{name} must have length 2, got {value}")
        lo, hi = int(value[0]), int(value[1])
        if lo <= 0 or hi <= 0:
            raise ValueError(f"{name} must contain positive integers, got {value}")
        if lo > hi:
            raise ValueError(f"{name} min must be <= max, got {value}")

    def _resolve_scene_files(self, scene_dir: Path, split_file: Path) -> list[Path]:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = scene_dir / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        scenes_base = scene_dir / "scenes"
        if not scenes_base.exists():
            scenes_base = scene_dir

        paths: list[Path] = []
        for line in split_path.read_text().splitlines():
            name = line.strip()
            if name:
                paths.append(scenes_base / name)
        return paths

    def _decode_meta(self, meta_raw: Any) -> dict[str, Any]:
        if hasattr(meta_raw, "item"):
            try:
                meta_raw = meta_raw.item()
            except ValueError:
                pass
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8")
        if isinstance(meta_raw, str):
            try:
                meta_raw = json.loads(meta_raw)
            except json.JSONDecodeError:
                return {}
        return dict(meta_raw) if isinstance(meta_raw, dict) else {}

    def _fallback_num_frames_from_payload(self, payload: dict[str, Any]) -> int:
        for key in ("ball_pos_norm", "ball_pos_world", "cam_0_ball_uv"):
            value = payload.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 1 and int(arr.shape[0]) > 0:
                return int(arr.shape[0])
        return 0

    def _resolve_num_frames(
        self,
        *,
        path: Path,
        meta: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        fallback = self._fallback_num_frames_from_payload(payload)
        meta_num_raw = meta.get("num_frames")
        meta_num: int | None = None
        if meta_num_raw is not None:
            try:
                meta_num = int(meta_num_raw)
            except (TypeError, ValueError):
                meta_num = None

        if meta_num is None or meta_num <= 0:
            warnings.warn(
                f"{path}: invalid meta['num_frames']={meta_num_raw!r}; using fallback={fallback}",
                stacklevel=2,
            )
            return fallback

        if fallback > 0 and meta_num > fallback:
            warnings.warn(
                f"{path}: meta['num_frames']={meta_num} exceeds available length {fallback}; "
                "using fallback",
                stacklevel=2,
            )
            return fallback

        return meta_num

    def _extract_scene_header(self, path: Path) -> NPZSceneHeader:
        with np.load(path, allow_pickle=True, mmap_mode="r") as npz:
            payload = {k: npz[k] for k in npz.files}
            meta = self._decode_meta(payload.get("meta", {}))
            num_frames = self._resolve_num_frames(path=path, meta=meta, payload=payload)
            try:
                num_cameras = int(np.asarray(payload["num_cameras"]).item())
            except Exception:
                num_cameras = 0
        return NPZSceneHeader(
            path=path,
            meta=dict(meta),
            num_frames=max(0, int(num_frames)),
            num_cameras=max(0, int(num_cameras)),
        )

    def _index_scene_headers(self, paths: list[Path]) -> list[NPZSceneHeader]:
        return [self._extract_scene_header(path) for path in paths]

    def _required_min_frames(self) -> int:
        return max(int(self.config.min_num_frames), int(self.config.seq_len_range[0]))

    def _required_min_cameras(self) -> int:
        return max(int(self.config.min_num_cameras), int(self.config.num_views_range[0]))

    def _passes_filters(self, header: NPZSceneHeader) -> bool:
        return (
            int(header.num_frames) >= self._required_min_frames()
            and int(header.num_cameras) >= self._required_min_cameras()
        )

    def __len__(self) -> int:
        return len(self.scenes)

    def get_scene_header(self, path: Path) -> NPZSceneHeader:
        try:
            return self._headers_by_path[path]
        except KeyError as exc:
            raise KeyError(f"Scene header not found for path: {path}") from exc

    def _load_scene(self, path: Path) -> NPZScene:
        header = self.get_scene_header(path)
        payload = self._scene_cache.get(path) if self._scene_cache is not None else _load_npz_payload(path)
        return NPZScene(
            path=path,
            data=payload,
            meta=dict(header.meta),
            num_frames=int(header.num_frames),
            num_cameras=int(header.num_cameras),
        )

    def select_camera(self, scene: NPZScene) -> int:
        return self.select_cameras(scene, num_views_range=(1, 1)).primary

    def select_cameras(
        self,
        scene: NPZScene,
        *,
        num_views_range: tuple[int, int] | None = None,
        camera_mode: str | int | None = None,
    ) -> CameraSelection:
        view_range = num_views_range or self.config.num_views_range
        self._validate_range(view_range, name="num_views_range")
        min_views, max_views = int(view_range[0]), int(view_range[1])

        if scene.num_cameras < min_views:
            raise ValueError(
                f"Scene {scene.path} has {scene.num_cameras} cameras, but min_views={min_views}"
            )

        nmax = min(max_views, scene.num_cameras)
        n = int(self.rng.integers(min_views, nmax + 1))
        mode = self.config.camera_mode if camera_mode is None else camera_mode

        if mode == "random":
            selected = self.rng.choice(scene.num_cameras, size=n, replace=False)
            return CameraSelection(indices=tuple(int(i) for i in selected.tolist()))

        if mode == "first":
            return CameraSelection(indices=tuple(range(n)))

        primary = 0
        if isinstance(mode, int):
            primary = int(mode)
        elif isinstance(mode, str) and mode.isdigit():
            primary = int(mode)
        primary = min(max(primary, 0), scene.num_cameras - 1)

        if n == 1:
            return CameraSelection(indices=(primary,))

        remaining = [i for i in range(scene.num_cameras) if i != primary]
        sampled = self.rng.choice(np.asarray(remaining), size=n - 1, replace=False)
        indices = (primary, *[int(i) for i in sampled.tolist()])
        return CameraSelection(indices=indices)

    def select_window(
        self,
        scene: NPZScene,
        *,
        full_len: int | None = None,
        seq_len_range: tuple[int, int] | None = None,
        crop_mode: Literal["random", "center"] | None = None,
    ) -> TemporalWindow:
        seq_range = seq_len_range or self.config.seq_len_range
        self._validate_range(seq_range, name="seq_len_range")
        min_seq, max_seq = int(seq_range[0]), int(seq_range[1])
        mode = self.config.crop_mode if crop_mode is None else crop_mode
        if mode not in ("random", "center"):
            raise ValueError(f"Unsupported crop_mode={mode}")

        full = int(scene.num_frames if full_len is None else full_len)
        if full < min_seq:
            raise ValueError(
                f"Scene too short for seq_len_range {seq_range}: full_len={full}, path={scene.path}"
            )

        actual_max = min(max_seq, full)
        seq_len = int(self.rng.integers(min_seq, actual_max + 1))

        if seq_len >= full:
            start = 0
        else:
            max_start = full - seq_len
            if mode == "random":
                start = int(self.rng.integers(0, max_start + 1))
            else:
                start = max_start // 2
        end = start + seq_len
        return TemporalWindow(start=start, end=end, seq_len=seq_len, full_len=full)

    def build_sample(self, scene: NPZScene) -> SampleT:
        """Build a task-specific sample from a loaded NPZScene."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SampleT:
        scene = self._load_scene(self.scenes[idx])
        return self.build_sample(scene)
