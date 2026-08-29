"""Durable court-line inference cache independent of alignment fitting."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.alignment.settings import CourtLineModelSettings
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import SceneCamera

_CACHE_SCHEMA = "court_line_inference_cache_v1"
_MIRROR_ROOT_ENV = "TENNIS_LAB_ALIGNMENT_INFERENCE_MIRROR_ROOT"


def court_line_inference_identity(
    settings: CourtLineModelSettings, *, seed: int
) -> dict[str, object]:
    """Fingerprint only inputs that can change raw model probabilities."""
    return {
        "schema": "court_line_inference_identity_v1",
        "checkpoint_sha256": _sha256_file(settings.checkpoint_path),
        "backbone_checkpoint_sha256": _sha256_file(
            settings.backbone_checkpoint_path
        ),
        "architecture": asdict(settings.architecture),
        "expected_short_side": settings.expected_short_side,
        "device": settings.device,
        "seed": seed,
    }


def load_or_predict_line_probabilities(
    *,
    scene: StandardSceneExport,
    cameras: tuple[SceneCamera, ...],
    inference_identity: Mapping[str, object],
    predict_probability: Callable[[NDArray[np.uint8]], NDArray[np.float32]],
    load_image: Callable[[SceneCamera], NDArray[np.uint8]],
) -> dict[str, NDArray[np.float32]]:
    """Reuse exact raw model outputs or persist each newly inferred view."""
    identity = _json_mapping(inference_identity, name="inference_identity")
    fingerprint = hashlib.sha256(_canonical_json(identity)).hexdigest()
    cache_root = _cache_base(scene) / fingerprint
    views_root = _ordinary_directory(cache_root / "views")
    mirror_root = _mirror_root(fingerprint)
    if mirror_root is not None:
        _ordinary_directory(mirror_root / "views")

    probabilities: dict[str, NDArray[np.float32]] = {}
    entries: list[dict[str, object]] = []
    for index, camera in enumerate(cameras):
        image_sha256 = _sha256_file(Path(camera.image_path))
        stem = f"view-{index:03d}-{image_sha256[:16]}"
        array_path = views_root / f"{stem}.npy"
        preview_path = views_root / f"{stem}.png"
        if array_path.exists() or array_path.is_symlink():
            probability = _load_probability(array_path)
        else:
            probability = _validated_probability(
                predict_probability(load_image(camera))
            )
            _atomic_save_array(array_path, probability)
        expected_preview = _render_probability(probability)
        if preview_path.exists() or preview_path.is_symlink():
            _validate_preview(preview_path, expected_preview)
        else:
            _atomic_save_png(preview_path, expected_preview)
        probabilities[camera.camera_id] = probability
        entries.append(
            {
                "index": index,
                "camera_id": camera.camera_id,
                "source_frame_index": camera.source_frame_index,
                "image_sha256": image_sha256,
                "probability_shape": list(probability.shape),
                "probability_file": f"views/{array_path.name}",
                "preview_file": f"views/{preview_path.name}",
            }
        )
        manifest = {
            "schema": _CACHE_SCHEMA,
            "scene_id": scene.scene_id,
            "fingerprint": fingerprint,
            "inference_identity": identity,
            "expected_view_count": len(cameras),
            "completed_view_count": len(entries),
            "views": entries,
        }
        manifest_path = cache_root / "manifest.json"
        _atomic_write_json(manifest_path, manifest)
        if mirror_root is not None:
            _mirror_file(array_path, mirror_root / "views" / array_path.name)
            _mirror_file(preview_path, mirror_root / "views" / preview_path.name)
            _mirror_file(manifest_path, mirror_root / "manifest.json")
    return probabilities


def _cache_base(scene: StandardSceneExport) -> Path:
    export_root = scene.export_root
    if export_root.name != "export" or export_root.parent.name != "reconstruction":
        raise ValueError("Court-line cache requires the canonical reconstruction export.")
    return _ordinary_directory(export_root.parent.parent / "court-line-inference")


def _mirror_root(fingerprint: str) -> Path | None:
    raw = os.environ.get(_MIRROR_ROOT_ENV)
    if raw is None:
        return None
    base = Path(raw)
    if not base.is_absolute():
        raise ValueError(f"{_MIRROR_ROOT_ENV} must be an absolute path.")
    return _ordinary_directory(base / fingerprint)


def _ordinary_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"Court-line cache directory cannot be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Court-line cache path is not a directory: {path}")
    return path


def _validated_probability(value: object) -> NDArray[np.float32]:
    probability = np.asarray(value)
    if probability.dtype != np.float32 or probability.ndim != 2:
        raise ValueError("Cached court-line probability must be a float32 2-D array.")
    if min(probability.shape) < 2 or not np.isfinite(probability).all():
        raise ValueError("Cached court-line probability is invalid or non-finite.")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("Cached court-line probability must lie in [0, 1].")
    probability = np.array(probability, dtype=np.float32, copy=True)
    probability.setflags(write=False)
    return cast(NDArray[np.float32], probability)


def _load_probability(path: Path) -> NDArray[np.float32]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Court-line cache entry must be an ordinary file: {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Court-line cache entry is unreadable: {path}") from error
    return _validated_probability(loaded)


def _atomic_save_array(path: Path, probability: NDArray[np.float32]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".uploading", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.save(output, probability, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_probability(probability: NDArray[np.float32]) -> NDArray[np.uint8]:
    intensity = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    colored = np.asarray(
        cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
    )
    colored[probability <= 0.0] = 0
    return cast(NDArray[np.uint8], colored)


def _atomic_save_png(path: Path, image_rgb: NDArray[np.uint8]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".uploading", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(image_rgb, mode="RGB").save(
            temporary, format="PNG", compress_level=6
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_preview(path: Path, expected: NDArray[np.uint8]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Court-line preview must be an ordinary file: {path}")
    with Image.open(path) as image:
        actual = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(actual, expected):
        raise ValueError(f"Court-line preview disagrees with cached values: {path}")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".uploading", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mirror_file(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".uploading", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Court-line inference input must be an ordinary file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_mapping(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    try:
        encoded = _canonical_json(value)
        decoded: Any = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-serializable.") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must be a mapping.")
    return cast(dict[str, object], decoded)


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


__all__ = [
    "court_line_inference_identity",
    "load_or_predict_line_probabilities",
]
