"""Immutable BLCS real-capture calibration artifacts for NHT feature fitting."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Self, cast
from urllib.parse import unquote, urlparse

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.scene_contract import ArtifactRef

BALL_CALIBRATION_BUNDLE_SCHEMA = "tennis_ball_nht_calibration_bundle_v1"
BALL_CALIBRATION_CAPTURE_SCHEMA = "tennis_ball_calibration_capture_v1"
BALL_CALIBRATION_IMPORT_SCHEMA = "tennis_ball_calibration_import_v1"

_CALIBRATION_ARRAY_KEYS = {
    "camera_to_asset",
    "intrinsics",
    "rgb",
    "mask",
    "split",
}
_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")
_ID_INITIAL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_MATRIX_ATOL = 1.0e-5


@dataclass(frozen=True)
class BallCalibrationBundle:
    """Verified calibration arrays in asset-local OpenCV coordinates."""

    root: Path
    manifest: dict[str, object]
    camera_to_asset: NDArray[np.float32]
    intrinsics: NDArray[np.float32]
    rgb: NDArray[np.uint8]
    mask: NDArray[np.bool_]
    split: NDArray[np.uint8]

    @property
    def width(self) -> int:
        return int(self.rgb.shape[2])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def train_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.split == 0))

    @property
    def validation_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.split == 1))


@dataclass(frozen=True)
class BallCalibrationCaptureView:
    """One verified RGB/mask pair with an explicit camera and split."""

    view_id: str
    split: str
    width: int
    height: int
    rgb: ArtifactRef
    mask: ArtifactRef
    camera_to_asset: tuple[float, ...]
    intrinsics: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_id(self.view_id, name="view_id")
        if self.split not in {"train", "validation"}:
            raise ValueError("capture view split must be 'train' or 'validation'.")
        if isinstance(self.width, bool) or self.width <= 1:
            raise ValueError("capture view width must be greater than one.")
        if isinstance(self.height, bool) or self.height <= 1:
            raise ValueError("capture view height must be greater than one.")
        camera = _matrix(
            self.camera_to_asset,
            rows=4,
            columns=4,
            name="camera_to_asset",
        )
        intrinsic = _matrix(
            self.intrinsics,
            rows=3,
            columns=3,
            name="intrinsics",
        )
        _validate_camera_to_asset(camera)
        _validate_intrinsics(
            intrinsic,
            width=self.width,
            height=self.height,
        )
        object.__setattr__(
            self,
            "camera_to_asset",
            tuple(float(value) for value in camera.ravel()),
        )
        object.__setattr__(
            self,
            "intrinsics",
            tuple(float(value) for value in intrinsic.ravel()),
        )

    @classmethod
    def from_dict(cls, value: object, *, root: Path) -> Self:
        """Parse one capture view and resolve its hash-checked local files."""
        raw = _strict_mapping(
            value,
            name="ball calibration capture view",
            keys={
                "view_id",
                "split",
                "width",
                "height",
                "rgb",
                "mask",
                "camera_to_asset",
                "intrinsics",
            },
        )
        width = _positive_int(raw["width"], name="width")
        height = _positive_int(raw["height"], name="height")
        rgb = _capture_artifact(
            raw["rgb"],
            root=root,
            artifact_id=f"{_string(raw['view_id'], name='view_id')}-rgb",
        )
        mask = _capture_artifact(
            raw["mask"],
            root=root,
            artifact_id=f"{_string(raw['view_id'], name='view_id')}-mask",
        )
        _load_rgb(rgb, width=width, height=height)
        _load_mask(mask, width=width, height=height)
        return cls(
            view_id=_string(raw["view_id"], name="view_id"),
            split=_string(raw["split"], name="split"),
            width=width,
            height=height,
            rgb=rgb,
            mask=mask,
            camera_to_asset=_number_tuple(
                raw["camera_to_asset"],
                length=16,
                name="camera_to_asset",
            ),
            intrinsics=_number_tuple(
                raw["intrinsics"],
                length=9,
                name="intrinsics",
            ),
        )


@dataclass(frozen=True)
class BallCalibrationCapture:
    """One strictly verified real-capture manifest."""

    root: Path
    manifest_path: Path
    capture_id: str
    views: tuple[BallCalibrationCaptureView, ...]

    @property
    def width(self) -> int:
        return self.views[0].width

    @property
    def height(self) -> int:
        return self.views[0].height


@dataclass(frozen=True)
class BallCalibrationImport:
    """Verified import provenance plus the published tensor bundle."""

    root: Path
    manifest: dict[str, object]
    capture: BallCalibrationCapture
    bundle: BallCalibrationBundle


def load_ball_calibration_bundle(path: str | Path) -> BallCalibrationBundle:
    """Load and hash-verify one exact calibration tensor bundle."""
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Calibration manifest is missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = _strict_mapping(
            json.load(handle),
            name="calibration manifest",
            keys={
                "schema",
                "bundle_id",
                "width",
                "height",
                "view_count",
                "train_views",
                "validation_views",
                "tensors",
                "content_fingerprint",
            },
        )
    if manifest["schema"] != BALL_CALIBRATION_BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported calibration schema: {manifest['schema']!r}.")
    _validate_id(_string(manifest["bundle_id"], name="bundle_id"), name="bundle_id")
    expected_fingerprint = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "content_fingerprint"}
    )
    if manifest["content_fingerprint"] != expected_fingerprint:
        raise ValueError("Calibration content fingerprint is invalid.")
    tensor_ref = _strict_mapping(
        manifest["tensors"],
        name="calibration tensor reference",
        keys={"relative_path", "sha256", "size_bytes"},
    )
    tensor_path = _resolve_relative_file(root, tensor_ref["relative_path"])
    _verify_relative_file(tensor_path, tensor_ref, name="calibration tensors")
    with np.load(tensor_path, allow_pickle=False) as arrays:
        if set(arrays.files) != _CALIBRATION_ARRAY_KEYS:
            raise ValueError(f"Calibration tensor keys differ: {sorted(arrays.files)}.")
        camera_to_asset = arrays["camera_to_asset"].copy()
        intrinsics = arrays["intrinsics"].copy()
        rgb = arrays["rgb"].copy()
        mask = arrays["mask"].copy()
        split = arrays["split"].copy()
    view_count = _positive_int(manifest["view_count"], name="view_count")
    width = _positive_int(manifest["width"], name="width")
    height = _positive_int(manifest["height"], name="height")
    _validate_calibration_arrays(
        camera_to_asset=camera_to_asset,
        intrinsics=intrinsics,
        rgb=rgb,
        mask=mask,
        split=split,
        view_count=view_count,
        width=width,
        height=height,
    )
    train_views = int((split == 0).sum())
    validation_views = int((split == 1).sum())
    if train_views != _positive_int(manifest["train_views"], name="train_views"):
        raise ValueError("Calibration train_views differs from split.")
    if validation_views != _positive_int(
        manifest["validation_views"],
        name="validation_views",
    ):
        raise ValueError("Calibration validation_views differs from split.")
    return BallCalibrationBundle(
        root=root,
        manifest=manifest,
        camera_to_asset=camera_to_asset,
        intrinsics=intrinsics,
        rgb=rgb,
        mask=mask,
        split=split,
    )


def write_ball_calibration_bundle(
    output_dir: str | Path,
    *,
    bundle_id: str,
    camera_to_asset: NDArray[np.float32],
    intrinsics: NDArray[np.float32],
    rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    split: NDArray[np.uint8],
) -> Path:
    """Atomically publish one validated calibration tensor bundle."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite calibration bundle: {destination}"
        )
    _validate_id(bundle_id, name="bundle_id")
    if rgb.ndim != 4:
        raise ValueError("rgb must have shape [V,H,W,3].")
    view_count, height, width = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
    _validate_calibration_arrays(
        camera_to_asset=camera_to_asset,
        intrinsics=intrinsics,
        rgb=rgb,
        mask=mask,
        split=split,
        view_count=view_count,
        width=width,
        height=height,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        tensors = temporary / "calibration.npz"
        np.savez(
            tensors,
            camera_to_asset=camera_to_asset,
            intrinsics=intrinsics,
            rgb=rgb,
            mask=mask,
            split=split,
        )
        unsigned = {
            "schema": BALL_CALIBRATION_BUNDLE_SCHEMA,
            "bundle_id": bundle_id,
            "width": width,
            "height": height,
            "view_count": view_count,
            "train_views": int((split == 0).sum()),
            "validation_views": int((split == 1).sum()),
            "tensors": _relative_file_ref(temporary, tensors),
        }
        manifest = {
            **unsigned,
            "content_fingerprint": _canonical_sha256(unsigned),
        }
        _write_json(temporary / "manifest.json", manifest)
        load_ball_calibration_bundle(temporary)
        temporary.rename(destination)
        return destination / "manifest.json"
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_ball_calibration_capture(path: str | Path) -> BallCalibrationCapture:
    """Load a capture manifest and verify every RGB/mask byte before use."""
    candidate = Path(path)
    manifest_path = candidate / "capture.json" if candidate.is_dir() else candidate
    manifest_path = manifest_path.resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        raw = _strict_mapping(
            json.load(handle),
            name="ball calibration capture",
            keys={"schema", "capture_id", "views"},
        )
    if raw["schema"] != BALL_CALIBRATION_CAPTURE_SCHEMA:
        raise ValueError(f"Unsupported capture schema: {raw['schema']!r}.")
    capture_id = _string(raw["capture_id"], name="capture_id")
    _validate_id(capture_id, name="capture_id")
    values = raw["views"]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("capture views must be a JSON array.")
    views = tuple(
        sorted(
            (
                BallCalibrationCaptureView.from_dict(
                    value,
                    root=manifest_path.parent,
                )
                for value in values
            ),
            key=lambda view: view.view_id,
        )
    )
    if len(views) < 3:
        raise ValueError("capture must contain at least three views.")
    _require_unique([view.view_id for view in views], name="capture view ids")
    _require_unique([view.rgb.sha256 for view in views], name="capture RGB contents")
    _require_unique([view.rgb.uri for view in views], name="capture RGB paths")
    _require_unique([view.mask.uri for view in views], name="capture mask paths")
    dimensions = {(view.width, view.height) for view in views}
    if len(dimensions) != 1:
        raise ValueError("capture views must share one width and height.")
    train_count = sum(view.split == "train" for view in views)
    validation_count = sum(view.split == "validation" for view in views)
    if train_count < 2:
        raise ValueError("capture import requires at least two training views.")
    if validation_count < 1:
        raise ValueError("capture import requires at least one validation view.")
    return BallCalibrationCapture(
        root=manifest_path.parent,
        manifest_path=manifest_path,
        capture_id=capture_id,
        views=views,
    )


def import_ball_calibration_capture(
    capture_manifest: str | Path,
    output_dir: str | Path,
    *,
    bundle_id: str,
) -> Path:
    """Atomically import individual capture records into a tensor bundle."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite calibration import: {destination}"
        )
    _validate_id(bundle_id, name="bundle_id")
    capture = load_ball_calibration_capture(capture_manifest)
    rgb_values: list[NDArray[np.uint8]] = []
    mask_values: list[NDArray[np.bool_]] = []
    cameras: list[NDArray[np.float32]] = []
    intrinsics: list[NDArray[np.float32]] = []
    split_values: list[int] = []
    for view in capture.views:
        rgb_values.append(_load_rgb(view.rgb, width=view.width, height=view.height))
        mask_values.append(_load_mask(view.mask, width=view.width, height=view.height))
        cameras.append(np.asarray(view.camera_to_asset, dtype=np.float32).reshape(4, 4))
        intrinsics.append(np.asarray(view.intrinsics, dtype=np.float32).reshape(3, 3))
        split_values.append(0 if view.split == "train" else 1)
    camera_array = np.stack(cameras).astype(np.float32, copy=False)
    intrinsic_array = np.stack(intrinsics).astype(np.float32, copy=False)
    rgb_array = np.stack(rgb_values).astype(np.uint8, copy=False)
    mask_array = np.stack(mask_values).astype(np.bool_, copy=False)
    split_array = np.asarray(split_values, dtype=np.uint8)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        bundle_manifest = write_ball_calibration_bundle(
            temporary / "bundle",
            bundle_id=bundle_id,
            camera_to_asset=camera_array,
            intrinsics=intrinsic_array,
            rgb=rgb_array,
            mask=mask_array,
            split=split_array,
        )
        capture_ref = _absolute_artifact(
            capture.manifest_path,
            artifact_id="source-capture-manifest",
        )
        unsigned: dict[str, object] = {
            "schema": BALL_CALIBRATION_IMPORT_SCHEMA,
            "status": "passed",
            "capture_id": capture.capture_id,
            "capture_manifest": capture_ref.to_dict(),
            "ordered_view_ids": [view.view_id for view in capture.views],
            "view_count": len(capture.views),
            "train_views": int((split_array == 0).sum()),
            "validation_views": int((split_array == 1).sum()),
            "bundle_manifest": _relative_file_ref(temporary, bundle_manifest),
            "importer": {
                "module": "src.synthetic_data_generation.dataset.blcs.artifacts.calibration",
                "sha256": _sha256_file(Path(__file__).resolve()),
                "numpy_version": np.__version__,
            },
        }
        manifest = {
            **unsigned,
            "content_fingerprint": _canonical_sha256(unsigned),
        }
        _write_json(temporary / "capture-import.json", manifest)
        load_ball_calibration_import(temporary)
        temporary.rename(destination)
        return destination / "capture-import.json"
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_ball_calibration_import(path: str | Path) -> BallCalibrationImport:
    """Load import provenance, source capture, and the exact tensor bundle."""
    candidate = Path(path)
    manifest_path = (
        candidate / "capture-import.json" if candidate.is_dir() else candidate
    ).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = _strict_mapping(
            json.load(handle),
            name="ball calibration import",
            keys={
                "schema",
                "status",
                "capture_id",
                "capture_manifest",
                "ordered_view_ids",
                "view_count",
                "train_views",
                "validation_views",
                "bundle_manifest",
                "importer",
                "content_fingerprint",
            },
        )
    if manifest["schema"] != BALL_CALIBRATION_IMPORT_SCHEMA:
        raise ValueError(f"Unsupported calibration import: {manifest['schema']!r}.")
    if manifest["status"] != "passed":
        raise ValueError("Calibration import status must be 'passed'.")
    expected_fingerprint = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "content_fingerprint"}
    )
    if manifest["content_fingerprint"] != expected_fingerprint:
        raise ValueError("Calibration import content fingerprint is invalid.")
    root = manifest_path.parent
    capture_artifact = ArtifactRef.from_dict(manifest["capture_manifest"])
    capture_path = _verify_absolute_artifact(capture_artifact)
    capture = load_ball_calibration_capture(capture_path)
    if capture.capture_id != manifest["capture_id"]:
        raise ValueError("Calibration import capture_id differs from source capture.")
    ordered_view_ids = manifest["ordered_view_ids"]
    if not isinstance(ordered_view_ids, Sequence) or isinstance(
        ordered_view_ids,
        (str, bytes),
    ):
        raise TypeError("ordered_view_ids must be a JSON array.")
    if list(ordered_view_ids) != [view.view_id for view in capture.views]:
        raise ValueError("Calibration import ordered_view_ids differ from capture.")
    bundle_ref = _strict_mapping(
        manifest["bundle_manifest"],
        name="imported calibration bundle reference",
        keys={"relative_path", "sha256", "size_bytes"},
    )
    bundle_manifest_path = _resolve_relative_file(root, bundle_ref["relative_path"])
    _verify_relative_file(
        bundle_manifest_path,
        bundle_ref,
        name="imported calibration bundle manifest",
    )
    bundle = load_ball_calibration_bundle(bundle_manifest_path.parent)
    counts = {
        "view_count": len(bundle.train_indices) + len(bundle.validation_indices),
        "train_views": len(bundle.train_indices),
        "validation_views": len(bundle.validation_indices),
    }
    for name, expected in counts.items():
        if _positive_int(manifest[name], name=name) != expected:
            raise ValueError(f"Calibration import {name} differs from bundle.")
    importer = _strict_mapping(
        manifest["importer"],
        name="calibration importer identity",
        keys={"module", "sha256", "numpy_version"},
    )
    _string(importer["module"], name="importer module")
    _sha256(_string(importer["sha256"], name="importer sha256"), name="importer sha256")
    _string(importer["numpy_version"], name="importer numpy_version")
    return BallCalibrationImport(
        root=root,
        manifest=manifest,
        capture=capture,
        bundle=bundle,
    )


def _validate_calibration_arrays(
    *,
    camera_to_asset: NDArray[np.generic],
    intrinsics: NDArray[np.generic],
    rgb: NDArray[np.generic],
    mask: NDArray[np.generic],
    split: NDArray[np.generic],
    view_count: int,
    width: int,
    height: int,
) -> None:
    if camera_to_asset.dtype != np.float32 or camera_to_asset.shape != (
        view_count,
        4,
        4,
    ):
        raise ValueError("camera_to_asset must have float32 shape [V,4,4].")
    if intrinsics.dtype != np.float32 or intrinsics.shape != (view_count, 3, 3):
        raise ValueError("intrinsics must have float32 shape [V,3,3].")
    if rgb.dtype != np.uint8 or rgb.shape != (view_count, height, width, 3):
        raise ValueError("rgb must have uint8 shape [V,H,W,3].")
    if mask.dtype != np.bool_ or mask.shape != (view_count, height, width):
        raise ValueError("mask must have bool shape [V,H,W].")
    if split.dtype != np.uint8 or split.shape != (view_count,):
        raise ValueError("split must have uint8 shape [V].")
    if not np.isin(split, np.asarray([0, 1], dtype=np.uint8)).all():
        raise ValueError("Calibration split values must be exactly 0 or 1.")
    if int((split == 0).sum()) <= 0 or int((split == 1).sum()) <= 0:
        raise ValueError("Calibration must contain train and validation views.")
    for index in range(view_count):
        _validate_camera_to_asset(camera_to_asset[index].astype(np.float64))
        _validate_intrinsics(
            intrinsics[index].astype(np.float64),
            width=width,
            height=height,
        )
    flattened_mask = mask.reshape(view_count, -1)
    if not flattened_mask.any(axis=1).all():
        raise ValueError("Every calibration view must contain foreground mask pixels.")
    if flattened_mask.all(axis=1).any():
        raise ValueError("Calibration foreground masks must not fill an entire view.")


def _validate_camera_to_asset(matrix: NDArray[np.float64]) -> None:
    if not np.isfinite(matrix).all():
        raise ValueError("camera_to_asset contains non-finite values.")
    if not np.allclose(
        matrix[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=_MATRIX_ATOL,
        rtol=0.0,
    ):
        raise ValueError("camera_to_asset must have bottom row [0,0,0,1].")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=_MATRIX_ATOL,
        rtol=0.0,
    ) or not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        abs_tol=_MATRIX_ATOL,
        rel_tol=0.0,
    ):
        raise ValueError("camera_to_asset rotation must be a proper rotation.")


def _validate_intrinsics(
    matrix: NDArray[np.float64],
    *,
    width: int,
    height: int,
) -> None:
    if not np.isfinite(matrix).all():
        raise ValueError("intrinsics contains non-finite values.")
    if not np.allclose(
        matrix[2],
        np.asarray([0.0, 0.0, 1.0]),
        atol=_MATRIX_ATOL,
        rtol=0.0,
    ):
        raise ValueError("intrinsics must have bottom row [0,0,1].")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("Calibration focal lengths must be positive.")
    if not 0.0 <= matrix[0, 2] < width or not 0.0 <= matrix[1, 2] < height:
        raise ValueError("Calibration principal point must lie inside the image.")


def _load_rgb(
    artifact: ArtifactRef,
    *,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    path = _verify_absolute_artifact(artifact)
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Calibration RGB must use mode RGB: {path}")
        if image.size != (width, height):
            raise ValueError(f"Calibration RGB dimensions differ: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


def _load_mask(
    artifact: ArtifactRef,
    *,
    width: int,
    height: int,
) -> NDArray[np.bool_]:
    path = _verify_absolute_artifact(artifact)
    with Image.open(path) as image:
        image.load()
        if image.mode != "L":
            raise ValueError(f"Calibration mask must use mode L: {path}")
        if image.size != (width, height):
            raise ValueError(f"Calibration mask dimensions differ: {path}")
        values = np.asarray(image, dtype=np.uint8).copy()
    if not np.isin(values, np.asarray([0, 255], dtype=np.uint8)).all():
        raise ValueError(f"Calibration mask must contain only 0 and 255: {path}")
    mask = cast(NDArray[np.bool_], values == 255)
    if not mask.any():
        raise ValueError(f"Calibration mask contains no foreground: {path}")
    if mask.all():
        raise ValueError(f"Calibration mask fills the whole image: {path}")
    return mask


def _capture_artifact(
    value: object,
    *,
    root: Path,
    artifact_id: str,
) -> ArtifactRef:
    raw = _strict_mapping(
        value,
        name="capture file reference",
        keys={"relative_path", "sha256", "size_bytes"},
    )
    path = _resolve_relative_file(root, raw["relative_path"])
    _verify_relative_file(path, raw, name="capture file")
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=path.resolve().as_uri(),
        sha256=_sha256(_string(raw["sha256"], name="sha256"), name="sha256"),
        size_bytes=_positive_int(raw["size_bytes"], name="size_bytes"),
    )


def _resolve_relative_file(root: Path, value: object) -> Path:
    relative = _string(value, name="relative_path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError("relative_path is unsafe.")
    path = root.joinpath(*pure.parts)
    if not path.is_file():
        raise FileNotFoundError(f"Referenced file is missing: {path}")
    return path


def _verify_relative_file(
    path: Path,
    value: Mapping[str, object],
    *,
    name: str,
) -> None:
    expected_size = _positive_int(value["size_bytes"], name=f"{name} size_bytes")
    expected_sha256 = _sha256(
        _string(value["sha256"], name=f"{name} sha256"),
        name=f"{name} sha256",
    )
    if path.stat().st_size != expected_size:
        raise ValueError(f"{name} size differs: {path}")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} SHA-256 differs: {path}")


def _verify_absolute_artifact(artifact: ArtifactRef) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"Artifact is not a local file URI: {artifact.uri!r}")
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise FileNotFoundError(f"Artifact is missing: {path}")
    if path.stat().st_size != artifact.size_bytes:
        raise ValueError(f"Artifact size differs: {path}")
    if _sha256_file(path) != artifact.sha256:
        raise ValueError(f"Artifact SHA-256 differs: {path}")
    return path


def _absolute_artifact(path: Path, *, artifact_id: str) -> ArtifactRef:
    resolved = path.resolve()
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=resolved.as_uri(),
        sha256=_sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _relative_file_ref(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _matrix(
    value: Sequence[float],
    *,
    rows: int,
    columns: int,
    name: str,
) -> NDArray[np.float64]:
    values = _number_tuple(value, length=rows * columns, name=name)
    return np.asarray(values, dtype=np.float64).reshape(rows, columns)


def _number_tuple(value: object, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    if len(value) != length:
        raise ValueError(f"{name} must contain {length} values.")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} values must be numbers.")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} values must be finite.")
        result.append(numeric)
    return tuple(result)


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    raw = {str(key): item for key, item in value.items()}
    missing = keys.difference(raw)
    extra = set(raw).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} keys differ; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return raw


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _validate_id(value: str, *, name: str) -> None:
    if (
        not value
        or value[0] not in _ID_INITIAL_CHARS
        or any(character not in _ID_CHARS for character in value)
    ):
        raise ValueError(
            f"{name} must start with lowercase alphanumeric and contain only "
            "lowercase alphanumeric, '.', '_', or '-'."
        )


def _require_unique(values: Sequence[str], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique.")
