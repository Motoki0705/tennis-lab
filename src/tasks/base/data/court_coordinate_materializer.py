"""Non-overwriting BLCS/PLCS dataset normalization materialization.

The materializer copies a complete scene-directory artifact into a new,
version-qualified root, changes only the task's normalized position array, and
adds canonical root/scene metadata. Root-level dynamic ``chunks/`` caches are
excluded because they are separate generated datasets whose normalized arrays
would otherwise retain the source contract. Source contracts are explicit and
are validated before values are interpreted; no shape/range-based version
inference or in-place migration is supported.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np

from src.tasks.base.configuration import (
    CourtCoordinateNormalizationConfig,
    as_config_mapping,
    exact_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.base.data.court_coordinate_contract import (
    COURT_COORDINATE_NORMALIZATION_METADATA_KEY,
    CourtCoordinateContractMismatchError,
    CourtCoordinateNormalizationMetadata,
    MissingCourtCoordinateMetadataError,
    extract_court_coordinate_normalization_metadata,
    validate_dataset_court_coordinate_contract,
)
from src.utils.io import load_json, save_json_atomic
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)

CourtCoordinateDatasetKind: TypeAlias = Literal["blcs", "plcs"]

__all__ = [
    "CourtCoordinateDatasetKind",
    "CourtCoordinateMaterializationConfig",
    "CourtCoordinateMaterializationError",
    "CourtCoordinateMaterializationResult",
    "materialize_court_coordinate_normalization_dataset",
]

_MATERIALIZATION_KEYS = frozenset(
    {
        "dataset_kind",
        "source_dir",
        "output_dir",
        "source_normalization_version",
        "max_abs_round_trip_error_m",
    }
)
_MANIFEST_FILENAME = "normalization_materialization.json"
_ROOT_CACHE_DIRECTORY_NAMES = frozenset({"chunks"})


class CourtCoordinateMaterializationError(ValueError):
    """Raised before publishing an unsafe or incompatible materialization."""


@dataclass(frozen=True, slots=True)
class _DatasetAdapter:
    dataset_kind: CourtCoordinateDatasetKind
    normalized_filename: str
    physical_filename: str | None

    def physical_position(
        self,
        scene_dir: Path,
        source_contract: CourtCoordinateNormalization,
    ) -> tuple[np.ndarray, Path]:
        normalized_path = scene_dir / self.normalized_filename
        if self.physical_filename is not None:
            physical_path = scene_dir / self.physical_filename
            physical = _load_position_array(physical_path)
            source_normalized = _load_position_array(normalized_path)
            expected = source_contract.normalize_position(physical)
            _assert_compatible_values(
                source_normalized,
                expected,
                location=normalized_path,
                description="declared source normalization",
            )
            return physical, physical_path

        source_normalized = _load_position_array(normalized_path)
        physical = source_contract.denormalize_position(source_normalized)
        return physical, normalized_path


_ADAPTERS: dict[CourtCoordinateDatasetKind, _DatasetAdapter] = {
    "blcs": _DatasetAdapter(
        dataset_kind="blcs",
        normalized_filename="ball_pos_norm.npy",
        physical_filename="ball_pos_world.npy",
    ),
    "plcs": _DatasetAdapter(
        dataset_kind="plcs",
        normalized_filename="position.npy",
        physical_filename=None,
    ),
}


@dataclass(frozen=True, slots=True)
class CourtCoordinateMaterializationConfig:
    """Strict input/output and source/target contracts for one materialization."""

    dataset_kind: CourtCoordinateDatasetKind
    source_dir: Path
    output_dir: Path
    source_contract: CourtCoordinateNormalization
    target_contract: CourtCoordinateNormalization
    max_abs_round_trip_error_m: float

    @classmethod
    def from_config(cls, value: object) -> CourtCoordinateMaterializationConfig:
        """Parse the materializer's composed Hydra root without fallbacks."""
        root = as_config_mapping(value, path="configuration")
        target = CourtCoordinateNormalizationConfig.from_config(value).contract
        mapping = exact_config_mapping(
            require_config_mapping(root, "materialization", path="configuration"),
            path="materialization",
            required_keys=_MATERIALIZATION_KEYS,
        )
        raw_kind = cast(
            "str",
            require_config_value(mapping, "dataset_kind", str, path="materialization"),
        )
        if raw_kind not in _ADAPTERS:
            raise CourtCoordinateMaterializationError(
                "materialization.dataset_kind must be 'blcs' or 'plcs'; "
                f"got {raw_kind!r}."
            )
        raw_source = cast(
            "str",
            require_config_value(mapping, "source_dir", str, path="materialization"),
        )
        raw_output = cast(
            "str",
            require_config_value(mapping, "output_dir", str, path="materialization"),
        )
        if not raw_source or not raw_output:
            raise CourtCoordinateMaterializationError(
                "materialization.source_dir and output_dir must be non-empty."
            )
        raw_source_version = cast(
            "str",
            require_config_value(
                mapping,
                "source_normalization_version",
                str,
                path="materialization",
            ),
        )
        raw_tolerance = cast(
            "float | int",
            require_config_value(
                mapping,
                "max_abs_round_trip_error_m",
                (float, int),
                path="materialization",
            ),
        )
        tolerance = float(raw_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise CourtCoordinateMaterializationError(
                "materialization.max_abs_round_trip_error_m must be finite and > 0; "
                f"got {raw_tolerance!r}."
            )
        return cls(
            dataset_kind=raw_kind,
            source_dir=_project_path(raw_source),
            output_dir=_project_path(raw_output),
            source_contract=resolve_court_coordinate_normalization(
                raw_source_version
            ),
            target_contract=target,
            max_abs_round_trip_error_m=tolerance,
        )


@dataclass(frozen=True, slots=True)
class CourtCoordinateMaterializationResult:
    """Published output paths and physical reconstruction evidence."""

    output_dir: Path
    manifest_path: Path
    scene_count: int
    max_abs_round_trip_error_m: float


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_position_array(path: Path) -> np.ndarray:
    try:
        array = cast("np.ndarray", np.load(path, allow_pickle=False))
    except FileNotFoundError as error:
        raise CourtCoordinateMaterializationError(
            f"Required position array does not exist: {path}."
        ) from error
    if array.ndim < 1 or array.shape[-1] != 3:
        raise CourtCoordinateMaterializationError(
            f"{path}: position array must have shape (..., 3); got {array.shape!r}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise CourtCoordinateMaterializationError(
            f"{path}: position array must be numeric; got dtype {array.dtype}."
        )
    if not np.isfinite(array).all():
        raise CourtCoordinateMaterializationError(
            f"{path}: position array contains non-finite values."
        )
    return array


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _assert_compatible_values(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    location: Path,
    description: str,
    tolerance: float = 1.0e-5,
) -> None:
    error = _max_abs_error(actual, expected)
    if not np.isfinite(error) or error > tolerance:
        raise CourtCoordinateMaterializationError(
            f"{location}: {description} mismatch; max_abs_error={error:.9g}, "
            f"tolerance={tolerance:.9g}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    value = load_json(path)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CourtCoordinateMaterializationError(
            f"{path}: expected a JSON object with string keys."
        )
    return cast("dict[str, object]", value)


def _replace_validated_source_metadata(
    document: dict[str, object],
    *,
    source_contract: CourtCoordinateNormalization,
    target_contract: CourtCoordinateNormalization,
    location: Path,
) -> dict[str, object]:
    """Replace only a previously validated source contract in a staged copy."""
    source_metadata = CourtCoordinateNormalizationMetadata.from_contract(
        source_contract
    )
    existing = extract_court_coordinate_normalization_metadata(
        document,
        location=str(location),
    )
    if existing is None:
        if source_contract.version != "v1":
            raise MissingCourtCoordinateMetadataError(
                f"{location}: source normalization metadata disappeared during "
                "materialization; metadata-free sources are legacy v1 only."
            )
    elif existing != source_metadata:
        raise CourtCoordinateContractMismatchError(
            f"{location}: source normalization changed during materialization; "
            f"expected {source_metadata.to_dict()!r}, got {existing.to_dict()!r}."
        )
    result = dict(document)
    result[COURT_COORDINATE_NORMALIZATION_METADATA_KEY] = (
        CourtCoordinateNormalizationMetadata.from_contract(target_contract).to_dict()
    )
    return result


def _write_contract_metadata(
    root: Path,
    scene_dirs: list[Path],
    *,
    source_contract: CourtCoordinateNormalization,
    target_contract: CourtCoordinateNormalization,
    validated_source_root_metadata_absent: bool,
) -> None:
    root_path = root / "meta.json"
    if root_path.exists():
        root_document = _load_json_object(root_path)
    elif validated_source_root_metadata_absent and source_contract.version == "v1":
        root_document = {}
    else:
        root_document = _load_json_object(root_path)
    root_metadata = _replace_validated_source_metadata(
        root_document,
        source_contract=source_contract,
        target_contract=target_contract,
        location=root_path,
    )
    save_json_atomic(root_metadata, root_path)
    for scene_dir in scene_dirs:
        metadata_path = scene_dir / "meta.json"
        metadata = _replace_validated_source_metadata(
            _load_json_object(metadata_path),
            source_contract=source_contract,
            target_contract=target_contract,
            location=metadata_path,
        )
        save_json_atomic(metadata, metadata_path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _validate_paths(config: CourtCoordinateMaterializationConfig) -> tuple[Path, Path]:
    source = config.source_dir.resolve(strict=True)
    target = config.output_dir.resolve(strict=False)
    if not source.is_dir():
        raise CourtCoordinateMaterializationError(
            f"Materialization source is not a directory: {source}."
        )
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing materialization output: {target}."
        )
    if target == source or target.is_relative_to(source):
        raise CourtCoordinateMaterializationError(
            "Materialization output must be separate from and outside the source "
            f"dataset: source={source}, output={target}."
        )
    version_tokens = {
        f"norm-{config.target_contract.version}",
        f"norm_{config.target_contract.version}",
    }
    if not any(token in target.name for token in version_tokens):
        expected = " or ".join(sorted(version_tokens))
        raise CourtCoordinateMaterializationError(
            "Materialization output directory name must identify the target "
            f"normalization version with {expected!r}; got {target.name!r}."
        )
    scenes_dir = source / "scenes"
    if not scenes_dir.is_dir():
        raise CourtCoordinateMaterializationError(
            f"Materialization source has no scenes directory: {scenes_dir}."
        )
    return source, target


def _scene_directories(root: Path) -> list[Path]:
    scenes = sorted(path for path in (root / "scenes").iterdir() if path.is_dir())
    if not scenes:
        raise CourtCoordinateMaterializationError(
            f"Materialization source contains no scene directories: {root / 'scenes'}."
        )
    return scenes


def _copy_source_dataset(source: Path, staging: Path) -> None:
    """Copy the canonical dataset while excluding source-version caches."""

    def ignore_root_caches(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() != source:
            return set()
        return set(_ROOT_CACHE_DIRECTORY_NAMES.intersection(names))

    shutil.copytree(
        source,
        staging,
        dirs_exist_ok=True,
        symlinks=False,
        ignore=ignore_root_caches,
    )


def _materialize_into_staging(
    *,
    config: CourtCoordinateMaterializationConfig,
    source: Path,
    staging: Path,
    validated_source_root_metadata_absent: bool,
) -> tuple[dict[str, object], float]:
    _copy_source_dataset(source, staging)
    source_scenes = _scene_directories(source)
    staging_scenes = [staging / "scenes" / path.name for path in source_scenes]
    adapter = _ADAPTERS[config.dataset_kind]
    scene_evidence: list[dict[str, object]] = []
    maximum_error = 0.0

    for source_scene, staging_scene in zip(
        source_scenes,
        staging_scenes,
        strict=True,
    ):
        physical, physical_source_path = adapter.physical_position(
            source_scene,
            config.source_contract,
        )
        normalized = config.target_contract.normalize_position(physical)
        output_path = staging_scene / adapter.normalized_filename
        np.save(output_path, normalized, allow_pickle=False)
        written = _load_position_array(output_path)
        reconstructed = config.target_contract.denormalize_position(written)
        error = _max_abs_error(reconstructed, physical)
        if not np.isfinite(error) or error > config.max_abs_round_trip_error_m:
            raise CourtCoordinateMaterializationError(
                f"{output_path}: physical round trip max_abs_error={error:.9g}m "
                "exceeds configured tolerance "
                f"{config.max_abs_round_trip_error_m:.9g}m."
            )
        maximum_error = max(maximum_error, error)
        scene_evidence.append(
            {
                "scene": source_scene.name,
                "physical_source": physical_source_path.name,
                "physical_source_sha256": _sha256(physical_source_path),
                "normalized_output": adapter.normalized_filename,
                "normalized_output_sha256": _sha256(output_path),
                "max_abs_round_trip_error_m": error,
            }
        )

    _write_contract_metadata(
        staging,
        staging_scenes,
        source_contract=config.source_contract,
        target_contract=config.target_contract,
        validated_source_root_metadata_absent=validated_source_root_metadata_absent,
    )
    validate_dataset_court_coordinate_contract(
        staging,
        config.target_contract,
        scene_paths=staging_scenes,
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_kind": config.dataset_kind,
        "source_dataset": _display_path(source),
        "source_normalization": CourtCoordinateNormalizationMetadata.from_contract(
            config.source_contract
        ).to_dict(),
        "target_normalization": CourtCoordinateNormalizationMetadata.from_contract(
            config.target_contract
        ).to_dict(),
        "normalized_array": adapter.normalized_filename,
        "scene_count": len(scene_evidence),
        "max_abs_round_trip_error_m": maximum_error,
        "round_trip_tolerance_m": config.max_abs_round_trip_error_m,
        "scenes": scene_evidence,
    }
    save_json_atomic(manifest, staging / _MANIFEST_FILENAME)
    return manifest, maximum_error


def materialize_court_coordinate_normalization_dataset(
    config: CourtCoordinateMaterializationConfig,
) -> CourtCoordinateMaterializationResult:
    """Publish one complete non-overwriting, version-qualified dataset copy."""
    source, target = _validate_paths(config)
    source_scenes = _scene_directories(source)
    source_validation = validate_dataset_court_coordinate_contract(
        source,
        config.source_contract,
        scene_paths=source_scenes,
    )
    validated_source_root_metadata_absent = (
        source_validation.legacy_metadata_free and not (source / "meta.json").exists()
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=target.parent,
        )
    )
    try:
        _manifest, maximum_error = _materialize_into_staging(
            config=config,
            source=source,
            staging=staging,
            validated_source_root_metadata_absent=validated_source_root_metadata_absent,
        )
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite output created during materialization: {target}."
            )
        staging.rename(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return CourtCoordinateMaterializationResult(
        output_dir=target,
        manifest_path=target / _MANIFEST_FILENAME,
        scene_count=len(source_scenes),
        max_abs_round_trip_error_m=maximum_error,
    )
