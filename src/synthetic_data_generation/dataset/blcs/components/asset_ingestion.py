"""Normalize production tennis-ball Gaussian assets for BLCS datasets."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self

import torch
from torch import Tensor

from src.synthetic_data_generation.composition.contracts import (
    ASSET_COORDINATE_FRAME,
    GAUSSIAN_ASSET_SCHEMA,
    METRE_UNIT,
    NHT_APPEARANCE_MODEL,
    NHT_TENSOR_ENCODING,
    GaussianAsset,
)
from src.synthetic_data_generation.composition.gaussians import (
    GaussianTensorSet,
    transform_gaussians,
)
from src.synthetic_data_generation.dataset.blcs.artifacts.asset_registry import (
    BallAssetEntry,
    BallAssetRegistry,
    load_ball_asset_registry,
    verify_local_ball_artifact,
    verify_local_gaussian_asset,
    write_ball_asset_registry,
)
from src.synthetic_data_generation.scene_contract import (
    ArtifactRef,
    SimilarityTransform,
)

BallSourceFormat = Literal[
    "shared_nht_tensor_pack_v1",
    "independent_nht_tensor_pack_v1",
    "vanilla_3dgs_ply_v1",
]
BallConversionMethod = Literal[
    "identity_shared_nht_v1",
    "frozen_target_nht_feature_optimization_v1",
]

BALL_ASSET_INGESTION_SCHEMA = "tennis_ball_asset_ingestion_spec_v1"
BALL_ASSET_CONVERSION_REPORT_SCHEMA = "tennis_ball_asset_conversion_report_v1"
BALL_ASSET_PUBLICATION_SCHEMA = "tennis_ball_asset_publication_v1"

SHARED_NHT_SOURCE: Final[BallSourceFormat] = "shared_nht_tensor_pack_v1"
INDEPENDENT_NHT_SOURCE: Final[BallSourceFormat] = "independent_nht_tensor_pack_v1"
VANILLA_3DGS_SOURCE: Final[BallSourceFormat] = "vanilla_3dgs_ply_v1"
IDENTITY_SHARED_NHT: Final[BallConversionMethod] = "identity_shared_nht_v1"
FROZEN_TARGET_NHT_FIT: Final[BallConversionMethod] = (
    "frozen_target_nht_feature_optimization_v1"
)

_SOURCE_FORMATS = {
    SHARED_NHT_SOURCE,
    INDEPENDENT_NHT_SOURCE,
    VANILLA_3DGS_SOURCE,
}
_CONVERSION_METHODS = {IDENTITY_SHARED_NHT, FROZEN_TARGET_NHT_FIT}
_TENSOR_KEYS = {
    "means",
    "quats",
    "scales",
    "opacities",
    "features",
    "instance_ids",
}
_MIN_CONVERSION_PSNR_DB = 20.0
_MAX_DIAMETER_RELATIVE_ERROR = 0.25
_MAX_ORIGIN_OFFSET_DIAMETER_FRACTION = 0.10
_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_ID_INITIAL_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


@dataclass(frozen=True)
class BallAssetIngestionSpec:
    """One user source plus its explicit target-NHT preparation declaration."""

    schema: str
    variant_id: str
    asset_id: str
    nominal_diameter_m: float
    source_format: BallSourceFormat
    source_artifacts: tuple[ArtifactRef, ...]
    prepared_tensors: ArtifactRef
    prepared_appearance_space_sha256: str
    prepared_appearance_payload: ArtifactRef
    conversion_method: BallConversionMethod
    conversion_report: ArtifactRef | None
    asset_from_prepared: SimilarityTransform

    def __post_init__(self) -> None:
        if self.schema != BALL_ASSET_INGESTION_SCHEMA:
            raise ValueError(
                f"Unsupported ball ingestion schema {self.schema!r}; "
                f"expected {BALL_ASSET_INGESTION_SCHEMA!r}."
            )
        _validate_id(self.variant_id, name="variant_id")
        _validate_id(self.asset_id, name="asset_id")
        diameter = _finite_float(
            self.nominal_diameter_m,
            name="nominal_diameter_m",
        )
        if not 0.05 <= diameter <= 0.09:
            raise ValueError("nominal_diameter_m must lie in [0.05, 0.09] metres.")
        source_artifacts = tuple(self.source_artifacts)
        if not source_artifacts:
            raise ValueError("source_artifacts must not be empty.")
        _require_unique(
            [artifact.artifact_id for artifact in source_artifacts],
            name="source artifact ids",
        )
        appearance_space = _sha256(
            self.prepared_appearance_space_sha256,
            name="prepared_appearance_space_sha256",
        )
        if self.source_format not in _SOURCE_FORMATS:
            raise ValueError(f"Unsupported source_format: {self.source_format!r}.")
        if self.conversion_method not in _CONVERSION_METHODS:
            raise ValueError(
                f"Unsupported conversion_method: {self.conversion_method!r}."
            )
        if self.conversion_method == IDENTITY_SHARED_NHT:
            if self.source_format != SHARED_NHT_SOURCE:
                raise ValueError(
                    "Vanilla or independently trained NHT sources require the "
                    "frozen-target conversion path."
                )
            if self.conversion_report is not None:
                raise ValueError(
                    "Identity shared-NHT ingestion must not claim a conversion report."
                )
            if not any(
                _same_artifact_content(self.prepared_tensors, artifact)
                for artifact in source_artifacts
            ):
                raise ValueError(
                    "Identity shared-NHT prepared_tensors must be one declared "
                    "source artifact."
                )
        else:
            if self.source_format == SHARED_NHT_SOURCE:
                raise ValueError(
                    "Already shared NHT sources must use identity ingestion."
                )
            if self.conversion_report is None:
                raise ValueError(
                    "Vanilla or independent NHT sources require a conversion report."
                )
        object.__setattr__(self, "nominal_diameter_m", diameter)
        object.__setattr__(self, "source_artifacts", source_artifacts)
        object.__setattr__(
            self,
            "prepared_appearance_space_sha256",
            appearance_space,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible source declaration."""
        return {
            "schema": self.schema,
            "variant_id": self.variant_id,
            "asset_id": self.asset_id,
            "nominal_diameter_m": self.nominal_diameter_m,
            "source_format": self.source_format,
            "source_artifacts": [
                artifact.to_dict() for artifact in self.source_artifacts
            ],
            "prepared_tensors": self.prepared_tensors.to_dict(),
            "prepared_appearance_space_sha256": (self.prepared_appearance_space_sha256),
            "prepared_appearance_payload": (self.prepared_appearance_payload.to_dict()),
            "conversion_method": self.conversion_method,
            "conversion_report": (
                None
                if self.conversion_report is None
                else self.conversion_report.to_dict()
            ),
            "asset_from_prepared": self.asset_from_prepared.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict source declaration without compatibility fallback."""
        raw = _strict_mapping(
            value,
            name="ball asset ingestion spec",
            keys={
                "schema",
                "variant_id",
                "asset_id",
                "nominal_diameter_m",
                "source_format",
                "source_artifacts",
                "prepared_tensors",
                "prepared_appearance_space_sha256",
                "prepared_appearance_payload",
                "conversion_method",
                "conversion_report",
                "asset_from_prepared",
            },
        )
        source_format_value = _string(raw["source_format"], name="source_format")
        if source_format_value not in _SOURCE_FORMATS:
            raise ValueError(f"Unsupported source_format: {source_format_value!r}.")
        conversion_value = _string(
            raw["conversion_method"],
            name="conversion_method",
        )
        if conversion_value not in _CONVERSION_METHODS:
            raise ValueError(f"Unsupported conversion_method: {conversion_value!r}.")
        source_artifacts = raw["source_artifacts"]
        if not isinstance(source_artifacts, Sequence) or isinstance(
            source_artifacts,
            (str, bytes),
        ):
            raise TypeError("source_artifacts must be a JSON array.")
        conversion_report = raw["conversion_report"]
        return cls(
            schema=_string(raw["schema"], name="schema"),
            variant_id=_string(raw["variant_id"], name="variant_id"),
            asset_id=_string(raw["asset_id"], name="asset_id"),
            nominal_diameter_m=_finite_float(
                raw["nominal_diameter_m"],
                name="nominal_diameter_m",
            ),
            source_format=source_format_value,
            source_artifacts=tuple(
                ArtifactRef.from_dict(artifact) for artifact in source_artifacts
            ),
            prepared_tensors=ArtifactRef.from_dict(raw["prepared_tensors"]),
            prepared_appearance_space_sha256=_string(
                raw["prepared_appearance_space_sha256"],
                name="prepared_appearance_space_sha256",
            ),
            prepared_appearance_payload=ArtifactRef.from_dict(
                raw["prepared_appearance_payload"]
            ),
            conversion_method=conversion_value,
            conversion_report=(
                None
                if conversion_report is None
                else ArtifactRef.from_dict(conversion_report)
            ),
            asset_from_prepared=SimilarityTransform.from_dict(
                raw["asset_from_prepared"]
            ),
        )


def publish_ball_asset_registry_from_sources(
    output_dir: Path,
    *,
    registry_id: str,
    target_background: GaussianAsset,
    sources: Sequence[BallAssetIngestionSpec],
) -> Path:
    """Canonicalize compatible sources and atomically publish one registry."""
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite ball asset publication: {destination}"
        )
    _validate_id(registry_id, name="registry_id")
    if target_background.role != "background":
        raise ValueError("target_background must be a background Gaussian asset.")
    verify_local_gaussian_asset(target_background)
    source_tuple = tuple(sorted(sources, key=lambda item: item.variant_id))
    if not source_tuple:
        raise ValueError("At least one ball asset source is required.")
    _require_unique(
        [source.variant_id for source in source_tuple],
        name="source variant ids",
    )
    _require_unique(
        [source.asset_id for source in source_tuple],
        name="source asset ids",
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
        entries = tuple(
            _ingest_one_source(
                temporary,
                destination=destination,
                target_background=target_background,
                source=source,
            )
            for source in source_tuple
        )
        registry = BallAssetRegistry.create(
            registry_id=registry_id,
            appearance_space_sha256=target_background.appearance_space_sha256,
            entries=entries,
        )
        write_ball_asset_registry(temporary / "registry.json", registry)
        temporary.replace(destination)
        published = destination / "registry.json"
        load_ball_asset_registry(published)
        return published
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _ingest_one_source(
    temporary: Path,
    *,
    destination: Path,
    target_background: GaussianAsset,
    source: BallAssetIngestionSpec,
) -> BallAssetEntry:
    for artifact in source.source_artifacts:
        verify_local_ball_artifact(artifact)
    prepared_path = verify_local_ball_artifact(source.prepared_tensors)
    verify_local_ball_artifact(source.prepared_appearance_payload)
    report_path = (
        None
        if source.conversion_report is None
        else verify_local_ball_artifact(source.conversion_report)
    )
    _verify_target_appearance(source, target_background=target_background)

    prepared = _load_prepared_tensor_set(
        prepared_path,
        appearance_space_sha256=source.prepared_appearance_space_sha256,
    )
    if prepared.feature_dim != target_background.feature_dim:
        raise ValueError(
            f"Prepared feature_dim {prepared.feature_dim} differs from target "
            f"{target_background.feature_dim}."
        )
    if report_path is not None:
        _verify_conversion_report(
            report_path,
            source=source,
            target_background=target_background,
            gaussian_count=prepared.gaussian_count,
            feature_dim=prepared.feature_dim,
        )
    canonical = transform_gaussians(prepared, source.asset_from_prepared)
    canonical = GaussianTensorSet(
        means=canonical.means,
        quats=canonical.quats,
        log_scales=canonical.log_scales,
        opacity_logits=canonical.opacity_logits,
        features=canonical.features,
        instance_ids=torch.zeros_like(canonical.instance_ids),
        appearance_space_sha256=canonical.appearance_space_sha256,
    )
    geometry = _metric_geometry(canonical, source.nominal_diameter_m)

    assets_dir = temporary / "assets"
    assets_dir.mkdir(exist_ok=True)
    tensor_path = assets_dir / f"{source.asset_id}.pt"
    torch.save(_tensor_payload(canonical), tensor_path)
    final_tensor_path = destination / "assets" / tensor_path.name
    tensor_artifact = _artifact(
        f"{source.asset_id}-metric-nht-tensors",
        tensor_path,
        published_path=final_tensor_path,
    )
    publication_path = assets_dir / f"{source.asset_id}.ingestion.json"
    final_publication_path = destination / "assets" / publication_path.name
    publication = {
        "schema": BALL_ASSET_PUBLICATION_SCHEMA,
        "variant_id": source.variant_id,
        "asset_id": source.asset_id,
        "nominal_diameter_m": source.nominal_diameter_m,
        "source_format": source.source_format,
        "conversion_method": source.conversion_method,
        "source_artifacts": [
            _artifact_content(artifact) for artifact in source.source_artifacts
        ],
        "prepared_tensors": _artifact_content(source.prepared_tensors),
        "prepared_appearance_space_sha256": (source.prepared_appearance_space_sha256),
        "prepared_appearance_payload": _artifact_content(
            source.prepared_appearance_payload
        ),
        "conversion_report": (
            None
            if source.conversion_report is None
            else _artifact_content(source.conversion_report)
        ),
        "target_background": {
            "asset_id": target_background.asset_id,
            "appearance_space_sha256": (target_background.appearance_space_sha256),
            "appearance_payload": _artifact_content(
                target_background.appearance_payload
            ),
            "feature_dim": target_background.feature_dim,
        },
        "asset_from_prepared": source.asset_from_prepared.to_dict(),
        "metric_geometry": geometry,
        "output_tensors": _artifact_content(tensor_artifact),
    }
    publication["publication_fingerprint"] = _canonical_sha256(publication)
    publication_path.write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publication_artifact = _artifact(
        f"{source.asset_id}-ingestion",
        publication_path,
        published_path=final_publication_path,
    )
    provenance = _unique_artifacts(
        (
            *source.source_artifacts,
            source.prepared_tensors,
            *(() if source.conversion_report is None else (source.conversion_report,)),
            publication_artifact,
        )
    )
    asset = GaussianAsset(
        schema=GAUSSIAN_ASSET_SCHEMA,
        asset_id=source.asset_id,
        asset_class="tennis-ball",
        role="movable",
        coordinate_frame=ASSET_COORDINATE_FRAME,
        unit=METRE_UNIT,
        metres_per_unit=1.0,
        gaussian_count=canonical.gaussian_count,
        feature_dim=canonical.feature_dim,
        tensor_encoding=NHT_TENSOR_ENCODING,
        tensors=tensor_artifact,
        appearance_model=NHT_APPEARANCE_MODEL,
        appearance_space_sha256=target_background.appearance_space_sha256,
        appearance_payload=target_background.appearance_payload,
        provenance=provenance,
    )
    return BallAssetEntry(
        variant_id=source.variant_id,
        asset=asset,
        nominal_diameter_m=source.nominal_diameter_m,
    )


def _load_prepared_tensor_set(
    path: Path,
    *,
    appearance_space_sha256: str,
) -> GaussianTensorSet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != _TENSOR_KEYS:
        actual = (
            sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__
        )
        raise ValueError(f"Prepared NHT tensor keys differ: {actual}.")
    tensors: dict[str, Tensor] = {}
    for name in _TENSOR_KEYS:
        value = payload[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"Prepared NHT field {name!r} must be a tensor.")
        tensors[name] = value
    floating_names = {"means", "quats", "scales", "opacities", "features"}
    if any(tensors[name].dtype != torch.float32 for name in floating_names):
        raise TypeError("Prepared NHT floating tensors must use torch.float32.")
    if tensors["instance_ids"].dtype != torch.int64:
        raise TypeError("Prepared NHT instance_ids must use torch.int64.")
    return GaussianTensorSet(
        means=tensors["means"],
        quats=tensors["quats"],
        log_scales=tensors["scales"],
        opacity_logits=tensors["opacities"],
        features=tensors["features"],
        instance_ids=tensors["instance_ids"],
        appearance_space_sha256=appearance_space_sha256,
    )


def _metric_geometry(
    gaussians: GaussianTensorSet,
    nominal_diameter_m: float,
) -> dict[str, object]:
    sigma = torch.exp(gaussians.log_scales).amax(dim=-1)
    if not bool(torch.isfinite(sigma).all()):
        raise ValueError("Gaussian metric scales overflow after canonicalization.")
    radius = torch.linalg.vector_norm(gaussians.means, dim=-1)
    envelope_radius = radius + 3.0 * sigma
    p99_radius = float(torch.quantile(envelope_radius, 0.99))
    effective_diameter = 2.0 * p99_radius
    minimum = gaussians.means.amin(dim=0)
    maximum = gaussians.means.amax(dim=0)
    aabb_midpoint = 0.5 * (minimum + maximum)
    origin_offset = float(torch.linalg.vector_norm(aabb_midpoint))
    diameter_relative_error = abs(effective_diameter - nominal_diameter_m) / (
        nominal_diameter_m
    )
    if diameter_relative_error > _MAX_DIAMETER_RELATIVE_ERROR:
        raise ValueError(
            "Metric-local Gaussian diameter differs from nominal diameter: "
            f"effective={effective_diameter:.6g} m, "
            f"nominal={nominal_diameter_m:.6g} m, "
            f"relative_error={diameter_relative_error:.6g}."
        )
    maximum_origin_offset = nominal_diameter_m * _MAX_ORIGIN_OFFSET_DIAMETER_FRACTION
    if origin_offset > maximum_origin_offset:
        raise ValueError(
            "Metric-local Gaussian origin does not coincide with ball centre: "
            f"offset={origin_offset:.6g} m > {maximum_origin_offset:.6g} m."
        )
    return {
        "gaussian_count": gaussians.gaussian_count,
        "feature_dim": gaussians.feature_dim,
        "aabb_min_m": [float(value) for value in minimum],
        "aabb_max_m": [float(value) for value in maximum],
        "aabb_midpoint_m": [float(value) for value in aabb_midpoint],
        "origin_offset_m": origin_offset,
        "p99_three_sigma_radius_m": p99_radius,
        "effective_diameter_m": effective_diameter,
        "nominal_diameter_m": nominal_diameter_m,
        "diameter_relative_error": diameter_relative_error,
        "maximum_diameter_relative_error": _MAX_DIAMETER_RELATIVE_ERROR,
        "maximum_origin_offset_m": maximum_origin_offset,
    }


def _verify_target_appearance(
    source: BallAssetIngestionSpec,
    *,
    target_background: GaussianAsset,
) -> None:
    if (
        source.prepared_appearance_space_sha256
        != target_background.appearance_space_sha256
    ):
        raise ValueError(
            "Prepared ball features do not declare the frozen target NHT "
            "appearance space."
        )
    if not _same_artifact_content(
        source.prepared_appearance_payload,
        target_background.appearance_payload,
    ):
        raise ValueError(
            "Prepared ball appearance payload differs from the target background "
            "deferred shader."
        )


def _verify_conversion_report(
    path: Path,
    *,
    source: BallAssetIngestionSpec,
    target_background: GaussianAsset,
    gaussian_count: int,
    feature_dim: int,
) -> None:
    with path.open(encoding="utf-8") as handle:
        raw = _strict_mapping(
            json.load(handle),
            name="ball conversion report",
            keys={
                "schema",
                "status",
                "method",
                "source_format",
                "target_appearance_space_sha256",
                "target_appearance_payload_sha256",
                "prepared_tensors_sha256",
                "gaussian_count",
                "feature_dim",
                "optimization_steps",
                "validation_views",
                "validation_psnr_db",
            },
        )
    if _string(raw["schema"], name="schema") != BALL_ASSET_CONVERSION_REPORT_SCHEMA:
        raise ValueError("Unsupported ball conversion report schema.")
    if _string(raw["status"], name="status") != "passed":
        raise ValueError("Ball conversion report status must be 'passed'.")
    expected_values = {
        "method": source.conversion_method,
        "source_format": source.source_format,
        "target_appearance_space_sha256": (target_background.appearance_space_sha256),
        "target_appearance_payload_sha256": (
            target_background.appearance_payload.sha256
        ),
        "prepared_tensors_sha256": source.prepared_tensors.sha256,
    }
    for name, expected in expected_values.items():
        actual = _string(raw[name], name=name)
        if actual != expected:
            raise ValueError(
                f"Ball conversion report {name} differs: {actual!r} != {expected!r}."
            )
    declared_count = _positive_int(raw["gaussian_count"], name="gaussian_count")
    declared_feature_dim = _positive_int(raw["feature_dim"], name="feature_dim")
    if declared_count != gaussian_count:
        raise ValueError(
            "Ball conversion report gaussian_count differs from prepared tensors: "
            f"{declared_count} != {gaussian_count}."
        )
    if declared_feature_dim != feature_dim:
        raise ValueError(
            "Ball conversion report feature_dim differs from prepared tensors: "
            f"{declared_feature_dim} != {feature_dim}."
        )
    for name in ("optimization_steps", "validation_views"):
        if _positive_int(raw[name], name=name) <= 0:
            raise ValueError(f"{name} must be positive.")
    psnr = _finite_float(raw["validation_psnr_db"], name="validation_psnr_db")
    if psnr < _MIN_CONVERSION_PSNR_DB:
        raise ValueError(
            f"Ball conversion PSNR {psnr:.6g} dB is below "
            f"{_MIN_CONVERSION_PSNR_DB:.6g} dB."
        )


def _tensor_payload(gaussians: GaussianTensorSet) -> dict[str, Tensor]:
    return {
        "means": gaussians.means.contiguous(),
        "quats": gaussians.quats.contiguous(),
        "scales": gaussians.log_scales.contiguous(),
        "opacities": gaussians.opacity_logits.contiguous(),
        "features": gaussians.features.contiguous(),
        "instance_ids": gaussians.instance_ids.contiguous(),
    }


def _artifact(
    artifact_id: str,
    path: Path,
    *,
    published_path: Path,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=published_path.resolve().as_uri(),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _unique_artifacts(artifacts: Sequence[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    by_id: dict[str, ArtifactRef] = {}
    for artifact in artifacts:
        previous = by_id.get(artifact.artifact_id)
        if previous is not None and not _same_artifact_content(previous, artifact):
            raise ValueError(
                f"Artifact id {artifact.artifact_id!r} refers to multiple contents."
            )
        by_id[artifact.artifact_id] = artifact
    return tuple(by_id[key] for key in sorted(by_id))


def _artifact_content(artifact: ArtifactRef) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def _same_artifact_content(first: ArtifactRef, second: ArtifactRef) -> bool:
    return bool(first.sha256 == second.sha256 and first.size_bytes == second.size_bytes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _sha256(value: str, *, name: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a full SHA-256 digest.")
    return digest


def _validate_id(value: str, *, name: str) -> None:
    if (
        not value
        or value[0] not in _ID_INITIAL_CHARS
        or any(character not in _ID_CHARS for character in value)
    ):
        raise ValueError(f"Invalid {name}: {value!r}.")


def _require_unique(values: Sequence[object], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates.")
