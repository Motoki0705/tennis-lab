"""Versioned BLCS ball-asset inventory and deterministic selection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from urllib.parse import unquote, urlparse

from src.synthetic_data_generation.composition.contracts import (
    GaussianAsset,
    gaussian_asset_content_identity,
)
from src.synthetic_data_generation.scene_contract import ArtifactRef

BALL_ASSET_REGISTRY_SCHEMA = "tennis_ball_gaussian_asset_registry_v1"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class BallAssetEntry:
    """One selectable tennis-ball appearance variant."""

    variant_id: str
    asset: GaussianAsset
    nominal_diameter_m: float

    def __post_init__(self) -> None:
        _validate_id(self.variant_id, name="variant_id")
        if self.asset.role != "movable":
            raise ValueError("Ball inventory entries must reference movable assets.")
        diameter = _finite_float(
            self.nominal_diameter_m,
            name="nominal_diameter_m",
        )
        if not 0.05 <= diameter <= 0.09:
            raise ValueError(
                "nominal_diameter_m must be a plausible tennis-ball diameter "
                "in [0.05, 0.09] metres."
            )
        object.__setattr__(self, "nominal_diameter_m", diameter)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible entry."""
        return {
            "variant_id": self.variant_id,
            "nominal_diameter_m": self.nominal_diameter_m,
            "asset": self.asset.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one exact registry entry."""
        raw = _strict_mapping(
            value,
            name="ball asset entry",
            keys={"variant_id", "nominal_diameter_m", "asset"},
        )
        return cls(
            variant_id=_string(raw["variant_id"], name="variant_id"),
            nominal_diameter_m=_finite_float(
                raw["nominal_diameter_m"],
                name="nominal_diameter_m",
            ),
            asset=GaussianAsset.from_dict(raw["asset"]),
        )


@dataclass(frozen=True)
class BallAssetRegistry:
    """Content-fingerprinted inventory in one shared NHT appearance space."""

    schema: str
    registry_id: str
    registry_fingerprint: str
    appearance_space_sha256: str
    entries: tuple[BallAssetEntry, ...]

    def __post_init__(self) -> None:
        if self.schema != BALL_ASSET_REGISTRY_SCHEMA:
            raise ValueError(
                f"Unsupported ball registry schema {self.schema!r}; "
                f"expected {BALL_ASSET_REGISTRY_SCHEMA!r}."
            )
        _validate_id(self.registry_id, name="registry_id")
        appearance_digest = _sha256(
            self.appearance_space_sha256,
            name="appearance_space_sha256",
        )
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("Ball asset registry must contain at least one entry.")
        _require_unique(
            [entry.variant_id for entry in entries],
            name="ball variant ids",
        )
        _require_unique(
            [entry.asset.asset_id for entry in entries],
            name="ball asset ids",
        )
        mismatches = [
            entry.variant_id
            for entry in entries
            if entry.asset.appearance_space_sha256 != appearance_digest
        ]
        if mismatches:
            raise ValueError(
                "Ball assets do not share the declared NHT appearance space: "
                f"{mismatches}."
            )
        fingerprint = _sha256(
            self.registry_fingerprint,
            name="registry_fingerprint",
        )
        expected = compute_ball_asset_registry_fingerprint(
            registry_id=self.registry_id,
            appearance_space_sha256=appearance_digest,
            entries=entries,
        )
        if fingerprint != expected:
            raise ValueError(
                "Ball registry fingerprint mismatch: "
                f"declared {fingerprint}, computed {expected}."
            )
        object.__setattr__(self, "appearance_space_sha256", appearance_digest)
        object.__setattr__(self, "registry_fingerprint", fingerprint)
        object.__setattr__(self, "entries", entries)

    @classmethod
    def create(
        cls,
        *,
        registry_id: str,
        appearance_space_sha256: str,
        entries: Sequence[BallAssetEntry],
    ) -> Self:
        """Create a registry with a path-independent content fingerprint."""
        entry_tuple = tuple(entries)
        return cls(
            schema=BALL_ASSET_REGISTRY_SCHEMA,
            registry_id=registry_id,
            registry_fingerprint=compute_ball_asset_registry_fingerprint(
                registry_id=registry_id,
                appearance_space_sha256=appearance_space_sha256,
                entries=entry_tuple,
            ),
            appearance_space_sha256=appearance_space_sha256,
            entries=entry_tuple,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact registry record."""
        return {
            "schema": self.schema,
            "registry_id": self.registry_id,
            "registry_fingerprint": self.registry_fingerprint,
            "appearance_space_sha256": self.appearance_space_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse and fingerprint-check one registry."""
        raw = _strict_mapping(
            value,
            name="ball asset registry",
            keys={
                "schema",
                "registry_id",
                "registry_fingerprint",
                "appearance_space_sha256",
                "entries",
            },
        )
        entries = raw["entries"]
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise TypeError("entries must be a JSON array.")
        return cls(
            schema=_string(raw["schema"], name="schema"),
            registry_id=_string(raw["registry_id"], name="registry_id"),
            registry_fingerprint=_string(
                raw["registry_fingerprint"],
                name="registry_fingerprint",
            ),
            appearance_space_sha256=_string(
                raw["appearance_space_sha256"],
                name="appearance_space_sha256",
            ),
            entries=tuple(BallAssetEntry.from_dict(entry) for entry in entries),
        )


@dataclass(frozen=True)
class BallAssetSelection:
    """Auditable deterministic choice for one persistent object identity."""

    entry: BallAssetEntry
    entry_index: int
    selection_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.entry_index, bool) or self.entry_index < 0:
            raise ValueError("entry_index must be a non-negative integer.")
        object.__setattr__(
            self,
            "selection_sha256",
            _sha256(self.selection_sha256, name="selection_sha256"),
        )


def compute_ball_asset_registry_fingerprint(
    *,
    registry_id: str,
    appearance_space_sha256: str,
    entries: Sequence[BallAssetEntry],
) -> str:
    """Hash registry semantics without publication paths or input ordering."""
    payload = {
        "schema": BALL_ASSET_REGISTRY_SCHEMA,
        "registry_id": registry_id,
        "appearance_space_sha256": appearance_space_sha256.lower(),
        "entries": [
            {
                "variant_id": entry.variant_id,
                "nominal_diameter_m": entry.nominal_diameter_m,
                "asset": gaussian_asset_content_identity(entry.asset),
            }
            for entry in sorted(entries, key=lambda item: item.variant_id)
        ],
    }
    return _canonical_sha256(payload)


def select_ball_asset(
    registry: BallAssetRegistry,
    *,
    seed: int,
    selection_key: str,
) -> BallAssetSelection:
    """Select one variant deterministically without mutable RNG state."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(selection_key, str) or not selection_key.strip():
        raise ValueError("selection_key must be a non-empty string.")
    ordered = sorted(registry.entries, key=lambda entry: entry.variant_id)
    selection_sha256 = _canonical_sha256(
        {
            "registry_fingerprint": registry.registry_fingerprint,
            "seed": seed,
            "selection_key": selection_key,
        }
    )
    entry_index = int(selection_sha256, 16) % len(ordered)
    return BallAssetSelection(
        entry=ordered[entry_index],
        entry_index=entry_index,
        selection_sha256=selection_sha256,
    )


def write_ball_asset_registry(path: Path, registry: BallAssetRegistry) -> None:
    """Atomically publish a registry and refuse replacement."""
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite ball registry: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(registry.to_dict(), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_ball_asset_registry(
    path: Path,
    *,
    verify_local_artifacts: bool = True,
) -> BallAssetRegistry:
    """Load a strict registry and optionally verify every referenced local byte."""
    with path.resolve().open(encoding="utf-8") as handle:
        registry = BallAssetRegistry.from_dict(json.load(handle))
    if verify_local_artifacts:
        verify_local_ball_asset_registry(registry)
    return registry


def verify_local_ball_asset_registry(registry: BallAssetRegistry) -> None:
    """Require every referenced tensor, appearance, and provenance file to match."""
    verified: set[tuple[str, str, int]] = set()
    for entry in registry.entries:
        for artifact in _asset_artifacts(entry.asset):
            key = (
                artifact.sha256,
                artifact.artifact_id,
                artifact.size_bytes,
            )
            if key not in verified:
                verify_local_ball_artifact(artifact)
                verified.add(key)


def verify_local_gaussian_asset(asset: GaussianAsset) -> None:
    """Verify every local byte referenced by one selected Gaussian asset."""
    for artifact in _asset_artifacts(asset):
        verify_local_ball_artifact(artifact)


def _asset_artifacts(asset: GaussianAsset) -> tuple[ArtifactRef, ...]:
    return (
        asset.tensors,
        asset.appearance_payload,
        *asset.provenance,
    )


def verify_local_ball_artifact(artifact: ArtifactRef) -> Path:
    """Verify one local BLCS artifact and return its resolved path."""
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(
            "Ball registries used for generation require verifiable local file URIs; "
            f"got {artifact.uri!r}."
        )
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise FileNotFoundError(f"Missing ball asset artifact: {path}")
    if path.stat().st_size != artifact.size_bytes:
        raise ValueError(f"Ball asset artifact size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    computed = digest.hexdigest()
    if computed != artifact.sha256:
        raise ValueError(
            f"Ball asset artifact hash mismatch for {path}: "
            f"declared {artifact.sha256}, computed {computed}."
        )
    return path.resolve()


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
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite.")
    return result


def _sha256(value: str, *, name: str) -> str:
    digest = value.lower()
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a full SHA-256 digest.")
    return digest


def _validate_id(value: str, *, name: str) -> None:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid {name}: {value!r}.")


def _require_unique(values: Sequence[object], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates.")
