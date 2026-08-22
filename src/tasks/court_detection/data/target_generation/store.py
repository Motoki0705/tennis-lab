"""Deterministic derived-target paths owned by Court detection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from PIL import Image, UnidentifiedImageError

from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtInputSpec,
    CourtSampleRecord,
    CourtSourceKind,
)

SEGMENTATION_TARGET_SCHEMA = "court_cell_segmentation_v1"
LINE_TARGET_SCHEMA = "court_line_binary_v1"
_DIGEST_LENGTH = 64
_DERIVED_METADATA_KEYS = {
    "schema",
    "target_kind",
    "source_kind",
    "source_schema",
    "source_sample_id",
    "stable_sample_id",
    "width",
    "height",
    "source_target_sha256",
    "sha256",
}


class CourtDerivedTargetStore:
    """Resolve derived targets without mutating either source dataset."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(
        self,
        *,
        source_kind: CourtSourceKind,
        derived_key: str,
        target_schema: str,
    ) -> Path:
        if source_kind not in {"tennis_court_detector", "synthetic_court"}:
            raise ValueError(f"Unsupported Court source kind: {source_kind!r}.")
        if not target_schema or target_schema != target_schema.strip():
            raise ValueError("Derived target schema must be non-empty and trimmed.")
        key = PurePosixPath(derived_key)
        if key.is_absolute() or not key.parts or any(
            part in {"", ".", ".."} for part in key.parts
        ):
            raise ValueError("Derived target key must be a safe relative POSIX path.")
        relative = Path(*key.parts)
        target = self.root / source_kind / target_schema / relative
        return target.with_suffix(".png")

    @staticmethod
    def metadata_path(target_path: Path) -> Path:
        if target_path.suffix.lower() != ".png":
            raise ValueError("Court derived target metadata requires a PNG path.")
        return target_path.with_suffix(".json")


def build_derived_target_metadata(
    record: CourtSampleRecord,
    *,
    input_spec: CourtInputSpec,
    target_kind: CourtDenseTargetKind,
    target_schema: str,
    target_sha256: str,
) -> dict[str, object]:
    """Build exact freshness metadata from a source-neutral record contract."""
    payload = record.payload
    if payload.get("source_schema") != input_spec.source_schema:
        raise ValueError("Court record source schema disagrees with input spec.")
    source_sample_id = payload.get("source_sample_id")
    width = payload.get("width")
    height = payload.get("height")
    source_digest = payload.get("source_target_sha256")
    if not isinstance(source_sample_id, str) or not source_sample_id:
        raise ValueError("Court record requires source_sample_id provenance.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in (width, height)
    ):
        raise ValueError("Court record requires positive source dimensions.")
    for name, digest in (
        ("source_target_sha256", source_digest),
        ("target_sha256", target_sha256),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != _DIGEST_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Court {name} must be a lowercase SHA-256 digest.")
    return {
        "schema": target_schema,
        "target_kind": target_kind,
        "source_kind": input_spec.source_kind,
        "source_schema": input_spec.source_schema,
        "source_sample_id": source_sample_id,
        "stable_sample_id": record.sample_id,
        "width": width,
        "height": height,
        "source_target_sha256": source_digest,
        "sha256": target_sha256,
    }


def validate_derived_target(
    record: CourtSampleRecord,
    *,
    input_spec: CourtInputSpec,
    target_kind: CourtDenseTargetKind,
    target_schema: str,
) -> None:
    """Reject missing, stale, symlinked, or provenance-mismatched targets."""
    try:
        target_path = record.dense_target_refs[target_kind]
    except KeyError as error:
        raise FileNotFoundError(
            f"Court sample {record.sample_id!r} has no {target_kind} target reference."
        ) from error
    metadata_path = CourtDerivedTargetStore.metadata_path(target_path)
    if (
        target_path.is_symlink()
        or not target_path.is_file()
        or metadata_path.is_symlink()
        or not metadata_path.is_file()
    ):
        raise FileNotFoundError(
            f"Precomputed Court {target_kind} target/metadata is missing or not ordinary: "
            f"{target_path}."
        )
    try:
        with Image.open(target_path) as image:
            if image.mode != "L" or image.size != (
                cast(int, record.payload.get("width")),
                cast(int, record.payload.get("height")),
            ):
                raise ValueError(
                    f"Court derived target dimensions/mode are stale: {target_path}."
                )
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(
            f"Court derived target is unreadable: {target_path}."
        ) from error
    target_digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    expected = build_derived_target_metadata(
        record,
        input_spec=input_spec,
        target_kind=target_kind,
        target_schema=target_schema,
        target_sha256=target_digest,
    )
    try:
        parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Court derived metadata is unreadable: {metadata_path}."
        ) from error
    if not isinstance(parsed, Mapping) or set(parsed) != _DERIVED_METADATA_KEYS:
        raise ValueError(f"Court derived metadata schema changed: {metadata_path}.")
    if dict(cast(Mapping[str, object], parsed)) != expected:
        raise ValueError(
            f"Court derived target is stale or belongs to another source: {target_path}."
        )


__all__ = [
    "CourtDerivedTargetStore",
    "LINE_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA",
    "build_derived_target_metadata",
    "validate_derived_target",
]
