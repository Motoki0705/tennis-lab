"""Deterministic derived-target paths owned by court detection."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from src.tasks.court_detection.data.contracts import CourtSourceKind

SEGMENTATION_TARGET_SCHEMA = "court_cell_segmentation_v1"
LINE_TARGET_SCHEMA = "court_line_binary_v1"


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


__all__ = [
    "CourtDerivedTargetStore",
    "LINE_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA",
]
