"""Derive larger/longer experiments without changing or regenerating paid inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .batch import import_reference, reuse_comparison
from .contracts import VariantConfig
from .generation import record_result
from .sampling import extend_indices
from .validation import load_attempt, validate_integrity, validate_ready
from .workspace import load_manifest, prepare, sha256, write_json


def derive_variant(
    parent_root: Path,
    output_root: Path,
    *,
    scene_id: str,
    sample_count: int,
    max_steps: int,
) -> dict[str, Any]:
    """Copy accepted results with their reviews; new views remain unaccepted."""
    parent_root, output_root = parent_root.resolve(), output_root.resolve()
    if parent_root == output_root:
        raise ValueError("Derived experiment needs a separate output directory")
    parent = load_manifest(parent_root)
    validate_ready(parent_root, parent)
    if parent.config.generation_provider != "openai_api":
        raise ValueError("Derivation currently requires saved API results")
    indices = extend_indices(
        [frame.source_index for frame in parent.frames], sample_count
    )
    config = VariantConfig.model_validate(
        {
            **parent.config.model_dump(),
            "output_root": output_root,
            "scene_id": scene_id,
            "sample_count": sample_count,
            "frame_indices": indices,
            "max_steps": max_steps,
        }
    )
    manifest = prepare(config)
    lineage = {
        "parent_root": str(parent_root),
        "parent_manifest_sha256": sha256(parent_root / "manifest.json"),
        "selection_policy": "preserve parent views; bisect largest gap, earlier gap on ties",
        "frame_indices": indices,
        "reused_frames": [frame.name for frame in parent.frames],
        "additional_indices": sorted(
            set(indices) - {frame.source_index for frame in parent.frames}
        ),
        "training_initialization": "from scratch; parent checkpoints are not imported",
    }
    provenance = output_root / "provenance/derived-from.json"
    if provenance.exists():
        if json.loads(provenance.read_text()) != lineage:
            raise ValueError("Derived parent provenance has changed")
    else:
        write_json(provenance, lineage)
    import_reference(
        output_root,
        parent_root / "reference/clay.png",
        review_notes="Byte-identical copy of the parent variant's accepted fixed reference",
    )
    for frame in parent.frames:
        assert frame.accepted_attempt is not None
        accepted = load_attempt(parent_root, frame.accepted_attempt)
        child = load_manifest(output_root)
        current = next(item for item in child.frames if item.name == frame.name)
        if current.accepted_attempt is not None:
            imported = load_attempt(output_root, current.accepted_attempt)
            if (
                imported.raw_sha256 != accepted.raw_sha256
                or imported.normalized_sha256 != accepted.normalized_sha256
            ):
                raise ValueError(f"Previously imported result differs: {frame.name}")
            continue
        result = reuse_comparison(
            output_root, (parent_root / frame.accepted_attempt).parent
        )
        recorded = record_result(
            output_root,
            result["request_id"],
            Path(result["path"]),
            accepted=True,
            review_notes=f"Inherited verified review from {parent_root / frame.accepted_attempt}: {accepted.review_notes}",
        )
        if (
            not recorded.accepted
            or recorded.normalized_sha256 != accepted.normalized_sha256
        ):
            raise ValueError(f"Imported training image differs: {frame.name}")
    manifest = load_manifest(output_root)
    validate_integrity(output_root, manifest, verify_source=True)
    return {
        "root": str(output_root),
        "status": manifest.status,
        "reused_images": len(parent.frames),
        "new_images_required": sample_count - len(parent.frames),
        "max_steps": max_steps,
        "splits": {
            split: sum(frame.split == split for frame in manifest.frames)
            for split in ("train", "validation")
        },
    }
