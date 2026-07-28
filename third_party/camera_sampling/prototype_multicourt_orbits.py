"""Publish immutable two-court circle/ellipse trajectories and seven-class labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.synthetic_data_generation.court.layout import load_multi_court_layout
from src.synthetic_data_generation.court.orbits import sample_orbit_families
from src.synthetic_data_generation.scene_contract import load_scene_contract

SCHEMA = "tennis_multicourt_orbit_plan_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    result = np.quantile(values, (0.0, 0.1, 0.5, 0.9, 1.0))
    return {
        key: float(value)
        for key, value in zip(
            ("minimum", "p10", "median", "p90", "maximum"),
            result,
            strict=True,
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--court-geometry", type=Path, required=True)
    parser.add_argument("--points-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26072814)
    parser.add_argument("--samples-per-orbit", type=int, default=24)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    for path in (args.contract, args.court_geometry, args.points_scene):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = load_scene_contract(args.contract)
    layout = load_multi_court_layout(
        args.court_geometry,
        contract,
        candidate_ids=("court-0", "court-1"),
    )
    result = sample_orbit_families(
        contract.cameras,
        layout,
        np.load(args.points_scene, allow_pickle=False),
        seed=args.seed,
        samples_per_orbit=args.samples_per_orbit,
    )
    coverage = Counter(
        court.coverage_bucket
        for frame in result.frames
        for court in frame.projection.courts
    )
    coverage_pairs = Counter(
        "|".join(court.coverage_bucket for court in frame.projection.courts)
        for frame in result.frames
    )
    accepted_by_family = Counter(frame.family_id for frame in result.frames)
    metrics = {
        "court_instance_count": len(layout.courts),
        "court_centres_reference_m": layout.centers_in_reference().tolist(),
        "court_centre_separation_m": float(
            np.linalg.norm(
                layout.centers_in_reference()[1]
                - layout.centers_in_reference()[0]
            )
        ),
        "family_count": len(result.families),
        "circle_family_count": sum(
            family.shape == "circle" for family in result.families
        ),
        "ellipse_family_count": sum(
            family.shape == "ellipse" for family in result.families
        ),
        "proposal_count": result.proposal_count,
        "accepted_frame_count": len(result.frames),
        "accepted_fraction": len(result.frames) / result.proposal_count,
        "rejection_counts": dict(result.rejection_counts),
        "accepted_frames_by_family": dict(sorted(accepted_by_family.items())),
        "coverage_bucket_counts": dict(sorted(coverage.items())),
        "coverage_pair_counts": dict(sorted(coverage_pairs.items())),
        "nearest_captured_translation_m": _quantiles(
            [
                frame.nearest_captured_translation_m
                for frame in result.frames
            ]
        ),
        "nearest_captured_rotation_deg": _quantiles(
            [
                frame.nearest_captured_rotation_deg
                for frame in result.frames
            ]
        ),
        "collision_clearance_m": _quantiles(
            [frame.collision_clearance_m for frame in result.frames]
        ),
        "maximum_physical_peaks_per_channel": 2 * len(layout.courts),
    }
    passed = (
        len(layout.courts) == 2
        and metrics["court_centre_separation_m"] > 12.0
        and len(result.families) == 18
        and len(result.frames) >= 0.95 * result.proposal_count
        and coverage["full"] > 0
        and coverage["near_full"] > 0
        and coverage["partial"] > 0
        and metrics["nearest_captured_translation_m"]["maximum"] > 5.0
        and metrics["nearest_captured_rotation_deg"]["maximum"] > 5.0
    )
    if not passed:
        raise RuntimeError(f"Multi-court orbit plan gate failed: {metrics}")

    unsigned = {
        "schema": SCHEMA,
        "status": "passed",
        "source": {
            "scene_contract": {
                "path": str(args.contract.resolve()),
                "sha256": _sha256(args.contract),
                "scene_fingerprint": contract.scene_fingerprint,
            },
            "court_geometry": {
                "path": str(args.court_geometry.resolve()),
                "sha256": _sha256(args.court_geometry),
                "artifact_fingerprint": layout.geometry_artifact_fingerprint,
            },
            "points_scene": {
                "path": str(args.points_scene.resolve()),
                "sha256": _sha256(args.points_scene),
            },
        },
        "label_schema": {
            "model_heatmap_channel_count": 7,
            "near_far_symmetry_removed": True,
            "court_instance_retained_in_annotation": True,
            "court_instance_grouping_is_training_target": False,
            "court_instance_grouping_stage": "postprocess",
            "multi_peak_composition": "pixelwise-maximum",
            "maximum_physical_peaks_per_channel": 2 * len(layout.courts),
        },
        "layout": {
            "reference_court_instance_id": layout.reference_court_instance_id,
            "courts": [asdict(court) for court in layout.courts],
        },
        "sampling": {
            "seed": result.seed,
            "captured_radius_x_m": result.captured_radius_x_m,
            "captured_radius_y_m": result.captured_radius_y_m,
            "complex_center_reference_m": result.complex_center_reference_m,
            "families": [asdict(family) for family in result.families],
        },
        "metrics": metrics,
        "frames": [
            {
                "family_id": frame.family_id,
                "family_frame_index": frame.frame_index,
                "camera": frame.camera.to_dict(),
                "projection": asdict(frame.projection),
                "nearest_captured_translation_m": (
                    frame.nearest_captured_translation_m
                ),
                "nearest_captured_rotation_deg": (
                    frame.nearest_captured_rotation_deg
                ),
                "collision_clearance_m": frame.collision_clearance_m,
            }
            for frame in result.frames
        ],
    }
    manifest = dict(unsigned)
    manifest["content_fingerprint"] = _canonical_sha256(unsigned)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            temporary.rmdir()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"content_fingerprint={manifest['content_fingerprint']}")


if __name__ == "__main__":
    main()
