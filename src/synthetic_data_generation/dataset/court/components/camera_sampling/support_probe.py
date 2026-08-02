"""Publish an immutable SfM-neighborhood camera-support probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.synthetic_data_generation.dataset.court.components.camera_sampling.sfm_neighborhood import (
    NovelViewThresholds,
    pose_distance_score,
    sample_safe_novel_views,
)
from src.synthetic_data_generation.scene_contract import load_scene_contract
from src.utils.schema.court import court_keypoints_3d

SCHEMA = "court_novel_view_probe_v1"


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
    quantiles = np.asarray(
        np.quantile(values, (0.0, 0.1, 0.5, 0.9, 1.0)),
        dtype=np.float64,
    ).reshape(5)
    return {
        key: float(quantiles[index])
        for index, key in enumerate(("minimum", "p10", "median", "p90", "maximum"))
    }


def _pairwise_minimum_score(
    matrices: list[np.ndarray],
    thresholds: NovelViewThresholds,
) -> float:
    minimum = np.inf
    for first_index, first in enumerate(matrices):
        for second in matrices[first_index + 1 :]:
            minimum = min(
                minimum,
                pose_distance_score(first, second, thresholds),
            )
    return float(minimum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--points-scene", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26072813)
    parser.add_argument("--proposals-per-anchor", type=int, default=64)
    parser.add_argument("--max-views", type=int, default=256)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    for path in (args.contract, args.points_scene, args.research, args.pins):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = load_scene_contract(args.contract)
    points_scene = np.load(args.points_scene, allow_pickle=False)
    thresholds = NovelViewThresholds()
    result = sample_safe_novel_views(
        contract.cameras,
        contract.alignment.court_from_scene,
        court_keypoints_3d().numpy(),
        points_scene,
        seed=args.seed,
        proposals_per_anchor=args.proposals_per_anchor,
        max_views=args.max_views,
        thresholds=thresholds,
    )
    selected = result.selected
    matrices = [
        np.asarray(camera.camera_to_court, dtype=np.float64).reshape(4, 4)
        for camera in selected
    ]
    nontrivial_count = sum(
        camera.nearest_captured_translation_m >= 0.01
        or camera.nearest_captured_rotation_deg >= 0.10
        for camera in selected
    )
    selected_to_anchor_expansion_factor = len(selected) / result.safe_anchor_count
    metrics = {
        "safe_captured_anchor_count": result.safe_anchor_count,
        "proposal_count": result.proposal_count,
        "accepted_candidate_count": result.accepted_candidate_count,
        "accepted_candidate_fraction": (
            result.accepted_candidate_count / result.proposal_count
        ),
        "selected_count": len(selected),
        "selected_to_anchor_expansion_factor": selected_to_anchor_expansion_factor,
        "unique_selected_anchor_count": len(
            {camera.anchor_camera_id for camera in selected}
        ),
        "nontrivial_novel_pose_count": nontrivial_count,
        "selected_pairwise_minimum_pose_score": _pairwise_minimum_score(
            matrices,
            thresholds,
        ),
        "extrapolation_score": _quantiles(
            [camera.extrapolation_score for camera in selected]
        ),
        "nearest_captured_translation_m": _quantiles(
            [camera.nearest_captured_translation_m for camera in selected]
        ),
        "nearest_captured_rotation_deg": _quantiles(
            [camera.nearest_captured_rotation_deg for camera in selected]
        ),
        "collision_clearance_m": _quantiles(
            [camera.collision_clearance_m for camera in selected]
        ),
        "min_court_depth_m": _quantiles(
            [camera.min_court_depth_m for camera in selected]
        ),
        "min_line_margin_px": _quantiles(
            [camera.min_line_margin_px for camera in selected]
        ),
    }
    passed = (
        metrics["selected_count"] == args.max_views
        and selected_to_anchor_expansion_factor >= 4.0
        and metrics["unique_selected_anchor_count"] == result.safe_anchor_count
        and nontrivial_count == len(selected)
        and max(camera.extrapolation_score for camera in selected)
        <= thresholds.support_score_limit + 1.0e-10
        and min(camera.collision_clearance_m for camera in selected)
        >= thresholds.min_collision_clearance_m
        and min(camera.min_court_depth_m for camera in selected)
        > thresholds.near_plane_m
        and min(camera.min_line_margin_px for camera in selected)
        >= thresholds.min_image_margin_px
        and all(all(camera.court_keypoints_visible[:14]) for camera in selected)
    )
    if not passed:
        raise RuntimeError(f"B00 novel-view probe failed: {metrics}")

    unsigned = {
        "schema": SCHEMA,
        "status": "passed",
        "p6_sampling_probe_passed": True,
        "source": {
            "scene_contract": {
                "path": str(args.contract.resolve()),
                "sha256": _sha256(args.contract),
                "scene_fingerprint": contract.scene_fingerprint,
            },
            "points_scene": {
                "path": str(args.points_scene.resolve()),
                "sha256": _sha256(args.points_scene),
            },
            "research": {
                "path": str(args.research.resolve()),
                "sha256": _sha256(args.research),
            },
            "pins": {
                "path": str(args.pins.resolve()),
                "sha256": _sha256(args.pins),
            },
        },
        "thresholds": asdict(thresholds),
        "sampling": {
            "seed": result.seed,
            "rejection_counts": dict(result.rejection_counts),
        },
        "metrics": metrics,
        "views": [asdict(camera) for camera in selected],
    }
    manifest = dict(unsigned)
    manifest["content_fingerprint"] = _canonical_sha256(unsigned)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
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
