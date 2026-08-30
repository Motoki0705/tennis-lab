"""Focused fixtures for publication contract and media validation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.alignment.contracts import (
    ALIGNMENT_TRACE_SCHEMA,
    GROUND_PLANE_FRAME_SCHEMA,
    AlignmentTracePhase,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    GROUND_PLANE_UV_COORDINATE_CONVENTION,
)
from src.synthetic_data_generation.visualization.publication.alignment import (
    ALIGNMENT_AGREEMENT_METRIC_SCHEMA,
)
from src.synthetic_data_generation.visualization.publication.cameras import (
    METRIC_CAMERA_COORDINATE_CONVENTION,
)
from src.synthetic_data_generation.visualization.publication.contracts import (
    PUBLICATION_BUNDLE_SCHEMA,
    PUBLICATION_COORDINATE_CONTRACT,
    PUBLICATION_MANIFEST_SCHEMA,
    PUBLICATION_REQUEST_SCHEMA,
    REQUIRED_PUBLICATION_ARTIFACTS,
    PublicationArtifactName,
    PublicationArtifactRecord,
    PublicationDrawingSettings,
    PublicationManifest,
)
from src.synthetic_data_generation.visualization.publication.datasets import (
    write_deterministic_gif,
)
from src.synthetic_data_generation.visualization.publication.figures import (
    CAMERA_COVERAGE_METRIC_SCHEMA,
    OVERVIEW_LAYOUT_SCHEMA,
)


def publication_config_payload(tmp_path: Path) -> dict[str, object]:
    """Return a complete, explicit configuration against temporary owners."""
    data_root = tmp_path / "runtime-data"
    scene_root = data_root / "scenes" / "scene-0"
    scene_root.mkdir(parents=True)
    return {
        "roots": {
            "project_root": str(tmp_path),
            "data_root": str(data_root),
            "checkpoint_root": str(tmp_path / "runtime-checkpoints"),
            "artifact_root": str(tmp_path / "runtime-artifacts"),
            "output_root": str(tmp_path / "runtime-output"),
            "cache_root": str(tmp_path / "runtime-cache"),
            "external_asset_root": str(tmp_path / "runtime-external"),
        },
        "publication": {
            "scene_id": "scene-0",
            "scene_root": "scenes/scene-0",
            "output_bundle": "publication/scene-0",
            "artifacts": [item.value for item in REQUIRED_PUBLICATION_ARTIFACTS],
            "court": {
                "trajectory_id": "trajectory-0",
                "frame_indices": [0, 2],
            },
            "blcs": {
                "logical_scene_id": "logical-0",
                "camera_id": "cam-0",
                "frame_indices": [0, 2],
                "camera_ids": ["cam-0", "cam-1"],
            },
            "plcs": {
                "logical_scene_id": "logical-0",
                "camera_id": "cam-0",
                "frame_indices": [0, 2],
                "camera_ids": ["cam-0", "cam-1"],
            },
            "captured": {"camera_ids": ["cam-0", "cam-1"]},
            "drawing": {
                "dataset_size": [64, 64],
                "alignment_size": [64, 64],
                "figure_size": [64, 64],
                "overview_size": [600, 400],
                "gif_duration_ms": 40,
                "frustum_depth_metres": 1.5,
                "line_width": 1.0,
                "font_size": 10,
                "history_frames": 2,
                "maximum_artifact_bytes": 1_000_000,
                "maximum_bundle_bytes": 2_000_000,
            },
        },
    }


@pytest.fixture
def publication_config(tmp_path: Path) -> dict[str, object]:
    """Provide a complete config payload for strict-boundary tests."""
    return publication_config_payload(tmp_path)


@pytest.fixture
def publication_drawing() -> PublicationDrawingSettings:
    """Provide valid, small deterministic drawing settings."""
    return PublicationDrawingSettings(
        dataset_size=(64, 64),
        alignment_size=(64, 64),
        figure_size=(64, 64),
        overview_size=(600, 400),
        gif_duration_ms=40,
        frustum_depth_metres=1.5,
        line_width=1.0,
        font_size=10,
        history_frames=2,
        maximum_artifact_bytes=1_000_000,
        maximum_bundle_bytes=2_000_000,
    )


def _content_digest(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()


def _write_png(path: Path) -> None:
    image: NDArray[np.uint8] = np.full((64, 64, 3), 245, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )


def _artifact_mapping(
    artifact: PublicationArtifactName,
) -> tuple[dict[str, object], ...]:
    if artifact in {
        PublicationArtifactName.DATASET_COURT,
        PublicationArtifactName.DATASET_BLCS,
        PublicationArtifactName.DATASET_PLCS,
    }:
        return ({"source_index": 0}, {"source_index": 2})
    if artifact is PublicationArtifactName.ALIGNMENT_PROGRESSION:
        return tuple(
            {
                "step_index": index,
                "phase": phase.value,
                "score_sum": float(index),
                "candidate_ids": ["candidate-0"],
                "candidate_scores": [float(index)],
            }
            for index, phase in enumerate(AlignmentTracePhase)
        )
    if artifact is PublicationArtifactName.ALIGNMENT_HEATMAP_COURT:
        return (
            {
                "candidate_ids": ["candidate-0"],
                "court_instance_ids": ["court-0"],
                "heatmap_camera_ids": ["cam-0"],
            },
        )
    if artifact in {
        PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY,
        PublicationArtifactName.BLCS_CAMERA_LAYOUT,
        PublicationArtifactName.PLCS_CAMERA_LAYOUT,
    }:
        return ({"camera_id": "cam-0"}, {"camera_id": "cam-1"})
    if artifact is PublicationArtifactName.CAMERA_LAYOUT_COMPARISON:
        return (
            {"camera_id": "cam-0"},
            {"camera_id": "cam-1"},
            {"camera_id": "cam-0"},
            {"camera_id": "cam-1"},
        )
    return tuple(
        {
            "panel": label,
            "source_artifact": artifact,
            "bounds_pixels": [index * 8, 0, index * 8 + 6, 6],
        }
        for index, (label, artifact) in enumerate(
            (
                ("Court dataset", "dataset-court.gif"),
                ("BLCS dataset", "dataset-blcs.gif"),
                ("PLCS dataset", "dataset-plcs.gif"),
                ("Alignment evidence", "alignment-heatmap-court.png"),
                ("Captured cameras", "captured-camera-trajectory.png"),
                ("BLCS / PLCS cameras", "camera-layout-comparison.png"),
            )
        )
    )


def _source_owners() -> dict[str, object]:
    common_dataset = {
        "owner_path": "datasets/court",
        "schema": "synthetic_dataset_v1",
        "scene_id": "scene-0",
        "domain": "court",
        "trajectory_id": "trajectory-0",
        "source_count": 3,
        "source_fps": None,
        "source_size": [128, 72],
        "output_size": [64, 64],
        "resize_filter": "Pillow LANCZOS",
        "selected_indices": [0, 2],
    }
    return {
        "court": common_dataset,
        "blcs": {
            "owner_path": "datasets/blcs",
            "schema": "synthetic_blcs_dataset_v1",
            "scene_id": "scene-0",
            "domain": "blcs",
            "logical_scene_id": "logical-0",
            "gif_camera_id": "cam-0",
            "camera_ids": ["cam-0", "cam-1"],
            "source_count": 3,
            "source_fps": 25.0,
            "source_size": [128, 72],
            "output_size": [64, 64],
            "resize_filter": "Pillow LANCZOS",
            "selected_indices": [0, 2],
        },
        "plcs": {
            "owner_path": "datasets/plcs",
            "schema": "synthetic_plcs_dataset_v1",
            "scene_id": "scene-0",
            "domain": "plcs",
            "logical_scene_id": "logical-0",
            "gif_camera_id": "cam-0",
            "camera_ids": ["cam-0", "cam-1"],
            "source_count": 3,
            "source_fps": None,
            "source_size": [128, 72],
            "output_size": [64, 64],
            "resize_filter": "Pillow LANCZOS",
            "selected_indices": [0, 2],
        },
        "alignment": {
            "owner_path": "alignment",
            "schema": "synthetic_alignment_v1",
            "scene_id": "scene-0",
            "ground_plane_frame_schema": GROUND_PLANE_FRAME_SCHEMA,
            "alignment_trace_schema": ALIGNMENT_TRACE_SCHEMA,
            "candidate_ids": ["candidate-0"],
            "court_instance_ids": ["court-0"],
            "heatmap_camera_ids": ["cam-0"],
        },
        "reconstruction": {
            "owner_path": "reconstruction/export",
            "schema": "nht_standard_cameras_v1",
            "scene_id": "scene-0",
            "camera_ids": ["cam-0", "cam-1"],
            "metric_conversion": "MetricSceneAdapter",
        },
    }


@pytest.fixture
def valid_publication_bundle(tmp_path: Path) -> Path:
    """Create a complete lightweight bundle for inventory/tamper tests."""
    bundle = tmp_path / "publication-bundle"
    bundle.mkdir()
    gif_frames: tuple[NDArray[np.uint8], ...] = tuple(
        np.full((64, 64, 3), color, dtype=np.uint8)
        for color in ((220, 20, 60), (20, 180, 90))
    )
    alignment_frames: tuple[NDArray[np.uint8], ...] = tuple(
        np.full((64, 64, 3), color, dtype=np.uint8)
        for color in ((220, 20, 60), (20, 120, 220), (20, 180, 90), (240, 180, 20))
    )
    for artifact in REQUIRED_PUBLICATION_ARTIFACTS:
        path = bundle / artifact.value
        if artifact is PublicationArtifactName.ALIGNMENT_PROGRESSION:
            write_deterministic_gif(alignment_frames, path, duration_ms=40)
        elif artifact.value.endswith(".gif"):
            write_deterministic_gif(gif_frames, path, duration_ms=40)
        else:
            _write_png(path)

    records = tuple(
        PublicationArtifactRecord(
            file_name=artifact,
            media_type="image/gif" if artifact.value.endswith(".gif") else "image/png",
            width=64,
            height=64,
            frame_count=4
            if artifact is PublicationArtifactName.ALIGNMENT_PROGRESSION
            else (2 if artifact.value.endswith(".gif") else 1),
            duration_ms=40 if artifact.value.endswith(".gif") else None,
            byte_size=(bundle / artifact.value).stat().st_size,
            content_digest_blake2b_256=_content_digest(bundle / artifact.value),
            mapping=_artifact_mapping(artifact),
        )
        for artifact in REQUIRED_PUBLICATION_ARTIFACTS
    )
    manifest = PublicationManifest(
        scene_id="scene-0",
        resolved_config={
            "schema": PUBLICATION_REQUEST_SCHEMA,
            "scene_id": "scene-0",
            "scene_root": ".",
            "output_bundle": ".",
            "artifact_names": [item.value for item in REQUIRED_PUBLICATION_ARTIFACTS],
            "court": {
                "dataset_root": "datasets/court",
                "trajectory_id": "trajectory-0",
                "frame_indices": [0, 2],
            },
            "blcs": {
                "dataset_root": "datasets/blcs",
                "logical_scene_id": "logical-0",
                "camera_id": "cam-0",
                "frame_indices": [0, 2],
                "camera_ids": ["cam-0", "cam-1"],
            },
            "plcs": {
                "dataset_root": "datasets/plcs",
                "logical_scene_id": "logical-0",
                "camera_id": "cam-0",
                "frame_indices": [0, 2],
                "camera_ids": ["cam-0", "cam-1"],
            },
            "captured": {
                "scene_json": "reconstruction/export/scene.json",
                "camera_ids": ["cam-0", "cam-1"],
            },
            "alignment_root": "alignment",
            "drawing": {
                "dataset_size": [64, 64],
                "alignment_size": [64, 64],
                "figure_size": [64, 64],
                "overview_size": [64, 64],
                "gif_duration_ms": 40,
                "frustum_depth_metres": 1.5,
                "line_width": 1.0,
                "font_size": 10,
                "history_frames": 2,
                "maximum_artifact_bytes": 1_000_000,
                "maximum_bundle_bytes": 2_000_000,
            },
        },
        source_owners=_source_owners(),
        artifacts=records,
        coordinate_contract=PUBLICATION_COORDINATE_CONTRACT,
        diagnostic_versions={
            "publication_request": PUBLICATION_REQUEST_SCHEMA,
            "publication_manifest": PUBLICATION_MANIFEST_SCHEMA,
            "publication_bundle": PUBLICATION_BUNDLE_SCHEMA,
            "alignment": "synthetic_alignment_v1",
            "alignment_trace": ALIGNMENT_TRACE_SCHEMA,
            "ground_plane_frame": GROUND_PLANE_FRAME_SCHEMA,
            "alignment_agreement_metrics": ALIGNMENT_AGREEMENT_METRIC_SCHEMA,
            "camera_coverage_metrics": CAMERA_COVERAGE_METRIC_SCHEMA,
            "overview_layout": OVERVIEW_LAYOUT_SCHEMA,
            "gif_encoder": "pillow-gif-fixed-palette-v1",
            "camera_coordinate_convention": METRIC_CAMERA_COORDINATE_CONVENTION,
            "ground_plane_uv_coordinate_convention": GROUND_PLANE_UV_COORDINATE_CONVENTION,
        },
        metrics={
            "alignment": {"schema": ALIGNMENT_AGREEMENT_METRIC_SCHEMA},
            "cameras": {},
        },
        asset_policy={
            "maximum_artifact_bytes": 1_000_000,
            "maximum_bundle_bytes": 2_000_000,
            "artifact_bytes": sum(item.byte_size for item in records),
            "artifact_count": len(records),
        },
    )
    (bundle / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle
