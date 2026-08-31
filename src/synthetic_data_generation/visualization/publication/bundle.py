"""Atomic generation and semantic validation of publication bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.alignment.contracts import (
    ALIGNMENT_SCHEMA,
    ALIGNMENT_TRACE_SCHEMA,
    GROUND_PLANE_FRAME_SCHEMA,
    AlignmentTracePhase,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    GROUND_PLANE_UV_COORDINATE_CONVENTION,
)
from src.synthetic_data_generation.reconstruction.scene_export import NHT_CAMERAS_SCHEMA
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.synthetic_data_generation.visualization.publication.alignment import (
    ALIGNMENT_AGREEMENT_METRIC_SCHEMA,
    AlignmentPublicationData,
    load_alignment_publication_data,
    render_alignment_heatmap_court_png,
    render_alignment_progression_gif,
)
from src.synthetic_data_generation.visualization.publication.cameras import (
    METRIC_CAMERA_COORDINATE_CONVENTION,
    PublicationCameraCollection,
    load_blcs_cameras,
    load_captured_cameras,
    load_plcs_cameras,
)
from src.synthetic_data_generation.visualization.publication.contracts import (
    CAMERA_DRAWING_POLICY_SCHEMA,
    CAPTURED_CAMERA_SELECTION_POLICY,
    PUBLICATION_BUNDLE_SCHEMA,
    PUBLICATION_COORDINATE_CONTRACT,
    PUBLICATION_MANIFEST_SCHEMA,
    PUBLICATION_REQUEST_SCHEMA,
    REQUIRED_PUBLICATION_ARTIFACTS,
    STATIC_RIG_SELECTION_POLICY,
    CameraRenderingSemantics,
    PublicationArtifactName,
    PublicationArtifactRecord,
    PublicationBundleResult,
    PublicationDrawingSettings,
    PublicationManifest,
    PublicationRequest,
)
from src.synthetic_data_generation.visualization.publication.datasets import (
    GIF_ENCODER,
    render_blcs_dataset_gif,
    render_court_dataset_gif,
    render_plcs_dataset_gif,
)
from src.synthetic_data_generation.visualization.publication.figures import (
    CAMERA_COVERAGE_METRIC_SCHEMA,
    CAMERA_RIG_COMPARISON_METRIC_SCHEMA,
    OVERVIEW_LAYOUT_SCHEMA,
    camera_collection_metrics,
    camera_forward_angle_differences_degrees,
    camera_render_indices,
    render_camera_comparison_figure,
    render_camera_figure,
    render_publication_overview,
)
from src.synthetic_data_generation.visualization.sources import (
    BLCSVisualizationSource,
    CourtVisualizationSource,
    PLCSVisualizationSource,
)

MANIFEST_FILE = "manifest.json"


def generate_publication_bundle(request: PublicationRequest) -> PublicationBundleResult:
    """Validate every owner, render in private staging, validate, then publish once."""
    if not isinstance(request, PublicationRequest):
        raise TypeError("generate_publication_bundle requires PublicationRequest.")
    inputs = _load_inputs(request)
    output = request.output_bundle
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Publication output appeared before generation: {output}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    try:
        manifest = _render_staging_bundle(request, staging=staging, inputs=inputs)
        validated = validate_publication_bundle(staging, expected_request=request)
        if validated.to_dict() != manifest.to_dict():
            raise ValueError(
                "Staging validator changed publication manifest semantics."
            )
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"Publication output appeared during generation: {output}"
            )
        os.rename(staging, output)
    finally:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
    return PublicationBundleResult(
        bundle_path=output,
        manifest_path=output / MANIFEST_FILE,
        manifest=manifest,
    )


def validate_publication_bundle(
    bundle_path: Path,
    *,
    expected_request: PublicationRequest,
) -> PublicationManifest:
    """Validate a bundle against its request and the current canonical sources."""
    if not isinstance(expected_request, PublicationRequest):
        raise TypeError(
            "validate_publication_bundle requires expected_request=PublicationRequest."
        )
    manifest = validate_publication_bundle_structure_only(bundle_path)
    if manifest.scene_id != expected_request.scene_id:
        raise ValueError("Manifest scene_id differs from the expected request.")
    if manifest.resolved_config != expected_request.to_resolved_config():
        raise ValueError("Manifest resolved config differs from the expected request.")
    inputs = _load_inputs(expected_request)
    _validate_authoritative_provenance(
        manifest,
        request=expected_request,
        inputs=inputs,
    )
    return manifest


def validate_publication_bundle_structure_only(
    bundle_path: Path,
) -> PublicationManifest:
    """Check bundle-local structure without authenticating claimed source provenance."""
    if bundle_path.is_symlink() or not bundle_path.is_dir():
        raise ValueError("Publication bundle must be an ordinary directory.")
    expected_files = {item.value for item in REQUIRED_PUBLICATION_ARTIFACTS} | {
        MANIFEST_FILE
    }
    actual_files = {path.name for path in bundle_path.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            "Publication bundle inventory differs; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}."
        )
    if any(path.is_symlink() or not path.is_file() for path in bundle_path.iterdir()):
        raise ValueError("Publication bundle may contain ordinary files only.")
    manifest_payload = json.loads(
        (bundle_path / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    manifest = PublicationManifest.from_dict(manifest_payload)
    total_bytes = 0
    for record in manifest.artifacts:
        path = bundle_path / record.file_name.value
        size = path.stat().st_size
        total_bytes += size
        if size != record.byte_size:
            raise ValueError(f"Artifact byte size changed: {record.file_name.value}")
        if _content_digest(path) != record.content_digest_blake2b_256:
            raise ValueError(
                f"Artifact content digest changed: {record.file_name.value}"
            )
        with Image.open(path) as image:
            expected_format = "GIF" if record.media_type == "image/gif" else "PNG"
            if image.format != expected_format or image.size != (
                record.width,
                record.height,
            ):
                raise ValueError(
                    f"Artifact media metadata changed: {record.file_name.value}"
                )
            if image.n_frames != record.frame_count:
                raise ValueError(
                    f"Artifact frame count changed: {record.file_name.value}"
                )
            for index in range(image.n_frames):
                image.seek(index)
                image.convert("RGB").load()
                if (
                    record.duration_ms is not None
                    and image.info.get("duration") != record.duration_ms
                ):
                    raise ValueError(
                        f"Artifact GIF timing changed: {record.file_name.value}"
                    )
    _validate_asset_policy(manifest, total_bytes=total_bytes)
    _validate_semantic_provenance(manifest)
    return manifest


class _LoadedInputs:
    def __init__(
        self,
        *,
        alignment: AlignmentPublicationData,
        court_source: CourtVisualizationSource,
        blcs_source: BLCSVisualizationSource,
        plcs_source: PLCSVisualizationSource,
        captured_cameras: PublicationCameraCollection,
        blcs_cameras: PublicationCameraCollection,
        plcs_cameras: PublicationCameraCollection,
    ) -> None:
        self.alignment = alignment
        self.court_source = court_source
        self.blcs_source = blcs_source
        self.plcs_source = plcs_source
        self.captured_cameras = captured_cameras
        self.blcs_cameras = blcs_cameras
        self.plcs_cameras = plcs_cameras


def _load_inputs(request: PublicationRequest) -> _LoadedInputs:
    alignment = load_alignment_publication_data(request.alignment_root)
    court_source = CourtVisualizationSource(
        request.dataset_root("court"), trajectory_id=request.court_trajectory_id
    )
    blcs_source = BLCSVisualizationSource(
        request.dataset_root("blcs"),
        logical_scene_id=request.blcs_logical_scene_id,
        camera_id=request.blcs_camera_id,
    )
    plcs_source = PLCSVisualizationSource(
        request.dataset_root("plcs"),
        logical_scene_id=request.plcs_logical_scene_id,
        camera_id=request.plcs_camera_id,
    )
    for source, domain in (
        (court_source, "court"),
        (blcs_source, "blcs"),
        (plcs_source, "plcs"),
    ):
        if source.dataset_scene_id != request.scene_id:
            raise ValueError(f"{domain} dataset belongs to a foreign scene.")
    captured_cameras = load_captured_cameras(
        request.reconstruction_scene_json,
        scene_id=request.scene_id,
        camera_ids=request.captured_camera_ids,
        metric_adapter=alignment.result.metric_adapter,
    )
    blcs_cameras = load_blcs_cameras(
        request.dataset_root("blcs"),
        scene_id=request.scene_id,
        logical_scene_id=request.blcs_logical_scene_id,
        camera_ids=request.blcs_camera_ids,
    )
    plcs_cameras = load_plcs_cameras(
        request.dataset_root("plcs"),
        scene_id=request.scene_id,
        logical_scene_id=request.plcs_logical_scene_id,
        camera_ids=request.plcs_camera_ids,
    )
    return _LoadedInputs(
        alignment=alignment,
        court_source=court_source,
        blcs_source=blcs_source,
        plcs_source=plcs_source,
        captured_cameras=captured_cameras,
        blcs_cameras=blcs_cameras,
        plcs_cameras=plcs_cameras,
    )


def _render_staging_bundle(
    request: PublicationRequest,
    *,
    staging: Path,
    inputs: _LoadedInputs,
) -> PublicationManifest:
    alignment = inputs.alignment
    drawing = request.drawing
    court_gif = render_court_dataset_gif(
        inputs.court_source,
        staging / PublicationArtifactName.DATASET_COURT.value,
        trajectory_id=request.court_trajectory_id,
        frame_indices=request.court_frame_indices,
        size=drawing.dataset_size,
        duration_ms=drawing.gif_duration_ms,
    )
    blcs_gif = render_blcs_dataset_gif(
        inputs.blcs_source,
        staging / PublicationArtifactName.DATASET_BLCS.value,
        logical_scene_id=request.blcs_logical_scene_id,
        camera_id=request.blcs_camera_id,
        frame_indices=request.blcs_frame_indices,
        size=drawing.dataset_size,
        duration_ms=drawing.gif_duration_ms,
        history_frames=drawing.history_frames,
    )
    plcs_gif = render_plcs_dataset_gif(
        inputs.plcs_source,
        staging / PublicationArtifactName.DATASET_PLCS.value,
        logical_scene_id=request.plcs_logical_scene_id,
        camera_id=request.plcs_camera_id,
        frame_indices=request.plcs_frame_indices,
        size=drawing.dataset_size,
        duration_ms=drawing.gif_duration_ms,
    )
    alignment_mapping = render_alignment_progression_gif(
        alignment,
        staging / PublicationArtifactName.ALIGNMENT_PROGRESSION.value,
        size=drawing.alignment_size,
        duration_ms=drawing.gif_duration_ms,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
    )
    render_alignment_heatmap_court_png(
        alignment,
        staging / PublicationArtifactName.ALIGNMENT_HEATMAP_COURT.value,
        size=drawing.figure_size,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
    )
    layout = alignment.result.layout
    captured_render_indices = camera_render_indices(
        len(inputs.captured_cameras.camera_ids),
        maximum_rendered_cameras=drawing.maximum_rendered_captured_cameras,
    )
    blcs_render_indices = tuple(range(len(inputs.blcs_cameras.camera_ids)))
    plcs_render_indices = tuple(range(len(inputs.plcs_cameras.camera_ids)))
    render_camera_figure(
        inputs.captured_cameras,
        layout,
        staging / PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY.value,
        size=drawing.figure_size,
        frustum_depth_metres=drawing.frustum_depth_metres,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
        rendering_semantics=CameraRenderingSemantics.CAPTURED_TRAJECTORY,
        rendered_camera_indices=captured_render_indices,
    )
    render_camera_figure(
        inputs.blcs_cameras,
        layout,
        staging / PublicationArtifactName.BLCS_CAMERA_LAYOUT.value,
        size=drawing.figure_size,
        frustum_depth_metres=drawing.frustum_depth_metres,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
        rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
        rendered_camera_indices=blcs_render_indices,
    )
    render_camera_figure(
        inputs.plcs_cameras,
        layout,
        staging / PublicationArtifactName.PLCS_CAMERA_LAYOUT.value,
        size=drawing.figure_size,
        frustum_depth_metres=drawing.frustum_depth_metres,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
        rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
        rendered_camera_indices=plcs_render_indices,
    )
    comparison_metrics = render_camera_comparison_figure(
        inputs.blcs_cameras,
        inputs.plcs_cameras,
        layout,
        staging / PublicationArtifactName.CAMERA_LAYOUT_COMPARISON.value,
        size=drawing.figure_size,
        frustum_depth_metres=drawing.frustum_depth_metres,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
        centre_tolerance_metres=drawing.coincident_centre_tolerance_metres,
        forward_angle_tolerance_degrees=(
            drawing.coincident_forward_angle_tolerance_degrees
        ),
    )
    camera_metrics = {
        "reconstruction": camera_collection_metrics(
            inputs.captured_cameras,
            rendering_semantics=CameraRenderingSemantics.CAPTURED_TRAJECTORY,
        ),
        "blcs": camera_collection_metrics(
            inputs.blcs_cameras,
            rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
        ),
        "plcs": camera_collection_metrics(
            inputs.plcs_cameras,
            rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
        ),
        "comparison": comparison_metrics,
    }
    overview_mapping = render_publication_overview(
        staging,
        staging / PublicationArtifactName.PUBLICATION_OVERVIEW.value,
        size=drawing.overview_size,
        scene_id=request.scene_id,
        alignment_metrics=alignment.metrics,
        camera_metrics=camera_metrics,
        font_size=drawing.font_size,
    )
    mapping_by_artifact: Mapping[
        PublicationArtifactName, tuple[Mapping[str, object], ...]
    ] = {
        PublicationArtifactName.DATASET_COURT: court_gif.mapping,
        PublicationArtifactName.DATASET_BLCS: blcs_gif.mapping,
        PublicationArtifactName.DATASET_PLCS: plcs_gif.mapping,
        PublicationArtifactName.ALIGNMENT_PROGRESSION: alignment_mapping,
        PublicationArtifactName.ALIGNMENT_HEATMAP_COURT: (
            {
                "candidate_ids": list(
                    alignment.evidence.alignment_trace.final_candidate_ids
                ),
                "court_instance_ids": [
                    item.court_instance_id for item in layout.courts
                ],
                "heatmap_camera_ids": list(alignment.heatmaps.camera_ids),
            },
        ),
        PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY: _camera_mapping(
            inputs.captured_cameras,
            rendering_semantics=CameraRenderingSemantics.CAPTURED_TRAJECTORY,
            rendered_camera_indices=captured_render_indices,
        ),
        PublicationArtifactName.BLCS_CAMERA_LAYOUT: _camera_mapping(
            inputs.blcs_cameras,
            rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
            rendered_camera_indices=blcs_render_indices,
        ),
        PublicationArtifactName.PLCS_CAMERA_LAYOUT: _camera_mapping(
            inputs.plcs_cameras,
            rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
            rendered_camera_indices=plcs_render_indices,
        ),
        PublicationArtifactName.CAMERA_LAYOUT_COMPARISON: _camera_comparison_mapping(
            blcs_mapping=_camera_mapping(
                inputs.blcs_cameras,
                rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
                rendered_camera_indices=blcs_render_indices,
            ),
            plcs_mapping=_camera_mapping(
                inputs.plcs_cameras,
                rendering_semantics=CameraRenderingSemantics.STATIC_RIG,
                rendered_camera_indices=plcs_render_indices,
            ),
            comparison_metrics=comparison_metrics,
        ),
        PublicationArtifactName.PUBLICATION_OVERVIEW: overview_mapping,
    }
    records = tuple(
        _artifact_record(
            staging / artifact.value,
            artifact=artifact,
            duration_ms=(
                drawing.gif_duration_ms if artifact.value.endswith(".gif") else None
            ),
            mapping=mapping_by_artifact[artifact],
        )
        for artifact in REQUIRED_PUBLICATION_ARTIFACTS
    )
    source_owners = _source_owner_manifest(request, inputs=inputs)
    total_bytes = sum(item.byte_size for item in records)
    manifest = PublicationManifest(
        scene_id=request.scene_id,
        resolved_config=request.to_resolved_config(),
        source_owners=source_owners,
        artifacts=records,
        coordinate_contract=PUBLICATION_COORDINATE_CONTRACT,
        diagnostic_versions={
            "publication_request": PUBLICATION_REQUEST_SCHEMA,
            "publication_manifest": PUBLICATION_MANIFEST_SCHEMA,
            "publication_bundle": PUBLICATION_BUNDLE_SCHEMA,
            "alignment": ALIGNMENT_SCHEMA,
            "alignment_trace": ALIGNMENT_TRACE_SCHEMA,
            "ground_plane_frame": GROUND_PLANE_FRAME_SCHEMA,
            "alignment_agreement_metrics": ALIGNMENT_AGREEMENT_METRIC_SCHEMA,
            "camera_coverage_metrics": CAMERA_COVERAGE_METRIC_SCHEMA,
            "camera_rig_comparison_metrics": CAMERA_RIG_COMPARISON_METRIC_SCHEMA,
            "camera_drawing_policy": CAMERA_DRAWING_POLICY_SCHEMA,
            "overview_layout": OVERVIEW_LAYOUT_SCHEMA,
            "gif_encoder": GIF_ENCODER,
            "camera_coordinate_convention": METRIC_CAMERA_COORDINATE_CONVENTION,
            "ground_plane_uv_coordinate_convention": (
                GROUND_PLANE_UV_COORDINATE_CONVENTION
            ),
        },
        metrics={"alignment": dict(alignment.metrics), "cameras": camera_metrics},
        asset_policy={
            "maximum_artifact_bytes": drawing.maximum_artifact_bytes,
            "maximum_bundle_bytes": drawing.maximum_bundle_bytes,
            "artifact_bytes": total_bytes,
            "artifact_count": len(records),
        },
    )
    _validate_asset_policy(manifest, total_bytes=total_bytes)
    (staging / MANIFEST_FILE).write_text(
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _source_owner_manifest(
    request: PublicationRequest,
    *,
    inputs: _LoadedInputs,
) -> Mapping[str, object]:
    alignment = inputs.alignment
    court_source = inputs.court_source
    blcs_source = inputs.blcs_source
    plcs_source = inputs.plcs_source
    captured = inputs.captured_cameras
    blcs = inputs.blcs_cameras
    plcs = inputs.plcs_cameras
    return {
        "court": {
            "owner_path": "datasets/court",
            "schema": court_source.dataset_schema,
            "scene_id": court_source.dataset_scene_id,
            "domain": "court",
            "trajectory_id": request.court_trajectory_id,
            "source_count": court_source.frame_count,
            "source_fps": None,
            "source_size": [court_source.width, court_source.height],
            "output_size": list(request.drawing.dataset_size),
            "resize_filter": "Pillow LANCZOS",
            "selected_indices": list(request.court_frame_indices),
        },
        "blcs": {
            "owner_path": "datasets/blcs",
            "schema": blcs_source.dataset_schema,
            "scene_id": blcs_source.dataset_scene_id,
            "domain": "blcs",
            "logical_scene_id": request.blcs_logical_scene_id,
            "gif_camera_id": request.blcs_camera_id,
            "camera_ids": list(blcs.camera_ids),
            "camera_rendering_semantics": CameraRenderingSemantics.STATIC_RIG.value,
            "source_count": blcs_source.frame_count,
            "source_fps": blcs_source.source_fps,
            "source_size": [blcs_source.width, blcs_source.height],
            "output_size": list(request.drawing.dataset_size),
            "resize_filter": "Pillow LANCZOS",
            "selected_indices": list(request.blcs_frame_indices),
        },
        "plcs": {
            "owner_path": "datasets/plcs",
            "schema": plcs_source.dataset_schema,
            "scene_id": plcs_source.dataset_scene_id,
            "domain": "plcs",
            "logical_scene_id": request.plcs_logical_scene_id,
            "gif_camera_id": request.plcs_camera_id,
            "camera_ids": list(plcs.camera_ids),
            "camera_rendering_semantics": CameraRenderingSemantics.STATIC_RIG.value,
            "source_count": plcs_source.frame_count,
            "source_fps": None,
            "source_size": [plcs_source.width, plcs_source.height],
            "output_size": list(request.drawing.dataset_size),
            "resize_filter": "Pillow LANCZOS",
            "selected_indices": list(request.plcs_frame_indices),
        },
        "alignment": {
            "owner_path": "alignment",
            "schema": ALIGNMENT_SCHEMA,
            "scene_id": request.scene_id,
            "ground_plane_frame_schema": GROUND_PLANE_FRAME_SCHEMA,
            "alignment_trace_schema": ALIGNMENT_TRACE_SCHEMA,
            "candidate_ids": list(
                alignment.evidence.alignment_trace.final_candidate_ids
            ),
            "court_instance_ids": [
                item.court_instance_id for item in alignment.result.layout.courts
            ],
            "heatmap_camera_ids": list(alignment.heatmaps.camera_ids),
        },
        "reconstruction": {
            "owner_path": "reconstruction/export",
            "schema": NHT_CAMERAS_SCHEMA,
            "scene_id": captured.scene_id,
            "camera_ids": list(captured.camera_ids),
            "camera_rendering_semantics": (
                CameraRenderingSemantics.CAPTURED_TRAJECTORY.value
            ),
            "metric_conversion": "MetricSceneAdapter",
        },
    }


def _validate_authoritative_provenance(
    manifest: PublicationManifest,
    *,
    request: PublicationRequest,
    inputs: _LoadedInputs,
) -> None:
    expected_owners = _source_owner_manifest(request, inputs=inputs)
    if manifest.source_owners != expected_owners:
        raise ValueError(
            "Manifest source owners differ from the validated request sources."
        )

    expected_dataset_mappings = {
        PublicationArtifactName.DATASET_COURT: _dataset_source_mapping(
            inputs.court_source.frame_order,
            frame_indices=request.court_frame_indices,
        ),
        PublicationArtifactName.DATASET_BLCS: _dataset_source_mapping(
            inputs.blcs_source.frame_order,
            frame_indices=request.blcs_frame_indices,
            identity={
                "logical_scene_id": request.blcs_logical_scene_id,
                "camera_id": request.blcs_camera_id,
            },
        ),
        PublicationArtifactName.DATASET_PLCS: _dataset_source_mapping(
            inputs.plcs_source.frame_order,
            frame_indices=request.plcs_frame_indices,
            identity={
                "logical_scene_id": request.plcs_logical_scene_id,
                "camera_id": request.plcs_camera_id,
            },
        ),
    }
    records = {record.file_name: record for record in manifest.artifacts}
    for artifact, expected_mapping in expected_dataset_mappings.items():
        if records[artifact].mapping != expected_mapping:
            raise ValueError(
                f"{artifact.value} mapping differs from the validated dataset source."
            )

    camera_sources = {
        PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY: (
            inputs.captured_cameras,
            CameraRenderingSemantics.CAPTURED_TRAJECTORY,
            camera_render_indices(
                len(inputs.captured_cameras.camera_ids),
                maximum_rendered_cameras=(
                    request.drawing.maximum_rendered_captured_cameras
                ),
            ),
        ),
        PublicationArtifactName.BLCS_CAMERA_LAYOUT: (
            inputs.blcs_cameras,
            CameraRenderingSemantics.STATIC_RIG,
            tuple(range(len(inputs.blcs_cameras.camera_ids))),
        ),
        PublicationArtifactName.PLCS_CAMERA_LAYOUT: (
            inputs.plcs_cameras,
            CameraRenderingSemantics.STATIC_RIG,
            tuple(range(len(inputs.plcs_cameras.camera_ids))),
        ),
    }
    for artifact, (collection, semantics, rendered_indices) in camera_sources.items():
        expected_mapping = _camera_mapping(
            collection,
            rendering_semantics=semantics,
            rendered_camera_indices=rendered_indices,
        )
        if records[artifact].mapping != expected_mapping:
            raise ValueError(
                f"{artifact.value} mapping differs from the validated camera source."
            )


def _dataset_source_mapping(
    frame_order: tuple[Mapping[str, object], ...],
    *,
    frame_indices: tuple[int, ...],
    identity: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, object], ...]:
    suffix = {} if identity is None else dict(identity)
    return tuple(
        {"source_index": index, **dict(frame_order[index]), **suffix}
        for index in frame_indices
    )


def _camera_mapping(
    collection: PublicationCameraCollection,
    *,
    rendering_semantics: CameraRenderingSemantics,
    rendered_camera_indices: tuple[int, ...],
) -> tuple[Mapping[str, object], ...]:
    rendered_camera_ids = tuple(
        collection.camera_ids[index] for index in rendered_camera_indices
    )
    selection_policy = (
        CAPTURED_CAMERA_SELECTION_POLICY
        if rendering_semantics is CameraRenderingSemantics.CAPTURED_TRAJECTORY
        else STATIC_RIG_SELECTION_POLICY
    )
    summary: Mapping[str, object] = {
        "mapping_type": "camera_rendering_policy",
        "owner": collection.owner,
        "logical_scene_id": collection.logical_scene_id,
        "rendering_semantics": rendering_semantics.value,
        "drawing_policy_schema": CAMERA_DRAWING_POLICY_SCHEMA,
        "selection_policy": selection_policy,
        "camera_count": len(collection.camera_ids),
        "rendered_camera_count": len(rendered_camera_indices),
        "rendered_camera_indices": list(rendered_camera_indices),
        "rendered_camera_ids": list(rendered_camera_ids),
    }
    poses = tuple(
        {
            "mapping_type": "camera_pose",
            "owner": collection.owner,
            "logical_scene_id": collection.logical_scene_id,
            "camera_index": index,
            "camera_id": camera.camera_id,
            "source_frame_index": camera.source_frame_index,
            "width": camera.width,
            "height": camera.height,
            "intrinsics": list(camera.intrinsics),
            "camera_to_metric_scene": [
                [float(value) for value in row]
                for row in collection.camera_to_metric_scene[index]
            ],
            "image_path": camera.image_path,
        }
        for index, camera in enumerate(collection.cameras)
    )
    return (summary, *poses)


def _camera_comparison_mapping(
    *,
    blcs_mapping: tuple[Mapping[str, object], ...],
    plcs_mapping: tuple[Mapping[str, object], ...],
    comparison_metrics: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    blcs_summary = blcs_mapping[0]
    plcs_summary = plcs_mapping[0]
    return (
        {
            "mapping_type": "camera_rig_comparison",
            "rendering_semantics": CameraRenderingSemantics.STATIC_RIG.value,
            "pose_matching": "strict_ordered_camera_id",
            "blcs_camera_ids": blcs_summary["rendered_camera_ids"],
            "plcs_camera_ids": plcs_summary["rendered_camera_ids"],
            "comparison_metrics": dict(comparison_metrics),
        },
        *blcs_mapping[1:],
        *plcs_mapping[1:],
    )


def _artifact_record(
    path: Path,
    *,
    artifact: PublicationArtifactName,
    duration_ms: int | None,
    mapping: tuple[Mapping[str, object], ...],
) -> PublicationArtifactRecord:
    with Image.open(path) as image:
        width, height = image.size
        frame_count = image.n_frames
    return PublicationArtifactRecord(
        file_name=artifact,
        media_type="image/gif" if artifact.value.endswith(".gif") else "image/png",
        width=width,
        height=height,
        frame_count=frame_count,
        duration_ms=duration_ms,
        byte_size=path.stat().st_size,
        content_digest_blake2b_256=_content_digest(path),
        mapping=mapping,
    )


def _content_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_asset_policy(manifest: PublicationManifest, *, total_bytes: int) -> None:
    policy = _exact_mapping(
        manifest.asset_policy,
        name="asset_policy",
        keys={
            "maximum_artifact_bytes",
            "maximum_bundle_bytes",
            "artifact_bytes",
            "artifact_count",
        },
    )
    maximum_artifact = _positive_integer(
        policy["maximum_artifact_bytes"], name="maximum_artifact_bytes"
    )
    maximum_bundle = _positive_integer(
        policy["maximum_bundle_bytes"], name="maximum_bundle_bytes"
    )
    if policy["artifact_count"] != len(REQUIRED_PUBLICATION_ARTIFACTS):
        raise ValueError("Asset policy artifact_count differs from fixed inventory.")
    if policy["artifact_bytes"] != total_bytes:
        raise ValueError("Asset policy artifact_bytes differs from measured media.")
    if any(item.byte_size > maximum_artifact for item in manifest.artifacts):
        raise ValueError("A publication artifact exceeds the configured byte limit.")
    if total_bytes > maximum_bundle:
        raise ValueError(
            "Publication artifacts exceed the configured bundle byte limit."
        )


def _validate_semantic_provenance(manifest: PublicationManifest) -> None:
    resolved_config = _exact_mapping(
        manifest.resolved_config,
        name="resolved_config",
        keys={
            "schema",
            "scene_id",
            "scene_root",
            "output_bundle",
            "artifact_names",
            "court",
            "blcs",
            "plcs",
            "captured",
            "alignment_root",
            "drawing",
        },
    )
    if (
        resolved_config["schema"] != PUBLICATION_REQUEST_SCHEMA
        or resolved_config["scene_id"] != manifest.scene_id
        or tuple(
            _sequence(
                resolved_config["artifact_names"],
                name="resolved_config.artifact_names",
            )
        )
        != tuple(item.value for item in REQUIRED_PUBLICATION_ARTIFACTS)
    ):
        raise ValueError("Resolved config request schema is missing or foreign.")
    drawing = _drawing_settings_from_manifest(resolved_config["drawing"])
    configured_cameras = {
        "reconstruction": _exact_mapping(
            resolved_config["captured"],
            name="resolved_config.captured",
            keys={"scene_json", "camera_ids"},
        ),
        "blcs": _exact_mapping(
            resolved_config["blcs"],
            name="resolved_config.blcs",
            keys={
                "dataset_root",
                "logical_scene_id",
                "camera_id",
                "frame_indices",
                "camera_ids",
            },
        ),
        "plcs": _exact_mapping(
            resolved_config["plcs"],
            name="resolved_config.plcs",
            keys={
                "dataset_root",
                "logical_scene_id",
                "camera_id",
                "frame_indices",
                "camera_ids",
            },
        ),
    }
    owner_keys = {
        "court": {
            "owner_path",
            "schema",
            "scene_id",
            "domain",
            "trajectory_id",
            "source_count",
            "source_fps",
            "source_size",
            "output_size",
            "resize_filter",
            "selected_indices",
        },
        "blcs": {
            "owner_path",
            "schema",
            "scene_id",
            "domain",
            "logical_scene_id",
            "gif_camera_id",
            "camera_ids",
            "camera_rendering_semantics",
            "source_count",
            "source_fps",
            "source_size",
            "output_size",
            "resize_filter",
            "selected_indices",
        },
        "plcs": {
            "owner_path",
            "schema",
            "scene_id",
            "domain",
            "logical_scene_id",
            "gif_camera_id",
            "camera_ids",
            "camera_rendering_semantics",
            "source_count",
            "source_fps",
            "source_size",
            "output_size",
            "resize_filter",
            "selected_indices",
        },
        "alignment": {
            "owner_path",
            "schema",
            "scene_id",
            "ground_plane_frame_schema",
            "alignment_trace_schema",
            "candidate_ids",
            "court_instance_ids",
            "heatmap_camera_ids",
        },
        "reconstruction": {
            "owner_path",
            "schema",
            "scene_id",
            "camera_ids",
            "camera_rendering_semantics",
            "metric_conversion",
        },
    }
    for name, keys in owner_keys.items():
        owner = _exact_mapping(
            manifest.source_owners[name], name=f"source_owners.{name}", keys=keys
        )
        if owner["scene_id"] != manifest.scene_id:
            raise ValueError("Source owner contains a foreign scene identity.")
    for owner_name, expected_semantics in (
        ("reconstruction", CameraRenderingSemantics.CAPTURED_TRAJECTORY),
        ("blcs", CameraRenderingSemantics.STATIC_RIG),
        ("plcs", CameraRenderingSemantics.STATIC_RIG),
    ):
        owner = cast(Mapping[str, object], manifest.source_owners[owner_name])
        if owner["camera_rendering_semantics"] != expected_semantics.value:
            raise ValueError(
                f"{owner_name} camera rendering semantics are missing or incorrect."
            )
        if tuple(
            _sequence(owner["camera_ids"], name=f"{owner_name}.camera_ids")
        ) != tuple(
            _sequence(
                configured_cameras[owner_name]["camera_ids"],
                name=f"resolved_config.{owner_name}.camera_ids",
            )
        ):
            raise ValueError(
                f"{owner_name} owner camera IDs differ from resolved configuration."
            )
        if (
            owner_name != "reconstruction"
            and owner["logical_scene_id"]
            != (configured_cameras[owner_name]["logical_scene_id"])
        ):
            raise ValueError(
                f"{owner_name} logical scene differs from resolved configuration."
            )
    for domain, artifact in (
        ("court", PublicationArtifactName.DATASET_COURT),
        ("blcs", PublicationArtifactName.DATASET_BLCS),
        ("plcs", PublicationArtifactName.DATASET_PLCS),
    ):
        owner = cast(Mapping[str, object], manifest.source_owners[domain])
        indices = tuple(
            _nonnegative_integer(value, name=f"{domain}.selected_indices")
            for value in _sequence(
                owner["selected_indices"], name=f"{domain}.selected_indices"
            )
        )
        source_count = _positive_integer(
            owner["source_count"], name=f"{domain}.source_count"
        )
        if not indices or indices[0] != 0 or indices[-1] != source_count - 1:
            raise ValueError("Dataset provenance selection is not endpoint-inclusive.")
        record = next(item for item in manifest.artifacts if item.file_name is artifact)
        output_size = tuple(
            _positive_integer(value, name=f"{domain}.output_size")
            for value in _sequence(owner["output_size"], name=f"{domain}.output_size")
        )
        source_size = tuple(
            _positive_integer(value, name=f"{domain}.source_size")
            for value in _sequence(owner["source_size"], name=f"{domain}.source_size")
        )
        if len(output_size) != 2 or output_size != (record.width, record.height):
            raise ValueError(
                "Dataset artifact dimensions differ from owner provenance."
            )
        if len(source_size) != 2 or owner["resize_filter"] != "Pillow LANCZOS":
            raise ValueError("Dataset source dimension/resize provenance is invalid.")
        source_fps = owner["source_fps"]
        if source_fps is not None and (
            isinstance(source_fps, bool)
            or not isinstance(source_fps, (int, float))
            or float(source_fps) <= 0.0
        ):
            raise ValueError("Dataset source_fps provenance must be null or positive.")
        mapped = tuple(
            _nonnegative_integer(item.get("source_index"), name="mapping.source_index")
            for item in record.mapping
        )
        if mapped != indices:
            raise ValueError(
                "Dataset artifact mapping differs from its owner selection."
            )
    alignment_record = next(
        item
        for item in manifest.artifacts
        if item.file_name is PublicationArtifactName.ALIGNMENT_PROGRESSION
    )
    expected_phases = tuple(phase.value for phase in AlignmentTracePhase)
    observed_phases = tuple(item.get("phase") for item in alignment_record.mapping)
    if observed_phases != expected_phases:
        raise ValueError("Alignment progression phases are missing or reordered.")
    alignment_owner = cast(Mapping[str, object], manifest.source_owners["alignment"])
    heatmap_record = next(
        item
        for item in manifest.artifacts
        if item.file_name is PublicationArtifactName.ALIGNMENT_HEATMAP_COURT
    )
    if len(heatmap_record.mapping) != 1 or heatmap_record.mapping[0] != {
        "candidate_ids": alignment_owner["candidate_ids"],
        "court_instance_ids": alignment_owner["court_instance_ids"],
        "heatmap_camera_ids": alignment_owner["heatmap_camera_ids"],
    }:
        raise ValueError("Alignment heatmap mapping differs from alignment provenance.")
    camera_artifacts = {
        "reconstruction": PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY,
        "blcs": PublicationArtifactName.BLCS_CAMERA_LAYOUT,
        "plcs": PublicationArtifactName.PLCS_CAMERA_LAYOUT,
    }
    camera_pose_matrices: dict[str, NDArray[np.float64]] = {}
    camera_pose_mappings: dict[str, tuple[Mapping[str, object], ...]] = {}
    expected_camera_metrics: dict[str, Mapping[str, object]] = {}
    for owner_name, artifact in camera_artifacts.items():
        owner = cast(Mapping[str, object], manifest.source_owners[owner_name])
        record = next(item for item in manifest.artifacts if item.file_name is artifact)
        semantics = (
            CameraRenderingSemantics.CAPTURED_TRAJECTORY
            if owner_name == "reconstruction"
            else CameraRenderingSemantics.STATIC_RIG
        )
        poses, matrices = _validate_camera_artifact_mapping(
            record.mapping,
            owner_name=owner_name,
            owner=owner,
            rendering_semantics=semantics,
            maximum_rendered_captured_cameras=(
                drawing.maximum_rendered_captured_cameras
            ),
        )
        camera_pose_mappings[owner_name] = poses
        camera_pose_matrices[owner_name] = matrices
        expected_camera_metrics[owner_name] = _camera_metrics_from_poses(
            owner_name=owner_name,
            rendering_semantics=semantics,
            matrices=matrices,
        )
    comparison_record = next(
        item
        for item in manifest.artifacts
        if item.file_name is PublicationArtifactName.CAMERA_LAYOUT_COMPARISON
    )
    blcs_ids = tuple(
        _nonempty_text(value, name="blcs.camera_ids")
        for value in _sequence(
            cast(Mapping[str, object], manifest.source_owners["blcs"])["camera_ids"],
            name="blcs.camera_ids",
        )
    )
    plcs_ids = tuple(
        _nonempty_text(value, name="plcs.camera_ids")
        for value in _sequence(
            cast(Mapping[str, object], manifest.source_owners["plcs"])["camera_ids"],
            name="plcs.camera_ids",
        )
    )
    if blcs_ids != plcs_ids:
        raise ValueError("BLCS/PLCS comparison requires identical ordered camera IDs.")
    expected_comparison_metrics = _camera_comparison_metrics_from_poses(
        camera_ids=blcs_ids,
        blcs_matrices=camera_pose_matrices["blcs"],
        plcs_matrices=camera_pose_matrices["plcs"],
        centre_tolerance_metres=drawing.coincident_centre_tolerance_metres,
        forward_angle_tolerance_degrees=(
            drawing.coincident_forward_angle_tolerance_degrees
        ),
    )
    expected_camera_metrics["comparison"] = expected_comparison_metrics
    _validate_camera_comparison_mapping(
        comparison_record.mapping,
        blcs_ids=blcs_ids,
        plcs_ids=plcs_ids,
        blcs_poses=camera_pose_mappings["blcs"],
        plcs_poses=camera_pose_mappings["plcs"],
        expected_metrics=expected_comparison_metrics,
    )
    overview_record = next(
        item
        for item in manifest.artifacts
        if item.file_name is PublicationArtifactName.PUBLICATION_OVERVIEW
    )
    expected_panel_sources = (
        PublicationArtifactName.DATASET_COURT.value,
        PublicationArtifactName.DATASET_BLCS.value,
        PublicationArtifactName.DATASET_PLCS.value,
        PublicationArtifactName.ALIGNMENT_HEATMAP_COURT.value,
        PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY.value,
        PublicationArtifactName.CAMERA_LAYOUT_COMPARISON.value,
    )
    if tuple(item.get("source_artifact") for item in overview_record.mapping) != (
        expected_panel_sources
    ):
        raise ValueError("Overview panels are missing or reordered.")
    for panel in overview_record.mapping:
        bounds = tuple(
            _nonnegative_integer(value, name="overview.bounds_pixels")
            for value in _sequence(
                panel.get("bounds_pixels"), name="overview.bounds_pixels"
            )
        )
        if (
            len(bounds) != 4
            or bounds[0] >= bounds[2]
            or bounds[1] >= bounds[3]
            or bounds[2] > overview_record.width
            or bounds[3] > overview_record.height
        ):
            raise ValueError("Overview panel bounds leave the rendered canvas.")
    versions = _exact_mapping(
        manifest.diagnostic_versions,
        name="diagnostic_versions",
        keys={
            "publication_request",
            "publication_manifest",
            "publication_bundle",
            "alignment",
            "alignment_trace",
            "ground_plane_frame",
            "alignment_agreement_metrics",
            "camera_coverage_metrics",
            "camera_rig_comparison_metrics",
            "camera_drawing_policy",
            "overview_layout",
            "gif_encoder",
            "camera_coordinate_convention",
            "ground_plane_uv_coordinate_convention",
        },
    )
    if versions.get("alignment_trace") != ALIGNMENT_TRACE_SCHEMA:
        raise ValueError("Manifest alignment trace diagnostic version is missing.")
    if versions.get("ground_plane_frame") != GROUND_PLANE_FRAME_SCHEMA:
        raise ValueError("Manifest ground-plane diagnostic version is missing.")
    if versions.get("camera_coverage_metrics") != CAMERA_COVERAGE_METRIC_SCHEMA:
        raise ValueError("Manifest camera coverage diagnostic version is missing.")
    if (
        versions.get("camera_rig_comparison_metrics")
        != CAMERA_RIG_COMPARISON_METRIC_SCHEMA
    ):
        raise ValueError("Manifest camera comparison diagnostic version is missing.")
    if versions.get("camera_drawing_policy") != CAMERA_DRAWING_POLICY_SCHEMA:
        raise ValueError("Manifest camera drawing policy version is missing.")
    metrics = _exact_mapping(
        manifest.metrics,
        name="metrics",
        keys={"alignment", "cameras"},
    )
    alignment_metrics = _mapping(metrics.get("alignment"), name="metrics.alignment")
    if alignment_metrics.get("schema") != ALIGNMENT_AGREEMENT_METRIC_SCHEMA:
        raise ValueError("Manifest alignment metric schema is missing or foreign.")
    cameras_metrics = _exact_mapping(
        metrics.get("cameras"),
        name="metrics.cameras",
        keys={"reconstruction", "blcs", "plcs", "comparison"},
    )
    if cameras_metrics != expected_camera_metrics:
        raise ValueError(
            "Camera metrics differ from full pose mappings or rendering semantics."
        )
    if manifest.coordinate_contract != PUBLICATION_COORDINATE_CONTRACT:
        raise ValueError(
            "Manifest coordinate contract differs from publication authority."
        )
    if (
        versions.get("camera_coordinate_convention")
        != METRIC_CAMERA_COORDINATE_CONVENTION
    ):
        raise ValueError("Manifest camera coordinate convention is missing.")
    if (
        versions.get("ground_plane_uv_coordinate_convention")
        != GROUND_PLANE_UV_COORDINATE_CONVENTION
    ):
        raise ValueError("Manifest ground-plane UV convention is missing.")


def _drawing_settings_from_manifest(value: object) -> PublicationDrawingSettings:
    raw = _exact_mapping(
        value,
        name="resolved_config.drawing",
        keys={
            "dataset_size",
            "alignment_size",
            "figure_size",
            "overview_size",
            "gif_duration_ms",
            "frustum_depth_metres",
            "line_width",
            "font_size",
            "history_frames",
            "maximum_rendered_captured_cameras",
            "coincident_centre_tolerance_metres",
            "coincident_forward_angle_tolerance_degrees",
            "maximum_artifact_bytes",
            "maximum_bundle_bytes",
        },
    )
    return PublicationDrawingSettings(
        dataset_size=_integer_pair(raw["dataset_size"], name="drawing.dataset_size"),
        alignment_size=_integer_pair(
            raw["alignment_size"], name="drawing.alignment_size"
        ),
        figure_size=_integer_pair(raw["figure_size"], name="drawing.figure_size"),
        overview_size=_integer_pair(raw["overview_size"], name="drawing.overview_size"),
        gif_duration_ms=_integer_value(
            raw["gif_duration_ms"], name="drawing.gif_duration_ms"
        ),
        frustum_depth_metres=_finite_number(
            raw["frustum_depth_metres"], name="drawing.frustum_depth_metres"
        ),
        line_width=_finite_number(raw["line_width"], name="drawing.line_width"),
        font_size=_integer_value(raw["font_size"], name="drawing.font_size"),
        history_frames=_integer_value(
            raw["history_frames"], name="drawing.history_frames"
        ),
        maximum_rendered_captured_cameras=_integer_value(
            raw["maximum_rendered_captured_cameras"],
            name="drawing.maximum_rendered_captured_cameras",
        ),
        coincident_centre_tolerance_metres=_finite_number(
            raw["coincident_centre_tolerance_metres"],
            name="drawing.coincident_centre_tolerance_metres",
        ),
        coincident_forward_angle_tolerance_degrees=_finite_number(
            raw["coincident_forward_angle_tolerance_degrees"],
            name="drawing.coincident_forward_angle_tolerance_degrees",
        ),
        maximum_artifact_bytes=_integer_value(
            raw["maximum_artifact_bytes"], name="drawing.maximum_artifact_bytes"
        ),
        maximum_bundle_bytes=_integer_value(
            raw["maximum_bundle_bytes"], name="drawing.maximum_bundle_bytes"
        ),
    )


def _validate_camera_artifact_mapping(
    mapping: tuple[Mapping[str, object], ...],
    *,
    owner_name: str,
    owner: Mapping[str, object],
    rendering_semantics: CameraRenderingSemantics,
    maximum_rendered_captured_cameras: int,
) -> tuple[tuple[Mapping[str, object], ...], NDArray[np.float64]]:
    camera_ids = tuple(
        _nonempty_text(value, name=f"{owner_name}.camera_ids")
        for value in _sequence(owner["camera_ids"], name=f"{owner_name}.camera_ids")
    )
    if len(mapping) != len(camera_ids) + 1:
        raise ValueError(
            "Camera artifact mapping must retain one policy and every camera pose."
        )
    summary = _exact_mapping(
        mapping[0],
        name=f"{owner_name}.camera_rendering_policy",
        keys={
            "mapping_type",
            "owner",
            "logical_scene_id",
            "rendering_semantics",
            "drawing_policy_schema",
            "selection_policy",
            "camera_count",
            "rendered_camera_count",
            "rendered_camera_indices",
            "rendered_camera_ids",
        },
    )
    expected_logical_scene_id = owner.get("logical_scene_id")
    expected_rendered_indices = (
        camera_render_indices(
            len(camera_ids),
            maximum_rendered_cameras=maximum_rendered_captured_cameras,
        )
        if rendering_semantics is CameraRenderingSemantics.CAPTURED_TRAJECTORY
        else tuple(range(len(camera_ids)))
    )
    expected_rendered_ids = tuple(
        camera_ids[index] for index in expected_rendered_indices
    )
    expected_selection_policy = (
        CAPTURED_CAMERA_SELECTION_POLICY
        if rendering_semantics is CameraRenderingSemantics.CAPTURED_TRAJECTORY
        else STATIC_RIG_SELECTION_POLICY
    )
    rendered_indices = tuple(
        _nonnegative_integer(value, name="rendered_camera_indices")
        for value in _sequence(
            summary["rendered_camera_indices"], name="rendered_camera_indices"
        )
    )
    rendered_ids = tuple(
        _nonempty_text(value, name="rendered_camera_ids")
        for value in _sequence(
            summary["rendered_camera_ids"], name="rendered_camera_ids"
        )
    )
    if (
        summary["mapping_type"] != "camera_rendering_policy"
        or summary["owner"] != owner_name
        or summary["logical_scene_id"] != expected_logical_scene_id
        or summary["rendering_semantics"] != rendering_semantics.value
        or summary["drawing_policy_schema"] != CAMERA_DRAWING_POLICY_SCHEMA
        or summary["selection_policy"] != expected_selection_policy
        or summary["camera_count"] != len(camera_ids)
        or summary["rendered_camera_count"] != len(expected_rendered_indices)
        or rendered_indices != expected_rendered_indices
        or rendered_ids != expected_rendered_ids
    ):
        raise ValueError(
            "Camera rendered indices/IDs differ from the deterministic drawing policy."
        )
    poses = tuple(mapping[1:])
    matrices: list[NDArray[np.float64]] = []
    for index, (camera_id, value) in enumerate(zip(camera_ids, poses, strict=True)):
        pose = _exact_mapping(
            value,
            name=f"{owner_name}.camera_pose",
            keys={
                "mapping_type",
                "owner",
                "logical_scene_id",
                "camera_index",
                "camera_id",
                "source_frame_index",
                "width",
                "height",
                "intrinsics",
                "camera_to_metric_scene",
                "image_path",
            },
        )
        if (
            pose["mapping_type"] != "camera_pose"
            or pose["owner"] != owner_name
            or pose["logical_scene_id"] != expected_logical_scene_id
            or pose["camera_index"] != index
            or pose["camera_id"] != camera_id
        ):
            raise ValueError("Camera pose identity/order differs from its exact owner.")
        _nonnegative_integer(
            pose["source_frame_index"], name="camera_pose.source_frame_index"
        )
        _positive_integer(pose["width"], name="camera_pose.width")
        _positive_integer(pose["height"], name="camera_pose.height")
        intrinsics = tuple(
            _finite_number(item, name="camera_pose.intrinsics")
            for item in _sequence(pose["intrinsics"], name="camera_pose.intrinsics")
        )
        if len(intrinsics) != 9:
            raise ValueError("Camera pose intrinsics must contain exactly 9 values.")
        _nonempty_text(pose["image_path"], name="camera_pose.image_path")
        matrices.append(
            _finite_matrix4(
                pose["camera_to_metric_scene"],
                name="camera_pose.camera_to_metric_scene",
            )
        )
    return poses, np.stack(matrices)


def _camera_metrics_from_poses(
    *,
    owner_name: str,
    rendering_semantics: CameraRenderingSemantics,
    matrices: NDArray[np.float64],
) -> Mapping[str, object]:
    centres = matrices[:, :3, 3]
    metrics: dict[str, object] = {
        "schema": CAMERA_COVERAGE_METRIC_SCHEMA,
        "owner": owner_name,
        "rendering_semantics": rendering_semantics.value,
        "camera_count": len(matrices),
        "centre_bounds_metric_scene": [
            [float(item) for item in np.min(centres, axis=0)],
            [float(item) for item in np.max(centres, axis=0)],
        ],
    }
    if rendering_semantics is CameraRenderingSemantics.CAPTURED_TRAJECTORY:
        displacements = np.linalg.norm(centres[1:] - centres[:-1], axis=1)
        metrics.update(
            {
                "trajectory_segment_count": max(0, len(matrices) - 1),
                "trajectory_length_metres": float(np.sum(displacements)),
                "maximum_adjacent_displacement_metres": (
                    0.0 if len(displacements) == 0 else float(np.max(displacements))
                ),
            }
        )
    return metrics


def _camera_comparison_metrics_from_poses(
    *,
    camera_ids: tuple[str, ...],
    blcs_matrices: NDArray[np.float64],
    plcs_matrices: NDArray[np.float64],
    centre_tolerance_metres: float,
    forward_angle_tolerance_degrees: float,
) -> Mapping[str, object]:
    if blcs_matrices.shape != plcs_matrices.shape or len(blcs_matrices) != len(
        camera_ids
    ):
        raise ValueError("BLCS/PLCS pose mappings differ in camera count.")
    centre_distances = np.linalg.norm(
        blcs_matrices[:, :3, 3] - plcs_matrices[:, :3, 3], axis=1
    )
    blcs_forward = blcs_matrices[:, :3, 2]
    plcs_forward = plcs_matrices[:, :3, 2]
    angles = camera_forward_angle_differences_degrees(blcs_forward, plcs_forward)
    coincident = (centre_distances <= centre_tolerance_metres) & (
        angles <= forward_angle_tolerance_degrees
    )
    coincident_count = int(np.count_nonzero(coincident))
    return {
        "schema": CAMERA_RIG_COMPARISON_METRIC_SCHEMA,
        "pose_matching": "strict_ordered_camera_id",
        "camera_count": len(camera_ids),
        "coincident_camera_count": coincident_count,
        "coincident_camera_fraction": float(coincident_count / len(camera_ids)),
        "maximum_centre_distance_metres": float(np.max(centre_distances)),
        "maximum_forward_angle_difference_degrees": float(np.max(angles)),
        "centre_tolerance_metres": centre_tolerance_metres,
        "forward_angle_tolerance_degrees": forward_angle_tolerance_degrees,
    }


def _validate_camera_comparison_mapping(
    mapping: tuple[Mapping[str, object], ...],
    *,
    blcs_ids: tuple[str, ...],
    plcs_ids: tuple[str, ...],
    blcs_poses: tuple[Mapping[str, object], ...],
    plcs_poses: tuple[Mapping[str, object], ...],
    expected_metrics: Mapping[str, object],
) -> None:
    expected_poses = (*blcs_poses, *plcs_poses)
    if len(mapping) != len(expected_poses) + 1:
        raise ValueError("Camera comparison mapping has incomplete pose provenance.")
    summary = _exact_mapping(
        mapping[0],
        name="camera_comparison.mapping",
        keys={
            "mapping_type",
            "rendering_semantics",
            "pose_matching",
            "blcs_camera_ids",
            "plcs_camera_ids",
            "comparison_metrics",
        },
    )
    if (
        summary["mapping_type"] != "camera_rig_comparison"
        or summary["rendering_semantics"] != CameraRenderingSemantics.STATIC_RIG.value
        or summary["pose_matching"] != "strict_ordered_camera_id"
        or tuple(_sequence(summary["blcs_camera_ids"], name="blcs_camera_ids"))
        != blcs_ids
        or tuple(_sequence(summary["plcs_camera_ids"], name="plcs_camera_ids"))
        != plcs_ids
        or summary["comparison_metrics"] != expected_metrics
        or tuple(mapping[1:]) != expected_poses
    ):
        raise ValueError(
            "Camera comparison mapping differs from strict ordered static-rig poses."
        )


def _finite_matrix4(value: object, *, name: str) -> NDArray[np.float64]:
    rows = _sequence(value, name=name)
    if len(rows) != 4:
        raise ValueError(f"{name} must contain four rows.")
    matrix = np.asarray(
        [
            [
                _finite_number(item, name=name)
                for item in _sequence(row, name=f"{name}.row")
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    RigidTransform.from_matrix(matrix)
    return matrix


def _integer_pair(value: object, *, name: str) -> tuple[int, int]:
    result = tuple(
        _integer_value(item, name=name) for item in _sequence(value, name=name)
    )
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two integers.")
    return result[0], result[1]


def _integer_value(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
    ):
        raise TypeError(f"{name} must be a finite number.")
    return float(value)


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object.")
    return cast(Mapping[str, object], value)


def _exact_mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    if set(result) != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}."
        )
    return result


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


__all__ = [
    "MANIFEST_FILE",
    "generate_publication_bundle",
    "validate_publication_bundle",
    "validate_publication_bundle_structure_only",
]
