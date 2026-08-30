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
    PUBLICATION_BUNDLE_SCHEMA,
    PUBLICATION_COORDINATE_CONTRACT,
    PUBLICATION_MANIFEST_SCHEMA,
    PUBLICATION_REQUEST_SCHEMA,
    REQUIRED_PUBLICATION_ARTIFACTS,
    PublicationArtifactName,
    PublicationArtifactRecord,
    PublicationBundleResult,
    PublicationManifest,
    PublicationRequest,
)
from src.synthetic_data_generation.visualization.publication.datasets import (
    GIF_ENCODER,
    DatasetGifResult,
    render_blcs_dataset_gif,
    render_court_dataset_gif,
    render_plcs_dataset_gif,
)
from src.synthetic_data_generation.visualization.publication.figures import (
    CAMERA_COVERAGE_METRIC_SCHEMA,
    OVERVIEW_LAYOUT_SCHEMA,
    camera_collection_metrics,
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
    expected_request: PublicationRequest | None = None,
) -> PublicationManifest:
    """Reject missing, extra, foreign, mismatched, partial, or tampered bundles."""
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
    if expected_request is not None:
        if manifest.scene_id != expected_request.scene_id:
            raise ValueError("Manifest scene_id differs from the expected request.")
        if manifest.resolved_config != expected_request.to_resolved_config():
            raise ValueError(
                "Manifest resolved config differs from the expected request."
            )
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
    render_camera_figure(
        inputs.captured_cameras,
        layout,
        staging / PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY.value,
        size=drawing.figure_size,
        frustum_depth_metres=drawing.frustum_depth_metres,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
    )
    render_camera_figure(
        inputs.blcs_cameras,
        layout,
        staging / PublicationArtifactName.BLCS_CAMERA_LAYOUT.value,
        size=drawing.figure_size,
        frustum_depth_metres=drawing.frustum_depth_metres,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
    )
    render_camera_figure(
        inputs.plcs_cameras,
        layout,
        staging / PublicationArtifactName.PLCS_CAMERA_LAYOUT.value,
        size=drawing.figure_size,
        frustum_depth_metres=drawing.frustum_depth_metres,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
    )
    render_camera_comparison_figure(
        inputs.blcs_cameras,
        inputs.plcs_cameras,
        layout,
        staging / PublicationArtifactName.CAMERA_LAYOUT_COMPARISON.value,
        size=drawing.figure_size,
        frustum_depth_metres=drawing.frustum_depth_metres,
        line_width=drawing.line_width,
        font_size=drawing.font_size,
    )
    camera_metrics = {
        "reconstruction": camera_collection_metrics(inputs.captured_cameras),
        "blcs": camera_collection_metrics(inputs.blcs_cameras),
        "plcs": camera_collection_metrics(inputs.plcs_cameras),
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
            inputs.captured_cameras
        ),
        PublicationArtifactName.BLCS_CAMERA_LAYOUT: _camera_mapping(
            inputs.blcs_cameras
        ),
        PublicationArtifactName.PLCS_CAMERA_LAYOUT: _camera_mapping(
            inputs.plcs_cameras
        ),
        PublicationArtifactName.CAMERA_LAYOUT_COMPARISON: (
            *(_camera_mapping(inputs.blcs_cameras)),
            *(_camera_mapping(inputs.plcs_cameras)),
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
    source_owners = _source_owner_manifest(
        request,
        alignment=alignment,
        court_gif=court_gif,
        blcs_gif=blcs_gif,
        plcs_gif=plcs_gif,
        captured=inputs.captured_cameras,
        blcs=inputs.blcs_cameras,
        plcs=inputs.plcs_cameras,
    )
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
    alignment: AlignmentPublicationData,
    court_gif: DatasetGifResult,
    blcs_gif: DatasetGifResult,
    plcs_gif: DatasetGifResult,
    captured: PublicationCameraCollection,
    blcs: PublicationCameraCollection,
    plcs: PublicationCameraCollection,
) -> Mapping[str, object]:
    return {
        "court": {
            "owner_path": "datasets/court",
            "schema": court_gif.dataset_schema,
            "scene_id": court_gif.dataset_scene_id,
            "domain": "court",
            "trajectory_id": request.court_trajectory_id,
            "source_count": court_gif.source_count,
            "source_fps": court_gif.source_fps,
            "source_size": [court_gif.source_width, court_gif.source_height],
            "output_size": [court_gif.width, court_gif.height],
            "resize_filter": "Pillow LANCZOS",
            "selected_indices": list(request.court_frame_indices),
        },
        "blcs": {
            "owner_path": "datasets/blcs",
            "schema": blcs_gif.dataset_schema,
            "scene_id": blcs_gif.dataset_scene_id,
            "domain": "blcs",
            "logical_scene_id": request.blcs_logical_scene_id,
            "gif_camera_id": request.blcs_camera_id,
            "camera_ids": list(blcs.camera_ids),
            "source_count": blcs_gif.source_count,
            "source_fps": blcs_gif.source_fps,
            "source_size": [blcs_gif.source_width, blcs_gif.source_height],
            "output_size": [blcs_gif.width, blcs_gif.height],
            "resize_filter": "Pillow LANCZOS",
            "selected_indices": list(request.blcs_frame_indices),
        },
        "plcs": {
            "owner_path": "datasets/plcs",
            "schema": plcs_gif.dataset_schema,
            "scene_id": plcs_gif.dataset_scene_id,
            "domain": "plcs",
            "logical_scene_id": request.plcs_logical_scene_id,
            "gif_camera_id": request.plcs_camera_id,
            "camera_ids": list(plcs.camera_ids),
            "source_count": plcs_gif.source_count,
            "source_fps": plcs_gif.source_fps,
            "source_size": [plcs_gif.source_width, plcs_gif.source_height],
            "output_size": [plcs_gif.width, plcs_gif.height],
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
            "metric_conversion": "MetricSceneAdapter",
        },
    }


def _camera_mapping(
    collection: PublicationCameraCollection,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "owner": collection.owner,
            "logical_scene_id": collection.logical_scene_id,
            "camera_id": camera.camera_id,
            "source_frame_index": camera.source_frame_index,
            "width": camera.width,
            "height": camera.height,
        }
        for camera in collection.cameras
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
            "metric_conversion",
        },
    }
    for name, keys in owner_keys.items():
        owner = _exact_mapping(
            manifest.source_owners[name], name=f"source_owners.{name}", keys=keys
        )
        if owner["scene_id"] != manifest.scene_id:
            raise ValueError("Source owner contains a foreign scene identity.")
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
    for owner_name, artifact in camera_artifacts.items():
        owner = cast(Mapping[str, object], manifest.source_owners[owner_name])
        expected_ids = tuple(
            _sequence(owner["camera_ids"], name=f"{owner_name}.camera_ids")
        )
        record = next(item for item in manifest.artifacts if item.file_name is artifact)
        if tuple(item.get("camera_id") for item in record.mapping) != expected_ids:
            raise ValueError(
                "Camera artifact order differs from its exact owner inventory."
            )
    comparison_record = next(
        item
        for item in manifest.artifacts
        if item.file_name is PublicationArtifactName.CAMERA_LAYOUT_COMPARISON
    )
    expected_comparison = tuple(
        item
        for artifact in (
            PublicationArtifactName.BLCS_CAMERA_LAYOUT,
            PublicationArtifactName.PLCS_CAMERA_LAYOUT,
        )
        for record in manifest.artifacts
        if record.file_name is artifact
        for item in record.mapping
    )
    if comparison_record.mapping != expected_comparison:
        raise ValueError("Camera comparison mapping differs from BLCS/PLCS owners.")
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
    metrics = _exact_mapping(
        manifest.metrics,
        name="metrics",
        keys={"alignment", "cameras"},
    )
    alignment_metrics = _mapping(metrics.get("alignment"), name="metrics.alignment")
    if alignment_metrics.get("schema") != ALIGNMENT_AGREEMENT_METRIC_SCHEMA:
        raise ValueError("Manifest alignment metric schema is missing or foreign.")
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
]
