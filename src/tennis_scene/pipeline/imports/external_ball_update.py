"""Add external ball 3D to a frozen clip store without constructing a runner."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np

from src.tennis_scene.generate_dataset.manifest import ClipManifest, DatasetClipRecord
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.assembly import assemble_automatic_scene
from src.tennis_scene.pipeline.clip_inputs import InputSnapshot, resolve_dataset_clip
from src.tennis_scene.pipeline.components.annotation_triangulation import (
    ANNOTATION_GEOMETRY_POLICY,
    triangulate_annotation_positions,
)
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    DECLARED_PIPELINE_CONTRACT,
    validate_dataset_result,
)
from src.tennis_scene.pipeline.imports import external_ball_positions
from src.tennis_scene.pipeline.imports.external_ball_geometry import PinnedBallCourt
from src.tennis_scene.pipeline.imports.external_ball_positions import (
    REPORTED_VISIBILITY,
    STATUSES,
    ExternalBallPositions,
    read_external_ball_positions,
)
from src.tennis_scene.pipeline.imports.external_ball_publication import (
    ScenePointers,
    current_publication,
    publish_scene_revision,
)
from src.tennis_scene.pipeline.storage.clip_store import (
    ArtifactRef,
    ClipStore,
    clip_store_lock,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256
from src.utils.geometry.keypoints import normalize_keypoints
from src.utils.geometry.triangulation import PointRejection
from src.utils.io import load_json
from src.utils.video import VideoInfo

Value = TypeVar("Value")

POSITIONS_NODE = "external_ball_positions"
TRIANGULATION_NODE = "ball_triangulation"
SHARED_ALIGNMENT_NODE = "external_ball_camera_geometry"
REJECTION_CODES = {
    "valid": int(PointRejection.VALID),
    "insufficient_views": int(PointRejection.INSUFFICIENT_VIEWS),
    "invalid_multiview_geometry": int(PointRejection.OPTIMIZATION_FAILED),
}


def _read_fixed(
    store: ClipStore,
    index: dict[str, Any],
    node: str,
    schema: str,
    version: int,
    output: type[Value],
) -> tuple[ArtifactRef, dict[str, Any], Value]:
    reference = ArtifactRef(**index["artifacts"][node])
    descriptor = read_component_descriptor(
        store.root, index["artifacts"][node], node=node, source_sha256=store.source_key
    )
    if (
        (reference.schema, reference.version, descriptor.get("status"))
        != (schema, version, "complete")
        or descriptor.get("execution_key") != reference.execution_key
        or document_digest(descriptor["identity"]) != reference.execution_key
    ):
        raise ValueError(
            f"Saved {node} does not match {schema} v{version} and its recorded identity"
        )
    value: Value = store.load(reference, ArtifactCodec(output))
    return reference, descriptor, value


def _statistics(
    positions: ExternalBallPositions, ball: BallReconstructionResult
) -> dict[str, Any]:
    trajectory = ball.trajectory
    views, frames = ball.visibility.shape
    view_count = ball.visibility.sum(0)
    error = trajectory.reprojection_px[trajectory.inliers]
    point_status = {}
    for code, label in enumerate(STATUSES):
        mask = positions.status == code
        point_status[label] = {
            "total": int(mask.sum()),
            "out_of_frame": int((mask & ~ball.visibility).sum()),
            "used_in_3d": int((mask & trajectory.inliers).sum()),
        }
    return {
        "frames": frames,
        "valid_3d": int(trajectory.valid.sum()),
        "valid_rate": float(trajectory.valid.mean()),
        "point_status": point_status,
        "view_count": {
            str(count): {
                "frames": int((view_count == count).sum()),
                "valid_3d": int(((view_count == count) & trajectory.valid).sum()),
            }
            for count in range(views + 1)
        },
        "ball_rejection": {
            label: int((trajectory.reasons == code).sum())
            for label, code in REJECTION_CODES.items()
        },
        "reprojection_px": None
        if not len(error)
        else dict(
            zip(
                ("p50", "p95", "max"),
                np.asarray(np.percentile(error, [50, 95, 100])).tolist(),
                strict=True,
            )
        ),
    }


def _saved_failure(
    path: Path, source: ClipSource, refs: dict[str, Any]
) -> dict[str, Any]:
    digest = dual_sha256(path)
    failure = load_json(path)
    if (
        failure["clip_id"] != source.clip_id
        or not failure["error"].startswith("ReconstructionUnavailable:")
        or "player_association" in refs
    ):
        raise ValueError(
            "Incomplete player reconstruction lacks the expected #2363 failure lineage"
        )
    reason = (
        "player_association_failed" if "court_side" in refs else "court_side_failed"
    )
    return {
        "status": "unavailable",
        "reason_code": reason,
        "saved_failure_sha256": digest,
        "error": failure["error"],
    }


def update_clip(
    dataset: Path,
    clip_record: DatasetClipRecord,
    positions_path: Path,
    court: PinnedBallCourt,
    code_sha256: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    clip = resolve_dataset_clip(dataset, clip_record)
    manifest = ClipManifest.load(clip.clip_dir)
    root = (clip.clip_dir / ANNOTATION_RELATIVE_DIR).resolve()
    current, pending = current_publication(root)
    before = current if pending is None else pending.before
    index = before.index()
    source = cast(ClipSource, restore_type(index["source"], ClipSource))
    inputs = dict(clip.snapshot.files)
    if (
        index.get("schema") != "tennis_scene_index_v1"
        or document_digest(source) != index["source_sha256"]
        or (manifest.dataset_id, manifest.video_id)
        != (court.side.decision.dataset_id, court.side.decision.video_id)
        or source.clip_id != clip_record.clip_id
        or source.camera_ids != clip.camera_ids
        or (source.num_frames, *source.size, source.fps)
        != (manifest.num_frames, manifest.width, manifest.height, manifest.fps)
        or {v.camera_id: v.sha256 for v in source.videos} != clip.media_sha256
        or tuple(p.resolve() for p in source.paths)
        != tuple(p.resolve() for p in clip.video_paths)
    ):
        raise ValueError(
            "Saved clip source does not match the dataset manifest or media"
        )
    positions = read_external_ball_positions(positions_path, manifest)
    inputs[positions_path] = positions.sha256
    store = ClipStore(root, json_value(source), memory_entries=0)
    refs = index["artifacts"]
    if {POSITIONS_NODE, TRIANGULATION_NODE, SHARED_ALIGNMENT_NODE} & refs.keys():
        raise ValueError(
            "External ball publication requires the saved player-only component index"
        )
    calibration_ref, _, calibration = _read_fixed(
        store,
        index,
        "court_calibration",
        "local_court_calibration",
        1,
        CourtCalibrationOutput,
    )
    marker = before.marker()
    saved_alignment = None
    alignment_ref: ArtifactRef | None = None
    player_state: dict[str, Any] = {"status": "reused"}
    if "camera_alignment" in refs:
        alignment_ref, alignment_doc, saved = _read_fixed(
            store,
            index,
            "camera_alignment",
            "aligned_cameras",
            1,
            CameraAlignmentOutput,
        )
        if (
            saved.geometry is None
            or marker is None
            or marker.get("clip_manifest_sha256") != clip.manifest_sha256
            or marker.get("media_sha256") != clip.media_sha256
            or alignment_doc["dependencies"].get("calibration")
            != json_value(calibration_ref)
        ):
            raise ValueError(
                "Saved camera alignment lacks its completed player scene or calibration"
            )
        saved_alignment = saved.geometry
    else:
        if marker is not None or "scene_assembly" in refs:
            raise ValueError("An existing player scene lacks a saved camera_alignment")
        failure_path = clip.clip_dir / "annotations" / "tennis_scene.failure.json"
        player_state = _saved_failure(failure_path, source, refs)
        inputs[failure_path] = player_state["saved_failure_sha256"]
    geometry, geometry_source = court.bind(
        source, calibration, positions, saved_alignment
    )
    calibration_done = time.perf_counter()
    snapshot = InputSnapshot(tuple(inputs.items()))
    identity = {
        "schema": "external_ball_update_v1",
        "source_sha256": store.source_key,
        "code_sha256": code_sha256,
        "clip_manifest_sha256": clip.manifest_sha256,
        "external_positions_sha256": positions.sha256,
        "geometry": geometry_source,
        "camera_alignment": refs.get("camera_alignment"),
        "court_calibration": json_value(calibration_ref),
        "previous_scene_assembly": refs.get("scene_assembly"),
        "previous_scene_export": index["exports"].get("scene"),
        "previous_pointer_sha256": document_digest(before),
        "players": player_state,
        "triangulation": dict(ANNOTATION_GEOMETRY_POLICY),
    }
    if pending is not None:
        if pending.inputs != identity:
            raise ValueError(
                "External ball publication inputs changed. Restore the previous scene before republishing."
            )
        with clip_store_lock(root):
            if ScenePointers.read(root) != current:
                raise ValueError("Clip pointers changed during external ball replay")
            snapshot.assert_unchanged()
            pending.activate(root, current)
        return {
            **pending.statistics,
            "reused_publication": True,
            "pointer_digests": pending.pointer_digests(),
            "elapsed_seconds": time.perf_counter() - start,
        }
    before.verify(root)
    assert_current_component_lineage(
        index, root, {"court_calibration": json_value(calibration_ref)}
    )
    usable = positions.usable
    uv = np.where(usable[..., None], positions.uv_px, 0).astype(np.float32)
    ball = triangulate_annotation_positions(uv, usable, geometry.cameras)
    triangulation_done = time.perf_counter()
    nodes: dict[str, ArtifactRef] = {}

    def stage(
        name: str,
        value: Value,
        codec: type[Value],
        schema: str,
        deps: dict[str, ArtifactRef],
        *,
        version: int = 1,
        origin: str = "component",
    ) -> ArtifactRef:
        ref = store.publish(
            name,
            value,
            ArtifactCodec(codec),
            schema=schema,
            version=version,
            identity={**identity, "node": name, "dependencies": json_value(deps)},
            dependencies=deps,
            provenance={
                "origin": origin,
                "execution": "external_ball_update",
                "geometry": geometry_source,
            },
            adopt=False,
        )
        nodes[name] = ref
        return ref

    if saved_alignment is None:
        alignment_ref = stage(
            SHARED_ALIGNMENT_NODE,
            CameraAlignmentOutput(geometry),
            CameraAlignmentOutput,
            "aligned_cameras",
            {"calibration": calibration_ref},
            origin="import",
        )
    assert alignment_ref is not None
    points_ref = stage(
        POSITIONS_NODE,
        positions,
        ExternalBallPositions,
        external_ball_positions.SCHEMA,
        {},
        origin="import",
    )
    ball_ref = stage(
        TRIANGULATION_NODE,
        ball,
        BallReconstructionResult,
        "external_ball_trajectory",
        {"positions": points_ref, "alignment": alignment_ref},
    )
    metadata = {
        "external_ball": {
            "input_file": positions_path.name,
            "positions": json_value(points_ref),
            "camera_geometry": json_value(alignment_ref),
            "geometry_provenance": geometry_source,
            "status_labels": list(STATUSES),
            "status_codes": positions.status.tolist(),
            "reported_visibility_labels": list(REPORTED_VISIBILITY),
            "ball_rejection_codes": REJECTION_CODES,
            "triangulation": dict(ANNOTATION_GEOMETRY_POLICY),
        },
        "player_reconstruction": player_state,
    }
    if saved_alignment is None:
        scene = assemble_automatic_scene(
            video_paths=source.paths,
            camera_ids=source.camera_ids,
            info=VideoInfo(source.fps, *source.size, source.num_frames),
            court=calibration.court,
            active_indices=tuple(
                view.source_index for view in calibration.calibration.views
            ),
            geometry=geometry,
            grouped_people=None,
            skeleton=None,
            players=None,
            ball=ball,
            metadata={
                **metadata,
                "status": "partial",
                "dataset_clip_id": source.clip_id,
                "pipeline_contract": DECLARED_PIPELINE_CONTRACT,
            },
        )
        dependencies = {"calibration": calibration_ref, "alignment": alignment_ref}
    else:
        _, original, previous = _read_fixed(
            store, index, "scene_assembly", "scene_result", 2, SceneResult
        )
        if (
            original["dependencies"].get("alignment") != json_value(alignment_ref)
            or original["dependencies"].get("calibration")
            != json_value(calibration_ref)
            or previous.metadata.get("court_reference") != geometry.document
            or previous.metadata.get("camera_ids") != list(source.camera_ids)
            or previous.ball_vis is None
            or previous.ball_vis.any()
            or previous.ball_3d_valid is None
            or previous.ball_3d_valid.any()
        ):
            raise ValueError(
                "Saved player scene disagrees with its fixed geometry or already contains ball observations"
            )
        trajectory = ball.trajectory
        scene = replace(
            previous,
            ball_uv=normalize_keypoints(uv, *source.size),
            ball_vis=usable,
            ball_3d=trajectory.positions,
            ball_3d_valid=trajectory.valid,
            ball_rejection_code=trajectory.reasons,
            metadata={
                **previous.metadata,
                **metadata,
                "status": previous.metadata["status"]
                if trajectory.valid.any()
                else "partial",
                "validity_statistics": {
                    **previous.metadata["validity_statistics"],
                    "ball_3d_frames": int(trajectory.valid.sum()),
                },
            },
        )
        dependencies = {
            port: ArtifactRef(**ref) for port, ref in original["dependencies"].items()
        }
    validate_dataset_result(scene, clip_record)
    scene_ref = stage(
        "scene_assembly",
        scene,
        SceneResult,
        "scene_result",
        {**dependencies, "ball": ball_ref},
        version=2,
    )
    staged_scene = store.load(scene_ref, ArtifactCodec(SceneResult))
    staged = time.perf_counter()
    stats = {
        **_statistics(positions, ball),
        "clip_id": source.clip_id,
        "scene_assembly": json_value(scene_ref),
        "previous_scene_assembly": identity["previous_scene_assembly"],
        "input_sha256": positions.sha256,
        "geometry": geometry_source["origin"],
        "player_reason": player_state.get("reason_code"),
        "prepare_seconds": calibration_done - start,
        "triangulation_seconds": triangulation_done - calibration_done,
        "stage_seconds": staged - triangulation_done,
    }
    receipt = publish_scene_revision(
        root, current, before, nodes, staged_scene, clip, snapshot, identity, stats
    )
    end = time.perf_counter()
    return {
        **receipt.statistics,
        "reused_publication": False,
        "pointer_digests": receipt.pointer_digests(),
        "export_seconds": end - staged,
        "elapsed_seconds": end - start,
    }
