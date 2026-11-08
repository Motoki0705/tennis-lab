"""Tiny saved-input fixtures; no video/model/GPT execution or real clip assets."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.annotation.video import (
    BallTarget,
    FrameIdentity,
    VideoAnnotationInput,
    VideoBallAnnotation,
    VideoIdentity,
    write_annotation,
)
from src.tasks.base.annotation.contracts import FrameRange
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.court_side.hypothesis import CourtSideConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
from src.tennis_scene.pipeline.components.ball_points import BallPointsModule
from src.tennis_scene.pipeline.components.camera_alignment import (
    CameraAlignmentModule,
    CameraAlignmentOutput,
)
from src.tennis_scene.pipeline.components.camera_geometry import CameraGeometryConfig
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationModule,
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule, CourtKPResult
from src.tennis_scene.pipeline.components.court_side import (
    CourtSideModule,
    CourtSideOutput,
)
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, ComponentIO
from src.tennis_scene.pipeline.imports import (
    BallAnnotationImporter,
    FixedGeometryImporter,
)
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.utils.checksum import dual_sha256
from tests.support.ball3d_evaluation import make_inputs, make_store


class NoExecution:
    version = 1

    def __init__(self, io: ComponentIO[Any, Any]) -> None:
        self.io, self.calls = io, 0

    def process(self, inputs: Any) -> Any:
        self.calls += 1
        raise AssertionError("Fixed geometry or annotation attempted model execution")

    def assemble(self, context: AssemblyContext, artifacts: Any) -> Any:
        raise AssertionError("Imported inputs must not be assembled from the current ball")


def write_fixed_bundle(store: ClipStore, root: Path, *, omit_court_arrays: bool = False) -> str:
    root.mkdir()
    for path in store.root.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".npy"):
            relative = path.relative_to(store.root)
            if omit_court_arrays and path.suffix == ".npy" and relative.parts[1] == "court_detection":
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    source = store.source
    write_json_atomic(root / "bundle.json", {
        "schema": "tennis_lab_ball3d_fixed_inputs.v1", "clip_id": source["clip_id"],
        "camera_ids": [v["camera_id"] for v in source["videos"]],
        "source_sha256": store.source_key, "source_index_sha256": dual_sha256(root / "scene.json"),
        "source_code_commit": "1" * 40, "source_code_evidence": "synthetic fixed geometry fixture",
        "source_archive": {"name": "synthetic.zip", "sha256": "2" * 64},
        "components": {k: json_value(v) for k, v in store.references().items() if not k.startswith("court_detection/")},
        "files": {str(p.relative_to(root)): dual_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()},
    })
    digest: str = dual_sha256(root / "bundle.json")
    (root / "bundle.sha256").write_text(digest + "\n")
    return digest


@dataclass(frozen=True)
class ImportFixture:
    source: ClipSource
    historical: ClipStore
    nodes: tuple[ComponentNode, ...]
    fixed: FixedGeometryImporter
    annotations: BallAnnotationImporter
    truth: NDArray[np.float64]

    def bound_nodes(self) -> tuple[ComponentNode, ...]:
        nodes: tuple[ComponentNode, ...] = self.annotations.bind_nodes(self.fixed.bind_nodes(self.nodes))
        return nodes


def make_import_fixture(tmp_path: Path, *, omit_court_arrays: bool = False) -> ImportFixture:
    inputs = make_inputs(frames=4)
    historical = make_store(tmp_path / "historical", inputs)
    source: ClipSource = restore_type(historical.source, ClipSource)
    calibration = historical.load(historical.references()["court_calibration"], ArtifactCodec(CourtCalibrationOutput))
    assert calibration.court.diagnostics is not None
    for i, c in enumerate(source.camera_ids):
        diagnostics = {"output_keypoint_contract": "camera_view_v2", "temporal_policy": "static_first_frame",
                       "cameras": [calibration.court.diagnostics["cameras"][i]]}
        value = CourtKPResult(calibration.court.keypoints[i:i + 1, :1], calibration.court.visibility[i:i + 1, :1],
                              np.array([0], np.int32), diagnostics)
        historical.publish(f"court_detection/{c}", value, ArtifactCodec(CourtKPResult), schema="court_observations",
            version=2, identity={"scope": c}, dependencies={}, provenance={"origin": "component", "fixture": True})
    # The old side/alignment depend on the OLD ball points. They must never
    # become outputs attributed to the newly imported annotation points.
    for name, cls in (("court_side", CourtSideOutput), ("camera_alignment", CameraAlignmentOutput)):
        ref = historical.references()[name]
        doc = historical.descriptor(ref)
        value = historical.load(ref, ArtifactCodec(cls))
        deps = {port: historical.references()["court_calibration" if port == "calibration" else "court_side"]
                for port in doc["dependencies"]}
        deps.update({f"ball_{c}": historical.references()[f"ball_points/{c}"] for c in source.camera_ids})
        historical.publish(name, value, ArtifactCodec(cls), schema=ref.schema, version=ref.version,
            identity={"dependencies": json_value(deps), "fixture": True}, dependencies=deps,
            provenance={"origin": "component", "fixture": True})
    digest = write_fixed_bundle(historical, tmp_path / "bundle", omit_court_arrays=omit_court_arrays)
    fixed = FixedGeometryImporter(tmp_path / "bundle/bundle.json", digest,
        "verified_calibration_frame0" if omit_court_arrays else "stored_artifact")

    geometry_config = CameraGeometryConfig()
    cal = CourtCalibrationModule(source.camera_ids, geometry_config, roi_margins=(0., 0.))
    side = CourtSideModule(source.camera_ids, CourtSideConfig(2., 1., 1, .5, .5, .1), max_frames=4)
    alignment = CameraAlignmentModule(source.camera_ids, geometry_config, player_reprojection_px=2.,
        ball_reprojection_px=2., joint_confidence=.1, max_frames=4)
    nodes = []

    def add(name: str, component: Any, bindings: dict[str, str], camera: str | None = None, *, execute: bool = False) -> None:
        nodes.append(ComponentNode(name, component, component.io, NoExecution(component.io), bindings,
            AssemblyContext(source, camera), {}, "fixture", "execute" if execute else "load"))

    for c in source.camera_ids:
        add(f"court_detection/{c}", NoExecution(CourtKPModule.io), {}, c)
        add(f"ball_detection/{c}", NoExecution(BallDetectionModule.io), {}, c)
        add(f"ball_points/{c}", BallPointsModule(), {"detections": f"ball_detection/{c}"}, c, execute=True)
    add("court_calibration", NoExecution(cal.io), {c: f"court_detection/{c}" for c in source.camera_ids})
    balls = {f"ball_{c}": f"ball_points/{c}" for c in source.camera_ids}
    add("court_side", NoExecution(side.io), {"calibration": "court_calibration", **balls})
    add("camera_alignment", NoExecution(alignment.io), {"calibration": "court_calibration", "side": "court_side",
        "identities": "unused_identities", **{f"pose_{c}": f"unused_pose/{c}" for c in source.camera_ids}, **balls})
    add("ball_triangulation", BallTriangulationModule(source.camera_ids, reprojection_px=.01, min_frames=100),
        {"alignment": "camera_alignment", "calibration": "court_calibration", **balls}, execute=True)

    # Independent axis-aligned pinhole arithmetic. The legacy detector's UVs
    # and masks are deliberately not used to produce these annotations.
    truth = np.array([[1., 2., 1.], [1.1, 1.8, 1.05], [1.2, 1.6, 1.1], [1.3, 1.4, 1.15]])
    layout = AnnotationLayout.testing(tmp_path / "annotation_store")
    paths, expected = {}, {}
    # Camera order is deliberately not lexical; final frame has exactly one usable view.
    kinds = (("observed", "observed", "estimated", "observed"),
             ("estimated", "interpolated", "occlusion_estimated", "unknown"),
             ("unknown", "unknown", "observed", "no_active_ball"))
    for i, (video, view) in enumerate(zip(source.videos, inputs.views, strict=True)):
        pinned = VideoAnnotationInput(source=VideoIdentity(recording_id="synthetic-recording", camera_id=video.camera_id,
            file_name=video.path.name, media_sha256=video.sha256, size_bytes=100 + i,
            width=video.width, height=video.height, frame_count=4, time_base="1/1000", nominal_fps="10"),
            target=BallTarget(target_semantics="active_ball", track_id="active"), frame_range=FrameRange(start=0, stop=4),
            frames=tuple(FrameIdentity(frame_index=t, pts=100 * i + p) for t, p in enumerate((13, 114, 215, 317))))
        camera = view.camera
        depth = camera.center[2] - truth[:, 2]
        xy = np.column_stack((camera.intrinsic[0, 0] * (truth[:, 0] - camera.center[0]) / depth + 160.,
                              camera.intrinsic[1, 1] * (camera.center[1] - truth[:, 1]) / depth + 120.))
        rows = []
        for t, kind in enumerate(kinds[i]):
            has_position = kind not in ("unknown", "no_active_ball")
            rows.append({"frame_index": t, "pts": pinned.frames[t].pts,
                "state": "in_frame" if has_position else kind,
                "occlusion": "not_applicable" if kind == "no_active_ball" else "unknown" if kind == "unknown"
                    else "occluded" if kind == "occlusion_estimated" else "not_occluded",
                "position": {"x": float(xy[t, 0]), "y": float(xy[t, 1]),
                    "provenance": {"kind": kind, "support_frames": [0, 2] if kind == "interpolated" else [0, 1] if "estimated" in kind else []}}
                    if has_position else None,
                "notes": "same-camera context" if "estimated" in kind else ""})
        annotation = VideoBallAnnotation.model_validate_json(json.dumps({
            **pinned.model_dump(mode="json"), "schema_version": "video_ball_annotation.v3",
            "coordinate_system": "native_pixels_top_left_endpoint", "frames": rows}))
        path = layout.path("annotations", video.camera_id, "revision.json")
        write_annotation(layout, path, annotation, expected_input=pinned, require_complete=True)
        paths[video.camera_id], expected[video.camera_id] = path, pinned
    return ImportFixture(source, historical, tuple(nodes), fixed, BallAnnotationImporter(paths, expected), truth)
