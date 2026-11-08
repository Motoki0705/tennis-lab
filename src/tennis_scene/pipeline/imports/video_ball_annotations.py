"""Saved v3 -> scene artifacts. Worker execution and dataset publication are upstream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.tasks.ball_detection.annotation.video import (
    VideoAnnotationInput,
    read_annotation,
    summarize_annotation,
    usage_masks,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.annotation_triangulation import (
    ANNOTATION_GEOMETRY_POLICY,
    ANNOTATION_TRAJECTORY_PORT,
    BallAnnotationTriangulationModule,
)
from src.tennis_scene.pipeline.components.ball_annotations import (
    BallAnnotationOutput,
    BallAnnotationPointsModule,
    ImportedBallAnnotationModule,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.imports.publish import bind_import, publish_import
from src.tennis_scene.pipeline.input_assembly.ball_annotations import (
    BallAnnotationPointsInputAssembler,
    BallAnnotationTriangulationInputAssembler,
)
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.utils.checksum import dual_sha256


@dataclass(frozen=True)
class BallAnnotationImporter:
    """Use inputs pinned by video registration, never identities taken from the annotation itself."""

    annotations: Mapping[str, Path]
    expected_inputs: Mapping[str, VideoAnnotationInput]

    def _validate_source(self, source: ClipSource) -> None:
        if set(self.annotations) != set(source.camera_ids) or set(self.expected_inputs) != set(source.camera_ids):
            raise ValueError("Annotation paths and pinned inputs must match every source camera")
        if len({i.source.recording_id for i in self.expected_inputs.values()}) != 1:
            raise ValueError("Annotation cameras must belong to one pinned recording")
        for video in source.videos:
            expected = self.expected_inputs[video.camera_id]
            identity = expected.source
            if ((identity.camera_id, identity.media_sha256, identity.width, identity.height)
                    != (video.camera_id, video.sha256, video.width, video.height)
                    or identity.frame_count < video.num_frames
                    or not np.isclose(float(Fraction(identity.nominal_fps)), video.fps, rtol=0, atol=1e-5)
                    or expected.frame_range.start != 0 or expected.frame_range.stop != video.num_frames):
                raise ValueError("Pinned annotation input differs from the scene source/camera/timeline")

    def bind_nodes(self, nodes: Sequence[ComponentNode]) -> tuple[ComponentNode, ...]:
        """Choose v3 schemas and the explicit equal-weight policy in a declared graph.

        Call FixedGeometryImporter.bind_nodes first. The old side/alignment
        algorithms consume real detector measurements and must not be retargeted
        to annotation positions. The generic runner needs no import exceptions.
        """
        by_name = {n.name: n for n in nodes}
        source = by_name["ball_triangulation"].context.source
        self._validate_source(source)
        for name in ("court_side", "camera_alignment"):
            if name not in by_name or by_name[name].bindings or by_name[name].source != "load":
                raise ValueError("v3 annotations require explicitly bound fixed geometry inputs")
        replacements: dict[str, ComponentNode] = {}
        for camera in source.camera_ids:
            name = f"ball_detection/{camera}"
            node = by_name[name]
            if node.source != "load":
                raise ValueError(f"{name} must be load-only for saved annotation import")
            detector = ImportedBallAnnotationModule()
            replacements[name] = replace(node, component=detector, io=detector.io,
                settings={"input": "saved_video_ball_annotation_v3"},
                required_import_identity={"importer": "video_ball_annotation_v3", "importer_version": 1,
                    "annotation_sha256": dual_sha256(self.annotations[camera]),
                    "expected_input": self.expected_inputs[camera].model_dump(mode="json"),
                    "policy": ANNOTATION_GEOMETRY_POLICY})
            name = f"ball_points/{camera}"
            points = BallAnnotationPointsModule()
            replacements[name] = replace(by_name[name], component=points, io=points.io,
                assembler=BallAnnotationPointsInputAssembler(), settings={"policy": ANNOTATION_GEOMETRY_POLICY})
        triangulation = BallAnnotationTriangulationModule(source.camera_ids)
        replacements["ball_triangulation"] = replace(by_name["ball_triangulation"], component=triangulation,
            io=triangulation.io, assembler=BallAnnotationTriangulationInputAssembler(),
            settings={"policy": ANNOTATION_GEOMETRY_POLICY})
        # The final assembly consumes the shared BallTriangulationOutput base
        # contract. Its binding still names the annotation-specific stored schema.
        if "scene_assembly" in by_name:
            node = by_name["scene_assembly"]
            replacements[node.name] = replace(node, io=replace(node.io,
                inputs={**node.io.inputs, "ball": ANNOTATION_TRAJECTORY_PORT}))
        return tuple(replacements.get(n.name, n) for n in nodes)

    def publish(self, nodes: Sequence[ComponentNode], store: ClipStore, source: ClipSource) -> dict[str, ArtifactRef]:
        self._validate_source(source)
        if store.source != json_value(source):
            raise ValueError("Annotation import store differs from the declared source")
        prepared = []
        # Validate all cameras before publishing any of them.
        for camera in source.camera_ids:
            path = self.annotations[camera].resolve()
            digest = dual_sha256(path)
            saved = read_annotation(path, expected_input=self.expected_inputs[camera], require_complete=True)
            if dual_sha256(path) != digest:
                raise ValueError("Annotation changed while being read")
            masks = usage_masks(saved)
            value = BallAnnotationOutput(camera, source.size,
                np.asarray([f.frame_index for f in saved.frames], np.int64),
                np.asarray([f.pts for f in saved.frames], np.int64),
                np.asarray([(0., 0.) if f.position is None else (f.position.x, f.position.y) for f in saved.frames], np.float32),
                np.asarray(masks.position_usable, np.float32), np.asarray(masks.position_usable, bool),
                np.asarray(masks.direct_observation, bool), np.asarray(masks.estimated, bool),
                tuple("" if f.position is None else f.position.provenance.kind for f in saved.frames),
                str(path), digest, saved.model_dump(mode="json"))
            bound = bind_import(nodes, f"ball_detection/{camera}", store)
            if bound.node.io != ImportedBallAnnotationModule.io or bound.node.context.source != source:
                raise ValueError("BallAnnotationImporter requires the v3 node contract")
            prepared.append((bound, value, summarize_annotation(saved)))
        return {bound.node.name: publish_import(bound, store, value, importer="video_ball_annotation_v3", version=1,
            identity={"annotation_sha256": value.annotation_sha256, "expected_input": json_value(self.expected_inputs[value.camera_id].model_dump(mode="json")),
                      "policy": ANNOTATION_GEOMETRY_POLICY},
            provenance={"annotation_path": value.annotation_path, "annotation_sha256": value.annotation_sha256,
                        "annotation_schema": value.annotation["schema_version"], "counts": json_value(summary),
                        "model_inference": False, "policy": ANNOTATION_GEOMETRY_POLICY})
            for bound, value, summary in prepared}
