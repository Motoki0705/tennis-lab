"""Import pinned historical geometry as external inputs, preserving the old lineage."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.imports.publish import bind_import, publish_import
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256

BUNDLE_SCHEMA = "tennis_lab_ball3d_fixed_inputs.v1"


def geometry_nodes(source: ClipSource) -> tuple[str, ...]:
    return (*(f"court_detection/{c}" for c in source.camera_ids),
            "court_calibration", "court_side", "camera_alignment")


def _path(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if Path(relative).is_absolute() or not result.is_relative_to(root):
        raise ValueError("Fixed bundle path escapes its root")
    return result


def _court_from_broadcast(document: dict[str, Any], calibration: CourtCalibrationOutput, view: int) -> CourtKPResult:
    """Explicit lossless recovery of omitted static arrays, proven by original NPY hashes."""
    if set(document["payload"]) != {"keypoints", "visibility", "frame_indices", "diagnostics"}:
        raise ValueError("Recovered court fields disagree with the declared artifact schema")
    if ((calibration.court.diagnostics or {}).get("temporal_policy") != "static_first_frame_broadcast"
            or (document["payload"]["diagnostics"] or {}).get("temporal_policy") != "static_first_frame"):
        raise ValueError("Court recovery requires declared static first-frame broadcast")
    values = {"keypoints": calibration.court.keypoints[view:view + 1, :1],
              "visibility": calibration.court.visibility[view:view + 1, :1],
              "frame_indices": np.asarray([0], np.int32)}
    if {document["payload"][field]["array"] for field in values} != set(document["arrays"]):
        raise ValueError("Recovered court array fields disagree with the declared artifact schema")
    for field, value in values.items():
        name = document["payload"][field]["array"]
        declared = document["arrays"][name]
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        if (list(value.shape) != declared["shape"] or str(value.dtype) != declared["dtype"]
                or hashlib.sha256(stream.getvalue()).hexdigest() != declared["sha256"]):
            raise ValueError("Recovered court array differs from the original artifact checksum")
    return CourtKPResult(**values, diagnostics=document["payload"]["diagnostics"])


@dataclass(frozen=True)
class FixedGeometryImporter:
    bundle_path: Path
    bundle_sha256: str
    court_detection: Literal["stored_artifact", "verified_calibration_frame0"] = "stored_artifact"

    def bind_nodes(self, nodes: Sequence[ComponentNode]) -> tuple[ComponentNode, ...]:
        """Declare fixed geometry as leaves, never as outputs of the current ball input.

        Only explicit load-only targets can receive imports. Original output
        types/schemas remain unchanged; original dependencies live in provenance.
        """
        if self.court_detection not in ("stored_artifact", "verified_calibration_frame0"):
            raise ValueError("Unknown fixed court detection storage policy")
        source = next(n.context.source for n in nodes if n.name == "court_calibration")
        wanted = set(geometry_nodes(source))
        if not wanted.issubset({n.name for n in nodes}):
            raise ValueError("Fixed geometry requires every declared geometry node")
        result = []
        for node in nodes:
            if node.name in wanted:
                if node.source != "load":
                    raise ValueError(f"{node.name} must be load-only to receive fixed geometry")
                node = replace(node, io=replace(node.io, inputs={}), bindings={},
                    settings={"fixed_geometry": {"bundle_sha256": self.bundle_sha256,
                                                 "court_detection": self.court_detection}},
                    required_import_identity={"importer": "fixed_geometry_bundle", "importer_version": 1,
                                              "bundle_sha256": self.bundle_sha256, "court_detection": self.court_detection})
            result.append(node)
        return tuple(result)

    def _read(self, source: ClipSource) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
        path = self.bundle_path.resolve()
        root = path.parent
        if dual_sha256(path) != self.bundle_sha256:
            raise ValueError("Fixed bundle manifest checksum mismatch")
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise ValueError("Unsupported fixed geometry bundle schema")
        commit, evidence = manifest.get("source_code_commit"), manifest.get("source_code_evidence")
        if (not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Fixed bundle requires historical code provenance")
        for name, checksum in manifest["files"].items():
            item = _path(root, name)
            if not item.is_file() or dual_sha256(item) != checksum:
                raise ValueError(f"Fixed bundle file checksum mismatch: {name}")
        index_path = _path(root, "scene.json")
        if dual_sha256(index_path) != manifest["source_index_sha256"]:
            raise ValueError("Fixed bundle scene index checksum mismatch")
        index = json.loads(index_path.read_text())
        if (index.get("schema") != "tennis_scene_index_v1"
                or document_digest(index["source"]) != index["source_sha256"]
                or manifest["source_sha256"] != index["source_sha256"]):
            raise ValueError("Fixed bundle source identity mismatch")
        original = restore_type(index["source"], ClipSource)
        if (manifest["clip_id"] != source.clip_id or manifest["camera_ids"] != list(source.camera_ids)
                or original.camera_ids != source.camera_ids):
            raise ValueError("Fixed bundle clip/camera axis mismatch")
        for old, current in zip(original.videos, source.videos, strict=True):
            # Paths may move across machines. Camera identity, bytes and timeline may not.
            if ((old.sha256, old.width, old.height, old.num_frames)
                    != (current.sha256, current.width, current.height, current.num_frames)
                    or not np.isfinite(current.fps) or not np.isfinite(old.fps)
                    or abs(old.fps - current.fps) > 1e-6):
                raise ValueError("Fixed bundle source video/timeline mismatch")
        names = geometry_nodes(source)
        for name in ("court_calibration", "court_side", "camera_alignment"):
            if manifest["components"][name] != index["artifacts"][name]:
                raise ValueError("Fixed bundle selected geometry reference mismatch")
        refs = {name: index["artifacts"][name] for name in names}
        assert_current_component_lineage(index, root, refs)
        documents = {name: read_component_descriptor(root, ref, node=name, source_sha256=index["source_sha256"])
                     for name, ref in refs.items()}
        for name, document in documents.items():
            if (document.get("schema") != "component_artifact_v1" or document.get("status") != "complete"
                    or document.get("execution_key") != refs[name]["execution_key"]):
                raise ValueError("Incomplete or inconsistent fixed geometry artifact")
        return root, manifest, index, documents

    def publish(self, nodes: Sequence[ComponentNode], store: ClipStore, source: ClipSource) -> dict[str, ArtifactRef]:
        if store.source != json_value(source):
            raise ValueError("Fixed geometry import store differs from the declared source")
        root, manifest, index, documents = self._read(source)
        if store.root.is_relative_to(root):
            raise ValueError("Fixed geometry must be imported into a separate store")
        bound = {name: bind_import(nodes, name, store) for name in geometry_nodes(source)}
        if any(target.node.context.source != source for target in bound.values()):
            raise ValueError("Fixed geometry node context differs from its source")
        values: dict[str, Any] = {}
        for name, cls in (("court_calibration", CourtCalibrationOutput),
                          ("court_side", CourtSideOutput), ("camera_alignment", CameraAlignmentOutput)):
            target = bound[name]
            reference = index["artifacts"][name]
            document = documents[name]
            if (target.references or target.node.io.inputs
                    or (reference["schema"], reference["version"]) != (target.node.io.output_schema, target.node.io.version)
                    or target.node.io.output_type is not cls):
                raise ValueError("Fixed geometry requires compatible external-input node contracts")
            values[name] = ArtifactCodec(cls).load(document["payload"],
                _path(root, reference["path"]).parent, document["arrays"])
        calibration = values["court_calibration"]
        for view, camera in enumerate(source.camera_ids):
            name = f"court_detection/{camera}"
            target, document, reference = bound[name], documents[name], index["artifacts"][name]
            if (target.references or target.node.io.inputs or target.node.io.output_type is not CourtKPResult
                    or (reference["schema"], reference["version"]) != (target.node.io.output_schema, target.node.io.version)):
                raise ValueError("Fixed court detection contract mismatch")
            if self.court_detection == "verified_calibration_frame0":
                court = _court_from_broadcast(document, calibration, view)
            else:
                court = ArtifactCodec(CourtKPResult).load(document["payload"],
                    _path(root, reference["path"]).parent, document["arrays"])
            if (not court.validate()[0] or court.keypoints.shape != (1, 1, 14, 2)
                    or not np.array_equal(court.frame_indices, [0])
                    or not np.array_equal(court.keypoints[0, 0], calibration.court.keypoints[view, 0])
                    or not np.array_equal(court.visibility[0, 0], calibration.court.visibility[view, 0])):
                raise ValueError("Fixed court detection/calibration source axes disagree")
            diagnostics = court.diagnostics or {}
            if (diagnostics.get("temporal_policy") != "static_first_frame"
                    or diagnostics.get("output_keypoint_contract") != "camera_view_v2"):
                raise ValueError("Fixed courts must declare camera-local static first-frame observations")
            cameras = diagnostics.get("cameras")
            if (not isinstance(cameras, list) or len(cameras) != 1
                    or ("camera_id" in cameras[0] and cameras[0]["camera_id"] != camera)):
                raise ValueError("Fixed court detection camera metadata disagrees with its source")
            values[name] = court
        self._validate_geometry(source, calibration, values["court_side"], values["camera_alignment"])
        if dual_sha256(self.bundle_path.resolve()) != self.bundle_sha256:
            raise ValueError("Fixed bundle changed while being read")
        # Only new immutable descriptors/arrays are published. Historical
        # descriptors and their dependencies are never rewritten or adopted.
        return {name: publish_import(target, store, values[name], importer="fixed_geometry_bundle", version=1,
            identity={"bundle_sha256": self.bundle_sha256, "source_artifact": index["artifacts"][name],
                      "court_detection": self.court_detection},
            provenance={"model_inference": False, "geometry_policy": "external_fixed_input",
                        "bundle_path": str(self.bundle_path.resolve()), "bundle_sha256": self.bundle_sha256,
                        "source_artifact": index["artifacts"][name],
                        "source_dependencies": documents[name]["dependencies"],
                        "source_identity": documents[name]["identity"],
                        "source_provenance": documents[name]["provenance"],
                        "source_code_commit": manifest["source_code_commit"],
                        "source_code_evidence": manifest["source_code_evidence"],
                        "court_detection_storage": self.court_detection})
                for name, target in bound.items()}

    @staticmethod
    def _validate_geometry(source: ClipSource, calibration: CourtCalibrationOutput,
                           side: CourtSideOutput, alignment: CameraAlignmentOutput) -> None:
        court, geometry = calibration.court, alignment.geometry
        active = tuple(v.source_index for v in calibration.calibration.views)
        if (not court.validate()[0] or court.keypoints.shape != (len(source.camera_ids), source.num_frames, 14, 2)
                or not np.array_equal(court.frame_indices, np.arange(source.num_frames))
                or not np.array_equal(court.keypoints, np.broadcast_to(court.keypoints[:, :1], court.keypoints.shape))
                or not np.array_equal(court.visibility, np.broadcast_to(court.visibility[:, :1], court.visibility.shape))
                or len(set(active)) != len(active) or any(i < 0 or i >= len(source.camera_ids) for i in active)):
            raise ValueError("Fixed calibration source camera/frame axis mismatch")
        cameras = tuple(source.camera_ids[i] for i in active)
        if (len(cameras) < 2 or geometry is None or cameras != calibration.calibration.camera_ids
                or side.camera_ids != cameras or geometry.camera_ids != cameras
                or tuple(c.camera_id for c in geometry.cameras) != cameras
                or geometry.selection.selected_camera_ids != cameras
                or geometry.selection.reference_camera_id != calibration.reference_camera
                or side.reference_camera != calibration.reference_camera
                or geometry.reference_camera != calibration.reference_camera
                or geometry.view_half_turns != side.view_half_turns
                or any(v.frame_index != 0 for v in calibration.calibration.views)):
            raise ValueError("Fixed calibration/side/alignment camera axis mismatch")
        for local, camera, turn in zip(calibration.calibration.views, geometry.cameras, side.view_half_turns, strict=True):
            expected = local.camera.half_turned(turn)
            if not all(np.array_equal(getattr(camera, field), getattr(expected, field))
                       for field in ("intrinsic", "rotation", "translation")):
                raise ValueError("Fixed alignment disagrees with the calibrated court sides")
