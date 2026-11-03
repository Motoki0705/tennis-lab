"""Read-only extraction of historical component artifacts into standalone inputs.

The original index/descriptors are byte copies. Complete dependency descriptors
are kept for lineage audit, while only the ten selected components carry arrays.
Neither an old artifact identity nor a source code fingerprint is rewritten.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import BallPointsOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.temporal_smoothing import TemporalSmoothingConfig
from src.utils.video.reader import probe_video_info

from .contracts import Ball3DResult, DetectorReplayConfig, MethodInput, ViewInput
from .storage import checked_path, read_json

BUNDLE_SCHEMA = "tennis_lab_ball3d_fixed_inputs.v1"


def selected_nodes(camera_ids: tuple[str, ...]) -> tuple[str, ...]:
    return ("court_calibration", "court_side", "camera_alignment",
            *(f"ball_detection/{camera}" for camera in camera_ids),
            *(f"ball_points/{camera}" for camera in camera_ids), "ball_triangulation")


@dataclass(frozen=True)
class FixedInputBundle:
    """Evaluation-side container. Pass only ``inputs`` to a method runner."""

    inputs: MethodInput
    baseline_reference: BallTriangulationOutput
    source: ClipSource
    manifest_sha256: str
    manifest: dict[str, Any]
    triangulation_settings: dict[str, Any]


def _source(index: dict[str, Any]) -> ClipSource:
    if index.get("schema") != "tennis_scene_index_v1" or document_digest(index["source"]) != index["source_sha256"]:
        raise ValueError("Store source/schema checksum mismatch")
    source = restore_type(index["source"], ClipSource)
    if len(source.videos) != 3 or not math.isfinite(source.fps):
        raise ValueError("Ball3D LOVO requires exactly three synchronized source cameras")
    return source


def _lineage(root: Path, index: dict[str, Any], nodes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    refs = {name: index["artifacts"][name] for name in nodes}
    assert_current_component_lineage(index, root.resolve(), refs)
    descriptors: dict[str, dict[str, Any]] = {}

    def visit(name: str) -> None:
        if name in descriptors:
            return
        ref = index["artifacts"][name]
        doc = read_component_descriptor(root.resolve(), ref, node=name, source_sha256=index["source_sha256"])
        if doc["provenance"].get("origin") != "component":
            raise ValueError("Annotation/imported component leakage in fixed input lineage")
        if doc["identity"].get("dependencies") != doc["dependencies"]:
            raise ValueError("Descriptor identity and dependency lineage disagree")
        descriptors[name] = doc
        for dependency in doc["dependencies"].values():
            producer = next(n for n, active in index["artifacts"].items() if active == dependency)
            visit(producer)

    for name in nodes:
        visit(name)
    return descriptors


def _load_component(root: Path, index: dict[str, Any], name: str, schema: str, version: int, cls: type[Any]) -> Any:
    ref = index["artifacts"][name]
    if (ref["schema"], ref["version"]) != (schema, version):
        raise ValueError(f"Unsupported fixed component schema: {name}")
    doc = read_component_descriptor(root.resolve(), ref, node=name, source_sha256=index["source_sha256"])
    return ArtifactCodec(cls).load(doc["payload"], checked_path(root, ref["path"]).parent, doc["arrays"])


def load_bundle(root: Path, *, expected_sha256: str | None = None) -> FixedInputBundle:
    root = root.resolve()
    digest = dual_sha256(root / "bundle.json")
    if digest != (root / "bundle.sha256").read_text().strip() or (expected_sha256 is not None and digest != expected_sha256):
        raise ValueError("Bundle manifest checksum mismatch")
    manifest = read_json(root / "bundle.json")
    fields = {"schema", "clip_id", "source_code_commit", "source_code_evidence", "source_archive",
              "source_index_sha256", "source_sha256", "camera_ids", "components", "files"}
    if set(manifest) != fields or manifest["schema"] != BUNDLE_SCHEMA:
        raise ValueError("Unsupported bundle schema/fields; annotations are not input components")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["source_code_commit"]) or not manifest["source_code_evidence"]:
        raise ValueError("Bundle requires explicit historical code provenance")
    for name, checksum in manifest["files"].items():
        path = checked_path(root, name)
        if not path.is_file() or dual_sha256(path) != checksum:
            raise ValueError(f"Bundle file checksum mismatch: {name}")
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if files != set(manifest["files"]) | {"bundle.json", "bundle.sha256"}:
        raise ValueError("Bundle contains undeclared files (possible annotation leakage)")
    index = read_json(root / "scene.json")
    source = _source(index)
    if (manifest["source_index_sha256"] != dual_sha256(root / "scene.json")
            or manifest["source_sha256"] != index["source_sha256"]
            or manifest["camera_ids"] != list(source.camera_ids)):
        raise ValueError("Bundle source/camera axis mismatch")
    nodes = selected_nodes(source.camera_ids)
    if manifest["components"] != {name: index["artifacts"][name] for name in nodes}:
        raise ValueError("Bundle selected component axis mismatch or annotation leakage")
    descriptors = _lineage(root, index, nodes)
    # Only selected arrays and complete lineage descriptors may be present.
    expected_files = {"scene.json"}
    for name, doc in descriptors.items():
        path = Path(index["artifacts"][name]["path"])
        expected_files.add(str(path))
        if name in nodes:
            expected_files.update(str(path.parent / array) for array in doc["arrays"])
    if expected_files != set(manifest["files"]):
        raise ValueError("Bundle file inventory disagrees with selected arrays and lineage")
    calibration = _load_component(root, index, "court_calibration", "local_court_calibration", 1, CourtCalibrationOutput)
    side = _load_component(root, index, "court_side", "court_side", 3, CourtSideOutput)
    alignment = _load_component(root, index, "camera_alignment", "aligned_cameras", 1, CameraAlignmentOutput)
    geometry = alignment.geometry
    if (calibration.court.keypoints.shape != (3, source.num_frames, 14, 2)
            or calibration.court.visibility.shape != (3, source.num_frames, 14)
            or not np.array_equal(calibration.court.frame_indices, np.arange(source.num_frames))):
        raise ValueError("Court calibration source camera/frame axis mismatch")
    if geometry is None or (calibration.calibration.camera_ids != source.camera_ids
            or side.camera_ids != source.camera_ids or geometry.camera_ids != source.camera_ids
            or tuple(c.camera_id for c in geometry.cameras) != source.camera_ids
            or geometry.view_half_turns != side.view_half_turns
            or geometry.reference_camera != side.reference_camera
            or calibration.reference_camera != side.reference_camera):
        raise ValueError("Calibration/side/alignment camera axis mismatch")
    views = []
    for i, camera_id in enumerate(source.camera_ids):
        local = calibration.calibration.views[i]
        if local.source_index != i or local.frame_index != 0:
            raise ValueError("Local calibration source camera/frame axis mismatch")
        detection = _load_component(root, index, f"ball_detection/{camera_id}", "ball_detections", 2, BallDetectionOutput)
        points = _load_component(root, index, f"ball_points/{camera_id}", "ball_points", 4, BallPointsOutput)
        config = descriptors[f"ball_detection/{camera_id}"]["identity"]["settings"]["config"]
        smoothing = descriptors[f"ball_points/{camera_id}"]["identity"]["settings"]["smoothing"]
        views.append(ViewInput(geometry.cameras[i], local, side.view_half_turns[i], detection, points,
                     DetectorReplayConfig(config["score_threshold"], config["subpixel_refine"],
                                          TrajectoryGateConfig(**config["trajectory_gate"])),
                     TemporalSmoothingConfig(**smoothing)))
    inputs = MethodInput(manifest["clip_id"], np.arange(source.num_frames, dtype=np.int64),
                         source.fps, source.size, tuple(views))
    reference = _load_component(root, index, "ball_triangulation", "ball_trajectory", 1, BallTriangulationOutput)
    if reference.ball is None or reference.ball.trajectory.positions.shape != (source.num_frames, 3):
        raise ValueError("Baseline comparison reference is missing or has a different frame axis")
    trajectory = reference.ball.trajectory
    Ball3DResult(inputs.camera_ids, inputs.frame_indices, trajectory.positions, trajectory.valid, trajectory.inliers)
    settings = descriptors["ball_triangulation"]["identity"]["settings"]
    return FixedInputBundle(inputs, reference, source, digest, manifest, settings)


def verify_videos(bundle: FixedInputBundle, videos: Mapping[str, Path]) -> dict[str, Any]:
    if set(videos) != set(bundle.source.camera_ids):
        raise ValueError("Bind source videos by exact camera ID")
    checked = {}
    for video in bundle.source.videos:
        path = videos[video.camera_id]
        info = probe_video_info(path)
        checksum = dual_sha256(path)
        if (checksum != video.sha256 or (info.width, info.height, info.frame_count) != (video.width, video.height, video.num_frames)
                or not math.isclose(info.fps, video.fps, abs_tol=1e-3, rel_tol=0)):
            raise ValueError(f"Video checksum/camera/timeline mismatch: {video.camera_id}")
        checked[video.camera_id] = {"sha256": checksum, "num_frames": info.frame_count,
                                    "fps": info.fps, "width": info.width, "height": info.height}
    return checked


def validate_source_candidate(candidate: Path, source_store: Path) -> dict[str, Any]:
    """Validate the parent's descriptor/array copy; do not call it a full store."""
    root = source_store.resolve()
    index = read_json(root / "scene.json")
    source = _source(index)
    nodes = selected_nodes(source.camera_ids)
    _lineage(root, index, nodes)
    manifest = read_json(candidate / "bundle_manifest.json")
    if (manifest.get("schema") != "tennis_lab_fixed_input_source_bundle.v1"
            or set(manifest["components"]) != set(nodes)
            or manifest["source_scene_sha256"] != dual_sha256(root / "scene.json")):
        raise ValueError("Source candidate does not match the adopted store")
    for declared, video in zip(manifest["videos"], source.videos, strict=True):
        expected = {"camera_id": video.camera_id, "fps": video.fps, "height": video.height,
                    "width": video.width, "num_frames": video.num_frames, "sha256": video.sha256}
        if any(declared[key] != value for key, value in expected.items()):
            raise ValueError("Candidate video camera/frame axis mismatch")
    arrays_count = 0
    for name in nodes:
        declared, ref = manifest["components"][name], index["artifacts"][name]
        if any(declared[key] != ref[key] for key in ("path", "sha256", "schema", "version")):
            raise ValueError("Candidate component camera/identity mismatch")
        doc = read_component_descriptor(candidate.resolve(), ref, node=name, source_sha256=index["source_sha256"])
        if declared["arrays"] != doc["arrays"]:
            raise ValueError("Candidate array declarations differ from original descriptor")
        for array, spec in doc["arrays"].items():
            path = checked_path(candidate, str(Path(ref["path"]).parent / array))
            values = np.load(path, allow_pickle=False, mmap_mode="r")
            if (dual_sha256(path) != spec["sha256"] or list(values.shape) != spec["shape"]
                    or str(values.dtype) != spec["dtype"]):
                raise ValueError("Candidate array checksum/shape/dtype mismatch")
            arrays_count += 1
    return {"manifest_sha256": dual_sha256(candidate / "bundle_manifest.json"),
            "components": len(nodes), "arrays": arrays_count,
            "source_code_commit": manifest["source_code_commit"]}


def create_bundle(source_store: Path, output: Path, *, clip_id: str, source_code_commit: str,
                  source_code_evidence: str, source_archive: Path | None = None) -> FixedInputBundle:
    root, output = source_store.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Use a new bundle output separate from the immutable source store")
    if not clip_id or not re.fullmatch(r"[0-9a-f]{40}", source_code_commit) or not source_code_evidence:
        raise ValueError("Explicit clip and historical code provenance are required")
    index = read_json(root / "scene.json")
    source = _source(index)
    nodes = selected_nodes(source.camera_ids)
    descriptors = _lineage(root, index, nodes)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ball3d-bundle-", dir=output.parent))
    try:
        shutil.copyfile(root / "scene.json", temporary / "scene.json")
        for name, doc in descriptors.items():
            relative = Path(index["artifacts"][name]["path"])
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked_path(root, str(relative)), target)
            if name in nodes:
                for array, spec in doc["arrays"].items():
                    src = checked_path(root, str(relative.parent / array))
                    if dual_sha256(src) != spec["sha256"]:
                        raise ValueError("Source array checksum mismatch")
                    shutil.copyfile(src, target.parent / array)
        manifest = {"schema": BUNDLE_SCHEMA, "clip_id": clip_id, "source_code_commit": source_code_commit,
                    "source_code_evidence": source_code_evidence,
                    "source_archive": None if source_archive is None else {"name": source_archive.name, "sha256": dual_sha256(source_archive)},
                    "source_index_sha256": dual_sha256(root / "scene.json"), "source_sha256": index["source_sha256"],
                    "camera_ids": list(source.camera_ids), "components": {name: index["artifacts"][name] for name in nodes},
                    "files": {str(p.relative_to(temporary)): dual_sha256(p) for p in sorted(temporary.rglob("*")) if p.is_file()}}
        write_json_atomic(temporary / "bundle.json", manifest)
        (temporary / "bundle.sha256").write_text(dual_sha256(temporary / "bundle.json") + "\n")
        load_bundle(temporary)  # Typed/axis/mask validation before publication.
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return load_bundle(output)
