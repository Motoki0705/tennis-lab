"""Pinned-input rejection and annotation-specific geometry policies."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.annotation_triangulation import (
    BallAnnotationTriangulationOutput,
)
from src.tennis_scene.pipeline.components.ball_annotations import BallAnnotationOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.imports import (
    FixedGeometryImporter,
)
from src.tennis_scene.pipeline.imports.fixed_geometry import _court_from_broadcast
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.support.tennis_scene.ball_imports import (
    make_import_fixture,
    write_fixed_bundle,
)


@pytest.mark.parametrize(("field", "value"), [
    ("camera_id", "other-camera"), ("media_sha256", "f" * 64),
    ("recording_id", "other-recording"), ("time_base", "1/2000"),
    ("pts", 115), ("target", "another-ball"), ("state", "unprocessed"),
])
def test_saved_annotation_must_match_external_pinned_input(tmp_path: Path, field: str, value: Any) -> None:
    fixture = make_import_fixture(tmp_path)
    path = fixture.annotations.annotations["right"]
    document = json.loads(path.read_text())
    if field == "pts":
        document["frames"][1]["pts"] = value
    elif field == "target":
        document["target"]["track_id"] = value
    elif field == "state":
        document["frames"][3].update(state=value, occlusion="unknown", position=None)
    else:
        document["source"][field] = value
    path.write_text(json.dumps(document))
    store = ClipStore(tmp_path / "new_scene", json_value(fixture.source))
    with pytest.raises(ValueError, match="expected input|unprocessed"):
        fixture.annotations.publish(fixture.bound_nodes(), store, fixture.source)
    assert store.references() == {}  # no camera is adopted before the whole input set validates


@pytest.mark.parametrize("change", ["manifest", "array", "camera_order", "camera_bytes", "clip"])
def test_fixed_geometry_rejects_changed_artifacts_and_foreign_sources(tmp_path: Path, change: str) -> None:
    fixture = make_import_fixture(tmp_path)
    source = fixture.source
    if change == "manifest":
        with fixture.fixed.bundle_path.open("ab") as handle:
            handle.write(b"\n")
    elif change == "array":
        ref = fixture.historical.references()["court_calibration"]
        path = tmp_path / "bundle" / ref.path
        name = next(iter(json.loads(path.read_text())["arrays"]))
        with (path.parent / name).open("ab") as handle:
            handle.write(b"corrupt")
    elif change == "camera_order":
        source = replace(source, videos=tuple(reversed(source.videos)))
    elif change == "camera_bytes":
        source = replace(source, videos=(replace(source.videos[0], sha256="a" * 64), *source.videos[1:]))
    else:
        source = replace(source, clip_id="different-clip")
    store = ClipStore(tmp_path / "new_scene", json_value(source))
    with pytest.raises(ValueError, match="checksum|camera axis|source video"):
        fixture.fixed.publish(fixture.fixed.bind_nodes(fixture.nodes), store, source)
    assert store.references() == {}


def test_fixed_geometry_rejects_an_internally_reordered_camera_payload(tmp_path: Path) -> None:
    fixture = make_import_fixture(tmp_path)
    store = fixture.historical
    ref = store.references()["camera_alignment"]
    value = store.load(ref, ArtifactCodec(CameraAlignmentOutput))
    assert value.geometry is not None
    changed = replace(value, geometry=replace(value.geometry, cameras=tuple(reversed(value.geometry.cameras))))
    deps = {k: store.references()["court_side" if k == "side" else "court_calibration" if k == "calibration" else f"ball_points/{k[5:]}"]
            for k in store.descriptor(ref)["dependencies"]}
    store.publish("camera_alignment", changed, ArtifactCodec(CameraAlignmentOutput), schema=ref.schema,
        version=ref.version, identity={"mutation": "camera_payload_order"}, dependencies=deps, provenance={"origin": "component"})
    digest = write_fixed_bundle(store, tmp_path / "reordered")
    importer = FixedGeometryImporter(tmp_path / "reordered/bundle.json", digest)
    target = ClipStore(tmp_path / "new_scene", json_value(fixture.source))
    with pytest.raises(ValueError, match="alignment camera axis"):
        importer.publish(importer.bind_nodes(fixture.nodes), target, fixture.source)
    assert target.references() == {}


def test_omitted_court_arrays_require_explicit_checksum_proven_recovery(tmp_path: Path) -> None:
    fixture = make_import_fixture(tmp_path, omit_court_arrays=True)
    stored = replace(fixture.fixed, court_detection="stored_artifact")
    store = ClipStore(tmp_path / "new_scene", json_value(fixture.source))
    with pytest.raises(ValueError, match="array checksum"):
        stored.publish(stored.bind_nodes(fixture.nodes), store, fixture.source)
    assert store.references() == {}
    calibration = fixture.historical.load(fixture.historical.references()["court_calibration"], ArtifactCodec(CourtCalibrationOutput))
    document = fixture.historical.descriptor(fixture.historical.references()["court_detection/right"])
    shifted = calibration.court.keypoints.copy()
    shifted[0, :, 0, 0] = .1
    changed = replace(calibration, court=replace(calibration.court, keypoints=shifted))
    with pytest.raises(ValueError, match="original artifact checksum"):
        _court_from_broadcast(document, changed, 0)


def test_import_loads_remain_bound_to_declared_bundle_and_annotation_revisions(tmp_path: Path) -> None:
    fixture = make_import_fixture(tmp_path)
    nodes = fixture.bound_nodes()
    store = ClipStore(tmp_path / "new_scene", json_value(fixture.source))
    fixture.fixed.publish(nodes, store, fixture.source)
    fixture.annotations.publish(nodes, store, fixture.source)
    # Binding another input is insufficient: loading requires that exact input
    # to have been imported. Old adopted artifacts cannot satisfy a new request.
    for target, field in (("camera_alignment", "bundle_sha256"), ("ball_detection/right", "annotation_sha256")):
        changed = tuple(replace(n, required_import_identity={**(n.required_import_identity or {}), field: "f" * 64})
                        if n.name == target else n for n in nodes)
        with pytest.raises(ValueError, match="import identity changed"):
            ComponentRunner(changed, store).run(targets=[target])
    with pytest.raises(ValueError, match="load-only"):
        fixture.fixed.bind_nodes(tuple(replace(n, source="execute") if n.name == "court_calibration" else n for n in fixture.nodes))
    with pytest.raises(ValueError, match="fixed geometry"):
        fixture.annotations.bind_nodes(fixture.nodes)


def test_equal_weight_solver_keeps_disagreeing_estimated_views(tmp_path: Path) -> None:
    fixture = make_import_fixture(tmp_path)
    path = fixture.annotations.annotations["near"]
    saved = json.loads(path.read_text())
    saved["frames"][2]["position"]["y"] += 12.
    path.write_text(json.dumps(saved))
    nodes = fixture.bound_nodes()
    store = ClipStore(tmp_path / "new_scene", json_value(fixture.source))
    fixture.fixed.publish(nodes, store, fixture.source)
    fixture.annotations.publish(nodes, store, fixture.source)
    runner = ComponentRunner(nodes, store)
    runner.run()
    result: BallAnnotationTriangulationOutput = runner.output("ball_triangulation")
    assert result.ball is not None
    geometry = runner.output("camera_alignment").geometry
    rows = [runner.output(f"ball_points/{c}") for c in fixture.source.camera_ids]

    def independent_residual(xyz: np.ndarray) -> NDArray[np.float64]:
        residual = []
        for camera, points in zip(geometry.cameras, rows, strict=True):
            # Parallel camera axes: derive pixels directly from their centres,
            # rather than the production matrix/project/refinement helpers.
            cx, cy, cz = camera.center
            depth = cz - xyz[2]
            prediction = [camera.intrinsic[0, 0] * (xyz[0] - cx) / depth + 160.,
                          camera.intrinsic[1, 1] * (cy - xyz[1]) / depth + 120.]
            residual.extend(np.asarray(prediction) - points.uv_px[2])
        result: NDArray[np.float64] = np.asarray(residual, dtype=np.float64)
        return result

    oracle = least_squares(independent_residual, fixture.truth[2], xtol=1e-13, ftol=1e-13, gtol=1e-13)
    assert oracle.success
    np.testing.assert_allclose(result.ball.trajectory.positions[2], oracle.x, atol=2e-6, rtol=0)
    assert result.used_views[:, 2].all() and result.ball.trajectory.valid[2]
    assert result.ball.trajectory.reprojection_px[:, 2].max() > 1.
    assert result.point_kind[1][2] == "occlusion_estimated" and not result.observed[1, 2]
    assert not np.allclose(result.ball.trajectory.positions[2], fixture.truth[2], atol=.01)


def test_scene_annotation_schema_rejects_mask_and_provenance_forgery(tmp_path: Path) -> None:
    fixture = make_import_fixture(tmp_path)
    nodes = fixture.bound_nodes()
    store = ClipStore(tmp_path / "new_scene", json_value(fixture.source))
    refs = fixture.annotations.publish(nodes, store, fixture.source)
    row = store.load(refs["ball_detection/near"], ArtifactCodec(BallAnnotationOutput))
    with pytest.raises(ValueError, match="observed differs"):
        replace(row, observed=row.position_usable.copy())
    with pytest.raises(ValueError, match="position_usable differs"):
        replace(row, position_usable=row.observed.copy())
    with pytest.raises(ValueError, match="provenance differ"):
        replace(row, point_kind=("observed",) * 4)


@pytest.mark.parametrize("failure", ["parallel_rays", "no_positions"])
def test_unreconstructable_annotation_positions_have_explicit_missing_reasons(tmp_path: Path, failure: str) -> None:
    fixture = make_import_fixture(tmp_path)
    for path in fixture.annotations.annotations.values():
        saved = json.loads(path.read_text())
        frame = saved["frames"][2]
        if failure == "parallel_rays":
            frame["position"].update(x=160., y=120.)
        else:
            frame.update(state="unknown", occlusion="unknown", position=None)
        path.write_text(json.dumps(saved))
    nodes = fixture.bound_nodes()
    store = ClipStore(tmp_path / "new_scene", json_value(fixture.source))
    fixture.fixed.publish(nodes, store, fixture.source)
    fixture.annotations.publish(nodes, store, fixture.source)
    runner = ComponentRunner(nodes, store)
    runner.run()
    result: BallAnnotationTriangulationOutput = runner.output("ball_triangulation")
    assert result.ball is not None and not result.ball.trajectory.valid[2]
    assert not result.used_views[:, 2].any()
    np.testing.assert_array_equal(result.ball.trajectory.positions[2], 0.)
    assert result.position_usable[:, 2].tolist() == [failure == "parallel_rays"] * 3
    assert result.missing_reasons[2] == ("invalid_multiview_geometry" if failure == "parallel_rays" else "insufficient_views")
