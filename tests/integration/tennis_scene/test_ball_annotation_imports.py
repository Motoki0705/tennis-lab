"""Saved task v3 -> formal scene inputs -> newly reconstructed, auditable 3D."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.annotation_triangulation import (
    BallAnnotationTriangulationOutput,
)
from src.tennis_scene.pipeline.components.ball_annotations import BallAnnotationOutput
from src.tennis_scene.pipeline.imports.fixed_geometry import geometry_nodes
from src.tennis_scene.pipeline.input_assembly.observations import gather_balls
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PointRejection
from tests.support.tennis_scene.ball_imports import NoExecution, make_import_fixture


@pytest.mark.parametrize("omit_court_arrays", [False, True])
def test_saved_v3_estimates_contribute_without_becoming_observations(tmp_path: Path, omit_court_arrays: bool) -> None:
    fixture = make_import_fixture(tmp_path, omit_court_arrays=omit_court_arrays)
    source, nodes = fixture.source, fixture.bound_nodes()
    original = {p: dual_sha256(p) for p in (tmp_path / "bundle").rglob("*") if p.is_file()}
    if omit_court_arrays:
        # Also exercise a pre-existing scene store: old model point/3D artifacts
        # remain on disk, but cannot satisfy the new annotation schemas.
        shutil.copytree(fixture.historical.root, tmp_path / "new_scene")
    store = ClipStore(tmp_path / "new_scene", json_value(source), memory_entries=0)
    rebuilt = [*(f"ball_points/{c}" for c in source.camera_ids), "ball_triangulation"]
    old_results = {name: store.active(name) for name in rebuilt}
    geometry_refs = fixture.fixed.publish(nodes, store, source)
    annotation_refs = fixture.annotations.publish(nodes, store, source)
    assert {name: store.active(name) for name in rebuilt} == old_results

    # Discard the writing store; every input consumed below is decoded from disk.
    store = ClipStore(store.root, json_value(source), memory_entries=0)
    runner = ComponentRunner(nodes, store)
    runner.run()
    assert all(runner.statuses[n] == "loaded" for n in geometry_nodes(source))
    assert all(n.component.calls == 0 for n in nodes if isinstance(n.component, NoExecution))
    assert all(runner.statuses[f"ball_points/{c}"] == "executed" for c in source.camera_ids)
    assert runner.statuses["ball_triangulation"] == "executed"
    assert all(runner.references[name] != old_results[name] for name in rebuilt)
    result = runner.output("ball_triangulation")
    assert isinstance(result, BallAnnotationTriangulationOutput) and result.ball is not None
    assert result.camera_ids == ("right", "near", "far")
    assert result.ball.trajectory.valid.tolist() == [True, True, True, False]
    np.testing.assert_allclose(result.ball.trajectory.positions[:3], fixture.truth[:3], atol=2e-6, rtol=0)
    np.testing.assert_array_equal(result.ball.trajectory.positions[3], 0)
    np.testing.assert_array_equal(result.used_views, [[True, True, True, False], [True, True, True, False], [False, False, True, False]])
    np.testing.assert_array_equal(result.position_usable, [[True] * 4, [True, True, True, False], [False, False, True, False]])
    np.testing.assert_array_equal(result.observed, [[True, True, False, True], [False] * 4, [False, False, True, False]])
    np.testing.assert_array_equal(result.estimated, [[False, False, True, False], [True, True, True, False], [False] * 4])
    assert result.missing_reasons == ("", "", "", "insufficient_views")
    assert result.ball.trajectory.reasons[3] == PointRejection.INSUFFICIENT_VIEWS
    assert result.point_kind[1] == ("estimated", "interpolated", "occlusion_estimated", "")
    assert result.annotations[1].annotation["frames"][1]["position"]["provenance"]["support_frames"] == [0, 2]
    assert result.annotations[1].annotation["frames"][2]["occlusion"] == "occluded"
    assert result.annotations[1].annotation["frames"][0]["notes"] == "same-camera context"
    with pytest.raises(ValueError, match="task-validated source annotations"):
        # Even internally consistent union masks cannot relabel estimates as
        # observations while retaining the original annotation provenance.
        replace(result, observed=result.position_usable.copy(), estimated=np.zeros_like(result.estimated))
    assert result.policy["smoothing"] == result.policy["outlier_rejection"] == "none"

    for c in source.camera_ids:
        points = runner.output(f"ball_points/{c}")
        imported = store.load(annotation_refs[f"ball_detection/{c}"], ArtifactCodec(BallAnnotationOutput))
        np.testing.assert_array_equal(points.uv_px, imported.uv_px)
        np.testing.assert_array_equal(points.confidence, points.position_usable.astype(np.float32))
        assert points.pts.tolist() == [f.pts for f in fixture.annotations.expected_inputs[c].frames]
        assert points.annotation_sha256 == dual_sha256(fixture.annotations.annotations[c])
        assert points.annotation["target"]["target_semantics"] == "active_ball"
        # The old consumer cannot accept this schema and silently drop estimates.
        with pytest.raises(TypeError, match="observed masks"):
            gather_balls(source, {f"ball_{camera}": points for camera in source.camera_ids})

    for n, ref in geometry_refs.items():
        descriptor = store.descriptor(ref)
        assert descriptor["dependencies"] == {}
        assert descriptor["provenance"]["geometry_policy"] == "external_fixed_input"
        assert descriptor["provenance"]["source_artifact"] == json_value(fixture.historical.references()[n])
        assert descriptor["provenance"]["bundle_sha256"] == fixture.fixed.bundle_sha256
    side = store.descriptor(geometry_refs["court_side"])
    assert side["provenance"]["source_dependencies"]["ball_near"] == json_value(fixture.historical.references()["ball_points/near"])
    assert side["provenance"]["source_dependencies"]["ball_near"] != json_value(runner.references["ball_points/near"])
    assert original == {p: dual_sha256(p) for p in original}

    # Idempotent import/load and a third store confirm persistent output types,
    # masks, values, provenance and lineage rather than in-memory-only joining.
    assert fixture.fixed.publish(nodes, store, source) == geometry_refs
    assert fixture.annotations.publish(nodes, store, source) == annotation_refs
    again = ComponentRunner(nodes, ClipStore(store.root, json_value(source), memory_entries=0))
    again.run()
    assert again.statuses["ball_triangulation"] == "cached"
    restored = again.output("ball_triangulation")
    np.testing.assert_array_equal(restored.used_views, result.used_views)
    np.testing.assert_array_equal(restored.ball.trajectory.positions, result.ball.trajectory.positions)
    assert json_value(restored.annotations) == json_value(result.annotations)
