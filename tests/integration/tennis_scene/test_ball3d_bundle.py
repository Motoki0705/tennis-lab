from dataclasses import fields, replace

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.baseline import baseline_from_bundle
from src.tennis_scene.evaluation.ball3d.bundle import create_bundle, load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method, save_run
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256
from tests.support.ball3d_evaluation import court_diagnostics, make_inputs, make_store


def _bundle(tmp_path):
    inputs = make_inputs()
    store = make_store(tmp_path / "store", inputs)
    bundle = create_bundle(
        store.root,
        tmp_path / "bundle",
        clip_id=inputs.clip_id,
        source_code_commit="a" * 40,
        source_code_evidence="Synthetic contract fixture, never real-clip GT",
    )
    return inputs, bundle


def test_real_component_codec_bundle_and_cpu_runner_roundtrip(tmp_path):
    expected, bundle = _bundle(tmp_path)
    assert bundle.inputs.camera_ids == ("right", "near", "far")
    assert (
        bundle.source.clip_id == "historical_fixture_id"
    )  # Original source is not relabelled.
    for before, after in zip(expected.views, bundle.inputs.views, strict=True):
        np.testing.assert_array_equal(after.detections.uv_px, before.detections.uv_px)
        assert not after.detections.evidence.heatmaps.flags.writeable
        np.testing.assert_array_equal(after.court.frame_indices, [0])
        for court_field in (
            "keypoints_px",
            "scores",
            "valid",
            "selected_for_homography",
        ):
            np.testing.assert_array_equal(
                getattr(after.court, court_field), getattr(before.court, court_field)
            )
            assert not getattr(after.court, court_field).flags.writeable
        np.testing.assert_array_equal(
            after.court.require_lines().uv_px, before.court.require_lines().uv_px
        )
    np.testing.assert_array_equal(
        bundle.inputs.views[1].court.keypoints_px[0, 0], [68.25, 79.5]
    )
    assert np.flatnonzero(
        bundle.inputs.views[1].court.selected_for_homography[0]
    ).tolist() == [0, 3, 13]
    np.testing.assert_array_equal(
        bundle.inputs.views[0].points.observed, [True, False, True, True]
    )
    np.testing.assert_array_equal(
        bundle.inputs.views[2].points.observed, [True, True, True, False]
    )
    runs = run_method(
        baseline_from_bundle(bundle),
        bundle.inputs,
        bundle_sha256=bundle.manifest_sha256,
        code_commit="b" * 40,
        code_fingerprint="c" * 64,
    )
    assert runs[0].result.valid.all()
    np.testing.assert_allclose(
        runs[0].result.positions,
        [[1.0, 2.0, 1.0], [1.1, 1.8, 1.05], [1.2, 1.6, 1.1], [1.3, 1.4, 1.15]],
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        runs[0].result.positions, bundle.baseline_reference.ball.trajectory.positions
    )
    # Distinctive missing frame stays missing for each retained camera pair.
    assert [r.result.valid.tolist() for r in runs[1:]] == [
        [True, True, False, False],
        [True, False, True, False],
        [True, False, False, True],
    ]
    for run in runs:
        if run.held_out_camera is None:
            changed_intrinsic = expected.views[0].camera.intrinsic.copy()
            changed_intrinsic[0, 0] = 123.0
            changed_cameras = (
                replace(expected.views[0].camera, intrinsic=changed_intrinsic),
                expected.views[1].camera,
                expected.views[2].camera,
            )
            run = replace(
                run,
                result=replace(
                    run.result,
                    covariance=np.tile(np.diag([1.0, 2.0, 3.0]), (4, 1, 1)),
                    cameras=changed_cameras,
                    time_offsets_seconds=np.array([0.01, 0.0, -0.02], np.float64),
                ),
            )
        folder = tmp_path / run.key
        checksum = save_run(folder, run)
        restored = load_run(folder, expected_sha256=checksum)
        assert restored.held_out_camera == run.held_out_camera
        for field in fields(run.result):
            before, after = (
                getattr(run.result, field.name),
                getattr(restored.result, field.name),
            )
            if isinstance(before, np.ndarray):
                np.testing.assert_array_equal(after, before)
                assert after.dtype == before.dtype
        if run.held_out_camera is None:
            assert restored.result.cameras is not None
            assert restored.result.covariance is not None
            assert restored.result.cameras[0].intrinsic[0, 0] == 123.0
            np.testing.assert_array_equal(
                restored.result.time_offsets_seconds, [0.01, 0.0, -0.02]
            )
            np.testing.assert_array_equal(
                restored.result.covariance[2], np.diag([1.0, 2.0, 3.0])
            )
    array_path = next((tmp_path / runs[0].key).glob("*.npy"))
    array_path.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_run(tmp_path / runs[0].key)


@pytest.mark.parametrize("mutation", ["checksum", "camera", "extra_annotation"])
def test_bundle_rejects_file_corruption_camera_reordering_and_undeclared_labels(
    tmp_path, mutation
):
    _, bundle = _bundle(tmp_path)
    root = tmp_path / "bundle"
    if mutation == "checksum":
        next(root.rglob("*.npy")).write_bytes(b"corrupt")
    elif mutation == "extra_annotation":
        (root / "annotations.json").write_text('{"ball": [1, 2, 3]}')
    else:
        manifest = dict(bundle.manifest)
        manifest["camera_ids"] = ["far", "near", "right"]
        write_json_atomic(root / "bundle.json", manifest)
        (root / "bundle.sha256").write_text(dual_sha256(root / "bundle.json") + "\n")
    with pytest.raises(ValueError, match="checksum|camera axis|undeclared"):
        load_bundle(root)


def test_annotation_origin_in_a_valid_checksummed_store_cannot_become_method_input(
    tmp_path,
):
    inputs = make_inputs()
    store = make_store(tmp_path / "store", inputs, detector_origin="annotation")
    with pytest.raises(ValueError, match="Annotation/imported"):
        create_bundle(
            store.root,
            tmp_path / "bundle",
            clip_id=inputs.clip_id,
            source_code_commit="a" * 40,
            source_code_evidence="Synthetic leakage contract",
        )


@pytest.mark.parametrize(
    "mutation", ["missing_raw_scores", "camera_order", "source_frame"]
)
def test_bundle_rejects_checksummed_court_observation_provenance_errors(
    tmp_path, mutation
):
    inputs = make_inputs()
    doc = court_diagnostics(inputs)
    if mutation == "missing_raw_scores":
        del doc["cameras"][1]["frames"][0]["raw_scores"]
    elif mutation == "camera_order":
        doc["cameras"] = list(reversed(doc["cameras"]))
    else:
        doc["cameras"][1]["frames"][0]["frame_index"] = 1
    store = make_store(tmp_path / "store", inputs, court_diagnostics_override=doc)
    with pytest.raises(ValueError, match="raw court"):
        create_bundle(
            store.root,
            tmp_path / "bundle",
            clip_id=inputs.clip_id,
            source_code_commit="a" * 40,
            source_code_evidence="Court provenance fault injection",
        )
