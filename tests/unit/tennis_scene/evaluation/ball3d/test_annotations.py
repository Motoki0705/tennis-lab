from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.ground_truth import sha256_file

FIXTURE = Path(__file__).parents[2] / "ground_truth/fixtures/reference.json"


def _bundle():
    # Trusted input identity specified independently of the JSON under test.
    inputs = SimpleNamespace(
        clip_id="synthetic/asymmetric",
        camera_ids=("right", "left"),
        num_frames=3,
        fps=29.97,
        source_size_wh=(80, 40),
        frame_indices=np.arange(3),
    )
    source = SimpleNamespace(
        videos=(
            SimpleNamespace(camera_id="right", sha256="a" * 64),
            SimpleNamespace(camera_id="left", sha256="b" * 64),
        )
    )
    return SimpleNamespace(inputs=inputs, source=source)


def test_reader_adapter_preserves_estimates_exclusions_axes_and_source_rows():
    bundle = _bundle()
    labels = load_annotations(FIXTURE, bundle, expected_sha256=sha256_file(FIXTURE))
    labels.validate_for_inputs(bundle.inputs)
    assert np.argwhere(labels.valid_for_scoring).tolist() == [[0, 0], [0, 1], [1, 0]]
    assert np.argwhere(labels.observed).tolist() == [[0, 0], [1, 0]]
    assert np.argwhere(labels.estimated).tolist() == [[0, 1]]
    np.testing.assert_array_equal(labels.uv_px[0, 1], [17.25, 8.5])
    assert np.isnan(labels.uv_px[0, 2]).all()
    assert labels.reference_kind.tolist() == [
        ["observed", "estimated", "excluded_extrapolation"],
        ["observed", "unknown", "occluded"],
    ]
    assert labels.reference.events[1]["frame_index"] == 1
    assert labels.reference.events[0]["note"] == "Contact uncertain by one frame."
    assert labels.reference.estimation_segments[0]["quality"] == "lower"
    assert labels.reference.position("right", 1).source_status == "unknown"
    assert not labels.uv_px.flags.writeable


@pytest.mark.parametrize("axis", ["camera", "frame"])
def test_reader_adapter_rejects_scoring_camera_or_frame_mismatch(axis):
    bundle = _bundle()
    labels = load_annotations(FIXTURE, bundle, expected_sha256=sha256_file(FIXTURE))
    if axis == "camera":
        bundle.inputs.camera_ids = ("left", "right")
    else:
        bundle.inputs.frame_indices = np.array([0, 2, 1])
    with pytest.raises(ValueError, match="axes"):
        labels.validate_for_inputs(bundle.inputs)
    # The public loader independently binds the identity to input camera order.
    if axis == "camera":
        with pytest.raises(ValueError, match="camera"):
            load_annotations(FIXTURE, bundle, expected_sha256=sha256_file(FIXTURE))
