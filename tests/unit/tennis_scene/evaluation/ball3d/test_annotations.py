import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from tests.support.ball3d_evaluation import make_inputs


def _document():
    inputs = make_inputs()
    source = SimpleNamespace(videos=tuple(SimpleNamespace(camera_id=c, sha256=str(i) * 64)
                                          for i, c in enumerate(inputs.camera_ids)))
    bundle = SimpleNamespace(inputs=inputs, source=source)
    doc = {"schema": "tennis_lab_ball2d_annotations.v1", "clip_id": inputs.clip_id,
           "camera_ids": list(inputs.camera_ids), "num_frames": 4, "fps": 10., "source_size_wh": [320, 240],
           "video_sha256": {v.camera_id: v.sha256 for v in source.videos},
           "provenance": {"kind": "direct_video_review", "model": "synthetic_worker_for_contract_test", "effort": "max"},
           "observations": [{"camera_id": "near", "frame_index": 0, "status": "observed", "uv_px": [123, 45],
                             "reviewed": True, "evidence": "synthetic-frame-0.png"},
                            {"camera_id": "far", "frame_index": 2, "status": "occluded", "uv_px": None,
                             "reviewed": True, "evidence": "synthetic-frame-2.png"}]}
    return bundle, doc


def test_sparse_native_pixel_labels_preserve_camera_frame_and_unreviewed_gaps(tmp_path):
    bundle, doc = _document()
    path = tmp_path / "labels.json"
    write_json_atomic(path, doc)
    result = load_annotations(path, bundle)
    assert np.argwhere(result.observed).tolist() == [[1, 0]]
    np.testing.assert_array_equal(result.uv_px[1, 0], [123, 45])
    assert result.state_counts()["near"] == {"unreviewed": 3, "observed": 1, "occluded": 0, "out_of_frame": 0, "unknown": 0}
    assert result.states[2, 2] == 2


@pytest.mark.parametrize("mutation", ["video", "duplicate", "unreviewed", "outside", "prediction"])
def test_annotation_source_and_review_contract_rejects_untrusted_points(tmp_path, mutation):
    bundle, original = _document()
    doc = copy.deepcopy(original)
    if mutation == "video":
        doc["video_sha256"]["near"] = "f" * 64
    elif mutation == "duplicate":
        doc["observations"].append(doc["observations"][0])
    elif mutation == "unreviewed":
        doc["observations"][0]["reviewed"] = False
    elif mutation == "outside":
        doc["observations"][0]["uv_px"] = [320, 10]
    else:
        doc["provenance"]["kind"] = "detector_prediction"
    path = tmp_path / "labels.json"
    write_json_atomic(path, doc)
    with pytest.raises(ValueError):
        load_annotations(path, bundle)
