from dataclasses import replace

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.contracts import Ball3DResult
from tests.support.ball3d_evaluation import make_inputs


@pytest.mark.parametrize("mutation", ["camera", "frame", "evidence_frame", "mask", "annotation"])
def test_asymmetric_input_axes_and_detector_masks_reject_corruption(mutation):
    inputs = make_inputs()
    near = inputs.views[1]
    with pytest.raises(ValueError):
        if mutation == "camera":
            replace(near, camera=inputs.views[0].camera)
        elif mutation == "frame":
            replace(inputs, frame_indices=np.array([0, 1, 3, 4], np.int64))
        elif mutation == "evidence_frame":
            evidence = replace(near.detections.evidence, selected_time_index=np.ones(4, np.int64))
            replace(near, detections=replace(near.detections, evidence=evidence))
        elif mutation == "mask":
            # A consistent smoothing partition can still be the wrong detector source mask.
            original = near.points.smoothing
            source = original.source_observed.copy()
            source[0] = False
            accepted = original.accepted.copy()
            accepted[0] = False
            coordinates = original.coordinates.copy()
            coordinates[0] = 0
            changed = replace(original, source_observed=source, accepted=accepted,
                coordinates=coordinates, smoothed=np.zeros(4, bool), consensus_supported=np.zeros(4, bool))
            points = replace(near.points, uv_px=coordinates, confidence=np.where(accepted, near.points.confidence, 0),
                             observed=accepted, smoothing=changed)
            replace(near, points=points)
        else:
            annotation = replace(near.detections, score_semantics="annotation_acceptance_not_probability", evidence=None)
            replace(near, detections=annotation)


@pytest.mark.parametrize("mutation", ["unsupported_valid", "invalid_position", "covariance", "camera_order"])
def test_result_rejects_mask_geometry_and_camera_corruption(mutation):
    arguments = dict(camera_ids=("b", "a"), frame_indices=np.arange(2, dtype=np.int64),
        positions=np.array([[1., 2., 3.], [0., 0., 0.]], np.float32), valid=np.array([True, False]),
        used_views=np.array([[True, False], [True, False]]))
    if mutation == "unsupported_valid":
        arguments["used_views"][1, 0] = False
    elif mutation == "invalid_position":
        arguments["positions"][1] = 9
    elif mutation == "covariance":
        arguments["covariance"] = np.tile(np.diag([1., -1., 1.]), (2, 1, 1))
    else:
        arguments["cameras"] = tuple(v.camera for v in make_inputs().views[:2])
    with pytest.raises(ValueError):
        Ball3DResult(**arguments)
