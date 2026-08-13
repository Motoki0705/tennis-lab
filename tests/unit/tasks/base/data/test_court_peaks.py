from __future__ import annotations

import pytest
import torch

from src.synthetic_data_generation.dataset.court.components.labels import (
    PHYSICAL_INDICES_BY_CLASS,
    SEMANTIC_CLASS_NAMES,
)
from src.tasks.base.data.court_peaks import (
    COURT_PHYSICAL_INDICES_BY_CLASS,
    COURT_SEMANTIC_CLASS_NAMES,
    CourtPeakBatch,
    CourtPeakFrame,
    assemble_court_peak_batch,
    ordered_court_to_semantic_peaks,
    predicted_peaks_to_normalized,
    reference_context_validity,
    reference_view_mask,
)
from src.tasks.court_detection.model_io import CourtKeypointPrediction
from src.utils.schema.court import (
    COURT_PHYSICAL_INDICES_BY_SEMANTIC_CLASS,
)
from src.utils.schema.court import (
    COURT_SEMANTIC_CLASS_NAMES as SCHEMA_CLASS_NAMES,
)


def test_courtkp7_schema_has_one_shared_owner_for_sources_and_consumers() -> None:
    assert COURT_SEMANTIC_CLASS_NAMES is SCHEMA_CLASS_NAMES
    assert SEMANTIC_CLASS_NAMES is SCHEMA_CLASS_NAMES
    assert PHYSICAL_INDICES_BY_CLASS is COURT_PHYSICAL_INDICES_BY_SEMANTIC_CLASS


def test_physical_kp14_source_is_grouped_directly_into_semantic_kp7() -> None:
    court = torch.arange(28, dtype=torch.float32).reshape(1, 1, 1, 14, 2) / 28
    visible = torch.ones(1, 1, 1, 14, dtype=torch.bool)
    visible[..., 3] = False

    peaks = ordered_court_to_semantic_peaks(court, visible)

    assert peaks.uv.shape == (1, 1, 1, 7, 2, 2)
    for class_index, physical_indices in enumerate(COURT_PHYSICAL_INDICES_BY_CLASS):
        for peak_index, physical_index in enumerate(physical_indices):
            if visible[..., physical_index].item():
                torch.testing.assert_close(
                    peaks.uv[..., class_index, peak_index, :],
                    court[..., physical_index, :],
                )
    torch.testing.assert_close(
        peaks.covariance[peaks.valid],
        torch.eye(2).expand(int(peaks.valid.sum()), -1, -1) * 0.01**2,
    )
    assert bool((peaks.covariance[~peaks.valid] == 0).all())


def test_predicted_pixel_covariance_is_scaled_to_normalized_coordinates() -> None:
    uv = torch.tensor([[[20.0, 10.0]]] * 7)
    score = torch.ones(7, 1)
    covariance = torch.eye(2).reshape(1, 1, 2, 2).expand(7, 1, 2, 2).clone()
    valid = torch.ones(7, 1, dtype=torch.bool)

    normalized_uv, _, normalized_covariance, _ = predicted_peaks_to_normalized(
        uv,
        score,
        covariance,
        valid,
        image_size_hw=(21, 41),
    )

    torch.testing.assert_close(normalized_uv[0, 0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(
        normalized_covariance[0, 0], torch.diag(torch.tensor([1 / 1600, 1 / 400]))
    )


def test_peak_contract_rejects_non_psd_covariance() -> None:
    uv = torch.zeros(1, 1, 1, 7, 1, 2)
    score = torch.ones(1, 1, 1, 7, 1)
    covariance = torch.zeros(1, 1, 1, 7, 1, 2, 2)
    covariance[..., 0, 0] = -1
    valid = torch.ones_like(score, dtype=torch.bool)

    with pytest.raises(ValueError, match="positive semidefinite"):
        CourtPeakBatch(uv, score, covariance, valid)


def test_reference_contract_keeps_only_missing_reference_slot_zero() -> None:
    detection = torch.zeros(1, 2, 2, 3, dtype=torch.bool)
    detection[:, 1, 0, 2] = True
    frame_mask = torch.ones(1, 2, dtype=torch.bool)
    view_mask = torch.ones(1, 2, dtype=torch.bool)
    reference = reference_view_mask(torch.tensor([0]), view_mask)

    state = reference_context_validity(
        detection,
        frame_mask=frame_mask,
        view_mask=view_mask,
        reference_mask=reference,
        mask_invisible_observations=True,
    )

    assert state.shape == (1, 2, 2, 3)
    assert bool(state[0, :, 0, 0].all())
    assert not bool(state[0, :, 0, 1:].any())
    assert bool(state[0, 0, 1, 2])
    assert not bool(state[0, 1, 1].any())


def test_reference_index_must_select_one_unpadded_view() -> None:
    with pytest.raises(ValueError, match="inside view_mask"):
        reference_view_mask(
            torch.tensor([1]),
            torch.tensor([[True, False]]),
        )


def test_indexed_predictor_and_dataset_frames_assemble_variable_peak_capacity() -> None:
    prediction = CourtKeypointPrediction(
        keypoints=torch.tensor([[[10.0, 5.0]]] * 7),
        scores=torch.ones(7, 1),
        valid=torch.ones(7, 1, dtype=torch.bool),
        covariance=torch.eye(2).reshape(1, 1, 2, 2).expand(7, 1, 2, 2),
        heatmaps=torch.zeros(7, 2, 2),
        semantic_class_names=COURT_SEMANTIC_CLASS_NAMES,
        image_size_hw=(11, 21),
    )
    dataset_output = {
        "keypoints": torch.full((7, 5, 2), 4.0),
        "scores": torch.ones(7, 5),
        "valid": torch.tensor(
            [[False] * 5, [True] * 5, *([[True, True, False, False, False]] * 5)],
            dtype=torch.bool,
        ),
        "covariance": torch.eye(2)
        .reshape(1, 1, 2, 2)
        .expand(7, 5, 2, 2),
        "image_size": torch.tensor([9, 9]),
        "semantic_class_names": COURT_SEMANTIC_CLASS_NAMES,
    }
    frames = [
        CourtPeakFrame.from_prediction(
            prediction, batch_index=0, view_index=0, frame_index=0
        ),
        CourtPeakFrame.from_dataset_output(
            dataset_output, batch_index=0, view_index=0, frame_index=1
        ),
    ]

    peaks = assemble_court_peak_batch(frames, expected_shape_bvt=(1, 1, 2))

    assert peaks.uv.shape == (1, 1, 2, 7, 5, 2)
    assert not bool(peaks.valid[0, 0, 1, 0].any())
    assert int(peaks.valid[0, 0, 1, 1].sum()) == 5
    assert int(peaks.valid[0, 0, 0].sum()) == 7
    assert bool((peaks.score[~peaks.valid] == 0).all())


def test_indexed_peak_assembly_rejects_missing_frames_and_schema() -> None:
    frame = CourtPeakFrame(
        batch_index=0,
        view_index=0,
        frame_index=0,
        keypoints_pixels=torch.zeros(7, 1, 2),
        scores=torch.ones(7, 1),
        covariance_pixels=torch.eye(2).reshape(1, 1, 2, 2).expand(7, 1, 2, 2),
        valid=torch.ones(7, 1, dtype=torch.bool),
        image_size_hw=(10, 10),
        semantic_class_names=tuple(reversed(COURT_SEMANTIC_CLASS_NAMES)),
    )
    with pytest.raises(ValueError, match="semantic class schema"):
        assemble_court_peak_batch([frame], expected_shape_bvt=(1, 1, 1))

    valid_frame = CourtPeakFrame(
        batch_index=0,
        view_index=0,
        frame_index=0,
        keypoints_pixels=frame.keypoints_pixels,
        scores=frame.scores,
        covariance_pixels=frame.covariance_pixels,
        valid=frame.valid,
        image_size_hw=frame.image_size_hw,
        semantic_class_names=COURT_SEMANTIC_CLASS_NAMES,
    )
    with pytest.raises(ValueError, match="expected 2"):
        assemble_court_peak_batch([valid_frame], expected_shape_bvt=(1, 1, 2))

    with pytest.raises(ValueError, match="covariance"):
        CourtPeakFrame.from_dataset_output(
            {
                "keypoints": frame.keypoints_pixels,
                "scores": frame.scores,
                "valid": frame.valid,
                "image_size": torch.tensor([10, 10]),
                "semantic_class_names": COURT_SEMANTIC_CLASS_NAMES,
            },
            batch_index=0,
            view_index=0,
            frame_index=0,
        )
