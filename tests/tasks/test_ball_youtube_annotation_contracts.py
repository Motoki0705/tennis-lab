from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.annotation.youtube_session import (
    LEFT_KEYS as ANNOTATION_LEFT_KEYS,
)
from src.tasks.ball_detection.annotation.youtube_session import (
    RIGHT_KEYS as ANNOTATION_RIGHT_KEYS,
)
from src.tasks.ball_detection.annotation.youtube_session import (
    BallAnnotationSessionConfig,
    FinalizeConfig,
    ZoomConfig,
    _view_crop,
    finalize_candidate,
    frame_completion_error,
)
from src.tasks.ball_detection.data.argumentation import HorizontalFlipArgumentation
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.ball_detection.youtube.candidate_workflow import (
    LEFT_KEYS as SELECTION_LEFT_KEYS,
)
from src.tasks.ball_detection.youtube.candidate_workflow import (
    RIGHT_KEYS as SELECTION_RIGHT_KEYS,
)
from src.tasks.ball_detection.youtube.candidate_workflow import (
    CandidatePredictionConfig,
    CandidateSelectionConfig,
    SelectionState,
    _render_selection,
    _resume_index,
    _validate_selection_config,
    build_window_starts,
    create_candidate,
    predict_candidates,
)
from src.utils.data.heatmaps import heatmaps_to_peaks


def test_linux_arrow_key_codes_are_supported() -> None:
    assert 65361 in SELECTION_LEFT_KEYS
    assert 65363 in SELECTION_RIGHT_KEYS
    assert 65361 in ANNOTATION_LEFT_KEYS
    assert 65363 in ANNOTATION_RIGHT_KEYS
    assert chr(65361 & 0xFF).lower() == "q"


def test_selection_notification_is_rendered() -> None:
    state = SelectionState(
        records=[{
            "video_id": "video_000001",
            "source_frame_index": 0,
            "timestamp_sec": 0.0,
        }],
        current_index=0,
        notification="SELECTED: video_000001_clip_000001",
        notification_until=float("inf"),
    )
    canvas = _render_selection(
        np.zeros((100, 640, 3), dtype=np.uint8),
        state,
        _selection_config(),
    )
    blue, green, red = (int(value) for value in canvas[2, 2])
    assert green > blue
    assert green > red


def _selection_config() -> CandidateSelectionConfig:
    return CandidateSelectionConfig(
        resume=True,
        start_index=None,
        window_name="test",
        max_display_width=800,
        max_display_height=600,
        min_frames=8,
        copy_mode="copy",
        overwrite=False,
        skip_small=1,
        skip_medium=10,
        skip_large=50,
    )


def _annotation_config(root: Path) -> BallAnnotationSessionConfig:
    return BallAnnotationSessionConfig(
        root=root,
        video_id="video_000001",
        candidate_id=None,
        start_index=None,
        window_name="test",
        max_display_width=800,
        max_display_height=600,
        point_radius=7,
        point_thickness=2,
        max_balls_per_frame=16,
        zoom=ZoomConfig(key="z", factor=4.0),
        finalize=FinalizeConfig(key="f", overwrite=False),
    )


def _make_raw_frames(root: Path, count: int = 12) -> list[dict[str, object]]:
    raw_dir = root / "frames" / "video_000001" / "raw"
    raw_dir.mkdir(parents=True)
    records = []
    for index in range(count):
        image_path = raw_dir / f"frame_{index:08d}.jpg"
        image = np.full((24, 32, 3), index, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        records.append({
            "frame_id": f"video_000001_f{index:08d}",
            "image_path": str(image_path.relative_to(root)),
            "video_id": "video_000001",
            "source_frame_index": index,
            "timestamp_sec": index / 30.0,
            "fps": 30.0,
            "width": 32,
            "height": 24,
            "split": "train",
            "source_url": "https://example.com",
            "source_title": "Example",
        })
    return records


def test_window_starts_cover_the_final_frame() -> None:
    assert build_window_starts(frame_count=10, sequence_length=8, stride=4) == [0, 2]


def test_heatmaps_to_peaks_returns_multiple_local_maxima() -> None:
    heatmaps = torch.zeros(1, 12, 16)
    heatmaps[0, 2, 3] = 0.9
    heatmaps[0, 9, 13] = 0.8
    coords, values, valid = heatmaps_to_peaks(
        heatmaps,
        threshold=0.5,
        nms_kernel=3,
        max_peaks=4,
    )
    assert valid.sum().item() == 2
    assert values[0, :2].tolist() == pytest.approx([0.9, 0.8])
    assert coords[0, 0].tolist() == pytest.approx([3 / 15, 2 / 11])
    assert coords[0, 1].tolist() == pytest.approx([13 / 15, 9 / 11])


def test_horizontal_flip_transforms_all_ball_instances() -> None:
    transform = HorizontalFlipArgumentation({"enabled": True, "prob": 1.0})
    frames = [np.zeros((8, 10, 3), dtype=np.float32)]
    _, coords, visibility = transform.forward(
        frames,
        [[(2.0, 3.0), (7.0, 4.0)]],
        [[1.0, 1.0]],
        rng=random.Random(0),
    )
    assert coords == [[(7.0, 3.0), (2.0, 4.0)]]
    assert visibility == [[1.0, 1.0]]


def test_multi_ball_metrics_match_instances_one_to_one() -> None:
    metrics = BallDetectionMetrics(
        peak_threshold=0.5,
        ball_distance_threshold=1.5,
        nms_kernel=3,
        max_predictions_per_frame=4,
    )
    heatmaps = torch.zeros(1, 1, 10, 10)
    heatmaps[0, 0, 2, 3] = 0.9
    heatmaps[0, 0, 7, 8] = 0.8
    metrics.update(
        heatmaps,
        target_coords=torch.tensor([[[3.0, 2.0]]]),
        target_visibility=torch.tensor([[1.0]]),
        original_size=torch.tensor([[10.0, 10.0]]),
        target_instance_coords=torch.tensor([[[[3.0, 2.0], [8.0, 7.0]]]]),
        target_instance_visibility=torch.tensor([[[1.0, 1.0]]]),
    )
    result = metrics.compute()
    assert result["precision"].item() == pytest.approx(1.0)
    assert result["recall"].item() == pytest.approx(1.0)
    assert result["f1"].item() == pytest.approx(1.0)


def test_selection_skip_is_capped_at_fifty() -> None:
    _validate_selection_config(_selection_config())
    invalid = CandidateSelectionConfig(
        **{**_selection_config().__dict__, "skip_large": 51}
    )
    with pytest.raises(ValueError, match="at most 50"):
        _validate_selection_config(invalid)


def test_candidate_selection_resumes_after_last_endpoint(tmp_path: Path) -> None:
    root = tmp_path / "youtube"
    records = _make_raw_frames(root)
    staging = root / "staging" / "video_000001"
    candidate = create_candidate(
        root=root,
        video_id="video_000001",
        staging_dir=staging,
        records=records,
        start_index=1,
        end_index=8,
        config=_selection_config(),
    )

    assert candidate["status"] == "selected"
    assert candidate["raw_start_index"] == 1
    assert candidate["raw_end_index"] == 8
    assert _resume_index([candidate], len(records)) == 9
    assert len(list((staging / "clip_000001").glob("*.jpg"))) == 8
    assert not (staging / "clip_000001" / "predictions.jsonl").exists()


def test_candidate_selection_resume_can_reach_end_of_video() -> None:
    candidates = [{"raw_end_index": 11}]
    assert _resume_index(candidates, frame_count=12) == 12


def test_prediction_runs_only_after_explicit_predict_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "youtube"
    records = _make_raw_frames(root, count=8)
    staging = root / "staging" / "video_000001"
    create_candidate(
        root=root,
        video_id="video_000001",
        staging_dir=staging,
        records=records,
        start_index=0,
        end_index=7,
        config=_selection_config(),
    )

    class FakePredictor:
        model_config = {"num_frames": 8}

        def predict(
            self,
            images: torch.Tensor,
            return_heatmaps: bool = False,
        ) -> dict[str, torch.Tensor]:
            batch, frames = images.shape[:2]
            heatmaps = torch.zeros(batch, frames, 24, 32)
            heatmaps[:, :, 6, 8] = 0.9
            heatmaps[:, :, 18, 24] = 0.8
            return {
                "coords": torch.full((batch, frames, 2), 0.5),
                "visibility": torch.full((batch, frames), 0.9),
                **({"heatmaps": heatmaps} if return_heatmaps else {}),
            }

    monkeypatch.setattr(
        "src.tasks.ball_detection.youtube.candidate_workflow."
        "BallDetectionPredictor.load_from_checkpoint",
        lambda *args, **kwargs: FakePredictor(),
    )
    predict_candidates(
        root=root,
        video_id="video_000001",
        staging_dir=staging,
        config=CandidatePredictionConfig(
            checkpoint=tmp_path / "model.ckpt",
            device="cpu",
            sequence_length=8,
            window_stride=1,
            batch_size=2,
            image_size=(24, 32),
            normalize_imagenet=False,
            imagenet_mean=(0.485, 0.456, 0.406),
            imagenet_std=(0.229, 0.224, 0.225),
            peak_threshold=0.5,
            nms_kernel=3,
            max_candidates_per_frame=4,
            aggregation="mean_heatmap",
            overwrite=False,
        ),
    )

    candidate_path = staging / "clip_000001" / "candidate.json"
    candidate = json.loads(candidate_path.read_text())
    assert candidate["status"] == "pseudo_labeled"
    assert all(frame["review_status"] == "pending" for frame in candidate["frames"])
    assert all(len(frame["predictions"]["candidates"]) == 2 for frame in candidate["frames"])
    assert all(len(frame["balls"]) == 2 for frame in candidate["frames"])
    assert (candidate_path.parent / "predictions.jsonl").exists()


@pytest.mark.parametrize(
    ("ball", "expected"),
    [
        ({"ball_id": "b001", "state": "visible", "x": 10.0, "y": 20.0}, None),
        ({"ball_id": "b001", "state": "occluded", "x": 10.0, "y": 20.0}, None),
        ({"ball_id": "b001", "state": "out_of_frame", "x": None, "y": None}, None),
        (
            {"ball_id": "b001", "state": "visible", "x": None, "y": None},
            "ball[0]: state=visible requires x/y coordinates",
        ),
        (
            {"ball_id": "b001", "state": "unreviewed", "x": None, "y": None},
            "ball[0]: unsupported ball state 'unreviewed'",
        ),
    ],
)
def test_frame_completion_contract(ball: dict[str, object], expected: str | None) -> None:
    assert frame_completion_error({"balls": [ball]}) == expected


def test_empty_frame_is_a_valid_no_ball_annotation() -> None:
    assert frame_completion_error({"balls": []}) is None


def test_prediction_centered_zoom_crops_around_prediction() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    crop, origin_x, origin_y = _view_crop(
        image,
        prediction=(100.0, 50.0),
        zoom_enabled=True,
        zoom_factor=4.0,
    )
    assert crop.shape[:2] == (32, 50)
    assert (origin_x, origin_y) == (75, 34)


def test_finalize_moves_only_completed_candidate_to_training_dataset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "youtube"
    records = _make_raw_frames(root, count=8)
    staging = root / "staging" / "video_000001"
    create_candidate(
        root=root,
        video_id="video_000001",
        staging_dir=staging,
        records=records,
        start_index=0,
        end_index=7,
        config=_selection_config(),
    )
    candidate_path = staging / "clip_000001" / "candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["status"] = "annotating"
    for index, frame in enumerate(candidate["frames"]):
        visible = index % 2 == 0
        frame["review_status"] = "completed"
        frame["predictions"] = {"candidates": []}
        frame["balls"] = (
            [
                {
                    "ball_id": "b001",
                    "prediction_id": None,
                    "x": 12.0,
                    "y": 8.0,
                    "state": "visible",
                    "role": "target",
                    "confidence": None,
                    "label_source": "manual",
                },
                {
                    "ball_id": "b002",
                    "prediction_id": None,
                    "x": 20.0,
                    "y": 16.0,
                    "state": "visible",
                    "role": "secondary",
                    "confidence": None,
                    "label_source": "manual",
                },
            ]
            if visible
            else []
        )
    candidate_path.write_text(json.dumps(candidate))

    clip_id = finalize_candidate(candidate_path, _annotation_config(root))

    clip_dir = root / "frames" / "video_000001" / "clip_000001"
    assert clip_id == "video_000001_clip_000001"
    assert not candidate_path.parent.exists()
    assert (clip_dir / "clip.json").exists()
    with (clip_dir / "Label.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert sum(int(row["visibility"]) for row in rows) == 8
    assert {row["instance id"] for row in rows if row["visibility"] == "1"} == {
        "b001",
        "b002",
    }

    annotation = json.loads((root / "annotations" / "train.json").read_text())
    assert annotation["items"][0]["dataset_entry"] == (
        "youtube/frames/video_000001/clip_000001"
    )
    dataset = BallDetectionDataset(
        data_dir=root.parent,
        split_file=root / "annotations" / "train.txt",
        config=OmegaConf.create({
            "model": {"num_frames": 8},
            "data": {
                "sample_stride": 1,
                "image_size": [24, 32],
                "heatmap_size": [12, 16],
                "sigma_ratio": 0.012,
            },
        }),
    )
    sample = dataset[0]
    assert tuple(sample["images"].shape) == (8, 3, 24, 32)
    assert sample["visibility"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert tuple(sample["instance_coords"].shape) == (8, 8, 2)
    assert tuple(sample["instance_visibility"].shape) == (8, 8)
    assert sample["instance_visibility"][0, :2].tolist() == [1.0, 1.0]
    assert sample["instance_visibility"][1].sum().item() == 0.0


def test_finalize_rejects_pending_candidate(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "youtube/staging/video_000001/clip_000001"
    candidate_dir.mkdir(parents=True)
    path = candidate_dir / "candidate.json"
    path.write_text(json.dumps({
        "clip_id": "video_000001_clip_000001",
        "frames": [{"frame_id": "frame_0", "review_status": "pending"}],
    }))
    with pytest.raises(ValueError, match="incomplete frames"):
        finalize_candidate(path, _annotation_config(tmp_path / "youtube"))


def test_dataset_accepts_lowercase_clip_directory(tmp_path: Path) -> None:
    clip_dir = tmp_path / "youtube" / "frames" / "video_000001" / "clip_000001"
    clip_dir.mkdir(parents=True)
    assert BallDetectionDataset._is_clip_dir(clip_dir)
