"""Test top-down pose row identity, coverage and backend declaration on CPU."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.evaluation.player_pose.bundle import MANIFEST_SCHEMA
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, ReferenceFrame
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend, ConditionUnavailable, Status
from src.utils.schema.player import COCO_KP_NAMES
from src.utils.video import FramePacket

RUN = (
    Path(__file__).resolve().parents[5]
    / "knowledge/runs/run-meiji-topdown-pose-comparison"
)


@pytest.fixture
def run_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ...]:
    monkeypatch.syspath_prepend(str(RUN))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return tuple(
        import_module(name) for name in ("evaluate", "fixed_inputs", "pose_contract")
    )


def test_cohort_checks_the_published_source_video_record(
    run_modules: tuple[ModuleType, ...], tmp_path: Path
) -> None:
    _, fixed_inputs, _ = run_modules
    bundle, media = tmp_path / "bundle", tmp_path / "media"
    bundle.mkdir()
    media.mkdir()
    video = media / "source.mp4"
    video.write_bytes(b"immutable video")
    camera = {
        "camera_id": "cam2",
        "path": video.name,
        "sha256": dual_sha256(video),
        "num_frames": 1,
    }
    source = media / "manifest.json"
    source.write_text(
        json.dumps(
            {
                "benchmark_id": "source",
                "clips": [{"clip_id": "video/clip", "cameras": [camera]}],
            }
        )
    )
    label = bundle / "pose.npz"
    np.savez(
        label,
        keypoints_px=np.ones((1, 1, 1, 17, 2)),
        boxes_xyxy=np.array([[[[1.0, 1.0, 7.0, 9.0]]]]),
        box_reasons=np.zeros((1, 1, 1), np.uint8),
        joint_reasons=np.zeros((1, 1, 1, 17), np.uint8),
        player_ids=[8],
        camera_ids=["cam2"],
        evaluation_frames=[0],
    )
    manifest = bundle / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark_id": "cohort",
                "schema": MANIFEST_SCHEMA,
                "joint_order": COCO_KP_NAMES,
                "source_benchmark": {
                    "benchmark_id": "source",
                    "manifest_sha256": dual_sha256(source),
                },
                "clips": [
                    {
                        "clip_id": "video/clip",
                        "player_ids": [8],
                        "camera_ids": ["cam2"],
                        "evaluation_frames": [0],
                        "source_frames": 1,
                        "width": 10,
                        "height": 12,
                        "source_videos": [camera],
                        "pseudo_ground_truth": {
                            "path": label.name,
                            "sha256": dual_sha256(label),
                        },
                    }
                ],
            }
        )
    )
    digest = dual_sha256(manifest)
    identity, refs, streams = fixed_inputs.load_cohort(bundle, digest, media)
    assert identity == "cohort"
    assert list(refs) == [FrameKey("video/clip", "cam2", 0)]
    assert streams[0].path == video
    video.write_bytes(b"changed video")
    with pytest.raises(ValueError, match="SHA-256"):
        fixed_inputs.load_cohort(bundle, digest, media)


def test_topdown_score_keeps_original_frame_and_player_identity(
    run_modules: tuple[ModuleType, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluate, fixed_inputs, contract = run_modules
    first, second, empty = (
        FrameKey("video/clip", "cam2", frame) for frame in (1, 3, 5)
    )
    input_path = tmp_path / "source.mp4"
    video = fixed_inputs.Video(input_path, "cam2", "video/clip", 32, 24, (1, 3, 5))

    def frames(path: Path, *, max_frames: int) -> list[Any]:
        assert path == input_path and max_frames == 6
        return [
            FramePacket(frame, np.full((24, 32, 3), frame, np.uint8), (32, 24))
            for frame in range(6)
        ]

    monkeypatch.setattr(fixed_inputs, "OpenCVVideoFrameReader", frames)

    def reference(
        ids: list[int], boxes: list[list[int]], noses: list[list[int]]
    ) -> ReferenceFrame:
        points = np.full((len(ids), 17, 2), np.nan)
        points[:, 0] = np.array(noses).reshape(-1, 2)
        return ReferenceFrame(
            np.array(ids, np.int64),
            np.array(boxes, float).reshape(-1, 4),
            points,
            np.isfinite(points).all(axis=2),
        )

    refs = {
        second: reference([9], [[8, 2, 18, 12]], [[9, 4]]),
        empty: reference([], [], []),
        first: reference([9, 4], [[4, 2, 14, 12], [20, 6, 30, 16]], [[5, 4], [20, 8]]),
    }
    batch_sizes = []

    def infer(rows: tuple[Any, ...], stages: Any) -> Any:
        batch_sizes.append(len(rows))

        def forward() -> Any:
            predictions: NDArray[np.float32] = np.full(
                (len(rows), 17, 3), np.nan, np.float32
            )
            for i, row in enumerate(rows):
                x, y = row.bbox_xyxy[:2]
                assert (
                    row.frame_bgr.dtype == np.uint8
                    and row.bbox_xyxy.dtype == np.float32
                )
                if row.frame_bgr[0, 0, 0] == 1:
                    predictions[i, 0] = [x + 1, y + 2, -1 + i]
            return contract.PoseBatch(
                predictions,
                {"visibility": np.array([row.bbox_xyxy[0] for row in rows])},
            )

        return stages.forward(Backend.PT32, forward)

    keys, result = evaluate.predict_rows(
        fixed_inputs.pose_inputs([video], refs), infer, Backend.PT32, 2
    )
    assert batch_sizes == [2, 1]
    assert keys == ((first, 9), (first, 4), (second, 9))
    report = evaluate.score_fixed(refs, keys, result)
    fixed = report["overall"]["pose"]["with_detection_coverage"]
    finite = report["finite_pairs"]["overall"]
    assert report["images"] == 3
    assert fixed["eligible_joints"] == 3 and fixed["predicted_joints"] == 2
    assert fixed["pixel_error_px"]["median"] == pytest.approx(0.5)
    assert fixed["pck"]["0.1"] == pytest.approx(2 / 3)
    assert finite["pck"]["0.1"] == 1.0
    assert (
        report["by_player"]["video/clip"]["4"]["pose"]["matched_only"][
            "pixel_error_px"
        ]["mean"]
        == 1.0
    )
    with pytest.raises(ValueError, match="pinned player addresses"):
        evaluate.score_fixed(refs, keys[::-1], result)
    artifact = evaluate.save_predictions(tmp_path, refs, keys, result)
    path = tmp_path / artifact["path"]
    assert artifact["sha256"] == dual_sha256(path)
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["source_frame"].tolist() == [1, 1, 3]
        assert arrays["player_id"].tolist() == [9, 4, 9]
        assert arrays["extra__visibility"].tolist() == [4, 20, 8]
        assert np.isnan(arrays["keypoints_xyc"][-1]).all()


def test_reader_rejects_a_missing_requested_frame(
    run_modules: tuple[ModuleType, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, fixed_inputs, _ = run_modules
    monkeypatch.setattr(
        fixed_inputs,
        "OpenCVVideoFrameReader",
        lambda *args, **kwargs: [
            FramePacket(0, np.zeros((3, 4, 3), np.uint8), (4, 3)),
            FramePacket(1, np.zeros((3, 4, 3), np.uint8), (4, 3)),
        ],
    )
    video = fixed_inputs.Video(Path("video.mp4"), "cam2", "video/clip", 4, 3, (1, 3))
    with pytest.raises(ValueError, match="Missing source frame 3"):
        list(fixed_inputs.read_video(video))


def test_accuracy_stages_reject_a_forward_declared_with_another_backend(
    run_modules: tuple[ModuleType, ...],
) -> None:
    evaluate, _, _ = run_modules
    stages = evaluate.AccuracyStages(Backend.TRT16)

    def unexpected() -> None:
        pytest.fail("wrong backend was executed")

    with pytest.raises(ConditionUnavailable) as error:
        stages.forward(Backend.PT16, unexpected)
    assert error.value.status is Status.BACKEND_MISMATCH
