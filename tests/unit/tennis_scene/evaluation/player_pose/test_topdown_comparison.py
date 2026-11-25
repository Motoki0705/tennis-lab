"""Test top-down pose row identity, coverage and backend declaration on CPU."""

from __future__ import annotations

import hashlib
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
from src.utils.checksum import FileIntegrityError, dual_sha256
from src.utils.inference_benchmark import Backend, ConditionUnavailable, Status
from src.utils.io import write_json_atomic
from src.utils.schema.player import COCO_KP_IDX, COCO_KP_NAMES
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


@pytest.fixture
def compare_speed(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(RUN.parent / "run-person-model-speed-baseline"))
    monkeypatch.syspath_prepend(str(RUN))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return import_module("compare_speed")


@pytest.fixture
def comparison_cli(
    compare_speed: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, Path]:
    index = json.loads(compare_speed.COMPARISON_INPUTS.read_bytes())
    index_path = tmp_path / "comparison_inputs.json"
    output = tmp_path / "out"
    monkeypatch.setattr(compare_speed, "COMPARISON_INPUTS", index_path)
    # Exercise the CLI evidence checks without the local video and crop cache.
    monkeypatch.setattr(compare_speed, "PinnedBenchmarkInputs", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_speed.py", "--inputs", str(tmp_path), "--output", str(output)],
    )
    return index, index_path, output


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

        return stages.measure_forward(Backend.PT32, forward, name="pose")

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
        stages.measure_forward(Backend.PT16, unexpected)
    assert error.value.status is Status.BACKEND_MISMATCH


@pytest.mark.parametrize("model", ["vitpose-h", "rtmpose-m"])
def test_replay_checks_fixed_rows_and_can_use_retained_queue_results(
    run_modules: tuple[ModuleType, ...], tmp_path: Path, model: str
) -> None:
    evaluate, fixed_inputs, contract = run_modules
    replay = import_module("replay_accuracy")
    rtmpose = import_module("rtmpose_accuracy")
    key = FrameKey("video/clip", "cam2", 4)
    refs = {
        key: ReferenceFrame(
            np.array([21, 7]),
            np.array([[1.0, 2.0, 4.0, 6.0], [8.0, 3.0, 9.0, 12.0]]),
            np.ones((2, 17, 2)),
            np.ones((2, 17), bool),
        ),
    }
    keys = fixed_inputs.row_keys(refs)
    columns = replay.row_columns(refs, keys)
    points: NDArray[np.float32] = np.full((2, 17, 3), np.nan, np.float32)
    points[1, 9] = (3.0, 4.0, 0.25)
    poses = contract.PoseBatch(points)
    source, queue = tmp_path / "inference", tmp_path / "queue"
    source.mkdir()
    stem = "pose-accuracy"
    repro = queue / "repro" / stem
    repro.mkdir(parents=True)
    (repro / "output_dir.txt").write_text(str(source) + "\n")
    (repro / "repro.sh").write_text("echo replay\n")
    write_json_atomic(repro / "run.json", {"run_id": stem, "commit": "a" * 40})
    log = queue / "logs" / f"{stem}.log"
    log.parent.mkdir()
    log.write_text("completed\n")
    artifact = evaluate.save_predictions(source, refs, keys, poses)
    scores = {
        "overall": {"pose": {"with_detection_coverage": {"mean_oks": 0.25}}},
        "finite_pairs": {"overall": {"mean_oks": 0.75}},
    }
    write_json_atomic(source / "scores.json", scores)
    write_json_atomic(
        source / "receipt.json",
        {
            "commit": "a" * 40,
            "status": "ok",
            "bundle_id": "cohort",
            "bundle_sha256": "b" * 64,
            "model": model,
            "backend": "PT32",
            "rows": 2,
            "batch_size": 1,
            "joint_order": COCO_KP_NAMES,
            "weights": {
                "checkpoint": {"sha256": rtmpose.MODELS[model].sha256, "bytes": 128}
            }
            if model == "rtmpose-m"
            else {},
            "engines": {},
            "environment": {},
            "predictions": artifact,
            "scores": {
                "path": "scores.json",
                "sha256": dual_sha256(source / "scores.json"),
            },
        },
    )
    arguments = (queue, f"{stem}.job", ("cohort", "b" * 64))
    wrong = {**columns, "player_id": np.array([7, 21])}
    with pytest.raises(ValueError, match="player_id differs"):
        replay.read_condition(*arguments, wrong, retain=True)
    assert not (repro / "predictions").exists()

    case, retained, previous, record = replay.read_condition(
        *arguments, columns, retain=True
    )
    assert case == (model, Backend.PT32)
    assert previous == scores
    assert record["prediction_sha256"] == artifact["sha256"]
    np.testing.assert_array_equal(retained.keypoints_xyc, points)
    if model == "rtmpose-m":
        # RTMPose writes its prediction bundle directly into repro.
        (repro / "predictions" / "queue.log").unlink()
        with pytest.raises(FileIntegrityError, match="predictions/queue.log"):
            replay.read_condition(*arguments, columns)
        replay.read_condition(*arguments, columns, retain=True)
    (source / artifact["path"]).unlink()
    log.unlink()
    _, copied, _, copied_record = replay.read_condition(*arguments, columns)
    np.testing.assert_array_equal(copied.keypoints_xyc, points)
    assert copied_record["log"]["path"] == f"repro/{stem}/predictions/queue.log"
    assert copied_record["log"]["sha256"] == record["log"]["sha256"]
    if model == "rtmpose-m":
        dedicated_case, dedicated_poses, dedicated_record = rtmpose.read_condition(
            *arguments, columns
        )
        assert dedicated_case == case
        np.testing.assert_array_equal(dedicated_poses, points)
        assert dedicated_record["files"]["log"] == copied_record["log"]


@pytest.mark.usefixtures("run_modules")
def test_overlay_uses_the_same_source_frame_and_eligible_joints() -> None:
    render = import_module("render_overlays")
    before = FrameKey("video/clip", "cam2", 2)
    target = FrameKey("video/clip", "cam2", 4)
    points = np.zeros((1, 17, 2))
    points[0, 0] = (4, 2)
    points[0, 1] = (5, 1)
    eligible: NDArray[np.bool_] = np.zeros((1, 17), bool)
    eligible[0, 0] = True
    reference = ReferenceFrame(
        np.array([5]), np.array([[1.0, 1.5, 6.0, 3.5]]), points, eligible
    )
    bounds = render.frame_slices({target: reference, before: reference})
    predicted: NDArray[np.float32] = np.full((2, 17, 3), np.nan, np.float32)
    predicted[0, 0] = (2, 2, 1)
    predicted[1, 0] = (5, 2, 1)
    image: NDArray[np.uint8] = np.zeros((4, 8, 3), np.uint8)
    grid = render._sheet(
        image, target, bounds[target], reference, {"model PT32": predicted}, None
    )
    assert grid.shape == (392, 1280, 3)
    # Both panels map the image's pixels at 80 display pixels per source pixel.
    assert grid[212, 320].tolist() == list(render.COLORS[1])
    assert grid[212, 1040].tolist() == list(render.COLORS[1])
    assert grid[212, 800].tolist() == [0, 0, 0]
    assert grid[132, 400].tolist() == [0, 0, 0]


@pytest.mark.usefixtures("run_modules")
def test_overlay_preserves_the_timing_of_irregular_source_frames() -> None:
    render = import_module("render_overlays")
    sampled = [(10, "a"), (12, "b"), (13, "c"), (15, "d")]
    assert list(render.hold_source_frames(sampled)) == ["a", "a", "b", "c", "c", "d"]
    with pytest.raises(ValueError, match="strictly increasing"):
        list(render.hold_source_frames([(2, "a"), (2, "b")]))


@pytest.mark.usefixtures("run_modules")
def test_wrist_speed_uses_original_timestamps_and_stops_at_inference_gaps() -> None:
    difficulty = import_module("pose_difficulty")
    frames = np.array([0, 2, 6, 7, 9])
    joints: NDArray[np.float64] = np.full((1, 10, 17, 3), np.nan)
    left, right = COCO_KP_IDX["left_wrist"], COCO_KP_IDX["right_wrist"]
    joints[0, frames[:4], left] = [(0, 0, 0), (0.3, 0, 0), (1, 0, 0), (1.3, 0, 0)]
    joints[0, :3:2, right] = [(0, 0, 0), (0, 0.4, 0)]
    inferred: NDArray[np.bool_] = np.ones((1, 10), bool)
    inferred[0, 9] = False
    speed = difficulty.wrist_speeds(joints, inferred, frames, 60.0)
    assert speed[0, 2] == pytest.approx(12.0)
    assert speed[0, 7] == pytest.approx(18.0)
    assert np.isnan(speed[0, [0, 6, 9]]).all()


def test_difficulty_counts_missing_and_aligns_jitter_with_the_center_row(
    run_modules: tuple[ModuleType, ...],
) -> None:
    _, inputs, contract = run_modules
    difficulty = import_module("pose_difficulty")
    keys = tuple(
        inputs.RowKey(FrameKey("video/clip", camera, frame), 5)
        for camera, frame in (("cam0", 0), ("cam0", 2), ("cam0", 4), ("cam1", 4))
    )
    windows = difficulty.temporal_triplets(keys)
    assert windows.tolist() == [[0, 1, 2]]
    xy: NDArray[np.float64] = np.zeros((4, 17, 2), np.float64)
    valid: NDArray[np.bool_] = np.zeros((4, 17), bool)
    valid[:, :2] = True
    poses: NDArray[np.float32] = np.full((4, 17, 3), np.nan, np.float32)
    poses[:, 0] = [(1, 0, -0.5), (2, 0, 0.0), (5, 0, 1.0), (10, 0, 1.0)]
    poses[0, 1] = poses[2, 1] = (2, 2, 0.4)
    group = np.array([False, True, False, False])
    cohort = difficulty.Difficulty(
        xy, valid, np.full(4, 100.0), {"fast_wrist": group}, windows, {}
    )
    result = difficulty.joint_metrics(cohort, contract.PoseBatch(poses))["fast_wrist"]
    assert result["eligible_joints"] == 2 and result["missing_joints"] == 1
    assert result["pck"] == {"0.1": 0.0, "0.2": 0.5}
    assert result["by_joint"]["nose"]["jitter_px"]["median"] == pytest.approx(2.0)
    eye = result["by_joint"]["left_eye"]
    assert eye["eligible_temporal_triplets"] == 1
    assert eye["jitter_px"]["count"] == 0 and eye["coverage"] == 0.0


def test_cli_rejects_replacement_of_only_the_report_summary(
    compare_speed: ModuleType,
    comparison_cli: tuple[dict[str, Any], Path, Path],
    tmp_path: Path,
) -> None:
    index, index_path, output = comparison_cli
    condition = {"model": "rtmpose-x", "backend": "TRT16", "batch_size": 1}
    report = {
        "case": {"name": condition["model"], "backend": "TRT16", "batch_size": 1},
        "summary": {"stage_wall": {"p50_ms_per_item": 9.0}},
    }
    original = json.dumps(report).encode()
    index["reports"] = [{**condition, "sha256": hashlib.sha256(original).hexdigest()}]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    path = tmp_path / "report.json"
    path.write_bytes(original.replace(b"9.0", b"1.0"))
    sys.argv.append(str(path))

    with pytest.raises(ValueError, match="Report SHA-256 differs from the fixed"):
        compare_speed.main()
    assert not output.exists()


@pytest.mark.parametrize(
    ("log_text", "error"),
    [
        (None, "Terminal log does not exist"),
        ("rtmpose-x/TRT16/B4 infeasible\n", "does not record a failure"),
        ("rtmpose-x/TRT16/B1 ok\n", "does not record a failure"),
    ],
)
def test_cli_rejects_omission_without_terminal_failure_for_that_case(
    compare_speed: ModuleType,
    comparison_cli: tuple[dict[str, Any], Path, Path],
    tmp_path: Path,
    log_text: str | None,
    error: str,
) -> None:
    index, index_path, output = comparison_cli
    log = tmp_path / "terminal.log"
    if log_text is not None:
        content = log_text.encode()
        log.write_bytes(content)
        index["jobs"].append(
            {"state": "done", "log": {"sha256": hashlib.sha256(content).hexdigest()}}
        )
    index["reports"] = []
    index_path.write_text(json.dumps(index), encoding="utf-8")
    omissions = tmp_path / "omissions.json"
    omissions.write_text(
        json.dumps(
            [
                {
                    "model": "rtmpose-x",
                    "backend": "TRT16",
                    "batch_size": 1,
                    "reason": "Conversion failed",
                    "evidence": str(log),
                }
            ]
        ),
        encoding="utf-8",
    )
    sys.argv.extend(("--omissions", str(omissions)))

    with pytest.raises(ValueError, match=error):
        compare_speed.main()
    assert not output.exists()


def test_terminal_log_covers_a_failed_adapter_without_a_report(
    compare_speed: ModuleType, tmp_path: Path
) -> None:
    log = tmp_path / "terminal.log"
    content = b"rtmpose-x/TRT16/B1 process exit 1, log case-logs/TRT16-b1.log\n"
    log.write_bytes(content)
    compare_speed._verify_omission(
        ("rtmpose-x", "TRT16", 1), log, {hashlib.sha256(content).hexdigest()}
    )


def test_comparison_labels_both_batches_in_rows_and_plot(
    compare_speed: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    model = ("vitpose-h", "TRT16")
    assets = {"weights": {}, "engines": {}, "batch_size": 8}
    score = {
        "rows": "5",
        "eligible_joints": "73",
        "missing_joints": "2",
        "oks": "0.76",
        "pck_0_10": "0.68",
        "pck_0_20": "0.91",
        "p95_px": "9.0",
        "evidence": "fixture.job",
    }
    speed = {
        "speed_status": "ok",
        "wall_p50_ms_person": 2.5,
        "nvml_peak_mib": 100,
        "persons_per_s": 400,
        "record": {**assets, "nvml_device_peak_sampled_bytes": 100 * 2**20},
    }
    rows = compare_speed._compare(
        {(*model, batch): speed for batch in (1, 4, 8)}, {model: score}, {model: assets}
    )
    assert [(row["speed_batch_size"], row["accuracy_batch_size"]) for row in rows] == [
        (1, 8),
        (4, 8),
        (8, 8),
    ]

    def check_title(figure: Figure, path: Path, **kwargs: Any) -> None:
        title = figure.get_suptitle()
        assert "Speed and VRAM at batch 1" in title
        assert "Accuracy from batch 8 predictions" in title
        assert "batches 1 and 4 unmeasured" in title
        path.touch()

    monkeypatch.setattr(Figure, "savefig", check_title)
    image = tmp_path / "pareto.png"
    compare_speed._plot(rows, image)
    assert image.is_file()
