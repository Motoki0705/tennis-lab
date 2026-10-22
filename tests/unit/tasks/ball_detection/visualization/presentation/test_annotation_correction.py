"""Classification, strict retained-input validation and real CPU video export."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import DictConfig
from PIL import Image

from src.tasks.annotation.migration.legacy_format.contracts import (
    Ball,
    BallFrameAnnotation,
    sha256_file,
)
from src.tasks.ball_detection.visualization.presentation.annotation_correction import (
    comparison_layout,
    draw_comparison_frame,
    playback_frames,
    render_comparison,
    run_visualization,
)
from src.tasks.ball_detection.visualization.presentation.annotation_correction_config import (
    build_runtime_config,
)
from src.tasks.ball_detection.visualization.presentation.annotation_correction_frames import (
    open_frame_input,
)
from src.tasks.ball_detection.visualization.presentation.annotation_correction_inputs import (
    Candidate,
    Correction,
    classify_frame,
    load_comparison,
    prepared_clips,
    rank_clips,
)
from src.tasks.base.annotation.media import encode_video, probe_video
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.ball_detection.store import frame as stored_frame
from tests.support.tasks.ball_detection.store import write_store_clip


def _peak(x: float, y: float, score: float = 0.9) -> Candidate:
    return Candidate(center_px=[x, y], score=score)


def _ball(
    center: list[float] | None, *, status: str = "visible", track_id: str = "ball"
) -> Ball:
    return Ball.model_validate(
        {
            "track_id": track_id,
            "center_px": center,
            "status": status,
            "interpolation_frames": [0, 2] if status == "interpolated" else None,
        }
    )


def _frame(*balls: Ball, reviewed: bool = True, index: int = 0) -> BallFrameAnnotation:
    return BallFrameAnnotation(
        frame_index=index,
        reviewed=reviewed,
        balls=list(balls),
        interpolation_break=False,
        notes="",
    )


@pytest.mark.parametrize(
    "peaks,frame,expected",
    [
        ([_peak(10, 10)], _frame(_ball([10, 10])), "agree"),
        ([_peak(10, 10)], _frame(_ball([18, 10])), "agree"),
        ([_peak(10, 10)], _frame(_ball([18.001, 10])), "moved"),
        ([_peak(10, 10), _peak(50, 50, 0.5)], _frame(_ball([58, 50])), "rerank"),
        ([_peak(10, 10)], _frame(_ball([50, 50])), "moved"),
        ([], _frame(_ball([50, 50])), "added"),
        ([], _frame(_ball([50, 50], status="occluded")), "added"),
        ([], _frame(_ball([50, 50], status="interpolated")), "added"),
        ([_peak(10, 10)], _frame(), "removed"),
        ([_peak(10, 10)], _frame(_ball(None, status="out_of_frame")), "removed"),
        ([_peak(10, 10)], _frame(_ball(None, status="unresolved")), "removed"),
        ([], _frame(_ball(None, status="occluded")), "both_none"),
        ([], _frame(), "both_none"),
        ([_peak(10, 10)], _frame(_ball([50, 50]), reviewed=False), "unreviewed"),
        ([], _frame(reviewed=False), "unreviewed"),
    ],
)
def test_classify_every_class_and_inclusive_radius(
    peaks: list[Candidate], frame: BallFrameAnnotation, expected: str
) -> None:
    assert classify_frame(peaks, frame, 8) == expected


@pytest.mark.parametrize(
    "positions,expected",
    [
        ([[10, 10], [50, 50]], "rerank"),
        ([[50, 50], [90, 90]], "moved"),
        ([[10, 10], None], "removed"),
        ([[90, 90], None], "removed"),
    ],
)
def test_multiple_tracks_take_strongest_class(
    positions: list[list[float] | None], expected: str
) -> None:
    frame = _frame(
        *[
            _ball(
                position,
                status="unresolved" if position is None else "visible",
                track_id=f"b{i}",
            )
            for i, position in enumerate(positions)
        ]
    )
    peaks = [_peak(10, 10), _peak(50, 50, 0.5)]
    assert classify_frame(peaks, frame, 8) == expected
    # No proposals: a located track outranks a null-position track.
    assert classify_frame([], frame, 8) == Correction.ADDED
    assert (
        classify_frame(peaks, frame.model_copy(update={"reviewed": False}), 8)
        == Correction.UNREVIEWED
    )


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), True])
def test_invalid_radius_is_rejected(radius: float) -> None:
    with pytest.raises(ValueError, match="radius"):
        classify_frame([], _frame(), radius)


@dataclass(frozen=True)
class Inputs:
    root: Path
    campaign: Path
    video: Path
    manifest_path: Path
    annotation_path: Path
    cache_path: Path


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    root, campaign = tmp_path / "annotations", tmp_path / "campaign"
    video = root / "videos/source/testclip.mp4"
    video.parent.mkdir(parents=True)
    frames = [
        av.VideoFrame.from_ndarray(
            np.full((90, 160, 3), (30 + i * 10, 60, 90), dtype=np.uint8), format="rgb24"
        )
        for i in range(8)
    ]
    encode_video(
        video,
        width=160,
        height=90,
        time_base=Fraction(1, 1000),
        rate=Fraction(10),
        frames=((frame, i * 100, 100) for i, frame in enumerate(frames)),
        crf=17,
        preset="fast",
    )
    timeline = probe_video(video)
    manifest = {
        "schema_version": "tennis_chat_clip.v1",
        "kit_version": "test",
        "kit_id": "c" * 64,
        "clip_id": "local",
        "filename": video.name,
        "sha256": sha256_file(video),
        "bytes": video.stat().st_size,
        "width": 160,
        "height": 90,
        "time_base": str(timeline.time_base),
        "nominal_fps": str(timeline.rate),
        "source_start_pts": 0,
        "media_range": {"start": 0, "stop": 8},
        "target_range": {"start": 0, "stop": 8},
        "policies": {"ball_max_gap_seconds": 0.1},
        "source": {
            "source_id": "source",
            "youtube_id": None,
            "url": None,
            "title": "synthetic",
            "filename": "source.mp4",
            "sha256": "a" * 64,
            "bytes": 100,
            "acquired_at": "2026-10-20",
        },
        "frames": [
            {
                "frame_index": i,
                "source_frame_index": i,
                "source_pts": pts,
                "clip_pts": pts,
                "duration_pts": duration,
                "is_target": True,
            }
            for i, (pts, duration) in enumerate(
                zip(timeline.pts, timeline.durations, strict=True)
            )
        ],
    }
    annotation_frames = [
        _frame(_ball([30, 30]), index=0),
        _frame(_ball([30, 30]), index=1),
        _frame(_ball([90, 60]), index=2),
        _frame(_ball([50, 30], status="occluded"), index=3),
        _frame(index=4),
        _frame(index=5),
        _frame(_ball([30, 30]), reviewed=False, index=6),
        _frame(_ball(None, status="out_of_frame"), index=7),
    ]
    annotation = {
        "schema_version": "tennis_chat_ball_annotation.v1",
        "clip_id": "testclip",
        "width": 160,
        "height": 90,
        "frame_count": 8,
        "status": "partial",
        "issues": ["One unreviewed frame"],
        "frames": [f.model_dump() for f in annotation_frames],
    }
    peaks = [
        [_peak(30, 30)],
        [_peak(10, 10), _peak(30, 30, 0.5)],
        [_peak(10, 10)],
        [],
        [_peak(10, 10)],
        [],
        [_peak(10, 10)],
        [],
    ]
    cache = {
        "schema": "campaign_ball_candidates.v1",
        "proposals_only": True,
        "identity": {
            "video_sha256": manifest["sha256"],
            "checkpoint_sha256": "b" * 64,
            "threshold": 0.05,
            "max_peaks": 3,
        },
        "frames": {
            str(i): {"candidates": [p.model_dump() for p in ps]}
            for i, ps in enumerate(peaks)
        },
    }
    result = Inputs(
        root,
        campaign,
        video,
        root / "_preparation/source/run/clips/local/clip_manifest.json",
        root / "annotated/processed/ball/testclip.json",
        campaign / "cache/cands_ball/testclip.json",
    )
    _save(result.manifest_path, manifest)
    _save(result.annotation_path, annotation)
    _save(result.cache_path, cache)
    return result


def _comparison(inputs: Inputs):
    return load_comparison(
        inputs.root,
        inputs.campaign,
        "testclip",
        radius_px=8,
        prepared=prepared_clips(inputs.root)["testclip"],
    )


def _config(
    inputs: Inputs, *, mode: str = "rank", overrides: Sequence[str] = ()
) -> DictConfig:
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks/ball_detection/configs"),
    ):
        return compose(
            config_name="visualize_annotation_correction",
            overrides=[
                f"paths.annotation_root={inputs.root}",
                f"paths.campaign_dir={inputs.campaign}",
                f"paths.output_root={inputs.root.parent / 'runs'}",
                f"mode={mode}",
                "clip_id=null",
                "frames.range=null",
                "frames.source=video",
                "frames.store_dir=null",
                *overrides,
            ],
        )


def test_selected_material_defaults_are_reproducible(inputs: Inputs) -> None:
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks/ball_detection/configs"),
    ):
        config = compose(
            config_name="visualize_annotation_correction",
            overrides=[
                f"paths.annotation_root={inputs.root}",
                f"paths.campaign_dir={inputs.campaign}",
                f"paths.data_root={inputs.root.parent / 'data'}",
                f"paths.output_root={inputs.root.parent / 'runs'}",
            ],
        )
    runtime = build_runtime_config(config)
    assert runtime.mode == "render"
    assert runtime.clip_id == "csu_MWYHS8k__3b250a3c0ef22d47__f000002998-000003776"
    assert runtime.frame_range == (217, 517)
    assert runtime.frame_source == "store"
    assert runtime.store_dir == inputs.root.parent / "data/ball_detection/ball-mix-v2"
    assert (runtime.width, runtime.height) == (1920, 1080)
    assert runtime.hold_on_change_frames == 0


def test_reader_and_width_scaled_radius(inputs: Inputs) -> None:
    clip = _comparison(inputs)
    assert list(clip.corrections) == [
        "agree",
        "rerank",
        "moved",
        "added",
        "removed",
        "both_none",
        "unreviewed",
        "both_none",
    ]
    assert clip.radius_px == pytest.approx(8 * 160 / 1920)
    info = clip.input_manifest()
    assert info["checkpoint_sha256"] == "b" * 64
    assert info["annotation"]["sha256"] == sha256_file(inputs.annotation_path)
    assert info["candidate_cache"]["sha256"] == sha256_file(inputs.cache_path)


@pytest.mark.parametrize(
    "case,match",
    [
        ("schema", "schema"),
        ("proposals", "proposals_only"),
        ("identity", "identity"),
        ("order", "descending"),
        ("missing", "every manifest frame"),
        ("extra_frame", "every manifest frame"),
        ("outside", "outside image"),
        ("nonfinite", "finite"),
    ],
)
def test_candidate_validation_rejects_bad_inputs(
    inputs: Inputs, case: str, match: str
) -> None:
    data = json.loads(inputs.cache_path.read_text())
    if case == "schema":
        data["schema"] = "unknown.v1"
    elif case == "proposals":
        data["proposals_only"] = False
    elif case == "identity":
        data["identity"]["video_sha256"] = "f" * 64
    elif case == "order":
        data["frames"]["1"]["candidates"].reverse()
    elif case == "missing":
        del data["frames"]["1"]
    elif case == "extra_frame":
        data["frames"]["8"] = {"candidates": []}
    elif case == "outside":
        data["frames"]["0"]["candidates"][0]["center_px"] = [160, 0]
    elif case == "nonfinite":
        data["frames"]["0"]["candidates"][0]["score"] = float("nan")
    _save(inputs.cache_path, data)
    with pytest.raises(ValueError, match=match):
        _comparison(inputs)


@pytest.mark.parametrize(
    "case",
    ["width", "height", "frame_count", "clip_id", "sparse", "track_ids", "schema"],
)
def test_annotation_validation_rejects_bad_inputs(inputs: Inputs, case: str) -> None:
    data = json.loads(inputs.annotation_path.read_text())
    if case in ("width", "height", "frame_count"):
        data[case] += 1
    elif case == "clip_id":
        data[case] = "anotherclip"
    elif case == "sparse":
        del data["frames"][1]
    elif case == "track_ids":
        data["frames"][0]["balls"] *= 2
    elif case == "schema":
        data["schema_version"] = "unknown.v1"
    _save(inputs.annotation_path, data)
    with pytest.raises(ValueError):
        _comparison(inputs)


def test_ranking_prioritizes_class_coverage_then_ratio(inputs: Inputs) -> None:
    original_manifest = json.loads(inputs.manifest_path.read_text())
    for name in ("more_changes", "only_moved"):
        manifest = {**original_manifest, "filename": f"{name}.mp4"}
        annotation = json.loads(inputs.annotation_path.read_text())
        annotation["clip_id"] = name
        cache = json.loads(inputs.cache_path.read_text())
        if name == "more_changes":
            cache["frames"]["5"]["candidates"] = [_peak(10, 10).model_dump()]
            annotation["frames"][6]["reviewed"] = True
        else:
            for index in range(8):
                cache["frames"][str(index)]["candidates"] = [_peak(10, 10).model_dump()]
                annotation["frames"][index] = _frame(
                    _ball([90, 60]), index=index
                ).model_dump()
        _save(
            inputs.manifest_path.parent.parent / name / "clip_manifest.json", manifest
        )
        _save(inputs.annotation_path.with_name(f"{name}.json"), annotation)
        _save(inputs.cache_path.with_name(f"{name}.json"), cache)
        shutil.copyfile(inputs.video, inputs.video.with_name(f"{name}.mp4"))
    rows = rank_clips(inputs.root, inputs.campaign, radius_px=8)
    assert [row["clip_id"] for row in rows] == [
        "more_changes",
        "testclip",
        "only_moved",
    ]
    assert [row["changed_ratio"] for row in rows] == [0.75, 0.5, 1.0]
    assert rows[0]["class_counts"] == {
        "unreviewed": 0,
        "agree": 1,
        "rerank": 1,
        "moved": 2,
        "added": 1,
        "removed": 2,
        "both_none": 1,
    }


def test_layout_and_strip_are_exact_and_panels_are_synchronized() -> None:
    layout = comparison_layout(1920, 1080)
    assert layout.left == (0, 274, 944, 531)
    assert layout.right == (976, 274, 944, 531)
    assert layout.strip == (0, 837, 1920, 36)
    image: NDArray[np.uint8] = np.full((90, 160, 3), (40, 60, 80), dtype=np.uint8)
    kinds = list(Correction)
    result = draw_comparison_frame(
        image,
        [],
        _frame(),
        kinds,
        source_frame=1,
        event_seconds=1,
        layout=layout,
        source_size=(160, 90),
    )
    assert result.shape == (1080, 1920, 3)
    np.testing.assert_array_equal(result[274:805, :944], result[274:805, 976:])
    # Moved occupies cell 3 in the full-clip strip, regardless of current frame.
    np.testing.assert_array_equal(result[850, 960], [244, 162, 97])
    cursor = int(1.5 * 1920 / len(kinds))
    np.testing.assert_array_equal(result[850, cursor], [255, 255, 255])
    np.testing.assert_array_equal(result[836, 960], [18, 18, 18])
    np.testing.assert_array_equal(result[873, 960], [18, 18, 18])


def test_unreviewed_draws_no_right_annotation_or_proposal() -> None:
    image: NDArray[np.uint8] = np.full((90, 160, 3), 50, dtype=np.uint8)
    layout = comparison_layout(960, 540)
    result = draw_comparison_frame(
        image,
        [_peak(10, 10)],
        _frame(_ball([80, 40]), reviewed=False),
        [Correction.UNREVIEWED],
        source_frame=0,
        event_seconds=1,
        layout=layout,
        source_size=(160, 90),
    )
    x, y, w, h = layout.right
    np.testing.assert_array_equal(result[y : y + h, x : x + w], np.full((h, w, 3), 50))


@pytest.mark.parametrize(
    "status,center_filled",
    [("visible", True), ("occluded", False), ("interpolated", False)],
)
def test_annotation_marker_distinguishes_status(
    status: str, center_filled: bool
) -> None:
    layout = comparison_layout(1920, 1080)
    result = draw_comparison_frame(
        np.zeros((90, 160, 3), dtype=np.uint8),
        [],
        _frame(_ball([80, 40], status=status)),
        [Correction.ADDED],
        source_frame=0,
        event_seconds=1,
        layout=layout,
        source_size=(160, 90),
    )
    point = (976 + round(80 * 944 / 160), 274 + round(40 * 531 / 90))
    assert bool(result[point[1], point[0], 1] > 100) is center_filled
    assert np.any(
        result[point[1] - 12 : point[1] + 12, point[0] - 12 : point[0] + 12, 1] > 100
    )


def test_hold_repeats_only_change_frames() -> None:
    kinds = [
        Correction.AGREE,
        Correction.MOVED,
        Correction.BOTH_NONE,
        Correction.UNREVIEWED,
    ]
    assert playback_frames(kinds, 0, 4, 0) == (0, 1, 2, 3)
    assert playback_frames(kinds, 0, 4, 2) == (0, 1, 1, 1, 2, 3)
    with pytest.raises(ValueError):
        playback_frames(kinds, 0, 5, 0)


def test_arrow_grows_for_point_three_seconds_and_border_expires_at_point_four() -> None:
    layout = comparison_layout(1920, 1080)
    image: NDArray[np.uint8] = np.zeros((90, 160, 3), dtype=np.uint8)

    def draw(seconds: float) -> NDArray[np.uint8]:
        return draw_comparison_frame(
            image,
            [_peak(20, 30)],
            _frame(_ball([120, 30])),
            [Correction.MOVED],
            source_frame=0,
            event_seconds=seconds,
            layout=layout,
            source_size=(160, 90),
        )

    halfway, complete, expired = draw(0.15), draw(0.3), draw(0.4)
    y, x = 274 + round(30 * 531 / 90), 976 + 550
    assert halfway[y, x, 1] == 0
    assert complete[y, x, 1] > 100
    np.testing.assert_array_equal(complete[274, 976], [244, 162, 97])
    np.testing.assert_array_equal(expired[274, 976], [0, 0, 0])


def test_lower_candidate_opacity_tracks_score() -> None:
    layout = comparison_layout(960, 540)
    image: NDArray[np.uint8] = np.zeros((90, 160, 3), dtype=np.uint8)
    result = draw_comparison_frame(
        image,
        [_peak(10, 10), _peak(80, 40, 0.75), _peak(120, 40, 0.25)],
        _frame(),
        [Correction.REMOVED],
        source_frame=0,
        event_seconds=1,
        layout=layout,
        source_size=(160, 90),
    )
    _, y, w, h = layout.left
    high = result[y + round(40 * h / 90), round(80 * w / 160), 0]
    low = result[y + round(40 * h / 90), round(120 * w / 160), 0]
    assert high == pytest.approx(3 * int(low), abs=3)


def test_synthetic_render_exports_native_fps_panels_gif_and_manifest(
    inputs: Inputs,
) -> None:
    config = build_runtime_config(
        _config(
            inputs,
            mode="render",
            overrides=[
                "clip_id=testclip",
                "frames.range=0:8",
                "resolution.width=320",
                "resolution.height=180",
                "gif.width=160",
                "gif.fps=5",
                "playback.hold_on_change_frames=2",
            ],
        )
    )
    assert run_visualization(config) == 0
    output = config.run_dir / "presentation"
    timeline = probe_video(output / "annotation_correction.mp4")
    assert (timeline.width, timeline.height, timeline.rate, len(timeline.pts)) == (
        320,
        180,
        Fraction(10),
        16,
    )
    with Image.open(output / "annotation_correction.gif") as gif:
        assert gif.size == (160, 90)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["playback"]["duplicated_frames"] == 8
    assert manifest["playback"]["output_to_source_frame"] == [
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
        6,
        7,
    ]
    assert manifest["inputs"]["checkpoint_sha256"] == "b" * 64
    assert manifest["provenance"]["right_codex_annotations"] == "measured"
    assert manifest["provenance"]["arrows_and_borders"] == "schematic"
    assert manifest["config"]["paths"]["annotation_root"] == str(inputs.root)
    assert len(manifest["git_commit"]) == 40
    assert (output / "stills/moved.png").is_file()
    with pytest.raises(FileExistsError):
        run_visualization(config)


def test_media_hash_mismatch_is_rejected_before_export(inputs: Inputs) -> None:
    clip = _comparison(inputs)
    bad_manifest = clip.prepared.manifest.model_copy(update={"sha256": "f" * 64})
    bad_clip = replace(clip, prepared=replace(clip.prepared, manifest=bad_manifest))
    config = build_runtime_config(
        _config(
            inputs, mode="render", overrides=["clip_id=testclip", "frames.range=0:8"]
        )
    )
    with pytest.raises(ValueError, match="filename/hash"):
        render_comparison(bad_clip, config, manifest={})
    assert not (config.run_dir / "presentation").exists()


def test_media_pts_mismatch_is_rejected_before_export(inputs: Inputs) -> None:
    clip = _comparison(inputs)
    frames = list(clip.prepared.manifest.frames)
    frames[1] = frames[1].model_copy(update={"clip_pts": frames[1].clip_pts + 1})
    bad_manifest = clip.prepared.manifest.model_copy(update={"frames": frames})
    bad_clip = replace(clip, prepared=replace(clip.prepared, manifest=bad_manifest))
    config = build_runtime_config(
        _config(
            inputs, mode="render", overrides=["clip_id=testclip", "frames.range=0:8"]
        )
    )
    with pytest.raises(ValueError, match="PTS/duration"):
        render_comparison(bad_clip, config, manifest={})
    assert not (config.run_dir / "presentation").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        ["radius_px=0"],
        ["radius_px=.nan"],
        ["playback.hold_on_change_frames=-1"],
        ["mode=bad"],
        ["resolution.width=961"],
        ["gif.fps=0"],
        ["mode=render", "clip_id=testclip", "frames.range=8:8"],
        ["mode=render", "clip_id=../escape", "frames.range=0:8"],
        ["clip_id=testclip"],
        ["paths.annotation_root=relative/path"],
        ["+playback.unknown=1"],
    ],
)
def test_configuration_rejects_invalid_or_ignored_options(
    inputs: Inputs, overrides: list[str]
) -> None:
    with pytest.raises(ValueError):
        build_runtime_config(_config(inputs, overrides=overrides))


def test_input_paths_are_required_and_rank_output_is_immutable(inputs: Inputs) -> None:
    config = _config(inputs)
    del config.paths.annotation_root
    with pytest.raises(ValueError):
        build_runtime_config(config)
    runtime = build_runtime_config(_config(inputs))
    assert run_visualization(runtime) == 0
    output = runtime.run_dir / "presentation"
    ranking = json.loads((output / "ranking.json").read_text())
    assert ranking["clips"][0]["clip_id"] == "testclip"
    with pytest.raises(FileExistsError):
        run_visualization(runtime)


def _store(inputs: Inputs, *, frame_count: int = 8) -> Path:
    manifest = json.loads(inputs.manifest_path.read_text())
    directory = inputs.root.parent / "store-data/fixture-store"
    write_store_clip(
        directory,
        "chat_annotation/testclip",
        [stored_frame(frame["clip_pts"]) for frame in manifest["frames"][:frame_count]],
        source="chat_annotation",
        size=(80, 45),
    )
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    # The fixture uses OpenCV's default Q95, so recording it also checks that
    # the reader does not assume the real dataset's Q90 encoding.
    metadata["jpeg_quality"] = 95
    metadata["clips"][0].update(
        source_width=160,
        source_height=90,
        media_sha256=manifest["sha256"],
        time_base=manifest["time_base"],
        fps=manifest["nominal_fps"],
    )
    _save(path, metadata)
    return directory


def _store_runtime(inputs: Inputs, directory: Path):
    return build_runtime_config(
        _config(
            inputs,
            mode="render",
            overrides=[
                "clip_id=testclip",
                "frames.range=0:8",
                "frames.source=store",
                f"paths.data_root={directory.parent}",
                f"frames.store_dir={directory.name}",
                "resolution.width=320",
                "resolution.height=180",
                "gif.width=160",
                "gif.fps=5",
            ],
        )
    )


def test_store_rgb_without_original_video_and_original_coordinate_geometry(
    inputs: Inputs,
) -> None:
    directory = _store(inputs)
    inputs.video.unlink()
    clip = _comparison(inputs)
    frames = open_frame_input(clip, source="store", store_dir=directory)
    decoded = list(frames.decode(2, 3))
    assert decoded[0].shape == (45, 80, 3)
    assert (
        frames.metadata["media_sha256"]
        == clip.cache.identity.video_sha256
        == clip.prepared.manifest.sha256
    )
    assert frames.metadata["coordinate_scale"] == [0.5, 0.5]
    config = _store_runtime(inputs, directory)
    assert run_visualization(config) == 0
    output = config.run_dir / "presentation"
    with Image.open(output / "stills/moved.png") as image:
        pixels = np.array(image)
    layout = comparison_layout(320, 180)
    x, y, w, h = layout.right
    # Source center (90,60) maps using 160x90, not the saved 80x45.
    center = (x + round(90 * w / 160), y + round(60 * h / 90))
    np.testing.assert_array_equal(pixels[center[1], center[0]], [138, 201, 38])
    manifest = json.loads((output / "manifest.json").read_text())
    info = manifest["frame_input"]
    assert info["source"] == "store"
    assert info["store_dir"] == str(directory)
    assert info["saved_resolution"] == [80, 45]
    assert info["source_resolution"] == [160, 90]
    assert info["jpeg_quality"] == 95
    assert info["metadata"]["sha256"] == sha256_file(directory / "metadata.json")
    assert info["index"]["sha256"] == sha256_file(directory / "index.npz")
    assert info["image_payload"]["sha256"] == sha256_file(
        directory / "shards/clip-00000.bin"
    )
    timeline = probe_video(output / "annotation_correction.mp4")
    assert (timeline.rate, len(timeline.pts)) == (Fraction(10), 8)


@pytest.mark.parametrize(
    "case,match",
    [
        ("media", "SHA-256"),
        ("candidate", "SHA-256"),
        ("frame_count", "frame_count"),
        ("source_size", "source dimensions"),
        ("pts", "timestamps"),
        ("fps", "FPS"),
        ("quality", "jpeg_quality"),
        ("clip_id", "no clip"),
    ],
)
def test_store_identity_timing_and_encoding_validation(
    inputs: Inputs, case: str, match: str
) -> None:
    directory = _store(inputs, frame_count=7 if case == "frame_count" else 8)
    path = directory / "metadata.json"
    data = json.loads(path.read_text())
    clip = _comparison(inputs)
    if case == "media":
        data["clips"][0]["media_sha256"] = "e" * 64
    elif case == "candidate":
        cache = clip.cache.model_copy(
            update={
                "identity": clip.cache.identity.model_copy(
                    update={"video_sha256": "e" * 64}
                )
            }
        )
        clip = replace(clip, cache=cache)
    elif case == "source_size":
        data["clips"][0].update(source_width=320, source_height=180)
    elif case == "pts":
        with np.load(directory / "index.npz") as archive:
            columns = {key: archive[key] for key in archive.files}
        columns["pts"][1] += 1
        np.savez(directory / "index.npz", **columns)
    elif case == "fps":
        data["clips"][0]["fps"] = "20"
    elif case == "quality":
        data["jpeg_quality"] = True
    elif case == "clip_id":
        data["clips"][0]["clip_id"] = "chat_annotation/anotherclip"
    _save(path, data)
    with pytest.raises((ValueError, KeyError), match=match):
        open_frame_input(clip, source="store", store_dir=directory)


def test_video_input_does_not_substitute_available_store(inputs: Inputs) -> None:
    directory = _store(inputs)
    config = _store_runtime(inputs, directory)
    assert config.frame_source == "store"
    inputs.video.unlink()
    with pytest.raises(FileNotFoundError):
        open_frame_input(_comparison(inputs), source="video", store_dir=None)
    assert build_runtime_config(_config(inputs)).frame_source == "video"


@pytest.mark.parametrize(
    "overrides",
    [
        ["frames.source=store"],
        ["frames.source=unknown"],
        ["frames.store_dir=fixture-store"],
        [
            "mode=render",
            "clip_id=testclip",
            "frames.range=0:8",
            "frames.source=store",
            "frames.store_dir=/absolute",
        ],
        ["frames.source=store", "frames.store_dir=fixture-store"],
    ],
)
def test_frame_source_selection_rejects_incomplete_or_unused_options(
    inputs: Inputs, overrides: list[str]
) -> None:
    with pytest.raises(ValueError):
        build_runtime_config(_config(inputs, overrides=overrides))
