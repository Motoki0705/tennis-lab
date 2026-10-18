from __future__ import annotations

import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ClipManifest,
    FrameMap,
    FrameRange,
    ModelRef,
    SourceIdentity,
)
from src.tasks.base.annotation.media import encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionContext
from src.tasks.player_detection.annotation.contracts import (
    DinoCheckpointSpec,
    DinoInferenceConfig,
    PlayerPredictions,
)
from src.tasks.player_detection.annotation.inference import (
    DinoPredictor,
    PlayerPredictionAdapter,
    observation_id,
)


class SpyDino:
    loads = 0
    frames: list[NDArray[np.uint8]] = []
    invalid = False

    def __init__(self, *_args: object) -> None:
        type(self).loads += 1

    def __call__(self, frame: NDArray[np.uint8]) -> SimpleNamespace:
        type(self).frames.append(frame.copy())
        if len(type(self).frames) % 2 == 0:
            return SimpleNamespace(
                boxes_xyxy=np.empty((0, 4), np.float32), scores=np.empty(0, np.float32)
            )
        return SimpleNamespace(
            boxes_xyxy=np.array(
                [[1.0, 2.0, 64.0, 32.0], [4.0, 5.0, 16.0, 20.0]], np.float32
            ),
            scores=np.array([0.9, np.nan if self.invalid else 0.7], np.float32),
        )

    def close(self) -> None:
        pass


@dataclass
class Fixture:
    layout: AnnotationLayout
    context: PredictionContext
    manifest: ClipManifest
    directory: Path


def example(tmp_path: Path) -> Fixture:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    inputs = layout.path("campaigns", "inputs")
    placeholder = write_asset(layout, inputs / "weights.pth", b"fixture")
    configuration = write_document(
        layout,
        inputs / "config.json",
        DinoInferenceConfig(confidence_threshold=0.7, short_side=32, max_long_side=64),
    )
    path = inputs / "video.mp4"
    image: NDArray[np.uint8] = np.zeros((32, 64, 3), np.uint8)
    image[:, :, 0] = 200
    encode_video(
        path,
        width=64,
        height=32,
        time_base=Fraction(1, 1000),
        rate=Fraction(10),
        frames=(
            (av.VideoFrame.from_ndarray(image, format="rgb24"), pts, duration)
            for pts, duration in ((0, 100), (100, 200), (300, 100), (400, 200))
        ),
        crf=15,
        preset="ultrafast",
    )
    media = annotation_reference(layout, path)
    timeline = probe_video(path)
    source = SourceIdentity(
        provider="local",
        channel_id="fixture",
        video_id="fixture",
        media=media,
        origin_url="local:fixture",
    )
    manifest = ClipManifest(
        source=source,
        media=media,
        preparation_digest="e" * 64,
        width=64,
        height=32,
        time_base=str(timeline.time_base),
        nominal_fps=str(timeline.rate),
        source_range=FrameRange(start=0, stop=4),
        target_range=FrameRange(start=1, stop=3),
        frames=tuple(
            FrameMap(
                frame_index=index,
                source_frame_index=index,
                source_pts=pts,
                clip_pts=pts,
                duration_pts=duration,
            )
            for index, (pts, duration) in enumerate(
                zip(timeline.pts, timeline.durations, strict=True)
            )
        ),
    )
    reference = write_document(layout, inputs / "manifest.json", manifest)
    context = PredictionContext(
        task="player",
        manifest=reference,
        clip_id=manifest.clip_id,
        model=ModelRef(
            task="player",
            version="fixture",
            training_weights=placeholder,
            inference_weights=placeholder,
            configuration=placeholder,
        ),
        configuration=configuration,
    )
    return Fixture(
        layout,
        context,
        manifest,
        layout.path("predictions", "player", manifest.clip_id, context.request_id),
    )


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    SpyDino.loads, SpyDino.frames, SpyDino.invalid = 0, [], False
    monkeypatch.setattr(
        "src.tasks.player_detection.annotation.inference.DinoPredictor", SpyDino
    )
    return example(tmp_path)


def test_native_boxes_empty_rows_bgr_pts_and_cache(fixture: Fixture) -> None:
    adapter = PlayerPredictionAdapter()
    reference = adapter.produce(
        fixture.layout, fixture.context, fixture.manifest, fixture.directory
    )
    adapter.validate(fixture.layout, fixture.context, fixture.manifest, reference)
    result = read_document(fixture.layout, reference, PlayerPredictions)
    assert len(result.rows) == 4 and [len(row.observations) for row in result.rows] == [
        2,
        0,
        2,
        0,
    ]
    assert result.output_kind == "bbox_proposals_unreviewed"
    assert all("reviewed" not in row.model_dump() for row in result.rows)
    assert SpyDino.loads == 1 and len(SpyDino.frames) == 4
    assert SpyDino.frames[0].shape == (32, 64, 3)
    assert (
        SpyDino.frames[0][:, :, 2].mean() > 190
        and SpyDino.frames[0][:, :, 0].mean() < 5
    )
    ids = []
    for row, frame in zip(result.rows, fixture.manifest.frames, strict=True):
        assert row.source_pts == frame.source_pts and row.clip_pts == frame.clip_pts
        for rank, proposal in enumerate(row.observations):
            assert proposal.observation_id == observation_id(
                fixture.context, row.frame_index, rank
            )
            ids.append(proposal.observation_id)
    assert len(ids) == len(set(ids))
    box = result.rows[0].observations[0].box
    assert (box.x1, box.y1, box.x2, box.y2) == (1.0, 2.0, 64.0, 32.0)
    assert result.rows[0].observations[1].confidence == float(np.float32(0.7))
    assert (
        PlayerPredictionAdapter().produce(
            fixture.layout, fixture.context, fixture.manifest, fixture.directory
        )
        == reference
    )
    assert SpyDino.loads == 1 and len(SpyDino.frames) == 4
    adapter.close()


def test_nonfinite_output_is_not_published(fixture: Fixture) -> None:
    SpyDino.invalid = True
    with pytest.raises(ValueError, match="nonfinite"):
        PlayerPredictionAdapter().produce(
            fixture.layout, fixture.context, fixture.manifest, fixture.directory
        )
    assert not (fixture.directory / "boxes.json").exists()


@pytest.mark.parametrize("kind", ["identity", "pts", "coordinates", "score"])
def test_corrupted_prediction_is_rejected(fixture: Fixture, kind: str) -> None:
    adapter = PlayerPredictionAdapter()
    reference = adapter.produce(
        fixture.layout, fixture.context, fixture.manifest, fixture.directory
    )
    result = read_document(fixture.layout, reference, PlayerPredictions)
    row = result.rows[0]
    proposal = row.observations[0]
    if kind == "identity":
        proposal = proposal.model_copy(update={"observation_id": "f" * 64})
    elif kind == "score":
        proposal = proposal.model_copy(update={"confidence": 0.1})
    elif kind == "coordinates":
        proposal = proposal.model_copy(
            update={"box": proposal.box.model_copy(update={"x2": 65.0})}
        )
    row = row.model_copy(update={"observations": (proposal, *row.observations[1:])})
    if kind == "pts":
        row = row.model_copy(update={"source_pts": row.source_pts + 1})
    broken = write_document(
        fixture.layout,
        fixture.directory / f"bad-{kind}.json",
        result.model_copy(update={"rows": (row, *result.rows[1:])}),
    )
    with pytest.raises(ValueError):
        adapter.validate(fixture.layout, fixture.context, fixture.manifest, broken)


def test_cache_context_and_input_hash_are_not_swapped(fixture: Fixture) -> None:
    adapter = PlayerPredictionAdapter()
    adapter.produce(
        fixture.layout, fixture.context, fixture.manifest, fixture.directory
    )
    other = fixture.context.model_copy(
        update={"model": fixture.context.model.model_copy(update={"version": "other"})}
    )
    with pytest.raises(ValueError, match="source/config"):
        adapter.produce(fixture.layout, other, fixture.manifest, fixture.directory)
    fixture.layout.artifact_path(fixture.manifest.media).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        adapter.produce(
            fixture.layout,
            fixture.context,
            fixture.manifest,
            fixture.directory / "uncached",
        )
    assert SpyDino.loads == 1


def test_export_pairing_and_repository_revision_are_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = example(tmp_path)
    repository = fixture.layout.repository / "third_party/DINO"
    repository.mkdir(parents=True)
    for command in (
        ("git", "init", "-q"),
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ),
    ):
        subprocess.run(command, cwd=repository, check=True)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    spec = write_document(
        fixture.layout,
        fixture.layout.path("campaigns", "inputs", "spec.json"),
        DinoCheckpointSpec(repository_commit=revision, epoch=3, global_step=20),
    )
    context = fixture.context.model_copy(
        update={
            "model": fixture.context.model.model_copy(update={"configuration": spec})
        }
    )
    config = read_document(fixture.layout, context.configuration, DinoInferenceConfig)

    class Detector:
        mismatch = False
        unloaded = 0

        def __init__(self, checkpoint: Path, repo: Path, **kwargs: object) -> None:
            assert (
                checkpoint
                == fixture.layout.artifact_path(context.model.inference_weights)
                and repo == repository
            )
            assert kwargs == {
                "device": "cuda",
                "confidence": 0.7,
                "short_side": 32,
                "max_long_side": 64,
            }
            self.checkpoint_info = SimpleNamespace(
                source_checkpoint_sha256="f" * 64
                if self.mismatch
                else context.model.training_weights.digest,
                epoch=3,
                global_step=20,
            )

        def load(self) -> None:
            pass

        def unload(self) -> None:
            type(self).unloaded += 1

    monkeypatch.setattr(
        "src.tasks.player_detection.inference.DinoPlayerDetector", Detector
    )
    predictor = DinoPredictor(fixture.layout, context, config)
    predictor.close()
    assert Detector.unloaded == 1
    Detector.mismatch = True
    with pytest.raises(ValueError, match="training checkpoint"):
        DinoPredictor(fixture.layout, context, config)
    assert Detector.unloaded == 2
    bad_spec = write_document(
        fixture.layout,
        fixture.layout.path("campaigns", "inputs", "wrong-spec.json"),
        DinoCheckpointSpec(repository_commit="a" * 40, epoch=3, global_step=20),
    )
    with pytest.raises(ValueError, match="repository revision"):
        DinoPredictor(
            fixture.layout,
            context.model_copy(
                update={
                    "model": context.model.model_copy(
                        update={"configuration": bad_spec}
                    )
                }
            ),
            config,
        )
