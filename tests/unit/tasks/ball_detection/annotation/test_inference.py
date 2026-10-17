from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.annotation.contracts import (
    BallPredictions,
    MDDCheckpointSpec,
    MDDInferenceConfig,
)
from src.tasks.ball_detection.annotation.inference import (
    BallPredictionAdapter,
    MDDPredictor,
)
from src.tasks.ball_detection.annotation.windows import window_owners, window_starts
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
)
from src.tasks.base.annotation.media import check_clip, decode_range
from src.tasks.base.annotation.predictions import (
    PredictionContext,
    PredictionReceipt,
)
from tests.support.tasks.annotation_predictions import Fixture, example


class SpyPredictor:
    calls: list[tuple[tuple[int, ...], NDArray[np.float32]]] = []
    loads = 0

    def __init__(self, *_args: object) -> None:
        type(self).loads += 1

    def __call__(
        self, rgb: NDArray[np.uint8], seconds: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        assert rgb.dtype == np.uint8 and seconds[0] == 0
        assert np.all(np.diff(seconds) > 0)
        type(self).calls.append((rgb.shape, seconds.copy()))
        # Distinct positions within each window make overlap ownership observable.
        return np.stack(
            (np.arange(32, dtype=np.float32) / 31, np.ones(32, np.float32)), axis=1
        )


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    SpyPredictor.loads = 0
    SpyPredictor.calls = []
    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.MDDPredictor", SpyPredictor
    )
    return example(tmp_path)


@pytest.mark.parametrize(
    "count,stride",
    [(32, 32), (33, 32), (47, 16), (63, 32), (64, 32), (65, 16), (83, 31), (101, 1)],
)
def test_windows_cover_only_real_frames(count: int, stride: int) -> None:
    starts = window_starts(count, stride)
    assert starts[0] == 0 and starts[-1] == count - 32
    assert len(starts) == len(set(starts))
    owners = window_owners(count, stride)
    for frame, owner in enumerate(owners):
        candidates = [start for start in starts if start <= frame < start + 32]
        assert owner == min(
            candidates, key=lambda start: (abs(frame - start - 15.5), start)
        )
    with pytest.raises(ValueError, match="padding"):
        window_starts(31, stride)


def test_native_pixels_vfr_and_cache_skip_inference(fixture: Fixture) -> None:
    result = fixture.service.run(fixture.reference, BallPredictionAdapter())
    receipt = read_document(fixture.layout, result, PredictionReceipt)
    document = read_document(
        fixture.layout, receipt.prediction.artifact, BallPredictions
    )
    assert len(document.rows) == 65
    assert SpyPredictor.loads == 1 and len(SpyPredictor.calls) == 4
    assert {shape for shape, _seconds in SpyPredictor.calls} == {(32, 3, 32, 64)}
    assert np.allclose(SpyPredictor.calls[0][1][:4], [0, 0.1, 0.3, 0.4])
    for row, owner, frame in zip(
        document.rows, window_owners(65, 16), fixture.manifest.frames, strict=True
    ):
        assert row.x == pytest.approx((row.frame_index - owner) / 31 * 63, abs=1e-5)
        assert row.y == 31.0
        assert (
            row.source_pts == frame.source_pts
            and row.source_frame_index == frame.source_frame_index
        )
    assert not {"confidence", "presence", "visibility"} & set(
        document.rows[0].model_dump()
    )
    assert fixture.service.run(fixture.reference, BallPredictionAdapter()) == result
    assert SpyPredictor.loads == 1 and len(SpyPredictor.calls) == 4
    with fixture.service.registry.transaction() as database:
        database.execute("UPDATE annotation_tasks SET state='published'")
    assert fixture.service.run(fixture.reference, BallPredictionAdapter()) == result
    assert SpyPredictor.loads == 1


@pytest.mark.parametrize("count", [32, 33, 47, 63, 65])
def test_streamed_windows_match_individual_native_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    fixture = example(tmp_path, count)
    video = fixture.layout.artifact_path(fixture.manifest.media)
    timeline = check_clip(video, fixture.manifest)
    starts = window_starts(count, 16)
    expected = [
        np.stack(
            [
                frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                for frame in decode_range(video, timeline, start, start + 32)
            ]
        )
        for start in starts
    ]
    calls = 0
    intervals = []
    decoded_frames = 0

    class PixelPredictor(SpyPredictor):
        def __call__(
            self, rgb: NDArray[np.uint8], seconds: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            nonlocal calls
            np.testing.assert_array_equal(rgb, expected[calls])
            calls += 1
            return super().__call__(rgb, seconds)

    def counted_decode(path, clock, start, stop):
        nonlocal decoded_frames
        intervals.append((start, stop))
        for frame in decode_range(path, clock, start, stop):
            decoded_frames += 1
            yield frame

    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.MDDPredictor", PixelPredictor
    )
    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.decode_range", counted_decode
    )
    fixture.service.run(fixture.reference, BallPredictionAdapter())
    assert calls == len(starts)
    assert intervals == [(0, count)] and decoded_frames == count


def test_decoder_final_validation_precedes_publication(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_decode(*args):
        yield from decode_range(*args)
        raise ValueError("final decoder validation failed")

    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.decode_range", invalid_decode
    )
    with pytest.raises(ValueError, match="final decoder validation failed"):
        fixture.service.run(fixture.reference, BallPredictionAdapter())
    assert not list(fixture.layout.root.rglob("coordinates.json"))


def test_prediction_receipt_gap_reuses_coordinates(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.predictions.write_document",
            lambda *_: (_ for _ in ()).throw(OSError("receipt gap")),
        )
        with pytest.raises(OSError, match="receipt gap"):
            fixture.service.run(fixture.reference, BallPredictionAdapter())
    calls = len(SpyPredictor.calls)
    fixture.service.run(fixture.reference, BallPredictionAdapter())
    assert len(SpyPredictor.calls) == calls and SpyPredictor.loads == 1


def test_database_gap_reuses_prediction_receipt(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            fixture.service,
            "_commit",
            lambda *_: (_ for _ in ()).throw(RuntimeError("database gap")),
        )
        with pytest.raises(RuntimeError, match="database gap"):
            fixture.service.run(fixture.reference, BallPredictionAdapter())
    fixture.service.run(fixture.reference, BallPredictionAdapter())
    assert SpyPredictor.loads == 1 and len(SpyPredictor.calls) == 4


def test_concurrent_inference_has_one_model_call_sequence(fixture: Fixture) -> None:
    def run(_number: int) -> ArtifactRef | None:
        try:
            return fixture.service.run(fixture.reference, BallPredictionAdapter())
        except BlockingIOError:
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts = [ref for ref in executor.map(run, range(4)) if ref is not None]
    assert receipts and len(set(receipts)) == 1
    assert SpyPredictor.loads == 1 and len(SpyPredictor.calls) == 4


def test_different_model_unknown_clip_and_tamper_are_not_reinferred(
    fixture: Fixture,
) -> None:
    first = fixture.service.run(fixture.reference, BallPredictionAdapter())
    changed = fixture.plan.tasks[0].model.model_copy(update={"version": "v2"})
    task = fixture.plan.tasks[0].model_copy(update={"model": changed})
    original = fixture.service.plan
    fixture.service.plan = original.model_copy(update={"tasks": (task,)})
    with pytest.raises(ValueError, match="newly registered"):
        fixture.service.run(fixture.reference, BallPredictionAdapter())
    fixture.service.plan = original
    source = fixture.manifest.source.model_copy(update={"video_id": "new"})
    manifest = fixture.manifest.model_copy(update={"source": source})
    unknown = write_document(
        fixture.layout,
        fixture.layout.path("campaigns", "inputs", "unknown.json"),
        manifest,
    )
    with pytest.raises(ValueError, match="newly registered"):
        fixture.service.run(unknown, BallPredictionAdapter())
    receipt = read_document(fixture.layout, first, PredictionReceipt)
    fixture.layout.artifact_path(receipt.prediction.artifact).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        fixture.service.run(fixture.reference, BallPredictionAdapter())
    assert SpyPredictor.loads == 1 and len(SpyPredictor.calls) == 4


def test_short_clip_fails_without_model_or_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = example(tmp_path, 31)
    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.MDDPredictor",
        lambda *_: pytest.fail("no model needed"),
    )
    with pytest.raises(ValueError, match="padding"):
        fixture.service.run(fixture.reference, BallPredictionAdapter())


@pytest.mark.parametrize("historical", [False, True])
def test_real_mdd_checkpoint_roundtrip_and_native_forward(tmp_path: Path, historical: bool) -> None:
    import torch

    from src.tasks.ball_detection.models.mdd import (
        DeepMDDQueryDetector,
        MDDPretrainConfig,
    )

    torch.set_num_threads(2)
    config = MDDPretrainConfig(
        stem_channels=(4, 8, 8, 8),
        mixed_channels=(8, 16),
        residual_blocks=(0, 1, 1, 1, 1, 1),
        decoder_channels=8,
        dim=16,
        heads=2,
        layers=1,
        ffn_dim=64,
        dropout=0.0,
        rope_base=10000.0,
        activation_checkpointing=False,
        encoder_variant="residual",
        temporal_mixing="dense3d",
        input_channels=2,
    )
    model = DeepMDDQueryDetector(config).eval()
    fixture = example(tmp_path)
    directory = fixture.layout.path("campaigns", "inputs")
    checkpoint = directory / "model.pt"
    input_contract = model.mdd.input_contract()
    architecture = asdict(config)
    if historical:
        architecture.pop("input_channels")
    recipe = {
        "model": architecture,
        "input_contract": input_contract,
        "image_decode": {"decoder": "fixture"},
        "parent": {
            "input_contract": input_contract,
            "image_decode": {"decoder": "fixture"},
        },
        "manifest_sha256": "f" * 64,
    }
    torch.save(
        {
            "schema": "mdd_query_posttraining.v1",
            "stage": "query_posttraining",
            "model_config": architecture,
            "recipe": recipe,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    weights = annotation_reference(fixture.layout, checkpoint)
    spec = write_document(
        fixture.layout,
        directory / "spec.json",
        MDDCheckpointSpec(
            checkpoint_schema="mdd_query_posttraining.v1",
            architecture=architecture,
            input_contract=input_contract,
            training_manifest_digest="f" * 64,
        ),
    )
    model_ref = fixture.plan.tasks[0].model.model_copy(
        update={
            "training_weights": weights,
            "inference_weights": weights,
            "configuration": spec,
        }
    )
    context = PredictionContext(
        task="ball",
        manifest=fixture.reference,
        clip_id=fixture.manifest.clip_id,
        model=model_ref,
        configuration=fixture.plan.tasks[0].inference,
    )
    inference = read_document(fixture.layout, context.configuration, MDDInferenceConfig)
    predictor = MDDPredictor(fixture.layout, context, inference)
    rgb: NDArray[np.uint8] = np.zeros((32, 3, 32, 64), np.uint8)
    seconds = np.arange(32, dtype=np.float32) / 30
    actual = predictor(rgb, seconds)
    with torch.inference_mode():
        expected = model(torch.from_numpy(rgb)[None], torch.from_numpy(seconds)[None])[
            0
        ].numpy()
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (32, 2) and np.isfinite(actual).all()
    bad_spec = write_document(
        fixture.layout,
        directory / "bad-spec.json",
        MDDCheckpointSpec(
            checkpoint_schema="mdd_query_posttraining.v1",
            architecture={**architecture, "dim": 32},
            input_contract=input_contract,
            training_manifest_digest="f" * 64,
        ),
    )
    with pytest.raises(ValueError, match="architecture"):
        MDDPredictor(
            fixture.layout,
            context.model_copy(
                update={
                    "model": model_ref.model_copy(update={"configuration": bad_spec})
                }
            ),
            inference,
        )
    with pytest.raises(ValueError, match="timestamps"):
        predictor(rgb, np.zeros(32, np.float32))
    with pytest.raises(ValueError, match="hash|size"):
        checkpoint.write_bytes(b"altered")
        MDDPredictor(fixture.layout, context, inference)
