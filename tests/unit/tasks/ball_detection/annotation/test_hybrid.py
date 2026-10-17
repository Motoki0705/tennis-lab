from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tasks.ball_detection.annotation.contracts import (
    HybridPredictions,
    HybridProposal,
    read_predictions,
)
from src.tasks.ball_detection.annotation.inference import BallPredictionAdapter
from src.tasks.ball_detection.annotation.validation import validate_predictions
from src.tasks.ball_detection.annotation.windows import window_owners
from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import AdoptionReceipt, ArtifactRef
from src.tasks.base.annotation.predictions import (
    PredictionReceipt,
    Predictions,
)
from tests.support.tasks.annotation_predictions import Fixture, example
from tests.support.tasks.ball_detection.hybrid_annotation import SpyHybrid, inference


def hybrid_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, jpeg: bool = False
) -> Fixture:
    fixture = example(tmp_path, jpeg=jpeg, inference_config=inference(jpeg=jpeg))
    SpyHybrid.loads, SpyHybrid.calls = 0, []
    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.HybridPredictor", SpyHybrid
    )
    return fixture


@pytest.mark.parametrize("jpeg", [False, True])
def test_hybrid_native_vfr_overlap_absence_and_cached_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jpeg: bool
) -> None:
    fixture = hybrid_fixture(tmp_path, monkeypatch, jpeg=jpeg)
    ref = fixture.service.run(fixture.reference, BallPredictionAdapter())
    receipt = read_document(fixture.layout, ref, PredictionReceipt)
    document = read_predictions(fixture.layout, receipt.prediction.artifact)
    assert isinstance(document, HybridPredictions)
    assert document.heatmap_size == (8, 16) and len(document.rows) == 65
    np.testing.assert_allclose(SpyHybrid.calls[0][:4], [0.0, 0.1, 0.3, 0.4])
    for row, frame, owner in zip(
        document.rows, fixture.manifest.frames, window_owners(65, 16), strict=True
    ):
        local = row.frame_index - owner
        assert row.source_pts == frame.source_pts and row.window_start == owner
        assert row.heatmap_peak == pytest.approx(0.9)
        assert (row.presence == "present") == (local % 2 == 0)
        if row.presence == "present":
            assert row.center == pytest.approx((local / 31 * 63, 31.0))
        else:
            assert row.center is None and row.x is None and row.y is None
    assert b"NaN" not in document.canonical_bytes()
    assert fixture.service.run(fixture.reference, BallPredictionAdapter()) == ref
    assert SpyHybrid.loads == 1 and len(SpyHybrid.calls) == 4


@pytest.mark.parametrize(
    "change",
    [
        {"presence": "absent"},
        {"x": None},
        {"presence_logit": float("nan")},
        {"heatmap_peak": -1.0},
        {"heatmap_peak": 2.0},
        {"presence_score": 2.0},
    ],
)
def test_hybrid_schema_rejects_inconsistent_or_nonfinite_proposals(
    change: dict,
) -> None:
    row = dict(
        frame_index=0,
        source_frame_index=0,
        source_pts=0,
        clip_pts=0,
        window_start=0,
        x=1.0,
        y=2.0,
        presence="present",
        presence_logit=2.0,
        presence_score=0.8807971,
        heatmap_peak=0.9,
    )
    with pytest.raises(ValueError):
        HybridProposal.model_validate({**row, **change})


def test_hybrid_threshold_uses_the_model_io_fp32_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    from src.tasks.ball_detection.annotation.hybrid import HybridWindow

    config = inference().model_copy(update={"presence_threshold": 0.7})
    fixture = example(tmp_path, count=32, inference_config=config)
    logit = torch.logit(torch.tensor(0.7, dtype=torch.float32))
    score = logit.sigmoid()
    assert float(score) < 0.7  # Decimal .7 is rounded down in FP32.

    class BoundaryPredictor(SpyHybrid):
        def __call__(self, rgb: np.ndarray, seconds: np.ndarray) -> HybridWindow:
            result = super().__call__(rgb, seconds)
            return HybridWindow(
                coordinates=np.zeros((32, 2), np.float32),
                presence_logits=np.full(32, logit.item(), np.float32),
                presence_scores=np.full(32, score.item(), np.float32),
                present=np.ones(32, np.bool_),
                heatmap_peaks=result.heatmap_peaks,
                heatmap_size=result.heatmap_size,
            )

    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.HybridPredictor",
        BoundaryPredictor,
    )
    ref = fixture.service.run(fixture.reference, BallPredictionAdapter())
    receipt = read_document(fixture.layout, ref, PredictionReceipt)
    doc = read_predictions(fixture.layout, receipt.prediction.artifact)
    assert isinstance(doc, HybridPredictions) and all(
        row.presence == "present" for row in doc.rows
    )


@pytest.mark.parametrize(
    "change", ["grid", "decision", "score", "coordinate", "pts", "family"]
)
def test_hybrid_saved_evidence_is_validated_without_reinference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    fixture = hybrid_fixture(tmp_path, monkeypatch)
    ref = fixture.service.run(fixture.reference, BallPredictionAdapter())
    receipt = read_document(fixture.layout, ref, PredictionReceipt)
    doc = read_predictions(fixture.layout, receipt.prediction.artifact)
    assert isinstance(doc, HybridPredictions)
    rows = list(doc.rows)
    if change == "grid":
        doc = doc.model_copy(update={"heatmap_size": (7, 16)})
    elif change == "family":
        from src.tasks.ball_detection.annotation.contracts import MDDInferenceConfig

        cfg = write_document(
            fixture.layout,
            fixture.layout.path("campaigns", "wrong.json"),
            MDDInferenceConfig(stride=16, device="cpu", precision="float32"),
        )
        receipt = receipt.model_copy(
            update={
                "context": receipt.context.model_copy(update={"configuration": cfg})
            }
        )
    else:
        update = {
            "decision": {"presence": "absent", "x": None, "y": None},
            "score": {"presence_score": 0.4},
            "coordinate": {"x": 100.0},
            "pts": {"source_pts": -1},
        }[change]
        rows[0] = rows[0].model_copy(update=update)
        doc = doc.model_copy(update={"rows": tuple(rows)})
    altered = write_document(
        fixture.layout, fixture.layout.path("campaigns", "altered.json"), doc
    )
    with pytest.raises(ValueError):
        validate_predictions(fixture.layout, receipt.context, fixture.manifest, altered)
    assert SpyHybrid.loads == 1 and len(SpyHybrid.calls) == 4


def test_mixed_published_pending_and_interrupted_hybrid_clips_preserve_teachers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.ball_detection.annotation.review.contracts import (
        BallEdits,
        ConfirmAbsence,
    )
    from src.tasks.base.annotation.contracts import FrameRange
    from src.tasks.base.annotation.editing import Revision
    from tests.support.tasks.ball_detection.annotation import Environment

    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.HybridPredictor", SpyHybrid
    )
    SpyHybrid.loads, SpyHybrid.calls = 0, []
    root = tmp_path / "a"

    def create(video_id: str, frame_count: int = 32) -> Environment:
        return Environment(
            root,
            video_id=video_id,
            frame_count=frame_count,
            inference_config=inference(),
            prediction_adapter=BallPredictionAdapter(),
        )

    old = create("old")
    image = old.page((1,))
    old.apply(
        BallEdits(
            operations=(
                ConfirmAbsence(
                    scope=FrameRange(start=1, stop=2),
                    out_of_frame_tracks=(),
                    evidence=(image.sha256,),
                ),
            ),
            images=(image,),
        )
    )
    adoption = old.adapter.prepare_adoption(
        old.layout,
        old.context,
        old.manifest,
        old.head(),
        (old.telemetry((image,)),),
        reason="synthetic reviewed absence",
    )
    old.store.seal(old.head(), adoption, old.adapter)
    revision = read_document(old.layout, old.head(), Revision)
    adopted = read_document(old.layout, adoption, AdoptionReceipt)
    protected = (
        old.context.prediction,
        old.head(),
        revision.payload,
        adoption,
        adopted.teacher_mask,
    )
    before = {ref: old.layout.artifact_path(ref).read_bytes() for ref in protected}
    service = Predictions(old.layout, old.plan)
    with service.registry.transaction() as db:
        db.execute(
            "UPDATE annotation_tasks SET state='published' WHERE clip_id=?",
            (old.manifest.clip_id,),
        )
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.predictions.write_document",
            lambda *_: (_ for _ in ()).throw(OSError("receipt gap")),
        )
        with pytest.raises(OSError, match="receipt gap"):
            create("interrupted", 33)
    with monkeypatch.context() as patch:
        patch.setattr(
            BallPredictionAdapter,
            "produce",
            lambda *_: (_ for _ in ()).throw(OSError("not started")),
        )
        with pytest.raises(OSError, match="not started"):
            create("new", 34)
    with service.registry.transaction() as db:
        registered = [
            ArtifactRef.model_validate_json(row[0])
            for row in db.execute("SELECT manifest FROM clips ORDER BY id")
        ]
    calls_before = len(SpyHybrid.calls)
    shared = BallPredictionAdapter()
    receipts = tuple(service.run(ref, shared) for ref in registered)
    assert (
        len(registered) == 3 and len(SpyHybrid.calls) == calls_before + 2
    )  # Only new clip's two windows.
    assert (
        tuple(service.run(ref, BallPredictionAdapter()) for ref in registered)
        == receipts
    )
    assert len(SpyHybrid.calls) == calls_before + 2
    assert {
        ref: old.layout.artifact_path(ref).read_bytes() for ref in protected
    } == before
    changed = old.plan.tasks[0].model_copy(
        update={"model": old.plan.tasks[0].model.model_copy(update={"version": "next"})}
    )
    service.plan = old.plan.model_copy(update={"tasks": (changed,)})
    for ref in registered:
        with pytest.raises(ValueError, match="newly registered"):
            service.run(ref, BallPredictionAdapter())
    assert len(SpyHybrid.calls) == calls_before + 2
