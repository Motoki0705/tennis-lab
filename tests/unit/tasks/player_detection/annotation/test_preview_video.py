from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.media import probe_video
from src.tasks.base.annotation.plan import file_digest
from src.tasks.base.annotation.predictions import PredictionReceipt
from src.tasks.player_detection.annotation.contracts import PlayerPredictions
from src.tasks.player_detection.annotation.preview_video import (
    VideoPreview,
    export_prediction_video,
)
from src.tasks.player_detection.scripts.preview_predictions import PATH_BOUNDARY
from src.utils.configuration import PathResolver
from tests.support.tasks.annotation_predictions import example
from tests.support.tasks.player_detection.annotation import ProposalFixture


def forbidden(*args: Any, **kwargs: Any) -> None:
    raise AssertionError(
        "saved video preview must not infer, load a model or open a ledger"
    )


@pytest.mark.parametrize("jpeg", [False, True])
def test_video_preserves_all_native_frames_pts_boxes_and_readonly_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jpeg: bool
) -> None:
    env = example(tmp_path, 5, task="player", jpeg=jpeg)
    prediction = env.service.run(env.reference, ProposalFixture())
    before = {
        p: (p.stat().st_mtime_ns, file_digest(p))
        for p in env.layout.root.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        "src.tasks.player_detection.annotation.inference.DinoPredictor", forbidden
    )
    monkeypatch.setattr(
        "src.tasks.base.annotation.predictions.Predictions.__init__", forbidden
    )
    output = tmp_path / "video"
    result = export_prediction_video(
        env.layout, prediction, output, reason="inspect missed players"
    )
    assert result == VideoPreview.model_validate_json(
        (output / "provenance.json").read_bytes()
    )
    assert result.context.context.model == env.plan.tasks[0].model
    assert (
        result.reason == "inspect missed players"
        and result.frames == env.manifest.frames
    )
    assert result.target_range == env.manifest.target_range
    timeline = probe_video(output / "preview.mp4")
    assert [p * timeline.time_base for p in timeline.pts] == [
        f.clip_pts * Fraction(env.manifest.time_base) for f in env.manifest.frames
    ]
    assert [d * timeline.time_base for d in timeline.durations] == [
        f.duration_pts * Fraction(env.manifest.time_base) for f in env.manifest.frames
    ]
    with av.open(str(output / "preview.mp4")) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    yellow = [
        np.any((f[:, :, 0] > 150) & (f[:, :, 1] > 130) & (f[:, :, 2] < 80))
        for f in frames
    ]
    assert yellow == [True, True, False, True, True]
    assert before == {
        p: (p.stat().st_mtime_ns, file_digest(p))
        for p in env.layout.root.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        export_prediction_video(env.layout, prediction, output, reason="retry")


@pytest.mark.parametrize(
    "fault",
    [
        "receipt",
        "proposal_hash",
        "configuration",
        "model_spec",
        "media",
        "pts",
        "observation",
        "bbox",
    ],
)
def test_video_rejects_corrupt_identity_before_creating_output(
    tmp_path: Path, fault: str
) -> None:
    env = example(tmp_path, 5, task="player")
    reference = env.service.run(env.reference, ProposalFixture())
    receipt = read_document(env.layout, reference, PredictionReceipt)
    if fault == "receipt":
        receipt = receipt.model_copy(
            update={
                "prediction": receipt.prediction.model_copy(
                    update={"model_digest": "f" * 64}
                )
            }
        )
    elif fault == "proposal_hash":
        env.layout.artifact_path(receipt.prediction.artifact).write_bytes(b"corrupt")
    elif fault in {"configuration", "model_spec", "media"}:
        reference_to_change = {
            "configuration": receipt.context.configuration,
            "model_spec": receipt.context.model.configuration,
            "media": env.manifest.media,
        }[fault]
        env.layout.artifact_path(reference_to_change).write_bytes(b"corrupt")
    else:
        predictions = read_document(
            env.layout, receipt.prediction.artifact, PlayerPredictions
        )
        row = predictions.rows[0]
        if fault == "pts":
            row = row.model_copy(update={"source_pts": -1})
        else:
            item = row.observations[0]
            item = (
                item.model_copy(update={"observation_id": "f" * 64})
                if fault == "observation"
                else item.model_copy(
                    update={"box": item.box.model_copy(update={"x2": 65.0})}
                )
            )
            row = row.model_copy(update={"observations": (item, *row.observations[1:])})
        proposals = write_document(
            env.layout,
            env.layout.path("campaigns", "bad-proposals.json"),
            predictions.model_copy(update={"rows": (row, *predictions.rows[1:])}),
        )
        receipt = receipt.model_copy(
            update={
                "prediction": receipt.prediction.model_copy(
                    update={"artifact": proposals}
                )
            }
        )
    bad = write_document(
        env.layout, env.layout.path("campaigns", "bad-receipt.json"), receipt
    )
    with pytest.raises(ValueError):
        export_prediction_video(env.layout, bad, tmp_path / "bad", reason="inspect")
    assert not (tmp_path / "bad").exists()


def test_video_refuses_data_outputs_and_symlinks_and_cleans_failed_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = example(tmp_path, 5, task="player")
    reference = env.service.run(env.reference, ProposalFixture())
    for output in (
        env.layout.root / "video",
        tmp_path / "data/video",
        tmp_path / "outputs/video",
    ):
        with pytest.raises(ValueError, match="separate"):
            export_prediction_video(env.layout, reference, output, reason="inspect")
    alias = tmp_path / "alias"
    alias.symlink_to(env.layout.root, target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        export_prediction_video(
            env.layout, reference, alias / "video", reason="inspect"
        )
    monkeypatch.setattr(
        "src.tasks.player_detection.annotation.preview_video.encode_video", forbidden
    )
    with pytest.raises(AssertionError):
        export_prediction_video(
            env.layout, reference, tmp_path / "failed", reason="inspect"
        )
    assert not (tmp_path / "failed").exists()


def test_cli_output_boundary_is_scratch_only(tmp_path: Path) -> None:
    env = example(tmp_path, 5, task="player")
    reference = env.service.run(env.reference, ProposalFixture())
    resolver = PathResolver(
        replace(env.layout.resolver.roots, cache_root=tmp_path / "tmp")
    )
    prediction = env.layout.artifact_path(reference)
    PATH_BOUNDARY.validate(
        {"prediction": prediction, "output": tmp_path / "tmp/video"}, resolver=resolver
    )
    with pytest.raises(ValueError):
        PATH_BOUNDARY.validate(
            {"prediction": prediction, "output": tmp_path / "outputs/video"},
            resolver=resolver,
        )
