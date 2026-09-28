"""The persisted contract preserves raw uncertainty and rejects incomplete evidence."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    BallDetectionInputAssembler,
)
from src.tennis_scene.pipeline.runner import ComponentNode, ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from tests.unit.tennis_scene.pipeline.components.test_ball_detection import (
    _process,
    _TypedBallPredictor,
)
from tests.unit.tennis_scene.pipeline.config_factories import make_ball_config


def test_artifact_roundtrip_is_exact_and_load_only_does_not_infer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _process(tmp_path, monkeypatch, num_frames=3)
    module = BallDetectionModule(make_ball_config(tmp_path))
    monkeypatch.setattr(module, "process", lambda _: output)
    source = ClipSource("clip", (SourceVideo("near", tmp_path / "near.mp4", "hash", 3, 30., 6, 4),))
    context = AssemblyContext(source, "near")
    node = ComponentNode("ball_detection/near", module, module.io, BallDetectionInputAssembler(), {},
                         context, {}, "test")
    store = ClipStore(tmp_path / "store", json_value(source), memory_entries=0)
    runner = ComponentRunner([node], store)
    runner.run()
    reference = runner.references[node.name]
    assert (reference.schema, reference.version) == ("ball_detections", 2)
    monkeypatch.setattr(module, "process", lambda _: pytest.fail("load-only must not infer"))
    resumed = ComponentRunner([replace(node, source="load")], ClipStore(store.root, json_value(source), memory_entries=0))
    resumed.run()
    restored = resumed.output(node.name)
    assert restored.evidence is not None and output.evidence is not None
    assert restored.evidence.config == output.evidence.config
    for field in fields(output.evidence):
        before, after = getattr(output.evidence, field.name), getattr(restored.evidence, field.name)
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(after, before)
            assert not after.flags.writeable
    assert resumed.statuses[node.name] == "loaded"
    # A v1 consumer cannot load a v2 artifact (and vice versa).
    with pytest.raises(ValueError, match="contract mismatch"):
        ComponentRunner([replace(node, io=replace(node.io, version=1), source="load")], store).run()


@pytest.mark.parametrize("mutation", ["missing", "timeline", "nan", "mask", "patch", "invalid_slot", "dtype"])
def test_corrupt_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    output = _process(tmp_path, monkeypatch, num_frames=2)
    evidence = output.evidence
    assert evidence is not None
    with pytest.raises(ValueError):
        if mutation == "missing":
            replace(output, evidence=None)
        elif mutation == "timeline":
            replace(output, frame_indices=np.arange(1, dtype=np.int64), uv_px=output.uv_px[:1],
                    observed=output.observed[:1], confidence=output.confidence[:1], point_kind=output.point_kind[:1])
        elif mutation == "nan":
            maps = evidence.heatmaps.copy()
            maps[0, 0, 0] = np.nan
            replace(evidence, heatmaps=maps)
        elif mutation == "mask":
            replace(evidence, patch_valid=~evidence.patch_valid)
        elif mutation == "patch":
            replace(evidence, patches=np.zeros_like(evidence.patches))
        elif mutation == "invalid_slot":
            points = evidence.candidate_uv_px.copy()
            points[~evidence.candidate_valid] = 1
            replace(evidence, candidate_uv_px=points)
        else:
            replace(evidence, candidate_cells=evidence.candidate_cells.astype(np.int32))


def test_annotation_or_disabled_outputs_cannot_claim_model_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _process(tmp_path, monkeypatch, num_frames=2)
    for semantics in ("disabled", "annotation_acceptance_not_probability"):
        with pytest.raises(ValueError, match="Only model"):
            replace(output, score_semantics=semantics)
    disabled = BallDetectionModule(make_ball_config(tmp_path), enabled=False)
    from src.tennis_scene.pipeline.components.ball_detection import BallDetectionInput

    result = disabled.process(BallDetectionInput(SourceVideo("near", tmp_path / "unused.mp4", "hash", 2, 30., 6, 4)))
    assert result.evidence is None and not result.observed.any()


def test_trajectory_gate_does_not_erase_raw_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.tennis_scene.pipeline.components.ball_detection as component
    from src.tasks.ball_detection.model_io.contracts import (
        BallCandidateConfig,
        BallPrediction,
    )
    from src.tennis_scene.pipeline.components.ball_detection import BallDetectionInput
    from src.utils.video import FramePacket
    from tests.unit.tennis_scene.pipeline.components.test_ball_detection import (
        _prediction,
    )

    packets = [FramePacket(index=i, frame=np.zeros((4, 6, 3), np.uint8), original_size=(1000, 500)) for i in range(12)]
    monkeypatch.setattr(component, "OpenCVVideoFrameReader", lambda *args, **kwargs: packets)

    class MovingBall(_TypedBallPredictor):
        configured_frames = 12

        def predict(self, images: torch.Tensor, *, candidate_config: BallCandidateConfig) -> BallPrediction:
            maps = torch.full((1, 12, 9, 50), .001)
            for i in range(12):
                maps[0, i, 4, (40 if i == 6 else 5 + i)] = .9
            return _prediction(maps, candidate_config)

    module = BallDetectionModule(replace(make_ball_config(tmp_path), image_size=(4, 6)))
    monkeypatch.setattr(module, "load", lambda: setattr(module, "_pipeline", MovingBall()))
    output = module.process(BallDetectionInput(SourceVideo("cam", tmp_path / "clip.mp4", "hash", 12, 30., 1000, 500)))
    assert not output.observed[6] and output.observed[[5, 7]].all()
    assert output.confidence[6] == 0
    assert output.evidence is not None
    assert output.evidence.candidate_scores[6, 0] == pytest.approx(.9)
    assert output.evidence.candidate_uv_px[6, 0, 0] == pytest.approx(40 / 49 * 999)
    assert output.evidence.heatmaps[6, 4, 40] == pytest.approx(.9)
