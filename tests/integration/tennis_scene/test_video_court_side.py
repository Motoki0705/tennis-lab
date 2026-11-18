"""Fixed geometry and measured ball tracks through the real component stores and single-clip runner."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.video_artifact import VideoArtifactConfig
from src.tennis_scene.pipeline.video_court_side import (
    PinnedSideInputs,
    build_video_court_side,
    prepare_video_court_side,
)
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from tests.integration.tennis_scene.test_video_court import (
    CAMERAS,
    CLIPS,
    shared_fixture,
)
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    _camera,
    materialize_assets,
    runtime,
)


def second_clip_balls(stages, kind):
    # The first clip retains the fixture's fully observed 24-frame rally.
    frames = 24
    xyz = np.c_[1 + np.arange(frames) * .05, -8 + np.arange(frames) * .3, np.full(frames, 1.4)]
    if kind == "ambiguous":
        xyz = np.c_[np.zeros(frames), np.zeros(frames), np.linspace(.4, 4.5, frames)]
    if kind == "static":
        xyz[:] = xyz[0]
    observed = np.arange(frames) < (10 if kind == "conflict" else frames)
    for camera, center in zip(CAMERAS, ([-8., -16., 9.], [8., -16., 9.], [5., 16., 9.]), strict=True):
        physical = _camera(camera, center).half_turned(camera == "cam2" and kind == "conflict")
        stage = stages[f"ball_detection/{camera}"]
        stage.result = replace(stage.result, uv_px=np.where(observed[:, None], physical.project(xyz)[0], 0).astype(np.float32),
                               observed=observed, confidence=(observed * .9).astype(np.float32), point_kind=observed.astype(np.uint8))


def prefix_stores(tmp_path, monkeypatch, kind):
    pipeline, paths, sources, _, _, court = shared_fixture(tmp_path, monkeypatch)
    pins = {}
    for i, source in enumerate(sources):
        if i == 1:
            second_clip_balls(pipeline.components, kind)
        store = ClipStore(tmp_path / f"clip-{i}", json_value(source))
        nodes = standard_definition(pipeline.config, source, code_identity=pipeline.code_identity, video_court=court,
                                    overrides=pipeline.components)
        runner = ComponentRunner(nodes, store)
        runner.run(targets=["court_calibration", *(f"ball_points/{c}" for c in CAMERAS)])
        pins[source.clip_id] = PinnedSideInputs.from_store(store.root)
    return pipeline, paths, sources, court, pins


def build(pipeline, court, pins, tmp_path, **options):
    return build_video_court_side(pins, court=court, directory=tmp_path / "side", thresholds=pipeline.config.court_side,
                                  max_frames=pipeline.config.sampling_max_frames, code_identity=pipeline.code_identity, **options)


def acquisition(tmp_path, **kwargs):
    return VideoArtifactConfig(directory=tmp_path / "side", dataset_id="fixture", video_id="video_000", rebuild=False, **kwargs)


@pytest.mark.parametrize(("kind", "status", "reason"), [
    ("agreement", "ok", None),
    ("static", "warning", "court_side_insufficient_frames"),
    ("ambiguous", "warning", "court_side_ambiguous_margin"),
    ("conflict", "failed", "court_side_video_shared_conflict"),
])
def test_clip_verdict_and_full_video_decision(tmp_path, monkeypatch, kind, status, reason):
    pipeline, paths, _, court, pins = prefix_stores(tmp_path, monkeypatch, kind)
    artifact = build(pipeline, court, pins, tmp_path)
    d = artifact.decision
    assert d.shared.hypotheses[0].view_half_turns == (False, False, True) and d.shared.reason is None
    assert d.evidence_clip_ids == CLIPS and artifact.clips[CLIPS[0]].standalone.reason is None
    assert d.shared.frames == sum(len(c.scored_frame_indices) for c in artifact.clips.values())
    assert all(c.scored_frame_indices[0] == 0 for c in artifact.clips.values())
    if kind == "static":
        assert artifact.clips[CLIPS[1]].scored_frame_indices == (0,)
    load = acquisition(tmp_path, action="load", artifact_digest=artifact.digest, request=None, request_sha256=None)
    pipeline.config = replace(pipeline.config, video_court_side=load)
    store = pins[CLIPS[1]].directory
    if status == "failed":
        with pytest.raises(ReconstructionUnavailable) as stopped:
            pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=store, clip_id=CLIPS[1])
        assert stopped.value.reason == reason and stopped.value.diagnostics["standalone"]["margin"] >= pipeline.config.court_side.min_margin
        assert not (store / "exports").exists()
    else:
        # Only execute the side target: a static/ambiguous ball may independently fail a later 3D gate.
        nodes = standard_definition(pipeline.config, pins[CLIPS[1]].source, code_identity=pipeline.code_identity,
                                    overrides=pipeline.components, video_court=court, video_court_side=artifact)
        runner = ComponentRunner(nodes, ClipStore(store, json_value(pins[CLIPS[1]].source)))
        runner.run(["court_side"])
        assert runner.output("court_side") == artifact.output()
        assert runner.nodes["court_side"].settings["reason"] == reason
    report = artifact.clip_report(CLIPS[1], "warn")
    assert (report["status"], report["reason"]) == (status, reason)
    if status == "warning":
        assert artifact.clip_report(CLIPS[1], "fail")["status"] == "failed"
    if status == "failed":
        assert json.loads((store / "run.json").read_text())["video_court_side"] == report
    assert len(report["standalone"]["hypotheses"]) == 4


def test_short_clips_only_decide_together_and_warn_in_single_entry(tmp_path, monkeypatch):
    pipeline, paths, _, court, pins = prefix_stores(tmp_path, monkeypatch, "agreement")
    pipeline.config = replace(pipeline.config, court_side=replace(pipeline.config.court_side, min_frames=30))
    artifact = build(pipeline, court, pins, tmp_path)
    assert [artifact.clips[c].standalone.frames for c in CLIPS] == [24, 24]
    assert artifact.output().frames == 48
    load = acquisition(tmp_path, action="load", artifact_digest=artifact.digest, request=None, request_sha256=None)
    pipeline.config = replace(pipeline.config, video_court_side=load)
    root = pins[CLIPS[0]].directory
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    assert scene.metadata["court_reference"]["view_half_turns"] == [False, False, True]
    receipt = json.loads((root / "run.json").read_text())["video_court_side"]
    assert receipt["status"] == "warning" and receipt["standalone"]["reason"] == "insufficient_frames"
    assert receipt["shared"]["reason"] is None and receipt["shared"]["frames"] == 48


def test_build_once_load_and_invalidate_only_side_consumers(tmp_path, monkeypatch):
    pipeline, paths, sources, court, pins = prefix_stores(tmp_path, monkeypatch, "agreement")
    request = tmp_path / "inputs.json"
    rows = {c: {"role": "artifact", "directory": pin.directory.name, "inputs_digest": pin.digest} for c, pin in pins.items()}
    write_json_atomic(request, {"schema": "video_court_side_inputs_v1", "clips": rows, "evidence_clip_ids": None})
    cfg = runtime(tmp_path, overrides=(
        "court_kp.mode=video_shared", "court_kp.video_shared.action=load", "court_kp.video_shared.directory=shared",
        "court_kp.video_shared.dataset_id=fixture", "court_kp.video_shared.video_id=video_000",
        f"court_kp.video_shared.artifact_digest={court.digest}", "court_side.mode=video_shared",
        "court_side.video_shared.action=build", "court_side.video_shared.directory=side",
        "court_side.video_shared.request=inputs.json", f"court_side.video_shared.request_sha256={dual_sha256(request)}",
    ))
    pipeline.config = cfg
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    root = pins[CLIPS[0]].directory
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    assert scene.metadata["court_reference"]["view_half_turns"] == [False, False, True]
    shared = pipeline.video_court_side
    assert shared is not None
    receipt = json.loads((root / "run.json").read_text())["video_court_side"]
    assert receipt["artifact_digest"] == shared.digest and receipt["status"] == "ok"
    before = pipeline.last_store.references()
    assert pipeline.publication_identity()["video_court_side_digest"] == shared.digest

    def cannot_score(*args, **kw):
        pytest.fail("build cache and explicit load must reuse the pinned measurements")
    scorer = "src.tennis_scene.pipeline.video_court_side.prepare.collect_side_evidence_with_frames"
    with monkeypatch.context() as cached:
        cached.setattr(scorer, cannot_score)
        assert build(pipeline, court, pins, tmp_path).digest == shared.digest

    # Change evidence only in the other clip. Its new pin must invalidate the
    # decision for this clip, while the original pinned build request fails.
    second_clip_balls(pipeline.components, "static")
    nodes = standard_definition(replace(cfg, video_court_side=None), sources[1], code_identity=pipeline.code_identity,
                                overrides=pipeline.components, video_court=court)
    other = ClipStore(pins[CLIPS[1]].directory, json_value(sources[1]))
    ComponentRunner(nodes, other, overwrite=True).run([f"ball_points/{c}" for c in CAMERAS])
    with pytest.raises(ValueError, match="inputs changed"):
        prepare_video_court_side(cfg.video_court_side, cfg.resolver, court=court, thresholds=cfg.court_side,
                                max_frames=cfg.sampling_max_frames, code_identity=pipeline.code_identity)
    pins[CLIPS[1]] = PinnedSideInputs.from_store(other.root)
    changed = build(pipeline, court, pins, tmp_path)
    assert changed.digest != shared.digest and changed.output().frames == shared.output().frames // 2 + 1
    monkeypatch.setattr(scorer, cannot_score)
    request.unlink()
    load = acquisition(tmp_path, action="load", artifact_digest=changed.digest, request=None, request_sha256=None)
    pipeline.config = replace(cfg, video_court_side=load)
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    after = pipeline.last_store.references()
    for node in ("ball_points/cam0", "court_calibration", "person_detection/cam0"):
        assert after[node] == before[node]
    for node in ("court_side", "camera_alignment", "scene_assembly"):
        assert after[node].execution_key != before[node].execution_key
    assert pipeline.publication_identity()["video_court_side_digest"] == changed.digest
    with pytest.raises(ValueError, match="digest"):
        standard_definition(pipeline.config, sources[0], code_identity=pipeline.code_identity, video_court=court, video_court_side=shared)
    with pytest.raises(ValueError, match="thresholds"):
        prepare_video_court_side(load, cfg.resolver, court=court, thresholds=replace(cfg.court_side, min_margin=.01),
                                max_frames=cfg.sampling_max_frames, code_identity=pipeline.code_identity)
    diagnostics_path = changed.directory / "objects" / f"{changed.decision.diagnostics_digest}.json"
    diagnostics = diagnostics_path.read_bytes()
    diagnostics_path.write_text("{}")
    with pytest.raises(ValueError, match="content changed"):
        pipeline.prepare_video_court_side(court)
    diagnostics_path.write_bytes(diagnostics)
    path = changed.directory / "objects" / f"{changed.digest}.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="content changed"):
        pipeline.prepare_video_court_side(court)


def test_reject_foreign_missing_and_stale_inputs(tmp_path, monkeypatch):
    pipeline, _, sources, court, pins = prefix_stores(tmp_path, monkeypatch, "static")
    with pytest.raises(ValueError, match="all clips"):
        build(pipeline, court, {CLIPS[0]: pins[CLIPS[0]]}, tmp_path)
    with pytest.raises(ValueError, match="membership"):
        build(pipeline, court, {**pins, CLIPS[1]: replace(pins[CLIPS[1]], source=replace(sources[1], clip_id="other/clip_a"))}, tmp_path)
    with pytest.raises(ValueError, match="repeated"):
        build(pipeline, court, pins, tmp_path, evidence_clip_ids=[CLIPS[0], CLIPS[0]])
    artifact = build(pipeline, court, pins, tmp_path)
    from src.tennis_scene.pipeline.video_court_side.component import (
        SharedCourtSideModule,
    )
    node = SharedCourtSideModule(CAMERAS, artifact, court, "warn")
    local = pins[CLIPS[0]].assemble()
    with pytest.raises(ValueError, match="observations differ"):
        node.process(replace(local, balls=pins[CLIPS[1]].assemble().balls))
    incorrect = replace(local.calibration, court=replace(local.calibration.court, diagnostics={"temporal_policy": "static_first_frame_broadcast"}))
    with pytest.raises(ValueError, match="pinned video_shared"):
        node.process(replace(local, calibration=incorrect))
    from src.tasks.court_detection.geometry.video_drift import DriftConfig
    from src.tennis_scene.pipeline.video_court.diagnostics import publish_video_drift
    from tests.integration.tennis_scene.test_video_court import (
        independent_drift_reports,
    )
    diagnosed = publish_video_drift(court, tmp_path / "drift", protocol={"settings": asdict(DriftConfig())},
                                    reports=independent_drift_reports(court))
    with pytest.raises(ValueError, match="drift evidence"):
        build(pipeline, diagnosed, pins, tmp_path)
    with pytest.raises(ValueError, match="drift evidence"):
        artifact.check_settings(diagnosed, pipeline.config.court_side, pipeline.config.sampling_max_frames)
    # Matching court provenance alone cannot excuse a changed numerical calibration.
    calibration = local.calibration.calibration
    views = calibration.views
    altered = replace(calibration, views=(replace(views[0], camera=views[0].camera.half_turned(True)), *views[1:]))
    changed = replace(local, calibration=replace(local.calibration, calibration=altered))
    assemble = PinnedSideInputs.assemble
    with monkeypatch.context() as different:
        different.setattr(PinnedSideInputs, "assemble", lambda pin: changed if pin.source.clip_id == CLIPS[0] else assemble(pin))
        with pytest.raises(ValueError, match="calibration changed"):
            build(pipeline, court, pins, tmp_path, rebuild=True)

    strict = SharedCourtSideModule(CAMERAS, artifact, court, "fail")
    with pytest.raises(ReconstructionUnavailable, match="insufficient_frames"):
        strict.process(pins[CLIPS[1]].assemble())
    undecided = build(pipeline, court, pins, tmp_path, evidence_clip_ids=[CLIPS[1]])
    assert undecided.decision.shared.reason is not None
    with pytest.raises(ReconstructionUnavailable, match="court_side_video_shared_insufficient_frames"):
        SharedCourtSideModule(CAMERAS, undecided, court, "warn").process(local)


@pytest.mark.parametrize(("overrides", "message"), [
    (("court_side.mode=video_shared",), "requires court_kp.mode=video_shared"),
    (("court_side.mode=guess",), "must be clip or video_shared"),
    (("court_side.video_shared.action=build",), "require court_side.mode=video_shared"),
])
def test_mode_must_be_explicit_and_calibrated(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        runtime(tmp_path, overrides=overrides)
