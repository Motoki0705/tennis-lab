"""Fixed geometry and measured ball tracks through the real component stores and single-clip runner."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.court_side import CourtSideModule
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


def prefix_stores(tmp_path, monkeypatch, kind, *, uncertain_drift=False):
    pipeline, paths, sources, _, _, court = shared_fixture(tmp_path, monkeypatch)
    first_balls = {name: stage.result for name, stage in pipeline.components.items() if name.startswith('ball_detection/')}
    # These stubs stand in for source-owned observations, including a later
    # explicit overwrite of another clip. Their model configuration stays fixed
    # while the measured outputs differ per clip.
    for camera in CAMERAS:
        monkeypatch.setattr(pipeline.components[f'ball_detection/{camera}'], 'cache_identity',
                            lambda: {'fixture': 'source_owned_ball_observations'})
    if uncertain_drift:
        from src.tasks.court_detection.geometry.video_drift import (
            DriftConfig,
            summarize_clip,
        )
        from src.tennis_scene.pipeline.video_court.diagnostics import (
            publish_video_drift,
        )
        from tests.integration.tennis_scene.test_video_court import (
            independent_drift_reports,
        )

        reports = independent_drift_reports(court)
        frame = reports["cam1"][CLIPS[1]]["frames"][0]
        frame.update(status="failed", reason="drift_unobservable")
        reports["cam1"][CLIPS[1]] = summarize_clip([frame])
        court = publish_video_drift(court, tmp_path / "drift", protocol={"settings": asdict(DriftConfig())}, reports=reports)
        pipeline.config = replace(pipeline.config, video_court_drift=court.drift, video_court_unobservable_policy="warn")
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
    # Each known output belongs to its source clip. Restoring clip A also keeps
    # its explicit override identity stable when clip B has different results.
    for name, output in first_balls.items():
        pipeline.components[name].result = output
    return pipeline, paths, sources, court, pins


def build(pipeline, court, pins, tmp_path, **options):
    return build_video_court_side(pins, court=court, directory=tmp_path / "side", thresholds=pipeline.config.court_side,
                                  max_frames=pipeline.config.sampling_max_frames, code_identity=pipeline.code_identity, **options)


def acquisition(tmp_path, **kwargs):
    return VideoArtifactConfig(directory=tmp_path / "side", dataset_id="fixture", video_id="video_000", rebuild=False, **kwargs)


def build_request(tmp_path, pins, selection=None):
    path = tmp_path / "inputs.json"
    rows = {clip: {"role": "artifact", "directory": pin.directory.name, "inputs_digest": pin.digest} for clip, pin in pins.items()}
    write_json_atomic(path, {"schema": "video_court_side_inputs_v1", "clips": rows, "evidence_clip_ids": selection})
    return acquisition(tmp_path, action="build", request=path, request_sha256=dual_sha256(path), artifact_digest=None)


def republish_second_ball_stream(pipeline, court, store, source):
    previous = {name: stage.result for name, stage in pipeline.components.items() if name.startswith('ball_detection/')}
    second_clip_balls(pipeline.components, "static")
    nodes = standard_definition(replace(pipeline.config, video_court_side=None), source, code_identity=pipeline.code_identity,
                                overrides=pipeline.components, video_court=court)
    ComponentRunner(nodes, ClipStore(store, json_value(source)), overwrite=True).run([f"ball_points/{c}" for c in CAMERAS])
    for name, output in previous.items():
        pipeline.components[name].result = output


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
        override = CourtSideModule(CAMERAS, pipeline.config.court_side, max_frames=pipeline.config.sampling_max_frames)
        with monkeypatch.context() as patch:
            patch.setitem(pipeline.components, "court_side", override)
            clip_nodes = standard_definition(replace(pipeline.config, video_court_side=None), pins[CLIPS[1]].source,
                code_identity=pipeline.code_identity, overrides=pipeline.components, video_court=court)
            assert next(n.component for n in clip_nodes if n.name == "court_side") is override
            with pytest.raises(ValueError, match="court_side.*override"):
                pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=store, clip_id=CLIPS[1])
            assert pipeline.last_runner is None and pipeline.last_store.active("court_side") is None
        with pytest.raises(ReconstructionUnavailable) as stopped:
            pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=store, clip_id=CLIPS[1])
        assert stopped.value.reason == reason and stopped.value.diagnostics["standalone"]["margin"] >= pipeline.config.court_side.min_margin
        assert not (store / "exports").exists()
    else:
        # Only execute the side target: a static/ambiguous ball may independently fail a later 3D gate.
        second_clip_balls(pipeline.components, kind)
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
    request = build_request(tmp_path, pins).request
    cfg = runtime(tmp_path, overrides=(
        "court_kp.mode=video_shared", "court_kp.video_shared.action=load", "court_kp.video_shared.directory=shared",
        "court_kp.video_shared.dataset_id=fixture", "court_kp.video_shared.video_id=video_000",
        f"court_kp.video_shared.artifact_digest={court.digest}", "court_side.mode=video_shared",
        "court_side.video_shared.action=build", "court_side.video_shared.directory=side",
        "court_side.video_shared.request=inputs.json", f"court_side.video_shared.request_sha256={dual_sha256(request)}",
    ))
    assert cfg.video_court_side is not None
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
        assert pipeline.publication_identity()["video_court_side_digest"] == shared.digest

    # Change evidence only in the other clip. Its new pin must invalidate the
    # decision for this clip, while the original pinned build request fails.
    other = pins[CLIPS[1]].directory
    republish_second_ball_stream(pipeline, court, other, sources[1])
    with pytest.raises(ValueError, match="inputs changed"):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    assert pipeline.last_runner is None and pipeline.last_receipt["status"] == "failed"
    pins[CLIPS[1]] = PinnedSideInputs.from_store(other)
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


def test_repeated_build_verifies_request_bytes_before_publication(tmp_path, monkeypatch):
    pipeline, _, _, court, pins = prefix_stores(tmp_path, monkeypatch, "agreement")
    config = build_request(tmp_path, pins)
    pipeline.config = replace(pipeline.config, video_court_side=config)
    materialize_assets(enabled_model_assets(pipeline.config).values(), tmp_path)
    assert pipeline.publication_identity()["video_court_side_digest"] == pipeline.video_court_side.digest
    config.request.write_bytes(config.request.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="build request changed"):
        pipeline.publication_identity()


def test_prepared_decision_must_match_request_selection_and_every_revision(tmp_path, monkeypatch):
    pipeline, _, sources, court, pins = prefix_stores(tmp_path, monkeypatch, "agreement")
    subset = build(pipeline, court, pins, tmp_path, evidence_clip_ids=[CLIPS[0]])

    def plan(config, artifact):
        return standard_definition(replace(pipeline.config, video_court_side=config), sources[0], code_identity=pipeline.code_identity,
                                   overrides=pipeline.components, video_court=court, video_court_side=artifact)

    # Both clips have diagnostics and agree on the winner, even for a subset build.
    full_request = build_request(tmp_path, pins)
    with pytest.raises(ValueError, match="evidence clip"):
        plan(full_request, subset)
    full = build(pipeline, court, pins, tmp_path)
    load_full = acquisition(tmp_path, action="load", artifact_digest=full.digest, request=None, request_sha256=None)
    with pytest.raises(ValueError, match="digest"):
        plan(load_full, subset)
    subset_request = build_request(tmp_path, pins, [CLIPS[0]])
    assert next(n for n in plan(subset_request, subset) if n.name == "court_side").settings["artifact_digest"] == subset.digest

    # B is diagnostic-only, so its new ball revision leaves the pooled score unchanged.
    republish_second_ball_stream(pipeline, court, pins[CLIPS[1]].directory, sources[1])
    pins[CLIPS[1]] = PinnedSideInputs.from_store(pins[CLIPS[1]].directory)
    revised = build_request(tmp_path, pins, [CLIPS[0]])
    with pytest.raises(ValueError, match="input revisions"):
        plan(revised, subset)
    current = build(pipeline, court, pins, tmp_path, evidence_clip_ids=[CLIPS[0]])
    assert current.output() == subset.output() and current.digest != subset.digest
    # The load request pins the selection and revisions via the immutable digest.
    load = acquisition(tmp_path, action="load", artifact_digest=current.digest, request=None, request_sha256=None)
    revised.request.unlink()
    with pytest.raises(ValueError, match="digest"):
        plan(load, subset)
    assert next(n for n in plan(load, current) if n.name == "court_side").settings["artifact_digest"] == current.digest


def test_reject_foreign_missing_and_stale_inputs(tmp_path, monkeypatch):
    pipeline, _, sources, court, pins = prefix_stores(tmp_path, monkeypatch, "static")
    local_pin = pins[CLIPS[0]]
    assert PinnedSideInputs.from_store(Path(os.path.relpath(local_pin.directory))) == local_pin
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
    (("court_kp.mode=video_shared", "court_kp.video_shared.action=load", "court_kp.video_shared.directory=shared",
      "court_kp.video_shared.dataset_id=fixture", "court_kp.video_shared.video_id=video_000",
      "court_kp.video_shared.artifact_digest=" + "a" * 64, "court_side.mode=video_shared",
      "court_side.video_shared.directory=side"), "acquisition and scope must be explicit"),
])
def test_mode_must_be_explicit_and_calibrated(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        runtime(tmp_path, overrides=overrides)


@pytest.mark.parametrize("action", ["build", "load"])
def test_current_drift_policy_covers_every_contributing_clip(tmp_path, monkeypatch, action):
    pipeline, paths, sources, court, pins = prefix_stores(tmp_path, monkeypatch, "agreement", uncertain_drift=True)
    full = build(pipeline, court, pins, tmp_path)
    selected = build(pipeline, court, pins, tmp_path, evidence_clip_ids=[CLIPS[0]])

    def configuration(artifact):
        if action == "build":
            return build_request(tmp_path, pins, list(artifact.decision.evidence_clip_ids))
        return acquisition(tmp_path, action="load", artifact_digest=artifact.digest, request=None, request_sha256=None)

    pipeline.config = replace(pipeline.config, video_court_side=configuration(full))
    assert pipeline.prepare_video_court_side(court).digest == full.digest
    pipeline.config = replace(pipeline.config, video_court_unobservable_policy="fail")
    if action == "build":
        # Reject both a disk hit and a forced fresh build before any scoring.
        for rebuild in (False, True):
            with pytest.raises(ReconstructionUnavailable, match="drift_unobservable"):
                build(pipeline, court, pins, tmp_path, court_policy="fail", rebuild=rebuild)

    def run_a():
        return pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS,
                            store_root=pins[CLIPS[0]].directory, clip_id=CLIPS[0])

    def bind(artifact, source):
        return standard_definition(pipeline.config, source, code_identity=pipeline.code_identity, overrides=pipeline.components,
                                   video_court=court, video_court_side=artifact)

    def prepare_from_disk():
        cfg = pipeline.config
        return prepare_video_court_side(cfg.video_court_side, cfg.resolver, court=court, thresholds=cfg.court_side,
                                        max_frames=cfg.sampling_max_frames, code_identity=pipeline.code_identity,
                                        court_policy=cfg.video_court_unobservable_policy)

    for adopt in (run_a, prepare_from_disk, lambda: bind(full, sources[0]), pipeline.publication_identity):
        with pytest.raises(ReconstructionUnavailable, match="drift_unobservable") as error:
            adopt()
        assert error.value.diagnostics["camera_id"] == "cam1" and error.value.diagnostics["clip_id"] == CLIPS[1]
    assert pipeline.last_runner is None

    # B keeps a standalone diagnostic. The strict court allows A to contribute and use its result.
    pipeline.config = replace(pipeline.config, video_court_side=configuration(selected))
    assert pipeline.prepare_video_court_side(court).digest == selected.digest
    assert next(n for n in bind(selected, sources[0]) if n.name == "court_side").settings["artifact_digest"] == selected.digest
    materialize_assets(enabled_model_assets(pipeline.config).values(), tmp_path)
    admitted = pipeline.publication_identity()["video_court_side_clips"]
    assert admitted[CLIPS[0]]["status"] == "ok" and "drift_unobservable" in admitted[CLIPS[1]]["detail"]
    with pytest.raises(ReconstructionUnavailable, match="drift_unobservable"):
        run = ComponentRunner(bind(selected, sources[1]), ClipStore(pins[CLIPS[1]].directory, json_value(sources[1])))
        run.run(["court_side"])


@pytest.mark.parametrize("node", ["ball_points/cam1", "court_calibration"])
def test_build_reuse_rejects_other_clips_damaged_array_without_rescoring(tmp_path, monkeypatch, node):
    pipeline, paths, sources, court, pins = prefix_stores(tmp_path, monkeypatch, "agreement")
    request = build_request(tmp_path, pins)
    pipeline.config = replace(pipeline.config, video_court_side=request)
    artifact = pipeline.prepare_video_court_side(court)
    materialize_assets(enabled_model_assets(pipeline.config).values(), tmp_path)
    monkeypatch.setattr("src.tennis_scene.pipeline.video_court_side.prepare.collect_side_evidence_with_frames",
                        lambda *a, **kw: pytest.fail("reused builds must not score"))
    assert pipeline.publication_identity()["video_court_side_digest"] == artifact.digest
    other = pins[CLIPS[1]]
    descriptor = other.directory / other.references[node]["path"]
    document = json.loads(descriptor.read_text())
    if node == "court_calibration":
        descriptor.with_name(next(iter(document["arrays"]))).unlink()
    else:
        descriptor.with_name(document["payload"]["uv_px"]["array"]).write_bytes(b"damaged ball coordinates")
    assert PinnedSideInputs.from_store(other.directory).digest == other.digest

    for adopt in (lambda: pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS,
                                       store_root=pins[CLIPS[0]].directory, clip_id=CLIPS[0]),
                  pipeline.publication_identity, lambda: build(pipeline, court, pins, tmp_path),
                  lambda: standard_definition(pipeline.config, sources[0], code_identity=pipeline.code_identity,
                      overrides=pipeline.components, video_court=court, video_court_side=artifact)):
        with pytest.raises(ValueError, match="array checksum"):
            adopt()
    assert pipeline.last_runner is None
    # Explicit load only needs the immutable video objects and the current clip's own inputs.
    pipeline.config = replace(pipeline.config, video_court_side=acquisition(tmp_path, action="load", artifact_digest=artifact.digest,
                                                                          request=None, request_sha256=None))
    assert pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=pins[CLIPS[0]].directory,
                        clip_id=CLIPS[0]).metadata["court_reference"]["view_half_turns"] == list(artifact.output().view_half_turns)


@pytest.mark.parametrize("kind,global_load", [("agreement", False), ("conflict", True)])
def test_shared_load_requires_its_component_and_an_admissible_clip(tmp_path, monkeypatch, kind, global_load):
    from src.tennis_scene.pipeline.imports.publish import bind_import, publish_import

    pipeline, _, sources, court, pins = prefix_stores(tmp_path, monkeypatch, kind)
    source = sources[1]
    store = ClipStore(pins[source.clip_id].directory, json_value(source))
    artifact = build(pipeline, court, pins, tmp_path)
    cfg = replace(pipeline.config, component_sources={**pipeline.config.component_sources, "court_side": "load"})
    ordinary_nodes = standard_definition(cfg, source, code_identity=pipeline.code_identity, overrides=pipeline.components, video_court=court)
    bound = bind_import(ordinary_nodes, "court_side", store)
    alternate = tuple(replace(h, view_half_turns=(*h.view_half_turns[:-1], not h.view_half_turns[-1]))
                      for h in artifact.output().hypotheses)
    imported = replace(artifact.output(), view_half_turns=alternate[0].view_half_turns, hypotheses=alternate)
    publish_import(bound, store, imported, importer="fixture", version=1, identity={"file_sha256": "a" * 64}, provenance={})
    regular = ComponentRunner(ordinary_nodes, store)
    regular.run(["court_side"])
    assert regular.output("court_side") == imported

    policy = acquisition(tmp_path, action="load", artifact_digest=artifact.digest, request=None, request_sha256=None)
    cfg = replace(cfg, video_court_side=policy, cache_source="load" if global_load else cfg.cache_source)
    nodes = standard_definition(cfg, source, code_identity=pipeline.code_identity, overrides=pipeline.components,
                                video_court=court, video_court_side=artifact)
    runner = ComponentRunner(nodes, store)
    if kind == "conflict":
        with pytest.raises(ReconstructionUnavailable, match="court_side_video_shared_conflict"):
            runner.run(["court_side"])
    else:
        with pytest.raises(ValueError, match="component origin"):
            runner.run(["court_side"])
    assert runner.statuses["court_side"] == "failed" and "court_side" not in runner.references
    with pytest.raises(ValueError, match="component origin"):
        bind_import(nodes, "court_side", store)
    with pytest.raises(ValueError, match="component origin"):
        publish_import(replace(bound, node=runner.nodes["court_side"]), store, imported, importer="fixture",
                       version=1, identity={}, provenance={})


def test_shared_marker_requires_an_admitted_clip_and_its_published_component(tmp_path, monkeypatch):
    import shutil

    from src.tennis_scene.pipeline.artifacts import document_digest
    from src.tennis_scene.pipeline.clip_inputs import (
        resolve_dataset_clip,
        select_dataset_clips,
    )
    from src.tennis_scene.pipeline.dataset_publication import (
        ANNOTATION_RELATIVE_DIR,
        completed_annotation,
        publish_annotation,
    )
    from src.tennis_scene.pipeline.imports.publish import bind_import, publish_import
    from tests.integration.tennis_scene.test_video_court import make_build_request

    pipeline, paths, sources, court, pins = prefix_stores(tmp_path, monkeypatch, "conflict")
    make_build_request(tmp_path, paths)
    dataset = tmp_path / "dataset"
    a, b = [resolve_dataset_clip(dataset, record) for record in select_dataset_clips(dataset, CLIPS)]
    cfg = replace(pipeline.config, video_court_side=build_request(tmp_path, pins))
    pipeline.config = cfg
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    identity = pipeline.publication_identity()
    with monkeypatch.context() as patch:
        patch.setitem(pipeline.components, "court_side", CourtSideModule(CAMERAS, cfg.court_side, max_frames=cfg.sampling_max_frames))
        with pytest.raises(ValueError, match="court_side.*override"):
            pipeline.publication_identity()
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=pins[CLIPS[0]].directory, clip_id=CLIPS[0])
    directory = a.clip_dir / ANNOTATION_RELATIVE_DIR
    shutil.copytree(pins[CLIPS[0]].directory, directory)
    marker = publish_annotation(a, scene, pipeline_config_yaml="fixture: true\n", publication_identity=identity)
    assert completed_annotation(a, identity) == marker
    with pytest.raises(ValueError, match="shared decision or policy"):
        publish_annotation(a, scene, pipeline_config_yaml="", publication_identity={**identity, "video_court_side_digest": "b" * 64})

    # Even an otherwise matching completion marker cannot admit the conflicting clip.
    other_marker = b.clip_dir / ANNOTATION_RELATIVE_DIR / "annotation.json"
    other_marker.parent.mkdir(parents=True)
    write_json_atomic(other_marker, {**json.loads(marker.read_text()), "clip_id": CLIPS[1],
                                     "publication_identity_sha256": document_digest(identity),
                                     "clip_manifest_sha256": b.manifest_sha256, "media_sha256": b.media_sha256})
    with pytest.raises(ReconstructionUnavailable, match="court_side_video_shared_conflict"):
        completed_annotation(b, identity)
    with pytest.raises(ReconstructionUnavailable, match="court_side_video_shared_conflict"):
        publish_annotation(b, scene, pipeline_config_yaml="", publication_identity=identity)

    # Replacing A's active side withdraws its scene export, but leaves the old marker file on disk.
    ordinary = replace(cfg, video_court_side=None, component_sources={**cfg.component_sources, "court_side": "load"})
    nodes = standard_definition(ordinary, sources[0], code_identity=pipeline.code_identity, overrides=pipeline.components, video_court=court)
    store = ClipStore(directory, json_value(sources[0]))
    publish_import(bind_import(nodes, "court_side", store), store, pipeline.video_court_side.output(),
                   importer="fixture", version=1, identity={}, provenance={})
    with pytest.raises(ValueError, match="component origin"):
        completed_annotation(a, identity)
    with pytest.raises(ValueError, match="component origin"):
        publish_annotation(a, scene, pipeline_config_yaml="", publication_identity=identity)


def test_shared_component_reuse_matches_origin_digest_and_policy(tmp_path, monkeypatch):
    from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
    from src.tennis_scene.pipeline.video_court_side.component import (
        SharedCourtSideModule,
    )

    pipeline, _, sources, court, pins = prefix_stores(tmp_path, monkeypatch, "agreement")
    shared = build(pipeline, court, pins, tmp_path)
    subset = build(pipeline, court, pins, tmp_path, evidence_clip_ids=[CLIPS[0]])
    store = ClipStore(pins[CLIPS[0]].directory, json_value(sources[0]))

    def runner(artifact, mode, *, policy="warn"):
        cfg = replace(pipeline.config, video_court_side=acquisition(tmp_path, action="load", artifact_digest=artifact.digest,
                      request=None, request_sha256=None), court_side_unobservable_policy=policy,
                      component_sources={**pipeline.config.component_sources, "court_side": mode})
        nodes = standard_definition(cfg, sources[0], code_identity=pipeline.code_identity, overrides=pipeline.components,
                                    video_court=court, video_court_side=artifact)
        return ComponentRunner(nodes, store)

    fresh = runner(shared, "execute")
    fresh.run(["court_side"])
    assert fresh.statuses["court_side"] == "executed"
    monkeypatch.setattr(SharedCourtSideModule, "process", lambda *a: pytest.fail("cache/load should reuse the shared component"))
    for mode, status in (("execute", "cached"), ("load", "loaded")):
        reuse = runner(shared, mode)
        reuse.run(["court_side"])
        assert reuse.statuses["court_side"] == status and reuse.output("court_side") == shared.output()
    for mismatch in (runner(subset, "load"), runner(shared, "load", policy="fail")):
        with pytest.raises(ValueError, match="component identity changed"):
            mismatch.run(["court_side"])

    # An execution key alone cannot authorize an import, even on an automatic cache hit.
    decision = fresh.plan.decide((CLIPS[0], "court_side"))
    node = fresh.nodes["court_side"]
    ref = store.publish("court_side", shared.output(), ArtifactCodec(node.io.output_type), schema=node.io.output_schema,
                        version=node.io.version, identity=decision.identity, dependencies=decision.dependencies, provenance={"origin": "import"})
    assert ref.execution_key == fresh.references["court_side"].execution_key
    with pytest.raises(ValueError, match="component origin"):
        runner(shared, "execute").run(["court_side"])


def test_shared_side_rejects_publication_only_resume_from_runtime_config(tmp_path, monkeypatch):
    from src.tennis_scene.pipeline.direct_annotation import DirectAnnotationConfig

    pipeline, paths, _, court, pins = prefix_stores(tmp_path, monkeypatch, "conflict")
    shared = build(pipeline, court, pins, tmp_path)
    direct = DirectAnnotationConfig(tmp_path / "request.json", "a" * 64, tmp_path / "bundle.json", "b" * 64,
                                    "stored_artifact", True, "publication")
    load = acquisition(tmp_path, action="load", artifact_digest=shared.digest, request=None, request_sha256=None)
    pipeline.config = replace(pipeline.config, video_court_side=load, direct_annotation=direct)
    with pytest.raises(ValueError, match="publication-only resume"):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=pins[CLIPS[1]].directory, clip_id=CLIPS[1])
    assert pipeline.last_runner is None and pipeline.last_receipt["status"] == "failed"
