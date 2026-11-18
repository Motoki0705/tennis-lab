"""Shared geometry, real stores/plans and entry-point provenance without models."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Lock
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.court import court_views_from_diagnostics
from src.tennis_scene.generate_dataset.manifest import register_exported_clip
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.clip_inputs import (
    prepare_clip_source,
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.dataset_publication import (
    completed_annotation,
    publish_annotation,
)
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
from src.tennis_scene.pipeline.imports.fixed_geometry import FixedGeometryImporter
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.video_court import VideoCourtConfig, prepare_video_court
from src.tennis_scene.pipeline.video_court.estimator import PROFILE
from src.tennis_scene.pipeline.video_court.store import VideoCourtStore
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    inputs,
    materialize_assets,
    runtime,
    setup_pipeline,
)

CAMERAS = ('cam0', 'cam1', 'cam2')
CLIPS = ('video_000/clip_a', 'video_000/clip_b')


def report(frames=24, *, failed=False):
    status = 'failed' if failed else 'ok'
    return {'status': status, 'frames': [{'frame_index': frames // 2, 'status': status,
                                         'reason': 'drift_unobservable' if failed else None}]}


def matrices():
    court, _, _ = inputs()
    assert court.diagnostics is not None
    return [np.asarray(camera['frames'][0]['homography_court_metres_to_image_pixels']) for camera in court.diagnostics['cameras']]


def shared_fixture(tmp_path, monkeypatch):
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    sources = [prepare_clip_source(paths, CAMERAS, clip_id=clip).source for clip in CLIPS]
    docs = {}
    for i, camera in enumerate(CAMERAS):
        members = {source.clip_id: {'sha256': source.video(camera).sha256, 'num_frames': source.num_frames, 'fps': source.fps} for source in sources}
        identity = {'profile': PROFILE, 'dataset_id': 'fixture', 'video_id': 'video_000', 'camera_id': camera,
                    'size_wh': [1280, 720], 'sources': members, 'implementation': 'fake-known-h'}
        docs[camera] = {'schema': 'video_court_camera_v1', 'dataset_id': 'fixture', 'video_id': 'video_000',
                        'camera_id': camera, 'size_wh': [1280, 720], 'identity': identity, 'sources': members,
                        'status': 'accepted', 'matrix': matrices()[i].tolist(), 'clips': {clip: report() for clip in CLIPS},
                        'diagnostics': {'profile': PROFILE, 'selected_initializer': f'{camera}/known-h', 'fit_frames': [[CLIPS[0], 1], [CLIPS[1], 2]]}}
    store = VideoCourtStore(tmp_path / 'shared')
    with store.lock():
        artifact = store.publish(docs)
    config = VideoCourtConfig('load', store.root, 'fixture', 'video_000', None, None, artifact.digest, False)
    pipeline.config = replace(pipeline.config, video_court=config,
                              processing_settings={**pipeline.config.processing_settings, 'court_kp': {'mode': 'video_shared'}})
    pipeline.components = {name: stage for name, stage in stages.items() if not name.startswith('court_detection/')}
    pipeline.code_identity = 'stable-test-implementation'
    return pipeline, paths, sources, docs, store, artifact


def test_shared_court_marker_tracks_appearance_implementation(tmp_path, monkeypatch):
    pipeline, _, _, _, _, _ = shared_fixture(tmp_path, monkeypatch)
    cfg = pipeline.config
    cfg = replace(cfg, player_association=replace(cfg.player_association,
        appearance=replace(cfg.player_association.appearance, encoder='dinov3_vitb16')))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    repository = cfg.roots.external_asset_root / 'dinov3'
    repository.mkdir(parents=True, exist_ok=True)
    source = repository / 'hubconf.py'
    source.write_text('# appearance implementation A\n')
    pipeline.config = cfg
    first = pipeline.publication_identity()
    source.write_text('# appearance implementation B\n')
    assert pipeline.publication_identity() != first


@pytest.mark.parametrize('separate_drift', [False, True])
def test_single_entry_and_multi_clip_plan_use_identical_shared_court_without_checkpoint(tmp_path, monkeypatch, separate_drift):
    pipeline, paths, sources, _, _, artifact = shared_fixture(tmp_path, monkeypatch)
    if separate_drift:
        from dataclasses import asdict

        from src.tasks.court_detection.geometry.video_drift import DriftConfig
        from src.tennis_scene.pipeline.video_court.diagnostics import (
            publish_video_drift,
        )

        artifact = publish_video_drift(artifact, tmp_path / 'drift', protocol={'settings': asdict(DriftConfig())},
                                       reports=independent_drift_reports(artifact))
        pipeline.config = replace(pipeline.config, video_court_drift=artifact.drift)
    # The checkpoint path is deliberately absent. All other model stages use
    # existing deterministic fixtures; shared court executes its real component.
    if pipeline.config.court_kp.checkpoint.exists():
        pipeline.config.court_kp.checkpoint.unlink()
    expected = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=tmp_path / 'single', clip_id=CLIPS[0])
    specs = []
    for i, source in enumerate(sources):
        nodes = standard_definition(pipeline.config, source, code_identity=pipeline.code_identity,
                                    overrides=pipeline.components, video_court=artifact)
        specs.append(ClipExecution(source.clip_id, nodes, ClipStore(tmp_path / f'multi-{i}', json_value(source))))
    executor = PlanExecutor(ExecutionPlan(specs))
    executor.run()
    first, _ = executor.finalize_clip(CLIPS[0])
    second, _ = executor.finalize_clip(CLIPS[1])
    np.testing.assert_array_equal(expected.court_kp, first.court_kp)
    np.testing.assert_array_equal(first.court_kp, second.court_kp)
    assert pipeline.last_runner is not None
    calibration = pipeline.last_runner.output('court_calibration')
    assert calibration.court.diagnostics['temporal_policy'] == 'static_video_shared_broadcast'
    assert calibration.court.diagnostics['observed_frame_indices'] == []
    with pytest.raises(ValueError, match='provenance'):
        court_views_from_diagnostics(calibration.court.diagnostics, CAMERAS, (1280, 720), 24)
    with pytest.raises(ValueError, match='only frame-zero'):
        FixedGeometryImporter(tmp_path / 'unused.json', 'a' * 64).bind_nodes(specs[0].nodes)


def test_warning_keeps_shared_geometry_and_is_visible_in_receipt_and_overlay(tmp_path, monkeypatch):
    from src.tennis_scene.visualization.review import gallery

    pipeline, paths, _, docs, store, original = shared_fixture(tmp_path, monkeypatch)
    warning = docs['cam0']['clips'][CLIPS[0]]
    warning['status'] = 'warning'
    warning['frames'][0].update(status='warning', reason='camera_moved_possible',
                                mean_displacement_px=5., max_displacement_px=5.)
    with store.lock():
        artifact = store.publish(docs)
    np.testing.assert_array_equal(artifact.cameras['cam0']['matrix'], original.cameras['cam0']['matrix'])
    pipeline.config = replace(pipeline.config, video_court=replace(pipeline.config.video_court, artifact_digest=artifact.digest))
    root = tmp_path / 'scene'
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    receipt = json.loads((root / 'run.json').read_text())
    assert receipt['status'] != 'failed'
    effective = receipt['video_court']['clip_diagnostics']['cam0']
    assert effective['status'] == warning['status']
    assert effective['unobservable_policy'] == 'warn'
    assert effective['frames'][0]['measurement_status'] == warning['frames'][0]['status']
    assert receipt['video_court']['clip_diagnostics']['cam1']['status'] == 'ok'
    messages = []
    original_text = gallery._text

    def record_text(frame, message, at, color=(255, 255, 255)):
        messages.append(message)
        original_text(frame, message, at, color)

    monkeypatch.setattr(gallery, '_text', record_text)
    monkeypatch.setattr(gallery.Review, 'frame', lambda *args: np.zeros((720, 1280, 3), np.uint8))
    review = gallery.Review(root / 'scene.json', tmp_path / 'review')
    payload, _ = review.payload('court_detection/cam0')
    _, details = review.render_court_detection('court_detection/cam0', 'cam0', payload)
    assert dict(details)['drift'] == 'warning'
    assert any('warning' in text and 'camera_moved_possible' in text for text in messages)


def test_camera_content_change_invalidates_its_court_and_descendants_only(tmp_path, monkeypatch):
    pipeline, paths, _, docs, store, artifact = shared_fixture(tmp_path, monkeypatch)
    root = tmp_path / 'clip-store'
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    docs['cam0']['matrix'][0][2] += .1
    with store.lock():
        changed = store.publish(docs)
    assert changed.manifest['cameras']['cam1'] == artifact.manifest['cameras']['cam1']
    pipeline.config = replace(pipeline.config, video_court=replace(pipeline.config.video_court, artifact_digest=changed.digest))
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    statuses = pipeline.last_runner.statuses
    assert statuses['court_detection/cam0'] == 'executed'
    assert statuses['court_detection/cam1'] == statuses['court_detection/cam2'] == 'cached'
    assert statuses['court_calibration'] == 'executed'
    assert statuses['person_detection/cam0'] == statuses['ball_detection/cam0'] == 'cached'
    assert statuses['scene_assembly'] == 'executed'


def test_corruption_is_rejected_even_with_complete_clip_cache_and_records_failure(tmp_path, monkeypatch):
    pipeline, paths, _, _, store, artifact = shared_fixture(tmp_path, monkeypatch)
    root = tmp_path / 'clip-store'
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    target = store.root / 'objects' / f'{artifact.manifest["cameras"]["cam1"]}.json'
    data = json.loads(target.read_text())
    data['matrix'][0][2] += 10
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='content changed'):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    assert json.loads((root / 'run.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('failure', ['no_h', 'drift', 'tolerance'])
def test_failed_camera_never_calls_clip_detector_or_uses_another_camera(tmp_path, monkeypatch, failure):
    pipeline, paths, _, docs, store, _ = shared_fixture(tmp_path, monkeypatch)
    docs['cam1']['clips'][CLIPS[0]] = report(failed=True)
    if failure == 'drift':
        pipeline.config = replace(pipeline.config, video_court_unobservable_policy='fail')
    elif failure == 'tolerance':
        docs['cam1']['clips'][CLIPS[0]]['frames'][0].update(reason='camera_drift_exceeds_tolerance',
                                                         mean_displacement_px=11., max_displacement_px=12.)
    if failure == 'no_h':
        docs['cam1']['status'], docs['cam1']['matrix'] = 'failed', None
        docs['cam1']['clips'] = {clip: report(failed=True) for clip in CLIPS}
    with store.lock():
        artifact = store.publish(docs)
    pipeline.config = replace(pipeline.config, video_court=replace(pipeline.config.video_court, artifact_digest=artifact.digest))
    with pytest.raises(ReconstructionUnavailable, match='cam1') as error:
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=tmp_path / 'clip-store', clip_id=CLIPS[0])
    assert error.value.reason == 'video_shared_court_failed'


@pytest.mark.parametrize('change', ['video', 'clip', 'camera', 'content', 'size', 'frames'])
def test_source_identity_mismatch_fails_before_court_execution(tmp_path, monkeypatch, change):
    _, _, sources, _, _, artifact = shared_fixture(tmp_path, monkeypatch)
    source = sources[0]
    if change in {'video', 'clip'}:
        source = replace(source, clip_id='other/clip_a' if change == 'video' else 'video_000/unknown')
    else:
        mutations: dict[str, dict[str, Any]] = {'camera': {'camera_id': 'foreign'}, 'content': {'sha256': '0' * 64},
                                               'size': {'width': 640}, 'frames': {'num_frames': 2}}
        values = mutations[change]
        videos = tuple(replace(v, **values) for v in source.videos) if change in {'size', 'frames'} else (replace(source.videos[0], **values), *source.videos[1:])
        source = replace(source, videos=videos)
    with pytest.raises(ValueError):
        artifact.bind(source)


def make_build_request(tmp_path, paths):
    dataset = tmp_path / 'dataset'
    for clip_id in CLIPS:
        folder = dataset / 'videos/video_000/clips' / clip_id.split('/')[1]
        folder.mkdir(parents=True)
        for path in paths:
            (folder / path.name).write_bytes(path.read_bytes())
        write_json_atomic(folder / 'clip.json', dict(version=2, dataset_id='fixture', clip_id=clip_id,
            video_id='video_000', clip_name=clip_id.split('/')[1], fps=30., num_frames=24, width=1280, height=720,
            global_start_sec=0., global_end_sec=.8, camera_ids=list(CAMERAS), video_paths=[p.name for p in paths],
            cameras=[], sync_source='clip_studio', exported_at='2026-11-17T00:00:00Z'))
        register_exported_clip(dataset, folder / 'clip.json')
    references = {}
    for camera in CAMERAS:
        path = tmp_path / f'seeds-{camera}.json'
        write_json_atomic(path, {'provenance': {'fixture': 'fake estimator'}})
        references[camera] = {'path': path.name, 'sha256': dual_sha256(path)}
    body = {'schema': 'video_court_build_v1', 'profile': PROFILE, 'dataset_directory': 'dataset', 'video_id': 'video_000',
            'candidate_manifests': references, 'seed_clip_ids': None, 'fit_clip_ids': None, 'bootstrap_clip_ids': None,
            'fit_numerators': [1, 2, 3], 'diagnostic_numerators': [4], 'denominator': 5, 'workers': 1}
    path = tmp_path / 'build.json'
    write_json_atomic(path, body)
    return VideoCourtConfig('build', tmp_path / 'built-shared', 'fixture', 'video_000', path, dual_sha256(path), None, False), body


def test_build_resume_load_identity_and_camera_specific_rebuild(tmp_path, monkeypatch):
    pipeline, paths, _, _, _, _ = shared_fixture(tmp_path, monkeypatch)
    config, body = make_build_request(tmp_path, paths)
    calls: Counter[str] = Counter()
    fail = [True]

    def fake_estimator(**kwargs):
        camera = kwargs['camera']
        calls[camera] += 1
        if camera == 'cam1' and fail[0]:
            raise RuntimeError('interrupted computation')
        return {'status': 'accepted', 'matrix': matrices()[CAMERAS.index(camera)].tolist(),
                'diagnostics': {'profile': PROFILE}, 'clips': {clip: report() for clip in kwargs['sources']}}

    monkeypatch.setattr('src.tennis_scene.pipeline.video_court.prepare.estimate_camera', fake_estimator)
    with pytest.raises(RuntimeError, match='interrupted'):
        prepare_video_court(config, pipeline.config.resolver, code_identity='test')
    fail[0] = False
    artifact = prepare_video_court(config, pipeline.config.resolver, code_identity='test')
    assert calls == {'cam0': 1, 'cam1': 2, 'cam2': 1}
    loaded = prepare_video_court(replace(config, action='load', request=None, request_sha256=None, artifact_digest=artifact.digest),
                                 pipeline.config.resolver, code_identity='test')
    assert loaded.digest == artifact.digest
    assert prepare_video_court(config, pipeline.config.resolver, code_identity='test').digest == artifact.digest
    assert calls == {'cam0': 1, 'cam1': 2, 'cam2': 1}
    path = tmp_path / body['candidate_manifests']['cam0']['path']
    write_json_atomic(path, {'provenance': {'fixture': 'changed fake candidate'}})
    body['candidate_manifests']['cam0']['sha256'] = dual_sha256(path)
    write_json_atomic(config.request, body)
    config = replace(config, request_sha256=dual_sha256(config.request))
    updated = prepare_video_court(config, pipeline.config.resolver, code_identity='test')
    assert calls == {'cam0': 2, 'cam1': 2, 'cam2': 1}
    assert updated.manifest['cameras']['cam0'] != artifact.manifest['cameras']['cam0']
    assert updated.manifest['cameras']['cam1'] == artifact.manifest['cameras']['cam1']
    damaged = config.directory / 'objects' / f'{updated.manifest["cameras"]["cam0"]}.json'
    damaged.write_text('{corrupt')
    before = calls.copy()
    with pytest.raises(ValueError):
        prepare_video_court(config, pipeline.config.resolver, code_identity='test')
    assert calls == before
    repaired = prepare_video_court(replace(config, rebuild=True), pipeline.config.resolver, code_identity='test')
    assert repaired.digest == updated.digest
    assert calls == {camera: before[camera] + 1 for camera in CAMERAS}


def test_build_input_change_prevents_camera_publication(tmp_path, monkeypatch):
    pipeline, paths, _, _, _, _ = shared_fixture(tmp_path, monkeypatch)
    config, _ = make_build_request(tmp_path, paths)

    def mutate(**kwargs):
        kwargs['paths'][CLIPS[0]].write_bytes(b'changed during RGB computation')
        return {'status': 'accepted', 'matrix': matrices()[0].tolist(), 'diagnostics': {'profile': PROFILE},
                'clips': {clip: report() for clip in kwargs['sources']}}

    monkeypatch.setattr('src.tennis_scene.pipeline.video_court.prepare.estimate_camera', mutate)
    with pytest.raises(ValueError, match='inputs changed'):
        prepare_video_court(config, pipeline.config.resolver, code_identity='test')
    assert not (config.directory / 'objects').exists()


def test_two_build_callers_share_the_lock_and_publish_one_result(tmp_path, monkeypatch):
    pipeline, paths, _, _, _, _ = shared_fixture(tmp_path, monkeypatch)
    config, _ = make_build_request(tmp_path, paths)
    started, second_attempt, release = Event(), Event(), Event()
    count_lock = Lock()
    attempts = [0]
    calls: Counter[str] = Counter()
    original_lock = VideoCourtStore.lock

    @contextmanager
    def observed_lock(store):
        with count_lock:
            attempts[0] += 1
            if attempts[0] == 2:
                second_attempt.set()
        with original_lock(store):
            yield

    def estimate(**kwargs):
        camera = kwargs['camera']
        calls[camera] += 1
        if camera == 'cam0' and calls[camera] == 1:
            started.set()
            assert release.wait(10)
        return {'status': 'accepted', 'matrix': matrices()[CAMERAS.index(camera)].tolist(),
                'diagnostics': {'profile': PROFILE}, 'clips': {clip: report() for clip in kwargs['sources']}}

    monkeypatch.setattr(VideoCourtStore, 'lock', observed_lock)
    monkeypatch.setattr('src.tennis_scene.pipeline.video_court.prepare.estimate_camera', estimate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(prepare_video_court, config, pipeline.config.resolver, code_identity='test')
        try:
            assert started.wait(10)
            second = pool.submit(prepare_video_court, config, pipeline.config.resolver, code_identity='test')
            assert second_attempt.wait(10)
        finally:
            release.set()
        assert first.result(timeout=10).digest == second.result(timeout=10).digest
    assert calls == {'cam0': 1, 'cam1': 1, 'cam2': 1}


@pytest.mark.parametrize('selection', [[], [CLIPS[0], CLIPS[0]], ['other/clip_a']])
def test_invalid_build_subset_stops_before_estimation(tmp_path, monkeypatch, selection):
    pipeline, paths, _, _, _, _ = shared_fixture(tmp_path, monkeypatch)
    config, body = make_build_request(tmp_path, paths)
    body['fit_clip_ids'] = selection
    write_json_atomic(config.request, body)
    config = replace(config, request_sha256=dual_sha256(config.request))
    with pytest.raises(ValueError, match='empty/duplicate/foreign/unknown'):
        prepare_video_court(config, pipeline.config.resolver, code_identity='test')


def test_shared_artifact_change_is_part_of_dataset_marker_identity(tmp_path, monkeypatch):
    pipeline, paths, _, docs, store, artifact = shared_fixture(tmp_path, monkeypatch)
    materialize_assets(enabled_model_assets(pipeline.config).values(), tmp_path)
    _, _ = make_build_request(tmp_path, paths)
    record = select_dataset_clips((tmp_path / 'dataset').resolve())[0]
    bound = resolve_dataset_clip((tmp_path / 'dataset').resolve(), record)
    identity = pipeline.publication_identity()
    result = pipeline.run(bound.video_paths, video_role=PathRole.DATA, camera_ids=CAMERAS,
                          store_root=bound.clip_dir / 'annotations/tennis_scene', clip_id=record.clip_id)
    destination = publish_annotation(bound, result, pipeline_config_yaml='fixture: video_court\n',
                                     publication_identity=identity)
    assert completed_annotation(bound, identity) == destination
    docs['cam0']['matrix'][0][2] += 1
    with store.lock():
        changed = store.publish(docs)
    pipeline.config = replace(pipeline.config, video_court=replace(pipeline.config.video_court, artifact_digest=changed.digest))
    with pytest.raises(ValueError, match='Stale annotation'):
        completed_annotation(bound, pipeline.publication_identity())


@pytest.mark.parametrize('overrides', [
    ('court_kp.mode=unknown',),
    ('court_kp.mode=video_shared',),
    ('court_kp.video_shared.action=load',),
    ('court_kp.mode=video_shared', 'court_kp.video_shared.action=load', 'court_kp.video_shared.directory=shared',
     'court_kp.video_shared.dataset_id=fixture', 'court_kp.video_shared.video_id=video_000',
     'court_kp.video_shared.artifact_digest=' + 'a' * 64, 'ball_detection.method=gpt_annotation'),
])
def test_invalid_mode_acquisition_and_annotation_combination_stop_at_configuration(tmp_path, overrides):
    with pytest.raises((ValueError, TypeError)):
        runtime(tmp_path, overrides=overrides)


def independent_drift_reports(base):
    from src.tasks.court_detection.geometry.video_drift import summarize_clip
    return {camera: {clip: summarize_clip([{'schema': 'signed_line_similarity_v1', 'frame_index': 3,
        'status': 'ok', 'reason': None, 'temporal_pooling': False, 'rgb_sha256': 'b' * 64,
        'mean_displacement_px': 0., 'max_displacement_px': 0., 'max_visible_projection_std_px': .8}])
        for clip in doc['clips']} for camera, doc in base.cameras.items()}


def test_separate_drift_changes_consumers_and_marker_without_mutating_shared_h(tmp_path, monkeypatch):
    from copy import deepcopy
    from dataclasses import asdict

    from src.tasks.court_detection.geometry.video_drift import (
        DriftConfig,
        summarize_clip,
    )
    from src.tennis_scene.pipeline.video_court.diagnostics import publish_video_drift

    pipeline, paths, sources, _, store, base = shared_fixture(tmp_path, monkeypatch)
    materialize_assets(enabled_model_assets(pipeline.config).values(), tmp_path)
    before = {p: p.read_bytes() for p in (store.root / 'objects').glob('*.json')}
    protocol = {'settings': asdict(DriftConfig())}
    reports = independent_drift_reports(base)
    diagnosed = publish_video_drift(base, tmp_path / 'drift', protocol=protocol, reports=reports)
    pipeline.config = replace(pipeline.config, video_court_drift=diagnosed.drift)
    root = tmp_path / 'clip'
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    first_identity = pipeline.publication_identity()
    original_kp = pipeline.last_runner.output('court_detection/cam0').keypoints.copy()
    changed = deepcopy(reports)
    frame = changed['cam0'][CLIPS[0]]['frames'][0]
    frame.update(status='warning', reason='camera_moved_possible', mean_displacement_px=5., max_displacement_px=5.)
    changed['cam0'][CLIPS[0]] = summarize_clip([frame])
    latest = publish_video_drift(base, tmp_path / 'drift', protocol=protocol, reports=changed)
    assert latest.digest == base.digest and latest.manifest == base.manifest
    assert latest.drift_manifest['cameras']['cam1'] == diagnosed.drift_manifest['cameras']['cam1']
    pipeline.config = replace(pipeline.config, video_court_drift=latest.drift)
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    np.testing.assert_array_equal(pipeline.last_runner.output('court_detection/cam0').keypoints, original_kp)
    statuses = pipeline.last_runner.statuses
    assert statuses['court_detection/cam0'] == statuses['court_calibration'] == 'executed'
    assert statuses['court_detection/cam1'] == statuses['person_detection/cam0'] == statuses['ball_detection/cam0'] == 'cached'
    assert pipeline.publication_identity() != first_identity
    receipt = json.loads((root / 'run.json').read_text())['video_court']
    assert receipt['artifact_digest'] == base.digest
    assert receipt['drift_artifact']['artifact_digest'] == latest.drift.digest
    assert receipt['clip_diagnostics']['cam0']['status'] == 'warning'
    assert before == {p: p.read_bytes() for p in (store.root / 'objects').glob('*.json')}
    # Multi-clip planning must consume the same explicitly selected diagnostic.
    for source in sources:
        standard_definition(pipeline.config, source, code_identity='fixed', video_court=latest)
    with pytest.raises(ValueError, match='drift artifact'):
        standard_definition(pipeline.config, sources[0], code_identity='fixed', video_court=base)


def test_explicit_fail_policy_rejects_unobservable_even_if_embedded_report_was_ok(tmp_path, monkeypatch):
    from dataclasses import asdict

    from src.tasks.court_detection.geometry.video_drift import (
        DriftConfig,
        summarize_clip,
    )
    from src.tennis_scene.pipeline.video_court.diagnostics import publish_video_drift

    pipeline, paths, _, _, _, base = shared_fixture(tmp_path, monkeypatch)
    reports = independent_drift_reports(base)
    frame = reports['cam0'][CLIPS[0]]['frames'][0]
    frame.update(status='failed', reason='drift_unobservable')
    reports['cam0'][CLIPS[0]] = summarize_clip([frame])
    artifact = publish_video_drift(base, tmp_path / 'drift', protocol={'settings': asdict(DriftConfig())}, reports=reports)
    pipeline.config = replace(pipeline.config, video_court_drift=artifact.drift, video_court_unobservable_policy='fail')
    with pytest.raises(ReconstructionUnavailable, match='cam0'):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=tmp_path / 'clip', clip_id=CLIPS[0])
    assert pipeline.last_receipt['error_diagnostics']['drift_digest'] == artifact.drift_manifest['cameras']['cam0']


def test_corrupt_or_wrong_parent_drift_never_reverts_to_embedded_reports(tmp_path, monkeypatch):
    from dataclasses import asdict

    from src.tasks.court_detection.geometry.video_drift import DriftConfig
    from src.tennis_scene.pipeline.video_court.diagnostics import (
        load_video_drift,
        publish_video_drift,
    )

    pipeline, paths, _, docs, store, base = shared_fixture(tmp_path, monkeypatch)
    artifact = publish_video_drift(base, tmp_path / 'drift', protocol={'settings': asdict(DriftConfig())}, reports=independent_drift_reports(base))
    pipeline.config = replace(pipeline.config, video_court_drift=artifact.drift)
    root = tmp_path / 'clip'
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    docs['cam0']['matrix'][0][2] += 1
    changed = store.publish(docs)
    with pytest.raises(ValueError, match='match shared H'):
        load_video_drift(changed, artifact.drift)
    path = artifact.drift.directory / 'objects' / f'{artifact.drift_manifest["cameras"]["cam0"]}.json'
    value = json.loads(path.read_text())
    value['clips'][CLIPS[0]]['frames'][0]['mean_displacement_px'] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='content changed'):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    assert pipeline.last_receipt['status'] == 'failed'


def test_drift_preparation_uses_exact_plan_and_preserves_common_artifact(tmp_path, monkeypatch):
    from dataclasses import asdict

    from src.tasks.court_detection.geometry.video_drift import DriftConfig
    from src.tennis_scene.pipeline.video_court import drift_build

    _, _, sources, _, _, base = shared_fixture(tmp_path, monkeypatch)
    calls = []
    template = independent_drift_reports(base)['cam0'][CLIPS[0]]['frames'][0]

    def measure(job):
        path, frame, camera, config = job
        calls.append((path, frame, camera['camera_id']))
        return {**template, 'frame_index': frame}

    monkeypatch.setattr(drift_build, '_measure', measure)
    plan = {camera: {clip: [3, 7] for clip in CLIPS} for camera in CAMERAS}
    before = base.digest
    result = drift_build.build_video_drift(base, tuple(sources), frame_plan=plan, directory=tmp_path / 'drift',
                                           provenance={'unit': 'exact frame selection'}, config=DriftConfig())
    assert len(calls) == 12
    assert result.digest == before
    assert result.drift_cameras['cam0']['protocol']['settings'] == asdict(DriftConfig())
    assert [f['frame_index'] for f in result.drift_cameras['cam0']['clips'][CLIPS[0]]['frames']] == [3, 7]
    plan['cam0'][CLIPS[0]] = [3, 3]
    with pytest.raises(ValueError, match='invalid frame'):
        drift_build.build_video_drift(base, tuple(sources), frame_plan=plan, directory=tmp_path / 'drift', provenance={}, config=DriftConfig())
    assert len(calls) == 12


@pytest.mark.parametrize('overrides', [
    ('court_kp.video_shared.drift_directory=drift',),
    ('court_kp.video_shared.drift_directory=drift', 'court_kp.video_shared.drift_digest=' + 'b' * 64),
    ('court_kp.mode=video_shared', 'court_kp.video_shared.action=load', 'court_kp.video_shared.directory=shared',
     'court_kp.video_shared.dataset_id=fixture', 'court_kp.video_shared.video_id=video_000',
     'court_kp.video_shared.artifact_digest=' + 'a' * 64, 'court_kp.video_shared.drift_digest=' + 'b' * 64),
])
def test_drift_reference_must_be_complete_and_cannot_activate_clip_mode(tmp_path, overrides):
    with pytest.raises(ValueError):
        runtime(tmp_path, overrides=overrides)


def test_drift_reader_hashes_rgb_not_opencv_bgr(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from src.tasks.court_detection.geometry.video_drift import DriftConfig
    from src.tennis_scene.pipeline.video_court import drift_build

    bgr = np.array([[[10, 70, 190], [5, 90, 160]], [[25, 95, 220], [0, 45, 150]]], dtype=np.uint8)
    rgb = bgr[..., ::-1].copy()
    monkeypatch.setattr(drift_build, 'read_video_frame', lambda *args: SimpleNamespace(frame=bgr, original_size=(2, 2)))

    def inspect(image, *args):
        np.testing.assert_array_equal(image, rgb)
        return {'status': 'failed', 'reason': 'drift_unobservable'}

    monkeypatch.setattr(drift_build, 'inspect_frame', inspect)
    report = drift_build._measure((tmp_path / 'clip.mp4', 7,
        {'size_wh': [2, 2], 'matrix': np.eye(3).tolist(), 'diagnostics': {'quality': {'supported_lines': ['near_baseline']}}}, DriftConfig()))
    assert report['rgb_sha256'] == hashlib.sha256(rgb.tobytes()).hexdigest()
    assert report['rgb_sha256'] != hashlib.sha256(bgr.tobytes()).hexdigest()


def test_runtime_configuration_loads_both_pinned_court_and_drift_references(tmp_path):
    cfg = runtime(tmp_path, overrides=(
        'court_kp.mode=video_shared', 'court_kp.video_shared.action=load',
        'court_kp.video_shared.directory=shared', 'court_kp.video_shared.dataset_id=fixture',
        'court_kp.video_shared.video_id=video_000', 'court_kp.video_shared.artifact_digest=' + 'a' * 64,
        'court_kp.video_shared.drift_directory=drift', 'court_kp.video_shared.drift_digest=' + 'b' * 64))
    assert cfg.video_court is not None and cfg.video_court_drift is not None
    assert cfg.video_court.artifact_digest == 'a' * 64
    assert cfg.video_court_drift.digest == 'b' * 64
    assert cfg.video_court_drift.directory.name == 'drift'
    assert cfg.video_court_unobservable_policy == 'warn'
    assert cfg.processing_settings['court_kp'] == {'mode': 'video_shared', 'unobservable_policy': 'warn'}


def test_warn_policy_uses_h_and_keeps_unobservable_frames_in_receipt_and_gallery(tmp_path, monkeypatch):
    from dataclasses import asdict

    from src.tasks.court_detection.geometry.video_drift import (
        DriftConfig,
        summarize_clip,
    )
    from src.tennis_scene.pipeline.video_court.diagnostics import publish_video_drift
    from src.tennis_scene.visualization.review import gallery

    pipeline, paths, _, _, store, base = shared_fixture(tmp_path, monkeypatch)
    materialize_assets(enabled_model_assets(pipeline.config).values(), tmp_path)
    snapshots = {p: p.read_bytes() for p in (store.root / 'objects').glob('*.json')}
    reports = independent_drift_reports(base)
    frame = reports['cam0'][CLIPS[0]]['frames'][0]
    frame.update(status='failed', reason='drift_unobservable', geometry_reasons=['non_similarity_or_conflicting_line'],
                 mean_displacement_px=None, max_displacement_px=None, max_visible_projection_std_px=None)
    reports['cam0'][CLIPS[0]] = summarize_clip([frame])
    artifact = publish_video_drift(base, tmp_path / 'drift', protocol={'settings': asdict(DriftConfig())}, reports=reports)
    pipeline.config = replace(pipeline.config, video_court_drift=artifact.drift)
    root = tmp_path / 'clip'
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    warn_identity = pipeline.publication_identity()
    receipt = pipeline.last_receipt['video_court']
    output = pipeline.last_runner.output('court_detection/cam0')
    assert receipt['unobservable_policy'] == 'warn'
    assert receipt['clip_diagnostics']['cam0'] == output.diagnostics['drift']
    assert output.diagnostics['drift']['status'] == 'warning'
    assert output.diagnostics['drift']['frames'][0]['measurement_status'] == 'failed'
    assert output.diagnostics['drift']['frames'][0]['frame_index'] == 3
    assert output.diagnostics['drift']['frames'][0]['mean_displacement_px'] is None
    assert artifact.drift_cameras['cam0']['clips'][CLIPS[0]]['status'] == 'failed'
    assert snapshots == {p: p.read_bytes() for p in (store.root / 'objects').glob('*.json')}
    monkeypatch.setattr(gallery.Review, 'frame', lambda *args: np.zeros((720, 1280, 3), np.uint8))
    review = gallery.Review(root / 'scene.json', tmp_path / 'review')
    payload, _ = review.payload('court_detection/cam0')
    _, details = review.render_court_detection('court_detection/cam0', 'cam0', payload)
    assert dict(details)['drift'] == 'warning' and dict(details)['unobservable policy'] == 'warn'
    assert 'f3: drift_unobservable' in dict(details)['diagnostic frames']
    pipeline.config = replace(pipeline.config, video_court_unobservable_policy='fail')
    assert pipeline.publication_identity() != warn_identity
    # The completed warn cache must not bypass the explicit strict policy.
    with pytest.raises(ReconstructionUnavailable, match='cam0'):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    assert pipeline.last_receipt['video_court']['clip_diagnostics']['cam0']['status'] == 'failed'


def test_policy_change_invalidates_court_descendants_but_reuses_independent_roots(tmp_path, monkeypatch):
    pipeline, paths, _, _, _, artifact = shared_fixture(tmp_path, monkeypatch)
    root = tmp_path / 'clip'
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    original = pipeline.last_runner.output('court_detection/cam0').keypoints.copy()
    pipeline.config = replace(pipeline.config, video_court_unobservable_policy='fail')
    pipeline.run(paths, video_role=PathRole.DATA, camera_ids=CAMERAS, store_root=root, clip_id=CLIPS[0])
    assert all(pipeline.last_runner.statuses[f'court_detection/{camera}'] == 'executed' for camera in CAMERAS)
    assert pipeline.last_runner.statuses['court_calibration'] == pipeline.last_runner.statuses['scene_assembly'] == 'executed'
    assert pipeline.last_runner.statuses['person_detection/cam0'] == pipeline.last_runner.statuses['ball_detection/cam0'] == 'cached'
    np.testing.assert_array_equal(pipeline.last_runner.output('court_detection/cam0').keypoints, original)
    assert pipeline.last_receipt['video_court']['artifact_digest'] == artifact.digest


@pytest.mark.parametrize('policy', ['ignore', 'null'])
def test_unknown_unobservable_policy_is_rejected_by_configuration(tmp_path, policy):
    with pytest.raises((ValueError, TypeError)):
        runtime(tmp_path, overrides=(f'court_kp.video_shared.unobservable_policy={policy}',))
