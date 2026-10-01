"""One reserved unseen inference batch; labels and scoring belong to the next run.

The immutable person manifest must already be pushed before plan opens metadata
or media. This entrypoint never fits, tunes, retries, or executes court_side.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from association_recalibration_dev import project_ids  # type: ignore[import-not-found]
from person_unseen_freeze import (  # type: ignore[import-not-found]
    CAMERAS,
    CLIPS,
    CODE,
    require_pushed,
    verify,
)
from person_unseen_resume import (  # type: ignore[import-not-found]
    load_addendum,
    load_preparation_addendum,
    require_unstarted_directory,
)
from person_unseen_video import render  # type: ignore[import-not-found]

from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.feature_tracks import evidence_appearance
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.camera_geometry import calibrate_local_courts
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import file_identity, standard_definition
from src.tennis_scene.pipeline.runner import ComponentNode, ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera

PERSON_STAGES = frozenset(('person_detection', 'person_tracking'))
STAGES = PERSON_STAGES | {'court_detection'}


def checked(record: dict[str, Any]) -> Path:
    path = Path(record['path'])
    if file_identity(path) != {k: record[k] for k in ('path', 'sha256')}:
        raise ValueError(f'Pinned input changed: {path}')
    return path


def claim(path: Path, receipt: dict[str, Any]) -> None:
    """Atomic and durable even on failure; a second attempt cannot reuse the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')


def clip_config(frozen: dict[str, Any], clip: Path) -> Any:
    from omegaconf import OmegaConf
    manifest = ClipManifest.load(clip)
    cfg = OmegaConf.create(frozen['pipeline_config'])
    cfg.video_paths = [str(manifest.media_path(c).relative_to(Path(cfg.paths.data_root))) for c in CAMERAS]
    cfg.camera_ids = list(CAMERAS)
    return cfg


def selected_sides(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for clip in CLIPS:
        matches = [row for row in document['clips'] if row['clip_id'] == clip]
        if len(matches) != 1:
            raise ValueError(f'Missing/duplicate side reference: {clip}')
        row = matches[0]
        if 'observe_failed' in row:
            result[clip] = {'status': 'stopped', 'reason': 'annotation_side_missing_after_court_failure',
                            'view_half_turns': None, 'observe_failed': row['observe_failed']}
            continue
        if row['camera_ids'] != list(CAMERAS) or not row['annotation']['decided']:
            raise ValueError(f'Annotation-ball side is unavailable: {clip}')
        turns = row['annotation']['view_half_turns']
        if len(turns) != 3 or any(type(x) is not bool for x in turns):
            raise ValueError('Invalid reference half-turns')
        result[clip] = {'status': 'ok', 'view_half_turns': turns}
    return result


def plan(freeze: Path, commit: str, report: Path, *, addendum: Path | None = None,
         addendum_commit: str | None = None, preparation_addendum: Path | None = None,
         preparation_commit: str | None = None) -> None:
    frozen = require_pushed(freeze, commit)
    if report != Path(frozen['report']):
        raise ValueError('Use the one frozen output directory')
    verify(frozen)
    if (addendum is None) != (addendum_commit is None):
        raise ValueError('Resumed preparation requires both addendum and its pushed commit')
    if (preparation_addendum is None) != (preparation_commit is None) \
            or (preparation_addendum is not None and addendum is None):
        raise ValueError('Preparation continuation requires the original and new pushed addenda')
    execution_addendum = None
    preparation_record = None
    budget = frozen['budget']
    opening_name = 'opening.json'
    if addendum is not None and addendum_commit is not None:
        resumed = load_addendum(addendum, addendum_commit, freeze, commit, frozen)
        execution_addendum = {'file': file_identity(addendum), 'commit': addendum_commit}
        budget = resumed['budget']
        opening_name = 'resumed-opening-r17.json'
        preparation = None
        if preparation_addendum is not None and preparation_commit is not None:
            preparation = load_preparation_addendum(preparation_addendum, preparation_commit,
                                                   file_identity(addendum), report, commit)
            preparation_record = {'file': file_identity(preparation_addendum), 'commit': preparation_commit}
            opening_name = 'resumed-opening-r17b.json'
        require_unstarted_directory(report, preparation)
    else:
        # Existing or failed preparation must be investigated, never silently replaced.
        report.mkdir(parents=True, exist_ok=False)
    claim(report / opening_name, {'freeze': file_identity(freeze), 'freeze_commit': commit,
        'execution_addendum': execution_addendum,
        'preparation_addendum': preparation_record,
        'time_unix': time.time(), 'event': 'pre-open gate passed; metadata/hash preparation begins',
        'inference_attempts': 0, 'scoring_batches': 0})
    repo = Path(frozen['repo'])
    dataset = repo / 'data/tennis_multivew/processed/meiji_3cam/dataset'
    reservation = json.loads(checked(frozen['reservation']).read_text())
    side_path = repo / 'outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json'
    sides = selected_sides(json.loads(side_path.read_text()))
    missing = [c for c in CLIPS if sides[c]['view_half_turns'] is None]
    if missing and (execution_addendum is None or missing != resumed['allowed_missing_sides']):
        raise ValueError('Missing side requires the explicit pushed execution addendum')
    records = []
    from omegaconf import OmegaConf
    for clip in CLIPS:
        video_id, clip_id = clip.split('/')
        clip_root = dataset / 'videos' / video_id / 'clips' / clip_id
        metadata_path = clip_root / 'clip.json'
        metadata = json.loads(metadata_path.read_text())
        reserved = reservation['clips'][clip]
        if dual_sha256(metadata_path) != reserved['manifest_sha256']:
            raise ValueError('Reserved clip metadata changed')
        config = clip_config(frozen, clip_root)
        runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=True)
        source = build_clip_source(runtime.video_paths, runtime.camera_ids, clip_id=clip)
        if source.camera_ids != CAMERAS or source.num_frames != reserved['num_frames'] \
                or not np.isclose(source.fps, reserved['fps'], rtol=0., atol=1e-6) or source.num_frames != metadata['num_frames']:
            raise ValueError('Reserved camera/timeline changed')
        for camera in CAMERAS:
            if file_identity(source.video(camera).path) != reserved['videos'][camera]:
                raise ValueError('Reserved media changed')
        config_path = report / clip / 'pipeline.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(OmegaConf.to_yaml(config, resolve=True))
        nodes = [n for n in standard_definition(runtime, source, code_identity=dual_sha256(freeze))
                 if n.name.split('/')[0] in STAGES]
        runner = ComponentRunner(nodes, ClipStore(report / clip / 'store', json_value(source), memory_entries=0))
        if len(runner.order) != 9 or any(n.source != 'execute' for n in nodes):
            raise ValueError('Unseen inference requires exactly nine fresh court/person nodes')
        records.append({'clip': clip, 'metadata': file_identity(metadata_path), 'config': file_identity(config_path),
                        'source': json_value(source), 'side': sides[clip], 'view_half_turns': sides[clip]['view_half_turns'],
                        'node_order': list(runner.order),
                        'node_settings': {n.name: dict(n.settings) for n in nodes},
                        'labels_exist': (clip_root / 'annotations/player_association/labels.json').exists()})
    receipt = {'schema': 'i964_unseen_plan_v2', 'freeze': file_identity(freeze), 'freeze_commit': commit,
               'execution_addendum': execution_addendum,
               'preparation_addendum': preparation_record,
               'side_reference': file_identity(side_path), 'records': records, 'budget': budget,
               'camera_frames': sum(r['source']['videos'][0]['num_frames'] * 3 for r in records),
               'labels_opened': False, 'scoring_batches': 0,
               'entrypoints': [file_identity(CODE / 'tests/benchmarks' / name) for name in
                   ('person_unseen.py', 'person_unseen_video.py', 'person_unseen_score.py', 'person_unseen.sh',
                    'person_unseen_resume.py', 'association_feature_guard.py', 'association_recalibration_dev.py',
                    'build_dino_extension.sh')]}
    write_json_atomic(report / 'plan.json', receipt)
    print(json.dumps({'frames': receipt['camera_frames'],
                      'clips': [{k: r[k] for k in ('clip', 'view_half_turns', 'labels_exist')} for r in records]}))


def audit_tracks(runner: ComponentRunner, source: ClipSource) -> dict[str, Any]:
    audit = {}
    for camera in CAMERAS:
        det = runner.output(f'person_detection/{camera}')
        raw = runner.output(f'person_tracking/{camera}')
        if det.source_rows is None or raw.evidence is None or raw.reconstruction is None \
                or len(det.frame_offsets) != source.num_frames + 1 or raw.observed.shape[1] != source.num_frames \
                or det.duplicate_merges or (det.confidence < np.float32(.3)).any():
            raise ValueError('Incomplete or non-frozen person output')
        if (raw.reconstruction.interpolated & raw.observed).any():
            raise ValueError('GSI was counted as observed')
        for frame in range(source.num_frames):
            a, b = det.frame_offsets[frame:frame+2]
            rows = raw.evidence.detection_rows[:, frame][raw.observed[:, frame]]
            indices = np.searchsorted(det.source_rows[a:b], rows)
            if (indices >= b-a).any() or not np.array_equal(det.source_rows[a:b][indices], rows):
                raise ValueError('Tracking lost detection-row provenance')
            np.testing.assert_array_equal(raw.boxes_xyxy[:, frame][raw.observed[:, frame]], det.boxes_xyxy[a:b][indices])
        audit[camera] = {'detections': len(det.confidence), 'raw_tracks': len(raw.track_ids),
                         'real_observations': int(raw.observed.sum()),
                         'synthetic_gsi': int(raw.reconstruction.interpolated.sum())}
    return audit


def predict(runner: ComponentRunner, source: ClipSource, cfg: PipelineRuntimeConfig,
            side: dict[str, Any], cameras: dict[str, PinholeCamera], target: Path,
            ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Same raw appearance / linked-group association adapter as the frozen dev batch."""
    turns = side['view_half_turns']
    raw_cameras, groups, mappings = [], [], []
    arrays: dict[str, np.ndarray] = {}
    selection_audit = {}
    scale = source.pixel_threshold_scale
    config = replace(cfg.player_association, footpoints=replace(cfg.player_association.footpoints,
                      bottom_border_px=cfg.player_association.footpoints.bottom_border_px * scale))
    sampling = CropSamplingConfig()
    sampling = replace(sampling, min_height_px=sampling.min_height_px*scale, border_px=sampling.border_px*scale)
    for index, camera in enumerate(CAMERAS):
        tracked = runner.output(f'person_tracking/{camera}')
        if tracked.evidence is None:
            raise ValueError('Unseen association requires original tracked CLIP features')
        arrays.update({f'{camera}_boxes': tracked.boxes_xyxy, f'{camera}_observed': tracked.observed,
                       f'{camera}_track_ids': tracked.track_ids,
                       f'{camera}_ids': np.full(tracked.observed.shape, -1, np.int64)})
        if camera not in cameras:
            # Same explicit empty selection as PlayerSelectionModule for an uncalibrated view.
            # Raw detections/tracks remain available for review and the three-camera video.
            arrays.update({f'{camera}_selected': np.zeros_like(tracked.observed),
                           f'{camera}_group_boxes': np.empty((0, source.num_frames, 4), np.float32),
                           f'{camera}_group_observed': np.zeros((0, source.num_frames), bool),
                           f'{camera}_group_track_ids': np.empty(0, np.int64),
                           f'{camera}_group_origins': np.empty((0, source.num_frames), np.int64),
                           f'{camera}_group_ids': np.empty((0, source.num_frames), np.int64)})
            selection_audit[camera] = {'status': 'stopped', 'reason': 'camera_not_calibrated'}
            continue
        appearance = evidence_appearance(tracked.boxes_xyxy, tracked.observed, tracked.evidence, source.size, sampling)
        # Selection is camera-local and half-turn invariant. An absent side is never
        # invented: the unturned local camera may select, but cannot enter association.
        local_camera = cameras[camera] if turns is None else cameras[camera].half_turned(turns[index])
        raw = CameraTracks(local_camera, source.size, tracked.track_ids,
                           tracked.boxes_xyxy, tracked.observed, appearance)
        mask, selection = select_linked_candidates(raw, source.fps, LinkingConfig(max_candidates=cfg.max_tracks_per_camera),
                                                   config.footpoints)
        group, mapping = linked_timeline(raw, selection)
        raw_cameras.append(raw)
        groups.append(group)
        mappings.append(mapping)
        selection_audit[camera] = {'status': 'ok', 'linking': selection,
                                   'orientation': 'camera_local_without_side' if turns is None else 'annotation_side'}
        for name, values in {'boxes': raw.boxes_xyxy, 'observed': raw.observed, 'track_ids': raw.track_ids,
                             'selected': mask, 'group_boxes': group.boxes_xyxy, 'group_observed': group.observed,
                             'group_track_ids': group.track_ids, 'group_origins': mapping,
                             'group_ids': np.full(group.observed.shape, -1, np.int64)}.items():
            arrays[f'{camera}_{name}'] = values
    result: dict[str, Any] = {'status': 'ok', 'clip': source.clip_id, 'view_half_turns': turns,
                              'labels_used': False, 'selection': selection_audit, 'side': side}
    if turns is None:
        result.update(status='stopped', reason=side['reason'])
    elif set(cameras) != set(CAMERAS):
        result.update(status='stopped', reason='court_calibration_unavailable',
                      missing_cameras=sorted(set(CAMERAS) - set(cameras)))
    else:
        try:
            association = associate(groups, source.fps, config)
        except AssociationUndecided as error:
            result.update(status='undecided', reason=error.reason, diagnostics=error.diagnostics)
        else:
            result['diagnostics'] = association.diagnostics
            for camera, raw, mapping, assigned in zip(CAMERAS, raw_cameras, mappings, association.player_ids, strict=True):
                arrays[f'{camera}_ids'] = project_ids(raw, mapping, assigned)
                arrays[f'{camera}_group_ids'] = assigned
    with target.open('xb') as stream:
        saved_arrays: dict[str, Any] = dict(arrays)
        np.savez_compressed(stream, **saved_arrays)
    with np.load(target, allow_pickle=False) as restored:
        for key, value in arrays.items():
            np.testing.assert_array_equal(restored[key], value)
    result['arrays'] = file_identity(target)
    return result, arrays


def run_person_and_court(nodes: list[ComponentNode], store: ClipStore, source: ClipSource,
                         cfg: PipelineRuntimeConfig, root: Path,
                         ) -> tuple[ComponentRunner, dict[str, PinholeCamera], dict[str, Any]]:
    """Run all person cameras before independent court observations; never repair geometry."""
    person = ComponentRunner([n for n in nodes if n.name.split('/')[0] in PERSON_STAGES], store)
    try:
        person.run()
    finally:
        write_json_atomic(root / 'person-execute.json', {'statuses': person.statuses, 'seconds': person.seconds,
                          'active_node': person.active_node, 'references': json_value(person.references)})
    if len(person.statuses) != 6 or set(person.statuses.values()) != {'executed'}:
        raise ValueError('All six person nodes must execute once, without cache substitution')
    audit = audit_tracks(person, source)
    write_json_atomic(root / 'person-audit.json', audit)
    cameras = {}
    court_receipt: dict[str, Any] = {}
    for camera in CAMERAS:
        court = ComponentRunner([n for n in nodes if n.name == f'court_detection/{camera}'], store)
        try:
            court.run()
        except ValueError as error:
            # Only the existing, explicit no-supported-court outcome is recoverable.
            # Checkpoint/schema/IO errors still fail the job, with person artifacts intact.
            if not str(error).startswith('No Court region meets model support/geometry requirements:'):
                raise
            court_receipt[camera] = {'status': 'stopped', 'reason': 'court_detection_unavailable',
                                     'error': str(error), 'statuses': court.statuses}
        else:
            if list(court.statuses.values()) != ['executed']:
                raise ValueError('Each court detector must execute once')
            calibration = calibrate_local_courts(court.output(f'court_detection/{camera}'), (camera,),
                                                size=source.size, config=cfg.camera_geometry)
            cameras.update({v.camera.camera_id: v.camera for v in calibration.views})
            court_receipt[camera] = {'status': 'ok' if calibration.views else 'stopped',
                                     'calibration': json_value(calibration), 'statuses': court.statuses,
                                     'references': json_value(court.references), 'seconds': court.seconds}
        write_json_atomic(root / 'court-execute.json', court_receipt)
    return person, cameras, audit


def execute(report: Path) -> None:
    from omegaconf import OmegaConf
    plan_doc = json.loads((report / 'plan.json').read_text())
    frozen = require_pushed(checked(plan_doc['freeze']), plan_doc['freeze_commit'])
    if report != Path(frozen['report']) or plan_doc['schema'] != 'i964_unseen_plan_v2':
        raise ValueError('Unexpected unseen plan')
    verify(frozen)
    expected_budget = frozen['budget']
    if plan_doc['execution_addendum'] is not None:
        resumed = plan_doc['execution_addendum']
        expected_budget = load_addendum(checked(resumed['file']), resumed['commit'], checked(plan_doc['freeze']),
                                       plan_doc['freeze_commit'], frozen)['budget']
    if plan_doc['budget'] != expected_budget:
        raise ValueError('Execution budget differs from the pushed authorization')
    if plan_doc['preparation_addendum'] is not None:
        preparation = plan_doc['preparation_addendum']
        load_preparation_addendum(checked(preparation['file']), preparation['commit'],
                                 plan_doc['execution_addendum']['file'], report, plan_doc['freeze_commit'])
    for record in (*plan_doc['entrypoints'], plan_doc['side_reference']):
        checked(record)
    claim(report / 'attempt.json', {'time_unix': time.time(), 'plan': file_identity(report / 'plan.json'),
                                    'inference_attempt': 1, 'scoring_batches': 0, 'retry_allowed': False})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(plan_doc['budget']['allocator_bytes'] / total, 0)
    progress: dict[str, Any] = {'status': 'running', 'clips': {}, 'scoring_batches': 0}
    start = time.monotonic()
    try:
        for record in plan_doc['records']:
            clip = record['clip']
            checked(record['metadata'])
            cfg = PipelineRuntimeConfig.from_config(OmegaConf.load(checked(record['config'])), bind_inputs=True)
            source = build_clip_source(cfg.video_paths, cfg.camera_ids, clip_id=clip)
            if json_value(source) != record['source']:
                raise ValueError('Source changed after plan')
            root = report / clip
            nodes = [n for n in standard_definition(cfg, source, code_identity=dual_sha256(checked(plan_doc['freeze'])))
                     if n.name.split('/')[0] in STAGES]
            if {n.name: dict(n.settings) for n in nodes} != record['node_settings']:
                raise ValueError('Component settings changed')
            clip_start = time.monotonic()
            runner, cameras, audit = run_person_and_court(
                nodes, ClipStore(root / 'store', json_value(source), memory_entries=0), source, cfg, root)
            prediction, arrays = predict(runner, source, cfg, record['side'], cameras, root / 'predictions.npz')
            write_json_atomic(root / 'prediction.json', prediction)
            video = render(source, arrays, root / 'three_camera_full.mp4', prediction['status'])
            progress['clips'][clip] = {'audit': audit, 'prediction': file_identity(root / 'prediction.json'),
                                      'person_execute': file_identity(root / 'person-execute.json'),
                                      'court_execute': file_identity(root / 'court-execute.json'),
                                      'association_status': prediction['status'], 'video': video,
                                      'seconds': time.monotonic() - clip_start}
            progress['elapsed_seconds'] = time.monotonic() - start
            write_json_atomic(report / 'progress.json', progress)
            print(f'{clip}: inference/association/video completed; no scoring', flush=True)
            del runner, arrays
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json_atomic(report / 'progress.json', progress)
        raise
    progress.update(status='ok', peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json_atomic(report / 'complete.json', progress)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('plan', 'execute'), required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--freeze-commit')
    parser.add_argument('--addendum', type=Path)
    parser.add_argument('--addendum-commit')
    parser.add_argument('--preparation-addendum', type=Path)
    parser.add_argument('--preparation-commit')
    args = parser.parse_args()
    if args.phase == 'plan':
        if args.freeze is None or args.freeze_commit is None:
            parser.error('plan requires --freeze and --freeze-commit')
        plan(args.freeze.resolve(), args.freeze_commit, args.report.resolve(),
             addendum=None if args.addendum is None else args.addendum.resolve(), addendum_commit=args.addendum_commit,
             preparation_addendum=None if args.preparation_addendum is None else args.preparation_addendum.resolve(),
             preparation_commit=args.preparation_commit)
    else:
        execute(args.report.resolve())
