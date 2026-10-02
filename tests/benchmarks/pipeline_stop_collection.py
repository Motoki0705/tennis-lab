"""Read-only CPU collection of the clip_000 qualification stopped at court_side.

This does not resume the runner, publish artifacts, open labels or claim scene
qualification. The output directory must be new and outside the source store.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionOutput
from src.tennis_scene.pipeline.definition import file_identity
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256

OUTPUT_TYPES: dict[str, type[Any]] = {
    'court_detection': CourtKPModule.io.output_type,
    'court_calibration': CourtCalibrationOutput,
    'ball_detection': BallDetectionOutput,
    'person_detection': PersonDetectionOutput,
    'person_tracking': PersonTrackingOutput,
    'player_selection': PlayerSelectionOutput,
    'pose_estimation': ObjectObservations,
}


def load_completed(store: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the whole active dependency graph and every array without a writer."""
    index = json.loads((store / 'scene.json').read_text())
    if document_digest(index['source']) != index['source_sha256']:
        raise ValueError('Store source hash mismatch')
    assert_current_component_lineage(index, store, index['artifacts'])
    outputs, descriptors = {}, {}
    for node, reference in index['artifacts'].items():
        descriptor = read_component_descriptor(store, reference, node=node, source_sha256=index['source_sha256'])
        if descriptor['status'] != 'complete' or descriptor['execution_key'] != document_digest(descriptor['identity']):
            raise ValueError(f'Incomplete or changed identity: {node}')
        unsigned = {k: v for k, v in descriptor.items() if k != 'artifact_id'}
        if document_digest(unsigned) != reference['artifact_id']:
            raise ValueError(f'Artifact identity mismatch: {node}')
        outputs[node] = ArtifactCodec(OUTPUT_TYPES[node.split('/')[0]]).load(
            descriptor['payload'], (store / reference['path']).parent, descriptor['arrays'])
        descriptors[node] = descriptor
    return index, outputs, descriptors


def audit_person(camera: str, frames: int, outputs: dict[str, Any]) -> dict[str, Any]:
    detections: PersonDetectionOutput = outputs[f'person_detection/{camera}']
    tracks: PersonTrackingOutput = outputs[f'person_tracking/{camera}']
    selection: PlayerSelectionOutput = outputs[f'player_selection/{camera}']
    poses: ObjectObservations = outputs[f'pose_estimation/{camera}']
    if len(detections.frame_offsets) != frames + 1 or detections.duplicate_merges \
            or detections.source_rows is None or tracks.observed.shape[1] != frames \
            or tracks.evidence is None or tracks.reconstruction is None:
        raise ValueError(f'{camera}: missing full production detection/tracking')
    if (detections.confidence < np.float32(.3)).any() \
            or (tracks.reconstruction.interpolated & tracks.observed).any():
        raise ValueError(f'{camera}: detector gate or synthetic observation changed')
    for frame in range(frames):
        start, end = detections.frame_offsets[frame:frame + 2]
        rows = tracks.evidence.detection_rows[:, frame][tracks.observed[:, frame]]
        source_rows = detections.source_rows[start:end]
        indices = np.searchsorted(source_rows, rows)
        if (indices >= len(source_rows)).any() or not np.array_equal(source_rows[indices], rows):
            raise ValueError(f'{camera}: tracking cites a non-source detection')
        np.testing.assert_array_equal(tracks.boxes_xyxy[:, frame][tracks.observed[:, frame]],
                                      detections.boxes_xyxy[start:end][indices])
    np.testing.assert_array_equal(selection.raw_track_ids, tracks.track_ids)
    if (selection.selected & ~tracks.observed).any() or len(selection.tracks.track_ids) > 6:
        raise ValueError(f'{camera}: selection introduced observations or exceeded cap')
    groups, times = np.nonzero(selection.tracks.observed)
    raw = selection.origin_rows[groups, times]
    np.testing.assert_array_equal(selection.tracks.boxes_xyxy[groups, times], tracks.boxes_xyxy[raw, times])
    evidence = selection.tracks.evidence
    if evidence is None:
        raise ValueError('Selected pose evidence missing')
    np.testing.assert_array_equal(evidence.detection_rows[groups, times], tracks.evidence.detection_rows[raw, times])
    np.testing.assert_array_equal(evidence.poses[groups, times], tracks.evidence.poses[raw, times])
    np.testing.assert_array_equal(poses.observed, selection.tracks.observed.T[None])
    np.testing.assert_array_equal(poses.uv_px, evidence.poses[..., :2].transpose(1, 0, 2, 3)[None])
    np.testing.assert_array_equal(poses.confidence, evidence.poses[..., 2].transpose(1, 0, 2)[None])
    return {'frames': frames, 'detections': len(detections.confidence),
            'min_detection_score': float(detections.confidence.min()),
            'raw_tracks': len(tracks.track_ids), 'observed_track_frames': int(tracks.observed.sum()),
            'gsi_synthetic_frames': int(tracks.reconstruction.interpolated.sum()),
            'source_track_ids': tracks.source_track_ids,
            'selected_raw_observations': int(selection.selected.sum()),
            'selected_groups': len(selection.tracks.track_ids),
            'group_observations': selection.tracks.observed.sum(1).tolist(),
            'selection_diagnostics': selection.diagnostics,
            'source_rows_boxes_poses_equal': True, 'synthetic_used_as_observation': 0}


def collect(report: Path, queue: Path, job: str, output: Path) -> None:
    executed = json.loads((report / 'execute.json').read_text())
    preflight = json.loads((report / 'preflight.json').read_text())
    run = executed['run']
    if executed['status'] != 'failed' or run['error_reason'] != 'court_side_ambiguous_margin' \
            or run['source']['clip_id'] != 'video_000/clip_000' or executed['imported_nodes'] \
            or executed['dev_labels_opened'] or executed['pid'] == os.getpid():
        raise ValueError('Expected the unimported clip_000 side stop from a separate process')
    if executed['preflight'] != file_identity(report / 'preflight.json'):
        raise ValueError('Preflight changed after execution')
    config = preflight['config']
    if file_identity(Path(config['path'])) != config:
        raise ValueError('Pinned config changed')
    index, outputs, descriptors = load_completed(report / 'store')
    expected = [n for n in preflight['order'] if run['stage_status'].get(n) == 'executed']
    if set(outputs) != set(expected) or index['artifacts'] != run['artifacts'] or index['exports']:
        raise ValueError('Partial store disagrees with execution receipt')
    source = run['source']
    if source != index['source'] or source != preflight['source']:
        raise ValueError('Source identity mismatch')
    for video in source['videos']:
        if dual_sha256(Path(video['path'])) != video['sha256']:
            raise ValueError('Source media changed')
    people = {v['camera_id']: audit_person(v['camera_id'], v['num_frames'], outputs) for v in source['videos']}
    snapshots = [report / n for n in ('execute.json', 'resource_guard.json', 'store/scene.json', 'store/run.json')]
    snapshots += [queue / 'logs' / f'{job}.log', queue / 'failed' / f'{job}.job', queue / 'state' / f'{job}.state']
    repro = queue / 'repro' / job
    snapshots += [repro / n for n in ('run.json', 'repro.sh', 'git_status.txt', 'uncommitted.patch')]
    output.mkdir(parents=True, exist_ok=False)
    for path in snapshots:
        relative = Path('queue_repro') / path.name if path.parent == repro else Path('snapshot') / path.name
        if path == report / 'store/run.json':
            relative = Path('snapshot/store_run.json')
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for node, descriptor in descriptors.items():
        write_json_atomic(output / 'descriptors' / f'{node.replace("/", "-")}.json', descriptor)
    record = {'schema': 'i964_partial_collection_v1', 'status': 'verified_partial_failure', 'pid': os.getpid(),
              'execute_pid': executed['pid'], 'job': job, 'labels_opened': False, 'imported_nodes': [],
              'source': source, 'resources': json.loads((report / 'resource_guard.json').read_text()),
              'executed_nodes': expected, 'failed_node': run['active_stage'],
              'not_executed': [n for n in preflight['order'] if n not in run['stage_status']],
              'stage_seconds': run['stage_seconds'], 'side_stop': run['error_diagnostics'],
              'artifacts': index['artifacts'], 'arrays_verified': sum(len(d['arrays']) for d in descriptors.values()),
              'people': people, 'inputs': [file_identity(p) for p in snapshots],
              'scene_export': False, 'qualification_validation': False, 'full_video': False,
              'settings': {n: d['identity']['settings'] for n, d in descriptors.items()}}
    write_json_atomic(output / 'collection.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.report.resolve(), args.queue.resolve(), args.job, args.output.resolve())
