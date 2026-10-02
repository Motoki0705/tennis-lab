"""CPU cache/geometry audit for run 12; deliberately has no evaluation label reader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from association_recalibration_features import (  # type: ignore[import-not-found]
    CODE,
    checked,
    restrict_appearance,
)

from src.tasks.person_tracking.archive import load_features, save_features
from src.tasks.person_tracking.features import FeatureConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.definition import file_identity
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256


def audit(repo: Path, report: Path, plan_path: Path, preflight_path: Path) -> None:
    if (report / 'reuse.json').exists():
        raise FileExistsError('Reuse audit is immutable')
    plan = json.loads(plan_path.read_text())
    full = json.loads(preflight_path.read_text())
    old_root = repo / 'outputs/person_tracking/evaluate/dev_features/i964-features-r7-20260930'
    old_plan = json.loads((old_root / 'plan.json').read_text())
    features_path = old_root / 'features.json'
    old = json.loads(features_path.read_text())
    encoder = plan['production_tracking']['encoder']
    if old['status'] != 'ok' or old['plan_sha256'] != dual_sha256(old_root / 'plan.json'):
        raise ValueError('Invalid completed dev features')
    source = json.loads(checked(old_plan['sources']).read_text())
    inference = json.loads(checked({'path': source['coco_inference'], 'sha256': source['coco_inference_sha256']}).read_text())
    if old_plan['source_name'] != 'coco_fullframe_0.30' or inference['scope'] != 'full_frame_no_roi' \
            or inference['baseline_size'] != [800, 1333] or plan['merge_duplicates']:
        raise ValueError('Dev detector scope/resize/threshold/merge changed')
    for role, saved in [('detector', old_plan['detector']), ('pose', old_plan['models']['pose']),
                        ('clip', old_plan['models'][encoder])]:
        checked(saved)
        if saved['sha256'] != plan['models'][role]['sha256']:
            raise ValueError(f'Dev {role} checkpoint differs')
    if old_plan['models']['pose']['runtime'] != plan['pose_runtime'] or old_plan['models']['pose']['precision'] != 'float32':
        raise ValueError('Pose runtime/precision differs')
    before, after = FeatureConfig(**old_plan['feature_config']), FeatureConfig(**plan['feature_config'])
    track_root = repo / 'outputs/person_tracking/evaluate/tracker_matrix/i964-default-merge-r11-20260930'
    identity = json.loads((track_root / 'identity.json').read_text())
    if identity['profile'] != plan['production_tracking'] or identity['aflink']['sha256'] != plan['models']['aflink']['sha256']:
        raise ValueError('Saved dev tracker profile differs')
    if checked(identity['features']) != features_path:
        raise ValueError('Saved tracks used a different dev feature manifest')
    for path, sha in identity['code'].items():
        if dual_sha256(CODE / path) != sha:
            raise ValueError(f'Production/selection/association code differs: {path}')
    records = {}
    for key, saved in old['records'][encoder].items():
        frames, provenance = load_features(checked(saved))
        checked(provenance['source'])
        checked(provenance['detection'])
        if provenance['config'] != old_plan['feature_config'] or any((f.scores < np.float32(.3)).any() for f in frames):
            raise ValueError('Dev feature settings/threshold differ')
        projected, changed = restrict_appearance(frames, before, after, (1920, 1080))
        changed_rows = [int(row) for a, b in zip(frames, projected, strict=True)
                        for row in a.rows[a.appearance_valid != b.appearance_valid]]
        selected = dict(saved)
        if changed:
            path = report / 'projected_dev' / f'{key}.features.npz'
            save_features(path, projected, {**provenance, 'config': plan['feature_config'],
                'projection': {'source_archive': saved, 'changed_rows': changed_rows,
                               'rule': 'production rounded/clipped crop height >=16 px; no model inference'}})
            roundtrip, _ = load_features(path)
            if any(not np.array_equal(getattr(a, field), getattr(b, field))
                   for a, b in zip(projected, roundtrip, strict=True)
                   for field in ('rows', 'boxes', 'scores', 'poses', 'embeddings', 'appearance_valid')):
                raise ValueError('Projected features changed during serialization')
            selected = file_identity(path)
        descriptor = track_root / 'tracks' / f'strongsort_pp_pose_merge_off__{encoder}' / f'{key}.json'
        tracked = json.loads(descriptor.read_text())
        checked(tracked['arrays'])
        checked(tracked['gsi'])
        if tracked['identity_sha256'] != dual_sha256(track_root / 'identity.json') or tracked['status'] != 'ok':
            raise ValueError('Incomplete or mismatched saved dev tracking')
        records[key] = {'source': saved, 'reuse_features': selected, 'changed_appearance_rows': changed_rows,
                        'detection_pose_rows': sum(len(f.rows) for f in frames),
                        'tracking_action': 'cpu_retrack_required' if changed else 'reuse_exact_input_track',
                        'previous_track': file_identity(descriptor)}
    previous = Path(plan['historical_observe']['path']).parent
    side_path = repo / 'outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json'
    sides = json.loads(side_path.read_text())
    courts = {}
    for clip in plan['selected_clips']:
        root = previous / 'stores' / clip
        scene = json.loads((root / 'scene.json').read_text())
        videos = [r['video'] for r in plan['inputs'] if r['clip'] == clip]
        if scene['source']['videos'] != videos:
            raise ValueError(f'{clip}: prior court source differs')
        store = ClipStore(root, scene['source'])
        reference = store.active('court_calibration')
        if reference is None:
            raise ValueError(f'{clip}: missing calibration')
        descriptor = store.descriptor(reference)
        if descriptor['identity']['settings'] != full['nodes']['court_calibration']['settings']:
            raise ValueError('Court calibration settings differ')
        calibration = store.load(reference, ArtifactCodec(CourtCalibrationOutput)).calibration
        if {view.camera.camera_id for view in calibration.views} != {'cam0', 'cam1', 'cam2'}:
            raise ValueError(f'{clip}: incomplete calibration')
        detection_refs = {}
        for camera in ('cam0', 'cam1', 'cam2'):
            ref = store.active(f'court_detection/{camera}')
            if ref is None:
                raise ValueError('Missing court source')
            d = store.descriptor(ref)
            settings = d['identity']['settings']
            expected = full['nodes'][f'court_detection/{camera}']['settings']
            if settings['checkpoint']['sha256'] != expected['checkpoint']['sha256'] \
                    or settings['config'] != expected['config']:
                raise ValueError('Court detector config/checkpoint differs')
            # descriptor verifies its checksum; load also validates every numerical array.
            store.load(ref, ArtifactCodec(CourtKPResult))
            detection_refs[camera] = json_value(ref)
        decisions = [r['annotation'] for r in sides['clips'] if r['clip_id'] == clip]
        if len(decisions) != 1 or not decisions[0]['decided']:
            raise ValueError(f'{clip}: no decided annotation-ball side')
        courts[clip] = {'store': str(root), 'calibration': json_value(reference),
                        'court_detection': detection_refs, 'side': decisions[0],
                        'purpose': 'frozen calibration geometry; not an end-to-end prediction'}
    write_json_atomic(report / 'reuse.json', {
        'schema': 'i964_recalibration_reuse_v1', 'fit_executed': False, 'dev_scoring_executed': False,
        'plan': file_identity(plan_path), 'preflight': file_identity(preflight_path),
        'old_features': file_identity(features_path), 'old_tracking': file_identity(track_root / 'identity.json'),
        'side_source': file_identity(side_path), 'dev': records, 'calibration_geometry': courts,
        'new_features_required': [r['key'] for r in plan['inputs']],
    })
    print(f'Reuse audited: {len(records)} dev cameras, {len(courts)} calibration clips; no fitting or scoring')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo.resolve(), args.report.resolve(), args.plan.resolve(), args.preflight.resolve())


if __name__ == '__main__':
    main()
