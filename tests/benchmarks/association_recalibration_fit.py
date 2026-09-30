"""CPU-only fit entrypoint, gated on all 18 complete calibration cameras.

Run only after feature completion. This entrypoint cannot read dev labels or
score dev; it writes a proposed YAML to a new report directory, never defaults.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
import yaml
from association_recalibration_features import (  # type: ignore[import-not-found]
    CAMERAS,
    CODE,
    checked,
    validate_plan,
)
from association_recalibration_resume import (  # type: ignore[import-not-found]
    resume_records,
    verify_completed,
)

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.feature_tracks import evidence_appearance
from src.tasks.person_tracking.sequence import TrackingConfig, track_sequence
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.association.config import (
    DEFAULT_CONFIG,
    load_association_config,
)
from src.tasks.player_association.calibration.samples import prepare_clip
from src.tasks.player_association.calibration.validation import calibrate
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.definition import file_identity
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256


def require_complete(plan: dict[str, Any], features: dict[str, Any]) -> None:
    clips = plan['selected_clips']
    expected = {f'{clip}/{camera}' for clip in clips for camera in CAMERAS}
    if len(clips) != 6 or len(expected) != 18 or features['status'] != 'ok' \
            or set(features['records']) != expected or set(features['detections']) != expected:
        raise ValueError('Fit is closed until all 18 planned cameras are complete')


def run(repo: Path, feature_root: Path, reuse_path: Path, report: Path) -> None:
    if report.exists():
        raise FileExistsError('Fit output is immutable; do not rerun or retune on dev')
    # Readiness is checked before loading features, constructing trackers or fitting.
    feature_path = feature_root / 'features.json'
    features = json.loads(feature_path.read_text())
    plan_path = feature_root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    require_complete(plan, features)
    validate_plan(plan, repo)
    resume_records(plan)
    verify_completed(plan, features, dual_sha256(plan_path))
    reuse = json.loads(reuse_path.read_text())
    if set(reuse['calibration_geometry']) != set(plan['selected_clips']):
        raise ValueError('Calibration geometry differs from the frozen split')
    sides = json.loads(checked(reuse['side_source']).read_text())
    old = json.loads(checked(reuse['old_tracking']).read_text())
    base = load_association_config(players_per_side=1)
    if asdict(base) != old['association'] or TrackingConfig().identity() != plan['production_tracking']:
        raise ValueError('Fixed association/tracker defaults changed')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    aflink = AFLink(checked(plan['models']['aflink']))
    identity = {'features': file_identity(feature_path), 'plan': file_identity(plan_path),
                'reuse': file_identity(reuse_path), 'base_config': file_identity(DEFAULT_CONFIG),
                'benchmark': file_identity(Path(__file__)),
                'fitter': {str(p.relative_to(CODE)): dual_sha256(p)
                           for p in (CODE / 'src/tasks/player_association/calibration').glob('*.py')}}
    write_json_atomic(report / 'identity.json', identity)
    clips, selections = [], {}
    inputs = {r['key']: r for r in plan['inputs']}
    for key in plan['selected_clips']:
        geometry = reuse['calibration_geometry'][key]
        root = Path(geometry['store'])
        scene = json.loads((root / 'scene.json').read_text())
        if scene['source']['videos'] != [r['video'] for r in plan['inputs'] if r['clip'] == key]:
            raise ValueError('Court calibration source timeline differs')
        store = ClipStore(root, scene['source'])
        ref = store.active('court_calibration')
        if ref is None or json_value(ref) != geometry['calibration']:
            raise ValueError('Court calibration artifact changed')
        court = store.load(ref, ArtifactCodec(CourtCalibrationOutput)).calibration
        cameras = {view.camera.camera_id: view.camera for view in court.views}
        side = next(s for s in sides['clips'] if s['clip_id'] == key)
        if side['annotation'] != geometry['side'] or not side['annotation']['decided']:
            raise ValueError('Frozen side decision changed')
        turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
        raw = []
        for camera_id in CAMERAS:
            if psutil.virtual_memory().available < 6 * 1024**3:
                raise RuntimeError('Require at least 6 GiB available host RAM')
            camera_key = f'{key}/{camera_id}'
            meta = inputs[camera_key]['video']
            frames, _ = load_features(checked(features['records'][camera_key]))
            sequence = track_sequence(frames, fps=meta['fps'], config=TrackingConfig(), aflink=aflink)
            appearance = evidence_appearance(sequence.boxes, sequence.observed, sequence.evidence,
                                            (meta['width'], meta['height']), CropSamplingConfig())
            raw.append(CameraTracks(cameras[camera_id].half_turned(turns[camera_id]),
                (meta['width'], meta['height']), sequence.track_ids, sequence.boxes, sequence.observed, tuple(appearance)))
            path = report / 'tracks' / f'{camera_key}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                np.savez_compressed(handle, track_ids=sequence.track_ids, boxes=sequence.boxes,
                                    observed=sequence.observed, origins=sequence.evidence.detection_rows)
        clip, selected = prepare_clip(key, inputs[f'{key}/cam0']['video']['fps'], tuple(raw), base)
        clips.append(clip)
        selections[key] = selected
    write_json_atomic(report / 'selection.json', selections)
    config, evidence = calibrate(clips, base)
    evidence.update(identity=file_identity(report / 'identity.json'), dev_scoring_executed=False)
    write_json_atomic(report / 'fit.json', evidence)
    if config is not None:
        values = asdict(config)
        values.pop('players_per_side')
        path = report / 'association.yaml'
        path.write_text(yaml.safe_dump(values, sort_keys=False))
        if load_association_config(path, players_per_side=1) != config:
            raise ValueError('Fitted config changed during YAML roundtrip')
    print(f'Fit status: {evidence["status"]}; commit fit evidence/YAML before any dev scoring')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'features', 'reuse', 'report'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.features.resolve(), args.reuse.resolve(), args.report.resolve())


if __name__ == '__main__':
    main()
