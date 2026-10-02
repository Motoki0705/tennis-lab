"""Durable CPU selection/CLIP/association for the pre-ROI development comparison."""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from person_selection_cpu import (  # type: ignore[import-not-found]  # sibling CLI
    calibration,
    dense_tracks,
    load_sources,
)

from src.tasks.person_tracking.appearance_cache import cached_appearance
from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    exclusive_region,
    select_linked_candidates,
)
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.person_tracking.selection_burden import track_burden
from src.tasks.person_tracking.selection_metrics import selection_units
from src.tasks.player_association.appearance.encoders import (
    build_encoder,
    encoder_weights,
)
from src.tasks.player_association.appearance.sampling import TrackAppearance
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import load_association_config
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.geometry.footpoints import ground_footpoints
from src.tasks.player_detection.evaluation.fullframe_sources import SOURCES
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.utils.checksum import dual_sha256

CODE_ROOT = Path(__file__).resolve().parents[2]


def load_camera(saved: dict[str, Any], camera: Any) -> tuple[CameraTracks, dict[str, np.ndarray]]:
    if dual_sha256(Path(saved['path'])) != saved['sha256']:
        raise ValueError('Selection camera archive changed')
    with np.load(saved['path'], allow_pickle=False) as a:
        arrays = dict(a)
    appearance = [TrackAppearance(arrays[f'appearance/{i}/frames'], arrays[f'appearance/{i}/embeddings']) for i in range(len(arrays['track_ids']))]
    return CameraTracks(camera, (1920, 1080), arrays['track_ids'], arrays['boxes'], arrays['observed'], tuple(appearance)), arrays


def select_next(repo: Path, report: Path, max_clips: int) -> None:
    source = load_sources(report)
    reservation = Path(source['reservation'])
    if dual_sha256(reservation) != source['reservation_sha256'] or set(json.loads(reservation.read_text())['clips']) & set(DEV_CLIPS):
        raise ValueError('Unseen reservation changed/overlaps')
    tracking = json.loads((report / 'tracks.json').read_text())
    if tracking['sources_sha256'] != dual_sha256(report / 'sources.json'):
        raise ValueError('Source manifest changed since tracking')
    config_path = CODE_ROOT / 'src/tasks/player_association/configs/association.yaml'
    association_config = load_association_config(config_path, players_per_side=1)
    config = LinkingConfig()
    sides_path = repo / 'outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json'
    sides = json.loads(sides_path.read_text())
    encoder_name = 'clipreid_vitb16_market1501'
    weights = encoder_weights(encoder_name, checkpoint_root=repo / 'ckpt', external_root=repo / 'third_party')
    identity = {'sources_sha256': dual_sha256(report / 'sources.json'), 'tracks_sha256': dual_sha256(report / 'tracks.json'),
        'linking': asdict(config), 'association': json_value(association_config), 'association_config_sha256': dual_sha256(config_path),
        'side_decisions': str(sides_path), 'side_sha256': dual_sha256(sides_path),
        'encoder': {'name': encoder_name, 'weights': str(weights), 'sha256': dual_sha256(weights), 'device': 'cpu'},
        'selection_code_sha256': dual_sha256(CODE_ROOT / 'src/tasks/person_tracking/court_linking.py'),
        'appearance_code_sha256': dual_sha256(CODE_ROOT / 'src/tasks/person_tracking/appearance_cache.py'),
        'stage': 'linked dwell primary; existing CLIP cross-camera association secondary; undecided has no guessed IDs'}
    manifest_path = report / 'selection_protocol.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError('Selection resume protocol changed')
    else:
        write_json_atomic(manifest_path, identity)
    encoder = None
    finished = 0
    for variant in SOURCES:
        for clip in DEV_CLIPS:
            target = report / 'selection' / variant / clip / 'result.json'
            if target.exists():
                saved = json.loads(target.read_text())
                if saved['protocol_sha256'] != dual_sha256(manifest_path):
                    raise ValueError('Completed selection protocol changed')
                for value in (*saved['cameras'].values(), saved['units']):
                    if dual_sha256(Path(value['path'])) != value['sha256']:
                        raise ValueError('Completed selection content changed')
                continue
            if finished >= max_clips:
                return
            records = sorted((r for r in source['inputs'] if r['clip'] == clip), key=lambda r: r['camera'])
            side = next(s for s in sides['clips'] if s['clip_id'] == clip)
            if not side['annotation']['decided']:
                raise ValueError('Reference court side undecided')
            turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
            labels = ClipLabels.load(Path(records[0]['label_path']))
            result: dict[str, Any] = {'source': variant, 'clip': clip, 'protocol_sha256': dual_sha256(manifest_path),
                'cameras': {}, 'burden': [], 'association': {}}
            raw_tracks, groups, origins = [], [], []
            units: list[dict[str, Any]] = []
            for record in records:
                cam = record['camera']
                camera = calibration(record, turns[cam])
                metadata = target.parent / f'{cam}.json'
                if metadata.exists():
                    saved = json.loads(metadata.read_text())
                    if saved['protocol_sha256'] != dual_sha256(manifest_path):
                        raise ValueError('Camera selection protocol changed')
                    tracks, arrays = load_camera(saved, camera)
                    mask = arrays['selected']
                else:
                    tracks, provenance = dense_tracks(source, tracking, variant, record, camera, repo)
                    xy, valid = ground_footpoints(tracks.boxes_xyxy, tracks.observed, camera, 1080, association_config.footpoints)
                    core = valid & exclusive_region(xy, config, core=True)
                    if encoder is None:
                        encoder = build_encoder(encoder_name, checkpoint_root=repo / 'ckpt', external_root=repo / 'third_party', device='cpu')
                    features, sampling = cached_appearance(tracks, core, record['video'], encoder, identity['encoder']['sha256'], report / 'clip_crop_cache')
                    tracks = replace(tracks, appearance=features)
                    mask, diagnostic = select_linked_candidates(tracks, record['video']['fps'], config, association_config.footpoints)
                    path = target.parent / f'{cam}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    appearance_arrays = {f'appearance/{i}/{field}': getattr(a, field) for i, a in enumerate(features) for field in ('frames', 'embeddings')}
                    with path.open('xb') as dst:
                        np.savez_compressed(dst, track_ids=tracks.track_ids, boxes=tracks.boxes_xyxy, observed=tracks.observed, selected=mask, **appearance_arrays)
                    saved = {'path': str(path), 'sha256': dual_sha256(path), 'protocol_sha256': dual_sha256(manifest_path),
                        'tracking': provenance, 'sampling': sampling, 'linked': diagnostic}
                    write_json_atomic(metadata, saved)
                    print(f"selected {variant} {clip}/{cam}: {len(tracks.track_ids)} tracks -> {diagnostic['selected_groups']} groups; CLIP {sampling['new_crops']} new/{sampling['cached_crops']} cached", flush=True)
                group, origin = linked_timeline(tracks, saved['linked'])
                raw_tracks.append(tracks)
                groups.append(group)
                origins.append(origin)
                result['cameras'][cam] = saved
                result['burden'].extend(track_burden(tracks, mask, labels))
                units.extend({'stage': 'selected', **u} for u in selection_units(tracks, mask, labels))
                group_path = target.parent / f'{cam}.groups.npz'
                if not group_path.exists():
                    with group_path.open('xb') as dst:
                        np.savez_compressed(dst, track_ids=group.track_ids, boxes=group.boxes_xyxy, observed=group.observed, origins=origin)
                result['cameras'][cam]['groups'] = {'path': str(group_path), 'sha256': dual_sha256(group_path)}
            try:
                associated = associate(groups, records[0]['video']['fps'], association_config)
            except AssociationUndecided as error:
                result['association'] = {'status': 'undecided', 'reason': error.reason, 'diagnostics': error.diagnostics}
            else:
                result['association'] = {'status': 'ok', 'diagnostics': associated.diagnostics}
                for index, tracks in enumerate(raw_tracks):
                    group_ids = associated.player_ids[index]
                    ids = np.full(tracks.observed.shape, -1, np.int64)
                    for row in range(len(group_ids)):
                        at = np.flatnonzero((group_ids[row] >= 0) & (origins[index][row] >= 0))
                        ids[origins[index][row, at], at] = group_ids[row, at]
                    units.extend({'stage': 'associated', **u} for u in selection_units(tracks, ids >= 0, labels))
                    path = target.parent / f'{tracks.camera.camera_id}.identities.npz'
                    with path.open('xb') as dst:
                        np.savez_compressed(dst, player_ids=ids, group_player_ids=group_ids)
                    result['cameras'][tracks.camera.camera_id]['identities'] = {'path': str(path), 'sha256': dual_sha256(path)}
            unit_path = target.parent / 'units.jsonl.gz'
            with gzip.open(unit_path, 'wt') as dst:
                for unit in units:
                    dst.write(json.dumps({'source': variant, **unit}) + '\n')
            result['units'] = {'path': str(unit_path), 'sha256': dual_sha256(unit_path)}
            write_json_atomic(target, result)
            finished += 1
            print(f"completed {variant} {clip}: association {result['association']['status']} {result['association'].get('reason', '')}", flush=True)
