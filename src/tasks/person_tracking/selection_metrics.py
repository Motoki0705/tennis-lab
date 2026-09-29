"""Selection against partial reviewed person/frame units, without GT selection.

Unmatched reference units remain in the denominator. For known nonplayers,
report both total exclusions and exclusions conditional on a tracker hit;
missing detections must not masquerade as successful geometric rejection.
Identity counts are derived from these units, separately from association IDs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.submodules.models import PersonDetectionResult
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.tasks.player_detection.evaluation.far_player import unit_rows
from src.utils.schema.court import HALF_DOUBLES_WIDTH

# Explicit strata from the reviewed #933 YAML (including its auxiliary keys).
# These are used only for scoring, never for selecting a predicted track.
NONPLAYER_KIND = {
    ('video_000/clip_000', 'X1'): 'adjacent_court',
    ('video_000/clip_007', 'X1'): 'off_court',
    ('video_000/clip_007', 'X2'): 'nonperson',
    ('video_001/clip_001', 'X1'): 'off_court',
    ('video_001/clip_001', 'X2'): 'adjacent_court',
    ('video_001/clip_001', 'X3'): 'nonperson',
    ('video_001/clip_001', 'X4'): 'off_court',
    ('video_002/clip_013', 'X1'): 'nonperson',
}


def selection_units(tracks: CameraTracks, selected: np.ndarray, labels: ClipLabels) -> list[dict[str, Any]]:
    if selected.shape != tracks.observed.shape or selected.dtype != np.bool_:
        raise ValueError('Selection must align with track/frame observations')
    camera = tracks.camera.camera_id
    output = []
    for frame in range(labels.num_frames):
        active = tracks.observed[:, frame]
        accepted = active & selected[:, frame]
        def rows(mask: np.ndarray, index: int) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = unit_rows(PersonDetectionResult(tracks.boxes_xyxy[mask, index].astype(np.float32), np.ones(int(mask.sum()), np.float32)),
                                                   labels.cameras[camera], labels.roles, index)
            return result
        all_rows, selected_rows = rows(active, frame), rows(accepted, frame)
        for before, after in zip(all_rows, selected_rows, strict=True):
            person = labels.people[before['person']]
            reference_xy, reference_valid = ground_footpoints(np.asarray(before['old_box']), np.asarray(True),
                tracks.camera, tracks.image_size[1], FootpointConfig())
            output.append({'clip': labels.clip_id, 'camera': camera, 'frame': frame, 'person': person.person_id,
                'role': person.role, 'kind': 'player' if person.role == 'player' else NONPLAYER_KIND[(labels.clip_id, person.person_id)],
                'near_far': before['near_far'], 'tracked_05': before['matched'], 'selected_05': after['matched'],
                'tracked_03': before['matched_iou03'], 'selected_03': after['matched_iou03'],
                'reference_ground_valid': bool(reference_valid),
                'reference_x_m': float(reference_xy[0]) if reference_valid else None,
                'reference_wide': bool(abs(reference_xy[0]) > HALF_DOUBLES_WIDTH) if reference_valid else None})
    return output


def aggregate_units(units: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ('player', 'non_player', 'adjacent_court', 'off_court', 'nonperson'):
        items = [u for u in units if u['role'] == kind or u['kind'] == kind]
        result[f'{kind}_units'] = len(items)
        for iou in ('03', '05'):
            tracked = sum(u[f'tracked_{iou}'] for u in items)
            selected = sum(u[f'selected_{iou}'] for u in items)
            rejected_tracked = sum(u[f'tracked_{iou}'] and not u[f'selected_{iou}'] for u in items)
            result.update({f'{kind}_tracked_{iou}': tracked, f'{kind}_kept_{iou}': selected,
                f'{kind}_excluded_{iou}': len(items) - selected,
                f'{kind}_rejected_among_tracked_{iou}': rejected_tracked})
            # Player identity retained: >=50% of its labelled units. Nonplayer
            # identity fully rejected: zero selected units. Both rules explicit.
            identities = {(u['clip'], u['person']) for u in items}
            by_identity = [[u for u in items if (u['clip'], u['person']) == key] for key in sorted(identities)]
            result[f'{kind}_identities'] = len(identities)
            result[f'{kind}_identities_kept50_{iou}'] = sum(sum(u[f'selected_{iou}'] for u in group) >= .5 * len(group) for group in by_identity)
            result[f'{kind}_identities_rejected_all_{iou}'] = sum(not any(u[f'selected_{iou}'] for u in group) for group in by_identity)
    for camera in ('cam0', 'cam1', 'cam2'):
        far = [u for u in units if u['role'] == 'player' and u['near_far'] == 'far' and u['camera'] == camera]
        result[f'{camera}_far_reference_frames'] = len(far)
        result[f'{camera}_far_covered_03'] = sum(u['selected_03'] for u in far)
        result[f'{camera}_far_covered_05'] = sum(u['selected_05'] for u in far)
    wide = [u for u in units if u['role'] == 'player' and u.get('reference_wide') is True]
    result['player_wide_units'] = len(wide)
    result['player_ground_unknown_units'] = sum(u['role'] == 'player' and not u.get('reference_ground_valid', False) for u in units)
    for iou in ('03', '05'):
        result[f'player_wide_tracked_{iou}'] = sum(u[f'tracked_{iou}'] for u in wide)
        result[f'player_wide_kept_{iou}'] = sum(u[f'selected_{iou}'] for u in wide)
    return result
